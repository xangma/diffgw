"""Check integrated peak continuation against frozen q8 ablation and q4 control."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch

from . import qualify_generator as q
from .diagnose_peak_anchor import public_mode
from .diagnose_q8_mode_phase import _replay
from .diagnose_q8_ringdown_peak_bound import _frequency_metrics


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/peak_condition_integrated_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign_path = q.ROOT / plan["campaign"]
    q8_path = q.ROOT / plan["q8_counterfactual"]
    q4_path = q.ROOT / plan["q4_control"]
    if (plan["schema"] != "diffgw.seobnrv4.peak-condition-integrated-plan.v1"
            or plan["rows"] != [
                dict(name="q8_high_aligned", level="ode_2", anchor_angular_frequency_M=0.1),
                dict(name="q4_mixed_spin", level="ode_2", anchor_angular_frequency_M=0.1)]
            or plan["release_qualified"] is not False
            or q.sha256(campaign_path) != plan["campaign_sha256"]):
        parser.error("frozen plan or campaign differs")
    campaign = json.loads(campaign_path.read_text())
    q8_prior = json.loads(q8_path.read_text())
    q4_prior = json.loads(q4_path.read_text())
    pin = campaign["lal_binary"]
    if (campaign["status"] != "diagnostic_complete"
            or q8_prior["status"] != "diagnostic_complete"
            or q4_prior["status"] != "diagnostic_complete"
            or q8_prior["lal_binary"] != pin or q4_prior["lal_binary"] != pin
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("frozen references or public binary differ")
    torch.set_num_threads(1)
    source_paths = (
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/qualify_generator.py",
        "src/torchwave/approximants/eob/merger_ringdown.py",
        "src/torchwave/approximants/eob/numerical_attachment.py",
    )
    report = dict(schema="diffgw.seobnrv4.peak-condition-integrated.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={p: q.sha256(q.ROOT / p) for p in (
                      plan["campaign"], plan["q8_counterfactual"], plan["q4_control"])},
                  source_sha256={p: q.sha256(q.ROOT / p) for p in source_paths},
                  lal_binary=pin, rows={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        for row_plan in plan["rows"]:
            name = row_plan["name"]
            item = campaign["rows"][name]
            prior = q8_prior if name == "q8_high_aligned" else q4_prior
            if item["runs"]["ode_2"]["level"]["name"] != row_plan["level"]:
                raise ValueError(f"unexpected ODE level for {name}")
            started = time.monotonic()
            run = _replay(item["row"], item["runs"]["ode_2"]["level"],
                          q.Settings(**campaign["settings"]), campaign["choice"],
                          item["fixed_restart_time_M"])
            summary = q.summarize_run(run)
            expected = prior["native"]["array_sha256"]
            if summary["array_sha256"]["time"] != expected["time"]:
                raise ValueError(f"{name} time-grid hash changed")
            report["rows"][name] = dict(
                row=item["row"], level=item["runs"]["ode_2"]["level"],
                array_sha256=summary["array_sha256"],
                expected_mode_sha256=expected["mode"],
                mode_matches_frozen_candidate=(summary["array_sha256"]["mode"] == expected["mode"]),
                elapsed_seconds=time.monotonic()-started)
            q.write_evidence(args.json_out, report)
            print(f"{name} native waveform complete", flush=True)
            nt, nm, rt, rm, mass_seconds = public_mode(item["row"], run)
            report["rows"][name]["comparison"] = _frequency_metrics(
                nt, nm, rt, rm, mass_seconds, row_plan["anchor_angular_frequency_M"])
            report["rows"][name]["prior_complex_relative_l2"] = (
                prior["comparison"]["metrics"]["complex_relative_l2"]
                if name == "q8_high_aligned" else
                prior["frequency_anchored"]["metrics"]["complex_relative_l2"])
            q.write_evidence(args.json_out, report)
        report["frozen_candidate_reproduced"] = all(
            row["mode_matches_frozen_candidate"] for row in report["rows"].values())
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          frozen_candidate_reproduced=report.get("frozen_candidate_reproduced"))), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

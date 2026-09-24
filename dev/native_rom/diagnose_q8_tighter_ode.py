"""Extend frozen q8 numerical convergence by one tighter ODE solve."""

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
from .diagnose_q8_mode_phase import _replay


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q8_tighter_ode_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign_path = q.ROOT / plan["campaign"]
    expected_level = dict(name="ode_3", atol=3e-15, rtol=3e-14, dt_geom=0.5)
    if (plan["schema"] != "diffgw.seobnrv4.q8-tighter-ode-plan.v1"
            or plan["row_name"] != "q8_high_aligned"
            or plan["baseline_level"] != "ode_2"
            or plan["additional_level"] != expected_level
            or plan["relative_l2_limit"] != 1e-6
            or plan["production_change"] is not False
            or plan["release_qualified"] is not False
            or q.sha256(campaign_path) != plan["campaign_sha256"]):
        parser.error("frozen plan or campaign differs")
    campaign = json.loads(campaign_path.read_text())
    item = campaign["rows"][plan["row_name"]]
    row, choice = item["row"], campaign["choice"]
    settings = q.Settings(**campaign["settings"])
    if (campaign["status"] != "diagnostic_complete"
            or item["check"]["numerical_check_passed"] is not False
            or item["runs"]["ode_2"]["level"]["dt_geom"] != 0.5):
        parser.error("frozen numerical baseline differs")
    torch.set_num_threads(1)
    source_paths = (
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/qualify_generator.py",
        "dev/native_rom/diagnose_q8_mode_phase.py",
        "src/torchwave/approximants/eob/numerical_attachment.py",
    )
    report = dict(schema="diffgw.seobnrv4.q8-tighter-ode.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  campaign_sha256=q.sha256(campaign_path),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in source_paths},
                  row=row, choice=choice, runs={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        runs = []
        for name, level in (("ode_2", item["runs"]["ode_2"]["level"]),
                            ("ode_3", expected_level)):
            started = time.monotonic()
            run = _replay(row, level, settings, choice,
                          item["fixed_restart_time_M"])
            summary = q.summarize_run(run)
            if name == "ode_2" and summary["array_sha256"] != (
                    item["runs"]["ode_2"]["native"]["array_sha256"]):
                raise ValueError("fine native waveform differs from frozen campaign")
            report["runs"][name] = dict(level=level, native=summary,
                                         elapsed_seconds=time.monotonic()-started)
            q.write_evidence(args.json_out, report)
            print(f"level={name} complete", flush=True)
            runs.append(run)
        report["comparison"] = q.compare_runs(runs[0], runs[1])
        report["additional_gate_passed"] = (
            report["comparison"]["complex_relative_l2"] <= plan["relative_l2_limit"]
            and report["comparison"]["amplitude_relative_l2"] <= plan["relative_l2_limit"]
            and report["comparison"]["match_time_M_abs_difference"] <= settings.event_time_limit_M
            and report["comparison"]["peak_time_M_abs_difference"] <= settings.event_time_limit_M
            and report["comparison"]["boundary_time_M_abs_difference"] <= settings.event_time_limit_M)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          additional_gate_passed=report.get("additional_gate_passed"))), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

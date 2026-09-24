"""Measure independent Kerr QNM impact on the frozen q8 peak-preserving mode."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q8_independent_qnm_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign_path, baseline_path = (q.ROOT / plan[key] for key in
                                    ("campaign", "integrated_baseline"))
    if (plan["schema"] != "diffgw.seobnrv4.q8-independent-qnm-plan.v1"
            or plan["row_name"] != "q8_high_aligned"
            or plan["level_name"] != "ode_2"
            or plan["qnm_source"] != "independent_kerr"
            or plan["anchor_angular_frequency_M"] != .1
            or plan["production_change"] is not False
            or plan["release_qualified"] is not False
            or q.sha256(campaign_path) != plan["campaign_sha256"]
            or q.sha256(baseline_path) != plan["integrated_baseline_sha256"]):
        parser.error("frozen plan or inputs differ")
    campaign = json.loads(campaign_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    item = campaign["rows"][plan["row_name"]]
    row, choice = item["row"], campaign["choice"]
    prior = baseline["rows"][plan["row_name"]]
    pin = campaign["lal_binary"]
    if (campaign["status"] != "diagnostic_complete"
            or baseline["status"] != "diagnostic_complete"
            or prior["row"] != row or prior["level"] != item["runs"]["ode_2"]["level"]
            or baseline["lal_binary"] != pin
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("frozen q8 row or public binary differs")
    torch.set_num_threads(1)
    source_paths = ("dev/native_rom/eob/torch/adaptive_driver_clean.py",
                    "dev/native_rom/qualify_generator.py",
                    "dev/native_rom/diagnose_q8_mode_phase.py",
                    "src/torchwave/approximants/kerr_qnm_generated.py")
    report = dict(schema="diffgw.seobnrv4.q8-independent-qnm.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in source_paths},
                  campaign_sha256=q.sha256(campaign_path),
                  integrated_baseline_sha256=q.sha256(baseline_path),
                  lal_binary=pin, row=row, choice=choice, qnm_source=plan["qnm_source"],
                  errors=[], process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                                          pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        run = _replay(row, item["runs"]["ode_2"]["level"],
                      q.Settings(**campaign["settings"]), choice,
                      item["fixed_restart_time_M"], return_boundary_data=True,
                      diagnostic_qnm_source=plan["qnm_source"])
        summary = q.summarize_run(run)
        if (summary["array_sha256"]["time"] != prior["array_sha256"]["time"]
                or summary["array_sha256"]["mode"] == prior["array_sha256"]["mode"]):
            raise ValueError("independent QNM did not isolate the frozen q8 mode")
        report["native"] = dict(array_sha256=summary["array_sha256"],
                                elapsed_seconds=time.monotonic()-started,
                                final_spin=summary["final_spin"],
                                final_mass_fraction=summary["final_mass_fraction"],
                                boundary_time_M=run["boundary_time_M"],
                                boundary_inputs=run["boundary_data"])
        q.write_evidence(args.json_out, report)
        print("independent_qnm_waveform_complete", flush=True)
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        result = _frequency_metrics(nt, nm, rt, rm, mass_seconds,
                                    plan["anchor_angular_frequency_M"])
        report["frequency_anchored"] = result
        report["baseline"] = dict(
            whole_complex_relative_l2=prior["comparison"]["metrics"]["complex_relative_l2"],
            postpeak_complex_relative_l2=prior["comparison"]["windows"]["post_peak"]["complex_relative_l2"])
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Repeat the aligned-spin interpolation control on the frozen finer grid."""

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
from .diagnose_interpolation_control import compare_interpolants
from .diagnose_peak_anchor import public_mode


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/fine_interpolation_control_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    old_path = q.ROOT / plan["prior_interpolation"]
    convergence_path = q.ROOT / plan["prior_convergence"]
    old, convergence = (json.loads(path.read_text()) for path in (old_path, convergence_path))
    choices = json.loads((q.ROOT / plan["convergence_plan"]).read_text())
    if (plan["schema"] != "diffgw.seobnrv4.fine-interpolation-control-plan.v1"
            or q.sha256(old_path) != plan["prior_interpolation_sha256"]
            or q.sha256(convergence_path) != plan["prior_convergence_sha256"]
            or old["status"] != "diagnostic_complete"
            or convergence["status"] != "diagnostic_complete"
            or plan["row_index"] != 1 or plan["level_name"] != "grid_2"
            or old["row"] != convergence["results"][1]["row"]):
        parser.error("frozen fine-grid plan or inputs differ")
    torch.set_num_threads(1)
    level = next(item for item in choices["grid_levels"] if item["name"] == plan["level_name"])
    settings = q.Settings(relative_l2_limit=choices["gates"]["complex_relative_l2"],
                          event_time_limit_M=choices["gates"]["event_time_difference_M"])
    row = old["row"]
    report = dict(schema="diffgw.seobnrv4.fine-interpolation-control.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  whole_waveform_parity_validated=False, plan_sha256=q.sha256(PLAN),
                  prior_sha256={"interpolation": q.sha256(old_path),
                                "convergence": q.sha256(convergence_path)},
                  script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in (
                      "dev/native_rom/eob/torch/warm_start.py",
                      "dev/native_rom/eob/torch/adaptive_driver_clean.py",
                      "src/torchwave/approximants/eob/merger_ringdown.py",
                      "dev/native_rom/qualify_generator.py")},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                               argv=sys.argv), row=row, result=None, errors=[])
    q.write_evidence(args.json_out, report)
    started = time.monotonic()
    try:
        run = q.native_run(
            row, level, settings,
            diagnostic_hermite_resampling=choices["hermite_resampling"],
            diagnostic_local_nqc_derivatives=choices["local_nqc_phase_derivatives"],
            diagnostic_restart_time_M=choices["fixed_restart_time_M"],
            diagnostic_coarse_initial_step_M=choices["coarse_first_step_M"],
            diagnostic_warm_start_Momega=choices["native_start_Momega"])
        native = q.summarize_run(run)
        frozen = convergence["results"][plan["row_index"]]["runs"]["grid_2"]
        if native["array_sha256"] != frozen["array_sha256"]:
            raise ValueError("fine-grid replay differs from frozen numerical solve")
        nt, nm, rt, rm, mass_seconds = public_mode(row, run)
        report["result"] = dict(native=native,
                                comparison=compare_interpolants(nt, nm, rt, rm,
                                                                mass_seconds, plan),
                                elapsed_seconds=time.monotonic() - started)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Repeat one native v4 solve against a finer pinned public output grid."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_interpolation_control import compare_interpolants
from .diagnose_peak_anchor import public_mode as public_at_native_step
from .diagnose_reference_grid import public_mode as public_at_step


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/fine_reference_control_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    frozen_path = q.ROOT / plan["prior_fine_interpolation"]
    reference_path = q.ROOT / plan["prior_reference_grid"]
    frozen = json.loads(frozen_path.read_text())
    reference = json.loads(reference_path.read_text())
    choices = json.loads((q.ROOT / plan["convergence_plan"]).read_text())
    if (plan["schema"] != "diffgw.seobnrv4.fine-reference-control-plan.v1"
            or q.sha256(frozen_path) != plan["prior_fine_interpolation_sha256"]
            or q.sha256(reference_path) != plan["prior_reference_grid_sha256"]
            or frozen["status"] != "diagnostic_complete"
            or reference["status"] != "diagnostic_complete"
            or frozen["row"] != reference["row"]
            or plan["reference_step_M"] != 0.03125
            or plan["native_level_name"] != "grid_2"
            or q.lal_binary_identity(reference["lal_binary"]["lal_version"],
                                     reference["lal_binary"]["lalsimulation_version"])
            != reference["lal_binary"]):
        parser.error("frozen plan, evidence, or public binary differs")
    torch.set_num_threads(1)
    level = next(item for item in choices["grid_levels"]
                 if item["name"] == plan["native_level_name"])
    settings = q.Settings(relative_l2_limit=choices["gates"]["complex_relative_l2"],
                          event_time_limit_M=choices["gates"]["event_time_difference_M"])
    row = frozen["row"]
    report = dict(schema="diffgw.seobnrv4.fine-reference-control.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  whole_waveform_parity_validated=False, plan_sha256=q.sha256(PLAN),
                  prior_sha256={"fine_interpolation": q.sha256(frozen_path),
                                "reference_grid": q.sha256(reference_path)},
                  script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in (
                      "dev/native_rom/eob/torch/warm_start.py",
                      "dev/native_rom/eob/torch/adaptive_driver_clean.py",
                      "src/torchwave/approximants/eob/merger_ringdown.py",
                      "dev/native_rom/qualify_generator.py")},
                  lal_binary=reference["lal_binary"],
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
        if native["array_sha256"] != frozen["result"]["native"]["array_sha256"]:
            raise ValueError("native replay differs from frozen fine-grid solve")
        nt, nm, rt0, rm0, mass_seconds = public_at_native_step(row, run)
        comparison0 = compare_interpolants(nt, nm, rt0, rm0, mass_seconds, plan)
        old = frozen["result"]["comparison"]["methods"]["quintic_complex"]["whole"]
        now = comparison0["methods"]["quintic_complex"]["whole"]
        if not np.isclose(now["complex_relative_l2"], old["complex_relative_l2"],
                          atol=1e-12, rtol=0):
            raise ValueError("same-grid public comparison differs from frozen result")
        rt1, rm1, mass1 = public_at_step(row, plan["reference_step_M"])
        if mass1 != mass_seconds:
            raise ValueError("mass seconds differ")
        comparison1 = compare_interpolants(nt, nm, rt1, rm1, mass1, plan)
        report["result"] = dict(native=native, same_grid=comparison0,
                                finer_reference=comparison1,
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

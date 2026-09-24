"""Frozen waveform and ODE check for the pinned v4 calibration decimals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch

from . import qualify_generator as q
from .diagnose_hamiltonian_source_conventions import source_coefficients
from .diagnose_peak_anchor import fitted_metrics, public_mode
from .eob.torch.source_calibration_hamiltonian import SOURCE_V4_COEFFICIENTS


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_calibration_waveform_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_waveform"]
    convention_path = q.ROOT / plan["same_state_conventions"]
    source_plan_path = q.ROOT / plan["source_convention_plan"]
    if (plan["schema"] != "diffgw.seobnrv4.source-calibration-waveform-plan.v1"
            or plan["levels"] != ["ode_0", "ode_1", "ode_2"]
            or plan["baseline_calibration"] != "published"
            or plan["candidate_calibration"] != "lal_v4_rounded"
            or q.sha256(prior_path) != plan["prior_waveform_sha256"]
            or q.sha256(convention_path) != plan["same_state_conventions_sha256"]
            or q.sha256(source_plan_path) != plan["source_convention_plan_sha256"]):
        parser.error("frozen plan or prior evidence differs")
    prior = json.loads(prior_path.read_text())
    convention = json.loads(convention_path.read_text())
    source_plan = json.loads(source_plan_path.read_text())
    if (prior["status"] != "diagnostic_complete"
            or not prior["check"]["numerical_check_passed"]
            or prior["release_qualified"] or convention["release_qualified"]
            or convention["status"] != "diagnostic_complete"):
        parser.error("prior prerequisites failed")
    parsed = source_coefficients(args.source_root, source_plan)
    embedded = {(name, i, j): value
                for name, rows in SOURCE_V4_COEFFICIENTS.items()
                for i, row in enumerate(rows) for j, value in enumerate(row)}
    if parsed != embedded or len(parsed) != source_plan["expected_coefficient_count"]:
        parser.error("embedded calibration differs from pinned source")
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity differs")
    numerical, physical = prior["choice"]["numerical"], prior["choice"]["physical"]
    placements = prior["choice"]["flux_placements"]
    settings, row = q.Settings(**prior["settings"]), prior["row"]
    torch.set_num_threads(1)
    sources = (
        "dev/native_rom/eob/torch/source_calibration_hamiltonian.py",
        "dev/native_rom/eob/torch/historical_hamiltonian.py",
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/qualify_generator.py",
        "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
        "src/torchwave/approximants/eob/seobnrv4_flux.py",
    )
    report = dict(
        schema="diffgw.seobnrv4.source-calibration-waveform.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        prior_waveform_sha256=q.sha256(prior_path),
        same_state_conventions_sha256=q.sha256(convention_path),
        source_coefficient_header_sha256=source_plan["source_coefficient_header_sha256"],
        source_sha256={path: q.sha256(q.ROOT / path) for path in sources},
        lal_binary=pin, row=row, settings=asdict(settings),
        choice=dict(numerical=numerical, physical=physical,
                    flux_placements=placements,
                    baseline_calibration=plan["baseline_calibration"],
                    candidate_calibration=plan["candidate_calibration"]),
        baseline=None, candidate={}, check=None, fitted_peak=None, errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        common = dict(
            diagnostic_hermite_resampling=numerical["hermite_resampling"],
            diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=numerical["fixed_restart_time_M"],
            diagnostic_coarse_initial_step_M=0.125,
            diagnostic_warm_start_Momega=0.03162277660168379,
            diagnostic_nqc_phase_window_M=2.0,
            diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
            diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
            diagnostic_mode_21_higher_spin=placements["mode_21_higher_spin"],
            diagnostic_mode_31_higher_spin=placements["mode_31_higher_spin"],
            diagnostic_mode_32_spin_square=placements["mode_32_spin_square"],
        )
        level = prior["candidate"]["ode_2"]["level"]
        start = time.monotonic()
        baseline = q.native_run(row, level, settings,
                                diagnostic_hamiltonian_calibration="published", **common)
        baseline_summary = q.summarize_run(baseline)
        if baseline_summary["array_sha256"] != prior["candidate"]["ode_2"]["native"]["array_sha256"]:
            raise ValueError("baseline no longer reproduces frozen waveform")
        report["baseline"] = dict(native=baseline_summary, level=level,
                                  elapsed_seconds=time.monotonic()-start)
        q.write_evidence(args.json_out, report)
        print("replay=baseline", flush=True)
        runs = []
        for name in plan["levels"]:
            level = prior["candidate"][name]["level"]
            start = time.monotonic()
            run = q.native_run(row, level, settings,
                               diagnostic_hamiltonian_calibration="lal_v4_rounded",
                               **common)
            report["candidate"][name] = dict(native=q.summarize_run(run), level=level,
                                             elapsed_seconds=time.monotonic()-start)
            runs.append(run)
            q.write_evidence(args.json_out, report)
            print(f"candidate={name}", flush=True)
        report["check"] = q.convergence_check(*runs, settings)
        nt, nm, rt, rm, mass_seconds = public_mode(row, runs[-1])
        report["fitted_peak"] = fitted_metrics(nt, nm, rt, rm, mass_seconds,
                                                "quadratic_power")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          passed=(report["check"] or {}).get("numerical_check_passed"))),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

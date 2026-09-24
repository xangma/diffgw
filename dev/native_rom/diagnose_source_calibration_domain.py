"""Check source-rounded calibration on two frozen boundary waveform rows."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_calibration_domain_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_domain"]
    source_plan_path = q.ROOT / plan["source_convention_plan"]
    aligned_path = q.ROOT / plan["aligned_waveform"]
    if (plan["schema"] != "diffgw.seobnrv4.source-calibration-domain-plan.v1"
            or plan["levels"] != ["ode_0", "ode_1", "ode_2"]
            or plan["rows"] != ["mass_ratio_four_nonspinning", "strong_antialigned_spin"]
            or plan["calibration"] != "lal_v4_rounded"
            or plan["release_qualified"] is not False
            or q.sha256(prior_path) != plan["prior_domain_sha256"]
            or q.sha256(source_plan_path) != plan["source_convention_plan_sha256"]
            or q.sha256(aligned_path) != plan["aligned_waveform_sha256"]):
        parser.error("frozen plan or input differs")
    prior = json.loads(prior_path.read_text())
    aligned = json.loads(aligned_path.read_text())
    source_plan = json.loads(source_plan_path.read_text())
    if (prior["status"] != "diagnostic_complete"
            or aligned["status"] != "diagnostic_complete"
            or not aligned["check"]["numerical_check_passed"]
            or set(prior["rows"]) != set(plan["rows"])
            or any(not prior["rows"][name]["check"]["numerical_check_passed"]
                   for name in plan["rows"])
            or prior["lal_binary"] != aligned["lal_binary"]):
        parser.error("prerequisite campaign is inconsistent")
    parsed = source_coefficients(args.source_root, source_plan)
    embedded = {(name, i, j): value
                for name, rows in SOURCE_V4_COEFFICIENTS.items()
                for i, row in enumerate(rows) for j, value in enumerate(row)}
    if parsed != embedded:
        parser.error("embedded calibration differs from pinned source")
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity differs")
    settings = q.Settings(**prior["settings"])
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
        schema="diffgw.seobnrv4.source-calibration-domain.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        prior_domain_sha256=q.sha256(prior_path),
        aligned_waveform_sha256=q.sha256(aligned_path),
        source_coefficient_header_sha256=source_plan["source_coefficient_header_sha256"],
        source_sha256={path: q.sha256(q.ROOT / path) for path in sources},
        lal_binary=pin, settings=asdict(settings),
        numerical_choice=prior["numerical_choice"],
        physical_choice=prior["physical_choice"],
        flux_placements=prior["flux_placements"],
        candidate_window_M=prior["candidate_window_M"],
        calibration=plan["calibration"], rows={}, errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    numerical = prior["numerical_choice"]
    physical = prior["physical_choice"]
    placements = prior["flux_placements"]
    for name in plan["rows"]:
        previous = prior["rows"][name]
        row = previous["row"]
        row_report = dict(
            row=row, fixed_restart_time_M=previous["fixed_restart_time_M"],
            baseline_fine_array_sha256=previous["candidate"]["ode_2"]["native"]["array_sha256"],
            baseline_fitted_peak_complex_relative_l2=(
                previous["fitted_peak"]["metrics"]["complex_relative_l2"]),
            candidate={}, check=None, fitted_peak=None, error=None)
        report["rows"][name] = row_report
        q.write_evidence(args.json_out, report)
        try:
            common = dict(
                diagnostic_hermite_resampling=numerical["hermite_resampling"],
                diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
                diagnostic_matching_second_derivatives=True,
                diagnostic_restart_time_M=previous["fixed_restart_time_M"],
                diagnostic_coarse_initial_step_M=0.125,
                diagnostic_warm_start_Momega=0.03162277660168379,
                diagnostic_nqc_phase_window_M=prior["candidate_window_M"],
                diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
                diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
                diagnostic_hamiltonian_calibration=plan["calibration"],
                diagnostic_mode_21_higher_spin=placements["mode_21_higher_spin"],
                diagnostic_mode_31_higher_spin=placements["mode_31_higher_spin"],
                diagnostic_mode_32_spin_square=placements["mode_32_spin_square"],
            )
            runs = []
            for level_name in plan["levels"]:
                level = previous["candidate"][level_name]["level"]
                started = time.monotonic()
                run = q.native_run(row, level, settings, **common)
                row_report["candidate"][level_name] = dict(
                    native=q.summarize_run(run), level=level,
                    elapsed_seconds=time.monotonic()-started)
                runs.append(run)
                q.write_evidence(args.json_out, report)
                print(f"row={name} candidate={level_name}", flush=True)
            row_report["check"] = q.convergence_check(*runs, settings)
            nt, nm, rt, rm, mass_seconds = public_mode(row, runs[-1])
            row_report["fitted_peak"] = fitted_metrics(
                nt, nm, rt, rm, mass_seconds, "quadratic_power")
        except Exception as exc:
            row_report["error"] = dict(type=type(exc).__name__, message=str(exc))
            report["errors"].append(dict(row=name, **row_report["error"]))
        q.write_evidence(args.json_out, report)
    report["status"] = "diagnostic_complete" if not report["errors"] else "failed"
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          passed={name: (item["check"] or {}).get("numerical_check_passed")
                                  for name, item in report["rows"].items()})), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

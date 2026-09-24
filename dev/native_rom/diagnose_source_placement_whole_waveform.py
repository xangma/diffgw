"""Frozen whole-waveform check for directly source-validated flux placements."""

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
from .diagnose_peak_anchor import fitted_metrics, public_mode


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_placement_whole_waveform_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_waveform"]
    flux_path = q.ROOT / plan["source_flux_result"]
    if (plan["schema"] != "diffgw.seobnrv4.source-placement-whole-waveform-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["levels"] != ["ode_0", "ode_1", "ode_2"]
            or plan["candidate_changes"] != {
                "spin_truncation": "tabulated",
                "mode_43_higher_spin": "lal_v4_rho"}
            or q.sha256(prior_path) != plan["prior_waveform_sha256"]
            or q.sha256(flux_path) != plan["source_flux_result_sha256"]):
        parser.error("frozen plan or prerequisites changed")
    prior = json.loads(prior_path.read_text())
    flux = json.loads(flux_path.read_text())
    if (prior["status"] != "diagnostic_complete"
            or not prior["check"]["numerical_check_passed"]
            or flux["status"] != "diagnostic_complete"
            or flux["release_qualified"] is not False
            or max(abs(row["source_placement_minus_source_flux_relative"])
                   for row in flux["rows"]) > plan["source_flux_max_relative_difference"]
            or max(abs(point["source_placement_minus_source_flux_relative"])
                   for group in flux["mode_trajectory"] for point in group["points"])
            > plan["source_flux_max_relative_difference"]
            or flux["source_placement_flux_choice"] != dict(
                spin_truncation="tabulated", retain_pan_cubic_spin=True,
                mode_21_higher_spin="rho_hybrid",
                mode_31_higher_spin="lal_v4_rho",
                mode_32_spin_square="lal_v4_zero",
                mode_33_higher_spin="lal_v4_rho",
                mode_43_higher_spin="lal_v4_rho")):
        parser.error("source or numerical prerequisite failed")
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity changed")
    row, settings = prior["row"], q.Settings(**prior["settings"])
    numerical, physical = prior["choice"]["numerical"], prior["choice"]["physical"]
    placements = prior["choice"]["flux_placements"]
    torch.set_num_threads(1)
    sources = (
        "src/torchwave/approximants/eob/seobnrv4_flux.py",
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/qualify_generator.py",
        "dev/native_rom/eob/torch/source_calibration_hamiltonian.py",
    )
    report = dict(
        schema="diffgw.seobnrv4.source-placement-whole-waveform.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        prior_waveform_sha256=q.sha256(prior_path),
        source_flux_result_sha256=q.sha256(flux_path),
        source_sha256={path: q.sha256(q.ROOT/path) for path in sources},
        lal_binary=pin, row=row, settings=asdict(settings),
        choice=dict(numerical=numerical, physical=physical,
                    baseline_flux_placements=placements,
                    candidate_changes=plan["candidate_changes"]),
        baseline=None, candidate={}, check=None, fitted_peak=None, errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        common = dict(
            diagnostic_hermite_resampling=numerical["hermite_resampling"],
            diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=numerical["fixed_restart_time_M"],
            diagnostic_coarse_initial_step_M=0.125,
            diagnostic_warm_start_Momega=0.03162277660168379,
            diagnostic_nqc_phase_window_M=prior["candidate_window_M"],
            diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
            diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
            diagnostic_mode_21_higher_spin=placements["mode_21_higher_spin"],
            diagnostic_mode_31_higher_spin=placements["mode_31_higher_spin"],
            diagnostic_mode_32_spin_square=placements["mode_32_spin_square"],
        )
        level = prior["candidate"]["ode_2"]["level"]
        start = time.monotonic()
        baseline = q.native_run(row, level, settings, **common)
        summary = q.summarize_run(baseline)
        if summary["array_sha256"] != prior["candidate"]["ode_2"]["native"]["array_sha256"]:
            raise ValueError("baseline no longer reproduces frozen waveform")
        report["baseline"] = dict(native=summary, level=level,
                                  elapsed_seconds=time.monotonic()-start)
        q.write_evidence(args.json_out, report)
        print("replay=baseline", flush=True)
        runs = []
        for name in plan["levels"]:
            level = prior["candidate"][name]["level"]
            start = time.monotonic()
            run = q.native_run(
                row, level, settings,
                diagnostic_spin_truncation=plan["candidate_changes"]["spin_truncation"],
                diagnostic_mode_43_higher_spin=plan["candidate_changes"]["mode_43_higher_spin"],
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

"""Frozen numerical and public-mode stress test for two untouched v4 rows."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_candidate_new_rows_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    expected_rows = {
        "q8_high_aligned": dict(mass1=40.0, mass2=5.0, chi1=0.85, chi2=0.85,
                                omega_start=0.04, fixed_restart_time_M=100.0),
        "q4_mixed_spin": dict(mass1=40.0, mass2=10.0, chi1=0.8, chi2=-0.6,
                              omega_start=0.04, fixed_restart_time_M=100.0),
    }
    if (plan["schema"] != "diffgw.seobnrv4.source-candidate-new-rows-plan.v1"
            or plan["levels"] != ["ode_0", "ode_1", "ode_2"]
            or plan["rows"] != expected_rows
            or plan["fixed_coarse_first_step_M"] != 0.125
            or plan["warm_start_Momega"] != 0.03162277660168379
            or plan["candidate_window_M"] != 1.0
            or plan["release_qualified"] is not False):
        parser.error("unexpected frozen plan")
    evidence = {}
    for label in ("prior_calibration_domain", "prior_source_placement"):
        path = q.ROOT / plan[label]
        if q.sha256(path) != plan[f"{label}_sha256"]:
            parser.error(f"{label} hash differs")
        evidence[label] = json.loads(path.read_text())
    calibration = evidence["prior_calibration_domain"]
    placement = evidence["prior_source_placement"]
    if (calibration["status"] != "diagnostic_complete"
            or placement["status"] != "diagnostic_complete"
            or not placement["check"]["numerical_check_passed"]
            or not all(row["check"]["numerical_check_passed"]
                       for row in calibration["rows"].values())
            or calibration["lal_binary"] != placement["lal_binary"]
            or calibration["settings"] != placement["settings"]):
        parser.error("prior evidence is inconsistent")
    pin = calibration["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity differs")
    settings = q.Settings(**calibration["settings"])
    numerical = calibration["numerical_choice"]
    physical = calibration["physical_choice"]
    flux = calibration["flux_placements"]
    if (calibration["calibration"] != "lal_v4_rounded"
            or numerical["hermite_resampling"] is not True
            or numerical["local_nqc_phase_derivatives"] is not True
            or physical != placement["choice"]["physical"]
            or flux != placement["choice"]["baseline_flux_placements"]
            or placement["choice"]["candidate_changes"] != {
                "mode_43_higher_spin": "lal_v4_rho", "spin_truncation": "tabulated"}):
        parser.error("candidate choices differ from frozen evidence")
    torch.set_num_threads(1)
    sources = (
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/eob/torch/historical_hamiltonian.py",
        "dev/native_rom/eob/torch/source_calibration_hamiltonian.py",
        "dev/native_rom/qualify_generator.py",
        "src/torchwave/approximants/eob/seobnrv4_flux.py",
    )
    report = dict(
        schema="diffgw.seobnrv4.source-candidate-new-rows.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        input_sha256={key: plan[f"{key}_sha256"] for key in evidence},
        source_sha256={path: q.sha256(q.ROOT / path) for path in sources},
        lal_binary=pin, settings=asdict(settings),
        choice=dict(numerical=numerical, physical=physical, flux=flux,
                    calibration="lal_v4_rounded", mode_43_higher_spin="lal_v4_rho",
                    spin_truncation="tabulated", nqc_phase_window_M=1.0),
        rows={}, errors=[], process=dict(
            host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    levels = {item["name"]: item for item in q.run_plan(settings)}
    for name, row_plan in plan["rows"].items():
        row = {key: value for key, value in row_plan.items()
               if key != "fixed_restart_time_M"}
        q.validate_rows([row])
        item = dict(row=row, fixed_restart_time_M=row_plan["fixed_restart_time_M"],
                    runs={}, check=None, fitted_peak=None, error=None)
        report["rows"][name] = item
        q.write_evidence(args.json_out, report)
        try:
            common = dict(
                diagnostic_hermite_resampling=True,
                diagnostic_local_nqc_derivatives=True,
                diagnostic_matching_second_derivatives=True,
                diagnostic_restart_time_M=row_plan["fixed_restart_time_M"],
                diagnostic_coarse_initial_step_M=plan["fixed_coarse_first_step_M"],
                diagnostic_warm_start_Momega=plan["warm_start_Momega"],
                diagnostic_nqc_phase_window_M=plan["candidate_window_M"],
                diagnostic_mode_21_higher_spin=flux["mode_21_higher_spin"],
                diagnostic_mode_31_higher_spin=flux["mode_31_higher_spin"],
                diagnostic_mode_32_spin_square=flux["mode_32_spin_square"],
                diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
                diagnostic_mode_43_higher_spin="lal_v4_rho",
                diagnostic_spin_truncation="tabulated",
                diagnostic_hamiltonian_log_derivative=physical[
                    "hamiltonian_log_derivative"],
                diagnostic_hamiltonian_calibration="lal_v4_rounded",
            )
            runs = []
            for level_name in plan["levels"]:
                started = time.monotonic()
                run = q.native_run(row, levels[level_name], settings, **common)
                item["runs"][level_name] = dict(
                    native=q.summarize_run(run), level=levels[level_name],
                    elapsed_seconds=time.monotonic()-started)
                runs.append(run)
                q.write_evidence(args.json_out, report)
                print(f"row={name} level={level_name}", flush=True)
            item["check"] = q.convergence_check(*runs, settings)
            nt, nm, rt, rm, mass_seconds = public_mode(row, runs[-1])
            item["fitted_peak"] = fitted_metrics(
                nt, nm, rt, rm, mass_seconds, "quadratic_power")
        except Exception as exc:
            item["error"] = dict(type=type(exc).__name__, message=str(exc))
            report["errors"].append(dict(row=name, **item["error"]))
        q.write_evidence(args.json_out, report)
    report["status"] = "diagnostic_complete" if not report["errors"] else "failed"
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          passed={name: (item["check"] or {}).get("numerical_check_passed")
                                  for name, item in report["rows"].items()})), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen source-backed flux and 1 M NQC check on two boundary rows."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_one_M_domain_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    rows = {
        "mass_ratio_four_nonspinning": {"fixed_restart_time_M": 165.0},
        "strong_antialigned_spin": {"fixed_restart_time_M": 100.0},
    }
    flux = {
        "mode_21_higher_spin": "rho_hybrid",
        "mode_31_higher_spin": "lal_v4_rho",
        "mode_32_spin_square": "lal_v4_zero",
    }
    if (plan["schema"] != "diffgw.seobnrv4.source-flux-one-M-domain-plan.v1"
            or plan["rows"] != rows or plan["flux_placements"] != flux
            or plan["candidate_window_M"] != 1.0
            or plan["fixed_coarse_first_step_M"] != 0.125
            or plan["warm_start_Momega"] != 0.03162277660168379
            or plan["levels"] != ["ode_0", "ode_1", "ode_2"]
            or plan["release_qualified"] is not False):
        parser.error("unexpected frozen plan")
    inputs = {}
    for name, sha_key in (
            ("baseline_domain_result", "baseline_domain_result_sha256"),
            ("aligned_source_flux_result", "aligned_source_flux_result_sha256"),
            ("antialigned_restart_plan", "antialigned_restart_plan_sha256")):
        path = q.ROOT / plan[name]
        if q.sha256(path) != plan[sha_key]:
            parser.error(f"frozen {name} differs")
        inputs[name] = json.loads(path.read_text())
    baseline = inputs["baseline_domain_result"]
    aligned = inputs["aligned_source_flux_result"]
    restart = inputs["antialigned_restart_plan"]
    if (baseline["status"] != "diagnostic_complete"
            or aligned["status"] != "diagnostic_complete"
            or baseline["candidate_window_M"] != 1.0
            or aligned["candidate_window_M"] != 1.0
            or restart["fixed_restart_time_M"] != 100.0
            or restart["row_name"] != "strong_antialigned_spin"
            or set(baseline["rows"]) != set(rows)
            or any(baseline["rows"][name]["fixed_restart_time_M"]
                   != row_plan["fixed_restart_time_M"]
                   or not baseline["rows"][name]["check"]["numerical_check_passed"]
                   for name, row_plan in rows.items())
            or not aligned["check"]["numerical_check_passed"]
            or aligned["choice"]["flux_placements"] != flux
            or aligned["choice"]["numerical"] != baseline["numerical_choice"]
            or aligned["choice"]["physical"] != baseline["candidate_choice"]
            or aligned["settings"] != baseline["settings"]
            or aligned["lal_binary"] != baseline["lal_binary"]):
        parser.error("frozen diagnostic inputs are inconsistent")
    sources = dict(baseline["source_sha256"])
    for path, digest in aligned["source_sha256"].items():
        if path in sources and sources[path] != digest:
            parser.error(f"frozen source identities disagree: {path}")
        sources[path] = digest
    if any(q.sha256(q.ROOT / path) != digest for path, digest in sources.items()):
        parser.error("native source differs from frozen diagnostics")
    pin = baseline["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity differs")
    settings = q.Settings(**baseline["settings"])
    numerical = baseline["numerical_choice"]
    physical = baseline["candidate_choice"]
    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.source-flux-one-M-domain.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        input_sha256={name: plan[key] for name, key in (
            ("baseline_domain_result", "baseline_domain_result_sha256"),
            ("aligned_source_flux_result", "aligned_source_flux_result_sha256"),
            ("antialigned_restart_plan", "antialigned_restart_plan_sha256"))},
        source_sha256=sources, lal_binary=pin, settings=asdict(settings),
        candidate_window_M=plan["candidate_window_M"],
        numerical_choice=numerical, physical_choice=physical,
        flux_placements=flux, rows={}, errors=[], process=dict(
            host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    for name, row_plan in rows.items():
        prior = baseline["rows"][name]
        row = prior["row"]
        row_report = dict(
            row=row, fixed_restart_time_M=row_plan["fixed_restart_time_M"],
            baseline_check=prior["check"],
            baseline_fitted_peak_complex_relative_l2=(
                prior["fitted_peak"]["metrics"]["complex_relative_l2"]),
            baseline_fine_array_sha256=prior["candidate"]["ode_2"]["native"]["array_sha256"],
            candidate={}, check=None, fitted_peak=None, error=None)
        report["rows"][name] = row_report
        q.write_evidence(args.json_out, report)
        try:
            common = dict(
                diagnostic_hermite_resampling=numerical["hermite_resampling"],
                diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
                diagnostic_matching_second_derivatives=True,
                diagnostic_restart_time_M=row_plan["fixed_restart_time_M"],
                diagnostic_coarse_initial_step_M=plan["fixed_coarse_first_step_M"],
                diagnostic_warm_start_Momega=plan["warm_start_Momega"],
                diagnostic_nqc_phase_window_M=plan["candidate_window_M"],
                diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
                diagnostic_hamiltonian_log_derivative=physical[
                    "hamiltonian_log_derivative"],
                diagnostic_mode_21_higher_spin=flux["mode_21_higher_spin"],
                diagnostic_mode_31_higher_spin=flux["mode_31_higher_spin"],
                diagnostic_mode_32_spin_square=flux["mode_32_spin_square"],
            )
            runs = []
            for level_name in plan["levels"]:
                level = prior["candidate"][level_name]["level"]
                print(f"row={name} candidate={level_name}", flush=True)
                started = time.monotonic()
                run = q.native_run(row, level, settings, **common)
                row_report["candidate"][level_name] = dict(
                    native=q.summarize_run(run), level=level,
                    elapsed_seconds=time.monotonic()-started)
                runs.append(run)
                q.write_evidence(args.json_out, report)
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
    print(json.dumps(dict(
        status=report["status"], errors=report["errors"],
        passed={name: (item["check"] or {}).get("numerical_check_passed")
                for name, item in report["rows"].items()})), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

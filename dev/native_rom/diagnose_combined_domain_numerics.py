"""Frozen boundary-row ODE stability check for the combined native v4 candidate."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/combined_domain_numerics_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_combined_numerics"]
    rows_path = q.ROOT / plan["prior_boundary_rows"]
    if (plan["schema"] != "diffgw.seobnrv4.combined-domain-numerics-plan.v1"
            or q.sha256(prior_path) != plan["prior_combined_numerics_sha256"]
            or q.sha256(rows_path) != plan["prior_boundary_rows_sha256"]
            or plan["row_names"] != ["mass_ratio_four_nonspinning",
                                       "strong_aligned_spin", "strong_antialigned_spin"]
            or plan["ode_tolerances"] != [
                {"atol": 1e-12, "rtol": 1e-11},
                {"atol": 1e-13, "rtol": 1e-12},
                {"atol": 1e-14, "rtol": 1e-13}]
            or plan["fixed_coarse_first_step_M"] != 0.125
            or plan["warm_start_Momega"] != 0.03162277660168379):
        parser.error("frozen domain plan or prior evidence differs")
    prior = json.loads(prior_path.read_text())
    boundary = json.loads(rows_path.read_text())
    if (prior["status"] != "diagnostic_complete"
            or not prior["check"]["numerical_check_passed"]
            or boundary["status"] not in {"failed", "diagnostic_complete"}):
        parser.error("expected frozen combined numerical pass and boundary rows")
    rows = {name: boundary["rows"][name]["inputs"] for name in plan["row_names"]}
    q.validate_rows(list(rows.values()))
    settings = q.Settings(**prior["settings"])
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity differs")
    numerical, candidate = prior["numerical_choice"], prior["candidate_choice"]
    torch.set_num_threads(1)
    sources = ["src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
               "src/torchwave/approximants/eob/seobnrv4_flux.py",
               "dev/native_rom/eob/torch/historical_hamiltonian.py",
               "dev/native_rom/eob/torch/warm_start.py",
               "dev/native_rom/eob/torch/adaptive_driver_clean.py",
               "dev/native_rom/qualify_generator.py"]
    report = dict(
        schema="diffgw.seobnrv4.combined-domain-numerics.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), prior_sha256=dict(
            combined=q.sha256(prior_path), boundary=q.sha256(rows_path)),
        source_sha256={name: q.sha256(q.ROOT / name) for name in sources},
        script_sha256=q.sha256(__file__), settings=asdict(settings),
        numerical_choice=numerical, candidate_choice=candidate,
        lal_binary=pin, rows={}, errors=[], process=dict(
            host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
            argv=sys.argv))
    q.write_evidence(args.json_out, report)
    for name, row in rows.items():
        item = dict(row=row, runs={}, check=None, fitted_peak=None, error=None)
        report["rows"][name] = item
        runs = []
        try:
            for i, tolerances in enumerate(plan["ode_tolerances"]):
                level = dict(name=f"ode_{i}", dt_geom=0.125, **tolerances)
                print(f"row={name} run=ode_{i}", flush=True)
                started = time.monotonic()
                run = q.native_run(
                    row, level, settings,
                    diagnostic_hermite_resampling=numerical["hermite_resampling"],
                    diagnostic_local_nqc_derivatives=numerical[
                        "local_nqc_phase_derivatives"],
                    diagnostic_matching_second_derivatives=True,
                    diagnostic_restart_time_M=numerical["fixed_restart_time_M"],
                    diagnostic_coarse_initial_step_M=plan["fixed_coarse_first_step_M"],
                    diagnostic_warm_start_Momega=plan["warm_start_Momega"],
                    diagnostic_mode_33_higher_spin=candidate["mode_33_higher_spin"],
                    diagnostic_hamiltonian_log_derivative=candidate[
                        "hamiltonian_log_derivative"])
                item["runs"][level["name"]] = dict(
                    level=level, native=q.summarize_run(run),
                    elapsed_seconds=time.monotonic()-started)
                runs.append(run)
                q.write_evidence(args.json_out, report)
            item["check"] = q.convergence_check(*runs, settings)
            nt, nm, rt, rm, mass_seconds = public_mode(row, runs[-1])
            item["fitted_peak"] = fitted_metrics(nt, nm, rt, rm, mass_seconds,
                                                   "quadratic_power")
        except Exception as exc:
            item["error"] = dict(type=type(exc).__name__, message=str(exc))
            report["errors"].append(dict(row=name, **item["error"]))
        q.write_evidence(args.json_out, report)
    report["status"] = "diagnostic_complete" if not report["errors"] else "failed"
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          passed={name: (item["check"] or {}).get(
                              "numerical_check_passed") for name, item in
                              report["rows"].items()})), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

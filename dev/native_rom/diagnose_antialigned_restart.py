"""Recheck the antialigned boundary row with a predeclared earlier restart."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/antialigned_restart_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_campaign"]
    if (plan["schema"] != "diffgw.seobnrv4.antialigned-restart-plan.v1"
            or q.sha256(prior_path) != plan["prior_campaign_sha256"]
            or plan["row_name"] != "strong_antialigned_spin"
            or plan["fixed_restart_time_M"] != 100.0
            or plan["fixed_coarse_first_step_M"] != 0.125
            or plan["warm_start_Momega"] != 0.03162277660168379
            or plan["ode_tolerances"] != [
                {"atol": 1e-12, "rtol": 1e-11},
                {"atol": 1e-13, "rtol": 1e-12},
                {"atol": 1e-14, "rtol": 1e-13}]):
        parser.error("frozen antialigned plan or prior evidence differs")
    prior = json.loads(prior_path.read_text())
    item = prior["rows"][plan["row_name"]]
    if (prior["status"] != "failed" or item["runs"]
            or item["error"]["message"] !=
            "continuous match_time must lie inside the full input grid"):
        parser.error("expected frozen antialigned fine-grid failure")
    settings = q.Settings(**prior["settings"])
    numerical, candidate = prior["numerical_choice"], prior["candidate_choice"]
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity differs")
    torch.set_num_threads(1)
    source_paths = [
        "dev/native_rom/eob/torch/historical_hamiltonian.py",
        "dev/native_rom/eob/torch/warm_start.py",
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/qualify_generator.py",
    ]
    report = dict(
        schema="diffgw.seobnrv4.antialigned-restart.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), prior_sha256=q.sha256(prior_path),
        script_sha256=q.sha256(__file__),
        source_sha256={p: q.sha256(q.ROOT / p) for p in source_paths},
        row=item["row"], settings=asdict(settings), numerical_choice=numerical,
        candidate_choice=candidate, lal_binary=pin,
        fixed_restart_time_M=plan["fixed_restart_time_M"],
        runs={}, check=None, fitted_peak=None, errors=[], process=dict(
            host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    runs = []
    try:
        for i, tolerances in enumerate(plan["ode_tolerances"]):
            level = dict(name=f"ode_{i}", dt_geom=0.125, **tolerances)
            print(f"run=ode_{i}", flush=True)
            started = time.monotonic()
            run = q.native_run(
                item["row"], level, settings,
                diagnostic_hermite_resampling=numerical["hermite_resampling"],
                diagnostic_local_nqc_derivatives=numerical[
                    "local_nqc_phase_derivatives"],
                diagnostic_matching_second_derivatives=True,
                diagnostic_restart_time_M=plan["fixed_restart_time_M"],
                diagnostic_coarse_initial_step_M=plan["fixed_coarse_first_step_M"],
                diagnostic_warm_start_Momega=plan["warm_start_Momega"],
                diagnostic_mode_33_higher_spin=candidate["mode_33_higher_spin"],
                diagnostic_hamiltonian_log_derivative=candidate[
                    "hamiltonian_log_derivative"])
            report["runs"][level["name"]] = dict(
                level=level, native=q.summarize_run(run),
                elapsed_seconds=time.monotonic()-started)
            runs.append(run)
            q.write_evidence(args.json_out, report)
        report["check"] = q.convergence_check(*runs, settings)
        nt, nm, rt, rm, mass_seconds = public_mode(item["row"], runs[-1])
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

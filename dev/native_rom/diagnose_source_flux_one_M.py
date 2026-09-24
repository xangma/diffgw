"""Frozen one-M NQC numerical check on the source-backed flux candidate."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_one_M_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-flux-one-M-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["candidate_window_M"] != 1.0
            or plan["baseline_window_M"] != 2.0
            or plan["levels"] != ["ode_0", "ode_1", "ode_2"]):
        parser.error("unexpected frozen plan")
    prior_path = q.ROOT / plan["source_flux_result"]
    dense_path = q.ROOT / plan["dense_reference"]
    if (q.sha256(prior_path) != plan["source_flux_result_sha256"]
            or q.sha256(dense_path) != plan["dense_reference_sha256"]):
        parser.error("frozen prerequisite evidence changed")
    prior = json.loads(prior_path.read_text())
    dense = json.loads(dense_path.read_text())
    if (prior["status"] != "diagnostic_complete"
            or not prior["check"]["numerical_check_passed"]
            or dense["status"] != "diagnostic_complete"):
        parser.error("frozen baseline numerical evidence is invalid")
    row, settings = prior["row"], q.Settings(**prior["settings"])
    choices = prior["choice"]
    numerical, physical, flux = (choices[key] for key in
                                 ("numerical", "physical", "flux_placements"))
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("pinned public binary differs")
    sources = {path: q.sha256(q.ROOT / path) for path in prior["source_sha256"]}
    wrapper = "dev/native_rom/qualify_generator.py"
    if (any(sources[path] != digest for path, digest in prior["source_sha256"].items()
            if path != wrapper)
            or sources[wrapper] != plan["current_wrapper_sha256"]):
        parser.error("source-backed candidate implementation changed")
    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.source-flux-one-M.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        source_flux_result_sha256=q.sha256(prior_path),
        dense_reference_sha256=q.sha256(dense_path), source_sha256=sources,
        wrapper_change=plan["wrapper_change"],
        lal_binary=pin, row=row, settings=asdict(settings), choice=choices,
        candidate_window_M=plan["candidate_window_M"], candidate={}, check=None,
        fitted_peak=None, errors=[], process=dict(
            host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        common = dict(
            diagnostic_hermite_resampling=numerical["hermite_resampling"],
            diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=numerical["fixed_restart_time_M"],
            diagnostic_coarse_initial_step_M=0.125,
            diagnostic_warm_start_Momega=0.03162277660168379,
            diagnostic_nqc_phase_window_M=plan["candidate_window_M"],
            diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
            diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
            diagnostic_mode_21_higher_spin=flux["mode_21_higher_spin"],
            diagnostic_mode_31_higher_spin=flux["mode_31_higher_spin"],
            diagnostic_mode_32_spin_square=flux["mode_32_spin_square"],
        )
        runs = []
        for name in plan["levels"]:
            level = prior["candidate"][name]["level"]
            started = time.monotonic()
            run = q.native_run(row, level, settings, **common)
            report["candidate"][name] = dict(
                native=q.summarize_run(run), level=level,
                elapsed_seconds=time.monotonic() - started)
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

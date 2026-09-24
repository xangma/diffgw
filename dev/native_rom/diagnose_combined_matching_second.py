"""Frozen NQC second-derivative sensitivity of the combined native v4 candidate."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/combined_matching_second_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_convergence"]
    trace_path = q.ROOT / plan["prior_tolerance_trace"]
    if (plan["schema"] != "diffgw.seobnrv4.combined-matching-second-plan.v1"
            or q.sha256(prior_path) != plan["prior_convergence_sha256"]
            or q.sha256(trace_path) != plan["prior_tolerance_trace_sha256"]
            or plan["levels"] != ["ode_coarse", "ode_medium", "middle"]
            or plan["matching_second_choice"] is not True):
        parser.error("frozen matching-second plan or input differs")
    prior, trace = (json.loads(path.read_text()) for path in (prior_path, trace_path))
    if (prior["status"] != "diagnostic_complete"
            or trace["status"] != "diagnostic_complete"
            or prior["checks"]["ode"]["numerical_check_passed"]
            or trace["comparison"]["pre_attachment"]["raw_mode"][
                "complex_relative_l2"] >= 1e-8):
        parser.error("expected frozen NQC-dominated ODE failure")
    wave_plan = json.loads((q.ROOT / "dev/native_rom/evidence/2026-09-24/"
                            "combined_warmstart_waveform_plan.json").read_text())
    wave_path = q.ROOT / "dev/native_rom/evidence/2026-09-24/combined_warmstart_waveform.json"
    wave = json.loads(wave_path.read_text())
    pin = wave["lal_binary"]
    if (wave["status"] != "diagnostic_complete"
            or q.sha256(wave_path) != prior["prior_sha256"]
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("public binary or frozen waveform differs")
    settings = q.Settings(**prior["settings"])
    row, choice, mode = (prior[key] for key in
                         ("row", "numerical_choice", "candidate_choice"))
    torch.set_num_threads(1)
    sources = ["src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
               "src/torchwave/approximants/eob/seobnrv4_flux.py",
               "dev/native_rom/eob/torch/historical_hamiltonian.py",
               "dev/native_rom/eob/torch/warm_start.py",
               "dev/native_rom/eob/torch/adaptive_driver_clean.py",
               "dev/native_rom/qualify_generator.py"]
    report = dict(
        schema="diffgw.seobnrv4.combined-matching-second.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), prior_sha256=dict(
            convergence=q.sha256(prior_path), trace=q.sha256(trace_path)),
        source_sha256={name: q.sha256(q.ROOT / name) for name in sources},
        script_sha256=q.sha256(__file__), row=row, settings=asdict(settings),
        numerical_choice=choice, candidate_choice=mode,
        lal_binary=pin, baseline_fitted_peak=wave["results"]["combined"]["fitted_peak"],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv), runs={}, check=None, fitted_peak=None,
        errors=[])
    q.write_evidence(args.json_out, report)
    runs = {}
    try:
        for name in plan["levels"]:
            level = prior["runs"][name]["level"]
            print(f"run={name}", flush=True)
            started = time.monotonic()
            run = q.native_run(
                row, level, settings,
                diagnostic_hermite_resampling=choice["hermite_resampling"],
                diagnostic_local_nqc_derivatives=choice["local_nqc_phase_derivatives"],
                diagnostic_matching_second_derivatives=True,
                diagnostic_restart_time_M=choice["fixed_restart_time_M"],
                diagnostic_coarse_initial_step_M=0.125,
                diagnostic_warm_start_Momega=wave_plan["warm_start_Momega"],
                diagnostic_mode_33_higher_spin=mode["mode_33_higher_spin"],
                diagnostic_hamiltonian_log_derivative=mode["hamiltonian_log_derivative"])
            report["runs"][name] = dict(
                level=level, native=q.summarize_run(run),
                elapsed_seconds=time.monotonic()-started)
            runs[name] = run
            q.write_evidence(args.json_out, report)
        report["check"] = q.convergence_check(
            runs["ode_coarse"], runs["ode_medium"], runs["middle"], settings)
        nt, nm, rt, rm, mass_seconds = public_mode(row, runs["middle"])
        report["fitted_peak"] = fitted_metrics(nt, nm, rt, rm, mass_seconds,
                                                "quadratic_power")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          numerical_passed=(report["check"] or {}).get(
                              "numerical_check_passed"))), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

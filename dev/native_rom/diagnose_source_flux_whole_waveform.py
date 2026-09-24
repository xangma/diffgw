"""Frozen waveform and ODE check for source-backed v4 flux placements."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_whole_waveform_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_waveform"]
    flux_path = q.ROOT / plan["same_state_flux"]
    if (plan["schema"] != "diffgw.seobnrv4.source-flux-whole-waveform-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["levels"] != ["ode_0", "ode_1", "ode_2"]
            or plan["baseline"] != {
                "mode_21_higher_spin": "taylor_f", "mode_31_higher_spin": "taylor_f",
                "mode_32_spin_square": "lal_v4_zero"}
            or plan["candidate"] != {
                "mode_21_higher_spin": "rho_hybrid", "mode_31_higher_spin": "lal_v4_rho",
                "mode_32_spin_square": "lal_v4_zero"}
            or q.sha256(prior_path) != plan["prior_waveform_sha256"]
            or q.sha256(flux_path) != plan["same_state_flux_sha256"]):
        parser.error("frozen plan or prior evidence differs")
    prior = json.loads(prior_path.read_text())
    flux = json.loads(flux_path.read_text())
    if (prior["status"] != "diagnostic_complete"
            or not prior["check"]["numerical_check_passed"]
            or flux["status"] != "diagnostic_complete"
            or prior["row"] != flux["row"]):
        parser.error("prior waveform or flux prerequisite failed")
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity differs")
    numerical, candidate = prior["choice"]["numerical"], prior["choice"]["physical"]
    settings, row = q.Settings(**prior["settings"]), prior["row"]
    torch.set_num_threads(1)
    sources = (
        "src/torchwave/approximants/eob/seobnrv4_flux.py",
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/qualify_generator.py",
        "dev/native_rom/eob/torch/historical_hamiltonian.py",
    )
    report = dict(
        schema="diffgw.seobnrv4.source-flux-whole-waveform.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        prior_waveform_sha256=q.sha256(prior_path), same_state_flux_sha256=q.sha256(flux_path),
        source_sha256={path: q.sha256(q.ROOT / path) for path in sources},
        lal_binary=pin, row=row, settings=asdict(settings),
        choice=dict(numerical=numerical, physical=candidate,
                    flux_placements=plan["candidate"]),
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
            diagnostic_mode_33_higher_spin=candidate["mode_33_higher_spin"],
            diagnostic_hamiltonian_log_derivative=candidate["hamiltonian_log_derivative"],
        )
        level = prior["candidate"]["ode_2"]["level"]
        start = time.monotonic()
        baseline = q.native_run(
            row, level, settings,
            diagnostic_mode_21_higher_spin=plan["baseline"]["mode_21_higher_spin"],
            diagnostic_mode_31_higher_spin=plan["baseline"]["mode_31_higher_spin"],
            diagnostic_mode_32_spin_square=plan["baseline"]["mode_32_spin_square"],
            **common)
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
            run = q.native_run(
                row, level, settings,
                diagnostic_mode_21_higher_spin=plan["candidate"]["mode_21_higher_spin"],
                diagnostic_mode_31_higher_spin=plan["candidate"]["mode_31_higher_spin"],
                diagnostic_mode_32_spin_square=plan["candidate"]["mode_32_spin_square"],
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

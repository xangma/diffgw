"""Frozen whole-waveform ODE check for a wider local NQC phase window."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics, public_mode


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/strong_spin_nqc_window_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.strong-spin-nqc-window-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["baseline_window_M"] != 0.5
            or plan["candidate_window_M"] != 2.0
            or plan["levels"] != ["ode_0", "ode_1", "ode_2"]):
        parser.error("unexpected frozen plan")
    campaign_path = q.ROOT / plan["campaign"]
    windows_path = q.ROOT / plan["window_sensitivity"]
    if (q.sha256(campaign_path) != plan["campaign_sha256"]
            or q.sha256(windows_path) != plan["window_sensitivity_sha256"]):
        parser.error("frozen evidence hash differs")
    campaign = json.loads(campaign_path.read_text())
    windows = json.loads(windows_path.read_text())
    row_report = campaign["rows"][plan["row_name"]]
    if (campaign["status"] != "failed"
            or windows["status"] != "diagnostic_complete"
            or row_report["check"]["numerical_check_passed"]):
        parser.error("expected frozen strong-spin numerical failure")
    row = row_report["row"]
    settings = q.Settings(**campaign["settings"])
    numerical, candidate = campaign["numerical_choice"], campaign["candidate_choice"]
    pin = campaign["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity differs")
    torch.set_num_threads(1)
    sources = [
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/eob/torch/historical_hamiltonian.py",
        "src/torchwave/approximants/eob/numerical_attachment.py",
        "dev/native_rom/qualify_generator.py",
    ]
    report = dict(
        schema="diffgw.seobnrv4.strong-spin-nqc-window.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        campaign_sha256=q.sha256(campaign_path),
        window_sensitivity_sha256=q.sha256(windows_path),
        source_sha256={p: q.sha256(q.ROOT / p) for p in sources},
        lal_binary=pin, row=row, settings=asdict(settings),
        baseline=None, candidate={}, check=None, fitted_peak=None, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        common = dict(
            diagnostic_hermite_resampling=numerical["hermite_resampling"],
            diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=numerical["fixed_restart_time_M"],
            diagnostic_coarse_initial_step_M=plan["fixed_coarse_first_step_M"],
            diagnostic_warm_start_Momega=plan["warm_start_Momega"],
            diagnostic_mode_33_higher_spin=candidate["mode_33_higher_spin"],
            diagnostic_hamiltonian_log_derivative=candidate["hamiltonian_log_derivative"],
        )
        level = row_report["runs"]["ode_2"]["level"]
        started = time.monotonic()
        baseline = q.native_run(
            row, level, settings, diagnostic_nqc_phase_window_M=0.5, **common)
        baseline_summary = q.summarize_run(baseline)
        if baseline_summary["array_sha256"] != row_report["runs"]["ode_2"]["native"]["array_sha256"]:
            raise ValueError("default window no longer reproduces frozen waveform")
        report["baseline"] = dict(native=baseline_summary, level=level,
                                  elapsed_seconds=time.monotonic()-started)
        q.write_evidence(args.json_out, report)
        print("replay=baseline", flush=True)
        runs = []
        for name in plan["levels"]:
            level = row_report["runs"][name]["level"]
            started = time.monotonic()
            run = q.native_run(
                row, level, settings, diagnostic_nqc_phase_window_M=2.0, **common)
            report["candidate"][name] = dict(
                native=q.summarize_run(run), level=level,
                elapsed_seconds=time.monotonic()-started)
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

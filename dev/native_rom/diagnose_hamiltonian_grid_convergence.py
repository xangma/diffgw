"""Frozen three-grid convergence check of the source-motivated v4 candidate."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch

from . import qualify_generator as q
from .compare_whole_waveforms import aligned_metrics
from .diagnose_peak_anchor import fitted_metrics, public_mode


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/hamiltonian_grid_convergence_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_whole_waveform"]
    if (plan["schema"] != "diffgw.seobnrv4.hamiltonian-grid-convergence-plan.v1"
            or q.sha256(prior_path) != plan["prior_whole_waveform_sha256"]
            or plan["grid_steps_M"] != [0.25, 0.125, 0.0625]
            or plan["peak_fit"] != "quadratic_power"):
        parser.error("frozen plan or prior result differs")
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "diagnostic_complete":
        parser.error("prior waveform result did not complete")
    row, original_level = prior["row"], prior["level"]
    settings, choice = q.Settings(**prior["settings"]), prior["numerical_choice"]
    torch.set_num_threads(1)
    sources = ["src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
               "src/torchwave/approximants/eob/seobnrv4_flux.py",
               "dev/native_rom/eob/torch/historical_hamiltonian.py",
               "dev/native_rom/eob/torch/adaptive_driver_clean.py",
               "dev/native_rom/qualify_generator.py"]
    report = dict(
        schema="diffgw.seobnrv4.hamiltonian-grid-convergence.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), prior_sha256=q.sha256(prior_path),
        source_sha256={name: q.sha256(q.ROOT / name) for name in sources},
        script_sha256=q.sha256(__file__), row=row, settings=asdict(settings),
        numerical_choice=choice,
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv), levels=[], convergence=None, errors=[])
    q.write_evidence(args.json_out, report)
    runs = []
    try:
        for step in plan["grid_steps_M"]:
            level = dict(original_level, dt_geom=step, name=f"grid_{step:g}")
            print(f"grid={step:g} M", flush=True)
            started = time.monotonic()
            run = q.native_run(
                row, level, settings,
                diagnostic_hermite_resampling=choice["hermite_resampling"],
                diagnostic_local_nqc_derivatives=choice["local_nqc_phase_derivatives"],
                diagnostic_restart_time_M=choice["fixed_restart_time_M"],
                diagnostic_mode_33_higher_spin="lal_v4_rho",
                diagnostic_hamiltonian_log_derivative="lal_v4")
            summary = q.summarize_run(run)
            if (step == 0.125
                    and summary["array_sha256"] != prior["results"]["candidate"]["native"]["array_sha256"]):
                raise ValueError("middle grid differs from frozen candidate")
            nt, nm, rt, rm, mass_seconds = public_mode(row, run)
            report["levels"].append(dict(
                level=level, native=summary,
                sampled_peak=aligned_metrics(nt, nm, rt, rm, mass_seconds),
                fitted_peak=fitted_metrics(nt, nm, rt, rm, mass_seconds, "quadratic_power"),
                elapsed_seconds=time.monotonic()-started))
            runs.append(run)
            q.write_evidence(args.json_out, report)
        strict = replace(settings, dt_levels_M=tuple(plan["grid_steps_M"]))
        report["convergence"] = q.convergence_check(*runs, strict)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          numerical_check_passed=(report["convergence"] or {}).get(
                              "numerical_check_passed"))), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

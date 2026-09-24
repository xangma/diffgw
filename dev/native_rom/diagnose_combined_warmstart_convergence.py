"""Frozen numerical convergence checks for the combined native v4 candidate."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/combined_warmstart_convergence_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_waveform"]
    if (plan["schema"] != "diffgw.seobnrv4.combined-warmstart-convergence-plan.v1"
            or q.sha256(prior_path) != plan["prior_waveform_sha256"]
            or plan["grid_steps_M"] != [0.25, 0.125, 0.0625]
            or plan["fixed_coarse_first_step_M"] != 0.125
            or plan["warm_start_Momega"] != 0.03162277660168379
            or plan["ode_tolerances"] != [
                {"atol": 1e-12, "rtol": 1e-11},
                {"atol": 1e-13, "rtol": 1e-12},
                {"atol": 1e-14, "rtol": 1e-13}]):
        parser.error("frozen plan or prior evidence differs")
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "diagnostic_complete":
        parser.error("prior waveform diagnostic did not complete")
    settings = q.Settings(**prior["settings"])
    if (settings.relative_l2_limit != plan["relative_l2_limit"]
            or settings.event_time_limit_M != plan["event_time_limit_M"]):
        parser.error("numerical budgets differ")
    row, choice = prior["row"], prior["numerical_choice"]
    mode = prior["results"]["combined"]["choice"]
    torch.set_num_threads(1)
    sources = ["src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
               "src/torchwave/approximants/eob/seobnrv4_flux.py",
               "dev/native_rom/eob/torch/historical_hamiltonian.py",
               "dev/native_rom/eob/torch/warm_start.py",
               "dev/native_rom/eob/torch/adaptive_driver_clean.py",
               "dev/native_rom/qualify_generator.py"]
    report = dict(
        schema="diffgw.seobnrv4.combined-warmstart-convergence.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), prior_sha256=q.sha256(prior_path),
        source_sha256={name: q.sha256(q.ROOT / name) for name in sources},
        script_sha256=q.sha256(__file__), row=row, settings=asdict(settings),
        numerical_choice=choice, candidate_choice=mode,
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv), runs={}, checks=None, errors=[])
    q.write_evidence(args.json_out, report)
    runs = {}
    try:
        levels = [("middle", dict(prior["level"])),
                  ("grid_coarse", dict(prior["level"], dt_geom=0.25)),
                  ("grid_fine", dict(prior["level"], dt_geom=0.0625)),
                  ("ode_coarse", dict(prior["level"], **plan["ode_tolerances"][0])),
                  ("ode_medium", dict(prior["level"], **plan["ode_tolerances"][1]))]
        for name, level in levels:
            level["name"] = name
            print(f"run={name}", flush=True)
            started = time.monotonic()
            run = q.native_run(
                row, level, settings,
                diagnostic_hermite_resampling=choice["hermite_resampling"],
                diagnostic_local_nqc_derivatives=choice["local_nqc_phase_derivatives"],
                diagnostic_restart_time_M=choice["fixed_restart_time_M"],
                diagnostic_coarse_initial_step_M=plan["fixed_coarse_first_step_M"],
                diagnostic_warm_start_Momega=plan["warm_start_Momega"],
                diagnostic_mode_33_higher_spin=mode["mode_33_higher_spin"],
                diagnostic_hamiltonian_log_derivative=mode["hamiltonian_log_derivative"])
            summary = q.summarize_run(run)
            if (name == "middle"
                    and summary["array_sha256"] != prior["results"]["combined"]["native"]["array_sha256"]):
                raise ValueError("middle run differs from frozen candidate")
            report["runs"][name] = dict(level=level, native=summary,
                                        elapsed_seconds=time.monotonic()-started)
            runs[name] = run
            q.write_evidence(args.json_out, report)
        report["checks"] = dict(
            grid=q.convergence_check(runs["grid_coarse"], runs["middle"],
                                     runs["grid_fine"], settings),
            ode=q.convergence_check(runs["ode_coarse"], runs["ode_medium"],
                                    runs["middle"], settings))
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          passed={key: value["numerical_check_passed"] for key, value in
                                  (report["checks"] or {}).items()})), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Nested-grid stability for two numerically stable boundary candidates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch

from . import qualify_generator as q


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/boundary_grid_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.boundary-grid-plan.v1"
            or plan["grid_steps_M"] != [0.25, 0.125, 0.0625]
            or plan["fixed_coarse_first_step_M"] != 0.125
            or plan["warm_start_Momega"] != 0.03162277660168379
            or list(plan["rows"]) != ["mass_ratio_four_nonspinning",
                                       "strong_antialigned_spin"]):
        parser.error("frozen boundary grid plan differs")
    priors = {}
    for name, spec in plan["rows"].items():
        path = q.ROOT / spec["prior"]
        if q.sha256(path) != spec["prior_sha256"]:
            parser.error(f"prior evidence for {name} differs")
        prior = json.loads(path.read_text())
        prior_check = (prior["rows"][name]["check"] if "rows" in prior
                       else prior["check"])
        if not prior_check["numerical_check_passed"]:
            parser.error(f"prior ODE check for {name} failed")
        priors[name] = prior
    torch.set_num_threads(1)
    source_paths = [
        "dev/native_rom/eob/torch/historical_hamiltonian.py",
        "dev/native_rom/eob/torch/warm_start.py",
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/qualify_generator.py",
    ]
    report = dict(
        schema="diffgw.seobnrv4.boundary-grid.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN),
        prior_sha256={name: spec["prior_sha256"] for name, spec in plan["rows"].items()},
        script_sha256=q.sha256(__file__),
        source_sha256={p: q.sha256(q.ROOT / p) for p in source_paths},
        rows={}, errors=[], process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                                         pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    for name, spec in plan["rows"].items():
        prior = priors[name]
        old = prior["rows"][name] if "rows" in prior else prior
        row = old["row"]
        settings = q.Settings(**prior["settings"])
        numerical, candidate = prior["numerical_choice"], prior["candidate_choice"]
        item = dict(row=row, fixed_restart_time_M=spec["fixed_restart_time_M"],
                    levels={}, check=None, error=None)
        report["rows"][name] = item
        runs = []
        try:
            for step in plan["grid_steps_M"]:
                level = dict(name=f"grid_{step:g}", dt_geom=step,
                             atol=1e-14, rtol=1e-13)
                print(f"row={name} grid={step:g}", flush=True)
                started = time.monotonic()
                run = q.native_run(
                    row, level, settings,
                    diagnostic_hermite_resampling=numerical["hermite_resampling"],
                    diagnostic_local_nqc_derivatives=numerical[
                        "local_nqc_phase_derivatives"],
                    diagnostic_matching_second_derivatives=True,
                    diagnostic_restart_time_M=spec["fixed_restart_time_M"],
                    diagnostic_coarse_initial_step_M=plan["fixed_coarse_first_step_M"],
                    diagnostic_warm_start_Momega=plan["warm_start_Momega"],
                    diagnostic_mode_33_higher_spin=candidate["mode_33_higher_spin"],
                    diagnostic_hamiltonian_log_derivative=candidate[
                        "hamiltonian_log_derivative"])
                summary = q.summarize_run(run)
                if step == 0.125 and summary["array_sha256"] != old["runs"]["ode_2"]["native"]["array_sha256"]:
                    raise ValueError("middle grid differs from frozen ODE run")
                item["levels"][level["name"]] = dict(
                    level=level, native=summary, elapsed_seconds=time.monotonic()-started)
                runs.append(run)
                q.write_evidence(args.json_out, report)
            item["check"] = q.convergence_check(*runs, settings)
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

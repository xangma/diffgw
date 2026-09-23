"""Frozen whole-waveform self-convergence after native inline pre-evolution."""

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


EVIDENCE = q.ROOT / "dev/native_rom/evidence/2026-09-23"
PLAN = EVIDENCE / "inline_warmstart_convergence_plan.json"


def _run(row, level, plan, settings):
    return q.native_run(
        row, level, settings,
        diagnostic_hermite_resampling=plan["hermite_resampling"],
        diagnostic_local_nqc_derivatives=plan["local_nqc_phase_derivatives"],
        diagnostic_restart_time_M=plan["fixed_restart_time_M"],
        diagnostic_coarse_initial_step_M=plan["coarse_first_step_M"],
        diagnostic_warm_start_Momega=plan["native_start_Momega"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_inline"]
    if (plan["schema"] != "diffgw.seobnrv4.inline-warmstart-convergence-plan.v1"
            or plan["rows"] != [0, 1]
            or q.sha256(prior_path) != plan["prior_inline_sha256"]
            or plan["ode_levels"][-1]["dt_geom"] != plan["grid_levels"][0]["dt_geom"]):
        parser.error("frozen convergence plan or input mismatch")
    prior = json.loads(prior_path.read_text())
    if (prior["status"] != "diagnostic_complete"
            or any(not item["diagnostic_passed"] for item in prior["results"])):
        parser.error("prior inline warm-start diagnostic did not pass")
    settings = q.Settings(
        relative_l2_limit=plan["gates"]["complex_relative_l2"],
        event_time_limit_M=plan["gates"]["event_time_difference_M"])
    report = dict(schema="diffgw.seobnrv4.inline-warmstart-convergence.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  physical_parity_validated=False, plan_sha256=q.sha256(PLAN),
                  prior_inline_sha256=q.sha256(prior_path),
                  script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in (
                      "dev/native_rom/eob/torch/warm_start.py",
                      "dev/native_rom/eob/torch/adaptive_driver_clean.py",
                      "dev/native_rom/qualify_generator.py")},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), results=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for index in plan["rows"]:
            row = prior["results"][index]["row"]
            runs = {}
            summaries = {}
            levels = [*plan["ode_levels"], *plan["grid_levels"][1:]]
            for level in levels:
                began = time.monotonic()
                run = _run(row, level, plan, settings)
                run["elapsed_seconds"] = time.monotonic()-began
                runs[level["name"]] = run
                summaries[level["name"]] = q.summarize_run(run)
                print(json.dumps(dict(index=index, name=level["name"],
                                      elapsed_seconds=run["elapsed_seconds"])), flush=True)
            ode = q.convergence_check(*(runs[level["name"]]
                                        for level in plan["ode_levels"]), settings)
            grid = q.convergence_check(*(runs[level["name"]]
                                         for level in plan["grid_levels"]), settings)
            gates = dict(ode=ode["numerical_check_passed"],
                         grid=grid["numerical_check_passed"],
                         ode_amplitude=ode["medium_to_fine"]["amplitude_relative_l2"]
                         <= plan["gates"]["amplitude_relative_l2"],
                         grid_amplitude=grid["medium_to_fine"]["amplitude_relative_l2"]
                         <= plan["gates"]["amplitude_relative_l2"])
            result = dict(row=row, runs=summaries, ode=ode, grid=grid,
                          gates=gates, numerical_check_passed=all(gates.values()))
            report["results"].append(result)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(index=index, gates=gates,
                                  ode_l2=ode["medium_to_fine"]["complex_relative_l2"],
                                  grid_l2=grid["medium_to_fine"]["complex_relative_l2"])),
                  flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen native-warm-start convergence candidate with local NQC second derivatives."""

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
PLAN = EVIDENCE / "inline_warmstart_local_second_plan.json"
PRIOR_PLAN = EVIDENCE / "inline_warmstart_convergence_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_convergence"]
    prior = json.loads(prior_path.read_text())
    choices = json.loads(PRIOR_PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.inline-warmstart-local-second-plan.v1"
            or q.sha256(prior_path) != plan["prior_convergence_sha256"]
            or prior["status"] != "diagnostic_complete"
            or q.sha256(PRIOR_PLAN) != prior["plan_sha256"]
            or plan["row_indices"] != [1, 0]
            or plan["local_second_window_M"] != 0.5
            or plan["local_second_degree"] != 6
            or plan["matching_second_derivatives"] is not True
            or plan["retain_prior_warm_start_restart_first_step_and_levels"] is not True
            or plan["gates"]["complex_relative_l2"] != choices["gates"]["complex_relative_l2"]
            or plan["gates"]["amplitude_relative_l2"] != choices["gates"]["amplitude_relative_l2"]
            or plan["gates"]["event_time_difference_M"] != choices["gates"]["event_time_difference_M"]):
        parser.error("frozen local-second plan or convergence input mismatch")

    settings = q.Settings(
        relative_l2_limit=plan["gates"]["complex_relative_l2"],
        event_time_limit_M=plan["gates"]["event_time_difference_M"])
    report = dict(schema="diffgw.seobnrv4.inline-warmstart-local-second.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), prior_sha256=q.sha256(prior_path),
                  script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT/path) for path in (
                      "dev/native_rom/eob/torch/warm_start.py",
                      "dev/native_rom/eob/torch/adaptive_driver_clean.py",
                      "dev/native_rom/qualify_generator.py")},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), results=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for index in plan["row_indices"]:
            row = prior["results"][index]["row"]
            runs = {}
            summaries = {}
            levels = [*choices["ode_levels"], *choices["grid_levels"][1:]]
            for level in levels:
                began = time.monotonic()
                run = q.native_run(
                    row, level, settings,
                    diagnostic_hermite_resampling=choices["hermite_resampling"],
                    diagnostic_local_nqc_derivatives=choices["local_nqc_phase_derivatives"],
                    diagnostic_matching_second_derivatives=True,
                    diagnostic_restart_time_M=choices["fixed_restart_time_M"],
                    diagnostic_coarse_initial_step_M=choices["coarse_first_step_M"],
                    diagnostic_warm_start_Momega=choices["native_start_Momega"])
                run["elapsed_seconds"] = time.monotonic()-began
                runs[level["name"]] = run
                summaries[level["name"]] = q.summarize_run(run)
                print(json.dumps(dict(index=index, level=level["name"],
                                      seconds=run["elapsed_seconds"])), flush=True)
            ode = q.convergence_check(*(runs[level["name"]]
                                        for level in choices["ode_levels"]), settings)
            grid = q.convergence_check(*(runs[level["name"]]
                                         for level in choices["grid_levels"]), settings)
            gates = dict(ode=ode["numerical_check_passed"],
                         grid=grid["numerical_check_passed"],
                         ode_amplitude=ode["medium_to_fine"]["amplitude_relative_l2"]
                         <= plan["gates"]["amplitude_relative_l2"],
                         grid_amplitude=grid["medium_to_fine"]["amplitude_relative_l2"]
                         <= plan["gates"]["amplitude_relative_l2"])
            report["results"].append(dict(row=row, runs=summaries,
                                          ode=ode, grid=grid, gates=gates,
                                          numerical_check_passed=all(gates.values())))
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

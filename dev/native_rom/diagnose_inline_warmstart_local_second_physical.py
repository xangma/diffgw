"""Paired, pinned-public-binary check of the local-second derivative candidate."""

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
from .diagnose_peak_anchor import fitted_metrics, public_mode


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/inline_warmstart_local_second_physical_plan.json"
CONVERGENCE_PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/inline_warmstart_convergence_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_convergence"]
    candidate_plan_path = q.ROOT / plan["local_second_plan"]
    candidate_path = q.ROOT / plan["local_second_result"]
    binary_path = q.ROOT / plan["pinned_public_waveform"]
    prior = json.loads(prior_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    binary = json.loads(binary_path.read_text())
    choices = json.loads(CONVERGENCE_PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.inline-warmstart-local-second-physical-plan.v1"
            or q.sha256(prior_path) != plan["prior_convergence_sha256"]
            or q.sha256(candidate_plan_path) != plan["local_second_plan_sha256"]
            or q.sha256(binary_path) != plan["pinned_public_waveform_sha256"]
            or any(item["status"] != "diagnostic_complete" for item in (prior, candidate, binary))
            or candidate["plan_sha256"] != q.sha256(candidate_plan_path)
            or plan["row_indices"] != [0, 1] or plan["level_name"] != "ode_2"
            or not all(result["numerical_check_passed"] for result in candidate["results"])
            or [item["row"] for item in candidate["results"]] != [prior["results"][i]["row"]
                                                                for i in [1, 0]]
            or binary["rows"] != [prior["results"][i]["row"] for i in plan["row_indices"]]
            or q.lal_binary_identity(binary["lal_binary"]["lal_version"],
                                     binary["lal_binary"]["lalsimulation_version"])
            != binary["lal_binary"]):
        parser.error("frozen physical-comparison plan or inputs differ")
    level = next(item for item in choices["ode_levels"]
                 if item["name"] == plan["level_name"])
    settings = q.Settings(relative_l2_limit=choices["gates"]["complex_relative_l2"],
                          event_time_limit_M=choices["gates"]["event_time_difference_M"])
    report = dict(schema="diffgw.seobnrv4.inline-warmstart-local-second-physical.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  whole_waveform_parity_validated=False,
                  plan_sha256=q.sha256(PLAN),
                  input_sha256=dict(prior=q.sha256(prior_path),
                                    candidate=q.sha256(candidate_path),
                                    binary=q.sha256(binary_path)),
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
            comparisons = {}
            for local_second in (False, True):
                name = "local_second" if local_second else "baseline"
                started = time.monotonic()
                run = q.native_run(
                    row, level, settings,
                    diagnostic_hermite_resampling=choices["hermite_resampling"],
                    diagnostic_local_nqc_derivatives=choices["local_nqc_phase_derivatives"],
                    diagnostic_matching_second_derivatives=local_second,
                    diagnostic_restart_time_M=choices["fixed_restart_time_M"],
                    diagnostic_coarse_initial_step_M=choices["coarse_first_step_M"],
                    diagnostic_warm_start_Momega=choices["native_start_Momega"])
                frozen = (candidate["results"][1-index] if local_second
                          else prior["results"][index])["runs"]["ode_2"]
                summary = q.summarize_run(run)
                if summary["array_sha256"] != frozen["array_sha256"]:
                    raise ValueError(f"{name} row {index} differs from frozen numerical solve")
                native_time, native_mode, reference_time, reference_mode, mass_seconds = (
                    public_mode(row, run))
                comparison = fitted_metrics(native_time, native_mode,
                                            reference_time, reference_mode,
                                            mass_seconds, "quadratic_power")
                runs[name] = run
                comparisons[name] = dict(waveform=summary, fitted_peak=comparison,
                                         elapsed_seconds=time.monotonic()-started)
                print(json.dumps(dict(index=index, choice=name,
                                      l2=comparison["metrics"]["complex_relative_l2"],
                                      seconds=comparisons[name]["elapsed_seconds"])), flush=True)
            report["results"].append(dict(
                row=row, comparisons=comparisons,
                candidate_minus_baseline_l2=(
                    comparisons["local_second"]["fitted_peak"]["metrics"]["complex_relative_l2"]
                    -comparisons["baseline"]["fitted_peak"]["metrics"]["complex_relative_l2"]),
                native_pair=q.compare_runs(runs["local_second"], runs["baseline"])))
            q.write_evidence(args.json_out, report)
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

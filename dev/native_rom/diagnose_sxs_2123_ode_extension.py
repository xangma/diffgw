"""Extend the selected SXS-row ODE study and localize its residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import qualify_generator as q


def regions(candidate: dict, reference: dict, peak_time: float) -> dict:
    if not np.array_equal(candidate["time"], reference["time"]):
        raise ValueError("regional diagnostic requires one exact output grid")
    time_axis = reference["time"]
    lhs, rhs = candidate["mode"], reference["mode"]
    error = abs(lhs-rhs)**2
    total_error = float(error.sum())
    total_reference = float((abs(rhs)**2).sum())
    windows = {"inspiral": time_axis < peak_time-100.,
               "late_inspiral": (time_axis >= peak_time-100.) & (time_axis < peak_time),
               "post_peak": time_axis >= peak_time}
    result = {}
    for name, mask in windows.items():
        if mask.sum() < 3:
            raise ValueError(f"empty diagnostic region: {name}")
        result[name] = dict(
            samples=int(mask.sum()), metrics=q.mode_metrics(lhs[mask], rhs[mask]),
            error_energy_fraction=float(error[mask].sum()/total_error),
            global_relative_l2_contribution=float(
                np.sqrt(error[mask].sum()/total_reference)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ode-plan", type=Path, required=True)
    parser.add_argument("--prior-ode", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    extension = json.loads(args.ode_plan.read_text())
    prior = json.loads(args.prior_ode.read_text())
    levels = extension["levels"]
    if (extension["schema"] != "diffgw.seobnrv4.sxs-2123-ode-extension-plan.v1"
            or extension["prior_ode_sha256"] != q.sha256(args.prior_ode)
            or prior["plan_sha256"] != q.sha256(args.plan)
            or [level["name"] for level in levels]
            != ["strict_replay", "tighter", "tightest"]
            or {k: v for k, v in levels[0].items() if k != "name"}
            != {k: v for k, v in prior["runs"]["strict"]["level"].items()
                if k != "name"}):
        raise ValueError("ODE extension does not match its frozen inputs")
    torch.set_num_threads(1)
    runs = []
    summaries = {}
    for level in levels:
        started = time.monotonic()
        run = q.native_run(plan["row"], level, q.Settings(max_run_seconds=600.),
                           return_nqc_coefficients=True, return_boundary_data=True)
        summary = q.summarize_run(run)
        if (level["name"] == "strict_replay" and summary["array_sha256"]
                != prior["runs"]["strict"]["native"]["array_sha256"]):
            raise ValueError("strict waveform changed since the earlier ODE study")
        summaries[level["name"]] = dict(level=level, native=summary,
                                         nqc_coefficients=run["nqc_coefficients"],
                                         boundary_data=run["boundary_data"],
                                         elapsed_seconds=time.monotonic()-started)
        runs.append(run)
        print(f"{level['name']} complete", flush=True)
    budget = extension["stability_budget"]
    checked = q.convergence_check(*runs, settings=q.Settings(
        max_run_seconds=600.,
        relative_l2_limit=budget["tightest_vs_tighter_complex_relative_l2_max"],
        event_time_limit_M=budget["tightest_vs_tighter_event_time_difference_M_max"]))
    checked["gates"]["amplitude_error_budget"] = (
        checked["medium_to_fine"]["amplitude_relative_l2"]
        <= budget["tightest_vs_tighter_amplitude_relative_l2_max"])
    checked["gates"]["error_decreased"] = (
        checked["medium_to_fine"]["complex_relative_l2"]
        <= checked["coarse_to_medium"]["complex_relative_l2"]
        + budget["error_decreases_allowing_roundoff"])
    checked["numerical_check_passed"] = all(checked["gates"].values())
    peak_time = prior["runs"]["strict"]["native"]["peak_time_M"]
    diagnostic = dict(
        peak_time_M=peak_time,
        strict_to_tighter=regions(runs[0], runs[1], peak_time),
        tighter_to_tightest=regions(runs[1], runs[2], peak_time))
    result = dict(schema="diffgw.seobnrv4.sxs-2123-ode-extension.v1",
                  plan_sha256=q.sha256(args.plan),
                  extension_plan_sha256=q.sha256(args.ode_plan),
                  prior_ode_sha256=q.sha256(args.prior_ode),
                  script_sha256=q.sha256(__file__),
                  generator_source_manifest_sha256=q.canonical_hash(q.source_snapshot()),
                  row=plan["row"], runs=summaries, convergence=checked,
                  regional_diagnostic=diagnostic, release_qualified=False,
                  interpretation="One selected Torch waveform's numerical tolerance trace. "
                                 "It does not bound domain-wide physical or ROM error.")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(numerical_check_passed=checked["numerical_check_passed"],
                          complex_relative_l2=checked["medium_to_fine"][
                              "complex_relative_l2"])), flush=True)


if __name__ == "__main__":
    main()

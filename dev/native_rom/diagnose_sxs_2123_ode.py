"""Check post-hoc ODE tolerance stability of the pinned SXS:BBH:2123 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from . import qualify_generator as q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--ode-plan", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    ode_plan = json.loads(args.ode_plan.read_text())
    prior = json.loads(args.prior.read_text())
    levels = ode_plan["levels"]
    if (plan["schema"] != "diffgw.seobnrv4.sxs-2123-plan.v1"
            or ode_plan["schema"] != "diffgw.seobnrv4.sxs-2123-ode-plan.v1"
            or ode_plan["primary_comparison_sha256"] != q.sha256(args.prior)
            or prior["plan_sha256"] != q.sha256(args.plan)
            or [level["name"] for level in levels] != ["baseline", "refined", "strict"]
            or levels[0] != dict(name="baseline", atol=plan["method"]["native_atol"],
                                 rtol=plan["method"]["native_rtol"],
                                 dt_geom=plan["method"]["grid_step_M"])):
        raise ValueError("ODE study does not match the frozen primary comparison")
    budget = ode_plan["stability_budget"]
    settings = q.Settings(max_run_seconds=600.)
    torch.set_num_threads(1)
    runs = []
    summaries = {}
    for level in levels:
        started = time.monotonic()
        run = q.native_run(plan["row"], level, settings)
        summary = q.summarize_run(run)
        if level["name"] == "baseline" and summary["array_sha256"] != prior["native"][
                "native"]["array_sha256"]:
            raise ValueError("baseline native waveform changed")
        summaries[level["name"]] = dict(level=level, native=summary,
                                         elapsed_seconds=time.monotonic()-started)
        runs.append(run)
        print(f"{level['name']} complete", flush=True)
    checked = q.convergence_check(*runs, settings=q.Settings(
        max_run_seconds=600.,
        relative_l2_limit=budget["strict_vs_refined_complex_relative_l2_max"],
        event_time_limit_M=budget["strict_vs_refined_event_time_difference_M_max"]))
    # The shared helper uses one budget for both norms. Keep the plan's
    # separately frozen amplitude bound explicit in the result.
    checked["gates"]["amplitude_error_budget"] = (
        checked["medium_to_fine"]["amplitude_relative_l2"]
        <= budget["strict_vs_refined_amplitude_relative_l2_max"])
    checked["gates"]["error_decreased"] = (
        checked["medium_to_fine"]["complex_relative_l2"]
        <= checked["coarse_to_medium"]["complex_relative_l2"]
        + budget["error_decreases_allowing_roundoff"])
    checked["numerical_check_passed"] = all(checked["gates"].values())
    result = dict(schema="diffgw.seobnrv4.sxs-2123-ode-diagnostic.v1",
                  plan_sha256=q.sha256(args.plan),
                  ode_plan_sha256=q.sha256(args.ode_plan),
                  primary_comparison_sha256=q.sha256(args.prior),
                  script_sha256=q.sha256(__file__),
                  generator_source_manifest_sha256=q.canonical_hash(q.source_snapshot()),
                  row=plan["row"], runs=summaries, convergence=checked,
                  release_qualified=False,
                  interpretation="One selected row's Torch numerical stability only; "
                                 "this neither bounds physical error nor qualifies the ROM.")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(numerical_check_passed=checked["numerical_check_passed"],
                          complex_relative_l2=checked["medium_to_fine"][
                              "complex_relative_l2"])), flush=True)


if __name__ == "__main__":
    main()

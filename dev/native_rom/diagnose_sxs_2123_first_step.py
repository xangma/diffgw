"""Test whether the coarse first trial controls SXS-row phase convergence."""

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
    parser.add_argument("--first-step-plan", type=Path, required=True)
    parser.add_argument("--prior-extension", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    first = json.loads(args.first_step_plan.read_text())
    prior = json.loads(args.prior_extension.read_text())
    levels = first["levels"]
    if (first["schema"] != "diffgw.seobnrv4.sxs-2123-first-step-plan.v1"
            or first["prior_extension_sha256"] != q.sha256(args.prior_extension)
            or prior["plan_sha256"] != q.sha256(args.plan)
            or [level["name"] for level in levels] != ["tighter", "tightest"]
            or any(level != prior["runs"][level["name"]]["level"] for level in levels)
            or first["coarse_first_step_M"] != 0.125):
        raise ValueError("first-step comparison does not match frozen inputs")
    torch.set_num_threads(1)
    runs = []
    summaries = {}
    for level in levels:
        started = time.monotonic()
        run = q.native_run(plan["row"], level, q.Settings(max_run_seconds=600.),
                           diagnostic_coarse_initial_step_M=first["coarse_first_step_M"])
        summaries[level["name"]] = dict(
            native=q.summarize_run(run), elapsed_seconds=time.monotonic()-started,
            original=prior["runs"][level["name"]]["native"])
        runs.append(run)
        print(f"{level['name']} complete", flush=True)
    comparison = q.compare_runs(*runs)
    result = dict(schema="diffgw.seobnrv4.sxs-2123-first-step.v1",
                  plan_sha256=q.sha256(args.plan),
                  first_step_plan_sha256=q.sha256(args.first_step_plan),
                  prior_extension_sha256=q.sha256(args.prior_extension),
                  script_sha256=q.sha256(__file__),
                  generator_source_manifest_sha256=q.canonical_hash(q.source_snapshot()),
                  row=plan["row"], coarse_first_step_M=first["coarse_first_step_M"],
                  runs=summaries, comparison=comparison,
                  below_original_complex_error=(comparison["complex_relative_l2"]
                                                < prior["convergence"]["medium_to_fine"]
                                                ["complex_relative_l2"]),
                  numerical_budget_passed=(comparison["complex_relative_l2"]
                                           <= first["numerical_budget_complex_relative_l2"]),
                  release_qualified=False,
                  interpretation="A one-row first-trial-step sensitivity diagnostic, "
                                 "not a source-backed numerical prescription.")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(complex_relative_l2=comparison["complex_relative_l2"],
                          numerical_budget_passed=result["numerical_budget_passed"])),
          flush=True)


if __name__ == "__main__":
    main()

"""Isolate fine-stage output sampling on the frozen SXS:BBH:2123 row."""

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
    parser.add_argument("--sampling-plan", type=Path, required=True)
    parser.add_argument("--prior-extension", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    sampling = json.loads(args.sampling_plan.read_text())
    prior = json.loads(args.prior_extension.read_text())
    name = sampling["level_name"]
    if (sampling["schema"] != "diffgw.seobnrv4.sxs-2123-fine-sampling-plan.v1"
            or sampling["prior_extension_sha256"] != q.sha256(args.prior_extension)
            or prior["plan_sha256"] != q.sha256(args.plan)
            or name != "tightest"
            or sampling["variant"] != "fine_stage_rkf45_output_sampling_only"
            or sampling["numerical_budget_complex_relative_l2"] != 1e-6):
        raise ValueError("fine-sampling comparison does not match frozen inputs")
    level = prior["runs"][name]["level"]
    torch.set_num_threads(1)
    runs = {}
    summaries = {}
    for label, alternative in (("default_replay", False), ("fine_rkf45", True)):
        started = time.monotonic()
        run = q.native_run(plan["row"], level, q.Settings(max_run_seconds=600.),
                           diagnostic_fine_rkf45_resampling=alternative,
                           return_accepted_hashes=True)
        summary = q.summarize_run(run)
        if (label == "default_replay" and summary["array_sha256"]
                != prior["runs"][name]["native"]["array_sha256"]):
            raise ValueError("default waveform differs from frozen tightest solve")
        runs[label] = run
        summaries[label] = dict(native=summary, elapsed_seconds=time.monotonic()-started)
        print(f"{label} complete", flush=True)
    accepted_equal = (runs["default_replay"]["accepted_array_sha256"]
                      == runs["fine_rkf45"]["accepted_array_sha256"])
    if not accepted_equal:
        raise ValueError("fine-sampling switch changed an accepted ODE array")
    comparison = q.compare_runs(runs["fine_rkf45"], runs["default_replay"])
    drift = prior["convergence"]["medium_to_fine"]["complex_relative_l2"]
    result = dict(schema="diffgw.seobnrv4.sxs-2123-fine-sampling.v1",
                  plan_sha256=q.sha256(args.plan),
                  sampling_plan_sha256=q.sha256(args.sampling_plan),
                  prior_extension_sha256=q.sha256(args.prior_extension),
                  script_sha256=q.sha256(__file__),
                  generator_source_manifest_sha256=q.canonical_hash(q.source_snapshot()),
                  row=plan["row"], level=level, runs=summaries,
                  accepted_arrays_identical=accepted_equal, comparison=comparison,
                  prior_tolerance_drift_complex_relative_l2=drift,
                  sampling_to_tolerance_drift_ratio=comparison["complex_relative_l2"] / drift,
                  numerical_budget_passed=(comparison["complex_relative_l2"]
                                           <= sampling["numerical_budget_complex_relative_l2"]),
                  release_qualified=False,
                  interpretation="One-row fine-stage output-sampling diagnostic with the "
                                 "accepted ODE histories held byte-identical; no physical "
                                 "model or domain validation.")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(complex_relative_l2=comparison["complex_relative_l2"],
                          sampling_to_tolerance_drift_ratio=(result[
                              "sampling_to_tolerance_drift_ratio"]),
                          numerical_budget_passed=result["numerical_budget_passed"])),
          flush=True)


if __name__ == "__main__":
    main()

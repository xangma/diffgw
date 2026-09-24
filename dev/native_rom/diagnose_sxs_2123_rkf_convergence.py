"""Check SXS-row numerical convergence with fine-stage RKF45 sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from . import qualify_generator as q
from .diagnose_sxs_2123_ode_extension import regions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--convergence-plan", type=Path, required=True)
    parser.add_argument("--prior-extension", type=Path, required=True)
    parser.add_argument("--prior-sampling", type=Path, required=True)
    parser.add_argument("--prior-reference", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.plan.read_text())
    frozen = json.loads(args.convergence_plan.read_text())
    extension = json.loads(args.prior_extension.read_text())
    sampling = json.loads(args.prior_sampling.read_text())
    reference = json.loads(args.prior_reference.read_text())
    if (frozen["schema"] != "diffgw.seobnrv4.sxs-2123-rkf-convergence-plan.v1"
            or frozen["prior_extension_sha256"] != q.sha256(args.prior_extension)
            or frozen["prior_sampling_sha256"] != q.sha256(args.prior_sampling)
            or frozen["prior_reference_sha256"] != q.sha256(args.prior_reference)
            or extension["plan_sha256"] != q.sha256(args.plan)
            or sampling["plan_sha256"] != q.sha256(args.plan)
            or reference["plan_sha256"] != q.sha256(args.plan)
            or frozen["levels"] != ["tighter", "tightest"]
            or frozen["runs"] != ["tighter_default", "tighter_fine_rkf45",
                                  "tightest_fine_rkf45"]):
        raise ValueError("RKF45 convergence does not match frozen inputs")
    torch.set_num_threads(1)
    runs = {}
    summaries = {}
    for label, level_name, alternative in (
            ("tighter_default", "tighter", False),
            ("tighter_fine_rkf45", "tighter", True),
            ("tightest_fine_rkf45", "tightest", True)):
        level = extension["runs"][level_name]["level"]
        started = time.monotonic()
        run = q.native_run(base["row"], level, q.Settings(max_run_seconds=600.),
                           diagnostic_fine_rkf45_resampling=alternative,
                           return_accepted_hashes=True,
                           return_nqc_coefficients=True, return_boundary_data=True)
        summary = q.summarize_run(run)
        expected = (extension["runs"]["tighter"]["native"]["array_sha256"]
                    if label == "tighter_default" else
                    sampling["runs"]["fine_rkf45"]["native"]["array_sha256"]
                    if label == "tightest_fine_rkf45" else None)
        if expected is not None and summary["array_sha256"] != expected:
            raise ValueError(f"{label} waveform differs from frozen replay")
        runs[label] = run
        summaries[label] = dict(native=summary,
                                nqc_coefficients=run["nqc_coefficients"],
                                boundary_data=run["boundary_data"],
                                elapsed_seconds=time.monotonic()-started)
        print(f"{label} complete", flush=True)
    if (runs["tighter_default"]["accepted_array_sha256"]
            != runs["tighter_fine_rkf45"]["accepted_array_sha256"]
            or runs["tightest_fine_rkf45"]["accepted_array_sha256"]
            != sampling["runs"]["fine_rkf45"]["native"]["accepted_array_sha256"]):
        raise ValueError("accepted ODE arrays changed across sampling choices or replay")
    fine_comparison = q.compare_runs(runs["tighter_fine_rkf45"],
                                     runs["tightest_fine_rkf45"])
    baseline = extension["convergence"]["medium_to_fine"]
    budget = frozen["stability_budget"]
    checks = dict(
        complex_error_budget=(fine_comparison["complex_relative_l2"]
                              <= budget["complex_relative_l2_max"]),
        amplitude_error_budget=(fine_comparison["amplitude_relative_l2"]
                                <= budget["amplitude_relative_l2_max"]),
        **{event: (fine_comparison[event + "_abs_difference"]
                   <= budget["event_time_difference_M_max"]) for event in q.EVENTS})
    result = dict(schema="diffgw.seobnrv4.sxs-2123-rkf-convergence.v1",
                  plan_sha256=q.sha256(args.plan),
                  convergence_plan_sha256=q.sha256(args.convergence_plan),
                  prior_sha256=dict(extension=q.sha256(args.prior_extension),
                                    sampling=q.sha256(args.prior_sampling),
                                    reference=q.sha256(args.prior_reference)),
                  script_sha256=q.sha256(__file__),
                  generator_source_manifest_sha256=q.canonical_hash(q.source_snapshot()),
                  row=base["row"], runs=summaries, accepted_arrays_identical=True,
                  same_level_sampler_comparison=q.compare_runs(
                      runs["tighter_fine_rkf45"], runs["tighter_default"]),
                  rkf_tolerance_comparison=fine_comparison,
                  default_tolerance_comparison=baseline,
                  rkf_regional_diagnostic=regions(
                      runs["tighter_fine_rkf45"], runs["tightest_fine_rkf45"],
                      extension["runs"]["tightest"]["native"]["peak_time_M"]),
                  checks=checks, numerical_budget_passed=all(checks.values()),
                  release_qualified=False,
                  interpretation="Selected-row output-sampling and ODE tolerance "
                                 "diagnostic only; not a physical or domain gate.")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(rkf_tolerance_complex_relative_l2=(fine_comparison[
        "complex_relative_l2"]), default_tolerance_complex_relative_l2=(baseline[
            "complex_relative_l2"]), checks=checks)), flush=True)


if __name__ == "__main__":
    main()

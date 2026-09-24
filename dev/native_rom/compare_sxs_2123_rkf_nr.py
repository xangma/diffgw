"""Check the selected fine-stage sampler against frozen SXS:BBH:2123 NR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from . import qualify_generator as q
from .compare_sxs_2123 import anchor, compare, load_mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--nr-plan", type=Path, required=True)
    parser.add_argument("--prior-comparison", type=Path, required=True)
    parser.add_argument("--prior-convergence", type=Path, required=True)
    parser.add_argument("--prior-extension", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    frozen = json.loads(args.nr_plan.read_text())
    comparison = json.loads(args.prior_comparison.read_text())
    convergence = json.loads(args.prior_convergence.read_text())
    extension = json.loads(args.prior_extension.read_text())
    if (frozen["schema"] != "diffgw.seobnrv4.sxs-2123-rkf-nr-plan.v1"
            or frozen["prior_comparison_sha256"] != q.sha256(args.prior_comparison)
            or frozen["prior_rkf_convergence_sha256"] != q.sha256(args.prior_convergence)
            or frozen["prior_extension_sha256"] != q.sha256(args.prior_extension)
            or comparison["plan_sha256"] != q.sha256(args.plan)
            or convergence["plan_sha256"] != q.sha256(args.plan)
            or extension["plan_sha256"] != q.sha256(args.plan)
            or frozen["level_name"] != "tightest"):
        raise ValueError("NR sampler comparison does not match frozen inputs")
    extraction_path = args.decoded_dir / "extraction.json"
    extraction = json.loads(extraction_path.read_text())
    if (extraction["schema"] != "diffgw.seobnrv4.sxs-2123-extraction.v1"
            or extraction["plan_sha256"] != q.sha256(args.plan)
            or q.sha256(extraction_path) != comparison["extraction_sha256"]):
        raise ValueError("SXS extraction differs from frozen comparison")
    for filename, details in plan["catalog"]["waveforms"].items():
        if extraction["archive_files"][filename]["md5"] != details["md5"]:
            raise ValueError(f"SXS archive differs from frozen plan: {filename}")
    nr = {}
    for lev in (3, 4):
        item = extraction["extracted"][f"Lev{lev}"]
        path = args.decoded_dir / item["path"]
        if q.sha256(path) != item["sha256"]:
            raise ValueError(f"decoded NR mode differs from frozen extraction: Lev{lev}")
        nr[f"Lev{lev}"] = anchor(load_mode(path))
    torch.set_num_threads(1)
    runs = {}
    summaries = {}
    level = extension["runs"]["tightest"]["level"]
    for label, alternative in (("default", False), ("fine_rkf45", True)):
        started = time.monotonic()
        run = q.native_run(plan["row"], level, q.Settings(max_run_seconds=600.),
                           diagnostic_fine_rkf45_resampling=alternative,
                           return_accepted_hashes=True)
        summary = q.summarize_run(run)
        expected = (convergence["runs"]["tightest_fine_rkf45"]["native"]["array_sha256"]
                    if alternative else
                    extension["runs"]["tightest"]["native"]["array_sha256"])
        if summary["array_sha256"] != expected:
            raise ValueError(f"{label} waveform differs from frozen numerical solve")
        runs[label] = run
        summaries[label] = dict(native=summary, elapsed_seconds=time.monotonic()-started)
        print(f"{label} complete", flush=True)
    if runs["default"]["accepted_array_sha256"] != runs["fine_rkf45"]["accepted_array_sha256"]:
        raise ValueError("sampler changed accepted ODE arrays")
    aligned = {key: anchor(dict(time=run["time"], mode=run["mode"]))
               for key, run in runs.items()}
    windows = plan["method"]["windows_relative_to_Lev4_amplitude_peak_M"]
    step = plan["method"]["grid_step_M"]
    comparisons = {key + "_vs_Lev4": compare(mode, nr["Lev4"], windows, step)
                   for key, mode in aligned.items()}
    comparisons["Lev3_vs_Lev4_on_native_support"] = compare(
        nr["Lev3"], nr["Lev4"], windows, step,
        support=(float(aligned["default"]["time"][0]),
                 float(aligned["default"]["time"][-1])))
    comparisons["fine_rkf45_vs_default"] = compare(
        aligned["fine_rkf45"], aligned["default"], windows, step)
    result = dict(schema="diffgw.seobnrv4.sxs-2123-rkf-nr.v1",
                  plan_sha256=q.sha256(args.plan), nr_plan_sha256=q.sha256(args.nr_plan),
                  prior_comparison_sha256=q.sha256(args.prior_comparison),
                  prior_convergence_sha256=q.sha256(args.prior_convergence),
                  prior_extension_sha256=q.sha256(args.prior_extension),
                  extraction_sha256=q.sha256(extraction_path),
                  script_sha256=q.sha256(__file__),
                  generator_source_manifest_sha256=q.canonical_hash(q.source_snapshot()),
                  row=plan["row"], level=level, runs=summaries,
                  accepted_arrays_identical=True,
                  anchor={name: {key: value for key, value in mode.items()
                                 if key not in ("time", "mode")}
                          for name, mode in {**nr, **aligned}.items()},
                  comparisons=comparisons, physical_validation_passed=False,
                  release_qualified=False,
                  interpretation="One post-hoc NR row with predeclared fixed anchor; "
                                 "no physical-domain or ROM validation.")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value["metrics"]["complex_relative_l2"]
                      for key, value in comparisons.items()}), flush=True)


if __name__ == "__main__":
    main()

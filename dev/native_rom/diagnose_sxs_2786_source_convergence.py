"""Measure post-hoc SXS:BBH:2786 source-choice waveform convergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import compare_sxs_2123 as c
from . import qualify_generator as q


LEVELS = ("loose", "mid", "tightish")
EVENTS = ("peak_time_M", "match_time_M", "boundary_time_M")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    root = q.ROOT / "dev/native_rom/evidence/2026-09-24"
    case_path = root / "sxs_second_holdout_plan.json"
    comparison_path = root / "sxs_second_holdout_comparison.json"
    case_plan = json.loads(case_path.read_text())
    comparison = json.loads(comparison_path.read_text())
    case = comparison["cases"]["SXS:BBH:2786"]
    if comparison["plan_sha256"] != q.sha256(case_path):
        raise ValueError("holdout plan changed")
    results = {}
    waves = {}
    for label in LEVELS:
        plan_path = root / f"sxs_2786_source_choice_{label}_plan.json"
        result_path = root / f"sxs_2786_source_choice_{label}.json"
        result = json.loads(result_path.read_text())
        array_path = Path(result["array"]["path"])
        if (result["plan_sha256"] != q.sha256(plan_path)
                or result["comparison_sha256"] != q.sha256(comparison_path)
                or result["array"]["sha256"] != q.sha256(array_path)
                or result["source_manifest_sha256_before"]
                != result["source_manifest_sha256_after"]
                or result["release_qualified"] is not False):
            raise ValueError(f"{label} prerequisites changed")
        with np.load(array_path, allow_pickle=False) as record:
            wave = dict(time=record["source_choice_time"],
                        mode=record["source_choice_mode"])
        wave.update(result["anchor"])
        results[label] = result
        waves[label] = wave
    if len({item["source_manifest_sha256_before"] for item in results.values()}) != 1:
        raise ValueError("generator sources differ across tolerance levels")
    windows = case_plan["method"][
        "windows_relative_to_highest_resolution_amplitude_peak_M"]
    support = case["comparisons"]["native_vs_high"]["common_support_M"]
    dt = case_plan["method"]["grid_step_M"]
    pairs = {}
    for left, right in zip(LEVELS[:-1], LEVELS[1:]):
        fixed = c.compare(waves[left], waves[right], windows, dt,
                          support=support)
        event_differences = {name: results[right]["native"][name]
                             - results[left]["native"][name] for name in EVENTS}
        overlap_differences = {}
        for mass in (10, 50, 58, 100, 200):
            def overlap(label):
                row = next(item for item in results[label]["rows"]
                           if item["total_mass_msun"] == mass)
                return row["results"]["source_choice_vs_nr"]["overlap"]
            overlap_differences[str(mass)] = overlap(right)-overlap(left)
        pairs[f"{left}_vs_{right}"] = dict(fixed_crossing=fixed,
                                           event_differences_M=event_differences,
                                           weighted_overlap_differences=overlap_differences)
    output = dict(schema="diffgw.seobnrv4.sxs-2786-source-convergence.v1",
                  case_plan_sha256=q.sha256(case_path),
                  comparison_sha256=q.sha256(comparison_path),
                  result_sha256={label: q.sha256(
                      root / f"sxs_2786_source_choice_{label}.json")
                      for label in LEVELS},
                  script_sha256=q.sha256(__file__), pairs=pairs,
                  posthoc_selected_case=True, release_qualified=False)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
    print(json.dumps({key: value["fixed_crossing"]["metrics"]["complex_relative_l2"]
                      for key, value in pairs.items()}))


if __name__ == "__main__":
    main()

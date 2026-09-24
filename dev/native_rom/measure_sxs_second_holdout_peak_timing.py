"""Secondary phase-aligned peak-timing proxy for the second SXS holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import measure_sxs_five_case_peak_timing as timing
from . import qualify_generator as q


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--timing-plan", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    case_plan = json.loads(args.case_plan.read_text())
    plan = json.loads(args.timing_plan.read_text())
    comparison = json.loads(args.comparison.read_text())
    extraction_path = args.decoded_dir / "extraction.json"
    if (case_plan["schema"] != "diffgw.seobnrv4.sxs-second-holdout-plan.v1"
            or plan["schema"] != "diffgw.seobnrv4.sxs-second-holdout-peak-timing-plan.v1"
            or plan["qualification_claim"] is not False
            or plan["case_plan_sha256"] != q.sha256(args.case_plan)
            or plan["extraction_sha256"] != q.sha256(extraction_path)
            or plan["phase_timing_function_source_sha256"] != q.sha256(timing.__file__)
            or comparison["schema"] != "diffgw.seobnrv4.sxs-second-holdout-comparison.v1"
            or comparison["plan_sha256"] != q.sha256(args.case_plan)
            or comparison["extraction_sha256"] != q.sha256(extraction_path)
            or not comparison["all_cases_succeeded"]
            or not comparison["sources_unchanged"]
            or set(comparison["cases"]) != set(case_plan["cases"])
            or set(plan["nr_high_metadata_sha256"]) != set(case_plan["cases"])
            or plan["pairs"] != ["native_vs_nr", "low_vs_high_nr"]
            or plan["sample_step_M"] != case_plan["method"]["grid_step_M"]):
        raise ValueError("second holdout peak-timing prerequisites changed")
    output = dict(schema="diffgw.seobnrv4.sxs-second-holdout-peak-timing.v1",
                  plan_sha256=q.sha256(args.timing_plan),
                  comparison_sha256=q.sha256(args.comparison),
                  script_sha256=q.sha256(__file__), release_qualified=False,
                  paper_exact_alignment=False, cases={})
    for simulation, case in case_plan["cases"].items():
        entry = comparison["cases"][simulation]
        high = max(case["levels"])
        metadata_path = (args.archive_dir / simulation.replace(":", "_")
                         / f"Lev{high}" / "metadata.json")
        array_path = Path(entry["anchored_array"]["path"])
        if (q.sha256(array_path) != entry["anchored_array"]["sha256"]
                or q.sha256(metadata_path)
                != plan["nr_high_metadata_sha256"][simulation]):
            raise ValueError(f"peak-timing input changed: {simulation}")
        metadata = json.loads(metadata_path.read_text())
        if metadata["reference_time"] != metadata["relaxation_time"]:
            raise ValueError(f"NR settled reference changed: {simulation}")
        with np.load(array_path, allow_pickle=False) as source:
            waves = {name: dict(time=source[f"{name}_time"],
                                mode=source[f"{name}_mode"],
                                peak_after_anchor_M=entry["anchor"][label]
                                ["peak_after_anchor_M"])
                     for name, label in (("native", "native"),
                                         ("nr_low", f"Lev{min(case['levels'])}"),
                                         ("nr_high", f"Lev{high}"))}
        relaxed_start = (metadata["relaxation_time"]
                         - entry["anchor"][f"Lev{high}"]["anchor_time_M"])
        start = relaxed_start + plan["guard_after_relaxation_M"]

        def fit(duration: float) -> dict:
            return {name: timing.phase_aligned_peak_lag(
                waves[candidate], waves["nr_high"], start=start,
                duration=duration, dt=plan["sample_step_M"],
                bounds=tuple(plan["allowed_time_shift_M"]),
                step=plan["time_shift_grid_step_M"])
                for name, candidate in (("native_vs_nr", "native"),
                                        ("low_vs_high_nr", "nr_low"))}

        output["cases"][simulation] = dict(
            array_sha256=q.sha256(array_path),
            nr_high_metadata_sha256=q.sha256(metadata_path),
            nr_relaxed_start_after_anchor_M=relaxed_start,
            primary=fit(plan["primary_window_duration_M"]),
            controls={str(duration): fit(duration)
                      for duration in plan["control_window_durations_M"]})
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2, sort_keys=True,
                                            allow_nan=False) + "\n")
        print(f"{simulation} peak timing complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

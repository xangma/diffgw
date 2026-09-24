"""Low-frequency phase-aligned peak timing proxy for five SXS cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from . import compare_sxs_2123 as c
from . import qualify_generator as q


def phase_aligned_peak_lag(candidate: dict, reference: dict, *, start: float,
                           duration: float, dt: float, bounds: tuple,
                           step: float) -> dict:
    grid = np.arange(math.ceil(start / dt),
                     math.floor((start + duration) / dt) + 1) * dt
    if len(grid) < 100 or grid[0] < reference["time"][0] or grid[-1] > reference["time"][-1]:
        raise ValueError("insufficient settled reference support")
    lower = max(bounds[0], float(candidate["time"][0] - grid[0]))
    upper = min(bounds[1], float(candidate["time"][-1] - grid[-1]))
    if upper - lower < 2 * step:
        raise ValueError("insufficient candidate support for low-frequency fit")
    candidate_phase = np.unwrap(np.angle(candidate["mode"]))
    reference_phase = np.interp(grid, reference["time"],
                                np.unwrap(np.angle(reference["mode"])))
    def score(lag):
        residual = np.interp(grid + lag, candidate["time"], candidate_phase) - reference_phase
        centered = residual - np.mean(residual)
        return float(np.sqrt(np.mean(centered**2)))
    lags = np.arange(math.ceil(lower / step), math.floor(upper / step) + 1) * step
    scores = np.array([score(value) for value in lags])
    best = int(np.argmin(scores))
    lo = max(lower, float(lags[best] - step))
    hi = min(upper, float(lags[best] + step))
    lag = float(minimize_scalar(score, bounds=(lo, hi), method="bounded",
                                options={"xatol": 1e-7}).x) if lo < hi else lo
    peak_lag = float(candidate["peak_after_anchor_M"] - lag
                     - reference["peak_after_anchor_M"])
    return dict(peak_lag_after_alignment_M=peak_lag,
                absolute_peak_lag_M=abs(peak_lag), fitted_time_shift_M=lag,
                phase_residual_rms_rad=score(lag),
                time_shift_at_boundary=bool(best in (0, len(lags) - 1)),
                fit_window_M=[float(grid[0]), float(grid[-1])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-plan", type=Path, required=True)
    parser.add_argument("--four-long-plan", type=Path, required=True)
    parser.add_argument("--q5-long-plan", type=Path, required=True)
    parser.add_argument("--timing-plan", type=Path, required=True)
    parser.add_argument("--long-dir", type=Path, required=True)
    parser.add_argument("--decoded-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.timing_plan.read_text())
    case_plan = json.loads(args.case_plan.read_text())
    extraction_path = args.decoded_dir / "extraction.json"
    if (plan["schema"] != "diffgw.seobnrv4.sxs-five-case-peak-timing-plan.v1"
            or plan["qualification_claim"] is not False
            or plan["case_plan_sha256"] != q.sha256(args.case_plan)
            or plan["extraction_sha256"] != q.sha256(extraction_path)
            or plan["four_long_start_plan_sha256"] != q.sha256(args.four_long_plan)
            or plan["sxs_2508_long_start_plan_sha256"] != q.sha256(args.q5_long_plan)
            or set(plan["nr_high_metadata_sha256"]) != set(case_plan["cases"])):
        raise ValueError("peak-timing prerequisites changed")
    extraction = json.loads(extraction_path.read_text())
    output = dict(schema="diffgw.seobnrv4.sxs-five-case-peak-timing.v1",
                  timing_plan_sha256=q.sha256(args.timing_plan),
                  script_sha256=q.sha256(__file__), release_qualified=False,
                  paper_exact_alignment=False, cases={})
    for simulation, case in case_plan["cases"].items():
        number = simulation.rsplit(":", 1)[-1]
        long_path = args.long_dir / f"sxs_{number}_long_start.json"
        long_result = json.loads(long_path.read_text())
        array_path = args.decoded_dir / f"sxs_{number}_long_start_arrays.npz"
        long_plan_hash = (plan["sxs_2508_long_start_plan_sha256"] if number == "2508"
                          else plan["four_long_start_plan_sha256"])
        high, low = max(case["levels"]), min(case["levels"])
        metadata_path = (args.archive_dir / simulation.replace(":", "_")
                         / f"Lev{high}" / "metadata.json")
        metadata = json.loads(metadata_path.read_text())
        if (long_result["long_plan_sha256"] != long_plan_hash
                or long_result["array_sha256"] != q.sha256(array_path)
                or not long_result["sources_unchanged"]
                or (number == "2508" and q.sha256(long_path)
                    != plan["sxs_2508_long_start_result_sha256"])
                or (number != "2508" and long_result["simulation"] != simulation)
                or q.sha256(metadata_path)
                    != plan["nr_high_metadata_sha256"][simulation]
                or metadata["reference_time"] != metadata["relaxation_time"]):
            raise ValueError(f"peak-timing provenance changed: {simulation}")
        low_record = extraction["cases"][simulation]["extracted"][f"Lev{low}"]
        low_path = args.decoded_dir / low_record["path"]
        if q.sha256(low_path) != low_record["sha256"]:
            raise ValueError(f"lower-resolution NR mode changed: {simulation}")
        with np.load(array_path, allow_pickle=False) as source:
            waves = {name: dict(time=source[f"{name}_time"],
                                mode=source[f"{name}_mode"],
                                peak_after_anchor_M=
                                long_result["anchors"][name]["peak_after_anchor_M"])
                     for name in ("native", "public", "nr")}
        waves["nr_low"] = c.anchor(c.load_mode(low_path), dt=plan["sample_step_M"])
        relaxed_start = (metadata["relaxation_time"]
                         - long_result["anchors"]["nr"]["anchor_time_M"])
        start = relaxed_start + plan["guard_after_relaxation_M"]
        def fit(duration):
            return {name: phase_aligned_peak_lag(
                waves[candidate], waves["nr"], start=start, duration=duration,
                dt=plan["sample_step_M"], bounds=tuple(plan["allowed_time_shift_M"]),
                step=plan["time_shift_grid_step_M"])
                for name, candidate in (("native_vs_nr", "native"),
                                        ("public_vs_nr", "public"),
                                        ("low_vs_high_nr", "nr_low"))}
        output["cases"][simulation] = dict(
            long_result_sha256=q.sha256(long_path), array_sha256=q.sha256(array_path),
            nr_high_metadata_sha256=q.sha256(metadata_path),
            nr_low_sha256=q.sha256(low_path),
            nr_relaxed_start_after_anchor_M=relaxed_start,
            primary=fit(plan["primary_window_duration_M"]),
            controls={str(duration): fit(duration)
                      for duration in plan["control_window_durations_M"]})
        print(f"{simulation} peak timing complete", flush=True)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2, allow_nan=False,
                                             sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

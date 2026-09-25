"""Dev-only attribution of source-backed waveform tolerance drift by stage."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_spliced_whole_waveform import (
    SOURCE_PLAN,
    _prepare_public,
    run,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "jax_source_tolerance_stage_plan.json"
LADDER_PLAN = EVIDENCE / "jax_source_tolerance_ladder_plan.json"


def _compare(left_time, left, right_time, right, chart_limit):
    count = min(len(left_time), len(right_time), len(left), len(right))
    chart_error = float(np.max(np.abs(left_time[:count] - right_time[:count])))
    if chart_error > chart_limit:
        raise ValueError(f"stage charts differ: {chart_error:g} M")
    differences = np.asarray(left[:count]) - np.asarray(right[:count])
    reference = np.asarray(right[:count])
    early = int(0.8 * count)
    return {
        "common_samples": count,
        "sample_count_difference": abs(len(left) - len(right)),
        "chart_max_abs_M": chart_error,
        "complex_relative_l2": float(np.linalg.norm(differences)
                                     / np.linalg.norm(reference)),
        "early_80_percent_relative_l2": float(np.linalg.norm(differences[:early])
                                               / np.linalg.norm(reference[:early])),
        "max_abs": float(np.max(np.abs(differences))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pilot", "prospective"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    ladder_plan = json.loads(LADDER_PLAN.read_text())
    case = plan["pilot_case"] if args.case == "pilot" else plan["prospective_case"]
    ladder_path = EVIDENCE / f"jax_source_tolerance_ladder_{case}.json"
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-tolerance-stage-plan.v1"
            or plan["release_qualified"] is not False
            or q.sha256(LADDER_PLAN) != plan["ladder_plan_sha256"]
            or q.sha256(ladder_path) != plan["ladder_result_sha256"][case]
            or plan["levels"] != ["baseline", "refined"]):
        raise ValueError("frozen stage-attribution inputs changed")
    ladder = json.loads(ladder_path.read_text())
    source_plan = json.loads(SOURCE_PLAN.read_text())
    identity = _prepare_public(source_plan)
    started = time.monotonic()
    output = EVIDENCE / f"jax_source_tolerance_stages_{case}.json"
    result = {
        "schema": "diffgw.seobnrv4.jax-source-tolerance-stages.v1",
        "status": "running", "case_id": case, "plan_sha256": q.sha256(PLAN),
        "release_qualified": False, "host": platform.node(),
        "cwd": os.getcwd(), "pid": os.getpid(), "levels": {},
        "stage_comparisons": {},
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    series = {}
    stages = {}
    for name in plan["levels"]:
        level = next(row for row in ladder_plan["levels"] if row["name"] == name)
        settings = {
            "coarse_output_step_M": ladder_plan["coarse_output_step_M"],
            "max_step_M": ladder_plan["max_step_M"],
            "atol": level["atol"], "rtol": level["rtol"],
        }
        diagnostic, torch_series, _, stage = run(
            case, source_plan, identity, variant="source", independent_init=True,
            numerical_settings=settings, write_result=False, return_series=True,
            return_stages=True)
        series[name], stages[name] = torch_series, stage
        result["levels"][name] = {
            "settings": settings,
            "whole_samples": len(torch_series[0]),
            "coarse_samples": len(stage["coarse_time"]),
            "fine_samples": len(stage["fine_time"]),
            "restart_index": stage["restart_index"],
            "orbital_peak_time_M": stage["orbital_peak_time_M"],
            "match_time_M": stage["match_time_M"],
            "boundary_time_M": stage["boundary_time_M"],
            "boundary_index": stage["boundary_index"],
            "nqc_coefficients": [x.tolist() for x in stage["nqc_coefficients"]],
            "whole_mode_relative_l2_torch_jax": diagnostic["whole_mode_relative_l2"],
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(case, name, result["levels"][name], flush=True)
    base, refined = stages["baseline"], stages["refined"]
    limit = plan["same_chart_tolerance_M"]
    for label, time_key, value_key in (
        ("coarse_state", "coarse_time", "coarse_state"),
        ("fine_state", "fine_time", "fine_state"),
        ("fine_base_mode", "fine_time", "fine_base_mode"),
        ("fine_corrected_inspiral", "fine_time", "fine_corrected_inspiral"),
        ("fine_orbital_frequency", "fine_time", "fine_orbital_frequency"),
    ):
        result["stage_comparisons"][label] = _compare(
            base[time_key], base[value_key], refined[time_key],
            refined[value_key], limit)
    result["stage_comparisons"]["whole_mode"] = _compare(
        series["baseline"][0], series["baseline"][1],
        series["refined"][0], series["refined"][1], limit)
    bcoef = np.concatenate(base["nqc_coefficients"])
    rcoef = np.concatenate(refined["nqc_coefficients"])
    result["nqc_coefficient_l2_difference"] = float(np.linalg.norm(bcoef-rcoef))
    result["nqc_coefficient_relative_l2"] = float(np.linalg.norm(bcoef-rcoef)
                                                        / np.linalg.norm(rcoef))
    result["landmark_differences_M"] = {
        key: float(base[key] - refined[key]) for key in
        ("orbital_peak_time_M", "match_time_M", "boundary_time_M")
    }
    expected = ladder["comparisons"]["torch_baseline_to_refined"]["complex_relative_l2"]
    observed = result["stage_comparisons"]["whole_mode"]["complex_relative_l2"]
    if abs(expected - observed) > plan["ladder_replay_relative_l2_tolerance"]:
        raise ValueError("whole-mode drift does not reproduce the frozen ladder")
    result["status"] = "diagnostic_complete"
    result["elapsed_s"] = time.monotonic() - started
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(case, json.dumps({"comparisons": result["stage_comparisons"],
                            "nqc_coefficient_l2_difference": result["nqc_coefficient_l2_difference"],
                            "landmark_differences_M": result["landmark_differences_M"]},
                           sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

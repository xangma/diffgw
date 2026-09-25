"""Dev-only source-backed Torch/JAX whole-waveform tolerance ladder."""

import argparse
import hashlib
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
PLAN = EVIDENCE / "jax_source_tolerance_ladder_plan.json"


def _series_summary(series):
    time_grid, mode = series
    return {
        "samples": len(time_grid),
        "start_M": float(time_grid[0]),
        "end_M": float(time_grid[-1]),
        "time_sha256": hashlib.sha256(np.asarray(time_grid).tobytes()).hexdigest(),
        "mode_sha256": hashlib.sha256(np.asarray(mode).tobytes()).hexdigest(),
    }


def _compare(left, right):
    lt, lm = left
    rt, rm = right
    count = min(len(lt), len(rt))
    chart_error = float(np.max(np.abs(lt[:count] - rt[:count])))
    if chart_error > 1e-8:
        raise ValueError("whole-waveform time charts differ on shared support")
    if abs(len(lt) - len(rt)) > 1:
        raise ValueError("whole-waveform support differs by more than one output sample")
    return {
        "common_samples": count,
        "support_sample_difference": abs(len(lt) - len(rt)),
        "time_max_abs_M": chart_error,
        "complex_relative_l2": float(np.linalg.norm(lm[:count] - rm[:count])
                                     / np.linalg.norm(rm[:count])),
    }


def _validate(plan, cases):
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-tolerance-ladder-plan.v1"
            or plan["release_qualified"] is not False
            or q.sha256(SOURCE_PLAN) != plan["source_plan_sha256"]
            or [level["name"] for level in plan["levels"]]
            != ["baseline", "refined", "finest"]):
        raise ValueError("frozen tolerance ladder changed")
    for case in cases:
        path = EVIDENCE / f"jax_source_independent_init_{case}.json"
        if q.sha256(path) != plan["baseline_result_sha256"][case]:
            raise ValueError(f"frozen independent-initial-state control changed: {case}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pilot", "prospective"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    cases = ((plan["pilot_case"],) if args.case == "pilot"
             else tuple(plan["prospective_cases"]))
    _validate(plan, cases)
    source_plan = json.loads(SOURCE_PLAN.read_text())
    identity = _prepare_public(source_plan)
    for case in cases:
        started = time.monotonic()
        output = EVIDENCE / f"jax_source_tolerance_ladder_{case}.json"
        result = {
            "schema": "diffgw.seobnrv4.jax-source-tolerance-ladder.v1",
            "status": "running", "case_id": case,
            "plan_sha256": q.sha256(PLAN), "release_qualified": False,
            "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
            "levels": {}, "comparisons": {}, "checks": {},
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        series = {}
        for level in plan["levels"]:
            name = level["name"]
            settings = {
                "coarse_output_step_M": plan["coarse_output_step_M"],
                "max_step_M": plan["max_step_M"],
                "atol": level["atol"], "rtol": level["rtol"],
            }
            diagnostic, torch_series, jax_series = run(
                case, source_plan, identity, variant="source",
                independent_init=True, numerical_settings=settings,
                write_result=False, return_series=True)
            if name == "baseline":
                prior = json.loads((EVIDENCE / f"jax_source_independent_init_{case}.json").read_text())
                if (diagnostic["whole_output_samples"] != prior["whole_output_samples"]
                        or abs(diagnostic["whole_mode_relative_l2"]
                               - prior["whole_mode_relative_l2"]) > 1e-12):
                    raise ValueError(f"baseline replay changed: {case}")
            series[name] = {"torch": torch_series, "jax": jax_series}
            result["levels"][name] = {
                "settings": settings,
                "torch": _series_summary(torch_series),
                "jax": _series_summary(jax_series),
                "torch_jax": _compare(torch_series, jax_series),
                "diagnostic_checks_passed": diagnostic["diagnostic_checks_passed"],
            }
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(case, name, result["levels"][name]["torch_jax"], flush=True)
        for backend in ("torch", "jax"):
            for left, right in (("baseline", "refined"), ("refined", "finest")):
                key = f"{backend}_{left}_to_{right}"
                result["comparisons"][key] = _compare(
                    series[left][backend], series[right][backend])
        limits = plan["checks"]
        result["checks"] = {
            "baseline_to_refined_complex_relative_l2_at_most": all(
                result["comparisons"][f"{backend}_baseline_to_refined"]["complex_relative_l2"]
                <= limits["baseline_to_refined_complex_relative_l2_at_most"]
                for backend in ("torch", "jax")),
            "refined_to_finest_complex_relative_l2_at_most": all(
                result["comparisons"][f"{backend}_refined_to_finest"]["complex_relative_l2"]
                <= limits["refined_to_finest_complex_relative_l2_at_most"]
                for backend in ("torch", "jax")),
            "torch_jax_finest_complex_relative_l2_at_most": (
                result["levels"]["finest"]["torch_jax"]["complex_relative_l2"]
                <= limits["torch_jax_finest_complex_relative_l2_at_most"]),
            "each_level_backend_checks_passed": all(
                value["diagnostic_checks_passed"] for value in result["levels"].values()),
        }
        result["diagnostic_checks_passed"] = all(result["checks"].values())
        result["elapsed_s"] = time.monotonic() - started
        result["status"] = "diagnostic_complete"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(case, json.dumps({"checks": result["checks"],
                                "comparisons": result["comparisons"]},
                               sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

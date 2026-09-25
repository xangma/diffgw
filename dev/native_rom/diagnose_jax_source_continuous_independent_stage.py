"""Dev-only attribution of continuous-boundary independent-grid drift."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_source_cadence import _shared_compare
from dev.native_rom.diagnose_jax_source_stationary_stage_audit import (
    _matrix_attribution,
    _shared,
    _system,
)
from dev.native_rom.diagnose_jax_spliced_whole_waveform import (
    SOURCE_PLAN,
    _prepare_public,
    run,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "jax_source_continuous_independent_stage_plan.json"


def _partition(series_a, series_b, start, end):
    time_a, mode_a = series_a
    time_b, mode_b = series_b
    count = min(len(time_a), len(time_b))
    if np.max(np.abs(time_a[:count] - time_b[:count])) > 1e-8:
        raise ValueError("whole output chart changed")
    selected = (time_a[:count] >= start) & (time_a[:count] < end)
    a, b = mode_a[:count][selected], mode_b[:count][selected]
    if len(a) == 0:
        raise ValueError("empty waveform partition")
    return {
        "samples": int(len(a)),
        "start_M": float(time_a[:count][selected][0]),
        "end_M": float(time_a[:count][selected][-1]),
        "complex_relative_l2": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
        "complex_max_abs": float(np.max(np.abs(a - b))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pilot", "prospective"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    case = plan[f"{args.case}_case"]
    prior_path = EVIDENCE / f"jax_source_continuous_independent_grid_{case}.json"
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-continuous-independent-stage-plan.v1"
            or plan["release_qualified"] is not False
            or plan["fine_refinement_factors"] != [256, 512]
            or q.sha256(SOURCE_PLAN) != plan["source_plan_sha256"]
            or q.sha256(prior_path) != plan["prior_independent_grid_sha256"][case]):
        raise ValueError("frozen independent-stage plan or prerequisites changed")
    prior = json.loads(prior_path.read_text())
    source_plan = json.loads(SOURCE_PLAN.read_text())
    identity = _prepare_public(source_plan)
    output = EVIDENCE / f"jax_source_continuous_independent_stage_{case}.json"
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.jax-source-continuous-independent-stage.v1",
        "status": "running", "case_id": case, "plan_sha256": q.sha256(PLAN),
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "levels": {}, "comparisons": {}, "checks": {}, "release_qualified": False,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stages, series = {}, {}
    try:
        for label, factor in zip(("baseline", "refined"), plan["fine_refinement_factors"]):
            settings = {
                "coarse_output_step_M": plan["output_step_M"],
                "fine_refinement_factor": factor,
                "max_step_M": plan["max_step_M"],
                "atol": plan["atol"], "rtol": plan["rtol"],
            }
            diagnostic, torch_series, _, stage = run(
                case, source_plan, identity, variant="source", independent_init=True,
                numerical_settings=settings, diagnostic_peak_policy="stationary",
                diagnostic_continuous_boundary=True,
                write_result=False, return_series=True, return_stages=True)
            stages[label], series[label] = stage, torch_series
            result["levels"][label] = {
                "settings": settings,
                "fine_input_samples": len(stage["fine_time"]),
                "match_time_M": stage["match_time_M"],
                "boundary_time_M": stage["boundary_time_M"],
                "orbital_peak_spline_slope": stage["orbital_peak_spline_slope"],
                "amplitude_system": _system(stage, "amplitude"),
                "phase_system": _system(stage, "phase"),
                "whole_mode_complex_relative_l2_vs_jax": diagnostic[
                    "whole_mode_relative_l2"],
            }
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(case, label, "stage captured", flush=True)
        a, b = stages["baseline"], stages["refined"]
        for key in ("fine_state", "fine_orbital_frequency", "fine_base_mode",
                    "fine_corrected_inspiral"):
            result["comparisons"][key] = _shared(a, b, key)
        for prefix in ("amplitude", "phase"):
            result["comparisons"][f"{prefix}_system"] = _matrix_attribution(
                result["levels"]["baseline"][f"{prefix}_system"],
                result["levels"]["refined"][f"{prefix}_system"])
        result["comparisons"]["match_time_difference_M"] = float(
            b["match_time_M"] - a["match_time_M"])
        boundary = min(a["boundary_time_M"], b["boundary_time_M"])
        end = max(a["boundary_time_M"], b["boundary_time_M"])
        restart_time = float(a["coarse_time"][a["restart_index"]])
        result["comparisons"]["waveform_partitions"] = {
            "coarse_prefix": _partition(series["baseline"], series["refined"],
                                        -np.inf, restart_time),
            "post_restart_pre_boundary": _partition(
                series["baseline"], series["refined"], restart_time, boundary),
            "post_boundary": _partition(series["baseline"], series["refined"],
                                        end, np.inf),
        }
        result["comparisons"]["whole_waveform"] = _shared_compare(
            series["baseline"], series["refined"])
        limits = plan["checks"]
        result["checks"] = {
            "shared_fine_time_max_abs_at_most_M": all(
                result["comparisons"][key]["time_max_abs_M"]
                <= limits["shared_fine_time_max_abs_at_most_M"]
                for key in ("fine_state", "fine_orbital_frequency", "fine_base_mode",
                            "fine_corrected_inspiral")),
            "nqc_linear_system_max_abs_residual_at_most": all(
                result["levels"][label][f"{prefix}_system"]["max_abs_residual"]
                <= limits["nqc_linear_system_max_abs_residual_at_most"]
                for label in ("baseline", "refined")
                for prefix in ("amplitude", "phase")),
            "whole_result_reproduced_abs_at_most": abs(
                result["comparisons"]["whole_waveform"]["complex_relative_l2"]
                - prior["comparisons"]["torch"]["complex_relative_l2"])
                <= limits["whole_result_reproduced_abs_at_most"],
        }
        result["diagnostic_checks_passed"] = all(result["checks"].values())
        result["status"] = "diagnostic_complete"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["elapsed_s"] = time.monotonic() - started
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(case, json.dumps({"status": result["status"], "checks": result["checks"],
                            "comparisons": result["comparisons"],
                            "error": result.get("error")}, sort_keys=True), flush=True)
    return 0 if result["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

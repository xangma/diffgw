"""Dev-only stationary orbital-peak counterfactual on fixed fine-input grids."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_source_cadence import _shared_compare
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _compare, _series_summary
from dev.native_rom.diagnose_jax_spliced_whole_waveform import (
    SOURCE_PLAN,
    _prepare_public,
    run,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "jax_source_stationary_peak_control_plan.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pilot", "prospective"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    case = plan[f"{args.case}_case"]
    prior_path = EVIDENCE / f"jax_source_peak_spacing_{case}.json"
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-stationary-peak-control-plan.v1"
            or plan["release_qualified"] is not False
            or plan["output_step_M"] != 0.5
            or plan["fine_refinement_factors"] != [128, 256]
            or q.sha256(SOURCE_PLAN) != plan["source_plan_sha256"]
            or q.sha256(prior_path) != plan["compatibility_result_sha256"][case]):
        raise ValueError("frozen stationary-peak plan or prerequisites changed")
    prior = json.loads(prior_path.read_text())
    identity = _prepare_public(json.loads(SOURCE_PLAN.read_text()))
    started = time.monotonic()
    output = EVIDENCE / f"jax_source_stationary_peak_control_{case}.json"
    result = {
        "schema": "diffgw.seobnrv4.jax-source-stationary-peak-control.v1",
        "status": "running", "case_id": case,
        "plan_sha256": q.sha256(PLAN), "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "peak_policy": "nearest negative-curvature derivative root to sampled maximum",
        "compatibility_shared_sample_complex_relative_l2": {
            backend: prior["comparisons"][backend]["complex_relative_l2"]
            for backend in ("torch", "jax")},
        "levels": {}, "comparisons": {}, "checks": {},
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    series = {}
    try:
        for label, factor in zip(("baseline", "refined"), plan["fine_refinement_factors"]):
            settings = {
                "coarse_output_step_M": plan["output_step_M"],
                "fine_refinement_factor": factor,
                "max_step_M": plan["max_step_M"],
                "atol": plan["atol"], "rtol": plan["rtol"],
            }
            diagnostic, torch_series, jax_series, stage = run(
                case, json.loads(SOURCE_PLAN.read_text()), identity,
                variant="source", independent_init=True,
                numerical_settings=settings, diagnostic_peak_policy="stationary",
                write_result=False, return_series=True, return_stages=True)
            for backend, (time_grid, _) in (("torch", torch_series), ("jax", jax_series)):
                if np.max(np.abs(np.diff(time_grid) - plan["output_step_M"])) > 1e-8:
                    raise ValueError(f"whole output chart changed: {case} {backend} {factor}")
            series[label] = {"torch": torch_series, "jax": jax_series}
            level = {
                "settings": settings,
                "torch": _series_summary(torch_series),
                "jax": _series_summary(jax_series),
                "torch_jax": _compare(torch_series, jax_series),
                "diagnostic_checks_passed": diagnostic["diagnostic_checks_passed"],
                "nqc_coefficients": [x.tolist() for x in stage["nqc_coefficients"]],
                "jax_nqc_coefficients": [x.tolist() for x in stage["jax_nqc_coefficients"]],
                "fine_input_samples": len(stage["fine_time"]),
                "jax_fine_input_samples": len(stage["jax_fine_time"]),
                "match_time_M": stage["match_time_M"],
                "jax_match_time_M": stage["jax_match_time_M"],
                "boundary_time_M": stage["boundary_time_M"],
                "jax_boundary_time_M": stage["jax_boundary_time_M"],
                "torch_peak": {key: stage[key] for key in (
                    "orbital_peak_time_M", "orbital_peak_sample_index",
                    "orbital_peak_spline_slope", "orbital_peak_spline_curvature")},
                "jax_peak": {key: stage[key] for key in (
                    "jax_orbital_peak_time_M", "jax_orbital_peak_sample_index",
                    "jax_orbital_peak_spline_slope", "jax_orbital_peak_spline_curvature")},
                "compatibility_match_time_M": {
                    backend: prior["levels"][label][
                        "match_time_M" if backend == "torch" else "jax_match_time_M"]
                    for backend in ("torch", "jax")},
            }
            for backend in ("torch", "jax"):
                for key in ("time_sha256",):
                    if level[backend][key] != prior["levels"][label][backend][key]:
                        raise ValueError(f"compatibility output chart changed: {case} {backend} {label}")
            result["levels"][label] = level
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(case, label, level["torch_jax"], flush=True)
        for backend in ("torch", "jax"):
            result["comparisons"][backend] = _shared_compare(
                series["baseline"][backend], series["refined"][backend])
            key = "nqc_coefficients" if backend == "torch" else "jax_nqc_coefficients"
            baseline = np.concatenate(result["levels"]["baseline"][key])
            refined = np.concatenate(result["levels"]["refined"][key])
            result["comparisons"][f"{backend}_nqc_coefficient_l2"] = float(
                np.linalg.norm(refined - baseline))
        limits = plan["checks"]
        result["checks"] = {
            "shared_sample_complex_relative_l2_at_most": all(
                result["comparisons"][backend]["complex_relative_l2"]
                <= limits["shared_sample_complex_relative_l2_at_most"]
                for backend in ("torch", "jax")),
            "each_level_backend_checks_passed": all(
                level["diagnostic_checks_passed"] for level in result["levels"].values()),
            "stationary_peak_slope_max_abs_at_most": all(
                abs(level["torch_peak"]["orbital_peak_spline_slope"])
                <= limits["stationary_peak_slope_max_abs_at_most"]
                and abs(level["jax_peak"]["jax_orbital_peak_spline_slope"])
                <= limits["stationary_peak_slope_max_abs_at_most"]
                for level in result["levels"].values()),
            "shared_time_max_abs_at_most_M": all(
                result["comparisons"][backend]["shared_time_max_abs_M"]
                <= limits["shared_time_max_abs_at_most_M"]
                for backend in ("torch", "jax")),
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

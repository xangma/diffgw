"""Dev-only combined local-NQC and continuous-boundary grid control."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_source_cadence import _shared_compare
from dev.native_rom.diagnose_jax_source_local_system_control import _system
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _compare, _series_summary
from dev.native_rom.diagnose_jax_spliced_whole_waveform import (
    SOURCE_PLAN,
    _prepare_public,
    run,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "jax_source_combined_numerical_control_plan.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pilot", "prospective"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    case = plan[f"{args.case}_case"]
    stage_path = EVIDENCE / f"jax_source_continuous_independent_stage_{case}.json"
    local_path = EVIDENCE / f"jax_source_local_system_control_{case}.json"
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-combined-numerical-control-plan.v1"
            or plan["release_qualified"] is not False
            or plan["fine_refinement_factors"] != [256, 512]
            or plan["polynomial"] != {"half_width_M": 2.0, "degree": 4}
            or q.sha256(SOURCE_PLAN) != plan["source_plan_sha256"]
            or q.sha256(stage_path) != plan["prior_continuous_stage_sha256"][case]
            or q.sha256(local_path) != plan["prior_local_system_sha256"][case]):
        raise ValueError("frozen combined-control plan or prerequisites changed")
    prior = json.loads(stage_path.read_text())
    source_plan = json.loads(SOURCE_PLAN.read_text())
    identity = _prepare_public(source_plan)
    output = EVIDENCE / f"jax_source_combined_numerical_control_{case}.json"
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.jax-source-combined-numerical-control.v1",
        "status": "running", "case_id": case, "plan_sha256": q.sha256(PLAN),
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "prior_continuous_only_complex_relative_l2": prior[
            "comparisons"]["whole_waveform"]["complex_relative_l2"],
        "levels": {}, "comparisons": {}, "checks": {}, "release_qualified": False,
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
                case, source_plan, identity, variant="source", independent_init=True,
                numerical_settings=settings, diagnostic_peak_policy="stationary",
                diagnostic_nqc_amplitude_derivatives=plan["polynomial"],
                diagnostic_nqc_phase_matrix=plan["polynomial"],
                diagnostic_nqc_rhs_derivatives=plan["polynomial"],
                diagnostic_continuous_boundary=True,
                write_result=False, return_series=True, return_stages=True)
            if abs(stage["match_time_M"] - prior["levels"][label]["match_time_M"]) > 1e-10:
                raise ValueError(f"stationary match changed: {case} {label}")
            for backend, (time_grid, _) in (("torch", torch_series), ("jax", jax_series)):
                if np.max(np.abs(np.diff(time_grid) - plan["output_step_M"])) > 1e-8:
                    raise ValueError(f"whole output chart changed: {case} {backend} {label}")
            series[label] = {"torch": torch_series, "jax": jax_series}
            result["levels"][label] = {
                "settings": settings,
                "torch": _series_summary(torch_series),
                "jax": _series_summary(jax_series),
                "torch_jax": _compare(torch_series, jax_series),
                "expected_backend_parity_broken": not diagnostic["diagnostic_checks_passed"],
                "match_time_M": stage["match_time_M"],
                "boundary_time_M": stage["boundary_time_M"],
                "orbital_peak_spline_slope": stage["orbital_peak_spline_slope"],
                "amplitude_system": _system(stage, "amplitude"),
                "phase_system": _system(stage, "phase"),
                "torch_public_complex_l2": diagnostic["torch_public"][
                    "amplitude_peak_anchored"]["metrics"]["complex_relative_l2"],
            }
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(case, label, result["levels"][label]["torch_jax"], flush=True)
        result["comparisons"]["torch"] = _shared_compare(
            series["baseline"]["torch"], series["refined"]["torch"])
        result["comparisons"]["jax_unchanged"] = _shared_compare(
            series["baseline"]["jax"], series["refined"]["jax"])
        limits = plan["checks"]
        result["checks"] = {
            "torch_shared_sample_complex_relative_l2_at_most": (
                result["comparisons"]["torch"]["complex_relative_l2"]
                <= limits["torch_shared_sample_complex_relative_l2_at_most"]),
            "shared_time_max_abs_at_most_M": all(
                item["shared_time_max_abs_M"] <= limits["shared_time_max_abs_at_most_M"]
                for item in result["comparisons"].values()),
            "stationary_peak_slope_max_abs_at_most": all(
                abs(level["orbital_peak_spline_slope"])
                <= limits["stationary_peak_slope_max_abs_at_most"]
                for level in result["levels"].values()),
            "nqc_linear_system_max_abs_residual_at_most": all(
                level[f"{prefix}_system"]["max_abs_residual"]
                <= limits["nqc_linear_system_max_abs_residual_at_most"]
                for level in result["levels"].values()
                for prefix in ("amplitude", "phase")),
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

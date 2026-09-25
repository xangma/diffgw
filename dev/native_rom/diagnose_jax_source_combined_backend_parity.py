"""Dev-only independent Torch/JAX parity for the combined numerical control."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_source_cadence import _shared_compare
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _compare
from dev.native_rom.diagnose_jax_spliced_whole_waveform import (
    SOURCE_PLAN,
    _prepare_public,
    run,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "jax_source_combined_backend_parity_plan.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pilot", "prospective"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    case = plan[f"{args.case}_case"]
    prior_path = EVIDENCE / f"jax_source_combined_numerical_control_{case}.json"
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-combined-backend-parity-plan.v1"
            or plan["release_qualified"] is not False
            or plan["fine_refinement_factors"] != [256, 512]
            or plan["polynomial"] != {"half_width_M": 2.0, "degree": 4}
            or q.sha256(SOURCE_PLAN) != plan["source_plan_sha256"]
            or q.sha256(prior_path) != plan["prior_torch_combined_sha256"][case]):
        raise ValueError("frozen combined-parity plan or prerequisites changed")
    prior = json.loads(prior_path.read_text())
    source_plan = json.loads(SOURCE_PLAN.read_text())
    identity = _prepare_public(source_plan)
    output = EVIDENCE / f"jax_source_combined_backend_parity_{case}.json"
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.jax-source-combined-backend-parity.v1",
        "status": "running", "case_id": case, "plan_sha256": q.sha256(PLAN),
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
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
                diagnostic_continuous_boundary=True, diagnostic_jax_local=True,
                write_result=False, return_series=True, return_stages=True)
            if abs(stage["match_time_M"] - prior["levels"][label]["match_time_M"]) > 1e-10:
                raise ValueError(f"Torch stationary match changed: {case} {label}")
            if abs(diagnostic["torch_public"]["amplitude_peak_anchored"]["metrics"][
                    "complex_relative_l2"] - prior["levels"][label][
                        "torch_public_complex_l2"]) > 1e-10:
                raise ValueError(f"Torch physical comparison changed: {case} {label}")
            for backend, (time_grid, _) in (("torch", torch_series), ("jax", jax_series)):
                if np.max(np.abs(np.diff(time_grid) - plan["output_step_M"])) > 1e-8:
                    raise ValueError(f"whole output chart changed: {case} {backend} {label}")
            series[label] = {"torch": torch_series, "jax": jax_series}
            result["levels"][label] = {
                "settings": settings,
                "torch_jax": _compare(torch_series, jax_series),
                "native_backend_checks_passed": diagnostic["diagnostic_checks_passed"],
                "match_time_difference_M": (stage["jax_match_time_M"]
                                            - stage["match_time_M"]),
                "boundary_time_difference_M": (stage["jax_boundary_time_M"]
                                               - stage["boundary_time_M"]),
                "amplitude_coefficient_max_abs": float(np.max(np.abs(
                    stage["nqc_coefficients"][0] - stage["jax_nqc_coefficients"][0]))),
                "phase_coefficient_max_abs": float(np.max(np.abs(
                    stage["nqc_coefficients"][1] - stage["jax_nqc_coefficients"][1]))),
            }
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(case, label, result["levels"][label], flush=True)
        result["comparisons"]["torch"] = _shared_compare(
            series["baseline"]["torch"], series["refined"]["torch"])
        result["comparisons"]["jax"] = _shared_compare(
            series["baseline"]["jax"], series["refined"]["jax"])
        limits = plan["checks"]
        result["checks"] = {
            "each_grid_torch_jax_complex_relative_l2_at_most": all(
                level["torch_jax"]["complex_relative_l2"]
                <= limits["each_grid_torch_jax_complex_relative_l2_at_most"]
                for level in result["levels"].values()),
            "jax_shared_sample_complex_relative_l2_at_most": (
                result["comparisons"]["jax"]["complex_relative_l2"]
                <= limits["jax_shared_sample_complex_relative_l2_at_most"]),
            "torch_result_reproduced_abs_at_most": abs(
                result["comparisons"]["torch"]["complex_relative_l2"]
                - prior["comparisons"]["torch"]["complex_relative_l2"])
                <= limits["torch_result_reproduced_abs_at_most"],
            "shared_time_max_abs_at_most_M": all(
                item["shared_time_max_abs_M"] <= limits["shared_time_max_abs_at_most_M"]
                for item in result["comparisons"].values()),
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

"""Dev-only shared-fine-grid stage attribution after stationary peak matching."""

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
PLAN = EVIDENCE / "jax_source_stationary_stage_audit_plan.json"


def _system(stage, prefix):
    matrix = np.asarray(stage[f"nqc_{prefix}_matrix"]).squeeze()
    rhs = np.asarray(stage[f"nqc_{prefix}_rhs"]).squeeze()
    coefficients = np.asarray(stage["nqc_coefficients"][
        0 if prefix == "amplitude" else 1]).squeeze()
    residual = matrix @ coefficients - rhs
    return {
        "matrix": matrix.tolist(), "rhs": rhs.tolist(),
        "coefficients": coefficients.tolist(),
        "condition": float(np.linalg.cond(matrix)),
        "max_abs_residual": float(np.max(np.abs(residual))),
    }


def _shared(stage_a, stage_b, key_a, key_b=None):
    key_b = key_b or key_a
    ta = np.asarray(stage_a["fine_time" if not key_a.startswith("jax_")
                            else "jax_fine_time"])
    tb = np.asarray(stage_b["fine_time" if not key_b.startswith("jax_")
                            else "jax_fine_time"])[::2]
    left = np.asarray(stage_a[key_a])
    right = np.asarray(stage_b[key_b])[::2]
    common = min(len(ta), len(tb), len(left), len(right))
    left, right = left[:common], right[:common]
    return {
        "samples": common,
        "time_max_abs_M": float(np.max(np.abs(ta[:common] - tb[:common]))),
        "relative_l2": float(np.linalg.norm(left - right) / np.linalg.norm(left)),
        "max_abs": float(np.max(np.abs(left - right))),
    }


def _matrix_attribution(baseline, refined):
    matrix_a, rhs_a = np.asarray(baseline["matrix"]), np.asarray(baseline["rhs"])
    matrix_b, rhs_b = np.asarray(refined["matrix"]), np.asarray(refined["rhs"])
    coefficients_a = np.asarray(baseline["coefficients"])
    coefficients_b = np.asarray(refined["coefficients"])
    return {
        "matrix_relative_l2": float(np.linalg.norm(matrix_b - matrix_a)
                                    / np.linalg.norm(matrix_a)),
        "rhs_relative_l2": float(np.linalg.norm(rhs_b - rhs_a)
                                 / np.linalg.norm(rhs_a)),
        "coefficient_l2": float(np.linalg.norm(coefficients_b - coefficients_a)),
        "matrix_first_coefficient_l2": float(np.linalg.norm(
            np.linalg.solve(matrix_b, rhs_a) - coefficients_a)),
        "rhs_first_coefficient_l2": float(np.linalg.norm(
            np.linalg.solve(matrix_a, rhs_b) - coefficients_a)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("pilot", "prospective"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    case = plan[f"{args.case}_case"]
    prior_path = EVIDENCE / f"jax_source_stationary_peak_control_{case}.json"
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-stationary-stage-audit-plan.v1"
            or plan["release_qualified"] is not False
            or plan["fine_refinement_factors"] != [128, 256]
            or q.sha256(SOURCE_PLAN) != plan["source_plan_sha256"]
            or q.sha256(prior_path) != plan["stationary_result_sha256"][case]):
        raise ValueError("frozen stationary-stage plan or prerequisites changed")
    prior = json.loads(prior_path.read_text())
    identity = _prepare_public(json.loads(SOURCE_PLAN.read_text()))
    started = time.monotonic()
    output = EVIDENCE / f"jax_source_stationary_stage_audit_{case}.json"
    result = {
        "schema": "diffgw.seobnrv4.jax-source-stationary-stage-audit.v1",
        "status": "running", "case_id": case,
        "plan_sha256": q.sha256(PLAN), "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "levels": {}, "comparisons": {}, "checks": {},
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stages = {}
    try:
        for label, factor in zip(("baseline", "refined"), plan["fine_refinement_factors"]):
            settings = {
                "coarse_output_step_M": plan["output_step_M"],
                "fine_refinement_factor": factor,
                "max_step_M": plan["max_step_M"],
                "atol": plan["atol"], "rtol": plan["rtol"],
            }
            diagnostic, _, _, stage = run(
                case, json.loads(SOURCE_PLAN.read_text()), identity,
                variant="source", independent_init=True,
                numerical_settings=settings, diagnostic_peak_policy="stationary",
                write_result=False, return_series=True, return_stages=True)
            if not diagnostic["diagnostic_checks_passed"]:
                raise ValueError(f"backend parity failed at {case} {label}")
            for backend in ("torch", "jax"):
                match_key = "match_time_M" if backend == "torch" else "jax_match_time_M"
                if abs(stage[match_key] - prior["levels"][label][match_key]) > 1e-10:
                    raise ValueError(f"stationary match changed: {case} {backend} {label}")
            stages[label] = stage
            result["levels"][label] = {
                "settings": settings,
                "fine_input_samples": len(stage["fine_time"]),
                "jax_fine_input_samples": len(stage["jax_fine_time"]),
                "torch_jax_checks_passed": diagnostic["diagnostic_checks_passed"],
                "match_time_M": stage["match_time_M"],
                "amplitude_system": _system(stage, "amplitude"),
                "phase_system": _system(stage, "phase"),
            }
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(case, label, "stage captured", flush=True)
        a, b = stages["baseline"], stages["refined"]
        for key in ("fine_state", "fine_orbital_frequency", "fine_base_mode",
                    "fine_corrected_inspiral", "jax_fine_state", "jax_fine_base_mode",
                    "jax_fine_corrected_inspiral"):
            result["comparisons"][key] = _shared(a, b, key)
        for prefix in ("amplitude", "phase"):
            result["comparisons"][f"{prefix}_system"] = _matrix_attribution(
                result["levels"]["baseline"][f"{prefix}_system"],
                result["levels"]["refined"][f"{prefix}_system"])
        result["comparisons"]["whole_waveform_relative_l2"] = prior[
            "comparisons"]["torch"]["complex_relative_l2"]
        limits = plan["checks"]
        result["checks"] = {
            "shared_fine_time_max_abs_at_most_M": all(
                comparison["time_max_abs_M"] <= limits["shared_fine_time_max_abs_at_most_M"]
                for comparison in result["comparisons"].values()
                if isinstance(comparison, dict) and "time_max_abs_M" in comparison),
            "raw_mode_shared_relative_l2_at_most": all(
                result["comparisons"][key]["relative_l2"]
                <= limits["raw_mode_shared_relative_l2_at_most"]
                for key in ("fine_base_mode", "jax_fine_base_mode")),
            "each_level_backend_checks_passed": all(
                level["torch_jax_checks_passed"] for level in result["levels"].values()),
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

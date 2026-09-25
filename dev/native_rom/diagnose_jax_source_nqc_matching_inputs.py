"""Dev-only attribution of source-backed NQC tolerance drift to matching inputs."""

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
PLAN = EVIDENCE / "jax_source_nqc_matching_inputs_plan.json"
LADDER_PLAN = EVIDENCE / "jax_source_tolerance_ladder_plan.json"


def _l2(left, right):
    return float(np.linalg.norm(left - right))


def _system(stage, kind):
    matrix = np.asarray(stage[f"nqc_{kind}_matrix"])
    rhs = np.asarray(stage[f"nqc_{kind}_rhs"])
    coefficients = np.asarray(stage["nqc_coefficients"][0 if kind == "amplitude" else 1])
    return matrix, rhs, coefficients


def _attribution(base, refined):
    mb, rb, cb = base
    mr, rr, cr = refined
    matrix_then_rhs = np.linalg.solve(mr, rb)
    rhs_then_matrix = np.linalg.solve(mb, rr)
    return {
        "matrix_frobenius_difference": _l2(mb, mr),
        "rhs_l2_difference": _l2(rb, rr),
        "coefficient_l2_difference": _l2(cb, cr),
        "base_to_refined_matrix_first_coefficient_shift": _l2(cb, matrix_then_rhs),
        "base_to_refined_rhs_second_coefficient_shift": _l2(matrix_then_rhs, cr),
        "base_to_refined_rhs_first_coefficient_shift": _l2(cb, rhs_then_matrix),
        "base_to_refined_matrix_second_coefficient_shift": _l2(rhs_then_matrix, cr),
        "matrix_first_coefficients": matrix_then_rhs.tolist(),
        "rhs_first_coefficients": rhs_then_matrix.tolist(),
        "matrix_row_l2_differences": np.linalg.norm(mb - mr, axis=1).tolist(),
        "rhs_row_abs_differences": np.abs(rb - rr).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("q4_nonspinning", "strong_aligned"), required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    case = args.case
    stage_path = EVIDENCE / f"jax_source_tolerance_stages_{case}.json"
    curvature_path = EVIDENCE / f"jax_source_nqc_curvature_{case}.json"
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-nqc-matching-inputs-plan.v1"
            or plan["release_qualified"] is not False
            or case not in plan["cases"]
            or plan["levels"] != ["baseline", "refined"]
            or q.sha256(SOURCE_PLAN) != plan["source_plan_sha256"]
            or q.sha256(LADDER_PLAN) != plan["ladder_plan_sha256"]
            or q.sha256(stage_path) != plan["stage_result_sha256"][case]
            or q.sha256(curvature_path) != plan["curvature_result_sha256"][case]):
        raise ValueError("frozen matching-input plan or source evidence changed")
    ladder = json.loads(LADDER_PLAN.read_text())
    source_plan = json.loads(SOURCE_PLAN.read_text())
    identity = _prepare_public(source_plan)
    started = time.monotonic()
    output = EVIDENCE / f"jax_source_nqc_matching_inputs_{case}.json"
    result = {
        "schema": "diffgw.seobnrv4.jax-source-nqc-matching-inputs.v1",
        "status": "running", "case_id": case,
        "plan_sha256": q.sha256(PLAN), "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "levels": {}, "attribution": {}, "checks": {},
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stages = {}
    reconstruction_errors = []
    replay_errors = []
    for name in plan["levels"]:
        level = next(value for value in ladder["levels"] if value["name"] == name)
        settings = {
            "coarse_output_step_M": ladder["coarse_output_step_M"],
            "max_step_M": ladder["max_step_M"],
            "atol": level["atol"], "rtol": level["rtol"],
        }
        diagnostic, _, _, stage = run(
            case, source_plan, identity, variant="source", independent_init=True,
            numerical_settings=settings, write_result=False, return_series=True,
            return_stages=True)
        stages[name] = stage
        systems = {}
        for kind in ("amplitude", "phase"):
            matrix, rhs, coefficients = _system(stage, kind)
            residual = float(np.max(np.abs(matrix @ coefficients - rhs)))
            replay = float(np.max(np.abs(np.linalg.solve(matrix, rhs) - coefficients)))
            reconstruction_errors.append(residual)
            replay_errors.append(replay)
            systems[kind] = {
                "matrix": matrix.tolist(), "rhs": rhs.tolist(),
                "coefficients": coefficients.tolist(),
                "condition": stage[f"nqc_{kind}_condition"],
                "linear_system_reconstruction_max_abs": residual,
                "coefficient_replay_max_abs": replay,
            }
        result["levels"][name] = {
            "settings": settings, "match_time_M": stage["match_time_M"],
            "whole_mode_relative_l2_torch_jax": diagnostic["whole_mode_relative_l2"],
            "systems": systems,
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(case, name, {kind: systems[kind]["condition"]
                           for kind in systems}, flush=True)
    result["attribution"] = {
        kind: _attribution(_system(stages["baseline"], kind),
                           _system(stages["refined"], kind))
        for kind in ("amplitude", "phase")
    }
    limits = plan["checks"]
    result["checks"] = {
        "linear_system_reconstruction_max_abs_at_most": max(reconstruction_errors)
        <= limits["linear_system_reconstruction_max_abs_at_most"],
        "coefficient_replay_max_abs_at_most": max(replay_errors)
        <= limits["coefficient_replay_max_abs_at_most"],
    }
    result["diagnostic_checks_passed"] = all(result["checks"].values())
    result["elapsed_s"] = time.monotonic() - started
    result["status"] = "diagnostic_complete"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(case, json.dumps({"checks": result["checks"],
                            "attribution": result["attribution"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Attribute corrected-source aligned-spin local NQC drift from frozen fits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

from . import qualify_generator as q


PLAN = (q.ROOT / "dev/native_rom/evidence/2026-09-25/"
        "source_aligned_local_nqc_matrix_attribution_plan.json")


def _system(fit: dict, name: str):
    matrix = np.asarray(fit[f"{name}_matrix"], dtype=np.float64)
    rhs = np.asarray(fit[f"{name}_rhs"], dtype=np.float64)
    return matrix, rhs


def _attribution(coarse: dict, fine: dict, name: str) -> dict:
    first_matrix, first_rhs = _system(coarse, name)
    second_matrix, second_rhs = _system(fine, name)
    solve = np.linalg.solve
    first = solve(first_matrix, first_rhs)
    second = solve(second_matrix, second_rhs)
    rhs_first = solve(first_matrix, second_rhs)
    matrix_first = solve(second_matrix, first_rhs)
    rhs_contribution = 0.5 * ((rhs_first-first)+(second-matrix_first))
    matrix_contribution = 0.5 * ((matrix_first-first)+(second-rhs_first))
    if not np.allclose(rhs_contribution+matrix_contribution,
                       second-first, rtol=0, atol=1e-10):
        raise ValueError(f"{name} attribution does not close")
    rows = {}
    for index in range(len(first_rhs)):
        substitute = first_matrix.copy()
        substitute[index] = second_matrix[index]
        rows[str(index)] = (solve(substitute, first_rhs)-first).tolist()
    return {
        "coefficient_delta": (second-first).tolist(),
        "coefficient_delta_l2": float(np.linalg.norm(second-first)),
        "rhs_only_from_coarse": (rhs_first-first).tolist(),
        "matrix_only_from_coarse": (matrix_first-first).tolist(),
        "symmetric_rhs_contribution": rhs_contribution.tolist(),
        "symmetric_matrix_contribution": matrix_contribution.tolist(),
        "one_matrix_row_at_a_time_from_coarse": rows,
        "matrix_delta_max_relative": float(np.max(
            abs((second_matrix-first_matrix)/first_matrix))),
        "rhs_delta_l2": float(np.linalg.norm(second_rhs-first_rhs)),
        "coarse_condition": float(np.linalg.cond(first_matrix)),
        "fine_condition": float(np.linalg.cond(second_matrix)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    names = ("ladder_evidence", "ladder_script", "finer_evidence",
             "finer_script")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-local-nqc-matrix-attribution-plan.v1"
            or plan["release_qualified"] is not False
            or plan["steps_M"] != [0.03125, 0.015625, 0.0078125,
                                   0.00390625, 0.001953125]
            or plan["fit_reconstruction_abs_tolerance"] != 1e-8
            or any(q.sha256(path) != plan[name + "_sha256"]
                   for name, path in paths.items())):
        parser.error("Frozen plan or input changed")
    ladder = json.loads(paths["ladder_evidence"].read_text())
    finer = json.loads(paths["finer_evidence"].read_text())
    if (ladder["status"] != "diagnostic_complete"
            or finer["status"] != "diagnostic_complete"
            or ladder["row"] != finer["row"]
            or ladder["release_qualified"] is not False
            or finer["release_qualified"] is not False
            or ladder["grids"]["0.00390625"]["local_mode_sha256"]
            != finer["grids"]["0.00390625"]["local_mode_sha256"]):
        parser.error("Aligned-spin source evidence changed")
    grids = {**ladder["grids"], **finer["grids"]}
    report = {
        "schema": "diffgw.seobnrv4.source-aligned-local-nqc-matrix-attribution.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": ladder["row"], "pairs": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        for step in plan["steps_M"]:
            fit = grids[str(step)]["local_fit"]
            reconstructed = np.concatenate([np.linalg.solve(*_system(fit, name))
                                            for name in ("amplitude", "phase")])
            if np.max(abs(reconstructed-np.asarray(fit["coefficients"]))) > plan[
                    "fit_reconstruction_abs_tolerance"]:
                raise ValueError(f"Saved local fit does not reconstruct at {step:g} M")
        for first_step, second_step in zip(plan["steps_M"][:-1],
                                           plan["steps_M"][1:], strict=True):
            first = grids[str(first_step)]["local_fit"]
            second = grids[str(second_step)]["local_fit"]
            pair = f"{first_step:g}_to_{second_step:g}"
            report["pairs"][pair] = {
                name: _attribution(first, second, name)
                for name in ("amplitude", "phase")
            }
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}))
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

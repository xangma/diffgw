"""Check whether the frozen strong-spin NQC linear solve is the error source.

This dev-only diagnostic uses saved matching matrices. It changes neither the
source trajectory nor the derivative estimator, NQC rule, or waveform.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

import mpmath as mp
import numpy as np

from . import qualify_generator as q


PLAN = (q.ROOT / "dev/native_rom/evidence/2026-09-25/"
        "source_aligned_nqc_solve_condition_plan.json")


def _system(fit: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(fit[f"{name}_matrix"], dtype=np.float64)
    rhs = np.asarray(fit[f"{name}_rhs"], dtype=np.float64)
    if (matrix.shape != (len(rhs), len(rhs)) or len(rhs) not in (2, 3)
            or not np.isfinite(matrix).all() or not np.isfinite(rhs).all()):
        raise ValueError(f"Invalid {name} matching system")
    return matrix, rhs


def _high_precision_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    # Decimal representations of the saved binary64 inputs are held exact here;
    # this tests the solve only, not the accuracy of the saved derivative rows.
    rows = [[mp.mpf(repr(float(value))) for value in row] for row in matrix]
    values = [mp.mpf(repr(float(value))) for value in rhs]
    return np.asarray([float(x) for x in mp.lu_solve(mp.matrix(rows), mp.matrix(values))])


def _record(fit: dict, name: str, saved: np.ndarray, gate: dict) -> dict:
    matrix, rhs = _system(fit, name)
    floating = np.linalg.solve(matrix, rhs)
    precise = _high_precision_solve(matrix, rhs)
    row_scale = 1.0 / np.max(abs(matrix), axis=1)
    row_scaled = row_scale[:, None] * matrix
    column_scale = 1.0 / np.max(abs(row_scaled), axis=0)
    equilibrated = row_scaled * column_scale[None, :]
    equilibrated_solve = column_scale * np.linalg.solve(equilibrated, row_scale * rhs)
    reconstruction = float(np.max(abs(floating - saved)))
    return {
        "coefficients_saved": saved.tolist(),
        "coefficients_binary64": floating.tolist(),
        "coefficients_100_digit": precise.tolist(),
        "saved_reconstruction_abs_linf": reconstruction,
        "binary64_vs_100_digit_abs_linf": float(np.max(abs(floating - precise))),
        "equilibrated_vs_100_digit_abs_linf": float(np.max(abs(equilibrated_solve - precise))),
        "residual_abs_linf": float(np.max(abs(matrix @ floating - rhs))),
        "condition_2": float(np.linalg.cond(matrix)),
        "equilibrated_condition_2": float(np.linalg.cond(equilibrated)),
        "gates": {
            "saved_reconstruction": reconstruction <= gate["saved_fit_reconstruction_abs_linf"],
            "binary64_vs_high_precision": float(np.max(abs(floating - precise)))
                <= gate["binary64_vs_high_precision_abs_linf"],
        },
    }


def _pair(first: dict, second: dict, name: str, gate: dict) -> dict:
    matrix0, rhs0 = _system(first, name)
    matrix1, rhs1 = _system(second, name)
    coefficient0 = np.linalg.solve(matrix0, rhs0)
    coefficient1 = np.linalg.solve(matrix1, rhs1)
    actual = coefficient1 - coefficient0
    delta_matrix = matrix1 - matrix0
    delta_rhs = rhs1 - rhs0
    first_order = np.linalg.solve(matrix0, delta_rhs - delta_matrix @ coefficient0)
    relative_error = float(np.linalg.norm(actual - first_order) / np.linalg.norm(actual))
    return {
        "actual_coefficient_delta": actual.tolist(),
        "actual_coefficient_delta_l2": float(np.linalg.norm(actual)),
        "first_order_coefficient_delta": first_order.tolist(),
        "first_order_pair_delta_relative_error": relative_error,
        "matrix_delta": delta_matrix.tolist(),
        "rhs_delta": delta_rhs.tolist(),
        "matrix_term_l2": float(np.linalg.norm(np.linalg.solve(matrix0, -delta_matrix @ coefficient0))),
        "rhs_term_l2": float(np.linalg.norm(np.linalg.solve(matrix0, delta_rhs))),
        "gate": relative_error <= gate["first_order_pair_delta_relative_error"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / entry["path"]
             for name, entry in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-nqc-solve-condition-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row_name"] != "strong_aligned_spin"
            or plan["pilot_stencils_M"] != [0.1, 0.2]
            or plan["fine_steps_M"] != [0.03125, 0.015625, 0.0078125,
                                        0.00390625, 0.001953125]
            or plan["mpmath_decimal_digits"] != 100
            or plan["gates"] != {
                "saved_fit_reconstruction_abs_linf": 1e-8,
                "binary64_vs_high_precision_abs_linf": 1e-5,
                "first_order_pair_delta_relative_error": 0.05,
            }
            or any(q.sha256(paths[name]) != entry["sha256"]
                   for name, entry in plan["inputs"].items())):
        parser.error("Frozen plan or inputs changed")
    pilot, ladder, fine = (json.loads(paths[name].read_text())
                           for name in ("field_pilot", "local_ladder", "very_fine"))
    if (any(item["status"] != "diagnostic_complete"
            or item["release_qualified"] is not False
            for item in (pilot, ladder, fine))
            or pilot["gates"]["coefficient_stencil"] is not False
            or pilot["stencil_coefficient_l2"] <= 0.01
            or ladder["row"] != fine["row"]
            or ladder["grids"]["0.00390625"]["local_mode_sha256"]
            != fine["grids"]["0.00390625"]["local_mode_sha256"]):
        parser.error("Frozen failed gate or aligned-spin captures changed")
    report = {
        "schema": "diffgw.seobnrv4.source-aligned-nqc-solve-condition.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "historical_coefficient_stencil_l2": pilot["stencil_coefficient_l2"],
        "historical_0p01_gate": False,
        "limitations": plan["interpretation"],
        "systems": {}, "pairs": {}, "gates": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        mp.mp.dps = plan["mpmath_decimal_digits"]
        fits = {
            "pilot_0.1": pilot["field"]["0.1"]["all_rows"],
            "pilot_0.2": pilot["field"]["0.2"]["all_rows"],
        }
        grids = {**ladder["grids"], **fine["grids"]}
        fits.update({f"grid_{step}": grids[str(step)]["local_fit"]
                     for step in plan["fine_steps_M"]})
        gate = plan["gates"]
        for label, fit in fits.items():
            saved = np.asarray(fit["coefficients"], dtype=np.float64)
            if saved.shape != (5,):
                raise ValueError(f"Invalid saved coefficients: {label}")
            report["systems"][label] = {
                "amplitude": _record(fit, "amplitude", saved[:3], gate),
                "phase": _record(fit, "phase", saved[3:], gate),
            }
        pairs = [("pilot_0.1", "pilot_0.2")]
        pairs.extend((f"grid_{a}", f"grid_{b}")
                     for a, b in zip(plan["fine_steps_M"][:-1],
                                     plan["fine_steps_M"][1:], strict=True))
        for first, second in pairs:
            report["pairs"][f"{first}_to_{second}"] = {
                name: _pair(fits[first], fits[second], name, gate)
                for name in ("amplitude", "phase")
            }
        report["gates"] = {
            "saved_reconstruction": all(
                system["gates"]["saved_reconstruction"]
                for record in report["systems"].values() for system in record.values()),
            "binary64_vs_high_precision": all(
                system["gates"]["binary64_vs_high_precision"]
                for record in report["systems"].values() for system in record.values()),
            "first_order_pair_delta": all(
                record["gate"] for pair in report["pairs"].values()
                for record in pair.values()),
            "historical_0p01_coefficient_stencil": False,
        }
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "gates": report["gates"],
                      "errors": report["errors"]}))
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

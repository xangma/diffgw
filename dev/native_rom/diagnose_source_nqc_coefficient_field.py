"""Replay source NQC fits with source-cubic and regular-field derivatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import qualify_generator as q


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_coefficient_field_plan.json"


def fit(derivatives: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Apply the pinned v4 NQC linear systems to seven value/derivative rows."""
    value, first, second = derivatives.T
    amplitude_matrix = np.stack((value[1:4], first[1:4], second[1:4]))
    amplitude_rhs = targets[:3] - np.array((value[0], first[0], second[0]))
    phase_matrix = -np.stack((first[5:7], second[5:7]))
    frequency = np.array((abs(first[4]),
                          abs(second[4]) if first[4] * second[4] > 0
                          else -abs(second[4])))
    phase_rhs = targets[3:] - frequency
    coefficients = np.r_[np.linalg.solve(amplitude_matrix, amplitude_rhs),
                         np.linalg.solve(phase_matrix, phase_rhs)]
    return coefficients, float(np.linalg.cond(amplitude_matrix)), float(np.linalg.cond(phase_matrix))


def dense_derivatives(stencil: dict) -> np.ndarray:
    return np.stack((stencil["center"], stencil["first"], stencil["second"]), axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-coefficient-field-plan.v1"
            or plan["release_qualified"] is not False
            or plan["rows"] != ["strong_aligned_spin", "nonspinning", "mild_mixed_spin"]
            or plan["factors"] != [8, 16, 32]
            or plan["selected_stencil_M"] != "0.1"
            or plan["replay_absolute_tolerance"] != 3e-4):
        parser.error("Frozen coefficient-field plan changed")
    report = {
        "schema": "diffgw.seobnrv4.source-nqc-coefficient-field.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "scope": dict(**q.SCOPE, development_only=True),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "source_derivative_order": ["amplitude", "A*q1", "A*q2", "A*q3",
                                    "mode_phase", "p1", "p1*pr2"],
        "source": "Captured GSL natural-cubic input derivatives",
        "counterfactual": "Centered DOP853 integration of pinned regular-v4 field from the source match state",
        "rows": {},
    }
    for name in plan["rows"]:
        by_factor = {}
        for factor in plan["factors"]:
            entry = plan["inputs"][name][str(factor)]
            path = q.ROOT / entry["path"]
            if q.sha256(path) != entry["sha256"]:
                parser.error(f"Frozen source diagnostic changed: {path}")
            source = json.loads(path.read_text())
            if (source["status"] != "diagnostic_complete"
                    or source["release_qualified"] is not False
                    or source["row_name"] != name
                    or source["resample_factor"] != factor):
                parser.error(f"Source diagnostic contract changed: {path}")
            targets = np.asarray(source["source"]["meta"][3:], dtype=float)
            captured = np.asarray(source["source"]["captured_gsl_derivatives"], dtype=float)
            active = np.asarray(source["source"]["active_nqc_coefficients"], dtype=float)
            if targets.shape != (5,) or captured.shape != (7, 3) or active.shape != (5,):
                parser.error(f"Source fit dimensions changed: {path}")
            replay, source_amp_cond, source_phase_cond = fit(captured, targets)
            replay_error = float(np.max(np.abs(replay - active)))
            if replay_error > plan["replay_absolute_tolerance"]:
                parser.error(f"Source coefficient replay failed: {path}: {replay_error}")
            selected = source["centered_dop853"][-1]["stencils"][plan["selected_stencil_M"]]
            counter, dense_amp_cond, dense_phase_cond = fit(dense_derivatives(selected), targets)
            if not np.all(np.isfinite(counter)):
                parser.error(f"Nonfinite dense-field fit: {path}")
            numerical_spread = {}
            for stencil_M in ("0.2", "0.1", "0.05"):
                coefficients = [fit(dense_derivatives(run["stencils"][stencil_M]), targets)[0]
                                for run in source["centered_dop853"]]
                numerical_spread[stencil_M] = {
                    "tolerance_pair_max_absolute": float(np.max(np.abs(coefficients[0] - coefficients[1]))),
                    "from_selected_max_absolute": float(max(np.max(np.abs(c - counter))
                                                        for c in coefficients)),
                }
            by_factor[str(factor)] = {
                "input_sha256": entry["sha256"],
                "input_array_sha256": source["source"]["input_array_sha256"],
                "match_time_M": source["source"]["match_time_M"],
                "source_active_coefficients": active.tolist(),
                "source_fit_replay_max_absolute": replay_error,
                "source_amplitude_condition": source_amp_cond,
                "source_phase_condition": source_phase_cond,
                "dense_field_coefficients": counter.tolist(),
                "dense_amplitude_condition": dense_amp_cond,
                "dense_phase_condition": dense_phase_cond,
                "dense_minus_source": (counter - active).tolist(),
                "dense_vs_source_max_relative": float(np.max(np.abs((counter - active) / active))),
                "dense_numerical_spread": numerical_spread,
            }
        x, y = by_factor["16"], by_factor["32"]
        report["rows"][name] = {
            "factors": by_factor,
            "factor_16_to_32": {
                "match_time_difference_M": y["match_time_M"] - x["match_time_M"],
                "source_coefficient_l2": float(np.linalg.norm(np.array(y["source_active_coefficients"])
                                                           - x["source_active_coefficients"])),
                "dense_coefficient_l2": float(np.linalg.norm(np.array(y["dense_field_coefficients"])
                                                          - x["dense_field_coefficients"])),
            },
        }
    q.write_evidence(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

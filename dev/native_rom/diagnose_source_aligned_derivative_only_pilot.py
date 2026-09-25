"""Replay one frozen source NQC solve with field-derived derivative rows.

This is a coefficient-level diagnostic. It does not generate or alter a waveform.
The field derivatives were independently integrated and recorded by
``diagnose_source_nqc_dense_field``; the captured match event, values, targets,
and historical coefficient solve remain fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/source_aligned_derivative_only_pilot_plan.json"
NAMES = ("amplitude", "A*q1", "A*q2", "A*q3", "mode_phase", "p1", "p1*pr2")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _solve(derivatives: np.ndarray, targets: np.ndarray) -> dict:
    """Use the recorded source matching equations, changing derivative data only."""
    amp_matrix = np.stack((derivatives[1:4, 0], derivatives[1:4, 1],
                           derivatives[1:4, 2]))
    amp_rhs = targets[:3] - derivatives[0, :3]
    phase_matrix = -np.stack((derivatives[5:7, 1], derivatives[5:7, 2]))
    raw_frequency, raw_derivative = derivatives[4, 1:3]
    signed_derivative = (abs(raw_derivative) if raw_frequency * raw_derivative > 0
                         else -abs(raw_derivative))
    phase_rhs = targets[3:] - (abs(raw_frequency), signed_derivative)
    amplitude = np.linalg.solve(amp_matrix, amp_rhs)
    phase = np.linalg.solve(phase_matrix, phase_rhs)
    return {
        "coefficients": np.concatenate((amplitude, phase)).tolist(),
        "amplitude_matrix_condition": float(np.linalg.cond(amp_matrix)),
        "phase_matrix_condition": float(np.linalg.cond(phase_matrix)),
        "amplitude_residual_linf": float(np.max(abs(amp_matrix @ amplitude - amp_rhs))),
        "phase_residual_linf": float(np.max(abs(phase_matrix @ phase - phase_rhs))),
        "amplitude_matrix": amp_matrix.tolist(),
        "phase_matrix": phase_matrix.tolist(),
        "amplitude_rhs": amp_rhs.tolist(),
        "phase_rhs": phase_rhs.tolist(),
    }


def _field_derivatives(captured: np.ndarray, stencil: dict) -> np.ndarray:
    # Keep values at the historical match event; change only first/second rows.
    result = captured.copy()
    result[:, 1] = stencil["first"]
    result[:, 2] = stencil["second"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    source_path = ROOT / plan["source_evidence"]
    if (plan["schema"] != "diffgw.seobnrv4.aligned-derivative-only-pilot-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_evidence_sha256"] != _sha256(source_path)
            or plan["script_sha256"] != _sha256(Path(__file__))
            or plan["series_names"] != list(NAMES)
            or plan["centered_stencils_M"] != [0.1, 0.2]
            or plan["fixed_captured_values"] is not True
            or plan["fixed_match_event"] is not True
            or plan["fixed_targets"] is not True):
        parser.error("Frozen pilot plan or input changed")
    source = json.loads(source_path.read_text())
    if (source["status"] != "diagnostic_complete"
            or source["row"] != plan["row"]
            or source["resample_factor"] != plan["resample_factor"]
            or source["source"]["match_time_M"] != plan["match_time_M"]
            or source["source"]["meta"][1] != plan["internal_step_M"]
            or len(source["centered_dop853"]) != 2):
        parser.error("Frozen strong-spin capture changed")

    captured = np.asarray(source["source"]["captured_gsl_derivatives"], dtype=np.float64)
    targets = np.asarray(source["source"]["meta"][3:], dtype=np.float64)
    active = np.asarray(source["source"]["active_nqc_coefficients"], dtype=np.float64)
    if captured.shape != (7, 3) or targets.shape != (5,) or active.shape != (5,):
        parser.error("Unexpected NQC evidence dimensions")
    report = {
        "schema": "diffgw.seobnrv4.aligned-derivative-only-pilot.v1",
        "status": "running", "release_qualified": False,
        "plan_sha256": _sha256(PLAN), "script_sha256": _sha256(Path(__file__)),
        "source_evidence_sha256": _sha256(source_path),
        "row": source["row"], "resample_factor": source["resample_factor"],
        "match_time_M": source["source"]["match_time_M"],
        "internal_step_M": source["source"]["meta"][1],
        "fixed": {"trajectory": source["source"]["input_array_sha256"],
                  "match_event_M": source["source"]["match_time_M"],
                  "targets": targets.tolist(), "ringdown": "not evaluated or modified"},
        "series_names": list(NAMES), "captured_active": active.tolist(),
        "captured_derivatives": captured.tolist(), "field": {}, "gates": {},
    }
    _write(args.json_out, report)
    try:
        baseline = _solve(captured, targets)
        report["baseline"] = baseline
        report["baseline_replay_linf"] = float(np.max(abs(
            np.asarray(baseline["coefficients"]) - active)))

        tight = source["centered_dop853"][1]
        loose = source["centered_dop853"][0]
        candidates = {}
        for step in plan["centered_stencils_M"]:
            key = str(step)
            dense = _field_derivatives(captured, tight["stencils"][key])
            loose_dense = _field_derivatives(captured, loose["stencils"][key])
            solved = _solve(dense, targets)
            amp_only = dense.copy()
            amp_only[4:] = captured[4:]
            phase_only = captured.copy()
            phase_only[4:] = dense[4:]
            amp_solve = _solve(amp_only, targets)
            phase_solve = _solve(phase_only, targets)
            report["field"][key] = {
                "derivatives": dense.tolist(),
                "minus_spline_first": (dense[:, 1]-captured[:, 1]).tolist(),
                "minus_spline_second": (dense[:, 2]-captured[:, 2]).tolist(),
                "tolerance_derivative_linf": float(np.max(abs(
                    dense[:, 1:]-loose_dense[:, 1:]))),
                "all_rows": solved,
                "amplitude_rows_only": amp_solve["coefficients"],
                "phase_rows_only": phase_solve["coefficients"],
                "all_minus_spline_coefficients": (
                    np.asarray(solved["coefficients"])-active).tolist(),
            }
            candidates[key] = (dense, solved)
        first, second = (candidates[str(step)] for step in plan["centered_stencils_M"])
        report["stencil_derivative_linf"] = float(np.max(abs(
            first[0][:, 1:]-second[0][:, 1:])))
        report["stencil_coefficient_l2"] = float(np.linalg.norm(
            np.asarray(first[1]["coefficients"])
            - np.asarray(second[1]["coefficients"])))

        gates = plan["gates"]
        report["gates"] = {
            "baseline_replay": report["baseline_replay_linf"] <= gates["baseline_replay_linf"],
            "dense_tolerance": max(x["tolerance_derivative_linf"] for x in report["field"].values())
                <= gates["dense_tolerance_derivative_linf"],
            "dense_stencil": report["stencil_derivative_linf"]
                <= gates["dense_stencil_derivative_linf"],
            "coefficient_stencil": report["stencil_coefficient_l2"]
                <= gates["coefficient_stencil_l2"],
            "linear_residual": max(
                result[key] for result in (baseline, *(item[1] for item in candidates.values()))
                for key in ("amplitude_residual_linf", "phase_residual_linf")
            ) <= gates["linear_residual_linf"],
        }
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    _write(args.json_out, report)
    print(json.dumps({"status": report["status"], "gates": report["gates"],
                      "error": report.get("error")}))
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

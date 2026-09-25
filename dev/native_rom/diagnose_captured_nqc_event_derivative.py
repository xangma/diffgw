"""Dev-only fixed-input SEOBNRv4 NQC event x derivative attribution.

The two pre-NQC arrays are old, pinned captures. This script neither invokes
the waveform generator nor modifies production code. The local polynomial is
a diagnostic derivative control, not a proposed production prescription.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from . import qualify_generator as q
from .diagnose_native_source_nqc_inputs import _fit
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_source_flux_initial import _libraries
from .diagnose_source_nqc_high_rate import LocalPolynomial


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/captured_nqc_event_derivative_plan.json"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stationary_peak(time: np.ndarray, omega: np.ndarray) -> dict:
    index = int(np.argmax(omega))
    if not 1 <= index < len(time) - 1:
        raise ValueError("Frequency maximum has no two-sided peak bracket")
    spline = CubicSpline(time, omega)
    left, right = time[index - 1], time[index + 1]
    slope = spline.derivative()
    if not slope(left) > 0 > slope(right):
        raise ValueError("Captured frequency does not bracket a stationary peak")
    root = float(brentq(slope, left, right, xtol=1e-12))
    return {
        "bracket_M": [float(left), float(right)],
        "sample_max_index": index,
        "stationary_time_M": root,
        "frequency": float(spline(root)),
        "slope": float(slope(root)),
        "curvature": float(spline.derivative(2)(root)),
    }


def _input_derivatives(method, time: np.ndarray, values: np.ndarray,
                       match: float) -> list[list[float]]:
    amplitude, phase, radius, momentum, omega = values.T
    p1 = momentum / (radius * omega)
    q1 = p1 * p1
    series = np.column_stack((amplitude, amplitude * q1, amplitude * q1 / radius,
                              amplitude * q1 / radius**1.5, phase, p1,
                              p1 * momentum * momentum))
    return [method.derivatives(time, series[:, j], match) for j in range(7)]


def _profile(time: np.ndarray, values: np.ndarray, coefficients: np.ndarray,
             match: float, relative: np.ndarray) -> np.ndarray:
    samples = CubicSpline(time, values, axis=0)(match + relative)
    amplitude, phase, radius, momentum, omega = samples.T
    p1 = momentum / (radius * omega)
    q1 = p1 * p1
    corrected_amplitude = amplitude * (1 + coefficients[0] * q1
                                       + coefficients[1] * q1 / radius
                                       + coefficients[2] * q1 / radius**1.5)
    corrected_phase = phase - coefficients[3] * p1 - coefficients[4] * p1 * momentum**2
    corrected_phase -= corrected_phase[len(relative) // 2]
    return corrected_amplitude * np.exp(1j * corrected_phase)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    keys = ("source_array", "native_array", "prior_fit", "prior_stability",
            "source_flux_plan", "native_waveform")
    paths = {key: q.ROOT / plan[key] for key in keys}
    if (plan["schema"] != "diffgw.seobnrv4.captured-nqc-event-derivative-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row_name"] != "strong_aligned_spin"
            or plan["event_rules"] != ["reported_match", "stationary_frequency_peak_minus_reported_offset"]
            or plan["derivative_rules"] != ["historical_gsl_cubic", "local_degree4_pm2M"]
            or plan["profile_relative_window_M"] != [-2.0, 2.0]
            or plan["profile_samples"] != 257
            or plan["historical_replay_linf"] != 0.001
            or any(_hash(paths[key]) != plan[key + "_sha256"] for key in keys)
            or _hash(Path(__file__)) != plan["script_sha256"]):
        parser.error("Frozen plan or captured evidence changed")
    prior = json.loads(paths["prior_fit"].read_text())
    stability = json.loads(paths["prior_stability"].read_text())
    native_waveform = json.loads(paths["native_waveform"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or stability["status"] != "diagnostic_complete"
            or stability["gates"]["coefficient_stencil"] is not False
            or stability["stencil_coefficient_l2"] <= 0.01
            or native_waveform["status"] != "diagnostic_complete"):
        parser.error("Historical replay or 0.01 stability failure changed")
    with np.load(paths["source_array"]) as source_data:
        source_values = source_data["values"].copy()
        source_meta = source_data["meta"].copy()
        source_active = source_data["coefficients"][[0, 1, 2, 6, 7]].copy()
    with np.load(paths["native_array"]) as native_data:
        native_time = native_data["time"].copy()
        native_values = native_data["values"].T.copy()
        native_match = float(native_data["match"])
        native_active = native_data["coefficients"].copy()
    source_time = np.arange(len(source_values), dtype=np.float64) * source_meta[1]
    source_match = float(source_meta[2])
    targets = source_meta[3:].copy()
    if (source_values.shape[1] != 5 or native_values.shape[1] != 5
            or not np.all(np.diff(native_time) > 0)
            or abs(source_meta[1] - plan["captured_step_M"]) > 1e-12
            or abs(np.median(np.diff(native_time)) - plan["captured_step_M"]) > 1e-12
            or abs(source_match - prior["comparison"]["source_match_time_M"]) > 1e-10
            or abs(native_match - prior["comparison"]["native_match_time_M"]) > 1e-10):
        parser.error("Captured row or sample grids changed")
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    native_level = native_waveform["candidate"]["ode_2"]["native"]
    native_reported_peak = (native_level["peak_time_M"]
                            - native_level["restart_time_M"])
    gsl = GSLCubic(_libraries(source_plan["pinned_wheel"])["libgsl.28.dylib"])
    local = LocalPolynomial(2.0, 4)
    report = {
        "schema": "diffgw.seobnrv4.captured-nqc-event-derivative.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": _hash(PLAN), "script_sha256": _hash(Path(__file__)),
        "input_sha256": {key: _hash(path) for key, path in paths.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": prior["comparison"].get("row", plan["row"]),
        "fixed": {"source_sample_count": len(source_time),
                  "native_sample_count": len(native_time),
                  "source_match_M": source_match, "native_match_M": native_match,
                  "source_targets": targets.tolist(),
                  "trajectory_and_raw_mode": "pinned arrays, unchanged",
                  "ringdown": "not evaluated or modified"},
        "historical_stability_control": {
            "different_capture_factor": 32,
            "coefficient_stencil_l2": stability["stencil_coefficient_l2"],
            "gate_at_0.01": stability["gates"]["coefficient_stencil"],
            "interpretation": "Prior failure retained; the present 0.015625 M arrays do not resolve it."},
        "rows": {}, "cross_row": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    started = time.monotonic()
    try:
        records = {}
        for name, time_axis, values, match, reported_peak, active, historical in (
                ("source", source_time, source_values, source_match, source_meta[0],
                 source_active, prior["comparison"]["source_fit"]),
                ("native", native_time, native_values, native_match, native_reported_peak,
                 native_active, prior["comparison"]["native_fit_with_source_targets"])):
            peak = _stationary_peak(time_axis, values[:, 4])
            offset = float(reported_peak - match)
            stationary_match = float(peak["stationary_time_M"] - offset)
            events = {"reported_match": match,
                      "stationary_frequency_peak_minus_reported_offset": stationary_match}
            fits = {}
            for event_name, event in events.items():
                fits[event_name] = {}
                for method_name, method in (("historical_gsl_cubic", gsl),
                                            ("local_degree4_pm2M", local)):
                    fit = _fit(method, time_axis, *values.T, event, targets)
                    coefficients = np.asarray(fit["coefficients"])
                    fit["amplitude_residual_linf"] = float(np.max(abs(
                        np.asarray(fit["amplitude_matrix"]) @ coefficients[:3]
                        - np.asarray(fit["amplitude_rhs"]))))
                    fit["phase_residual_linf"] = float(np.max(abs(
                        np.asarray(fit["phase_matrix"]) @ coefficients[3:]
                        - np.asarray(fit["phase_rhs"]))))
                    fit["input_derivatives_value_first_second"] = _input_derivatives(
                        method, time_axis, values, event)
                    fits[event_name][method_name] = fit
            historical_fit = fits["reported_match"]["historical_gsl_cubic"]
            replay_active = float(np.max(abs(np.asarray(historical_fit["coefficients"]) - active)))
            replay_prior = float(np.max(abs(np.asarray(historical_fit["coefficients"])
                                            - np.asarray(historical["coefficients"]))))
            if max(replay_active if name == "source" else replay_prior,
                   replay_prior) > plan["historical_replay_linf"]:
                raise ValueError(f"{name} historical GSL fit failed replay")
            report["rows"][name] = {
                "array_sha256": _hash(paths[name + "_array"]),
                "reported_peak_time_M": float(reported_peak), "stationary_peak": peak,
                "peak_minus_reported_M": float(peak["stationary_time_M"] - reported_peak),
                "reported_peak_slope": float(CubicSpline(time_axis, values[:, 4]).derivative()(reported_peak)),
                "event_offset_M": offset, "stationary_match_time_M": stationary_match,
                "historical_active_coefficients": active.tolist(),
                "historical_replay_active_linf": replay_active,
                "historical_replay_prior_linf": replay_prior,
                "fits": fits,
            }
            records[name] = (time_axis, values, match, fits)
        relative = np.linspace(*plan["profile_relative_window_M"], plan["profile_samples"])
        for event_name in plan["event_rules"]:
            report["cross_row"][event_name] = {}
            for method_name in plan["derivative_rules"]:
                source_fit = records["source"][3][event_name][method_name]
                native_fit = records["native"][3][event_name][method_name]
                source_coeff = np.asarray(source_fit["coefficients"])
                native_coeff = np.asarray(native_fit["coefficients"])
                source_time_axis, source_samples, source_match_time, _ = records["source"]
                native_time_axis, native_samples, native_match_time, _ = records["native"]
                source_mode = _profile(source_time_axis, source_samples, source_coeff,
                                       source_match_time, relative)
                native_mode = _profile(native_time_axis, native_samples, native_coeff,
                                       native_match_time, relative)
                report["cross_row"][event_name][method_name] = {
                    "coefficient_gap_l2": float(np.linalg.norm(source_coeff - native_coeff)),
                    "corrected_pre_attachment_complex_relative_l2": float(
                        np.linalg.norm(source_mode - native_mode) / np.linalg.norm(source_mode)),
                    "profile_samples": len(relative),
                    "phase_alignment": "independent corrected phase at historical match",
                }
        report["historical_gap_replay_l2"] = report["cross_row"]["reported_match"][
            "historical_gsl_cubic"]["coefficient_gap_l2"]
        prior_fit_gap = float(np.linalg.norm(
            np.asarray(prior["comparison"]["source_fit"]["coefficients"])
            - np.asarray(prior["comparison"]["native_fit_with_source_targets"]["coefficients"])))
        report["prior_historical_fit_gap_l2"] = prior_fit_gap
        if abs(report["historical_gap_replay_l2"] - prior_fit_gap) > plan["historical_replay_linf"]:
            raise ValueError("Historical source-native coefficient gap changed")
        report["elapsed_seconds"] = time.monotonic() - started
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

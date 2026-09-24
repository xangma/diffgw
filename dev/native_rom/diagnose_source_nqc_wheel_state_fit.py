"""Fit source NQC systems using the wheel's public high-rate states."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import brentq

from . import qualify_generator as q
from .diagnose_nqc_c_same_input import bind, compile_wrapper
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_public_nqc_mode_boundary import public_call
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_coefficient_field import fit
from .diagnose_source_nqc_dense_field import source_series


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_wheel_state_fit_plan.json"
ACTIVE = (0, 1, 2, 6, 7)


def derivatives(gsl: GSLCubic, axis: np.ndarray, inputs: np.ndarray,
                match: float) -> np.ndarray:
    basis = source_series(inputs)
    basis[:, 4] = np.unwrap(basis[:, 4])
    return np.asarray([gsl.derivatives(axis, basis[:, i], match)
                       for i in range(basis.shape[1])])


def local_indices(axis: np.ndarray, center: float, half_width: float) -> np.ndarray:
    indices = np.flatnonzero(abs(axis - center) <= half_width)
    if len(indices) < 30 or not axis[indices[0]] < center < axis[indices[-1]]:
        raise ValueError("Local fit window does not contain enough centered samples")
    return indices


def source_peak_rule(gsl: GSLCubic, axis: np.ndarray,
                     omega: np.ndarray) -> dict:
    """Replay the pinned generator's first-decline and derivative-bisection rule."""
    declines = np.flatnonzero(np.diff(omega) <= 0)
    if not len(declines):
        raise ValueError("No sampled orbital-frequency decline")
    peak = int(declines[0] + 1)
    if peak < 3 or peak >= len(axis)-3:
        raise ValueError("Orbital peak is at a window boundary")

    def derivative(time: float) -> float:
        return gsl.derivatives(axis, omega, time)[1]

    first_decline = peak
    time1 = float(axis[peak])
    first_derivative = derivative(time1)
    if first_derivative > 0:
        time2 = float(axis[peak + 1])
    else:
        time2 = time1
        time1 = float(axis[peak - 1])
        peak -= 1
    left_derivative = derivative(time1)
    initial_left_derivative = left_derivative
    initial_right_derivative = derivative(time2)
    while True:
        time_peak = (time1 + time2) / 2
        mid_derivative = derivative(time_peak)
        if mid_derivative * left_derivative < 0:
            time2 = time_peak
        else:
            left_derivative = mid_derivative
            time1 = time_peak
        if time2 - time1 <= 1e-5:
            break
    sampled_maximum = int(np.argmax(omega))
    left, right = float(axis[sampled_maximum-1]), float(axis[sampled_maximum+1])
    true_spline_peak = float(brentq(derivative, left, right, xtol=1e-12))
    return {
        "time_M": float(time_peak),
        "first_decline_index": first_decline,
        "first_decline_time_M": float(axis[first_decline]),
        "sampled_maximum_time_M": float(axis[sampled_maximum]),
        "initial_left_derivative": initial_left_derivative,
        "initial_right_derivative": initial_right_derivative,
        "initial_derivatives_bracket_root": initial_left_derivative * initial_right_derivative < 0,
        "true_spline_peak_M": true_spline_peak,
        "rule_minus_true_spline_peak_M": float(time_peak-true_spline_peak),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    names = ("source_inputs", "matched_public", "source_wheel_mode",
             "source_flux_plan", "field_adapter", "mode_adapter", "nqc_adapter",
             "high_adapter", "generator_adapter", "exact_state_adapter")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-wheel-state-fit-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row"] != "strong_aligned_spin"
            or plan["step_M"] != 0.125 or plan["high_step_M"] != 0.015625
            or plan["local_half_width_M"] != 4.0
            or plan["peak_half_width_M"] != 4.0
            or plan["diagnostic_match_shifts_M"] != [
                -0.0625, -0.046875, -0.03125, -0.015625, 0.0,
                0.015625, 0.03125, 0.046875, 0.0625]
            or any(q.sha256(paths[name]) != plan[name + "_sha256"] for name in names)):
        parser.error("Frozen wheel-state fit plan or prerequisites changed")
    source_wheel = json.loads(paths["source_wheel_mode"].read_text())
    matched_public = json.loads(paths["matched_public"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    frozen = next(item for item in source_wheel["rows"] if item["name"] == plan["row"])
    public_frozen = next(item for item in matched_public["rows"] if item["name"] == plan["row"])
    row = frozen["row"]
    array = np.load(paths["source_inputs"])
    meta, values, coefficients = (array[key] for key in ("meta", "values", "coefficients"))
    if (meta.shape != (8,) or values.shape[1] != 5 or coefficients.shape != (10,)
            or not np.array_equal(coefficients, frozen["source_nqc_coefficients"])):
        parser.error("Frozen source high-rate inputs do not reproduce")
    libraries = _libraries(source_plan["pinned_wheel"])
    gsl = GSLCubic(libraries["libgsl.28.dylib"])
    report = {
        "schema": "diffgw.seobnrv4.source-nqc-wheel-state-fit.v1",
        "scope": dict(
            q.SCOPE, development_only=True,
            description="Diagnostic source NQC fits on captured source and public wheel states.",
            known_limitations=[
                "The wheel's private raw-mode and NQC input arrays are unavailable.",
                "The source C fit is a separate compilation, not the installed wheel build.",
                "One strong-spin row and a match-time scan do not qualify the model.",
                "No gradients, batching, accelerator, or ROM training are qualified.",
            ],
        ),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "pinned_libraries_sha256": {name: q.sha256(path) for name, path in libraries.items()},
        "row": row, "source": {}, "wheel": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        source_axis = np.arange(len(values)) * meta[1]
        source_match = float(meta[2])
        source_local = local_indices(source_axis, source_match,
                                     plan["local_half_width_M"])
        source_peak_result = source_peak_rule(gsl, source_axis, values[:, 4])
        source_peak = source_peak_result["time_M"]
        full_derivatives = derivatives(gsl, source_axis, values, source_match)
        local_derivatives = derivatives(gsl, source_axis[source_local],
                                        values[source_local], source_match)
        full_fit = fit(full_derivatives, meta[3:])[0]
        local_fit = fit(local_derivatives, meta[3:])[0]
        if abs(source_peak-meta[0]) > 1e-5 or np.max(abs(full_fit-coefficients[list(ACTIVE)])) > 3e-4:
            raise ValueError("Source NQC peak or full-array fit fails replay")
        report["source"] = {
            "peak_time_M": source_peak, "match_time_M": source_match,
            "peak_rule": source_peak_result,
            "peak_minus_capture_M": source_peak-float(meta[0]),
            "target": meta[3:].tolist(),
            "reported_coefficients": coefficients[list(ACTIVE)].tolist(),
            "full_array_fit": full_fit.tolist(), "local_window_fit": local_fit.tolist(),
            "local_minus_full_max_absolute": float(np.max(abs(local_fit-full_fit))),
            "local_minus_full_derivatives_max_absolute": float(np.max(abs(local_derivatives-full_derivatives))),
        }
        q.write_evidence(args.json_out, report)
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-nqc-wheel-state-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
            field_library = Path(tmp) / "field.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["field_adapter"], field_library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            mode_library = Path(tmp) / "mode.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["mode_adapter"], mode_library,
                     source_plan["flux_source_includes"],
                     source_plan["flux_public_includes"])
            high_library = Path(tmp) / "high.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["high_adapter"], high_library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            exact_library = Path(tmp) / "exact.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["exact_state_adapter"], exact_library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            nqc_library = Path(tmp) / "nqc.dylib"
            compile_wrapper(private, args.headers, args.gsl_include,
                            [str(path) for path in libraries.values()], nqc_library)
            c_fit, c_peak_offset = bind(nqc_library)

            def source_c_fit(inputs: np.ndarray, peak_time: float) -> np.ndarray:
                c_arrays = [np.ascontiguousarray(inputs[:, i], dtype=np.float64)
                            for i in range(5)]
                result = np.empty(5, dtype=np.float64)
                ptr = ctypes.POINTER(ctypes.c_double)
                rc = c_fit(*(value.ctypes.data_as(ptr) for value in c_arrays),
                           len(inputs), peak_time, float(meta[1]),
                           row["mass1"], row["mass2"],
                           (row["chi1"]-row["chi2"])/2,
                           (row["chi1"]+row["chi2"])/2,
                           result.ctypes.data_as(ptr))
                if rc or not np.isfinite(result).all():
                    raise RuntimeError(f"Pinned source C NQC fit failed: {rc}")
                return result

            offset = c_peak_offset(row["mass1"], row["mass2"],
                                   (row["chi1"]-row["chi2"])/2,
                                   (row["chi1"]+row["chi2"])/2)
            if abs(source_peak-source_match-offset) > 1e-10:
                raise ValueError("Pinned source NQC peak-to-match offset differs")
            c_source = source_c_fit(values, source_peak)
            if np.max(abs(c_source-coefficients[list(ACTIVE)])) > 1e-9:
                raise ValueError("Pinned source C NQC fit fails source-input replay")
            report["source"]["c_fit"] = c_source.tolist()
            report["source"]["c_minus_python_max_absolute"] = float(
                np.max(abs(c_source-full_fit)))
            field_function = ctypes.CDLL(str(field_library)).diffgw_v4_source_field
            field_function.argtypes = [ctypes.c_double]*8 + [ctypes.POINTER(ctypes.c_double)]
            field_function.restype = ctypes.c_int
            mode_function = ctypes.CDLL(str(mode_library)).diffgw_v4_mode22_same_state
            mode_function.argtypes = [ctypes.c_double]*9 + [ctypes.POINTER(ctypes.c_double)]
            mode_function.restype = ctypes.c_int
            exact_function = ctypes.CDLL(str(exact_library)).diffgw_v4_source_nqc_state
            exact_function.argtypes = [ctypes.c_double]*8 + [ctypes.POINTER(ctypes.c_double)]
            exact_function.restype = ctypes.c_int
            masses = [row[key] for key in ("mass1", "mass2", "chi1", "chi2")]
            fraction = row["mass1"] / (row["mass1"]+row["mass2"])
            def field_mode(state: np.ndarray, need_mode: bool) -> tuple[float, complex | None]:
                field_out = (ctypes.c_double*6)()
                if field_function(*masses, *state, field_out):
                    raise RuntimeError("Source field failed on a wheel state")
                omega = float(field_out[1])
                if not need_mode:
                    return omega, None
                mode_out = (ctypes.c_double*39)()
                if mode_function(fraction, 1-fraction, row["chi1"], row["chi2"],
                                 *state, omega, mode_out):
                    raise RuntimeError("Source raw mode failed on a wheel state")
                return omega, complex(mode_out[0], mode_out[1])
            def exact_mode(state: np.ndarray) -> tuple[float, complex]:
                output = (ctypes.c_double*3)()
                if exact_function(*masses, *state, output):
                    raise RuntimeError("Exact source NQC input calls failed")
                return float(output[0]), complex(output[1], output[2])
            high_function = ctypes.CDLL(str(high_library)).diffgw_v4_full_source_high_dynamics
            high_function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_size_t),
                                      ctypes.POINTER(ctypes.c_double)]
            high_function.restype = ctypes.c_int
            high_capacity = 50000
            high_buffer = np.empty(5*high_capacity, dtype=np.float64)
            high_count = ctypes.c_size_t()
            high_rc = high_function(*masses, row["omega_start"], plan["step_M"],
                                    high_capacity, ctypes.byref(high_count),
                                    high_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            if high_rc or high_count.value != len(values):
                raise ValueError(f"Pinned source public high dynamics differ in length: "
                                 f"{high_rc}, {high_count.value}, {len(values)}")
            source_high = high_buffer[:5*high_count.value].reshape(5, -1)
            source_high_axis = source_high[0]-source_high[0, 0]
            if np.max(abs(source_high_axis-source_axis)) > 1e-10:
                raise ValueError("Pinned source public high dynamics differ in time grid")
            source_states = source_high[1:, source_local]
            source_reconstructed = [field_mode(source_states[:, i], True)
                                    for i in range(source_states.shape[1])]
            source_reconstructed_mode = np.asarray([item[1] for item in source_reconstructed])
            source_reconstructed_inputs = np.column_stack((
                abs(source_reconstructed_mode),
                np.unwrap(np.angle(source_reconstructed_mode)),
                source_high[1, source_local], source_high[3, source_local],
                [item[0] for item in source_reconstructed]))
            source_reconstructed_fit = fit(derivatives(
                gsl, source_axis[source_local], source_reconstructed_inputs,
                source_match), meta[3:])[0]
            source_c_reconstructed = source_c_fit(
                source_reconstructed_inputs,
                source_peak-float(source_axis[source_local[0]]))
            phase_delta = np.angle(np.exp(1j*(
                source_reconstructed_inputs[:, 1]-values[source_local, 1])))
            report["source"]["public_state_reconstruction"] = {
                "max_absolute_r": float(np.max(abs(source_high[1]-values[:, 2]))),
                "max_absolute_pr_star": float(np.max(abs(source_high[3]-values[:, 3]))),
                "max_absolute_omega_local": float(np.max(abs(
                    source_reconstructed_inputs[:, 4]-values[source_local, 4]))),
                "max_relative_raw_amplitude_local": float(np.max(abs(
                    source_reconstructed_inputs[:, 0]/values[source_local, 0]-1))),
                "max_absolute_raw_phase_mod_2pi_local": float(np.max(abs(phase_delta))),
                "fit": source_reconstructed_fit.tolist(),
                "c_fit": source_c_reconstructed.tolist(),
                "fit_minus_c_max_absolute": float(np.max(abs(
                    source_reconstructed_fit-source_c_reconstructed))),
                "fit_minus_captured_l2": float(np.linalg.norm(
                    source_reconstructed_fit-coefficients[list(ACTIVE)])),
            }
            exact_source_items = [exact_mode(source_states[:, i])
                                  for i in range(source_states.shape[1])]
            exact_source_mode = np.asarray([item[1] for item in exact_source_items])
            exact_source_inputs = np.column_stack((
                abs(exact_source_mode), np.unwrap(np.angle(exact_source_mode)),
                source_high[1, source_local], source_high[3, source_local],
                [item[0] for item in exact_source_items]))
            exact_source_fit = fit(derivatives(
                gsl, source_axis[source_local], exact_source_inputs,
                source_match), meta[3:])[0]
            exact_source_c_fit = source_c_fit(
                exact_source_inputs,
                source_peak-float(source_axis[source_local[0]]))
            if (np.max(abs(exact_source_inputs[:, 4]-values[source_local, 4])) > 1e-10
                    or np.max(abs(exact_source_inputs[:, 0]/values[source_local, 0]-1)) > 1e-10
                    or np.linalg.norm(exact_source_c_fit-coefficients[list(ACTIVE)]) > 1e-3):
                raise ValueError("Exact source NQC input reconstruction fails control")
            exact_phase_delta = np.angle(np.exp(1j*(
                exact_source_inputs[:, 1]-values[source_local, 1])))
            report["source"]["exact_generator_calls_reconstruction"] = {
                "max_absolute_omega_local": float(np.max(abs(
                    exact_source_inputs[:, 4]-values[source_local, 4]))),
                "max_relative_raw_amplitude_local": float(np.max(abs(
                    exact_source_inputs[:, 0]/values[source_local, 0]-1))),
                "max_absolute_raw_phase_mod_2pi_local": float(np.max(abs(
                    exact_phase_delta))),
                "fit": exact_source_fit.tolist(),
                "c_fit": exact_source_c_fit.tolist(),
                "fit_minus_c_max_absolute": float(np.max(abs(
                    exact_source_fit-exact_source_c_fit))),
                "fit_minus_captured_l2": float(np.linalg.norm(
                    exact_source_fit-coefficients[list(ACTIVE)])),
                "c_fit_minus_captured_l2": float(np.linalg.norm(
                    exact_source_c_fit-coefficients[list(ACTIVE)])),
            }
            q.write_evidence(args.json_out, report)
            coefficient_output, reported = public_call(row, plan["step_M"], 1)
            if (list(coefficient_output) != [None, None, None]
                    or np.max(abs(reported-np.asarray(frozen["wheel_nqc_coefficients"]))) > 1e-9):
                raise ValueError("Pinned public NQC coefficients fail replay")
            output, _ = public_call(row, plan["step_M"], 0)
            high = np.asarray(output[2].data, dtype=np.float64).reshape(5, -1)
            if (q.canonical_hash(high.tolist()) != public_frozen["public_high_dynamics_sha256"]
                    or high.shape[1] != public_frozen["public_high_dynamics_length"]):
                raise ValueError("Pinned public high-rate dynamics fail replay")
            axis = high[0]-high[0, 0]
            if np.max(abs(np.diff(axis)-plan["high_step_M"])) > 1e-9:
                raise ValueError("Public high-rate step changed")
            shared = min(source_high.shape[1], high.shape[1])
            if np.max(abs(source_high_axis[:shared]-axis[:shared])) > 1e-9:
                raise ValueError("Source and wheel relative high-rate grids differ")
            report["source_wheel_public_states"] = {
                "source_high_samples": int(source_high.shape[1]),
                "wheel_high_samples": int(high.shape[1]),
                "source_start_time_M": float(source_high[0, 0]),
                "wheel_start_time_M": float(high[0, 0]),
                "source_end_time_M": float(source_high[0, -1]),
                "wheel_end_time_M": float(high[0, -1]),
                "max_absolute_state_difference_at_source_match": np.max(abs(
                    source_high[1:, source_local]-high[1:, source_local]),
                    axis=1).tolist(),
            }
            peak_indices = local_indices(axis, source_peak, plan["peak_half_width_M"])
            peak_omega = np.asarray([field_mode(high[1:, i], False)[0]
                                     for i in peak_indices])
            wheel_peak_result = source_peak_rule(gsl, axis[peak_indices], peak_omega)
            wheel_peak = wheel_peak_result["time_M"]
            wheel_match = wheel_peak-(float(meta[0])-source_match)
            indices = local_indices(axis, wheel_match, plan["local_half_width_M"])
            data = [field_mode(high[1:, i], True) for i in indices]
            wheel_omega = np.asarray([x[0] for x in data])
            raw_mode = np.asarray([x[1] for x in data])
            inputs = np.column_stack((abs(raw_mode), np.unwrap(np.angle(raw_mode)),
                                      high[1, indices], high[3, indices], wheel_omega))
            wheel_derivatives = derivatives(gsl, axis[indices], inputs, wheel_match)
            wheel_fit = fit(wheel_derivatives, meta[3:])[0]
            c_wheel = source_c_fit(inputs, wheel_peak-float(axis[indices[0]]))
            if np.max(abs(c_wheel-wheel_fit)) > 1e-4:
                raise ValueError("Pinned C and GSL wheel-state NQC fits differ")
            exact_peak_omega = np.asarray([exact_mode(high[1:, i])[0]
                                           for i in peak_indices])
            exact_peak_result = source_peak_rule(gsl, axis[peak_indices],
                                                 exact_peak_omega)
            exact_wheel_peak = exact_peak_result["time_M"]
            exact_wheel_match = exact_wheel_peak-(float(meta[0])-source_match)
            exact_indices = local_indices(axis, exact_wheel_match,
                                          plan["local_half_width_M"])
            exact_items = [exact_mode(high[1:, i]) for i in exact_indices]
            exact_raw_mode = np.asarray([item[1] for item in exact_items])
            exact_inputs = np.column_stack((
                abs(exact_raw_mode), np.unwrap(np.angle(exact_raw_mode)),
                high[1, exact_indices], high[3, exact_indices],
                [item[0] for item in exact_items]))
            exact_fit = fit(derivatives(gsl, axis[exact_indices], exact_inputs,
                                        exact_wheel_match), meta[3:])[0]
            exact_c_fit = source_c_fit(
                exact_inputs, exact_wheel_peak-float(axis[exact_indices[0]]))
            if np.max(abs(exact_fit-exact_c_fit)) > 1e-4:
                raise ValueError("Exact wheel-state C and GSL NQC fits differ")
            same_time_indices = local_indices(axis, source_match,
                                              plan["local_half_width_M"])
            same_time_data = [field_mode(high[1:, i], True) for i in same_time_indices]
            same_time_omega = np.asarray([x[0] for x in same_time_data])
            same_time_mode = np.asarray([x[1] for x in same_time_data])
            same_time_inputs = np.column_stack((
                abs(same_time_mode), np.unwrap(np.angle(same_time_mode)),
                high[1, same_time_indices], high[3, same_time_indices], same_time_omega))
            fixed_match_fit = fit(derivatives(gsl, axis[same_time_indices],
                                              same_time_inputs, source_match), meta[3:])[0]
            wheel_coefficients = np.asarray(frozen["wheel_nqc_coefficients"])[list(ACTIVE)]
            shift_scan = []
            for shift in plan["diagnostic_match_shifts_M"]:
                shifted_fit = fit(derivatives(gsl, axis[indices], inputs,
                                              wheel_match + shift), meta[3:])[0]
                shift_scan.append({
                    "shift_M": shift,
                    "coefficients": shifted_fit.tolist(),
                    "minus_reported_l2": float(np.linalg.norm(
                        shifted_fit-wheel_coefficients)),
                })
            report["wheel"] = {
                "high_dynamics_sha256": public_frozen["public_high_dynamics_sha256"],
                "peak_time_M": wheel_peak, "match_time_M": wheel_match,
                "peak_rule": wheel_peak_result,
                "peak_minus_source_M": wheel_peak-source_peak,
                "match_minus_source_M": wheel_match-source_match,
                "reported_coefficients": wheel_coefficients.tolist(),
                "source_equations_fit_on_wheel_states": wheel_fit.tolist(),
                "source_c_fit_on_wheel_states": c_wheel.tolist(),
                "c_minus_python_max_absolute": float(np.max(abs(c_wheel-wheel_fit))),
                "c_minus_reported_l2": float(np.linalg.norm(c_wheel-wheel_coefficients)),
                "source_equations_fit_at_source_match": fixed_match_fit.tolist(),
                "fit_minus_reported": (wheel_fit-wheel_coefficients).tolist(),
                "fit_minus_reported_l2": float(np.linalg.norm(wheel_fit-wheel_coefficients)),
                "diagnostic_match_shift_scan": shift_scan,
                "source_local_minus_reported_l2": float(np.linalg.norm(local_fit-wheel_coefficients)),
                "wheel_vs_source_local_fit_l2": float(np.linalg.norm(wheel_fit-local_fit)),
                "wheel_states_at_match": high[1:, int(np.argmin(abs(axis-wheel_match)))].tolist(),
                "sample_count": len(indices),
                "exact_generator_calls_reconstruction": {
                    "peak_rule": exact_peak_result,
                    "match_time_M": exact_wheel_match,
                    "source_equations_fit": exact_fit.tolist(),
                    "source_c_fit": exact_c_fit.tolist(),
                    "fit_minus_c_max_absolute": float(np.max(abs(exact_fit-exact_c_fit))),
                    "fit_minus_reported_l2": float(np.linalg.norm(
                        exact_fit-wheel_coefficients)),
                    "c_fit_minus_reported_l2": float(np.linalg.norm(
                        exact_c_fit-wheel_coefficients)),
                },
            }
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "diagnostic_failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        q.write_evidence(args.json_out, report)
        raise
    q.write_evidence(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

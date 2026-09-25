"""Dev-only q4 NQC fit after an evidence-derived source match-time shift."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np
from scipy.interpolate import CubicSpline
import torch

from . import qualify_generator as q
from .diagnose_native_source_nqc_inputs import _fit, _native_samples
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_high_rate import LocalPolynomial
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture
from .diagnose_source_q4_ultrafine_chart import _native


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/q4_native_source_nqc_match_shift_fit_plan.json"


def _fit_values(gsl, time, values, match, targets):
    return _fit(gsl, time, *values.T, match, targets)


def _raw_mode_profile(time, native, source, match, shift, half_width):
    near = abs(time-match) <= half_width
    if near.sum() < 100:
        raise ValueError("Insufficient near-match raw-mode samples")
    amplitude = CubicSpline(time, source[:, 0])
    phase = CubicSpline(time, source[:, 1])
    native_phase = CubicSpline(time, native[:, 1])
    shifted_time = time[near]+shift
    phase_offset = float(phase(match+shift)-native_phase(match))
    amp_residual = native[near, 0]-amplitude(shifted_time)
    phase_residual = native[near, 1]-phase(shifted_time)+phase_offset
    return {
        "samples": int(near.sum()),
        "amplitude_rms": float(np.sqrt(np.mean(amp_residual**2))),
        "amplitude_max_abs": float(np.max(abs(amp_residual))),
        "phase_rms_after_constant_alignment": float(np.sqrt(np.mean(phase_residual**2))),
        "phase_max_abs_after_constant_alignment": float(np.max(abs(phase_residual))),
    }


def _fit_gap_attribution(native, shifted):
    result = {}
    for mode in ("amplitude", "phase"):
        native_matrix = np.asarray(native[f"{mode}_matrix"])
        shifted_matrix = np.asarray(shifted[f"{mode}_matrix"])
        native_rhs = np.asarray(native[f"{mode}_rhs"])
        shifted_rhs = np.asarray(shifted[f"{mode}_rhs"])
        native_solution = np.linalg.solve(native_matrix, native_rhs)
        rhs_effect = np.linalg.solve(shifted_matrix, shifted_rhs-native_rhs)
        matrix_effect = np.linalg.solve(shifted_matrix, native_rhs)-native_solution
        total = np.linalg.solve(shifted_matrix, shifted_rhs)-native_solution
        if np.max(abs(total-rhs_effect-matrix_effect)) > 1e-10:
            raise ValueError(f"{mode} crossed-system decomposition failed")
        result[mode] = {
            "gap_l2": float(np.linalg.norm(total)),
            "rhs_first_effect_l2": float(np.linalg.norm(rhs_effect)),
            "matrix_second_effect_l2": float(np.linalg.norm(matrix_effect)),
            "matrix_max_abs_difference": float(np.max(abs(shifted_matrix-native_matrix))),
            "rhs_max_abs_difference": float(np.max(abs(shifted_rhs-native_rhs))),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    keys = ("prior_alignment", "prior_fit", "source_inputs", "capture_adapter",
            "source_flux_plan", "settings")
    paths = {key: q.ROOT / plan[key] for key in keys}
    if (plan["schema"] != "diffgw.seobnrv4.q4-native-source-nqc-match-shift-fit-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row_name"] != "q4_nonspinning"
            or plan["output_step_M"] != 0.015625
            or plan["capacity"] != 50000
            or plan["common_grid_resampling"] != "scipy_cubic_spline"
            or plan["local_polynomial_degree"] != 4
            or plan["local_polynomial_half_width_M"] != 2.0
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in paths.items())):
        parser.error("Frozen plan or prerequisites changed")
    prior = json.loads(paths["prior_alignment"].read_text())
    fit_prior = json.loads(paths["prior_fit"].read_text())
    source_inputs = json.loads(paths["source_inputs"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    key = str(plan["output_step_M"])
    if (any(item["status"] != "diagnostic_complete"
            for item in (prior, fit_prior, source_inputs))
            or not (prior["row"] == fit_prior["row"] == source_inputs["row"])):
        parser.error("Pinned NQC evidence changed")
    fields = prior["comparison"]["fields"]
    orbital_shift = float(np.median([fields[name]["best_source_time_shift_M"]
                                     for name in ("radius", "radial_momentum",
                                                  "orbital_frequency")]))
    peaks = prior["comparison"]["orbital_peaks"]
    peak_shift = float(peaks["source_high_grid"]["spline_time_relative_to_match_M"]
                       - peaks["native_high_grid"]["spline_time_relative_to_match_M"])
    if not (0 < orbital_shift < 0.05 and 0 < peak_shift < 0.05):
        parser.error("Frozen alignment shifts changed sign or range")
    report = {
        "schema": "diffgw.seobnrv4.q4-native-source-nqc-match-shift-fit.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in paths.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": prior["row"], "output_step_M": plan["output_step_M"],
        "comparison": None, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        torch.set_num_threads(1)
        libraries = _libraries(source_plan["pinned_wheel"])
        gsl = GSLCubic(libraries["libgsl.28.dylib"])
        polynomial = LocalPolynomial(plan["local_polynomial_half_width_M"],
                                     plan["local_polynomial_degree"])
        with tempfile.TemporaryDirectory(prefix="diffgw-q4-nqc-shift-fit-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("Pinned-source NQC instrumentation anchor changed")
            source_file.write_text(original.replace(INCLUDE, INCLUDE + CAPTURE))
            library = Path(tmp) / "source_nqc_inputs.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["capture_adapter"], library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            function = ctypes.CDLL(str(library)).diffgw_v4_full_source_nqc_inputs
            function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            meta, source_values, source_coefficients = _capture(
                function, prior["row"], plan["output_step_M"], plan["capacity"])
            expected = source_inputs["grids"][key]
            if (hashlib.sha256(source_values.tobytes()).hexdigest() != expected["input_sha256"]
                    or abs(meta[1]-expected["internal_step_M"]) > 1e-12
                    or abs(meta[2]-expected["match_time_M"]) > 1e-10
                    or np.max(abs(source_coefficients[[0, 1, 2, 6, 7]]
                                  - np.asarray(expected["source_active"]))) > 1e-9):
                raise ValueError("Pinned-source fine NQC inputs changed")
        run = _native(prior["row"], plan["output_step_M"],
                      {"atol": plan["atol"], "rtol": plan["rtol"]},
                      q.Settings(**json.loads(paths["settings"].read_text())["settings"]),
                      return_nqc_coefficients=True, return_nqc_samples=True)
        native_time, native_columns, native_match = _native_samples(run["nqc_samples"])
        prior_comp = fit_prior["comparison"]
        if (q.summarize_run(run)["array_sha256"] != prior_comp["native_mode_sha256"]
                or hashlib.sha256(native_columns.tobytes()).hexdigest()
                != prior_comp["native_nqc_sample_sha256"]):
            raise ValueError("Native fine replay changed")
        source_time = np.arange(len(source_values), dtype=np.float64)*meta[1]
        source_targets = np.asarray(expected["targets"])
        native_fit = _fit_values(gsl, native_time, native_columns.T,
                                 native_match, source_targets)
        source_fit = _fit_values(gsl, source_time, source_values,
                                 meta[2], source_targets)
        native_local_fit = _fit_values(polynomial, native_time, native_columns.T,
                                       native_match, source_targets)
        source_local_fit = _fit_values(polynomial, source_time, source_values,
                                       meta[2], source_targets)
        prior_native = np.asarray(prior_comp["native_gsl_with_source_targets"]["coefficients"])
        prior_source = np.asarray(prior_comp["source_gsl_with_source_targets"]["coefficients"])
        native_coeff = np.asarray(native_fit["coefficients"])
        source_coeff = np.asarray(source_fit["coefficients"])
        if (np.linalg.norm(native_coeff-prior_native) > 1e-8
                or np.linalg.norm(source_coeff-prior_source) > 1e-8):
            raise ValueError("Unshifted source/native GSL fits do not replay prior controls")
        translation = native_match-meta[2]
        native_indices = np.rint((source_time+translation-native_time[0])/meta[1]).astype(int)
        valid = ((native_indices >= 0) & (native_indices < len(native_time))
                 & (source_time+max(orbital_shift, peak_shift) <= source_time[-1]))
        source_axis = source_time[valid]
        native_indices = native_indices[valid]
        if len(source_axis) < 1000 or not np.all(np.diff(native_indices) == 1):
            raise ValueError("Insufficient common source/native samples")
        if np.max(abs(native_time[native_indices]-source_axis-translation)) > 1e-8:
            raise ValueError("Source/native grid alignment changed")
        common_native = native_columns.T[native_indices]
        common_source = source_values[valid]
        common_native_fit = _fit_values(gsl, source_axis, common_native,
                                         meta[2], source_targets)
        common_source_fit = _fit_values(gsl, source_axis, common_source,
                                         meta[2], source_targets)
        if max(np.linalg.norm(np.asarray(common_native_fit["coefficients"])-native_coeff),
               np.linalg.norm(np.asarray(common_source_fit["coefficients"])-source_coeff)) > 1e-5:
            raise ValueError("Common-grid unshifted GSL controls changed")
        source_interpolant = CubicSpline(source_time, source_values, axis=0,
                                         extrapolate=False)
        baseline_gap = float(np.linalg.norm(source_coeff-native_coeff))
        native_local_coeff = np.asarray(native_local_fit["coefficients"])
        source_local_coeff = np.asarray(source_local_fit["coefficients"])
        shifts = {}
        for name, shift in (("orbital_field_median", orbital_shift),
                            ("high_grid_peak_difference", peak_shift)):
            shifted_fit = _fit_values(gsl, source_time, source_values,
                                      meta[2]+shift, source_targets)
            shifted_local_fit = _fit_values(polynomial, source_time, source_values,
                                            meta[2]+shift, source_targets)
            shifted_coeff = np.asarray(shifted_fit["coefficients"])
            shifted_local_coeff = np.asarray(shifted_local_fit["coefficients"])
            shifted_common = source_interpolant(source_axis+shift)
            shifted_common_fit = _fit_values(gsl, source_axis, shifted_common,
                                              meta[2], source_targets)
            shifted_common_coeff = np.asarray(shifted_common_fit["coefficients"])
            shifts[name] = {
                "shift_M": shift,
                "gsl_fit": shifted_fit,
                "coefficient_gap_to_native_l2": float(np.linalg.norm(
                    shifted_coeff-native_coeff)),
                "coefficient_gap_ratio_to_unshifted": float(np.linalg.norm(
                    shifted_coeff-native_coeff))/baseline_gap,
                "gsl_gap_attribution": _fit_gap_attribution(native_fit, shifted_fit),
                "local_polynomial_fit": shifted_local_fit,
                "local_polynomial_gap_to_native_l2": float(np.linalg.norm(
                    shifted_local_coeff-native_local_coeff)),
                "local_polynomial_gap_attribution": _fit_gap_attribution(
                    native_local_fit, shifted_local_fit),
                "common_grid_resampled_source_fit": shifted_common_fit,
                "common_grid_resampled_gap_to_native_l2": float(np.linalg.norm(
                    shifted_common_coeff-np.asarray(common_native_fit["coefficients"]))),
                "common_grid_resampled_gap_attribution": _fit_gap_attribution(
                    common_native_fit, shifted_common_fit),
                "raw_mode_profile": _raw_mode_profile(
                    source_axis, common_native, common_source, meta[2], shift,
                    plan["near_match_half_width_M"]),
            }
        report["comparison"] = {
            "elapsed_seconds": time.monotonic()-started,
            "source_input_sha256": hashlib.sha256(source_values.tobytes()).hexdigest(),
            "native_input_sha256": hashlib.sha256(native_columns.tobytes()).hexdigest(),
            "unshifted_gap_l2": baseline_gap,
            "native_gsl_fit": native_fit,
            "source_gsl_fit": source_fit,
            "native_local_polynomial_fit": native_local_fit,
            "source_local_polynomial_fit": source_local_fit,
            "common_grid_samples": len(source_axis),
            "common_grid_native_fit": common_native_fit,
            "common_grid_source_fit": common_source_fit,
            "unshifted_local_polynomial_gap_l2": float(np.linalg.norm(
                source_local_coeff-native_local_coeff)),
            "unshifted_raw_mode_profile": _raw_mode_profile(
                source_axis, common_native, common_source, meta[2], 0.0,
                plan["near_match_half_width_M"]),
            "shifts": shifts,
        }
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

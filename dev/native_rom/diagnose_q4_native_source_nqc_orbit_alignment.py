"""Dev-only equal-radius and local-time alignment of q4 NQC orbital inputs."""

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
from scipy.optimize import minimize_scalar
import torch

from . import qualify_generator as q
from .diagnose_native_source_nqc_inputs import _native_samples
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture
from .diagnose_source_q4_ultrafine_chart import _native


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/q4_native_source_nqc_orbit_alignment_plan.json"
FIELDS = {"radius": 2, "radial_momentum": 3, "orbital_frequency": 4}


def _errors(native: np.ndarray, source: np.ndarray) -> dict:
    difference = native-source
    return {
        "rms": float(np.sqrt(np.mean(difference**2))),
        "max_abs": float(np.max(abs(difference))),
        "mean": float(np.mean(difference)),
    }


def _local_peak(time: np.ndarray, frequency: np.ndarray, match: float,
                reported_peak_relative: float) -> dict:
    index = int(np.argmax(frequency))
    if index < 2 or index + 2 >= len(time):
        raise ValueError("High-grid orbital-frequency peak is not interior")
    spline = CubicSpline(time, frequency, bc_type="natural")
    optimum = minimize_scalar(lambda coordinate: -float(spline(coordinate)),
                              bounds=(float(time[index-2]), float(time[index+2])),
                              method="bounded", options={"xatol": 1e-12})
    if not optimum.success or float(spline(optimum.x, 2)) >= 0:
        raise ValueError("High-grid orbital-frequency peak fit failed")
    return {
        "sample_index": index,
        "sample_time_relative_to_match_M": float(time[index]-match),
        "spline_time_relative_to_match_M": float(optimum.x-match),
        "spline_minus_reported_time_M": float(optimum.x-match-reported_peak_relative),
        "frequency": float(spline(optimum.x)),
        "spline_derivative_at_reported_peak": float(
            spline(match+reported_peak_relative, 1)),
        "spline_curvature_at_reported_peak": float(
            spline(match+reported_peak_relative, 2)),
    }


def _compare(time: np.ndarray, native: np.ndarray, source: np.ndarray,
             match: float, near_width: float, reference_width: float,
             max_shift: float) -> dict:
    near = abs(time-match) <= near_width
    reference = abs(time-match) <= reference_width
    if near.sum() < 100 or reference.sum() < 200:
        raise ValueError("Insufficient samples for the frozen near-match windows")
    near_time = time[near]
    result = {"near_samples": int(near.sum()), "reference_samples": int(reference.sum()),
              "fields": {}}
    for name, index in FIELDS.items():
        baseline = _errors(native[near, index], source[near, index])
        spline = CubicSpline(time, source[:, index])

        def objective(shift: float) -> float:
            return float(np.mean((native[near, index]-spline(near_time+shift))**2))

        optimum = minimize_scalar(objective, bounds=(-max_shift, max_shift),
                                  method="bounded", options={"xatol": 1e-12})
        if not optimum.success:
            raise ValueError(f"Local time-shift fit failed for {name}")
        shifted = _errors(native[near, index], spline(near_time+optimum.x))
        result["fields"][name] = {
            "match_relative_time": baseline,
            "best_source_time_shift_M": float(optimum.x),
            "time_shifted": shifted,
            "time_shift_rms_ratio": shifted["rms"]/baseline["rms"],
            "shift_at_bound": bool(abs(optimum.x) > 0.99*max_shift),
        }
    source_radius = source[reference, FIELDS["radius"]]
    if not np.all(np.diff(source_radius) < 0):
        raise ValueError("Source radius is not strictly decreasing in reference window")
    native_radius = native[near, FIELDS["radius"]]
    if native_radius.min() < source_radius.min() or native_radius.max() > source_radius.max():
        raise ValueError("Native near-match radius leaves source reference range")
    result["equal_radius"] = {}
    for name in ("radial_momentum", "orbital_frequency"):
        source_field = source[reference, FIELDS[name]]
        curve = CubicSpline(source_radius[::-1], source_field[::-1])
        if np.max(abs(curve(source_radius)-source_field)) > 1e-12:
            raise ValueError(f"Equal-radius source interpolation failed for {name}")
        baseline = result["fields"][name]["match_relative_time"]
        equal_radius = _errors(native[near, FIELDS[name]], curve(native_radius))
        result["equal_radius"][name] = {
            "error": equal_radius,
            "rms_ratio_to_equal_time": equal_radius["rms"]/baseline["rms"],
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
    keys = ("prior_field_swap", "prior_fit", "source_inputs", "capture_adapter",
            "source_flux_plan", "settings")
    paths = {key: q.ROOT / plan[key] for key in keys}
    if (plan["schema"] != "diffgw.seobnrv4.q4-native-source-nqc-orbit-alignment-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row_name"] != "q4_nonspinning"
            or plan["output_step_M"] != 0.015625
            or plan["capacity"] != 50000
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in paths.items())):
        parser.error("Frozen plan or prerequisites changed")
    prior = json.loads(paths["prior_field_swap"].read_text())
    prior_fit = json.loads(paths["prior_fit"].read_text())
    source_inputs = json.loads(paths["source_inputs"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    key = str(plan["output_step_M"])
    if (prior["status"] != "diagnostic_complete"
            or prior_fit["status"] != "diagnostic_complete"
            or source_inputs["status"] != "diagnostic_complete"
            or not (prior["row"] == source_inputs["row"] == prior_fit["row"])):
        parser.error("Pinned NQC input or prior field swap changed")
    report = {
        "schema": "diffgw.seobnrv4.q4-native-source-nqc-orbit-alignment.v1",
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
        with tempfile.TemporaryDirectory(prefix="diffgw-q4-nqc-orbit-alignment-") as tmp:
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
        if (q.summarize_run(run)["array_sha256"]
                != prior_fit["comparison"]["native_mode_sha256"]):
            raise ValueError("Native fine mode replay changed")
        if hashlib.sha256(native_columns.tobytes()).hexdigest() != prior["comparison"]["native_input_sha256"]:
            raise ValueError("Native fine NQC input replay changed")
        source_time = np.arange(len(source_values), dtype=np.float64)*meta[1]
        translation = native_match-meta[2]
        native_indices = np.rint((source_time+translation-native_time[0])/meta[1]).astype(int)
        valid = (native_indices >= 0) & (native_indices < len(native_time))
        source_axis = source_time[valid]
        native_indices = native_indices[valid]
        if (len(source_axis) != prior["comparison"]["common_samples"]
                or np.max(abs(native_time[native_indices]-source_axis-translation)) > 1e-8
                or not np.all(np.diff(native_indices) == 1)):
            raise ValueError("Source/native NQC grids do not replay prior alignment")
        common_native = native_columns.T[native_indices]
        common_source = source_values[valid]
        comparison = _compare(
            source_axis, common_native, common_source, meta[2],
            plan["near_match_half_width_M"],
            plan["radius_reference_half_width_M"],
            plan["max_local_time_shift_M"])
        source_reported_relative = expected["orbital_peak_M"]-expected["match_time_M"]
        native_reported_relative = run["peak_time_M"]-run["match_time_M"]
        source_peak = _local_peak(source_axis, common_source[:, 4], meta[2],
                                  source_reported_relative)
        native_peak = _local_peak(source_axis, common_native[:, 4], meta[2],
                                  native_reported_relative)
        comparison["orbital_peaks"] = {
            "source_high_grid": source_peak,
            "native_high_grid": native_peak,
            "native_minus_source_high_grid_M": (
                native_peak["spline_time_relative_to_match_M"]
                - source_peak["spline_time_relative_to_match_M"]),
            "source_reported_relative_to_match_M": source_reported_relative,
            "native_reported_relative_to_match_M": native_reported_relative,
        }
        comparison.update({
            "elapsed_seconds": time.monotonic()-started,
            "source_input_sha256": hashlib.sha256(source_values.tobytes()).hexdigest(),
            "native_input_sha256": hashlib.sha256(native_columns.tobytes()).hexdigest(),
            "match_time_translation_M": float(translation),
        })
        report["comparison"] = comparison
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

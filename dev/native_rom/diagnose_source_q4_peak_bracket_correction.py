"""Dev-only private-source orbital-peak bracket correction on q4 fine grids."""

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

import lal
import numpy as np

from . import qualify_generator as q
from .diagnose_native_source_nqc_inputs import _fit
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_peak_anchor import fitted_metrics
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture
from .diagnose_source_q4_nqc_cadence_ladder import _local_peak
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/source_q4_peak_bracket_correction_plan.json"
ANCHOR = """  do
    {
      timePeak = (time1 + time2) / 2.;"""
CORRECTION = """  /* Development-only: bracket the stationary point before bisection. */
  while (omegaDeriv1 <= 0. && peakIdx > 1)
    {
      time2 = time1;
      time1 = dynamicsHi->data[--peakIdx];
      omegaDeriv1 = gsl_spline_eval_deriv (spline, time1, acc);
    }
  if (omegaDeriv1 <= 0. || gsl_spline_eval_deriv (spline, time2, acc) >= 0.)
    XLAL_ERROR (XLAL_EFUNC);

"""


def _function(library: Path, name: str):
    function = getattr(ctypes.CDLL(str(library)), name)
    function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t)] + [ctypes.POINTER(ctypes.c_double)]*3
    function.restype = ctypes.c_int
    return function


def _historical_peak_bracket(gsl: GSLCubic, time: np.ndarray,
                             omega: np.ndarray) -> dict:
    first = int(np.flatnonzero(omega[1:] <= omega[:-1])[0]+1)
    left = float(time[first])
    left_derivative = gsl.derivatives(time, omega, left)[1]
    if left_derivative > 0:
        right = float(time[first+1])
    else:
        right = left
        left = float(time[first-1])
        left_derivative = gsl.derivatives(time, omega, left)[1]
    right_derivative = gsl.derivatives(time, omega, right)[1]
    original = {
        "first_decline_index": first,
        "left_time_M": left, "right_time_M": right,
        "left_derivative": left_derivative,
        "right_derivative": right_derivative,
        "opposite_derivative_signs": bool(left_derivative*right_derivative < 0),
    }
    while True:
        middle = (left+right)/2
        derivative = gsl.derivatives(time, omega, middle)[1]
        if derivative*left_derivative < 0:
            right = middle
        else:
            left, left_derivative = middle, derivative
        if right-left <= 1e-5:
            break
    original["replayed_peak_M"] = middle
    return original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    keys = ("source_nqc_ladder", "whole_ladder", "source_flux_plan",
            "capture_adapter", "mode_adapter")
    paths = {key: q.ROOT / plan[key] for key in keys}
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-peak-bracket-correction-plan.v1"
            or plan["release_qualified"] is not False
            or plan["steps_M"] != [0.015625, 0.0078125, 0.00390625]
            or plan["capture_capacity"] != 50000
            or plan["mode_capacity"] != 200000
            or plan["peak_fit"] != "quadratic_power"
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in paths.items())):
        parser.error("Frozen plan or inputs changed")
    nqc = json.loads(paths["source_nqc_ladder"].read_text())
    whole = json.loads(paths["whole_ladder"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (nqc["status"] != "diagnostic_complete"
            or whole["status"] != "diagnostic_complete"
            or nqc["row"] != whole["row"]):
        parser.error("Pinned source q4 evidence changed")
    row = nqc["row"]
    mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
    report = {
        "schema": "diffgw.seobnrv4.source-q4-peak-bracket-correction.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in paths.items()},
        "correction_sha256": hashlib.sha256(CORRECTION.encode()).hexdigest(),
        "capture_instrumentation_sha256": hashlib.sha256(CAPTURE.encode()).hexdigest(),
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": row, "grids": {}, "adjacent_corrected": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        libraries = _libraries(source_plan["pinned_wheel"])
        gsl = GSLCubic(libraries["libgsl.28.dylib"])
        previous = None
        with tempfile.TemporaryDirectory(prefix="diffgw-source-q4-peak-bracket-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(ANCHOR) != 1 or original.count(INCLUDE) != 1:
                raise ValueError("Pinned source peak or NQC anchor changed")
            source_file.write_text(original.replace(ANCHOR, CORRECTION+ANCHOR))
            mode_library = Path(tmp) / "corrected_mode.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["mode_adapter"], mode_library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            mode_function = _function(mode_library, "diffgw_v4_full_source_mode")
            source_file.write_text(source_file.read_text().replace(INCLUDE, INCLUDE+CAPTURE))
            capture_library = Path(tmp) / "corrected_capture.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["capture_adapter"], capture_library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            capture_function = _function(
                capture_library, "diffgw_v4_full_source_nqc_inputs")
            for step in plan["steps_M"]:
                meta, values, coefficients = _capture(
                    capture_function, row, step, plan["capture_capacity"])
                expected_nqc = nqc["grids"][str(step)]
                if (hashlib.sha256(values.tobytes()).hexdigest()
                        != expected_nqc["input_sha256"]
                        or abs(meta[1]-expected_nqc["internal_step_M"]) > 1e-12
                        or not np.array_equal(meta[3:], np.asarray(expected_nqc["targets"]))):
                    raise ValueError(f"Private peak patch changed NQC inputs at {step:g} M")
                time_axis = np.arange(len(values), dtype=np.float64)*meta[1]
                peak = _local_peak(time_axis, values[:, 4], meta[0])
                bracket = _historical_peak_bracket(gsl, time_axis, values[:, 4])
                if abs(bracket["replayed_peak_M"]-expected_nqc["orbital_peak_M"]) > 1e-12:
                    raise ValueError(f"Historical peak replay changed at {step:g} M")
                fit = _fit(gsl, time_axis, *values.T, meta[2], meta[3:])
                active = coefficients[[0, 1, 2, 6, 7]]
                if np.linalg.norm(np.asarray(fit["coefficients"])-active) > 1e-6:
                    raise ValueError(f"Corrected source NQC fit replay failed at {step:g} M")
                source_epoch, source_dt, source_mode = _source(
                    mode_function, row, step, plan["mode_capacity"])
                wheel_epoch, wheel_dt, wheel_mode, _ = _public(row, step)
                expected_whole = whole["grids"][str(step)]
                if (hashlib.sha256(wheel_mode.tobytes()).hexdigest()
                        != expected_whole["public_mode_sha256"]
                        or abs(source_dt/mass_seconds-step) > 1e-10
                        or abs(wheel_dt/mass_seconds-step) > 1e-10):
                    raise ValueError(f"Pinned public mode or cadence changed at {step:g} M")
                source_time = source_epoch+np.arange(len(source_mode))*source_dt
                wheel_time = wheel_epoch+np.arange(len(wheel_mode))*wheel_dt
                comparison = fitted_metrics(source_time, source_mode,
                                            wheel_time, wheel_mode,
                                            mass_seconds, plan["peak_fit"])
                report["grids"][str(step)] = {
                    "input_sha256": expected_nqc["input_sha256"],
                    "historical_peak_bracket": bracket,
                    "baseline_orbital_peak_M": expected_nqc["orbital_peak_M"],
                    "corrected_orbital_peak_M": float(meta[0]),
                    "corrected_minus_baseline_peak_M": float(meta[0]-expected_nqc["orbital_peak_M"]),
                    "corrected_peak_fit": peak,
                    "baseline_source_active": expected_nqc["source_active"],
                    "corrected_source_active": active.tolist(),
                    "wheel_active": expected_whole["public_active_nqc"],
                    "corrected_nqc_fit": fit,
                    "baseline_to_wheel_coefficient_gap_l2": expected_whole["nqc_coefficient_gap_l2"],
                    "corrected_to_wheel_coefficient_gap_l2": float(np.linalg.norm(
                        active-np.asarray(expected_whole["public_active_nqc"]))),
                    "baseline_source_mode_sha256": expected_whole["source_mode_sha256"],
                    "corrected_source_mode_sha256": hashlib.sha256(source_mode.tobytes()).hexdigest(),
                    "baseline_source_wheel": expected_whole["source_vs_public"],
                    "corrected_source_wheel": comparison,
                }
                if previous is not None:
                    old_step, old_time, old_mode, old_active = previous
                    report["adjacent_corrected"][f"{old_step:g}_to_{step:g}"] = fitted_metrics(
                        old_time, old_mode, source_time, source_mode,
                        mass_seconds, plan["peak_fit"])
                    report["adjacent_corrected"][f"{old_step:g}_to_{step:g}"][
                        "nqc_coefficient_difference_l2"] = float(np.linalg.norm(
                            active-old_active))
                previous = (step, source_time, source_mode, active)
                q.write_evidence(args.json_out, report)
                print(f"step={step:g} peak_shift={meta[0]-expected_nqc['orbital_peak_M']:.9g} "
                      f"whole={comparison['metrics']['complex_relative_l2']:.9g}", flush=True)
        report["elapsed_seconds"] = time.monotonic()-started
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

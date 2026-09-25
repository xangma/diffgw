"""Dev-only pinned-source q4 NQC fit at finer output cadences."""

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

from . import qualify_generator as q
from .diagnose_native_source_nqc_inputs import _fit
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture
from .diagnose_source_nqc_high_rate import LocalPolynomial


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/source_q4_nqc_cadence_ladder_plan.json"


def _local_peak(time, frequency, reported_peak):
    index = int(np.argmax(frequency))
    if index < 2 or index+2 >= len(time):
        raise ValueError("High-grid orbital-frequency peak is not interior")
    spline = CubicSpline(time, frequency, bc_type="natural")
    optimum = minimize_scalar(lambda value: -float(spline(value)),
                              bounds=(float(time[index-2]), float(time[index+2])),
                              method="bounded", options={"xatol": 1e-12})
    if not optimum.success or float(spline(optimum.x, 2)) >= 0:
        raise ValueError("High-grid orbital-frequency peak fit failed")
    return {
        "sample_index": index,
        "spline_minus_reported_time_M": float(optimum.x-reported_peak),
        "spline_slope_at_reported_peak": float(spline(reported_peak, 1)),
        "spline_frequency": float(spline(optimum.x)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    keys = ("prior_inputs", "capture_adapter", "source_flux_plan")
    paths = {key: q.ROOT / plan[key] for key in keys}
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-nqc-cadence-ladder-plan.v1"
            or plan["release_qualified"] is not False
            or plan["steps_M"] != [0.015625, 0.0078125, 0.00390625]
            or plan["local_polynomial_degree"] != 4
            or plan["local_polynomial_half_width_M"] != 2.0
            or plan["capacity"] != 50000
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in paths.items())):
        parser.error("Frozen plan or prerequisites changed")
    prior = json.loads(paths["prior_inputs"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or prior["row"] != plan["row"]):
        parser.error("Pinned q4 source capture changed")
    report = {
        "schema": "diffgw.seobnrv4.source-q4-nqc-cadence-ladder.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in paths.items()},
        "instrumentation_sha256": hashlib.sha256(CAPTURE.encode()).hexdigest(),
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": plan["row"], "grids": {}, "adjacent": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        libraries = _libraries(source_plan["pinned_wheel"])
        gsl = GSLCubic(libraries["libgsl.28.dylib"])
        polynomial = LocalPolynomial(plan["local_polynomial_half_width_M"],
                                     plan["local_polynomial_degree"])
        with tempfile.TemporaryDirectory(prefix="diffgw-source-q4-nqc-ladder-") as tmp:
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
            for step in plan["steps_M"]:
                meta, values, coefficients = _capture(
                    function, plan["row"], step, plan["capacity"])
                active = coefficients[[0, 1, 2, 6, 7]]
                time_axis = np.arange(len(values), dtype=np.float64)*meta[1]
                fit = _fit(gsl, time_axis, *values.T, meta[2], meta[3:])
                local_fit = _fit(polynomial, time_axis, *values.T, meta[2], meta[3:])
                fit_error = float(np.linalg.norm(
                    np.asarray(fit["coefficients"])-active))
                if fit_error > 1e-6:
                    raise ValueError(f"Source GSL NQC replay failed at step {step}")
                if step == plan["steps_M"][0]:
                    expected = prior["grids"][str(step)]
                    if (hashlib.sha256(values.tobytes()).hexdigest()
                            != expected["input_sha256"]
                            or abs(meta[1]-expected["internal_step_M"]) > 1e-12
                            or abs(meta[2]-expected["match_time_M"]) > 1e-10
                            or np.max(abs(active-np.asarray(expected["source_active"]))) > 1e-9):
                        raise ValueError("Existing q4 source NQC baseline changed")
                report["grids"][str(step)] = {
                    "samples": len(values),
                    "input_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                    "internal_step_M": float(meta[1]),
                    "orbital_peak_M": float(meta[0]),
                    "match_time_M": float(meta[2]),
                    "orbital_peak_relative_to_match_M": float(meta[0]-meta[2]),
                    "high_grid_peak": _local_peak(
                        time_axis, values[:, 4], meta[0]),
                    "targets": meta[3:].tolist(),
                    "source_active": active.tolist(),
                    "replayed_fit": fit,
                    "replay_error_l2": fit_error,
                    "local_polynomial_fit": local_fit,
                }
                q.write_evidence(args.json_out, report)
                print(f"step={step:g} samples={len(values)} "
                      f"internal_step={meta[1]:.9g} fit_error={fit_error:.3g}", flush=True)
        for coarse, fine in zip(plan["steps_M"], plan["steps_M"][1:]):
            a = report["grids"][str(coarse)]
            b = report["grids"][str(fine)]
            report["adjacent"][f"{coarse:g}_to_{fine:g}"] = {
                "coefficient_difference_l2": float(np.linalg.norm(
                    np.asarray(b["source_active"])-np.asarray(a["source_active"]))),
                "internal_step_ratio": a["internal_step_M"]/b["internal_step_M"],
                "local_polynomial_coefficient_difference_l2": float(np.linalg.norm(
                    np.asarray(b["local_polynomial_fit"]["coefficients"])
                    - np.asarray(a["local_polynomial_fit"]["coefficients"]))),
                "peak_relative_to_match_difference_M": (
                    b["orbital_peak_relative_to_match_M"]
                    - a["orbital_peak_relative_to_match_M"]),
                "target_max_abs_difference": float(np.max(abs(
                    np.asarray(b["targets"])-np.asarray(a["targets"])))),
            }
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

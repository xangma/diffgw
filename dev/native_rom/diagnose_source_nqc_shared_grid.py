"""Compare pinned-source NQC inputs on common timestamps of two high-rate grids.

The high-grid decimation is a counterfactual fit, not a waveform generator.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile

import numpy as np

from . import qualify_generator as q
from .diagnose_native_source_nqc_inputs import _fit
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_high_rate import RESAMPLE_ANCHOR
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_shared_grid_plan.json"
NAMES = ("amplitude", "phase", "radius", "radial_momentum", "orbital_frequency")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    names = ("source_high_rate", "source_wheel_mode", "source_flux_plan", "adapter")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-shared-grid-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["output_step_M"] != 0.125
            or plan["resample_factors"] != [16, 32]
            or plan["capacity"] != 50000
            or plan["near_match_half_width_M"] != 2.0
            or plan["release_qualified"] is not False
            or any(q.sha256(paths[name]) != plan[name + "_sha256"] for name in names)):
        parser.error("Frozen plan or prerequisites changed")
    prior = json.loads(paths["source_high_rate"].read_text())
    wheel_mode = json.loads(paths["source_wheel_mode"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or wheel_mode["status"] != "diagnostic_complete"):
        parser.error("Frozen high-rate result is unavailable")
    row = next(item["row"] for item in wheel_mode["rows"]
               if item["name"] == plan["row_name"])
    libraries = _libraries(source_plan["pinned_wheel"])
    gsl = GSLCubic(libraries["libgsl.28.dylib"])
    report = dict(schema="diffgw.seobnrv4.source-nqc-shared-grid.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={name: q.sha256(path) for name, path in paths.items()},
                  instrumentation_sha256=hashlib.sha256(CAPTURE.encode()).hexdigest(),
                  pinned_libraries_sha256={name: q.sha256(path)
                                           for name, path in libraries.items()},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), rows=[], comparisons={},
                  errors=[])
    q.write_evidence(args.json_out, report)
    try:
        baseline = {item["resample_factor"]: item for item in prior["rows"]}
        captures = {}
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-nqc-shared-grid-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1 or original.count(RESAMPLE_ANCHOR) != 1:
                raise ValueError("Pinned source diagnostic anchors changed")
            for factor in plan["resample_factors"]:
                item = baseline[factor]
                instrumented = original.replace(INCLUDE, INCLUDE + CAPTURE)
                instrumented = instrumented.replace(
                    RESAMPLE_ANCHOR,
                    f"\n\n  resampFac = {factor}u; /* diagnostic high-rate grid */"
                    + RESAMPLE_ANCHOR)
                source_file.write_text(instrumented)
                if q.sha256(source_file) != item["instrumented_source_sha256"]:
                    raise ValueError("Pinned source instrumentation differs from prior capture")
                dylib = Path(tmp) / f"source_nqc_shared_grid_{factor}.dylib"
                _compile(private, args.headers, args.gsl_include, libraries,
                         paths["adapter"], dylib,
                         source_plan["full_generator_source_includes"],
                         source_plan["full_generator_public_includes"],
                         require_wheel_flux=False)
                function = ctypes.CDLL(str(dylib)).diffgw_v4_full_source_nqc_inputs
                function.argtypes = [ctypes.c_double] * 6 + [ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
                function.restype = ctypes.c_int
                meta, values, coefficients = _capture(function, row,
                    plan["output_step_M"], plan["capacity"])
                array_hash = hashlib.sha256(values.tobytes()).hexdigest()
                active = coefficients[[0, 1, 2, 6, 7]]
                if (array_hash != item["input_array_sha256"]
                        or len(values) != item["samples"]
                        or abs(meta[1] - item["internal_step_M"]) > 1e-12
                        or abs(meta[2] - item["match_time_M"]) > 1e-12
                        or np.max(abs(active-item["active_nqc_coefficients"])) > 1e-9):
                    raise ValueError(f"Factor {factor} fails frozen source replay")
                captures[factor] = (meta, values)
                report["rows"].append(dict(resample_factor=factor, samples=len(values),
                    input_array_sha256=array_hash, internal_step_M=float(meta[1]),
                    match_time_M=float(meta[2]), active_nqc_coefficients=active.tolist()))
                q.write_evidence(args.json_out, report)
                print(json.dumps(dict(replayed_factor=factor, samples=len(values))), flush=True)
        low_meta, low = captures[16]
        high_meta, high = captures[32]
        stride = 2
        if (len(high[::stride]) < len(low)
                or abs(high_meta[1]*stride-low_meta[1]) > 1e-12
                or abs(high_meta[2]-low_meta[2]) > 1e-12):
            raise ValueError("Source grids do not share the expected timestamps")
        high_shared = high[::stride][:len(low)]
        difference = high_shared-low
        time = np.arange(len(low), dtype=np.float64)*low_meta[1]
        match = float(high_meta[2])
        near = abs(time-match) <= plan["near_match_half_width_M"]
        if not np.any(near):
            raise ValueError("No common samples near matching")
        report["comparisons"]["shared_inputs"] = {
            name: dict(max_abs_all=float(np.max(abs(difference[:, j]))),
                       max_abs_near_match=float(np.max(abs(difference[near, j]))),
                       max_near_match_over_local_scale=float(
                           np.max(abs(difference[near, j])) /
                           np.max(abs(low[near, j]))),
                       nearest_match_difference=float(
                           difference[np.argmin(abs(time-match)), j]))
            for j, name in enumerate(NAMES)}
        report["comparisons"]["shared_input_count"] = len(low)
        report["comparisons"]["near_match_count"] = int(near.sum())
        report["comparisons"]["time_step_M"] = float(low_meta[1])
        high_truncated = high[:stride*(len(low)-1)+1]
        fits = {}
        for name, values, step in (("factor16", low, low_meta[1]),
                                   ("factor32", high, high_meta[1]),
                                   ("factor32_truncated", high_truncated, high_meta[1]),
                                   ("factor32_decimated", high_shared, low_meta[1])):
            axis = np.arange(len(values), dtype=np.float64)*step
            fits[name] = _fit(gsl, axis, *values.T, match, high_meta[3:])
        for name in ("factor16", "factor32"):
            item = baseline[int(name[6:])]["common_match_counterfactual"]
            if np.max(abs(np.asarray(fits[name]["coefficients"])
                          -item["gsl_coefficients"])) > 1e-9:
                raise ValueError(f"{name} common-match GSL fit fails frozen replay")
        report["fits"] = {name: dict(coefficients=fit["coefficients"],
            amplitude_second_derivative=fit["amplitude_derivatives"][2],
            amplitude_condition=fit["amplitude_condition"],
            phase_condition=fit["phase_condition"]) for name, fit in fits.items()}
        report["comparisons"]["coefficient_l2"] = {
            "actual_16_to_32": float(np.linalg.norm(np.asarray(
                fits["factor16"]["coefficients"])-fits["factor32"]["coefficients"])),
            "shared_samples_16_to_32_decimated": float(np.linalg.norm(np.asarray(
                fits["factor16"]["coefficients"])-fits["factor32_decimated"]["coefficients"])),
            "same_run_decimated_to_full": float(np.linalg.norm(np.asarray(
                fits["factor32_decimated"]["coefficients"])-fits["factor32"]["coefficients"])),
            "same_run_truncated_to_full": float(np.linalg.norm(np.asarray(
                fits["factor32_truncated"]["coefficients"])-fits["factor32"]["coefficients"])),
            "same_span_decimated_to_full": float(np.linalg.norm(np.asarray(
                fits["factor32_decimated"]["coefficients"])-fits["factor32_truncated"]["coefficients"])),
        }
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Dev-only q4 corrected-source cadence test with finest NQC coefficients fixed."""

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
from .diagnose_peak_anchor import fitted_metrics
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import INCLUDE
from .diagnose_source_nqc_whole_mode_override import OVERRIDE, override_source
from .diagnose_source_q4_peak_bracket_correction import ANCHOR, CORRECTION
from .diagnose_source_wheel_mode import _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/source_q4_corrected_peak_fixed_nqc_plan.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    keys = ("peak_correction", "peak_script", "source_flux_plan", "adapter")
    paths = {key: q.ROOT / plan[key] for key in keys}
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-corrected-peak-fixed-nqc-plan.v1"
            or plan["release_qualified"] is not False
            or plan["steps_M"] != [0.015625, 0.0078125, 0.00390625]
            or plan["fixed_coefficient_step_M"] != 0.00390625
            or plan["capacity"] != 200000
            or plan["peak_fit"] != "quadratic_power"
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in paths.items())):
        parser.error("Frozen plan or input changed")
    corrected = json.loads(paths["peak_correction"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (corrected["status"] != "diagnostic_complete"
            or corrected["release_qualified"] is not False
            or set(corrected["grids"]) != set(map(str, plan["steps_M"]))):
        parser.error("Corrected-source evidence changed")
    row = corrected["row"]
    mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    fixed = np.ascontiguousarray(
        corrected["grids"][str(plan["fixed_coefficient_step_M"])]["corrected_source_active"],
        dtype=np.float64)
    report = {
        "schema": "diffgw.seobnrv4.source-q4-corrected-peak-fixed-nqc.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in paths.items()},
        "correction_sha256": hashlib.sha256(CORRECTION.encode()).hexdigest(),
        "override_sha256": hashlib.sha256(OVERRIDE.encode()).hexdigest(),
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": row, "fixed_active_nqc": fixed.tolist(),
        "grids": {}, "adjacent_fixed": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        previous = None
        with tempfile.TemporaryDirectory(prefix="diffgw-source-q4-corrected-fixed-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(ANCHOR) != 1 or original.count(INCLUDE) != 1:
                raise ValueError("Pinned source patch anchors changed")
            source_file.write_text(original.replace(ANCHOR, CORRECTION+ANCHOR)
                                   .replace(INCLUDE, INCLUDE+OVERRIDE))
            library_path = Path(tmp) / "corrected_fixed_nqc.dylib"
            _compile(private, args.headers, args.gsl_include,
                     _libraries(source_plan["pinned_wheel"]), paths["adapter"],
                     library_path, source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            library = ctypes.CDLL(str(library_path))
            baseline_function = library.diffgw_v4_full_source_mode
            baseline_function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            baseline_function.restype = ctypes.c_int
            override_function = library.diffgw_v4_full_source_mode_with_nqc
            override_function.argtypes = baseline_function.argtypes + [
                ctypes.POINTER(ctypes.c_double)]
            override_function.restype = ctypes.c_int
            for step in plan["steps_M"]:
                baseline = _source(baseline_function, row, step, plan["capacity"])
                frozen = corrected["grids"][str(step)]
                if (hashlib.sha256(baseline[2].tobytes()).hexdigest()
                        != frozen["corrected_source_mode_sha256"]
                        or abs(baseline[1]/mass_seconds-step) > 1e-10):
                    raise ValueError(f"Corrected source replay changed at {step:g} M")
                own = np.ascontiguousarray(frozen["corrected_source_active"],
                                           dtype=np.float64)
                noop = override_source(override_function, row, step,
                                       plan["capacity"], own)
                if noop[0] != baseline[0] or noop[1] != baseline[1] or not np.array_equal(
                        noop[2], baseline[2]):
                    raise ValueError(f"Own-coefficient control failed at {step:g} M")
                modified = override_source(override_function, row, step,
                                           plan["capacity"], fixed)
                baseline_time = baseline[0] + np.arange(len(baseline[2]))*baseline[1]
                modified_time = modified[0] + np.arange(len(modified[2]))*modified[1]
                difference = fitted_metrics(baseline_time, baseline[2],
                                            modified_time, modified[2],
                                            mass_seconds, plan["peak_fit"])
                report["grids"][str(step)] = {
                    "corrected_source_mode_sha256": frozen["corrected_source_mode_sha256"],
                    "noop_mode_sha256": hashlib.sha256(noop[2].tobytes()).hexdigest(),
                    "fixed_mode_sha256": hashlib.sha256(modified[2].tobytes()).hexdigest(),
                    "own_active_nqc": own.tolist(),
                    "fixed_minus_own_active_l2": float(np.linalg.norm(fixed-own)),
                    "fixed_minus_own_mode": difference,
                    "fixed_minus_own_epoch_M": (modified[0]-baseline[0])/mass_seconds,
                    "fixed_minus_own_samples": len(modified[2])-len(baseline[2]),
                }
                if previous is not None:
                    old_step, old_time, old_mode = previous
                    report["adjacent_fixed"][f"{old_step:g}_to_{step:g}"] = fitted_metrics(
                        old_time, old_mode, modified_time, modified[2],
                        mass_seconds, plan["peak_fit"])
                previous = (step, modified_time, modified[2])
                q.write_evidence(args.json_out, report)
                print(f"step={step:g} fixed/own="
                      f"{difference['metrics']['complex_relative_l2']:.9g}", flush=True)
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

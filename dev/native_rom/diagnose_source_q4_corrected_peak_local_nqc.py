"""Dev-only local NQC fit on corrected-source q4 fine grids."""

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
from .diagnose_peak_anchor import fitted_metrics
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_high_rate import LocalPolynomial
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture
from .diagnose_source_nqc_whole_mode_override import OVERRIDE, override_source
from .diagnose_source_q4_peak_bracket_correction import ANCHOR, CORRECTION, _function
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/source_q4_corrected_peak_local_nqc_plan.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    keys = ("peak_correction", "peak_script", "whole_ladder", "source_flux_plan",
            "capture_adapter", "override_adapter", "polynomial_impl", "fit_impl")
    paths = {key: q.ROOT / plan[key] for key in keys}
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-corrected-peak-local-nqc-plan.v1"
            or plan["release_qualified"] is not False
            or plan["steps_M"] != [0.015625, 0.0078125, 0.00390625]
            or plan["polynomial_degree"] != 4
            or plan["polynomial_half_width_M"] != 2.0
            or plan["capture_capacity"] != 50000
            or plan["mode_capacity"] != 200000
            or plan["peak_fit"] != "quadratic_power"
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in paths.items())):
        parser.error("Frozen plan or input changed")
    corrected = json.loads(paths["peak_correction"].read_text())
    whole = json.loads(paths["whole_ladder"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (corrected["status"] != "diagnostic_complete"
            or corrected["release_qualified"] is not False
            or whole["status"] != "diagnostic_complete"
            or whole["row"] != corrected["row"]
            or set(corrected["grids"]) != set(map(str, plan["steps_M"]))):
        parser.error("Corrected-source evidence changed")
    row = corrected["row"]
    mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    report = {
        "schema": "diffgw.seobnrv4.source-q4-corrected-peak-local-nqc.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in paths.items()},
        "correction_sha256": hashlib.sha256(CORRECTION.encode()).hexdigest(),
        "capture_sha256": hashlib.sha256(CAPTURE.encode()).hexdigest(),
        "override_sha256": hashlib.sha256(OVERRIDE.encode()).hexdigest(),
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": row, "grids": {}, "adjacent_local": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        libraries = _libraries(source_plan["pinned_wheel"])
        polynomial = LocalPolynomial(plan["polynomial_half_width_M"],
                                     plan["polynomial_degree"])
        with tempfile.TemporaryDirectory(prefix="diffgw-source-q4-corrected-local-") as tmp:
            functions = {}
            for kind, patch, adapter in (
                ("capture", CAPTURE, "capture_adapter"),
                ("override", OVERRIDE, "override_adapter"),
            ):
                private = Path(tmp) / kind
                _snapshot(args.source_root, source_plan["pinned_source"],
                          private, args.headers)
                source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
                original = source_file.read_text()
                if original.count(ANCHOR) != 1 or original.count(INCLUDE) != 1:
                    raise ValueError("Pinned source patch anchors changed")
                source_file.write_text(original.replace(ANCHOR, CORRECTION+ANCHOR)
                                       .replace(INCLUDE, INCLUDE+patch))
                library_path = Path(tmp) / f"{kind}.dylib"
                _compile(private, args.headers, args.gsl_include, libraries,
                         paths[adapter], library_path,
                         source_plan["full_generator_source_includes"],
                         source_plan["full_generator_public_includes"],
                         require_wheel_flux=False)
                if kind == "capture":
                    functions["capture"] = _function(
                        library_path, "diffgw_v4_full_source_nqc_inputs")
                else:
                    library = ctypes.CDLL(str(library_path))
                    baseline = library.diffgw_v4_full_source_mode
                    baseline.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
                    baseline.restype = ctypes.c_int
                    override = library.diffgw_v4_full_source_mode_with_nqc
                    override.argtypes = baseline.argtypes + [ctypes.POINTER(ctypes.c_double)]
                    override.restype = ctypes.c_int
                    functions.update(baseline=baseline, override=override)
            previous = None
            for step in plan["steps_M"]:
                frozen = corrected["grids"][str(step)]
                meta, values, coefficients = _capture(
                    functions["capture"], row, step, plan["capture_capacity"])
                own = np.ascontiguousarray(coefficients[[0, 1, 2, 6, 7]])
                if (hashlib.sha256(values.tobytes()).hexdigest() != frozen["input_sha256"]
                        or not np.array_equal(own, np.asarray(frozen["corrected_source_active"]))
                        or abs(meta[0]-frozen["corrected_orbital_peak_M"]) > 1e-12):
                    raise ValueError(f"Corrected NQC inputs changed at {step:g} M")
                time_axis = np.arange(len(values), dtype=np.float64)*meta[1]
                fit = _fit(polynomial, time_axis, *values.T, meta[2], meta[3:])
                local_active = np.ascontiguousarray(fit["coefficients"], dtype=np.float64)
                baseline_mode = _source(functions["baseline"], row, step,
                                        plan["mode_capacity"])
                if (hashlib.sha256(baseline_mode[2].tobytes()).hexdigest()
                        != frozen["corrected_source_mode_sha256"]):
                    raise ValueError(f"Corrected source mode changed at {step:g} M")
                noop = override_source(functions["override"], row, step,
                                       plan["mode_capacity"], own)
                if noop[:2] != baseline_mode[:2] or not np.array_equal(
                        noop[2], baseline_mode[2]):
                    raise ValueError(f"Own-coefficient control failed at {step:g} M")
                local_mode = override_source(functions["override"], row, step,
                                             plan["mode_capacity"], local_active)
                wheel = _public(row, step)
                if (hashlib.sha256(wheel[2].tobytes()).hexdigest()
                        != whole["grids"][str(step)]["public_mode_sha256"]):
                    raise ValueError(f"Public wheel mode changed at {step:g} M")
                baseline_time = baseline_mode[0]+np.arange(len(baseline_mode[2]))*baseline_mode[1]
                local_time = local_mode[0]+np.arange(len(local_mode[2]))*local_mode[1]
                wheel_time = wheel[0]+np.arange(len(wheel[2]))*wheel[1]
                report["grids"][str(step)] = {
                    "input_sha256": frozen["input_sha256"],
                    "corrected_source_mode_sha256": frozen["corrected_source_mode_sha256"],
                    "noop_mode_sha256": hashlib.sha256(noop[2].tobytes()).hexdigest(),
                    "local_mode_sha256": hashlib.sha256(local_mode[2].tobytes()).hexdigest(),
                    "source_active": own.tolist(), "local_fit": fit,
                    "local_minus_source_active_l2": float(np.linalg.norm(local_active-own)),
                    "local_minus_source_mode": fitted_metrics(
                        local_time, local_mode[2], baseline_time, baseline_mode[2],
                        mass_seconds, plan["peak_fit"]),
                    "local_minus_public_mode": fitted_metrics(
                        local_time, local_mode[2], wheel_time, wheel[2],
                        mass_seconds, plan["peak_fit"]),
                    "local_minus_source_epoch_M": (local_mode[0]-baseline_mode[0])/mass_seconds,
                    "local_minus_source_samples": len(local_mode[2])-len(baseline_mode[2]),
                }
                if previous is not None:
                    old_step, old_time, old_mode, old_active = previous
                    pair = f"{old_step:g}_to_{step:g}"
                    report["adjacent_local"][pair] = fitted_metrics(
                        old_time, old_mode, local_time, local_mode[2],
                        mass_seconds, plan["peak_fit"])
                    report["adjacent_local"][pair]["coefficient_difference_l2"] = float(
                        np.linalg.norm(local_active-old_active))
                previous = (step, local_time, local_mode[2], local_active)
                q.write_evidence(args.json_out, report)
                print(f"step={step:g} local/own="
                      f"{report['grids'][str(step)]['local_minus_source_mode']['metrics']['complex_relative_l2']:.9g}",
                      flush=True)
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

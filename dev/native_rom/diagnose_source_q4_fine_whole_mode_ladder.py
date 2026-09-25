"""Dev-only pinned-source versus public q4 whole modes at fine cadences."""

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
from .diagnose_source_wheel_mode import _public, _public_nqc, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/source_q4_fine_whole_mode_ladder_plan.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    keys = ("source_nqc_ladder", "prior_source_mode", "source_flux_plan",
            "large_adapter")
    paths = {key: q.ROOT / plan[key] for key in keys}
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-fine-whole-mode-ladder-plan.v1"
            or plan["release_qualified"] is not False
            or plan["steps_M"] != [0.015625, 0.0078125, 0.00390625]
            or plan["capacity"] != 200000
            or plan["peak_fit"] != "quadratic_power"
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in paths.items())):
        parser.error("Frozen plan or prerequisites changed")
    nqc = json.loads(paths["source_nqc_ladder"].read_text())
    prior = json.loads(paths["prior_source_mode"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (nqc["status"] != "diagnostic_complete"
            or prior["status"] != "diagnostic_complete"
            or nqc["row"] != prior["row"]):
        parser.error("Pinned q4 prerequisites changed")
    identity = q.lal_binary_identity(prior["public_binary"]["lal_version"],
                                      prior["public_binary"]["lalsimulation_version"])
    if identity != prior["public_binary"]:
        parser.error("Installed public binary changed")
    row = nqc["row"]
    mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
    report = {
        "schema": "diffgw.seobnrv4.source-q4-fine-whole-mode-ladder.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in paths.items()},
        "public_binary_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": row, "grids": {}, "adjacent_source": {}, "adjacent_public": {},
        "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        libraries = _libraries(source_plan["pinned_wheel"])
        previous = None
        with tempfile.TemporaryDirectory(prefix="diffgw-source-q4-whole-ladder-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            library = Path(tmp) / "source_full_mode.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["large_adapter"], library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            function = ctypes.CDLL(str(library)).diffgw_v4_full_source_mode
            function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            for step in plan["steps_M"]:
                source_epoch, source_dt, source_mode = _source(
                    function, row, step, plan["capacity"])
                public_epoch, public_dt, public_mode, _ = _public(row, step)
                public_nqc = _public_nqc(row, step)[[0, 1, 2, 6, 7]]
                if (abs(source_dt/mass_seconds-step) > 1e-10
                        or abs(public_dt/mass_seconds-step) > 1e-10):
                    raise ValueError(f"Output cadence changed at {step:g} M")
                source_hash = hashlib.sha256(source_mode.tobytes()).hexdigest()
                if step == plan["steps_M"][0]:
                    baseline = prior["grids"][str(step)]
                    if (source_hash != baseline["source_mode_sha256"]
                            or np.max(abs(public_nqc-
                                          np.asarray(baseline["wheel_active"]))) > 1e-9):
                        raise ValueError("Baseline source/public q4 mode changed")
                source_time = source_epoch + np.arange(len(source_mode))*source_dt
                public_time = public_epoch + np.arange(len(public_mode))*public_dt
                comparison = fitted_metrics(
                    source_time, source_mode, public_time, public_mode,
                    mass_seconds, plan["peak_fit"])
                if (step == plan["steps_M"][0]
                        and abs(comparison["metrics"]["complex_relative_l2"]
                                - baseline["fitted_baseline"]["metrics"]["complex_relative_l2"])
                        > 1e-12):
                    raise ValueError("Baseline source/public whole-mode metric changed")
                report["grids"][str(step)] = {
                    "source_samples": len(source_mode),
                    "public_samples": len(public_mode),
                    "source_mode_sha256": source_hash,
                    "public_mode_sha256": hashlib.sha256(public_mode.tobytes()).hexdigest(),
                    "source_active_nqc": nqc["grids"][str(step)]["source_active"],
                    "public_active_nqc": public_nqc.tolist(),
                    "nqc_coefficient_gap_l2": float(np.linalg.norm(
                        np.asarray(nqc["grids"][str(step)]["source_active"])-public_nqc)),
                    "source_vs_public": comparison,
                }
                if previous is not None:
                    old_step, old_source_time, old_source_mode, old_public_time, old_public_mode = previous
                    pair = f"{old_step:g}_to_{step:g}"
                    report["adjacent_source"][pair] = fitted_metrics(
                        old_source_time, old_source_mode, source_time, source_mode,
                        mass_seconds, plan["peak_fit"])
                    report["adjacent_public"][pair] = fitted_metrics(
                        old_public_time, old_public_mode, public_time, public_mode,
                        mass_seconds, plan["peak_fit"])
                previous = (step, source_time, source_mode, public_time, public_mode)
                q.write_evidence(args.json_out, report)
                print(f"step={step:g} source/public whole="
                      f"{comparison['metrics']['complex_relative_l2']:.9g}", flush=True)
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

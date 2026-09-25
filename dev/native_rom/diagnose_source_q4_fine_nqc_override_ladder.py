"""Dev-only causal NQC override across the pinned q4 fine whole-mode ladder."""

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
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/source_q4_fine_nqc_override_ladder_plan.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    keys = ("whole_ladder", "prior_override", "source_flux_plan", "adapter")
    paths = {key: q.ROOT / plan[key] for key in keys}
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-fine-nqc-override-ladder-plan.v1"
            or plan["release_qualified"] is not False
            or plan["steps_M"] != [0.015625, 0.0078125, 0.00390625]
            or plan["capacity"] != 200000
            or plan["peak_fit"] != "quadratic_power"
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in paths.items())):
        parser.error("Frozen plan or prerequisites changed")
    whole = json.loads(paths["whole_ladder"].read_text())
    prior = json.loads(paths["prior_override"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (whole["status"] != "diagnostic_complete"
            or prior["status"] != "diagnostic_complete"
            or whole["row"] != prior["row"]
            or whole["release_qualified"] is not False
            or prior["release_qualified"] is not False):
        parser.error("Pinned q4 prerequisites changed")
    row = whole["row"]
    mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    report = {
        "schema": "diffgw.seobnrv4.source-q4-fine-nqc-override-ladder.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in paths.items()},
        "instrumentation_sha256": hashlib.sha256(OVERRIDE.encode()).hexdigest(),
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": row, "grids": {}, "adjacent_modified": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        previous = None
        with tempfile.TemporaryDirectory(prefix="diffgw-source-q4-fine-nqc-override-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("Pinned source NQC include anchor changed")
            source_file.write_text(original.replace(INCLUDE, INCLUDE + OVERRIDE))
            library = Path(tmp) / "source_q4_fine_nqc_override.dylib"
            _compile(private, args.headers, args.gsl_include,
                     _libraries(source_plan["pinned_wheel"]), paths["adapter"], library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            lib = ctypes.CDLL(str(library))
            baseline_function = lib.diffgw_v4_full_source_mode
            baseline_function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            baseline_function.restype = ctypes.c_int
            override_function = lib.diffgw_v4_full_source_mode_with_nqc
            override_function.argtypes = baseline_function.argtypes + [
                ctypes.POINTER(ctypes.c_double)]
            override_function.restype = ctypes.c_int
            for step in plan["steps_M"]:
                source_epoch, source_dt, source_mode = _source(
                    baseline_function, row, step, plan["capacity"])
                wheel_epoch, wheel_dt, wheel_mode, _ = _public(row, step)
                expected = whole["grids"][str(step)]
                if (hashlib.sha256(source_mode.tobytes()).hexdigest()
                        != expected["source_mode_sha256"]
                        or hashlib.sha256(wheel_mode.tobytes()).hexdigest()
                        != expected["public_mode_sha256"]
                        or abs(source_dt/mass_seconds-step) > 1e-10
                        or abs(wheel_dt/mass_seconds-step) > 1e-10):
                    raise ValueError(f"Pinned whole modes changed at {step:g} M")
                source_active = np.asarray(expected["source_active_nqc"], dtype=np.float64)
                wheel_active = np.asarray(expected["public_active_nqc"], dtype=np.float64)
                noop = override_source(override_function, row, step,
                                       plan["capacity"], source_active)
                modified = override_source(override_function, row, step,
                                           plan["capacity"], wheel_active)
                if (noop[0] != source_epoch or noop[1] != source_dt
                        or not np.array_equal(noop[2], source_mode)
                        or modified[1] != source_dt or len(modified[2]) < 10):
                    raise ValueError(f"NQC intervention control changed at {step:g} M")
                source_time = source_epoch + np.arange(len(source_mode))*source_dt
                wheel_time = wheel_epoch + np.arange(len(wheel_mode))*wheel_dt
                modified_time = modified[0] + np.arange(len(modified[2]))*modified[1]
                baseline = fitted_metrics(source_time, source_mode,
                                          wheel_time, wheel_mode, mass_seconds,
                                          plan["peak_fit"])
                updated = fitted_metrics(modified_time, modified[2],
                                         wheel_time, wheel_mode, mass_seconds,
                                         plan["peak_fit"])
                if abs(baseline["metrics"]["complex_relative_l2"]
                       - expected["source_vs_public"]["metrics"]["complex_relative_l2"]) > 1e-12:
                    raise ValueError(f"Pinned whole-mode comparison changed at {step:g} M")
                modified_hash = hashlib.sha256(modified[2].tobytes()).hexdigest()
                if step == plan["steps_M"][0]:
                    control = prior["grids"][str(step)]
                    if (modified_hash != control["modified_mode_sha256"]
                            or abs(updated["metrics"]["complex_relative_l2"]
                                   - control["fitted_modified"]["metrics"]["complex_relative_l2"])
                            > 1e-12):
                        raise ValueError("Prior causal NQC override changed")
                report["grids"][str(step)] = {
                    "source_active": source_active.tolist(),
                    "wheel_active": wheel_active.tolist(),
                    "source_mode_sha256": expected["source_mode_sha256"],
                    "wheel_mode_sha256": expected["public_mode_sha256"],
                    "noop_mode_sha256": hashlib.sha256(noop[2].tobytes()).hexdigest(),
                    "modified_mode_sha256": modified_hash,
                    "modified_minus_source_epoch_M": (modified[0]-source_epoch)/mass_seconds,
                    "modified_minus_source_samples": len(modified[2])-len(source_mode),
                    "baseline": baseline, "modified": updated,
                }
                if previous is not None:
                    old_step, old_time, old_mode = previous
                    pair = f"{old_step:g}_to_{step:g}"
                    report["adjacent_modified"][pair] = fitted_metrics(
                        old_time, old_mode, modified_time, modified[2],
                        mass_seconds, plan["peak_fit"])
                previous = (step, modified_time, modified[2])
                q.write_evidence(args.json_out, report)
                print(f"step={step:g} source/wheel="
                      f"{baseline['metrics']['complex_relative_l2']:.9g}->"
                      f"{updated['metrics']['complex_relative_l2']:.9g}", flush=True)
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

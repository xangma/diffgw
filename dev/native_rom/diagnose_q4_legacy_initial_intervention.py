"""Dev-only whole-mode effect of the pinned source's legacy initial momentum."""

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
from unittest.mock import patch

import lal
import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_q4_ultrafine_chart import _native
from .diagnose_source_wheel_mode import _source
from .eob.torch import warm_start
from torchwave.constants import C_SI, MPC_SI


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/q4_legacy_initial_intervention_plan.json"
SETTINGS = q.ROOT / "dev/native_rom/evidence/2026-09-24/combined_domain_numerics.json"


def _native_chart(row: dict, run: dict) -> tuple[np.ndarray, np.ndarray]:
    mass_seconds = (row["mass1"] + row["mass2"]) * float(lal.MTSUN_SI)
    return ((run["time"] - run["peak_time_M"]) * mass_seconds,
            run["mode"] * mass_seconds * float(C_SI) / (100. * float(MPC_SI)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("Pinned source adapter requires macOS")
    plan = json.loads(PLAN.read_text())
    inputs = {key: q.ROOT / plan[key] for key in (
        "prior_chart", "source_flux_plan", "initial_adapter", "mode_adapter")}
    if (plan["schema"] != "diffgw.seobnrv4.q4-legacy-initial-intervention-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row_name"] != "q4_nonspinning"
            or plan["component"] != "pr_star"
            or plan["output_step_M"] != 0.03125
            or plan["warm_start_Momega"] != 0.03162277660168379
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in inputs.items())):
        parser.error("Frozen plan or prerequisites changed")
    prior = json.loads(inputs["prior_chart"].read_text())
    source_plan = json.loads(inputs["source_flux_plan"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or prior["row"]["mass1"] != 40.
            or prior["row"]["mass2"] != 10.
            or prior["row"]["chi1"] != 0.
            or prior["row"]["chi2"] != 0.):
        parser.error("Frozen q4 case changed")
    row = prior["row"]
    step = plan["output_step_M"]
    old = prior["grids"][str(step)]
    settings = q.Settings(**json.loads(SETTINGS.read_text())["settings"])
    torch.set_num_threads(1)
    report = {
        "schema": "diffgw.seobnrv4.q4-legacy-initial-intervention.v1",
        "status": "running", "release_qualified": False, "scope": q.SCOPE,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in inputs.items()},
        "settings_sha256": q.sha256(SETTINGS),
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": row, "source_initial": None, "native_initial": None,
        "baseline": None, "intervention": None, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-q4-legacy-initial-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            libraries = _libraries(source_plan["pinned_wheel"])
            initial_lib = Path(tmp) / "initial.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     inputs["initial_adapter"], initial_lib,
                     source_plan["flux_source_includes"],
                     source_plan["flux_public_includes"])
            initial_fn = ctypes.CDLL(str(initial_lib)).diffgw_v4_initial_conditions
            initial_fn.argtypes = [ctypes.c_double] * 5 + [
                ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
            initial_fn.restype = ctypes.c_int
            initial = (ctypes.c_double * 3)()
            rc = initial_fn(row["mass1"], row["mass2"], row["chi1"],
                            row["chi2"], plan["warm_start_Momega"], 0, initial)
            if rc:
                raise RuntimeError(f"Pinned source legacy initial state failed: {rc}")
            report["source_initial"] = dict(zip(("r", "pr_star", "pphi"), initial))
            mode_lib = Path(tmp) / "mode.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     inputs["mode_adapter"], mode_lib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            source_fn = ctypes.CDLL(str(mode_lib)).diffgw_v4_full_source_mode
            source_fn.argtypes = [ctypes.c_double] * 6 + [
                ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)]
            source_fn.restype = ctypes.c_int
            epoch, delta_t, source_mode = _source(
                source_fn, row, step, plan["capacity"])
            if hashlib.sha256(source_mode.tobytes()).hexdigest() != old[
                    "source"]["mode_sha256"]:
                raise ValueError("Pinned source mode differs from frozen chart")
            source_time = epoch + np.arange(source_mode.size) * delta_t
            mass_seconds = (row["mass1"] + row["mass2"]) * float(lal.MTSUN_SI)
            started = time.monotonic()
            baseline = _native(row, step, {"atol": 1e-14, "rtol": 1e-13}, settings)
            baseline_summary = q.summarize_run(baseline)
            if baseline_summary["array_sha256"] != old["native"]["array_sha256"]:
                raise ValueError("Native baseline differs from frozen chart")
            baseline_chart = _native_chart(row, baseline)
            baseline_metric = fitted_metrics(
                *baseline_chart, source_time, source_mode,
                mass_seconds, plan["peak_fit"])
            if abs(baseline_metric["metrics"]["complex_relative_l2"] - old[
                    "native_vs_source_fitted"]["metrics"]["complex_relative_l2"]) > 1e-12:
                raise ValueError("Native/source baseline comparison changed")
            report["baseline"] = {
                "native": baseline_summary, "vs_source": baseline_metric,
                "elapsed_seconds": time.monotonic() - started}
            q.write_evidence(args.json_out, report)
            original = warm_start.find_quasicircular_initial_conditions
            calls = []

            def substitute(*positional, **named):
                radius, radial, angular = original(*positional, **named)
                calls.append(dict(r=float(radius), pr_star=float(radial),
                                  pphi=float(angular)))
                return radius, radial.new_tensor(initial[1]), angular

            started = time.monotonic()
            with patch.object(warm_start, "find_quasicircular_initial_conditions",
                              substitute):
                candidate = _native(
                    row, step, {"atol": 1e-14, "rtol": 1e-13}, settings)
            if len(calls) != 1:
                raise ValueError("Expected one warm-start initial-state substitution")
            report["native_initial"] = calls[0]
            candidate_chart = _native_chart(row, candidate)
            candidate_metric = fitted_metrics(
                *candidate_chart, source_time, source_mode,
                mass_seconds, plan["peak_fit"])
            report["intervention"] = {
                "native": q.summarize_run(candidate),
                "vs_source": candidate_metric,
                "relative_l2_change": candidate_metric["metrics"][
                    "complex_relative_l2"] - baseline_metric["metrics"][
                        "complex_relative_l2"],
                "elapsed_seconds": time.monotonic() - started}
            report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"],
                      "relative_l2_change": (report["intervention"] or {}).get(
                          "relative_l2_change")}), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

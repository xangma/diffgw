"""Dev-only identical-initial-state regular/optimized pinned-source evolution."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import numpy as np

from . import qualify_generator as q
from .diagnose_optimized_source_field import FIELDS
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/regular_optimized_trajectory_plan.json"


def _integrate(function, row, initial, samples, plan):
    calls = 0

    def rhs(_time, state):
        nonlocal calls
        out = (ctypes.c_double * 6)()
        rc = function(*[row[key] for key in (
            "mass1", "mass2", "chi1", "chi2")], *state, out)
        if rc:
            raise RuntimeError(f"Pinned source field failed after {calls} calls: {rc}")
        calls += 1
        return np.asarray(out[:4], dtype=np.float64)

    started = time.monotonic()
    step = plan["step_M"]
    every = round(plan["sample_spacing_M"] / step)
    count = round(plan["duration_M"] / step)
    if (abs(every * step - plan["sample_spacing_M"]) > 1e-12
            or abs(count * step - plan["duration_M"]) > 1e-12
            or count // every + 1 != samples.size):
        raise ValueError("Fixed solver step does not divide sample grid")
    state = initial.copy()
    states = [state.copy()]
    for index in range(1, count + 1):
        k1 = rhs(0., state)
        k2 = rhs(0., state + step * k1 / 2)
        k3 = rhs(0., state + step * k2 / 2)
        k4 = rhs(0., state + step * k3)
        state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if not np.isfinite(state).all() or state[0] <= 0:
            raise RuntimeError(f"Invalid fixed-step trajectory at index {index}")
        if index % every == 0:
            states.append(state.copy())
    return {
        "calls": calls, "elapsed_seconds": time.monotonic() - started,
        "states": [{field: float(value[i]) for i, field in enumerate(FIELDS)}
                   for value in states],
    }


def _difference(a, b, samples):
    rows = []
    for time_m, left, right in zip(samples, a["states"], b["states"]):
        rows.append({"time_M": float(time_m),
                     "delta": {field: left[field] - right[field]
                               for field in FIELDS}})
    return {
        "samples": rows,
        "max_absolute_by_field": {
            field: max(abs(item["delta"][field]) for item in rows)
            for field in FIELDS},
    }


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
        "source_flux_plan", "initial_boundary_result", "regular_field_result",
        "regular_adapter", "optimized_adapter")}
    if (plan["schema"] != "diffgw.seobnrv4.regular-optimized-trajectory-plan.v1"
            or plan["release_qualified"] is not False
            or plan["solver"] != "fixed_rk4"
            or plan["steps_M"] != [2., 1., 0.5]
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in inputs.items())):
        parser.error("Frozen plan or prerequisites changed")
    source_plan = json.loads(inputs["source_flux_plan"].read_text())
    initial_result = json.loads(inputs["initial_boundary_result"].read_text())
    regular_result = json.loads(inputs["regular_field_result"].read_text())
    cases = [point for point in initial_result["points"]
             if point["name"] == plan["row_name"]
             and point["frequency_Momega"] == plan["initial_Momega"]]
    if (initial_result["status"] != "diagnostic_complete"
            or regular_result["status"] != "diagnostic_complete"
            or len(cases) != 1):
        parser.error("Frozen initial state or source field changed")
    case = cases[0]
    row = case["physical"]
    start = case["source"]["legacy"]
    initial = np.asarray([start["r"], 0., start["pr_star"], start["pphi"]])
    samples = np.arange(
        0., plan["duration_M"] + plan["sample_spacing_M"] / 2,
        plan["sample_spacing_M"], dtype=np.float64)
    report = {
        "schema": "diffgw.seobnrv4.regular-optimized-trajectory.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in inputs.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": row, "initial_Momega": plan["initial_Momega"],
        "initial_state": dict(zip(FIELDS, initial.tolist())),
        "sample_times_M": samples.tolist(), "runs": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-field-trajectory-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            functions = {}
            for route, symbol in (
                    ("regular", "diffgw_v4_source_field"),
                    ("optimized", "diffgw_v4_optimized_source_field")):
                library_path = Path(tmp) / f"{route}.dylib"
                _compile(private, args.headers, args.gsl_include,
                         _libraries(source_plan["pinned_wheel"]),
                         inputs[f"{route}_adapter"], library_path,
                         source_plan["full_generator_source_includes"],
                         source_plan["full_generator_public_includes"],
                         require_wheel_flux=False)
                function = getattr(ctypes.CDLL(str(library_path)), symbol)
                function.argtypes = [ctypes.c_double] * 8 + [
                    ctypes.POINTER(ctypes.c_double)]
                function.restype = ctypes.c_int
                functions[route] = function
            for route, function in functions.items():
                report["runs"][route] = {}
                for step in plan["steps_M"]:
                    run_plan = dict(plan, step_M=step)
                    report["runs"][route][str(step)] = _integrate(
                        function, row, initial, samples, run_plan)
                    q.write_evidence(args.json_out, report)
                    print(route, step, flush=True)
        coarse, middle, fine = (str(step) for step in plan["steps_M"])
        report["self_refinement"] = {
            route: {
                "middle_minus_coarse": _difference(
                    report["runs"][route][middle],
                    report["runs"][route][coarse], samples),
                "fine_minus_middle": _difference(
                    report["runs"][route][fine],
                    report["runs"][route][middle], samples),
            }
            for route in functions}
        report["regular_minus_optimized"] = _difference(
            report["runs"]["regular"][fine],
            report["runs"]["optimized"][fine], samples)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"],
                      "route_delta": (report.get("regular_minus_optimized") or {}).get(
                          "max_absolute_by_field")}), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

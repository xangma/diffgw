"""Dev-only optimized pinned-source/native vector field on frozen boundary states."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import torch

from . import qualify_generator as q
from .diagnose_optimized_source_field import FIELDS, _point
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/optimized_field_boundary_plan.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("Pinned wheel comparison requires macOS")
    plan = json.loads(PLAN.read_text())
    inputs = {key: q.ROOT / plan[key] for key in (
        "hamiltonian_grid_plan", "source_flux_plan", "optimized_field_plan",
        "adapter")}
    if (plan["schema"] != "diffgw.seobnrv4.optimized-field-boundary-plan.v1"
            or plan["release_qualified"] is not False
            or plan["phi_rule"] != "zero; aligned-spin field is axisymmetric"
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in inputs.items())):
        parser.error("Frozen plan or prerequisites changed")
    grid = json.loads(inputs["hamiltonian_grid_plan"].read_text())
    source_plan = json.loads(inputs["source_flux_plan"].read_text())
    optimized_plan = json.loads(inputs["optimized_field_plan"].read_text())
    if (optimized_plan["adapter_sha256"] != plan["adapter_sha256"]
            or [row["name"] for row in grid["frozen_states"]] != [
                "q4_nonspinning", "q3_strong_antialigned",
                "q10_near_extreme_aligned", "q3_near_extreme_mixed"]
            or any(row["indices"] != [0, 1, 2]
                   for row in grid["frozen_states"])):
        parser.error("Boundary states or source adapter changed")
    libraries = _libraries(source_plan["pinned_wheel"])
    torch.set_num_threads(1)
    report = {
        "schema": "diffgw.seobnrv4.optimized-field-boundary.v1",
        "status": "running", "release_qualified": False, "scope": q.SCOPE,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in inputs.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "points": [], "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-boundary-field-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            library_path = Path(tmp) / "field.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     inputs["adapter"], library_path,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            function = ctypes.CDLL(str(library_path)).diffgw_v4_optimized_source_field
            function.argtypes = [ctypes.c_double] * 8 + [ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            for spec in grid["frozen_states"]:
                for point in spec["explicit_states"]:
                    state = dict(point["state"], phi=0.)
                    if set(state) != set(FIELDS):
                        raise ValueError("Boundary state fields changed")
                    result = _point(function, f"{spec['name']}_{point['index']}",
                                    spec["physical"], state,
                                    optimized_plan["source_placement_flux_choice"])
                    report["points"].append(result)
                q.write_evidence(args.json_out, report)
                print(spec["name"], max(
                    max(abs(value) for value in point["native_minus_source_relative"].values())
                    for point in report["points"][-3:]), flush=True)
        report["max_abs_native_source_field_relative_difference"] = max(
            abs(value) for point in report["points"]
            for value in point["native_minus_source_relative"].values())
        report["max_abs_source_regular_optimized_flux_relative_difference"] = max(
            abs(point["regular_minus_optimized_flux_relative"])
            for point in report["points"])
        report["comparison_gates_passed"] = (
            report["max_abs_native_source_field_relative_difference"]
            <= plan["max_native_source_field_relative_difference"]
            and report["max_abs_source_regular_optimized_flux_relative_difference"]
            <= plan["max_source_regular_optimized_flux_relative_difference"])
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({key: report.get(key) for key in (
        "status", "comparison_gates_passed", "errors")}), flush=True)
    return 0 if (report["status"] == "diagnostic_complete"
                 and report["comparison_gates_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

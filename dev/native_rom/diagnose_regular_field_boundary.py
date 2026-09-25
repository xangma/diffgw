"""Dev-only pinned regular-source/native vector field on frozen boundary states."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

from . import qualify_generator as q
from .diagnose_optimized_source_field import FIELDS, _native
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/regular_field_boundary_plan.json"


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
        "source_flux_plan", "hamiltonian_grid_plan",
        "optimized_field_result", "optimized_field_plan", "adapter")}
    if (plan["schema"] != "diffgw.seobnrv4.regular-field-boundary-plan.v1"
            or plan["release_qualified"] is not False
            or plan["phi_rule"] != "zero; aligned-spin field is axisymmetric"
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in inputs.items())):
        parser.error("Frozen plan or prerequisites changed")
    source_plan = json.loads(inputs["source_flux_plan"].read_text())
    grid = json.loads(inputs["hamiltonian_grid_plan"].read_text())
    optimized = json.loads(inputs["optimized_field_result"].read_text())
    optimized_plan = json.loads(inputs["optimized_field_plan"].read_text())
    if ([spec["name"] for spec in grid["frozen_states"]] != plan["rows"]
            or optimized["status"] != "diagnostic_complete"
            or not optimized["comparison_gates_passed"]
            or len(optimized["points"]) != 12):
        parser.error("Frozen boundary or optimized control changed")
    report = {
        "schema": "diffgw.seobnrv4.regular-field-boundary.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in inputs.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "points": [], "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-regular-field-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            library_path = Path(tmp) / "regular_field.dylib"
            _compile(private, args.headers, args.gsl_include,
                     _libraries(source_plan["pinned_wheel"]),
                     inputs["adapter"], library_path,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            function = ctypes.CDLL(str(library_path)).diffgw_v4_source_field
            function.argtypes = [ctypes.c_double] * 8 + [
                ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            index = 0
            for spec in grid["frozen_states"]:
                row = spec["physical"]
                for point in spec["explicit_states"]:
                    name = f"{spec['name']}_{point['index']}"
                    state = dict(point["state"], phi=0.)
                    old = optimized["points"][index]
                    if (old["name"] != name or old["row"] != row
                            or old["state"] != state or set(state) != set(FIELDS)):
                        raise ValueError(f"Optimized control state changed: {name}")
                    out = (ctypes.c_double * 6)()
                    rc = function(*[row[key] for key in (
                        "mass1", "mass2", "chi1", "chi2")],
                        *[state[key] for key in FIELDS], out)
                    if rc:
                        raise RuntimeError(f"Regular source field failed at {name}: {rc}")
                    native = _native(
                        row, state,
                        optimized_plan["source_placement_flux_choice"])
                    if any(abs(native[i] - old["native_derivative"][field])
                           > 1e-13 * max(1., abs(native[i]))
                           for i, field in enumerate(FIELDS)):
                        raise ValueError(f"Native optimized control changed: {name}")
                    regular = dict(zip(FIELDS, out[:4]))
                    native_field = dict(zip(FIELDS, native))
                    optimized_field = old["source_optimized_derivative"]
                    floor = plan["relative_denominator_floor"]
                    record = {
                        "name": name, "row": row, "state": state,
                        "source_regular_derivative": regular,
                        "source_optimized_derivative": optimized_field,
                        "native_derivative": native_field,
                        "native_minus_regular_relative": {
                            field: (native_field[field] - regular[field])
                            / max(abs(regular[field]), floor) for field in FIELDS},
                        "regular_minus_optimized_relative": {
                            field: (regular[field] - optimized_field[field])
                            / max(abs(optimized_field[field]), floor)
                            for field in FIELDS},
                        "regular_flux_over_nu": out[4],
                        "torque_implied_flux_over_nu": out[5],
                        "flux_relative_difference": out[4] / out[5] - 1.,
                    }
                    report["points"].append(record)
                    index += 1
                q.write_evidence(args.json_out, report)
                print(spec["name"], flush=True)
        report["max_native_regular_relative_by_field"] = {
            field: max(abs(point["native_minus_regular_relative"][field])
                       for point in report["points"]) for field in FIELDS}
        report["max_regular_optimized_relative_by_field"] = {
            field: max(abs(point["regular_minus_optimized_relative"][field])
                       for point in report["points"]) for field in FIELDS}
        report["comparison_gate_passed"] = max(
            report["max_native_regular_relative_by_field"].values()
        ) <= plan["max_native_regular_field_relative_difference"]
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({key: report.get(key) for key in (
        "status", "comparison_gate_passed", "errors")}), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

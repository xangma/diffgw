"""Dev-only source, wheel, and native NQC-free flux at frozen boundary inputs."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

import torch

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _native_flux, _snapshot


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/flux_boundary_grid_plan.json"


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
    inputs = {
        key: q.ROOT / plan[key]
        for key in ("source_flux_plan", "hamiltonian_grid_plan",
                    "source_calibration_result", "adapter")
    }
    if (plan["schema"] != "diffgw.seobnrv4.flux-boundary-grid-plan.v1"
            or plan["release_qualified"] is not False
            or plan["omega_rule"] != "Momega=r^(-3/2), predeclared Kepler comparison input independent of each implementation's derivative"
            or plan["radial_momentum_rule"] != "zero, as required by the pinned source/wheel PA-flux adapter"
            or plan["indices"] != [0, 1, 2]
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in inputs.items())):
        parser.error("Frozen plan or prerequisites changed")
    source_plan = json.loads(inputs["source_flux_plan"].read_text())
    grid = json.loads(inputs["hamiltonian_grid_plan"].read_text())
    calibration = json.loads(inputs["source_calibration_result"].read_text())
    if ([row["name"] for row in grid["frozen_states"]] != plan["rows"]
            or any(row["indices"] != plan["indices"] for row in grid["frozen_states"])
            or source_plan["adapter"]["sha256"] != plan["adapter_sha256"]):
        parser.error("Boundary rows or pinned adapter changed")
    choices = dict(calibration["flux_choice"],
                   **source_plan["source_placement_flux_choice_overrides"])
    libraries = _libraries(source_plan["pinned_wheel"])
    torch.set_num_threads(1)
    report = {
        "schema": "diffgw.seobnrv4.flux-boundary-grid.v1",
        "status": "running", "release_qualified": False,
        "scope": {"physics_validated": False, "production_asset": False,
                  "production_promotion_allowed": False,
                  "training_generator_qualified": False},
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in inputs.items()},
        "flux_choices": choices,
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "rows": [], "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-flux-boundary-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            library_path = Path(tmp) / "flux_boundary.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     inputs["adapter"], library_path,
                     source_plan["flux_source_includes"],
                     source_plan["flux_public_includes"])
            function = ctypes.CDLL(str(library_path)).diffgw_v4_flux_same_state
            function.argtypes = [ctypes.c_double] * 7 + [ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            for spec in grid["frozen_states"]:
                physical = spec["physical"]
                total_mass = physical["mass1"] + physical["mass2"]
                x1, x2 = physical["mass1"] / total_mass, physical["mass2"] / total_mass
                row = {"name": spec["name"], "physical": physical, "points": []}
                report["rows"].append(row)
                for point in spec["explicit_states"]:
                    radius = point["state"]["r"]
                    pphi = point["state"]["pphi"]
                    omega = radius ** -1.5
                    values = (ctypes.c_double * 9)()
                    rc = function(x1, x2, physical["chi1"], physical["chi2"],
                                  radius, pphi, omega, values)
                    if rc:
                        raise RuntimeError(f"{spec['name']} index {point['index']}: C rc={rc}")
                    source, wheel = values[0], values[1]
                    native = _native_flux(physical, radius, pphi, omega, choices)
                    if (not all(math.isfinite(value) for value in (source, wheel, native))
                            or source <= 0):
                        raise ValueError(f"Nonfinite or nonpositive flux at {spec['name']} {point['index']}")
                    row["points"].append({
                        "index": point["index"], "r": radius, "pphi": pphi,
                        "omega": omega, "source_flux_over_nu": source,
                        "wheel_flux_over_nu": wheel, "native_flux_over_nu": native,
                        "wheel_minus_source_relative": (wheel - source) / source,
                        "native_minus_source_relative": (native - source) / source,
                        "native_minus_wheel_relative_to_source": (native - wheel) / source,
                    })
                row["max_abs_wheel_minus_source_relative"] = max(
                    abs(point["wheel_minus_source_relative"]) for point in row["points"])
                row["max_abs_native_minus_source_relative"] = max(
                    abs(point["native_minus_source_relative"]) for point in row["points"])
                row["max_abs_native_minus_wheel_relative_to_source"] = max(
                    abs(point["native_minus_wheel_relative_to_source"])
                    for point in row["points"])
                q.write_evidence(args.json_out, report)
                print(spec["name"], row["max_abs_wheel_minus_source_relative"],
                      row["max_abs_native_minus_source_relative"], flush=True)
        report["max_abs_wheel_minus_source_relative"] = max(
            row["max_abs_wheel_minus_source_relative"] for row in report["rows"])
        report["max_abs_native_minus_source_relative"] = max(
            row["max_abs_native_minus_source_relative"] for row in report["rows"])
        report["max_abs_native_minus_wheel_relative_to_source"] = max(
            row["max_abs_native_minus_wheel_relative_to_source"]
            for row in report["rows"])
        report["comparison_gates_passed"] = (
            report["max_abs_wheel_minus_source_relative"]
            <= plan["source_wheel_max_relative_difference"]
            and report["max_abs_native_minus_source_relative"]
            <= plan["native_source_max_relative_difference"])
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

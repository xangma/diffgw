"""Dev-only pinned-source/native 35-mode flux comparison on frozen boundary states."""

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
from .diagnose_source_flux_initial import (
    MODES, _compile, _libraries, _native_mode_and_flux, _snapshot,
)


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/flux_boundary_modes_plan.json"


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
        "boundary_grid_plan", "source_flux_plan", "source_calibration_result",
        "adapter")}
    if (plan["schema"] != "diffgw.seobnrv4.flux-boundary-modes-plan.v1"
            or plan["release_qualified"] is not False
            or plan["mode_count"] != len(MODES)
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in inputs.items())):
        parser.error("Frozen plan or prerequisites changed")
    boundary_plan = json.loads(inputs["boundary_grid_plan"].read_text())
    source_plan = json.loads(inputs["source_flux_plan"].read_text())
    calibration = json.loads(inputs["source_calibration_result"].read_text())
    grid_path = q.ROOT / boundary_plan["hamiltonian_grid_plan"]
    if (q.sha256(grid_path) != boundary_plan["hamiltonian_grid_plan_sha256"]
            or source_plan["adapter"]["sha256"] != plan["adapter_sha256"]):
        parser.error("Boundary states or source adapter changed")
    grid = json.loads(grid_path.read_text())
    if ([row["name"] for row in grid["frozen_states"]] != boundary_plan["rows"]
            or any(row["indices"] != boundary_plan["indices"]
                   for row in grid["frozen_states"])):
        parser.error("Boundary grid rows changed")
    choices = dict(calibration["flux_choice"],
                   **source_plan["source_placement_flux_choice_overrides"])
    libraries = _libraries(source_plan["pinned_wheel"])
    torch.set_num_threads(1)
    report = {
        "schema": "diffgw.seobnrv4.flux-boundary-modes.v1",
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
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-flux-modes-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            library_path = Path(tmp) / "flux_modes.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     inputs["adapter"], library_path,
                     source_plan["flux_source_includes"],
                     source_plan["flux_public_includes"])
            function = ctypes.CDLL(str(library_path)).diffgw_v4_mode22_same_state
            function.argtypes = [ctypes.c_double] * 9 + [ctypes.POINTER(ctypes.c_double)]
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
                    values = (ctypes.c_double * 39)()
                    rc = function(x1, x2, physical["chi1"], physical["chi2"],
                                  radius, 0., 0., pphi, omega, values)
                    if rc:
                        raise RuntimeError(f"{spec['name']} index {point['index']}: C rc={rc}")
                    source_total = values[3]
                    _, native_hreal, native_total, native_components = (
                        _native_mode_and_flux(physical, radius, 0., 0., pphi,
                                              omega, choices,
                                              source_plan["native_mode_choices"]))
                    source_components = list(values[4:])
                    if (not math.isfinite(source_total) or source_total <= 0.
                            or len(source_components) != len(MODES)
                            or len(native_components) != len(MODES)):
                        raise ValueError("Invalid source or native mode flux")
                    components = [{
                        "ell": ell, "m": m, "source_over_nu": source,
                        "native_over_nu": native,
                        "native_minus_source_relative_to_source_total":
                            (native - source) / source_total,
                    } for (ell, m), source, native in zip(
                        MODES, source_components, native_components)]
                    row["points"].append({
                        "index": point["index"], "r": radius, "pphi": pphi,
                        "omega": omega, "source_total_over_nu": source_total,
                        "native_total_over_nu": native_total,
                        "source_mode_sum_relative":
                            sum(source_components) / source_total - 1.,
                        "native_minus_source_total_relative":
                            (native_total - source_total) / source_total,
                        "native_minus_source_hreal_absolute":
                            native_hreal - values[2],
                        "components": components,
                        "max_abs_component_difference_relative_to_source_total":
                            max(abs(entry["native_minus_source_relative_to_source_total"])
                                for entry in components),
                    })
                q.write_evidence(args.json_out, report)
                print(spec["name"], max(
                    point["max_abs_component_difference_relative_to_source_total"]
                    for point in row["points"]), flush=True)
        points = [point for row in report["rows"] for point in row["points"]]
        report["max_abs_source_mode_sum_relative"] = max(
            abs(point["source_mode_sum_relative"]) for point in points)
        report["max_abs_native_minus_source_total_relative"] = max(
            abs(point["native_minus_source_total_relative"]) for point in points)
        report["max_abs_native_minus_source_hreal_absolute"] = max(
            abs(point["native_minus_source_hreal_absolute"]) for point in points)
        report["max_abs_component_difference_relative_to_source_total"] = max(
            point["max_abs_component_difference_relative_to_source_total"]
            for point in points)
        report["comparison_gates_passed"] = (
            report["max_abs_source_mode_sum_relative"]
            <= plan["source_mode_sum_max_relative_difference"]
            and report["max_abs_native_minus_source_total_relative"]
            <= plan["native_source_max_total_relative_difference"]
            and report["max_abs_component_difference_relative_to_source_total"]
            <= plan["native_source_max_component_difference_relative_to_total"]
            and report["max_abs_native_minus_source_hreal_absolute"]
            <= plan["native_source_max_hreal_absolute_difference"])
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

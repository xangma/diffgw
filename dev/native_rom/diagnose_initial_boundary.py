"""Dev-only pinned-source/native initial conditions on a frozen physical grid."""

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
from .diagnose_source_calibration_initial import _state
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .eob.torch.historical_hamiltonian import HistoricalLogDerivative
from .eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian,
)
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/initial_boundary_plan.json"
FIELDS = ("r", "pr_star", "pphi")


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
        "source_flux_plan", "hamiltonian_grid_plan",
        "source_calibration_initial_plan", "adapter")}
    if (plan["schema"] != "diffgw.seobnrv4.initial-boundary-plan.v1"
            or plan["release_qualified"] is not False
            or plan["frequencies_Momega"] != [0.01, 0.02]
            or plan["native_choices"] != ["published", "source_convention"]
            or plan["source_choices"] != ["legacy", "optimized"]
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in inputs.items())):
        parser.error("Frozen plan or prerequisites changed")
    source_plan = json.loads(inputs["source_flux_plan"].read_text())
    grid = json.loads(inputs["hamiltonian_grid_plan"].read_text())
    calibration = json.loads(inputs["source_calibration_initial_plan"].read_text())
    if [row["name"] for row in grid["frozen_states"]] != plan["rows"]:
        parser.error("Frozen physical rows changed")
    flux_choice = dict(calibration["flux_choice"],
                       **source_plan["source_placement_flux_choice_overrides"])
    flux = SEOBNRv4PublishedFlux(**flux_choice)
    libraries = _libraries(source_plan["pinned_wheel"])
    torch.set_num_threads(1)
    report = {
        "schema": "diffgw.seobnrv4.initial-boundary.v1",
        "status": "running", "release_qualified": False, "scope": q.SCOPE,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in inputs.items()},
        "native_source_sha256": {key: q.sha256(q.ROOT / key) for key in (
            "src/torchwave/approximants/eob/initial_conditions.py",
            "src/torchwave/approximants/eob/seobnrv4_flux.py",
            "dev/native_rom/eob/torch/source_calibration_hamiltonian.py")},
        "flux_choice": flux_choice,
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "points": [], "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-initial-boundary-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            library_path = Path(tmp) / "initial.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     inputs["adapter"], library_path,
                     source_plan["flux_source_includes"],
                     source_plan["flux_public_includes"])
            function = ctypes.CDLL(str(library_path)).diffgw_v4_initial_conditions
            function.argtypes = [ctypes.c_double] * 5 + [
                ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            for spec in grid["frozen_states"]:
                row = spec["physical"]
                for omega in plan["frequencies_Momega"]:
                    source = {}
                    for choice, optimized in (("legacy", 0), ("optimized", 1)):
                        values = (ctypes.c_double * 3)()
                        rc = function(row["mass1"], row["mass2"], row["chi1"],
                                      row["chi2"], omega, optimized, values)
                        if rc:
                            raise RuntimeError(
                                f"Pinned source IC failed: {spec['name']} {omega}"
                                f" {choice}, code {rc}")
                        source[choice] = dict(zip(FIELDS, values))
                    native = {
                        "published": _state(row, omega, flux,
                                            HistoricalLogDerivative, 0.001),
                        "source_convention": _state(
                            row, omega, flux,
                            SourceCalibratedHistoricalHamiltonian, 0.001),
                    }
                    difference = {}
                    for name, state in native.items():
                        for source_name, source_state in source.items():
                            difference[f"{name}_minus_{source_name}"] = {
                                key: state[key] - source_state[key] for key in FIELDS}
                    relative = {
                        name: {key: abs(value) / max(
                            abs(source["legacy"][key]),
                            plan["relative_denominator_floor"])
                               for key, value in residual.items()}
                        for name, residual in difference.items()
                        if name.endswith("_minus_legacy")}
                    report["points"].append({
                        "name": spec["name"], "physical": row,
                        "frequency_Momega": omega, "source": source,
                        "native": native, "native_minus_source": difference,
                        "native_legacy_relative": relative,
                        "source_optimized_minus_legacy": {
                            key: source["optimized"][key] - source["legacy"][key]
                            for key in FIELDS},
                    })
                q.write_evidence(args.json_out, report)
                print(spec["name"], flush=True)
        report["max_source_convention_absolute_by_field"] = {
            reference: {key: max(abs(point["native_minus_source"][
                f"source_convention_minus_{reference}"][key])
                for point in report["points"]) for key in FIELDS}
            for reference in plan["source_choices"]}
        report["max_source_optimized_minus_legacy_absolute_by_field"] = {
            key: max(abs(point["source_optimized_minus_legacy"][key])
                     for point in report["points"]) for key in FIELDS}
        report["max_source_convention_relative_to_legacy"] = max(
            value for point in report["points"]
            for value in point["native_legacy_relative"][
                "source_convention_minus_legacy"].values())
        report["comparison_gate_passed"] = (
            report["max_source_convention_relative_to_legacy"]
            <= plan["relative_initial_state_gate"])
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

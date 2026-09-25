"""Dev-only boundary-row comparison of both historical initial-data paths."""

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
from .diagnose_initial_finite_probe import FIELDS
from .diagnose_source_calibration_initial import _state
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian,
)
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/initial_finite_probe_domain_plan.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("pinned wheel comparison requires macOS")
    plan = json.loads(PLAN.read_text())
    source_path = q.ROOT / plan["source_flux_plan"]
    calibration_path = q.ROOT / plan["source_calibration_initial_plan"]
    adapter_path = q.ROOT / plan["adapter"]
    if (plan["schema"] != "diffgw.seobnrv4.initial-finite-probe-domain-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_choices"] != ["legacy", "optimized"]
            or plan["radial_probe"] != 0.001
            or len(plan["rows"]) != 4
            or plan["frequencies_Momega"] != [0.012, 0.018]
            or q.sha256(Path(__file__)) != plan["script_sha256"]
            or q.sha256(source_path) != plan["source_flux_plan_sha256"]
            or q.sha256(calibration_path)
            != plan["source_calibration_initial_plan_sha256"]
            or q.sha256(adapter_path) != plan["adapter_sha256"]):
        parser.error("frozen plan or pinned input changed")
    source_plan = json.loads(source_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    choice = dict(calibration["flux_choice"], **source_plan[
        "source_placement_flux_choice_overrides"])
    flux = SEOBNRv4PublishedFlux(**choice)
    libraries = _libraries(source_plan["pinned_wheel"])
    torch.set_num_threads(1)
    report = {
        "schema": "diffgw.seobnrv4.initial-finite-probe-domain.v1",
        "status": "running", "release_qualified": False, "scope": q.SCOPE,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "native_source_sha256": {name: q.sha256(q.ROOT / name) for name in (
            "src/torchwave/approximants/eob/initial_conditions.py",
            "src/torchwave/approximants/eob/seobnrv4_flux.py",
            "dev/native_rom/eob/torch/source_calibration_hamiltonian.py")},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "points": [], "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-initial-domain-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            library_path = Path(tmp) / "initial.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     adapter_path, library_path,
                     source_plan["flux_source_includes"],
                     source_plan["flux_public_includes"])
            function = ctypes.CDLL(str(library_path)).diffgw_v4_initial_conditions
            function.argtypes = [ctypes.c_double] * 5 + [
                ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            for row in plan["rows"]:
                for omega in plan["frequencies_Momega"]:
                    native = _state(row, omega, flux,
                                    SourceCalibratedHistoricalHamiltonian,
                                    plan["radial_probe"])
                    finite = {**native,
                              "pr_star": native["finite_radial_response_pr_star"]}
                    sources = {}
                    for source_choice in plan["source_choices"]:
                        values = (ctypes.c_double * 3)()
                        rc = function(row["mass1"], row["mass2"], row["chi1"],
                                      row["chi2"], omega,
                                      int(source_choice == "optimized"), values)
                        if rc:
                            raise RuntimeError(
                                f"{source_choice} initial state failed: "
                                f"{row['name']} {omega}, code {rc}")
                        source = dict(zip(FIELDS, values))
                        sources[source_choice] = {
                            "source": source,
                            "relative_difference": {key:
                                abs(finite[key]-source[key]) /
                                max(abs(source[key]),
                                    plan["relative_denominator_floor"])
                                for key in FIELDS},
                            "analytic_radial_absolute_difference":
                                abs(native["pr_star"]-source["pr_star"]),
                            "finite_radial_absolute_difference":
                                abs(finite["pr_star"]-source["pr_star"]),
                        }
                    report["points"].append({
                        "name": row["name"], "physical": row,
                        "frequency_Momega": omega,
                        "native_analytic": {key: native[key] for key in FIELDS},
                        "native_finite_probe": {key: finite[key] for key in FIELDS},
                        "finite_over_analytic_radial_slope":
                            native["finite_over_analytic_radial_slope"],
                        "sources": sources,
                    })
                q.write_evidence(args.json_out, report)
                print(row["name"], flush=True)
        report["max_relative_difference"] = {
            choice: max(value for point in report["points"]
                        for value in point["sources"][choice][
                            "relative_difference"].values())
            for choice in plan["source_choices"]}
        report["optimized_gate_passed"] = (
            report["max_relative_difference"]["optimized"]
            <= plan["optimized_relative_gate"])
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__,
                                 "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({key: report.get(key) for key in (
        "status", "optimized_gate_passed", "max_relative_difference", "errors")}),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

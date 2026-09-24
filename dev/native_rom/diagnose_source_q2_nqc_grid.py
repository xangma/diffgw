"""Measure q2 NQC coefficient sensitivity in pinned source and public wheel."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import traceback

import numpy as np

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_wheel_mode import _public_nqc, _source_nqc


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_q2_nqc_grid_plan.json"


def _comparison(candidate: np.ndarray, reference: np.ndarray) -> dict:
    delta = candidate-reference
    return dict(relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(reference)),
                max_abs=float(np.max(np.abs(delta))), delta=delta.tolist())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / path for name, path in plan["inputs"].items()}
    row = dict(mass1=30.0, mass2=15.0, chi1=0.4, chi2=-0.3, omega_start=0.04)
    active = [0, 1, 2, 6, 7]
    if (plan["schema"] != "diffgw.seobnrv4.source-q2-nqc-grid-plan.v1"
            or plan["row"] != row or plan["output_steps_M"] != [0.5, 0.25]
            or plan["active_indices"] != active
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen q2 NQC-grid plan or inputs differ")
    reference_grid = json.loads(paths["reference_grid"].read_text())
    prior = json.loads(paths["q2_probe"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (reference_grid["status"] != "diagnostic_complete"
            or prior["status"] != "diagnostic_complete"
            or reference_grid["row"] != row or prior["row"] != row
            or q.lal_binary_identity(prior["lal_binary"]["lal_version"],
                                     prior["lal_binary"]["lalsimulation_version"])
               != prior["lal_binary"]):
        parser.error("pinned q2 reference or wheel binary differs")
    scope = dict(q.SCOPE)
    scope["description"] = "Full-generator q2 NQC fits at two requested output steps."
    scope["known_limitations"] = [
        "Coefficient differences alone do not explain a whole-mode residual without shared-state evaluation.",
        "This selected q2 row and two output steps cannot qualify the parameter domain or ROM.",
    ]
    report = dict(schema="diffgw.seobnrv4.source-q2-nqc-grid.v1",
                  status="running", release_qualified=False, scope=scope,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], row=row, coefficients={},
                  comparisons={}, errors=[], lal_binary=prior["lal_binary"],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-q2-nqc-grid-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            dylib = Path(tmp) / "source_q2_nqc.dylib"
            _compile(private, args.headers, args.gsl_include,
                     _libraries(source_plan["pinned_wheel"]),
                     paths["mode_adapter"], dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            function = ctypes.CDLL(str(dylib)).diffgw_v4_full_source_nqc
            function.argtypes = [ctypes.c_double]*6 + [ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            values = {}
            for step in plan["output_steps_M"]:
                key = str(step)
                source = _source_nqc(function, row, step)
                wheel = _public_nqc(row, step)
                if not (np.isfinite(source).all() and np.isfinite(wheel).all()):
                    raise ValueError("non-finite q2 NQC coefficients")
                values[key] = dict(source=source[active], wheel=wheel[active])
                report["coefficients"][key] = dict(
                    source_all=source.tolist(), wheel_all=wheel.tolist(),
                    source_active=source[active].tolist(),
                    wheel_active=wheel[active].tolist())
                q.write_evidence(args.json_out, report)
                print(f"step={key} complete", flush=True)
        native = np.array([*prior["native"]["nqc_coefficients"]["amplitude"],
                           *prior["native"]["nqc_coefficients"]["phase"]])
        report["coefficients"]["native_0.5_active"] = native.tolist()
        report["comparisons"] = dict(
            source_fine_against_coarse=_comparison(values["0.25"]["source"],
                                                   values["0.5"]["source"]),
            wheel_fine_against_coarse=_comparison(values["0.25"]["wheel"],
                                                  values["0.5"]["wheel"]),
            wheel_against_source_coarse=_comparison(values["0.5"]["wheel"],
                                                    values["0.5"]["source"]),
            wheel_against_source_fine=_comparison(values["0.25"]["wheel"],
                                                  values["0.25"]["source"]),
            native_against_source_coarse=_comparison(native,
                                                      values["0.5"]["source"]))
        report.update(elapsed_seconds=time.monotonic()-started,
                      status="diagnostic_complete")
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc),
                                     traceback=traceback.format_exc()))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

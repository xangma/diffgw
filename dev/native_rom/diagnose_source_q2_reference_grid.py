"""Compare q2 source and wheel full modes across two requested output steps."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import traceback

import lal
import numpy as np

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_q4_q8_mode import _comparison
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_q2_reference_grid_plan.json"


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
    windows = [dict(name="full_inspiral", start_M=-1000.0, end_M=-120.0),
               dict(name="late_inspiral", start_M=-120.0, end_M=-30.0),
               dict(name="merger_neighborhood", start_M=-30.0, end_M=20.0),
               dict(name="ringdown", start_M=20.0, end_M=100.0)]
    if (plan["schema"] != "diffgw.seobnrv4.source-q2-reference-grid-plan.v1"
            or plan["row"] != row or plan["output_steps_M"] != [0.5, 0.25]
            or plan["phase_anchor_M"] != -100.0
            or plan["windows_M"] != windows or plan["capacity"] != 50000
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen q2 reference-grid plan or inputs differ")
    prior = json.loads(paths["q2_probe"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    native_grid = json.loads(paths["q2_native_grid"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or native_grid["status"] != "diagnostic_complete"
            or native_grid["grid_gate_passed"] is not True
            or prior["row"] != row or native_grid["row"] != row
            or q.lal_binary_identity(prior["lal_binary"]["lal_version"],
                                     prior["lal_binary"]["lalsimulation_version"])
               != prior["lal_binary"]):
        parser.error("pinned q2 predecessor or wheel binary differs")
    scope = dict(q.SCOPE)
    scope["description"] = "q2 source and wheel full-mode sensitivity to requested output step."
    scope["known_limitations"] = [
        "Peak-centered clocks and one early phase gauge remove two conventions, not physical differences.",
        "Changing the requested step can change a reference binary's trajectory and NQC fit.",
        "These selected two grids do not qualify source parity, the native generator, or a ROM.",
    ]
    report = dict(schema="diffgw.seobnrv4.source-q2-reference-grid.v1",
                  status="running", release_qualified=False, scope=scope,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], row=row, modes={},
                  comparisons={}, errors=[], lal_binary=prior["lal_binary"],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
        samples = {}
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-q2-ref-grid-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            dylib = Path(tmp) / "source_q2_mode.dylib"
            _compile(private, args.headers, args.gsl_include,
                     _libraries(source_plan["pinned_wheel"]),
                     paths["mode_adapter"], dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            function = ctypes.CDLL(str(dylib)).diffgw_v4_full_source_mode
            function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            for step in plan["output_steps_M"]:
                key = str(step)
                source_epoch, source_dt, source_mode = _source(
                    function, row, step, plan["capacity"])
                wheel_epoch, wheel_dt, wheel_mode, _ = _public(row, step)
                if (not math.isclose(source_dt/mass_seconds, step, rel_tol=0, abs_tol=1e-10)
                        or not math.isclose(wheel_dt/mass_seconds, step, rel_tol=0,
                                            abs_tol=1e-10)):
                    raise ValueError(f"reference output step differs: {key}")
                source_t = (source_epoch+np.arange(source_mode.size)*source_dt)/mass_seconds
                wheel_t = (wheel_epoch+np.arange(wheel_mode.size)*wheel_dt)/mass_seconds
                source_hash = hashlib.sha256(source_mode.tobytes()).hexdigest()
                wheel_hash = hashlib.sha256(wheel_mode.tobytes()).hexdigest()
                if step == 0.5 and (source_hash != prior["source_mode_sha256"]
                                    or wheel_hash != prior["wheel_mode_sha256"]):
                    raise ValueError("q2 source or wheel baseline mode changed")
                samples[key] = dict(source=(source_t, source_mode),
                                    wheel=(wheel_t, wheel_mode))
                report["modes"][key] = dict(
                    source=dict(samples=int(source_mode.size), epoch_s=source_epoch,
                                dt_s=source_dt, sha256=source_hash),
                    wheel=dict(samples=int(wheel_mode.size), epoch_s=wheel_epoch,
                               dt_s=wheel_dt, sha256=wheel_hash))
                q.write_evidence(args.json_out, report)
                print(f"step={key} complete", flush=True)
        anchor = plan["phase_anchor_M"]
        coarse, fine = samples["0.5"], samples["0.25"]
        report["comparisons"] = dict(
            source_fine_against_coarse=_comparison(
                *coarse["source"], *fine["source"], anchor, windows),
            wheel_fine_against_coarse=_comparison(
                *coarse["wheel"], *fine["wheel"], anchor, windows),
            wheel_against_source_fine=_comparison(
                *fine["source"], *fine["wheel"], anchor, windows))
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

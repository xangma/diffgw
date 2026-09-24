"""Compare a frozen new q2 native mode with source and public wheel modes."""

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
import torch

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_q4_q8_mode import _comparison, _run
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_q2_domain_probe_plan.json"


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
    level = dict(name="ode_3", atol=3e-15, rtol=3e-14, dt_geom=0.5)
    windows = [dict(name="full_inspiral", start_M=-1000.0, end_M=-120.0),
               dict(name="late_inspiral", start_M=-120.0, end_M=-30.0),
               dict(name="merger_neighborhood", start_M=-30.0, end_M=20.0),
               dict(name="ringdown", start_M=20.0, end_M=100.0)]
    if (plan["schema"] != "diffgw.seobnrv4.source-q2-domain-probe-plan.v1"
            or plan["row"] != row or plan["level"] != level
            or plan["restart_time_M"] != 100.0
            or plan["output_step_M"] != 0.5
            or plan["phase_anchor_M"] != -100.0
            or plan["windows_M"] != windows
            or plan["capacity"] != 50000
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen q2 domain plan or inputs differ")
    campaign = json.loads(paths["campaign"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (campaign["status"] != "diagnostic_complete"
            or q.lal_binary_identity(campaign["lal_binary"]["lal_version"],
                                     campaign["lal_binary"]["lalsimulation_version"])
               != campaign["lal_binary"]):
        parser.error("pinned campaign or wheel binary differs")
    scope = dict(q.SCOPE)
    scope["description"] = "Predeclared q2 development probe outside the q4/q8 selection set."
    scope["known_limitations"] = [
        "This is a development probe, not an untouched release holdout.",
        "The native row is not independently numerically converged here.",
        "An early phase gauge is removed; no time or amplitude fit is applied.",
        "One row cannot qualify the physical parameter domain or ROM.",
    ]
    report = dict(schema="diffgw.seobnrv4.source-q2-domain-probe.v1",
                  status="running", release_qualified=False, scope=scope,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], row=row, level=level,
                  lal_binary=campaign["lal_binary"], errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-q2-mode-") as tmp:
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
            epoch, source_dt, source_mode = _source(
                function, row, plan["output_step_M"], plan["capacity"])
            report["source_mode_sha256"] = hashlib.sha256(source_mode.tobytes()).hexdigest()
            q.write_evidence(args.json_out, report)
            print("source_complete", flush=True)
        settings = q.Settings(**campaign["settings"])
        native = _run(row, level, settings, campaign["choice"], plan["restart_time_M"])
        native_summary = q.summarize_run(native)
        report["native"] = native_summary
        q.write_evidence(args.json_out, report)
        print("native_complete", flush=True)
        wheel_epoch, wheel_dt, wheel_mode, _ = _public(row, plan["output_step_M"])
        mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
        if (not math.isclose(source_dt/mass_seconds, plan["output_step_M"],
                             rel_tol=0, abs_tol=1e-10)
                or not math.isclose(wheel_dt/mass_seconds, plan["output_step_M"],
                                    rel_tol=0, abs_tol=1e-10)):
            raise ValueError("source or wheel output step differs")
        source_t = (epoch+np.arange(source_mode.size)*source_dt)/mass_seconds
        native_t = native["time"]-native["peak_time_M"]
        wheel_t = (wheel_epoch+np.arange(wheel_mode.size)*wheel_dt)/mass_seconds
        native_h = native["mode"]*mass_seconds*lal.C_SI/(100e6*lal.PC_SI)
        report.update(
            source_samples=int(source_mode.size),
            wheel_samples=int(wheel_mode.size),
            wheel_mode_sha256=hashlib.sha256(wheel_mode.tobytes()).hexdigest(),
            source_epoch_s=epoch, wheel_epoch_s=wheel_epoch,
            native_against_source=_comparison(
                source_t, source_mode, native_t, native_h,
                plan["phase_anchor_M"], windows),
            wheel_against_source=_comparison(
                source_t, source_mode, wheel_t, wheel_mode,
                plan["phase_anchor_M"], windows),
            elapsed_seconds=time.monotonic()-started,
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

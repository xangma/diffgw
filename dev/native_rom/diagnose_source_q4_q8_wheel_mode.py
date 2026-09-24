"""Measure pinned-source versus public-wheel q4/q8 full-mode reference drift."""

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

import numpy as np
import lal

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_q4_q8_mode import _comparison
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_q4_q8_wheel_mode_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / path for name, path in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-q8-wheel-mode-plan.v1"
            or plan["rows"] != ["q4_mixed_spin", "q8_high_aligned"]
            or plan["output_step_M"] != 0.5 or plan["phase_anchor_M"] != -100.
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen source/wheel plan or inputs differ")
    campaign = json.loads(paths["campaign"].read_text())
    source = json.loads(paths["source_attachment"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (campaign["status"] != "diagnostic_complete"
            or source["status"] != "diagnostic_complete"
            or q.lal_binary_identity(campaign["lal_binary"]["lal_version"],
                                     campaign["lal_binary"]["lalsimulation_version"])
               != campaign["lal_binary"]):
        parser.error("pinned source or public binary differs")
    scope = dict(q.SCOPE)
    scope["description"] = ("Pinned separately compiled source versus public wheel "
                            "on a common orbital-peak chart and fixed phase gauge.")
    scope["known_limitations"] = [
        item for item in q.SCOPE["known_limitations"]
        if item != "No time, phase, amplitude, or coefficient fitting is performed."]
    scope["known_limitations"].append(
        "The fixed early-time phase anchor removes the arbitrary initial gauge; "
        "the wheel orbital-peak chart follows the pinned source epoch convention.")
    report = dict(schema="diffgw.seobnrv4.source-q4-q8-wheel-mode.v1",
                  scope=scope, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv),
                  rows={}, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-q4-q8-wheel-mode-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            dylib = Path(tmp) / "source_q4_q8_wheel_mode.dylib"
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
            for name in plan["rows"]:
                row = campaign["rows"][name]["row"]
                mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
                source_epoch, source_dt, source_mode = _source(
                    function, row, plan["output_step_M"], plan["capacity"])
                wheel_epoch, wheel_dt, wheel_mode, _ = _public(
                    row, plan["output_step_M"])
                if (not math.isclose(source_dt/mass_seconds, 0.5, abs_tol=1e-10)
                        or not math.isclose(wheel_dt/mass_seconds, 0.5, abs_tol=1e-10)):
                    raise ValueError(f"{name} source/wheel step differs")
                source_t = (source_epoch+np.arange(source_mode.size)*source_dt)/mass_seconds
                wheel_t = (wheel_epoch+np.arange(wheel_mode.size)*wheel_dt)/mass_seconds
                boundary = source["rows"][name]["peak_relative"][
                    "source_boundary_from_peak_M"]
                windows = [dict(name="early", start_M=-120., end_M=-60.),
                           dict(name="late_inspiral", start_M=-60., end_M=boundary),
                           dict(name="attachment", start_M=boundary, end_M=boundary+30.),
                           dict(name="ringdown", start_M=boundary+30., end_M=100.)]
                comparison = _comparison(source_t, source_mode, wheel_t,
                                         wheel_mode, plan["phase_anchor_M"], windows)
                report["rows"][name] = dict(row=row,
                    source_mode_sha256=hashlib.sha256(source_mode.tobytes()).hexdigest(),
                    wheel_mode_sha256=hashlib.sha256(wheel_mode.tobytes()).hexdigest(),
                    source_minus_wheel_epoch_M=(source_epoch-wheel_epoch)/mass_seconds,
                    comparison=comparison)
                q.write_evidence(args.json_out, report)
                print(f"compared={name}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

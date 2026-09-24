"""Locate q8 native/source/wheel phase drift on the common orbital clock."""

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

import lal
import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_q4_q8_mode import _run
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_q8_phase_profile_plan.json"


def phase_at(times: np.ndarray, mode: np.ndarray, probes: np.ndarray) -> np.ndarray:
    if (not np.all(np.diff(times) > 0) or not np.isfinite(mode).all()
            or probes[0] < times[0] or probes[-1] > times[-1]):
        raise ValueError("phase probes outside valid mode support")
    return np.interp(probes, times, np.unwrap(np.angle(mode)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / path for name, path in plan["inputs"].items()}
    probes = np.asarray(plan["probe_times_M"], dtype=float)
    if (plan["schema"] != "diffgw.seobnrv4.source-q8-phase-profile-plan.v1"
            or plan["row"] != "q8_high_aligned" or plan["level"] != "ode_2"
            or plan["output_step_M"] != 0.5 or plan["phase_anchor_M"] != -100.
            or probes.tolist() != [-1100., -1000., -800., -600., -400., -200.,
                                   -120., -100., -80., -60., -40., -25.,
                                   -15., -5., 0., 10., 30., 60., 90.]
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen phase-profile plan or inputs differ")
    campaign = json.loads(paths["campaign"].read_text())
    previous = json.loads(paths["source_native_mode"].read_text())
    wheel_previous = json.loads(paths["source_wheel_mode"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    row = campaign["rows"][plan["row"]]["row"]
    if (campaign["status"] != "diagnostic_complete"
            or previous["status"] != "diagnostic_complete"
            or wheel_previous["status"] != "diagnostic_complete"
            or q.lal_binary_identity(campaign["lal_binary"]["lal_version"],
                                     campaign["lal_binary"]["lalsimulation_version"])
               != campaign["lal_binary"]):
        parser.error("pinned native/source/wheel references differ")
    torch.set_num_threads(1)
    scope = dict(q.SCOPE)
    scope["description"] = (
        "Diagnostic source/native/wheel phase profile; does not qualify native physics.")
    scope["known_limitations"] = [
        item for item in q.SCOPE["known_limitations"]
        if item != "No time, phase, amplitude, or coefficient fitting is performed."]
    scope["known_limitations"].append(
        "One predeclared early-time phase anchor removes the arbitrary phase gauge; "
        "there is no time, amplitude, or phase-slope fit.")
    report = dict(schema="diffgw.seobnrv4.source-q8-phase-profile.v1",
                  status="running", release_qualified=False, scope=scope,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], row=row,
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-q8-phase-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            dylib = Path(tmp) / "source_q8_phase.dylib"
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
            start = time.monotonic()
            item = campaign["rows"][plan["row"]]
            native = _run(row, item["runs"]["ode_2"]["level"],
                          q.Settings(**campaign["settings"]), campaign["choice"],
                          item["fixed_restart_time_M"])
            native_hash = q.summarize_run(native)["array_sha256"]
            if native_hash != previous["rows"][plan["row"]]["native_array_sha256"]:
                raise ValueError("native replay changed")
            source_epoch, source_dt, source_mode = _source(
                function, row, plan["output_step_M"], plan["capacity"])
            wheel_epoch, wheel_dt, wheel_mode, _ = _public(
                row, plan["output_step_M"])
            if (hashlib.sha256(source_mode.tobytes()).hexdigest()
                    != previous["rows"][plan["row"]]["source_mode_sha256"]
                    or hashlib.sha256(wheel_mode.tobytes()).hexdigest()
                    != wheel_previous["rows"][plan["row"]]["wheel_mode_sha256"]):
                raise ValueError("source or wheel mode changed")
            mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
            if (not math.isclose(source_dt/mass_seconds, 0.5, rel_tol=0, abs_tol=1e-10)
                    or not math.isclose(wheel_dt/mass_seconds, 0.5,
                                        rel_tol=0, abs_tol=1e-10)):
                raise ValueError("source or wheel output step changed")
            native_t = native["time"]-native["peak_time_M"]
            source_t = (source_epoch+np.arange(source_mode.size)*source_dt)/mass_seconds
            wheel_t = (wheel_epoch+np.arange(wheel_mode.size)*wheel_dt)/mass_seconds
            phase = {"native": phase_at(native_t, native["mode"], probes),
                     "source": phase_at(source_t, source_mode, probes),
                     "wheel": phase_at(wheel_t, wheel_mode, probes)}
            amplitude = {
                "native": np.interp(probes, native_t, np.abs(native["mode"]))
                          * mass_seconds*lal.C_SI/(100e6*lal.PC_SI),
                "source": np.interp(probes, source_t, np.abs(source_mode)),
                "wheel": np.interp(probes, wheel_t, np.abs(wheel_mode))}
            anchor_index = int(np.flatnonzero(probes == plan["phase_anchor_M"])[0])
            pairs = {}
            for candidate, reference in (("native", "source"),
                                         ("wheel", "source"),
                                         ("native", "wheel")):
                difference = phase[candidate]-phase[reference]
                difference -= difference[anchor_index]
                name = f"{candidate}_minus_{reference}"
                pairs[name] = dict(phase_from_anchor_rad=difference.tolist(),
                                   amplitude_ratio=(amplitude[candidate]
                                                    /amplitude[reference]).tolist())
            if not all(0.9 < value < 1.1 for value in
                       pairs["native_minus_source"]["amplitude_ratio"]):
                raise ValueError("native/source amplitude units or support differ")
            closure = (np.asarray(pairs["native_minus_source"]["phase_from_anchor_rad"])
                       - np.asarray(pairs["native_minus_wheel"]["phase_from_anchor_rad"])
                       - np.asarray(pairs["wheel_minus_source"]["phase_from_anchor_rad"]))
            report.update(probe_times_M=probes.tolist(), pairs=pairs,
                          phase_closure_max_abs_rad=float(np.max(np.abs(closure))),
                          elapsed_seconds=time.monotonic()-start,
                          native_array_sha256=native_hash,
                          source_mode_sha256=hashlib.sha256(source_mode.tobytes()).hexdigest(),
                          wheel_mode_sha256=hashlib.sha256(wheel_mode.tobytes()).hexdigest())
            if report["phase_closure_max_abs_rad"] > 1e-12:
                raise ValueError("phase decomposition failed closure")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

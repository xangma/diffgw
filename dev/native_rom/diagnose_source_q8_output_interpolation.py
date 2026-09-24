"""Audit q8 output-phase interpolation on pinned native and source modes."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time

import lal
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
import torch

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_q4_q8_mode import _run
from .diagnose_source_wheel_mode import _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_q8_output_interpolation_plan.json"


def _anchored(values, anchor):
    values = np.asarray(values)
    return values-values[anchor]


def _phase_interpolants(times, mode, probes, anchor):
    if (not np.all(np.diff(times) > 0) or probes[0] < times[0]
            or probes[-1] > times[-1] or not np.isfinite(mode).all()):
        raise ValueError("invalid output mode or phase-probe support")
    phase = np.unwrap(np.angle(mode))
    return {
        "linear": _anchored(np.interp(probes, times, phase), anchor),
        "cubic": _anchored(CubicSpline(times, phase)(probes), anchor),
        "pchip": _anchored(PchipInterpolator(times, phase)(probes), anchor),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT/path for name, path in plan["inputs"].items()}
    probes = np.asarray(plan["probe_times_M"], dtype=float)
    if (plan["schema"] != "diffgw.seobnrv4.source-q8-output-interpolation-plan.v1"
            or plan["row"] != "q8_high_aligned" or plan["level"] != "ode_2"
            or plan["output_step_M"] != 0.5
            or probes.tolist() != [-120., -100., -80., -60., -40., -25.]
            or plan["phase_anchor_M"] != -100.
            or plan["interpolants"] != ["linear", "cubic", "pchip"]
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen output-interpolation plan or inputs differ")
    campaign = json.loads(paths["campaign"].read_text())
    raw = json.loads(paths["raw_phase"].read_text())
    previous = json.loads(paths["source_native_mode"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    row = campaign["rows"][plan["row"]]["row"]
    if (any(x["status"] != "diagnostic_complete"
            for x in (campaign, raw, previous))
            or raw["row"] != row or raw["probe_times_M"] != probes.tolist()
            or q.lal_binary_identity(campaign["lal_binary"]["lal_version"],
                                     campaign["lal_binary"]["lalsimulation_version"])
               != campaign["lal_binary"]):
        parser.error("pinned q8 evidence or wheel changed")
    anchor = int(np.flatnonzero(probes == plan["phase_anchor_M"])[0])
    scope = dict(q.SCOPE)
    scope["description"] = "Post-hoc interpolation-order control on one q8 row."
    scope["known_limitations"] = [
        "Cubic and PCHIP controls were selected after the raw-phase residual was observed.",
        "Smooth output interpolation is not an independent waveform reference.",
        "This one-row diagnostic cannot qualify generator or ROM physics.",
    ]
    report = dict(schema="diffgw.seobnrv4.source-q8-output-interpolation.v1",
                  status="running", release_qualified=False,
                  scope=scope,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], row=row,
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), errors=[])
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        torch.set_num_threads(1)
        item = campaign["rows"][plan["row"]]
        native = _run(row, item["runs"][plan["level"]]["level"],
                      q.Settings(**campaign["settings"]), campaign["choice"],
                      item["fixed_restart_time_M"])
        native_hash = q.summarize_run(native)["array_sha256"]
        if native_hash != previous["rows"][plan["row"]]["native_array_sha256"]:
            raise ValueError("native q8 output mode changed")
        report["native_array_sha256"] = native_hash
        q.write_evidence(args.json_out, report)
        print("native_q8_replayed", flush=True)
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-q8-output-interp-") as tmp:
            private = Path(tmp)/"source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            dylib = Path(tmp)/"source_mode.dylib"
            _compile(private, args.headers, args.gsl_include,
                     _libraries(source_plan["pinned_wheel"]), paths["mode_adapter"],
                     dylib, source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            pointer = ctypes.POINTER(ctypes.c_double)
            function = ctypes.CDLL(str(dylib)).diffgw_v4_full_source_mode
            function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                                ctypes.POINTER(ctypes.c_size_t), pointer,
                                pointer, pointer]
            function.restype = ctypes.c_int
            source_epoch, source_dt, source_mode = _source(
                function, row, plan["output_step_M"], plan["capacity"])
            source_hash = hashlib.sha256(source_mode.tobytes()).hexdigest()
            if source_hash != previous["rows"][plan["row"]]["source_mode_sha256"]:
                raise ValueError("source q8 output mode changed")
            mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
            source_t = (source_epoch+np.arange(source_mode.size)*source_dt)/mass_seconds
            native_t = native["time"]-native["peak_time_M"]
            modes = {
                "source": _phase_interpolants(source_t, source_mode, probes, anchor),
                "native": _phase_interpolants(native_t, native["mode"], probes, anchor),
            }
            observed = np.asarray(raw["observed_phase_rad"])
            modeled = np.asarray(raw["modeled_phase_rad"])
            if np.max(abs(modes["native"]["linear"]
                          -modes["source"]["linear"]-observed)) > 1e-10:
                raise ValueError("linear output phase differs from pinned raw audit")
            results = {}
            for method in plan["interpolants"]:
                source_bias = modes["source"]["linear"]-modes["source"][method]
                native_bias = modes["native"]["linear"]-modes["native"][method]
                source_error = np.asarray(raw["source_reconstruction_error_rad"])-source_bias
                native_error = np.asarray(raw["native_reconstruction_error_rad"])-native_bias
                observed_method = modes["native"][method]-modes["source"][method]
                residual = observed_method-modeled
                if np.max(abs(residual-(native_error-source_error))) > 1e-10:
                    raise ValueError(f"{method} reconstruction closure failed")
                results[method] = dict(
                    source_linear_minus_method_rad=source_bias.tolist(),
                    native_linear_minus_method_rad=native_bias.tolist(),
                    source_reconstruction_error_rad=source_error.tolist(),
                    native_reconstruction_error_rad=native_error.tolist(),
                    observed_phase_rad=observed_method.tolist(),
                    residual_phase_rad=residual.tolist(),
                    residual_max_abs_rad=float(np.max(abs(residual))),
                    residual_rms_rad=float(np.sqrt(np.mean(residual**2))))
            report.update(probe_times_M=probes.tolist(),
                          source_mode_sha256=source_hash, results=results,
                          elapsed_seconds=time.monotonic()-started)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

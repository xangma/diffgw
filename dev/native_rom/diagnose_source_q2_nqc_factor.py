"""Evaluate q2 source/wheel NQC coefficient differences on shared public states."""

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
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_q2_nqc_factor_plan.json"


def _nqc_factor(coefficients: np.ndarray, r: np.ndarray,
                pr: np.ndarray, omega: np.ndarray) -> np.ndarray:
    ratio = pr/(r*omega)
    amplitude = 1+ratio**2*(coefficients[0]+coefficients[1]/r
                            +coefficients[2]/r**1.5)
    phase = coefficients[6]*ratio+coefficients[7]*ratio*pr**2
    return amplitude*np.exp(1j*phase)


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
    if (plan["schema"] != "diffgw.seobnrv4.source-q2-nqc-factor-plan.v1"
            or plan["row"] != row or plan["output_steps_M"] != [0.5, 0.25]
            or plan["prepeak_cutoff_M"] != -25.0
            or plan["active_indices"] != [0, 1, 2, 6, 7]
            or plan["capacity"] != 50000
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen q2 NQC-factor plan or inputs differ")
    nqc_grid = json.loads(paths["nqc_grid"].read_text())
    reference_grid = json.loads(paths["reference_grid"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (nqc_grid["status"] != "diagnostic_complete"
            or reference_grid["status"] != "diagnostic_complete"
            or nqc_grid["row"] != row or reference_grid["row"] != row
            or nqc_grid["lal_binary"] != reference_grid["lal_binary"]
            or q.lal_binary_identity(nqc_grid["lal_binary"]["lal_version"],
                                     nqc_grid["lal_binary"]["lalsimulation_version"])
               != nqc_grid["lal_binary"]):
        parser.error("pinned q2 predecessors or wheel binary differ")
    scope = dict(q.SCOPE)
    scope["description"] = "Same-public-state coefficient-only NQC counterfactual on q2."
    scope["known_limitations"] = [
        "The source and wheel complete modes may differ in trajectory and pre-NQC amplitude/phase.",
        "Public low-rate state interpolation and derivative sampling are not exact hidden high-rate fit inputs.",
        "A coefficient-only prediction cannot establish native physical or ROM qualification.",
    ]
    report = dict(schema="diffgw.seobnrv4.source-q2-nqc-factor.v1",
                  status="running", release_qualified=False, scope=scope,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], row=row, steps={}, errors=[],
                  lal_binary=nqc_grid["lal_binary"],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-q2-nqc-factor-") as tmp:
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
            mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
            for step in plan["output_steps_M"]:
                key = str(step)
                source_epoch, source_dt, source_mode = _source(
                    function, row, step, plan["capacity"])
                wheel_epoch, wheel_dt, wheel_mode, low = _public(row, step)
                stored = reference_grid["modes"][key]
                source_hash = hashlib.sha256(source_mode.tobytes()).hexdigest()
                wheel_hash = hashlib.sha256(wheel_mode.tobytes()).hexdigest()
                if (source_hash != stored["source"]["sha256"]
                    or wheel_hash != stored["wheel"]["sha256"]
                    or not math.isclose(source_epoch, wheel_epoch, rel_tol=0,
                                        abs_tol=1e-12)
                    or source_dt != wheel_dt
                    or not math.isclose(source_dt/mass_seconds, step, rel_tol=0,
                                        abs_tol=1e-10)):
                    raise ValueError(f"q2 reference modes or clocks changed: {key}; "
                                     f"source={source_hash}, wheel={wheel_hash}, "
                                     f"source_epoch={source_epoch}, wheel_epoch={wheel_epoch}, "
                                     f"source_dt={source_dt}, wheel_dt={wheel_dt}")
                source_coeff = np.asarray(
                    nqc_grid["coefficients"][key]["source_all"], dtype=float)
                wheel_coeff = np.asarray(
                    nqc_grid["coefficients"][key]["wheel_all"], dtype=float)
                orbital_omega = np.gradient(low[2], step, edge_order=2)
                orbital_peak = int(np.argmax(orbital_omega))
                mode_peak = int(np.argmax(np.abs(wheel_mode)))
                offset_real = orbital_peak+wheel_epoch/wheel_dt
                offset = round(offset_real)
                end = mode_peak-math.ceil(-plan["prepeak_cutoff_M"]/step)
                if (not 0 <= offset < low.shape[1] or end < 10
                        or end > min(source_mode.size, wheel_mode.size)
                        or offset+end > low.shape[1]):
                    raise ValueError(f"cannot map q2 mode to public states: {key}")
                r, pr = low[1, offset:offset+end], low[3, offset:offset+end]
                omega = orbital_omega[offset:offset+end]
                ratio = (_nqc_factor(source_coeff, r, pr, omega)
                         /_nqc_factor(wheel_coeff, r, pr, omega))
                reference = source_mode[:end]
                prediction = wheel_mode[:end]*ratio
                report["steps"][key] = dict(
                    samples=end, last_sample_before_wheel_peak_M=(end-1-mode_peak)*step,
                    mode_to_low_index_offset=offset,
                    mode_to_low_index_remainder=offset_real-offset,
                    factor_amplitude_range=[float(np.min(abs(ratio))),
                                            float(np.max(abs(ratio)))],
                    factor_phase_range_rad=[float(np.min(np.angle(ratio))),
                                            float(np.max(np.angle(ratio)))],
                    wheel_against_source=q.mode_metrics(wheel_mode[:end], reference),
                    wheel_times_coefficient_ratio_against_source=q.mode_metrics(
                        prediction, reference))
                q.write_evidence(args.json_out, report)
                print(f"step={key} complete", flush=True)
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

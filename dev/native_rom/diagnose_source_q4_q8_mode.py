"""Compare frozen native q4/q8 modes with the pinned source on an orbital clock.

The source and native orbital-frequency peaks define each time origin. A single
predeclared early-time phase anchor removes the arbitrary initial phase gauge;
no relative time, amplitude, parameter, or phase-slope fitting is performed.
"""

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

import numpy as np
import torch
import lal

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_peak
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_wheel_mode import _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_q4_q8_mode_plan.json"
ACTIVE = (0, 1, 2, 6, 7)


def _run(row, level, settings, choice, restart):
    numerical, flux, physical = choice["numerical"], choice["flux"], choice["physical"]
    return q.native_run(
        row, level, settings,
        diagnostic_hermite_resampling=numerical["hermite_resampling"],
        diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
        diagnostic_matching_second_derivatives=True,
        diagnostic_restart_time_M=restart,
        diagnostic_coarse_initial_step_M=0.125,
        diagnostic_warm_start_Momega=0.03162277660168379,
        diagnostic_nqc_phase_window_M=choice["nqc_phase_window_M"],
        diagnostic_mode_21_higher_spin=flux["mode_21_higher_spin"],
        diagnostic_mode_31_higher_spin=flux["mode_31_higher_spin"],
        diagnostic_mode_32_spin_square=flux["mode_32_spin_square"],
        diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
        diagnostic_mode_43_higher_spin=choice["mode_43_higher_spin"],
        diagnostic_spin_truncation=choice["spin_truncation"],
        diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
        diagnostic_hamiltonian_calibration=choice["calibration"],
        return_nqc_coefficients=True)


def _comparison(source_t, source_h, native_t, native_h, anchor, windows):
    common = ((source_t >= native_t[0]) & (source_t <= native_t[-1])
              & (source_t >= windows[0]["start_M"])
              & (source_t <= windows[-1]["end_M"]))
    time_M = source_t[common]
    reference = source_h[common]
    candidate = (np.interp(time_M, native_t, native_h.real)
                 + 1j*np.interp(time_M, native_t, native_h.imag))
    native_amplitude = np.interp(time_M, native_t, np.abs(native_h))
    native_phase = np.interp(time_M, native_t, np.unwrap(np.angle(native_h)))
    polar_candidate = native_amplitude*np.exp(1j*native_phase)
    if time_M.size < 10 or not (time_M[0] <= anchor <= time_M[-1]):
        raise ValueError("insufficient common orbital-peak support")
    h_native = np.interp(anchor, native_t, native_h.real) + 1j*np.interp(
        anchor, native_t, native_h.imag)
    h_source = np.interp(anchor, source_t, source_h.real) + 1j*np.interp(
        anchor, source_t, source_h.imag)
    rotation = float(np.angle(h_native/h_source))
    candidate *= np.exp(-1j*rotation)
    polar_anchor = np.interp(anchor, native_t, np.abs(native_h))*np.exp(
        1j*np.interp(anchor, native_t, np.unwrap(np.angle(native_h))))
    polar_rotation = float(np.angle(polar_anchor/h_source))
    polar_candidate *= np.exp(-1j*polar_rotation)
    output = {}
    for window in windows:
        mask = ((time_M >= window["start_M"]) & (time_M < window["end_M"]))
        if mask.sum() < 10:
            raise ValueError(f"too few samples in {window['name']}")
        output[window["name"]] = dict(
            time_M=[float(time_M[mask][0]), float(time_M[mask][-1])],
            metrics=q.mode_metrics(candidate[mask], reference[mask]),
            polar_interpolation_metrics=q.mode_metrics(
                polar_candidate[mask], reference[mask]))
    return dict(phase_rotation_rad=rotation,
                polar_phase_rotation_rad=polar_rotation, windows=output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / path for name, path in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-q8-mode-plan.v1"
            or plan["rows"] != ["q4_mixed_spin", "q8_high_aligned"]
            or plan["level"] != "ode_2" or plan["output_step_M"] != 0.5
            or plan["phase_anchor_M"] != -100.0
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen source/native mode plan or inputs differ")
    campaign = json.loads(paths["campaign"].read_text())
    source = json.loads(paths["source_attachment"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    integrated = json.loads(paths["integrated_candidate"].read_text())
    if (campaign["status"] != "diagnostic_complete"
            or source["status"] != "diagnostic_complete"
            or integrated["status"] != "diagnostic_complete"
            or integrated["input_sha256"][
                "dev/native_rom/evidence/2026-09-24/source_candidate_new_rows.json"]
               != plan["input_sha256"]["campaign"]
            or q.lal_binary_identity(campaign["lal_binary"]["lal_version"],
                                     campaign["lal_binary"]["lalsimulation_version"])
               != campaign["lal_binary"]):
        parser.error("pinned campaign, source, or public binary differs")
    torch.set_num_threads(1)
    scope = dict(q.SCOPE)
    scope["description"] = "Diagnostic source/native full-mode comparison on a common orbital clock."
    scope["known_limitations"] = [
        item for item in q.SCOPE["known_limitations"]
        if item != "No time, phase, amplitude, or coefficient fitting is performed."]
    scope["known_limitations"].append(
        "One fixed early-time phase anchor removes only the arbitrary initial "
        "phase gauge; native values are interpolated to source samples both "
        "componentwise and through amplitude plus unwrapped phase.")
    report = dict(schema="diffgw.seobnrv4.source-q4-q8-mode.v1", scope=scope,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], lal_binary=campaign["lal_binary"],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), rows={}, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-q4-q8-mode-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            dylib = Path(tmp) / "source_q4_q8_mode.dylib"
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
                started = time.monotonic()
                item = campaign["rows"][name]
                row = item["row"]
                run = _run(row, item["runs"]["ode_2"]["level"],
                           q.Settings(**campaign["settings"]), campaign["choice"],
                           item["fixed_restart_time_M"])
                replay = q.summarize_run(run)
                if replay["array_sha256"] != integrated["rows"][name]["array_sha256"]:
                    raise ValueError(f"{name} native replay differs from integrated variant")
                source_epoch, source_dt, source_mode = _source(
                    function, row, plan["output_step_M"], plan["capacity"])
                mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
                if not math.isclose(source_dt/mass_seconds, plan["output_step_M"],
                                    abs_tol=1e-10, rel_tol=0):
                    raise ValueError(f"{name} source output step differs")
                source_t = (source_epoch+np.arange(source_mode.size)*source_dt)/mass_seconds
                native_t = run["time"]-run["peak_time_M"]
                native_mode = run["mode"]*mass_seconds*lal.C_SI/(100e6*lal.PC_SI)
                source_fit = fitted_peak(source_t, source_mode, "quadratic_power")
                known = source["rows"][name]["fitted_waveform_peak"][
                    "source_mode_peak_from_orbital_peak_M"]
                if abs(source_fit["time_s"]-known) > 1e-6:
                    raise ValueError(f"{name} source mode peak differs from capture")
                boundary = source["rows"][name]["peak_relative"][
                    "source_boundary_from_peak_M"]
                windows = [dict(name="early", start_M=-120., end_M=-60.),
                           dict(name="late_inspiral", start_M=-60., end_M=boundary),
                           dict(name="attachment", start_M=boundary, end_M=boundary+30.),
                           dict(name="ringdown", start_M=boundary+30., end_M=100.)]
                comparison = _comparison(source_t, source_mode, native_t,
                                         native_mode, plan["phase_anchor_M"], windows)
                coefficients = run["nqc_coefficients"]
                native_coeff = np.r_[coefficients["amplitude"],
                                     coefficients["phase"]].astype(np.float64)
                if native_coeff.size != 5 or not np.isfinite(native_coeff).all():
                    raise ValueError(f"{name} native NQC coefficients invalid")
                source_coeff = np.asarray(source["rows"][name]["source"][
                    "nqc_coefficients"], dtype=np.float64)[list(ACTIVE)]
                wheel_coeff = np.asarray(source["rows"][name]["wheel"][
                    "nqc_coefficients"], dtype=np.float64)[list(ACTIVE)]
                report["rows"][name] = dict(row=row, elapsed_seconds=time.monotonic()-started,
                    source_samples=int(source_mode.size), native_samples=int(native_mode.size),
                    source_mode_sha256=hashlib.sha256(source_mode.tobytes()).hexdigest(),
                    native_array_sha256=replay["array_sha256"],
                    source_epoch_s=source_epoch, source_boundary_from_orbital_peak_M=boundary,
                    native_boundary_from_orbital_peak_M=(run["boundary_time_M"]-run["peak_time_M"]),
                    comparison=comparison,
                    nqc_coefficients=dict(native=native_coeff.tolist(),
                                          source=source_coeff.tolist(),
                                          wheel=wheel_coeff.tolist(),
                                          native_minus_source_relative_l2=float(
                                              np.linalg.norm(native_coeff-source_coeff)
                                              /np.linalg.norm(source_coeff)),
                                          native_minus_wheel_relative_l2=float(
                                              np.linalg.norm(native_coeff-wheel_coeff)
                                              /np.linalg.norm(wheel_coeff))))
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

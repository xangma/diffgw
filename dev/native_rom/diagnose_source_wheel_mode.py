"""Compare complete v4 (2,2) modes from pinned source and wheel generators."""

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

import lal
import lalsimulation as lalsim
import numpy as np

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_wheel_mode_plan.json"


def _public(row: dict, step_M: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
    output = lalsim.SimIMRSpinAlignedEOBModes(
        step_M*mass_seconds, row["mass1"]*lal.MSUN_SI,
        row["mass2"]*lal.MSUN_SI, row["omega_start"]/(math.pi*mass_seconds),
        100e6*lal.PC_SI, row["chi1"], row["chi2"], 4,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
        lal.CreateREAL8Vector(10), 0, lal.CreateDict(), lal.CreateDict())
    mode = lalsim.SphHarmTimeSeriesGetMode(output[0], 2, 2)
    low = np.asarray(output[1].data, dtype=np.float64).reshape(5, -1).copy()
    return (float(mode.epoch), float(mode.deltaT),
            np.asarray(mode.data.data, dtype=np.complex128).copy(), low)


def _source(function, row: dict, step_M: float, capacity: int
            ) -> tuple[float, float, np.ndarray]:
    values = np.empty(2*capacity, dtype=np.float64)
    count = ctypes.c_size_t()
    epoch, delta_t = ctypes.c_double(), ctypes.c_double()
    rc = function(row["mass1"], row["mass2"], row["chi1"], row["chi2"],
                  row["omega_start"], step_M, capacity, ctypes.byref(count),
                  ctypes.byref(epoch), ctypes.byref(delta_t),
                  values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if rc or count.value < 10 or count.value > capacity:
        raise RuntimeError(f"Source full-generator mode failed: {rc}")
    mode = values[:2*count.value].reshape(-1, 2)
    return epoch.value, delta_t.value, mode[:, 0]+1j*mode[:, 1]


def _public_nqc(row: dict, step_M: float) -> np.ndarray:
    mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
    coefficients = lal.CreateREAL8Vector(10)
    lalsim.SimIMRSpinAlignedEOBModes(
        step_M*mass_seconds, row["mass1"]*lal.MSUN_SI,
        row["mass2"]*lal.MSUN_SI, row["omega_start"]/(math.pi*mass_seconds),
        100e6*lal.PC_SI, row["chi1"], row["chi2"], 4,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
        coefficients, 1, lal.CreateDict(), lal.CreateDict())
    return np.asarray(coefficients.data, dtype=np.float64).copy()


def _source_nqc(function, row: dict, step_M: float) -> np.ndarray:
    values = np.empty(10, dtype=np.float64)
    rc = function(row["mass1"], row["mass2"], row["chi1"], row["chi2"],
                  row["omega_start"], step_M,
                  values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if rc or not np.isfinite(values).all():
        raise RuntimeError(f"Source full-generator NQC failed: {rc}")
    return values


def _compare(name: str, row: dict, step_M: float,
             source: tuple[float, float, np.ndarray],
             wheel: tuple[float, float, np.ndarray, np.ndarray],
             source_nqc: np.ndarray, wheel_nqc: np.ndarray) -> dict:
    source_epoch, source_dt, source_mode = source
    wheel_epoch, wheel_dt, wheel_mode, low = wheel
    mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
    if (not np.isfinite(source_mode).all() or not np.isfinite(wheel_mode).all()
            or abs(source_dt/mass_seconds-step_M) > 1e-10
            or abs(wheel_dt/mass_seconds-step_M) > 1e-10):
        raise ValueError(f"Source/wheel mode invalid: {name}")
    source_time = source_epoch + np.arange(source_mode.size)*source_dt
    wheel_time = wheel_epoch + np.arange(wheel_mode.size)*wheel_dt
    count = min(source_mode.size, wheel_mode.size)
    direct = q.mode_metrics(source_mode[:count], wheel_mode[:count])
    peak = int(np.argmax(abs(wheel_mode[:count])))
    omega = np.gradient(low[2], step_M, edge_order=2)
    low_peak = int(np.argmax(omega))
    offset_real = low_peak+wheel_epoch/wheel_dt
    offset = round(offset_real)
    if (abs(offset_real-offset) > 0.5 or offset < 0
            or offset+peak-160 >= low.shape[1]):
        raise ValueError(f"Cannot map {name} mode to low dynamics")
    # Both public and source NQC factors use p=pr_star for these 22 modes.
    # Evaluate the coefficient-only counterfactual on the *same* public
    # trajectory, at least 20 M before the amplitude peak/attachment.
    end = peak-160
    sample = low[:, offset:offset+end]
    r, pr, om = sample[1], sample[3], omega[offset:offset+end]
    ratio = pr/(r*om)
    def nqc(coefficients: np.ndarray) -> np.ndarray:
        amp = 1+ratio**2*(coefficients[0]+coefficients[1]/r
                           + coefficients[2]/r**1.5)
        phase = coefficients[6]*ratio+coefficients[7]*ratio*pr**2
        return amp*np.exp(1j*phase)
    coefficient_prediction = nqc(source_nqc)/nqc(wheel_nqc)
    predicted_source = wheel_mode[:end]*coefficient_prediction
    predicted_metrics = q.mode_metrics(predicted_source, source_mode[:end])
    uncorrected_metrics = q.mode_metrics(wheel_mode[:end], source_mode[:end])
    sample_indices = sorted(set((0, count//4, count//2, max(0, peak-160),
                                 peak, min(count-1, peak+160))))
    samples = []
    for index in sample_indices:
        ratio = source_mode[index]/wheel_mode[index]
        samples.append(dict(index=index, time_M=index*step_M,
                            source_over_wheel_amplitude=float(abs(ratio)),
                            source_minus_wheel_phase_rad=float(np.angle(ratio)),
                            wheel_amplitude=float(abs(wheel_mode[index]))))
    fitted = fitted_metrics(source_time, source_mode, wheel_time, wheel_mode,
                            mass_seconds, "quadratic_power")
    return dict(name=name, row=row, step_M=step_M,
                source_samples=int(source_mode.size), wheel_samples=int(wheel_mode.size),
                source_epoch_s=source_epoch, wheel_epoch_s=wheel_epoch,
                source_minus_wheel_epoch_M=(source_epoch-wheel_epoch)/mass_seconds,
                source_delta_t_s=source_dt, wheel_delta_t_s=wheel_dt,
                source_mode_sha256=hashlib.sha256(source_mode.tobytes()).hexdigest(),
                wheel_mode_sha256=hashlib.sha256(wheel_mode.tobytes()).hexdigest(),
                wheel_peak_index=peak, low_orbital_peak_index=low_peak,
                mode_to_low_index_offset=offset,
                mode_to_low_index_offset_remainder=offset_real-offset,
                pre_attachment_prediction=dict(
                    samples=end, last_mode_time_before_peak_M=(end-peak)*step_M,
                    wheel_against_source=uncorrected_metrics,
                    wheel_times_coefficient_ratio_against_source=predicted_metrics),
                same_index_samples=samples,
                same_index_metrics=direct, fitted=fitted,
                source_nqc_coefficients=source_nqc.tolist(),
                wheel_nqc_coefficients=wheel_nqc.tolist(),
                source_minus_wheel_nqc=(source_nqc-wheel_nqc).tolist())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {key: q.ROOT / plan[key] for key in
             ("source_flux_result", "source_flux_plan", "dynamics_result", "adapter")}
    if (plan["schema"] != "diffgw.seobnrv4.source-wheel-mode-plan.v1"
            or plan["rows"] != ["nonspinning", "mild_mixed_spin", "strong_aligned_spin"]
            or plan["step_M"] != 0.125 or plan["max_samples"] != 50000
            or any(q.sha256(paths[key]) != plan[key+"_sha256"] for key in paths)):
        parser.error("Frozen plan or inputs changed")
    prior = json.loads(paths["source_flux_result"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    dynamics = json.loads(paths["dynamics_result"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or dynamics["status"] != "diagnostic_complete"
            or [item["name"] for item in dynamics["rows"]] != plan["rows"]):
        parser.error("Prerequisite source/wheel dynamics unavailable")
    libraries = _libraries(source_plan["pinned_wheel"])
    report = dict(schema="diffgw.seobnrv4.source-wheel-mode.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={key: q.sha256(path) for key, path in paths.items()},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), rows=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-mode-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            dylib = Path(tmp) / "full_source_mode.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["adapter"], dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            function = ctypes.CDLL(str(dylib)).diffgw_v4_full_source_mode
            function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            nqc_function = ctypes.CDLL(str(dylib)).diffgw_v4_full_source_nqc
            nqc_function.argtypes = [ctypes.c_double]*6 + [
                ctypes.POINTER(ctypes.c_double)]
            nqc_function.restype = ctypes.c_int
            for item in dynamics["rows"]:
                name, row = item["name"], item["row"]
                source = _source(function, row, plan["step_M"], plan["max_samples"])
                wheel = _public(row, plan["step_M"])
                source_nqc = _source_nqc(nqc_function, row, plan["step_M"])
                wheel_nqc = _public_nqc(row, plan["step_M"])
                report["rows"].append(_compare(name, row, plan["step_M"],
                                                source, wheel, source_nqc,
                                                wheel_nqc))
                q.write_evidence(args.json_out, report)
                print(f"row={name}", flush=True)
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

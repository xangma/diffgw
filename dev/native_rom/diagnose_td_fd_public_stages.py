"""Replay public LAL conditioning and FFT on identical TD inputs.

This development diagnostic uses LAL only as an external oracle. Its rows do
not validate the native SEOBNRv4 generator or the ROM training waveform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

import lal
import lalsimulation as lalsim
import numpy as np
import torch

from . import qualify_generator as q
from torchwave.approximants.eob.td_fd_conditioning_clean import (
    condition_td,
    fft_length_and_sampling,
    pair_leading_pad_final_crop,
    plan_td_support,
)


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/td_fd_public_stages_plan.json"


def _series(name: str, data: np.ndarray, dt: float):
    series = lal.CreateREAL8TimeSeries(
        name, lal.LIGOTimeGPS(0), 0.0, dt, lal.DimensionlessUnit, len(data)
    )
    series.data.data[:] = data
    return series


def _raw_input(row: dict, plan: dict, support, dt: float) -> tuple[np.ndarray, np.ndarray, float | None]:
    if row["kind"] == "analytic":
        t = np.arange(row["samples"], dtype=np.float64) * dt
        phase = 2.0 * math.pi * (16.0 * t + 0.5 * 95.0 * t**2 + 2.5 * t**3)
        envelope = np.sqrt(0.4 + t) * np.exp(-0.5 * ((t - 1.55) / 0.19) ** 2) + 0.03
        hp = envelope * np.cos(phase)
        hc = 0.7 * envelope * np.sin(phase + 0.2)
        hp[-row["trailing_zeros"]:] = 0.0
        hc[-row["trailing_zeros"]:] = 0.0
        return hp, hc, None
    if row["kind"] != "public_seobnrv4":
        raise ValueError("unknown frozen input kind")
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        row["mass1"] * lal.MSUN_SI, row["mass2"] * lal.MSUN_SI,
        0.0, 0.0, row["chi1z"], 0.0, 0.0, row["chi2z"],
        plan["public_distance_mpc"] * 1e6 * lal.PC_SI,
        plan["public_inclination_rad"], plan["public_reference_phase_rad"],
        0.0, 0.0, 0.0, dt, float(support.start_frequency),
        plan["public_reference_frequency_hz"], lal.CreateDict(), lalsim.SEOBNRv4,
    )
    if hp.deltaT != dt or hc.deltaT != dt:
        raise ValueError("public TD sample spacing changed")
    return np.array(hp.data.data, copy=True), np.array(hc.data.data, copy=True), float(hp.epoch)


def _relative_errors(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if (reference.shape != candidate.shape or not np.isfinite(reference).all()
            or not np.isfinite(candidate).all() or np.linalg.norm(reference) == 0):
        raise ValueError("output shape, finiteness, or normalization differs")
    difference = candidate - reference
    return {
        "rms_relative": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        "max_absolute_over_reference_peak": float(
            np.max(np.abs(difference)) / np.max(np.abs(reference))
        ),
    }


def _lal_fft(values: np.ndarray, dt: float) -> np.ndarray:
    n = len(values)
    td = _series("conditioned", values, dt)
    fd = lal.CreateCOMPLEX16FrequencySeries(
        "conditioned_fd", lal.LIGOTimeGPS(0), 0.0, 1.0 / (n * dt),
        lal.DimensionlessUnit, n // 2 + 1,
    )
    plan = lal.CreateForwardREAL8FFTPlan(n, 0)
    if lal.REAL8TimeFreqFFT(fd, td, plan) != 0:
        raise ValueError("public LAL real FFT failed")
    return np.array(fd.data.data, copy=True)


def _public_fd(row: dict, plan: dict):
    hp, hc = lalsim.SimInspiralFD(
        row["mass1"] * lal.MSUN_SI, row["mass2"] * lal.MSUN_SI,
        0.0, 0.0, row["chi1z"], 0.0, 0.0, row["chi2z"],
        plan["public_distance_mpc"] * 1e6 * lal.PC_SI,
        plan["public_inclination_rad"], plan["public_reference_phase_rad"],
        0.0, 0.0, 0.0, plan["public_fd_delta_f_hz"], row["f_low"],
        plan["public_fd_fmax_hz"], plan["public_reference_frequency_hz"],
        lal.CreateDict(), lalsim.SEOBNRv4,
    )
    if hp.deltaF != plan["public_fd_delta_f_hz"] or hc.deltaF != hp.deltaF:
        raise ValueError("public FD frequency spacing differs")
    return hp, hc


def _run_row(row: dict, plan: dict, dt: float) -> dict:
    support = plan_td_support(
        row["mass1"], row["mass2"], row["f_low"], row["chi1z"], row["chi2z"]
    )
    hp, hc, raw_epoch = _raw_input(row, plan, support, dt)
    raw_hash = hashlib.sha256(hp.tobytes() + hc.tobytes()).hexdigest()
    public_hp, public_hc = _series("hp", hp, dt), _series("hc", hc, dt)
    if lalsim.SimInspiralTDConditionStage1(
        public_hp, public_hc, float(support.first_taper_duration), row["f_low"]
    ) != 0:
        raise ValueError("public first conditioning stage failed")
    if lalsim.SimInspiralTDConditionStage2(
        public_hp, public_hc, float(support.f_safe), float(support.f_isco)
    ) != 0:
        raise ValueError("public second conditioning stage failed")
    native = condition_td(
        torch.from_numpy(hp), torch.from_numpy(hc), dt=dt, support=support,
        return_metadata=True,
    )
    public_values = [np.array(series.data.data, copy=True) for series in (public_hp, public_hc)]
    native_values = [native.h_plus.numpy(), native.h_cross.numpy()]
    td_errors = {
        name: _relative_errors(reference, candidate)
        for name, reference, candidate in zip(
            ("plus", "cross"), public_values, native_values
        )
    }
    sampling = fft_length_and_sampling(
        plan["public_fd_fmax_hz"], plan["public_fd_delta_f_hz"], dt=dt
    )
    fd_errors = {}
    fft_only_errors = {}
    public_fd_errors = {}
    public_fd_epoch_error_seconds = None
    public_fd = _public_fd(row, plan) if row["kind"] == "public_seobnrv4" else None
    for name, public_value, native_value in zip(
        ("plus", "cross"), public_values, native_values
    ):
        public_pad, _ = pair_leading_pad_final_crop(
            torch.from_numpy(public_value), torch.from_numpy(public_value), sampling.n_time
        )
        native_pad, _ = pair_leading_pad_final_crop(
            torch.from_numpy(native_value), torch.from_numpy(native_value), sampling.n_time
        )
        public_fft = _lal_fft(public_pad.numpy(), dt)
        native_fft = (dt * torch.fft.rfft(native_pad)).numpy()
        fft_only = (dt * torch.fft.rfft(public_pad)).numpy()
        fd_errors[name] = _relative_errors(public_fft, native_fft)
        fft_only_errors[name] = _relative_errors(public_fft, fft_only)
        if public_fd is not None:
            series = public_fd[0] if name == "plus" else public_fd[1]
            public_fd_errors[name] = _relative_errors(
                np.array(series.data.data, copy=True), native_fft
            )
            epoch_expected = raw_epoch - (sampling.n_time - len(public_value)) * dt
            epoch_error = abs(float(series.epoch) - epoch_expected)
            public_fd_epoch_error_seconds = max(
                public_fd_epoch_error_seconds or 0.0, epoch_error
            )
    max_td_rms = max(value["rms_relative"] for value in td_errors.values())
    max_td_peak = max(value["max_absolute_over_reference_peak"] for value in td_errors.values())
    max_fd_rms = max(value["rms_relative"] for value in fd_errors.values())
    max_fd_peak = max(value["max_absolute_over_reference_peak"] for value in fd_errors.values())
    passed = (
        max_td_rms <= plan["full_series_rms_relative_tolerance"]
        and max_td_peak <= plan["full_series_max_absolute_over_peak_tolerance"]
        and max_fd_rms <= plan["fft_rms_relative_tolerance"]
        and max_fd_peak <= plan["fft_max_absolute_over_peak_tolerance"]
        and all(
            error["rms_relative"] <= plan["fft_rms_relative_tolerance"]
            and error["max_absolute_over_reference_peak"]
            <= plan["fft_max_absolute_over_peak_tolerance"]
            for error in public_fd_errors.values()
        )
        and (public_fd_epoch_error_seconds is None
             or public_fd_epoch_error_seconds <= plan["public_fd_epoch_tolerance_seconds"])
    )
    return {
        "name": row["name"], "kind": row["kind"], "raw_td_sha256": raw_hash,
        "raw_samples": len(hp), "conditioned_samples": len(public_values[0]),
        "trailing_zero_samples": int(native.n_trailing_zeros),
        "first_taper_samples": int(native.first_taper_samples),
        "second_begin_samples": int(native.second_begin_samples),
        "second_end_samples": int(native.second_end_samples),
        "f_safe_hz": float(support.f_safe),
        "start_frequency_hz": float(support.start_frequency),
        "fft_samples": sampling.n_time,
        "delta_f_hz": float(sampling.delta_f),
        "td_errors": td_errors, "fd_errors": fd_errors,
        "fft_only_errors": fft_only_errors,
        "public_full_fd_errors": public_fd_errors,
        "public_fd_epoch_error_seconds": public_fd_epoch_error_seconds,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    frozen_rows = [
        dict(name="synthetic_chirp_with_padding", kind="analytic", mass1=30.0,
             mass2=20.0, chi1z=0.0, chi2z=0.0, f_low=20.0, samples=8192,
             trailing_zeros=97),
        dict(name="public_v4_mixed_spin", kind="public_seobnrv4", mass1=30.0,
             mass2=20.0, chi1z=0.3, chi2z=-0.2, f_low=20.0),
        dict(name="public_v4_safe_frequency_cap", kind="public_seobnrv4",
             mass1=50.0, mass2=30.0, chi1z=0.8, chi2z=0.5, f_low=60.0),
    ]
    if (plan["schema"] != "diffgw.seobnrv4.td-fd-public-stages-plan.v1"
            or plan["release_qualified"] is not False
            or plan["rows"] != frozen_rows
            or plan["sample_rate_hz"] != 4096
            or plan["public_distance_mpc"] != 1.0
            or plan["public_inclination_rad"] != 0.7
            or plan["public_reference_phase_rad"] != 0.0
            or plan["public_reference_frequency_hz"] != 0.0
            or plan["public_fd_fmax_hz"] != 2048.0
            or plan["public_fd_delta_f_hz"] != 0.25
            or plan["public_fd_epoch_tolerance_seconds"] != 1e-9
            or any(plan[key] != 1e-10 for key in (
                "full_series_rms_relative_tolerance",
                "full_series_max_absolute_over_peak_tolerance",
                "fft_rms_relative_tolerance",
                "fft_max_absolute_over_peak_tolerance",
            ))):
        parser.error("frozen conditioning plan differs")
    binary_file = q.ROOT / plan["binary_evidence"]
    if q.sha256(binary_file) != plan["binary_evidence_sha256"]:
        parser.error("pinned binary evidence differs")
    frozen_binary = json.loads(binary_file.read_text())["lal_binary"]
    binary = q.lal_binary_identity(
        frozen_binary["lal_version"], frozen_binary["lalsimulation_version"]
    )
    if binary != frozen_binary:
        parser.error("installed public binary differs from frozen oracle")
    report = {
        "schema": "diffgw.seobnrv4.td-fd-public-stages.v1",
        "status": "running", "release_qualified": False,
        "scope": dict(q.SCOPE,
                      description="Identical-input TD conditioning, real FFT, and selected public SEOBNRv4 FD values/epoch.",
                      known_limitations=[
                          "Two physical rows and one analytic row do not cover the full parameter domain.",
                          "The public SEOBNRv4 binary generates the two physical TD inputs; native generator fidelity is not assessed.",
                          "The public complex FD values and FFT epoch are checked only at f_ref=0 on two physical rows; general phase/time conventions remain open.",
                          "Native whole waveforms, broader frequency support, and ROM holdouts remain open.",
                      ]),
        "plan_sha256": q.sha256(PLAN),
        "script_sha256": q.sha256(__file__),
        "lal_binary": binary, "rows": [], "errors": [],
        "process": {"host": socket.gethostname(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
    }
    q.write_evidence(args.json_out, report)
    try:
        dt = 1.0 / plan["sample_rate_hz"]
        for row in plan["rows"]:
            result = _run_row(row, plan, dt)
            report["rows"].append(result)
            q.write_evidence(args.json_out, report)
            print(row["name"], result["passed"], flush=True)
        report["status"] = "passed" if all(row["passed"] for row in report["rows"]) else "failed"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

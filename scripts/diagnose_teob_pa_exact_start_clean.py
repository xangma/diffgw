#!/usr/bin/env python3
"""Bounded current exact-start TEOB PA-only isolation diagnostic.

The public PA-only call is compared on its full uniform elapsed-time grid
with the cached native PA prefix reconstructed from its saved pre-NQC
amplitude and lifted phase. The same prefix is also compared with the
cached processed native polarizations. No alignment, fit, runtime edit, or
native ODE evolution is performed.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import lal
import lalsimulation as lalsim
import numpy as np
import torch

from torchwave.approximants.eob.numerical_attachment import natural_cubic_spline

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs/reviews/2026-09-13-parity-review"
NATIVE = REVIEW / "verification/teob-native-replay-public-clean/replay-native-output.npz"
NATIVE_REPORT = REVIEW / "verification/teob-native-replay-public-clean/replay-report.json"
PUBLIC_ARRAYS = REVIEW / "verification/teob-conditioner-public-clean/arrays.npz"
PUBLIC_REPORT = REVIEW / "verification/teob-conditioner-public-clean/report.json"
OUT = REVIEW / "verification/teob-pa-exact-start-clean"

M1, M2, MTOT = 30.0, 20.0, 50.0
DISTANCE_MPC = 100.0
F_START = 16.900094009839023
F_REF = 20.0
DT_SECONDS = 1.0 / 2048.0
R0, R_MIN, DR, PA_ORDER = 18.017491181997485, 14.0, 0.1, 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def arr_hash(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    return hashlib.sha256(x.view(np.uint8)).hexdigest()


def finite(x: Any) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def stats(x: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    return {"count": int(x.size), "rms": finite(np.sqrt(np.mean(x * x))),
            "max_abs": finite(np.max(np.abs(x))), "mean": finite(np.mean(x))}


def public_options() -> lal.Dict:
    p = lal.CreateDict()
    lal.DictInsertINT4Value(p, "TEOB_use_postadiabatic", 2)
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(p, 0)
    modes = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(modes, 2, 2)
    lalsim.SimInspiralWaveformParamsInsertModeArray(p, modes)
    # PNSpinOrder is omitted deliberately: preserve the caller/default branch.
    return p


def public_pa() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    hp_s, hc_s = lalsim.SimIMRTEOBResumS(
        0.0, DT_SECONDS, M1 * lal.MSUN_SI, M2 * lal.MSUN_SI,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        DISTANCE_MPC * 1.0e6 * lal.PC_SI, 0.0, 0.0, public_options(),
        0.0, 0.0, F_START, F_REF,
    )
    epoch = float(hp_s.epoch)
    delta_t = float(hp_s.deltaT)
    hp = -np.asarray(hp_s.data.data, dtype=np.float64)
    hc = -np.asarray(hc_s.data.data, dtype=np.float64)
    dt_m = delta_t / (MTOT * lal.MTSUN_SI)
    t = np.arange(hp.size, dtype=np.float64) * dt_m
    meta = {"samples": int(hp.size), "epoch_seconds": epoch,
            "delta_t_seconds": delta_t, "dt_seconds_requested": DT_SECONDS,
            "dt_M": dt_m, "dt_M_expected": DT_SECONDS / (MTOT * lal.MTSUN_SI),
            "dt_M_difference": finite(dt_m - DT_SECONDS / (MTOT * lal.MTSUN_SI)),
            "time_start_M": finite(t[0]), "time_end_M": finite(t[-1]),
            "elapsed_seconds": finite(time.perf_counter() - started),
            "array_hashes": {"h_plus": arr_hash(hp), "h_cross": arr_hash(hc), "time": arr_hash(t)}}
    if not (hp.size == hc.size > 1 and epoch == 0.0 and delta_t == DT_SECONDS and
            np.all(np.isfinite(hp)) and np.all(np.isfinite(hc)) and np.all(np.diff(t) > 0)):
        raise RuntimeError("public PA output violates exact request/start/grid contract")
    return t, hp, hc, meta


def spline_complex(t: np.ndarray, amplitude: np.ndarray, phase: np.ndarray,
                   query: np.ndarray) -> np.ndarray:
    if t.size < 2 or not np.all(np.diff(t) > 0):
        raise RuntimeError("spline knots must be strictly increasing")
    if query[0] < t[0] - 1e-10 or query[-1] > t[-1] + 1e-10:
        raise RuntimeError("spline query would extrapolate")
    x = torch.as_tensor(t, dtype=torch.float64)
    q = torch.as_tensor(query, dtype=torch.float64)
    a = natural_cubic_spline(x, torch.as_tensor(amplitude, dtype=torch.float64), scalar_output=True).evaluate(q)
    p = natural_cubic_spline(x, torch.as_tensor(phase, dtype=torch.float64), scalar_output=True).evaluate(q)
    return (a * torch.exp(-1j * p)).numpy()


def metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa, bb = np.abs(a), np.abs(b)
    phase = np.unwrap(np.angle(a * np.conj(b)))
    return {"samples": int(a.size),
            "raw_relative_l2": finite(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)),
            "amplitude_relative_l2": finite(np.linalg.norm(aa - bb) / max(np.linalg.norm(bb), 1e-300)),
            "amplitude_fractional": stats(aa / np.maximum(bb, 1e-300) - 1.0),
            "phase_difference_radians": stats(phase),
            "phase_initial_radians": finite(phase[0]), "phase_final_radians": finite(phase[-1]),
            "alignment": "none"}


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    native = np.load(NATIVE, allow_pickle=False)
    public_t, public_hp, public_hc, public_meta = public_pa()
    native_t = np.asarray(native["td_time"], dtype=np.float64)
    native_hp = np.asarray(native["h_plus_td"], dtype=np.float64)
    native_hc = np.asarray(native["h_cross_td"], dtype=np.float64)
    pa_t = np.asarray(native["pa_time"], dtype=np.float64)
    pa_r = np.asarray(native["pa_radius"], dtype=np.float64)
    nqc_t = np.asarray(native["nqc_original_time"], dtype=np.float64)
    nqc_a = np.asarray(native["nqc_original_amplitude"], dtype=np.float64)
    nqc_p = np.asarray(native["nqc_original_phase"], dtype=np.float64)
    if not np.array_equal(nqc_t[:pa_t.size], pa_t):
        raise RuntimeError("native saved NQC-original prefix is not the reviewed PA grid")
    same_elapsed_grid = bool(np.array_equal(native_t[:public_t.size], public_t))
    if not same_elapsed_grid:
        raise RuntimeError("native and public elapsed grids differ; refusing regridding")
    common = public_t
    pa_a, pa_p = nqc_a[:pa_t.size], nqc_p[:pa_t.size]
    pre_mode = spline_complex(pa_t, pa_a, pa_p, common)
    native_mode = native_hp[:common.size] - 1j * native_hc[:common.size]
    public_mode = public_hp - 1j * public_hc
    scale = math.sqrt(5.0 / (4.0 * math.pi)) * (MTOT * lal.MTSUN_SI) * lal.C_SI / (
        DISTANCE_MPC * 1.0e6 * lal.PC_SI)
    pre_h, native_h, public_h = scale * pre_mode, native_mode, public_mode
    report = {
        "status": "complete", "diagnostic": "current exact-start PA-only pre-NQC isolation",
        "run": {"host": socket.gethostname(), "platform": platform.platform(), "cwd": str(Path.cwd()),
                "command": " ".join(sys.argv), "elapsed_seconds": finite(time.perf_counter() - started)},
        "request": {"api": "SimIMRTEOBResumS", "masses_msun": [M1, M2], "distance_mpc": DISTANCE_MPC,
                    "inclination": 0.0, "phi_ref": 0.0, "f_start_hz": F_START, "f_ref_hz": F_REF,
                    "dt_seconds": DT_SECONDS, "TEOB_use_postadiabatic": 2, "ModeArray": [[2, 2]],
                    "PNTidalOrder": 0, "PNSpinOrder": "omitted; caller/default preserved",
                    "spins": [0.0] * 6, "tidal_parameters": [0.0] * 4},
        "public_pa_output": public_meta,
        "native_cache_contract": {"native_samples": int(native_t.size), "native_pa_knots": int(pa_t.size),
                                   "native_pa_end_time_M": finite(pa_t[-1]),
                                   "native_support_start_frequency_hz": finite(native["support_start_frequency"]),
                                   "native_initial_radius": finite(native["initial_radius"]),
                                   "requested_start_frequency_matches_native": bool(float(native["support_start_frequency"]) == F_START),
                                   "requested_start_radius_matches_native": bool(float(native["initial_radius"]) == R0),
                                   "native_processed_prefix_samples": int(common.size),
                                   "same_elapsed_grid": same_elapsed_grid,
                                   "maximum_elapsed_grid_difference_M": finite(np.max(np.abs(native_t[:common.size] - common))),
                                   "native_processed_includes_nqc": True},
        "grid_contract": {"comparison_grid": "full public PA-only uniform elapsed grid", "samples": int(common.size),
                           "start_M": finite(common[0]), "end_M": finite(common[-1]), "dt_M": public_meta["dt_M"],
                           "pa_radius_start": finite(pa_r[0]), "pa_radius_end": finite(pa_r[-1]),
                           "requested_r0": R0, "requested_r_min": R_MIN, "requested_dr": DR,
                           "requested_pa_order": PA_ORDER,
                           "no_extrapolation": True, "pa_grid_hash": arr_hash(pa_t), "radius_grid_hash": arr_hash(pa_r)},
        "nqc_isolation": {"public": "PA_ONLY contract: TEOB_use_postadiabatic=2 stops at PA endpoint and skips BBH NQC/ringdown",
                          "pre_nqc_reference": "native nqc_original_amplitude/phase PA prefix; nqc22=1 before correction",
                          "native_processed": "current cached polarizations include NQC across all 4087 TD samples",
                          "contract_basis": "coordinator supplied sanitized public binary behavior contract; no source inspected",
                          "public_epoch": "zero; no peak located",
                          "limitation": "binary API exposes the PA route through its documented option; no separate NQC toggle used"},
        "phase_convention": "complex h = h_plus - i*h_cross; pre-NQC h = positive scale*A*exp(-i*saved lifted phase); fixed pi/2-phiRef reference, no frequency-phase adjustment",
        "metrics_on_public_pa_grid": {"pre_nqc_over_public_pa_only": metrics(pre_h, public_h),
                                       "pre_nqc_over_native_processed": metrics(pre_h, native_h),
                                       "native_processed_over_public_pa_only": metrics(native_h, public_h)},
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in
                           (NATIVE, NATIVE_REPORT, PUBLIC_ARRAYS, PUBLIC_REPORT,
                            ROOT / "src/torchwave/approximants/eob/numerical_attachment.py",
                            ROOT / "scripts/diagnose_teob_pa_exact_start_clean.py")},
        "methods": {"interpolation": "natural cubic amplitude and saved lifted phase separately",
                    "fitted_alignment": False, "thresholds_or_fits": False, "full_native_ode": False,
                    "runtime_edit": False},
    }
    np.savez_compressed(OUT / "arrays.npz", time_M=common, pa_radius=pa_r,
                        pre_nqc_h=pre_h, public_pa_h=public_h, native_processed_h=native_h,
                        pre_nqc_h_plus=pre_h.real, pre_nqc_h_cross=-pre_h.imag,
                        public_pa_h_plus=public_h.real, public_pa_h_cross=-public_h.imag,
                        native_processed_h_plus=native_h.real, native_processed_h_cross=-native_h.imag)
    report["output_hashes"] = {"arrays.npz": sha256(OUT / "arrays.npz")}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(OUT), "samples": int(common.size),
                      "metrics": report["metrics_on_public_pa_grid"]}, indent=2))


if __name__ == "__main__":
    main()

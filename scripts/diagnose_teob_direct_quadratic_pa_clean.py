#!/usr/bin/env python3
"""Cache-only comparison of direct-quadratic replay PA and the old PA cache.

The first 41 saved ``nqc_original`` knots are the pre-NQC PA prefix.  They are
interpolated onto the existing public PA output grid; no public call, ODE
integration, alignment, or fitted quantity is used here.
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
import numpy as np
import torch

from torchwave.approximants.eob.numerical_attachment import natural_cubic_spline

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs/reviews/2026-09-13-parity-review"
DIRECT = REVIEW / "verification/teob-direct-quadratic-replay-clean"
DIRECT_CACHE = DIRECT / "replay-native-output.npz"
DIRECT_REPORT = DIRECT / "replay-report.json"
OLD = REVIEW / "verification/teob-native-replay-public-clean/replay-native-output.npz"
OLD_REPORT = REVIEW / "verification/teob-native-replay-public-clean/replay-report.json"
PA = REVIEW / "verification/teob-pa-exact-start-clean/arrays.npz"
PA_REPORT = REVIEW / "verification/teob-pa-exact-start-clean/report.json"
OUT = DIRECT / "pa-comparison-report.json"

MTOT = 50.0
DISTANCE_MPC = 100.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def arr_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def finite(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def stats(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value, dtype=np.float64)
    return {"count": int(value.size), "rms": finite(np.sqrt(np.mean(value * value))),
            "max_abs": finite(np.max(np.abs(value))), "mean": finite(np.mean(value))}


def spline_complex(knots_t: np.ndarray, amplitude: np.ndarray, phase: np.ndarray,
                   query_t: np.ndarray) -> np.ndarray:
    if knots_t.size < 2 or not np.all(np.diff(knots_t) > 0):
        raise RuntimeError("PA knots must be strictly increasing")
    if query_t[0] < knots_t[0] or query_t[-1] > knots_t[-1]:
        raise RuntimeError("public PA grid is outside saved direct PA knots")
    x = torch.as_tensor(knots_t, dtype=torch.float64)
    q = torch.as_tensor(query_t, dtype=torch.float64)
    a = natural_cubic_spline(x, torch.as_tensor(amplitude, dtype=torch.float64),
                             scalar_output=True).evaluate(q)
    p = natural_cubic_spline(x, torch.as_tensor(phase, dtype=torch.float64),
                             scalar_output=True).evaluate(q)
    return (a * torch.exp(-1j * p)).numpy()


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        raise RuntimeError(f"shape mismatch: {actual.shape} versus {reference.shape}")
    phase = np.unwrap(np.angle(actual * np.conj(reference)))
    aa, rr = np.abs(actual), np.abs(reference)
    return {
        "samples": int(actual.size),
        "raw_relative_l2": finite(np.linalg.norm(actual - reference) /
                                   max(np.linalg.norm(reference), 1e-300)),
        "amplitude_relative_l2": finite(np.linalg.norm(aa - rr) /
                                         max(np.linalg.norm(rr), 1e-300)),
        "amplitude_fractional": stats(aa / np.maximum(rr, 1e-300) - 1.0),
        "phase_difference_radians": stats(phase),
        "phase_initial_radians": finite(phase[0]),
        "phase_final_radians": finite(phase[-1]),
        "alignment": "none",
    }


def endpoint_change(new: np.ndarray, old: np.ndarray) -> dict[str, Any]:
    delta = new - old
    return {"new": finite(new[-1]), "old": finite(old[-1]),
            "delta": finite(delta[-1]), "max_abs_delta": finite(np.max(np.abs(delta)))}


def main() -> None:
    started = time.perf_counter()
    direct = np.load(DIRECT_CACHE, allow_pickle=False)
    old = np.load(OLD, allow_pickle=False)
    pa = np.load(PA, allow_pickle=False)

    grid = np.asarray(pa["time_M"], dtype=np.float64)
    old_pre = np.asarray(pa["pre_nqc_h"], dtype=np.complex128)
    public = np.asarray(pa["public_pa_h"], dtype=np.complex128)
    old_t = np.asarray(old["pa_time"], dtype=np.float64)
    new_t = np.asarray(direct["pa_time"], dtype=np.float64)
    new_phase_knots = np.asarray(direct["nqc_original_phase"][:new_t.size], dtype=np.float64)
    new_amp_knots = np.asarray(direct["nqc_original_amplitude"][:new_t.size], dtype=np.float64)
    if not np.array_equal(np.asarray(direct["nqc_original_time"][:new_t.size]), new_t):
        raise RuntimeError("direct nqc_original prefix does not equal its PA grid")
    if not np.array_equal(grid, np.asarray(pa["time_M"], dtype=np.float64)):
        raise RuntimeError("public PA grid changed while loading cache")
    if grid[0] < new_t[0] or grid[-1] > new_t[-1]:
        raise RuntimeError("public PA grid is outside direct PA endpoint")

    scale = math.sqrt(5.0 / (4.0 * math.pi)) * (MTOT * lal.MTSUN_SI) * lal.C_SI / (
        DISTANCE_MPC * 1.0e6 * lal.PC_SI)
    direct_pre = scale * spline_complex(new_t, new_amp_knots, new_phase_knots, grid)
    if direct_pre.shape != public.shape or old_pre.shape != public.shape:
        raise RuntimeError("cached PA arrays do not share the expected 2637-sample grid")

    old_phase = np.asarray(old["pa_phase"], dtype=np.float64)
    new_phase = np.asarray(direct["pa_phase"], dtype=np.float64)
    old_radius = np.asarray(old["pa_radius"], dtype=np.float64)
    new_radius = np.asarray(direct["pa_radius"], dtype=np.float64)
    report = {
        "status": "complete",
        "diagnostic": "direct-quadratic replay cache-only PA pre-NQC comparison",
        "run": {"host": socket.gethostname(), "platform": platform.platform(),
                "cwd": str(Path.cwd()), "command": " ".join(sys.argv),
                "elapsed_seconds": finite(time.perf_counter() - started)},
        "inputs": {"direct_cache": {"path": str(DIRECT_CACHE), "sha256": sha256(DIRECT_CACHE)},
                   "direct_report": {"path": str(DIRECT_REPORT), "sha256": sha256(DIRECT_REPORT)},
                   "old_cache": {"path": str(OLD), "sha256": sha256(OLD)},
                   "old_report": {"path": str(OLD_REPORT), "sha256": sha256(OLD_REPORT)},
                   "public_pa_arrays": {"path": str(PA), "sha256": sha256(PA)},
                   "public_pa_report": {"path": str(PA_REPORT), "sha256": sha256(PA_REPORT)}},
        "grid_contract": {"grid": "existing public PA-only uniform elapsed grid",
                          "samples": int(grid.size), "start_M": finite(grid[0]),
                          "end_M": finite(grid[-1]), "dt_M": finite(grid[1] - grid[0]),
                          "exact_public_grid_reused": True, "direct_pa_knots": int(new_t.size),
                          "no_extrapolation": True, "grid_hash": arr_hash(grid)},
        "methods": {"interpolation": "natural cubic amplitude and saved lifted phase separately",
                    "positive_projection_scale": finite(scale), "alignment": False,
                    "fits_or_tuning": False, "public_call": False, "full_native_ode": False,
                    "runtime_edit": False},
        "nqc_isolation": {"direct_pa_prefix": "saved nqc_original amplitude/phase first 41 knots",
                          "public_pa_contract": "PA-only route stops at PA endpoint and skips BBH NQC/ringdown",
                          "limitation": "comparison uses cached public PA output and replay arrays only",
                          "contract_basis": "coordinator supplied sanitized public binary behavior contract; no source inspected"},
        "metrics_on_public_pa_grid": {
            "direct_pre_nqc_over_public_pa_only": metrics(direct_pre, public),
            "direct_pre_nqc_over_old_analytic_pre_nqc": metrics(direct_pre, old_pre),
            "old_analytic_pre_nqc_over_public_pa_only": metrics(old_pre, public),
        },
        "pa_endpoint_and_phase_changes": {
            "pa_time_M": endpoint_change(new_t, old_t),
            "pa_phase_radians": endpoint_change(new_phase, old_phase),
            "pa_radius_M": endpoint_change(new_radius, old_radius),
            "pa_phase_grid": {"max_abs_delta": finite(np.max(np.abs(new_phase - old_phase))),
                              "rms_delta": finite(np.sqrt(np.mean((new_phase - old_phase) ** 2)))},
            "pa_time_grid": {"max_abs_delta": finite(np.max(np.abs(new_t - old_t))),
                             "rms_delta": finite(np.sqrt(np.mean((new_t - old_t) ** 2)))},
        },
        "prefix_knot_changes": {"nqc_original_time_M": endpoint_change(
                                      np.asarray(direct["nqc_original_time"][:new_t.size]),
                                      np.asarray(old["nqc_original_time"][:old_t.size])),
                                 "nqc_original_amplitude": endpoint_change(new_amp_knots,
                                      np.asarray(old["nqc_original_amplitude"][:old_t.size])),
                                 "nqc_original_phase_radians": endpoint_change(new_phase_knots,
                                      np.asarray(old["nqc_original_phase"][:old_t.size]))},
        "source_hashes": {"script": sha256(Path(__file__)),
                          "numerical_attachment": sha256(ROOT / "src/torchwave/approximants/eob/numerical_attachment.py")},
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "output": str(OUT),
                      "metrics": report["metrics_on_public_pa_grid"],
                      "pa_endpoint_and_phase_changes": report["pa_endpoint_and_phase_changes"]}, indent=2))


if __name__ == "__main__":
    main()

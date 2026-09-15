#!/usr/bin/env python3
"""Compare the clean analytic PA start with the value-only squared-root start.

This diagnostic reuses the existing conditioned-waveform harness for summaries,
loads its frozen ``SimInspiralFD`` public cache, and runs only the opt-in
``quadratic-value`` PA policy.  The analytic native caches are retained as the
baseline.  It is a bounded attribution experiment: no alignment, fitting,
support truncation, parameter search, or canonical routing is performed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from diagnose_teob_conditioned_waveform_clean import (
    _cache_arrays,
    _complex_series_metrics,
    _event_summary,
    _float,
    _native_case_summary,
    _nqc_summary,
    _sha256,
    _tensor_numpy,
)
from torchwave.approximants.eob.teob_adaptive_continuation_clean import (
    DEFAULT_ATOL,
    DEFAULT_CONTINUATION_DURATION,
    DEFAULT_FIXED_STEP_CAP,
    DEFAULT_MAX_ACCEPTED_STEPS,
    DEFAULT_MAX_DURATION,
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_MAX_REJECTED_STEPS,
    DEFAULT_RTOL,
)
from torchwave.approximants.eob.teob_conditioned_waveform_clean import conditioned_teob_fd


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-pa-initializer-fd-clean"
PUBLIC_FD_CACHE = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-conditioned-waveform-clean/public-fd-sim-inspiral-output.npz"
PUBLIC_FD_REPORT = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-conditioned-waveform-clean/public-fd-sim-inspiral-report.json"
ANALYTIC_CACHES = {
    "30+20": ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-conditioned-waveform-clean/native-output.npz",
    "1.4+1.4": ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-conditioned-waveform-clean/catalog-output.npz",
}
CASES = {
    "30+20": {"mass1": 30.0, "mass2": 20.0, "f_low": 20.0, "prefix": "30p20"},
    "1.4+1.4": {"mass1": 1.4, "mass2": 1.4, "f_low": 30.0, "prefix": "1_4p1_4"},
}


def _source_hashes() -> dict[str, str]:
    paths = {
        "script": Path(__file__).resolve(),
        "adapter": ROOT / "src/torchwave/approximants/eob/teob_conditioned_waveform_clean.py",
        "postadiabatic": ROOT / "src/torchwave/approximants/eob/teob_postadiabatic_clean.py",
        "adaptive_continuation": ROOT / "src/torchwave/approximants/eob/teob_adaptive_continuation_clean.py",
        "adaptive_driver": ROOT / "src/torchwave/approximants/eob/adaptive_driver_clean.py",
        "radiation": ROOT / "src/torchwave/approximants/eob/teob_radiation_clean.py",
        "state_mode": ROOT / "src/torchwave/approximants/eob/teob_waveform_state_clean.py",
        "nqc": ROOT / "src/torchwave/approximants/eob/teob_merger_nqc_clean.py",
        "ringdown": ROOT / "src/torchwave/approximants/eob/teob_ringdown_assembly_clean.py",
        "postmerger": ROOT / "src/torchwave/approximants/eob/teob_postmerger.py",
        "conditioning": ROOT / "src/torchwave/approximants/eob/td_fd_conditioning_clean.py",
        "projection": ROOT / "src/torchwave/approximants/eob/conditioned_waveform_clean.py",
        "remnant_uib": ROOT / "src/torchwave/approximants/eob/teob_uib2016_remnant.py",
        "remnant_hybrid": ROOT / "src/torchwave/approximants/eob/teob_healy2014_remnant_clean.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _np(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return _tensor_numpy(value)
    return np.asarray(value)


def _band_metrics(native_f: np.ndarray, native: np.ndarray, public_f: np.ndarray,
                  public: np.ndarray, lo: float, hi: float) -> dict[str, object]:
    count = min(native_f.size, public_f.size, native.size, public.size)
    exact = np.abs(native_f[:count] - public_f[:count]) <= 1.0e-12
    selected = exact & (native_f[:count] >= lo) & (native_f[:count] <= hi)
    indices = np.flatnonzero(selected)
    if not indices.size:
        raise ValueError(f"no exact frequency bins in {lo:g}--{hi:g} Hz")
    metrics = _complex_series_metrics(native[indices], public[indices])
    metrics.update({
        "frequency_index_start": int(indices[0]),
        "frequency_index_end": int(indices[-1]),
        "frequency_start_hz": float(native_f[indices[0]]),
        "frequency_end_hz": float(native_f[indices[-1]]),
        "bins": int(indices.size),
        "alignment": False,
        "optimization": False,
        "support_truncation": False,
    })
    return metrics


def _phase_amplitude(native: np.ndarray, public: np.ndarray) -> dict[str, float | int]:
    mask = np.abs(public) > 0.0
    ratio = native[mask] / public[mask]
    phase = np.angle(ratio)
    amplitude = np.abs(ratio) - 1.0
    return {
        "bins": int(mask.sum()),
        "phase_error_rad_rms": float(np.sqrt(np.mean(phase.square()))),
        "phase_error_rad_max_abs": float(np.max(np.abs(phase))),
        "amplitude_relative_error_rms": float(np.sqrt(np.mean(amplitude.square()))),
        "amplitude_relative_error_max_abs": float(np.max(np.abs(amplitude))),
    }


def _load_public(prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(PUBLIC_FD_CACHE, allow_pickle=False) as archive:
        return (
            np.asarray(archive[f"{prefix}_public_frequencies"], dtype=np.float64),
            np.asarray(archive[f"{prefix}_public_h_plus_fd"], dtype=np.complex128),
            np.asarray(archive[f"{prefix}_public_h_cross_fd"], dtype=np.complex128),
        )


def _compare(result: object, analytic: dict[str, np.ndarray], public: tuple[np.ndarray, ...],
             f_low: float) -> dict[str, object]:
    public_f, public_hp, public_hc = public
    native_f = _np(result.frequencies).reshape(-1).astype(np.float64)
    native_hp = _np(result.h_plus).reshape(-1).astype(np.complex128)
    native_hc = _np(result.h_cross).reshape(-1).astype(np.complex128)
    bands = {"fullband": (f_low, 1024.0), "catalog_subband": (f_low, 256.0 if f_low == 20.0 else 500.0)}
    output: dict[str, object] = {}
    for name, (lo, hi) in bands.items():
        output[name] = {
            "h_plus": _band_metrics(native_f, native_hp, public_f, public_hp, lo, hi),
            "h_cross": _band_metrics(native_f, native_hc, public_f, public_hc, lo, hi),
            "combined_polarizations": _band_metrics(
                np.concatenate((native_f, native_f)), np.concatenate((native_hp, native_hc)),
                np.concatenate((public_f, public_f)), np.concatenate((public_hp, public_hc)), lo, hi,
            ),
            "phase_amplitude": {
                "h_plus": _phase_amplitude(native_hp, public_hp),
                "h_cross": _phase_amplitude(native_hc, public_hc),
            },
        }
    output["analytic_baseline_delta"] = {}
    analytic_f = np.asarray(analytic["frequencies"], dtype=np.float64).reshape(-1)
    analytic_hp = np.asarray(analytic["h_plus_fd"], dtype=np.complex128).reshape(-1)
    analytic_hc = np.asarray(analytic["h_cross_fd"], dtype=np.complex128).reshape(-1)
    for name, (lo, hi) in bands.items():
        output["analytic_baseline_delta"][name] = {
            "h_plus": _band_metrics(native_f, native_hp, analytic_f, analytic_hp, lo, hi),
            "h_cross": _band_metrics(native_f, native_hc, analytic_f, analytic_hc, lo, hi),
            "combined_polarizations": _band_metrics(
                np.concatenate((native_f, native_f)), np.concatenate((native_hp, native_hc)),
                np.concatenate((analytic_f, analytic_f)), np.concatenate((analytic_hp, analytic_hc)), lo, hi,
            ),
        }
    return output


def main() -> int:
    VERIFY.mkdir(parents=True, exist_ok=True)
    report_path = VERIFY / "diagnostic.json"
    command = shlex.join([sys.executable, *sys.argv])
    source_before = _source_hashes()
    report: dict[str, object] = {
        "status": "running",
        "scope": "value-only PA initializer attribution; no full trajectory validation",
        "method": "clean conditioned_teob_fd with exact policies and frozen public SimInspiralFD cache",
        "policies": {
            "pa_initializer": "quadratic-value",
            "analytic_baseline": "frozen native cache generated with canonical analytic initializer",
            "radial_acceleration": "frozen-mobility",
            "horizon": "leading-order",
            "remnant": "uib2016",
            "nqc_targets": "published-2017",
            "pa": {"r_min": 14.0, "dr": 0.1, "pa_order": 8},
            "continuation": {"atol": DEFAULT_ATOL, "rtol": DEFAULT_RTOL, "fixed_step_cap": DEFAULT_FIXED_STEP_CAP,
                            "continuation_duration": DEFAULT_CONTINUATION_DURATION,
                            "max_rejected_steps": DEFAULT_MAX_REJECTED_STEPS, "max_evaluations": DEFAULT_MAX_EVALUATIONS,
                            "max_accepted_steps": DEFAULT_MAX_ACCEPTED_STEPS, "max_duration": DEFAULT_MAX_DURATION},
        },
        "run_control": {"host": socket.gethostname(), "cwd": os.getcwd(), "command": command,
                        "pid": os.getpid(), "python": sys.executable, "platform": platform.platform(),
                        "started_unix": time.time(), "log_path": str((VERIFY / "diagnostic.log").resolve()),
                        "expected_next_check": "inspect diagnostic.json and quadratic-*output.npz after process exit",
                        "stop_command": f"kill {os.getpid()}"},
        "public_cache": {"path": str(PUBLIC_FD_CACHE.resolve()), "sha256": _sha256(PUBLIC_FD_CACHE),
                        "report_path": str(PUBLIC_FD_REPORT.resolve()), "report_sha256": _sha256(PUBLIC_FD_REPORT),
                        "method": "SimInspiralFD", "read_only": True},
        "analytic_caches": {label: {"path": str(path.resolve()), "sha256": _sha256(path), "read_only": True}
                            for label, path in ANALYTIC_CACHES.items()},
        "source_sha256_before": source_before,
        "cases": {},
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    continuation_kwargs = dict(report["policies"]["continuation"])
    pa_kwargs = {"r_min": 14.0, "dr": 0.1, "pa_order": 8}
    try:
        for label, case in CASES.items():
            print(f"stage: quadratic-value {label} evolution", flush=True)
            result = conditioned_teob_fd(
                case["mass1"], case["mass2"], f_low=case["f_low"], f_max=1024.0, delta_f=0.5,
                distance=100.0, inclination=0.0, phic=0.0, tc=0.0,
                radial_acceleration_policy="frozen-mobility", horizon_policy="leading-order",
                remnant_policy="uib2016", nqc_target_policy="published-2017",
                pa_initializer_policy="quadratic-value", pa_kwargs=pa_kwargs,
                continuation_kwargs=continuation_kwargs, dtype=torch.float64, device="cpu",
            )
            public = _load_public(case["prefix"])
            with np.load(ANALYTIC_CACHES[label], allow_pickle=False) as archive:
                analytic = {name: np.asarray(archive[name]).copy() for name in ("frequencies", "h_plus_fd", "h_cross_fd")}
            cache_path = VERIFY / f"quadratic-{case['prefix']}-output.npz"
            cache_arrays = _cache_arrays(result)
            np.savez_compressed(cache_path, **cache_arrays)
            report["cases"][label] = {
                "request": {"mass1_msun": case["mass1"], "mass2_msun": case["mass2"], "f_low_hz": case["f_low"],
                            "f_max_hz": 1024.0, "delta_f_hz": 0.5, "distance_mpc": 100.0,
                            "inclination": 0.0, "phic": 0.0, "tc_seconds": 0.0, "spins": [0.0, 0.0], "tides": [0.0, 0.0]},
                "native": _native_case_summary(result),
                "pa": {"initializer_policy": result.pa.initializer_policy,
                       "initializer_value_only": result.pa.metadata["initializer_value_only"],
                       "endpoint_time": _float(result.pa.endpoint_time), "endpoint_phase": _float(result.pa.endpoint_phase),
                       "samples": int(result.pa.time.numel())},
                "nqc": _nqc_summary(result.nqc),
                "ringdown_events": _event_summary(result.ringdown.events, total_mass_seconds=result.total_mass_seconds),
                "comparison": _compare(result, analytic, public, case["f_low"]),
                "cache": {"path": str(cache_path.resolve()), "sha256": _sha256(cache_path),
                          "bytes": cache_path.stat().st_size, "keys": sorted(cache_arrays), "read_only_after_write": True},
            }
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"stage: {label} cache written", flush=True)
        report["interpretation"] = {
            "purpose": "component attribution only; compare numerical effect of the PA initializer",
            "no_alignment_or_fitting": True,
            "no_support_truncation": True,
            "no_parameter_search": True,
            "full_trajectory_validation": False,
            "promotion": "not supported: quadratic-value uses the lower zero-G' squared root under no_grad and has no autograd contract; analytic remains default",
            "gradient_limit": "zero-spin r=14..20 value probe has finite values but autograd is not promised and must not be repaired with a straight-through derivative",
        }
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["source_sha256_after"] = _source_hashes()
    report["source_hashes_unchanged"] = report["source_sha256_before"] == report["source_sha256_after"]
    report["elapsed_seconds"] = time.perf_counter() - started
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
                      "report": str(report_path.resolve()), "source_hashes_unchanged": report["source_hashes_unchanged"]}, sort_keys=True), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

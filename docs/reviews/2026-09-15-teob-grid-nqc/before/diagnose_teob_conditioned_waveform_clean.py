#!/usr/bin/env python3
"""Run a bounded native clean TEOB TD-to-FD composition probe.

The two attribution policies are required on the command line.  The probe
then composes the reviewed PA profile, adaptive-to-fixed continuation, clean
TEOB state carrier, published 2017 NQC target provider, UIB2016 remnant, the
clean ringdown assembler, and the existing TD conditioning helper.  It is an
opt-in diagnostic; no canonical driver is imported or changed.

The default numbers are a small 30+20 solar-mass smoke configuration.  The
public binary fixture is read only for provenance and support-contract
context.  Its waveform is not aligned, fitted, or silently interpolated to
the native adapter grid.  The diagnostic also calls the public
``SimInspiralChooseFDWaveform`` API for the same 30+20 request and for a
separate 1.4+1.4 catalog case, comparing only exact frequency indices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import lal
import lalsimulation as lalsim

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
from torchwave.approximants.eob.teob_conditioned_waveform_clean import (
    HORIZON_POLICIES,
    PA_INITIALIZER_POLICIES,
    RADIAL_ACCELERATION_POLICIES,
    conditioned_teob_fd,
    project_teob22_direct,
    project_teob22_generic,
)


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-conditioned-waveform-clean"
DEFAULT_OUTPUT = VERIFY / "diagnostic.json"
DEFAULT_CACHE = VERIFY / "native-output.npz"
DEFAULT_CATALOG_CACHE = VERIFY / "catalog-output.npz"
DEFAULT_FD_CACHE = VERIFY / "public-fd-output.npz"
DEFAULT_LOG = VERIFY / "run.log"
PUBLIC_FIXTURE = Path("/tmp/torchwave-public-teob-q1p5-r20-dt0p5M.npz")
PUBLIC_METADATA = Path("/tmp/torchwave-public-teob-q1p5-r20-dt0p5M.json")
PUBLIC_FIXTURE_SHA256 = "09eaa8b00288c94947b2bb4b40ada7780dc51ac98491a48cbb8f60ed50ab686c"
PUBLIC_SUPPORT_REPORT = ROOT / (
    "docs/reviews/2026-09-13-parity-review/verification/"
    "teob-conditioned-root-review/public-support-start-conditioning.json"
)
PUBLIC_PROJECTION_REPORT = ROOT / (
    "docs/reviews/2026-09-13-parity-review/verification/"
    "teob-conditioned-root-review/public-projection-contract.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().reshape(-1)[0])
    return int(value)


def _tensor_numpy(value: torch.Tensor) -> np.ndarray:
    # ``torch.conj`` may leave a lazy conjugate bit on the returned view;
    # resolve it before serializing a reusable cache.
    return value.detach().cpu().resolve_conj().resolve_neg().numpy()


def _shape(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.shape]
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.shape]
    return None


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def _series_summary(value: torch.Tensor, *, complex_series: bool = False) -> dict[str, Any]:
    real = value.real if complex_series or value.is_complex() else value
    result: dict[str, Any] = {
        "shape": _shape(value),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "finite": _finite(value),
        "samples": int(value.shape[-1]) if value.ndim else 1,
    }
    if value.numel():
        result["first"] = _float(value.reshape(-1)[0].real if value.is_complex() else value.reshape(-1)[0])
        result["last"] = _float(value.reshape(-1)[-1].real if value.is_complex() else value.reshape(-1)[-1])
        result["abs_min"] = _float(value.abs().min())
        result["abs_max"] = _float(value.abs().max())
        if real.numel():
            result["real_min"] = _float(real.min())
            result["real_max"] = _float(real.max())
    return result


def _source_paths() -> dict[str, Path]:
    """Return only clean sources used by this adapter and this diagnostic."""

    return {
        "script": Path(__file__).resolve(),
        "adapter": ROOT / "src/torchwave/approximants/eob/teob_conditioned_waveform_clean.py",
        "postadiabatic": ROOT / "src/torchwave/approximants/eob/teob_postadiabatic_clean.py",
        "adaptive_continuation": ROOT / "src/torchwave/approximants/eob/teob_adaptive_continuation_clean.py",
        "adaptive_driver": ROOT / "src/torchwave/approximants/eob/adaptive_driver_clean.py",
        "vacuum_driver": ROOT / "src/torchwave/approximants/eob/teob_vacuum_driver_clean.py",
        "trajectory": ROOT / "src/torchwave/approximants/eob/teob_trajectory.py",
        "radial_acceleration": ROOT / "src/torchwave/approximants/eob/teob_radial_acceleration_clean.py",
        "radiation": ROOT / "src/torchwave/approximants/eob/teob_radiation_clean.py",
        "state_mode": ROOT / "src/torchwave/approximants/eob/teob_waveform_state_clean.py",
        "hamiltonian": ROOT / "src/torchwave/approximants/teobresums.py",
        "nqc": ROOT / "src/torchwave/approximants/eob/teob_merger_nqc_clean.py",
        "nqc_basis": ROOT / "src/torchwave/approximants/eob/teob_nqc_clean.py",
        "ringdown": ROOT / "src/torchwave/approximants/eob/teob_ringdown_assembly_clean.py",
        "postmerger": ROOT / "src/torchwave/approximants/eob/teob_postmerger.py",
        "uib_remnant": ROOT / "src/torchwave/approximants/eob/teob_uib2016_remnant.py",
        "conditioning": ROOT / "src/torchwave/approximants/eob/td_fd_conditioning_clean.py",
        "projection": ROOT / "src/torchwave/approximants/eob/conditioned_waveform_clean.py",
        "assembly_contract": ROOT / "docs/reviews/2026-09-13-parity-review/teob-merger-assembly-contract.md",
        "conditioning_contract": ROOT / "docs/reviews/2026-09-13-parity-review/td-fd-conditioning-contract.md",
    }


def _fixture_info(path: Path, metadata_path: Path) -> dict[str, Any]:
    path = path.resolve()
    metadata_path = metadata_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"public fixture is unavailable: {path}")
    digest = _sha256(path)
    if digest != PUBLIC_FIXTURE_SHA256:
        raise ValueError(
            f"public fixture hash mismatch: expected {PUBLIC_FIXTURE_SHA256}, got {digest}"
        )
    with np.load(path, allow_pickle=False) as archive:
        shapes = {name: [int(item) for item in archive[name].shape] for name in archive.files}
        dtypes = {name: str(archive[name].dtype) for name in archive.files}
        signal_norm = {
            name: float(np.linalg.norm(archive[name]))
            for name in ("hp", "hc")
            if name in archive.files
        }
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "keys": sorted(shapes),
        "shapes": shapes,
        "dtypes": dtypes,
        "signal_norm": signal_norm,
    }
    if metadata_path.is_file():
        result["metadata_path"] = str(metadata_path)
        result["metadata_sha256"] = _sha256(metadata_path)
        result["metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        result["metadata_path"] = str(metadata_path)
        result["metadata_unavailable"] = True
    return result


def _public_fd_options() -> lal.Dict:
    """Build the public zero-tide, (2,+2)-only TEOBResumS options."""

    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(params, 0)
    mode_array = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 2, 2)
    lalsim.SimInspiralWaveformParamsInsertModeArray(params, mode_array)
    return params


def _public_fd_call(
    *,
    mass1: float,
    mass2: float,
    f_low: float,
    f_max: float,
    delta_f: float,
    distance: float,
    inclination: float,
    phic: float,
    tc: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the public FD API and return its arrays plus exact metadata.

    SimInspiralChooseFDWaveform has no separate tc positional argument.  The
    physical requests here use tc=0; the value is retained in metadata to
    make that convention explicit.
    """

    if tc != 0.0:
        raise ValueError("the public FD comparison supports only tc=0")
    started = time.perf_counter()
    approximant = lalsim.GetApproximantFromString("TEOBResumS")
    fd_implemented = bool(lalsim.SimInspiralImplementedFDApproximants(approximant))
    try:
        hp, hc = lalsim.SimInspiralChooseFDWaveform(
            mass1 * lal.MSUN_SI,
            mass2 * lal.MSUN_SI,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            distance * 1.0e6 * lal.PC_SI,
            inclination,
            phic,
            0.0,
            0.0,
            0.0,
            delta_f,
            f_low,
            f_max,
            f_low,
            _public_fd_options(),
            approximant,
        )
    except Exception as exc:
        return (
            {
                "status": "unavailable",
                "api": "SimInspiralChooseFDWaveform",
                "approximant": "TEOBResumS",
                "approximant_enum": int(approximant),
                "public_fd_implemented": fd_implemented,
                "lal_version": getattr(lal, "__version__", None),
                "lalsimulation_version": getattr(lalsim, "__version__", None),
                "mass1_msun": float(mass1),
                "mass2_msun": float(mass2),
                "f_low_hz": float(f_low),
                "f_max_hz": float(f_max),
                "delta_f_hz": float(delta_f),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            },
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.complex128),
            np.empty(0, dtype=np.complex128),
        )
    hp_values = np.asarray(hp.data.data, dtype=np.complex128).copy()
    hc_values = np.asarray(hc.data.data, dtype=np.complex128).copy()
    delta = float(hp.deltaF)
    f0 = float(getattr(hp, "f0", 0.0))
    frequencies = f0 + delta * np.arange(hp_values.size, dtype=np.float64)
    metadata = {
        "status": "ok",
        "api": "SimInspiralChooseFDWaveform",
        "approximant": "TEOBResumS",
        "approximant_enum": int(approximant),
        "public_fd_implemented": fd_implemented,
        "lal_version": getattr(lal, "__version__", None),
        "lalsimulation_version": getattr(lalsim, "__version__", None),
        "mass1_msun": float(mass1),
        "mass2_msun": float(mass2),
        "chi1": [0.0, 0.0, 0.0],
        "chi2": [0.0, 0.0, 0.0],
        "lambda1": 0.0,
        "lambda2": 0.0,
        "distance_mpc": float(distance),
        "inclination": float(inclination),
        "phic": float(phic),
        "tc_seconds": float(tc),
        "delta_f_hz": delta,
        "f0_hz": f0,
        "f_low_hz": float(f_low),
        "f_max_hz": float(f_max),
        "f_ref_hz": float(f_low),
        "mode_array": "[(2,+2)]",
        "pn_tidal_order": 0,
        "samples": int(hp_values.size),
        "hp_epoch_seconds": float(hp.epoch),
        "hc_epoch_seconds": float(hc.epoch),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return metadata, frequencies, hp_values, hc_values


def _complex_series_metrics(native: np.ndarray, public: np.ndarray) -> dict[str, Any]:
    native = np.asarray(native, dtype=np.complex128)
    public = np.asarray(public, dtype=np.complex128)
    native_norm = float(np.linalg.norm(native))
    public_norm = float(np.linalg.norm(public))
    residual_norm = float(np.linalg.norm(native - public))
    if public_norm == 0.0:
        relative_l2 = None
    else:
        relative_l2 = residual_norm / public_norm
    if native_norm == 0.0 or public_norm == 0.0:
        overlap = None
        mismatch = None
    else:
        overlap = float(np.real(np.vdot(public, native)) / (public_norm * native_norm))
        mismatch = 1.0 - overlap
    return {
        "signal_norm_native": native_norm,
        "signal_norm_public": public_norm,
        "residual_norm": residual_norm,
        "relative_l2_unaligned": relative_l2,
        "complex_norm_ratio_native_over_public": (
            None if public_norm == 0.0 else native_norm / public_norm
        ),
        "flat_complex_overlap": overlap,
        "mismatch_unaligned": mismatch,
    }


def _public_fd_comparison(
    result: Any,
    public_frequencies: np.ndarray,
    public_hp: np.ndarray,
    public_hc: np.ndarray,
    *,
    f_low: float,
    delta_f: float,
) -> dict[str, Any]:
    """Compare exact shared FD bins without alignment or optimization."""

    if not np.asarray(public_frequencies).size:
        return {
            "status": "unavailable",
            "reason": "public SimInspiralChooseFDWaveform returned no series",
        }
    native_frequencies = _tensor_numpy(result.frequencies).reshape(-1).astype(np.float64)
    public_frequencies = np.asarray(public_frequencies, dtype=np.float64).reshape(-1)
    public_hp = np.asarray(public_hp, dtype=np.complex128).reshape(-1)
    public_hc = np.asarray(public_hc, dtype=np.complex128).reshape(-1)
    native_hp = _tensor_numpy(result.h_plus).reshape(-1).astype(np.complex128)
    native_hc = _tensor_numpy(result.h_cross).reshape(-1).astype(np.complex128)
    common = min(native_frequencies.size, public_frequencies.size,
                native_hp.size, native_hc.size, public_hp.size, public_hc.size)
    if common == 0:
        raise ValueError("native/public FD comparison has no common samples")
    axis_delta = np.abs(native_frequencies[:common] - public_frequencies[:common])
    exact_axis = axis_delta <= max(1.0e-12, 1.0e-12 * abs(delta_f))
    selected = exact_axis & (native_frequencies[:common] >= float(f_low))
    indices = np.flatnonzero(selected)
    if indices.size == 0:
        raise ValueError("native/public FD comparison has no exact bins at or above f_low")
    hp_metrics = _complex_series_metrics(native_hp[indices], public_hp[indices])
    hc_metrics = _complex_series_metrics(native_hc[indices], public_hc[indices])
    combined_metrics = _complex_series_metrics(
        np.concatenate((native_hp[indices], native_hc[indices])),
        np.concatenate((public_hp[indices], public_hc[indices])),
    )

    def support_counts(values: np.ndarray) -> dict[str, int]:
        selected_values = values[indices]
        return {
            "total_nonzero_bins": int(np.count_nonzero(values)),
            "compared_nonzero_bins": int(np.count_nonzero(selected_values)),
            "compared_zero_bins": int(selected_values.size - np.count_nonzero(selected_values)),
        }

    return {
        "status": "ok",
        "frequency_index_rule": "same integer indices with exact native/public frequency and f >= f_low",
        "alignment": "none",
        "optimization": "none",
        "native_samples_total": int(native_frequencies.size),
        "public_samples_total": int(public_frequencies.size),
        "common_samples_considered": int(common),
        "exact_frequency_axis": bool(np.all(exact_axis)),
        "maximum_common_frequency_difference_hz": float(axis_delta.max()),
        "compared_index_start": int(indices[0]),
        "compared_index_end": int(indices[-1]),
        "compared_bins": int(indices.size),
        "compared_frequency_start_hz": float(native_frequencies[indices[0]]),
        "compared_frequency_end_hz": float(native_frequencies[indices[-1]]),
        "requested_f_low_hz": float(f_low),
        "requested_delta_f_hz": float(delta_f),
        "native_support": {
            "h_plus": support_counts(native_hp),
            "h_cross": support_counts(native_hc),
        },
        "public_support": {
            "h_plus": support_counts(public_hp),
            "h_cross": support_counts(public_hc),
        },
        "h_plus": hp_metrics,
        "h_cross": hc_metrics,
        "combined_polarizations": combined_metrics,
        "mismatch_definition": (
            "1 - Re(vdot(public, native))/(||public||_2 ||native||_2) over the "
            "compared complex bins; flat weighting and no time/phase maximization"
        ),
    }


def _fd_cache_arrays(
    cases: list[tuple[str, Any, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Save native and public FD arrays for every paired diagnostic case."""

    arrays: dict[str, np.ndarray] = {}
    for label, result, frequencies, hp, hc in cases:
        key = label.replace("+", "p").replace(".", "_")
        arrays[f"{key}_native_frequencies"] = _tensor_numpy(result.frequencies).copy()
        arrays[f"{key}_native_h_plus_fd"] = _tensor_numpy(result.h_plus).copy()
        arrays[f"{key}_native_h_cross_fd"] = _tensor_numpy(result.h_cross).copy()
        arrays[f"{key}_public_frequencies"] = np.asarray(frequencies).copy()
        arrays[f"{key}_public_h_plus_fd"] = np.asarray(hp).copy()
        arrays[f"{key}_public_h_cross_fd"] = np.asarray(hc).copy()
    return arrays


def _native_case_summary(result: Any) -> dict[str, Any]:
    """Summarize a second native case while its full stages live in its cache."""

    return {
        "support": _support_summary(result.support),
        "sampling": _sampling_summary(result.sampling, result.total_mass_seconds),
        "native_start": {
            "omega_start": _float(result.omega_start),
            "initial_radius": _float(result.initial_radius),
            "formula": "max(r_min, omega_start**(-2/3))",
            "coordinate_epoch": _float(result.coordinate_epoch),
            "physical_epoch_seconds": _float(result.physical_epoch_seconds),
            "epoch_policy": "first accepted continuation orbital-frequency-decrease event",
            "continuation_peak_event_time": _float(result.continuation.peak_event_time),
        },
        "pa": {
            "status": result.pa.status,
            "samples": int(result.pa.time.numel()),
            "start_radius": _float(result.pa.start_radius),
            "endpoint_time": _float(result.pa.endpoint_time),
            "grid_spacing": result.pa.grid_spacing,
        },
        "continuation": {
            "status": result.continuation.status,
            "samples": int(result.continuation.time.numel()),
            "valid_samples": int(result.continuation.valid.sum()),
            "adaptive_samples": int(result.continuation.adaptive_time.numel()),
            "fixed_samples": int(result.continuation.fixed_time.numel()),
            "peak_event_index": result.continuation.peak_event_index,
            "peak_event_time": _float(result.continuation.peak_event_time),
            "peak_event_frequency": _float(result.continuation.peak_event_frequency),
            "endpoint_time": _float(result.continuation.endpoint_time),
            "phase_seam_error": _float(result.continuation.phase_seam_error),
            "statistics": _statistics_summary(result.continuation.statistics),
        },
        "nqc": _nqc_summary(result.nqc),
        "ringdown_events": _event_summary(
            result.ringdown.events, total_mass_seconds=result.total_mass_seconds
        ),
        "stages": {
            "mode_22": _series_summary(result.mode_22, complex_series=True),
            "mode_2m2": _series_summary(result.mode_2m2, complex_series=True),
            "h_plus_td": _series_summary(result.h_plus_td),
            "h_cross_td": _series_summary(result.h_cross_td),
            "h_plus_fd": _series_summary(result.h_plus),
            "h_cross_fd": _series_summary(result.h_cross),
        },
        "signal_norm": {
            "native_td_h_plus": _float(torch.linalg.vector_norm(result.h_plus_td)),
            "native_td_h_cross": _float(torch.linalg.vector_norm(result.h_cross_td)),
            "native_fd_h_plus": _float(torch.linalg.vector_norm(result.h_plus)),
            "native_fd_h_cross": _float(torch.linalg.vector_norm(result.h_cross)),
        },
    }


def _json_file_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.resolve()), "available": False}
    return {
        "path": str(path.resolve()),
        "available": True,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "content": json.loads(path.read_text(encoding="utf-8")),
    }


def _support_summary(support: Any) -> dict[str, Any]:
    fields = (
        "total_mass_seconds", "f_low", "f_safe", "start_frequency",
        "support_time", "first_taper_duration", "chirp_time", "plunge_time",
        "ringdown_time", "plunge_ringdown_time", "qnm_dimensionless_frequency",
        "qnm_quality", "qnm_damping_time",
    )
    return {name: _float(getattr(support, name)) for name in fields}


def _sampling_summary(sampling: Any, total_mass_seconds: torch.Tensor) -> dict[str, Any]:
    return {
        "n_time": int(sampling.n_time),
        "n_frequency": int(sampling.n_frequency),
        "n_nyquist_intervals": int(sampling.n_nyquist_intervals),
        "f_max_requested": _float(sampling.f_max_requested),
        "f_nyquist": _float(sampling.f_nyquist),
        "delta_f": _float(sampling.delta_f),
        "dt_seconds": _float(sampling.dt),
        "dt_geometric_initial_mass": _float(sampling.dt / total_mass_seconds),
        "automatic": bool(sampling.automatic),
    }


def _statistics_summary(statistics: Any) -> dict[str, Any]:
    return {
        "adaptive_accepted_steps": int(statistics.adaptive_accepted_steps),
        "adaptive_rejected_steps": int(statistics.adaptive_rejected_steps),
        "adaptive_evaluations": int(statistics.adaptive_evaluations),
        "fixed_accepted_steps": int(statistics.fixed_accepted_steps),
        "fixed_rejected_steps": int(statistics.fixed_rejected_steps),
        "fixed_evaluations": int(statistics.fixed_evaluations),
        "accepted_steps": int(statistics.accepted_steps),
        "rejected_steps": int(statistics.rejected_steps),
        "evaluations": int(statistics.evaluations),
        "termination": statistics.termination,
    }


def _event_summary(events: Any, *, total_mass_seconds: torch.Tensor) -> dict[str, Any]:
    fields = (
        "peak_knot_index", "peak_time", "peak_first_derivative",
        "peak_second_derivative", "merger_time", "match_time",
        "match_time_final_mass", "strictly_earlier_index", "attachment_index",
        "attachment_time", "attachment_time_final_mass",
    )
    result = {name: _int(getattr(events, name)) if "index" in name else _float(getattr(events, name)) for name in fields}
    result["peak_time_seconds_from_coordinate_start"] = _float(events.peak_time * total_mass_seconds)
    result["merger_time_seconds_from_coordinate_start"] = _float(events.merger_time * total_mass_seconds)
    result["match_time_seconds_from_coordinate_start"] = _float(events.match_time * total_mass_seconds)
    result["attachment_time_seconds_from_coordinate_start"] = _float(events.attachment_time * total_mass_seconds)
    result["peak_polynomial_coefficients"] = _tensor_numpy(events.peak_polynomial_coefficients).tolist()
    return result


def _nqc_summary(nqc: Any) -> dict[str, Any]:
    return {
        "target_provenance": nqc.target_provenance,
        "targets": {
            "amplitude": _float(nqc.targets.amplitude),
            "amplitude_derivative": _float(nqc.targets.amplitude_derivative),
            "frequency": _float(nqc.targets.frequency),
            "frequency_derivative": _float(nqc.targets.frequency_derivative),
        },
        "amplitude_coefficients": _tensor_numpy(nqc.coefficients.amplitude).tolist(),
        "phase_coefficients": _tensor_numpy(nqc.coefficients.phase).tolist(),
        "matching_index_auxiliary": int(nqc.matching_index),
        "cut_start_index_original": int(nqc.cut_start_index),
        "cut_last_index_original": int(nqc.cut_last_index),
        "t_peak_grid": _float(nqc.t_peak_grid),
        "t_nqc": _float(nqc.t_nqc),
        "auxiliary_spacing": _float(nqc.auxiliary_spacing),
        "auxiliary_samples": int(nqc.auxiliary.time.numel()),
        "amplitude_condition": _float(nqc.match.amplitude_condition),
        "phase_condition": _float(nqc.match.phase_condition),
        "amplitude_residual": _tensor_numpy(nqc.match.amplitude_residual).tolist(),
        "phase_residual": _tensor_numpy(nqc.match.phase_residual).tolist(),
        "negative_auxiliary_corrected_amplitude_samples": int(nqc.auxiliary_negative_amplitude.sum()),
    }


def _projection_summary(result: Any) -> dict[str, Any]:
    rows = []
    for inclination, phic in ((0.0, 0.0), (0.7, 0.0), (0.0, 0.4), (0.7, 0.4)):
        generic = project_teob22_generic(
            result.mode_22,
            result.mode_2m2,
            mass1=result.mass1,
            mass2=result.mass2,
            distance=result.distance,
            inclination=inclination,
            phic=phic,
        )
        direct = project_teob22_direct(
            result.mode_22,
            result.mode_2m2,
            mass1=result.mass1,
            mass2=result.mass2,
            distance=result.distance,
            inclination=inclination,
            phic=phic,
        )
        plus_residual = torch.max(torch.abs(generic[0] + direct[0]))
        cross_residual = torch.max(torch.abs(generic[1] + direct[1]))
        rows.append({
            "inclination": inclination,
            "phic": phic,
            "samples": int(generic[0].shape[-1]),
            "max_abs_plus_sum": _float(plus_residual),
            "max_abs_cross_sum": _float(cross_residual),
            "l2_plus_sum": _float(torch.linalg.vector_norm(generic[0] + direct[0])),
            "l2_cross_sum": _float(torch.linalg.vector_norm(generic[1] + direct[1])),
        })
    return {
        "relation": "direct = -generic for the same clean 22 pair and extrinsic parameters",
        "rows": rows,
        "max_abs_sum": max(max(row["max_abs_plus_sum"], row["max_abs_cross_sum"]) for row in rows),
    }


def _cache_arrays(result: Any) -> dict[str, np.ndarray]:
    """Copy every reusable stage array without serializing Python objects."""

    pa = result.pa
    continuation = result.continuation
    nqc = result.nqc
    ringdown = result.ringdown
    arrays: dict[str, np.ndarray] = {}

    def add(name: str, value: Any) -> None:
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            arrays[name] = _tensor_numpy(value).copy()
        else:
            arrays[name] = np.asarray(value)

    for name, value in {
        "frequencies": result.frequencies,
        "h_plus_fd": result.h_plus,
        "h_cross_fd": result.h_cross,
        "td_time": result.td_time,
        "mode_22": result.mode_22,
        "mode_2m2": result.mode_2m2,
        "h_plus_td": result.h_plus_td,
        "h_cross_td": result.h_cross_td,
        "pa_radius": pa.radius,
        "pa_time": pa.time,
        "pa_phase": pa.phase,
        "pa_pr_star": pa.pr_star,
        "pa_p_phi": pa.p_phi,
        "pa_frequency": pa.omega,
        "continuation_time": continuation.time,
        "continuation_state": continuation.state,
        "continuation_frequency": continuation.frequency,
        "continuation_valid": continuation.valid,
        "adaptive_time": continuation.adaptive_time,
        "adaptive_state": continuation.adaptive_state,
        "adaptive_frequency": continuation.adaptive_frequency,
        "fixed_time": continuation.fixed_time,
        "fixed_state": continuation.fixed_state,
        "fixed_frequency": continuation.fixed_frequency,
        "nqc_original_time": nqc.original.time,
        "nqc_original_state": nqc.original.state,
        "nqc_original_amplitude": nqc.original.amplitude,
        "nqc_original_phase": nqc.original.phase,
        "nqc_original_orbital_frequency": nqc.original.orbital_frequency,
        "nqc_original_radial_acceleration": nqc.original.radial_acceleration,
        "nqc_original_basis_amplitude": nqc.original.basis_amplitude,
        "nqc_original_basis_phase": nqc.original.basis_phase,
        "nqc_original_corrected_amplitude": nqc.original.corrected_amplitude,
        "nqc_original_corrected_phase": nqc.original.corrected_phase,
        "nqc_auxiliary_time": nqc.auxiliary.time,
        "nqc_auxiliary_state": nqc.auxiliary.state,
        "nqc_auxiliary_amplitude": nqc.auxiliary.amplitude,
        "nqc_auxiliary_phase": nqc.auxiliary.phase,
        "nqc_auxiliary_orbital_frequency": nqc.auxiliary.orbital_frequency,
        "nqc_auxiliary_radial_acceleration": nqc.auxiliary.radial_acceleration,
        "nqc_auxiliary_basis_amplitude": nqc.auxiliary.basis_amplitude,
        "nqc_auxiliary_basis_phase": nqc.auxiliary.basis_phase,
        "nqc_auxiliary_corrected_amplitude": nqc.auxiliary.corrected_amplitude,
        "nqc_auxiliary_corrected_phase": nqc.auxiliary.corrected_phase,
        "nqc_amplitude_coefficients": nqc.coefficients.amplitude,
        "nqc_phase_coefficients": nqc.coefficients.phase,
        "nqc_target_amplitude": nqc.targets.amplitude,
        "nqc_target_amplitude_derivative": nqc.targets.amplitude_derivative,
        "nqc_target_frequency": nqc.targets.frequency,
        "nqc_target_frequency_derivative": nqc.targets.frequency_derivative,
        "ringdown_time": ringdown.time,
        "ringdown_mode": ringdown.mode,
        "ringdown_amplitude": ringdown.amplitude,
        "ringdown_phase": ringdown.phase,
        "ringdown_joined_time": ringdown.joined_time,
        "ringdown_joined_mode": ringdown.joined_mode,
        "ringdown_joined_amplitude": ringdown.joined_amplitude,
        "ringdown_joined_phase": ringdown.joined_phase,
        "ringdown_extended_time": ringdown.extended_time,
        "ringdown_extended_mode": ringdown.extended_mode,
        "ringdown_extended_amplitude": ringdown.extended_amplitude,
        "ringdown_extended_phase": ringdown.extended_phase,
        "ringdown_tail_mode": ringdown.ringdown_mode,
        "ringdown_auxiliary_time": ringdown.auxiliary_time,
        "ringdown_auxiliary_amplitude": ringdown.auxiliary_amplitude,
        "ringdown_auxiliary_phase": ringdown.auxiliary_phase,
        "ringdown_auxiliary_orbital_frequency": ringdown.auxiliary_orbital_frequency,
        "radial_acceleration": nqc.original.radial_acceleration,
    }.items():
        add(name, value)

    events = ringdown.events
    for name in (
        "peak_knot_index", "peak_time", "peak_first_derivative", "peak_second_derivative",
        "merger_time", "match_time", "match_time_final_mass", "strictly_earlier_index",
        "attachment_index", "attachment_time", "attachment_time_final_mass",
    ):
        add(f"event_{name}", getattr(events, name))
    for name, value in {
        "omega_start": result.omega_start,
        "initial_radius": result.initial_radius,
        "total_mass_seconds": result.total_mass_seconds,
        "coordinate_epoch": result.coordinate_epoch,
        "physical_epoch_seconds": result.physical_epoch_seconds,
        "sampling_dt_seconds": result.sampling.dt,
        "sampling_delta_f": result.sampling.delta_f,
        "sampling_f_nyquist": result.sampling.f_nyquist,
        "support_start_frequency": result.support.start_frequency,
        "support_time": result.support.support_time,
        "support_first_taper_duration": result.support.first_taper_duration,
        "nqc_matching_index": np.asarray(nqc.matching_index, dtype=np.int64),
    }.items():
        add(name, value)
    return arrays


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radial-acceleration-policy", required=True, choices=RADIAL_ACCELERATION_POLICIES)
    parser.add_argument("--horizon-policy", required=True, choices=HORIZON_POLICIES)
    parser.add_argument("--pa-initializer-policy", default="analytic", choices=PA_INITIALIZER_POLICIES)
    parser.add_argument("--mass1", type=float, default=30.0)
    parser.add_argument("--mass2", type=float, default=20.0)
    parser.add_argument("--f-low", type=float, default=20.0)
    parser.add_argument("--f-max", type=float, default=1024.0)
    parser.add_argument("--delta-f", type=float, default=0.5)
    parser.add_argument("--distance", type=float, default=100.0)
    parser.add_argument("--inclination", type=float, default=0.0)
    parser.add_argument("--phic", type=float, default=0.0)
    parser.add_argument("--tc", type=float, default=0.0)
    parser.add_argument("--r-min", type=float, default=14.0)
    parser.add_argument("--dr", type=float, default=0.1)
    parser.add_argument("--pa-order", type=int, default=8)
    parser.add_argument("--fixed-step-cap", type=float, default=DEFAULT_FIXED_STEP_CAP)
    parser.add_argument("--continuation-duration", type=float, default=DEFAULT_CONTINUATION_DURATION)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--max-rejected-steps", type=int, default=DEFAULT_MAX_REJECTED_STEPS)
    parser.add_argument("--max-evaluations", type=int, default=DEFAULT_MAX_EVALUATIONS)
    parser.add_argument("--max-accepted-steps", type=int, default=DEFAULT_MAX_ACCEPTED_STEPS)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--public", type=Path, default=PUBLIC_FIXTURE)
    parser.add_argument("--public-metadata", type=Path, default=PUBLIC_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--catalog-cache", type=Path, default=DEFAULT_CATALOG_CACHE)
    parser.add_argument("--fd-cache", type=Path, default=DEFAULT_FD_CACHE)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.catalog_cache.parent.mkdir(parents=True, exist_ok=True)
    args.fd_cache.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    started = time.perf_counter()
    source_paths = _source_paths()
    source_before = {name: _sha256(path) for name, path in source_paths.items()}
    command = shlex.join([sys.executable, *sys.argv])
    report: dict[str, Any] = {
        "status": "running",
        "diagnostic_scope": (
            "native opt-in clean TEOB scalar zero-spin zero-tide mode-22 "
            "TD-to-FD composition; no full trajectory validation"
        ),
        "run_control": {
            "host": socket.gethostname(),
            "cwd": os.getcwd(),
            "command": command,
            "pid": os.getpid(),
            "python": sys.executable,
            "platform": platform.platform(),
            "device": "cpu",
            "started_unix": time.time(),
            "log_path": str(args.log_path.resolve()),
            "expected_next_check": "inspect diagnostic.json and native-output.npz after process exit",
            "stop_command": f"kill {os.getpid()}",
        },
        "probe": {
            "mass1_msun": args.mass1,
            "mass2_msun": args.mass2,
            "f_low_hz": args.f_low,
            "f_max_hz": args.f_max,
            "delta_f_hz": args.delta_f,
            "distance_mpc": args.distance,
            "inclination": args.inclination,
            "phic": args.phic,
            "tc_seconds": args.tc,
            "chi1z": 0.0,
            "chi2z": 0.0,
            "tides": "zero",
            "mode": [2, 2],
            "radial_acceleration_policy": args.radial_acceleration_policy,
            "horizon_policy": args.horizon_policy,
            "nqc_target_policy": "published-2017",
            "nqc_target_source": "clean published 2017 provider",
            "r_min": args.r_min,
            "dr": args.dr,
            "pa_order": args.pa_order,
            "fixed_step_cap": args.fixed_step_cap,
            "continuation_duration": args.continuation_duration,
            "atol": args.atol,
            "rtol": args.rtol,
            "max_rejected_steps": args.max_rejected_steps,
            "max_evaluations": args.max_evaluations,
            "max_accepted_steps": args.max_accepted_steps,
            "max_duration": args.max_duration,
            "radius_provenance": "max(r_min, omega_start**(-2/3)) from native support start",
            "conditioning_policy": "existing clean td_fd_conditioning_clean with native dt and requested f_max/delta_f",
            "alignment": "none",
            "fitting": "none",
        },
        "source_sha256_before": source_before,
        "output_path": str(args.output.resolve()),
        "cache_path": str(args.cache.resolve()),
        "catalog_cache_path": str(args.catalog_cache.resolve()),
        "fd_cache_path": str(args.fd_cache.resolve()),
        "public_fixture": None,
        "public_support_component_evidence": _json_file_info(PUBLIC_SUPPORT_REPORT),
        "public_projection_component_evidence": _json_file_info(PUBLIC_PROJECTION_REPORT),
    }

    # Make the running state visible before the potentially lengthy native
    # evolution.  A redirected stdout log still carries stage heartbeats.
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        report["public_fixture"] = _fixture_info(args.public, args.public_metadata)
        print("stage: public fixture provenance loaded", flush=True)

        pa_kwargs = {
            "r_min": args.r_min,
            "dr": args.dr,
            "pa_order": args.pa_order,
        }
        continuation_kwargs = {
            "atol": args.atol,
            "rtol": args.rtol,
            "fixed_step_cap": args.fixed_step_cap,
            "continuation_duration": args.continuation_duration,
            "max_rejected_steps": args.max_rejected_steps,
            "max_evaluations": args.max_evaluations,
            "max_accepted_steps": args.max_accepted_steps,
            "max_duration": args.max_duration,
        }
        print("stage: PA and adaptive/fixed TEOB evolution", flush=True)
        result = conditioned_teob_fd(
            args.mass1,
            args.mass2,
            f_low=args.f_low,
            f_max=args.f_max,
            delta_f=args.delta_f,
            distance=args.distance,
            inclination=args.inclination,
            phic=args.phic,
            tc=args.tc,
            radial_acceleration_policy=args.radial_acceleration_policy,
            horizon_policy=args.horizon_policy,
            pa_initializer_policy=args.pa_initializer_policy,
            pa_kwargs=pa_kwargs,
            continuation_kwargs=continuation_kwargs,
            dtype=torch.float64,
            device="cpu",
        )
        print("stage: NQC, remnant, ringdown, conditioning complete", flush=True)

        report["support"] = _support_summary(result.support)
        report["sampling"] = _sampling_summary(result.sampling, result.total_mass_seconds)
        report["native_start"] = {
            "omega_start": _float(result.omega_start),
            "initial_radius": _float(result.initial_radius),
            "formula": "max(r_min, omega_start**(-2/3))",
            "coordinate_epoch": _float(result.coordinate_epoch),
            "physical_epoch_seconds": _float(result.physical_epoch_seconds),
            "epoch_policy": "first accepted continuation orbital-frequency-decrease event",
            "continuation_peak_event_time": _float(result.continuation.peak_event_time),
        }
        report["pa"] = {
            "status": result.pa.status,
            "samples": int(result.pa.time.numel()),
            "start_radius": _float(result.pa.start_radius),
            "endpoint_time": _float(result.pa.endpoint_time),
            "endpoint_state": _tensor_numpy(result.pa.endpoint_state).tolist(),
            "grid_spacing": result.pa.grid_spacing,
        }
        report["continuation"] = {
            "status": result.continuation.status,
            "samples": int(result.continuation.time.numel()),
            "valid_samples": int(result.continuation.valid.sum()),
            "adaptive_samples": int(result.continuation.adaptive_time.numel()),
            "fixed_samples": int(result.continuation.fixed_time.numel()),
            "first_ode_step": _float(result.continuation.first_ode_step),
            "adaptive_next_step": _float(result.continuation.adaptive_next_step),
            "fixed_step": _float(result.continuation.fixed_step),
            "peak_event_index": result.continuation.peak_event_index,
            "peak_event_time": _float(result.continuation.peak_event_time),
            "peak_event_frequency": _float(result.continuation.peak_event_frequency),
            "endpoint_time": _float(result.continuation.endpoint_time),
            "endpoint_state": _tensor_numpy(result.continuation.endpoint_state).tolist(),
            "phase_seam_error": _float(result.continuation.phase_seam_error),
            "statistics": _statistics_summary(result.continuation.statistics),
        }
        report["nqc"] = _nqc_summary(result.nqc)
        report["ringdown"] = {
            "output": _series_summary(result.ringdown.mode, complex_series=True),
            "joined_samples": int(result.ringdown.joined_time.numel()),
            "extended_samples": int(result.ringdown.extended_time.numel()),
            "extension_samples": _int(result.ringdown.extension_samples),
            "auxiliary_spacing": _float(result.ringdown.auxiliary_spacing),
            "ringdown_phase0": _float(result.ringdown.ringdown_phase0),
            "auxiliary_amplitude_negative_samples": int(result.ringdown.auxiliary_amplitude_negative.sum()),
            "events": _event_summary(result.ringdown.events, total_mass_seconds=result.total_mass_seconds),
            "coordinate_start_definition": "ringdown.epoch = first returned coordinate time",
        }
        report["stages"] = {
            "mode_22": _series_summary(result.mode_22, complex_series=True),
            "mode_2m2": _series_summary(result.mode_2m2, complex_series=True),
            "h_plus_td": _series_summary(result.h_plus_td),
            "h_cross_td": _series_summary(result.h_cross_td),
            "h_plus_fd": _series_summary(result.h_plus),
            "h_cross_fd": _series_summary(result.h_cross),
        }
        report["signal_norm"] = {
            "native_td_h_plus": _float(torch.linalg.vector_norm(result.h_plus_td)),
            "native_td_h_cross": _float(torch.linalg.vector_norm(result.h_cross_td)),
            "native_fd_h_plus": _float(torch.linalg.vector_norm(result.h_plus)),
            "native_fd_h_cross": _float(torch.linalg.vector_norm(result.h_cross)),
            "public_fixture_td_h_plus": report["public_fixture"].get("signal_norm", {}).get("hp"),
            "public_fixture_td_h_cross": report["public_fixture"].get("signal_norm", {}).get("hc"),
            "comparison": (
                "Euclidean norms are reported for scale context only; native and public "
                "arrays have different support/sampling and are not residual-compared"
            ),
        }
        report["projection_contract"] = _projection_summary(result)
        report["metadata"] = {
            "mode": [2, 2],
            "radial_acceleration_policy": result.radial_acceleration_policy,
            "horizon_policy": result.horizon_policy,
            "nqc_target_policy": result.nqc_target_policy,
            "nqc_target_provenance": result.nqc_target_provenance,
            "total_mass_seconds": _float(result.total_mass_seconds),
            "mass1_msun": _float(result.mass1),
            "mass2_msun": _float(result.mass2),
            "distance_mpc": _float(result.distance),
        }

        print("stage: writing reusable native cache", flush=True)
        cache_arrays = _cache_arrays(result)
        np.savez_compressed(args.cache, **cache_arrays)
        report["cache"] = {
            "path": str(args.cache.resolve()),
            "sha256": _sha256(args.cache),
            "bytes": args.cache.stat().st_size,
            "keys": sorted(cache_arrays),
            "array_shapes": {name: [int(item) for item in value.shape] for name, value in cache_arrays.items()},
            "read_only_after_write": True,
        }
        fd_cases: list[tuple[str, Any, np.ndarray, np.ndarray, np.ndarray]] = []
        public_fd_cases: dict[str, Any] = {}

        print("stage: public 30+20 SimInspiralChooseFDWaveform comparison", flush=True)
        public_meta, public_frequencies, public_hp, public_hc = _public_fd_call(
            mass1=args.mass1,
            mass2=args.mass2,
            f_low=args.f_low,
            f_max=args.f_max,
            delta_f=args.delta_f,
            distance=args.distance,
            inclination=args.inclination,
            phic=args.phic,
            tc=args.tc,
        )
        fd_cases.append(("30+20", result, public_frequencies, public_hp, public_hc))
        public_fd_cases["30+20"] = {
            "request": {
                "mass1_msun": args.mass1,
                "mass2_msun": args.mass2,
                "f_low_hz": args.f_low,
                "f_max_hz": args.f_max,
                "delta_f_hz": args.delta_f,
                "distance_mpc": args.distance,
                "inclination": args.inclination,
                "phic": args.phic,
                "tc_seconds": args.tc,
                "chi1": [0.0, 0.0, 0.0],
                "chi2": [0.0, 0.0, 0.0],
                "lambda1": 0.0,
                "lambda2": 0.0,
            },
            "public": public_meta,
            "comparison": _public_fd_comparison(
                result, public_frequencies, public_hp, public_hc,
                f_low=args.f_low, delta_f=args.delta_f,
            ),
            "native_cache": str(args.cache.resolve()),
        }

        print("stage: native 1.4+1.4 catalog case", flush=True)
        catalog_mass1 = 1.4
        catalog_mass2 = 1.4
        catalog_f_low = 30.0
        catalog = conditioned_teob_fd(
            catalog_mass1,
            catalog_mass2,
            f_low=catalog_f_low,
            f_max=args.f_max,
            delta_f=args.delta_f,
            distance=args.distance,
            inclination=args.inclination,
            phic=args.phic,
            tc=args.tc,
            radial_acceleration_policy=args.radial_acceleration_policy,
            horizon_policy=args.horizon_policy,
            pa_initializer_policy=args.pa_initializer_policy,
            pa_kwargs=pa_kwargs,
            continuation_kwargs=continuation_kwargs,
            dtype=torch.float64,
            device="cpu",
        )
        catalog_cache_arrays = _cache_arrays(catalog)
        np.savez_compressed(args.catalog_cache, **catalog_cache_arrays)
        report["catalog_case"] = {
            "request": {
                "mass1_msun": catalog_mass1,
                "mass2_msun": catalog_mass2,
                "f_low_hz": catalog_f_low,
                "f_max_hz": args.f_max,
                "delta_f_hz": args.delta_f,
                "distance_mpc": args.distance,
                "inclination": args.inclination,
                "phic": args.phic,
                "tc_seconds": args.tc,
                "chi1": [0.0, 0.0, 0.0],
                "chi2": [0.0, 0.0, 0.0],
                "lambda1": 0.0,
                "lambda2": 0.0,
                "radial_acceleration_policy": args.radial_acceleration_policy,
                "horizon_policy": args.horizon_policy,
            },
            "native": _native_case_summary(catalog),
            "cache": {
                "path": str(args.catalog_cache.resolve()),
                "sha256": _sha256(args.catalog_cache),
                "bytes": args.catalog_cache.stat().st_size,
                "keys": sorted(catalog_cache_arrays),
                "array_shapes": {
                    name: [int(item) for item in value.shape]
                    for name, value in catalog_cache_arrays.items()
                },
                "read_only_after_write": True,
            },
        }

        print("stage: public 1.4+1.4 SimInspiralChooseFDWaveform comparison", flush=True)
        catalog_public_meta, catalog_public_frequencies, catalog_public_hp, catalog_public_hc = _public_fd_call(
            mass1=catalog_mass1,
            mass2=catalog_mass2,
            f_low=catalog_f_low,
            f_max=args.f_max,
            delta_f=args.delta_f,
            distance=args.distance,
            inclination=args.inclination,
            phic=args.phic,
            tc=args.tc,
        )
        fd_cases.append((
            "1.4+1.4",
            catalog,
            catalog_public_frequencies,
            catalog_public_hp,
            catalog_public_hc,
        ))
        public_fd_cases["1.4+1.4"] = {
            "request": {
                "mass1_msun": catalog_mass1,
                "mass2_msun": catalog_mass2,
                "f_low_hz": catalog_f_low,
                "f_max_hz": args.f_max,
                "delta_f_hz": args.delta_f,
                "distance_mpc": args.distance,
                "inclination": args.inclination,
                "phic": args.phic,
                "tc_seconds": args.tc,
                "chi1": [0.0, 0.0, 0.0],
                "chi2": [0.0, 0.0, 0.0],
                "lambda1": 0.0,
                "lambda2": 0.0,
            },
            "public": catalog_public_meta,
            "comparison": _public_fd_comparison(
                catalog, catalog_public_frequencies, catalog_public_hp, catalog_public_hc,
                f_low=catalog_f_low, delta_f=args.delta_f,
            ),
            "native_cache": str(args.catalog_cache.resolve()),
        }
        fd_cache_arrays = _fd_cache_arrays(fd_cases)
        np.savez_compressed(args.fd_cache, **fd_cache_arrays)
        report["public_fd_comparisons"] = {
            "api": "SimInspiralChooseFDWaveform",
            "approximant": "TEOBResumS",
            "options": {
                "mode_array": "[(2,+2)]",
                "PNTidalOrder": 0,
                "spins": "zero aligned spins",
                "lambda1_lambda2": [0.0, 0.0],
                "distance_mpc": args.distance,
                "inclination": args.inclination,
                "phic": args.phic,
                "tc_seconds": args.tc,
            },
            "cases": public_fd_cases,
            "cache": {
                "path": str(args.fd_cache.resolve()),
                "sha256": _sha256(args.fd_cache),
                "bytes": args.fd_cache.stat().st_size,
                "keys": sorted(fd_cache_arrays),
                "array_shapes": {
                    name: [int(item) for item in value.shape]
                    for name, value in fd_cache_arrays.items()
                },
                "read_only_after_write": True,
            },
            "comparison_policy": (
                "Unaligned exact-index comparison; no time/phase optimization, "
                "interpolation, support truncation, or parameter search."
            ),
        }
        report["comparison_scope"] = {
            "public_fixture_waveform": (
                "provenance only; no same-grid waveform residual is reported because "
                "the fixture uses dt=0.5M and f_min=14.45059083415894 while this "
                "native smoke uses support-derived f_low=20Hz and Fourier dt"
            ),
            "public_fixture_radius": "metadata records 19.999999999999996 from its public inverse-Kepler start",
            "native_radius": "derived from this run's support start; no hard-coded public radius",
            "public_support_conditioning": "existing paired public-start evidence is linked above; no oracle enters runtime",
            "public_fd_waveform": (
                "SimInspiralChooseFDWaveform with approximant TEOBResumS is compared "
                "on the exact common frequency indices at or above each request's f_low; "
                "the public binary is diagnostic-only."
            ),
            "full_trajectory_validation": False,
            "alignment": False,
            "parameter_search": False,
        }
        report["status"] = "complete"
    except Exception as exc:  # pragma: no cover - exercised by failed smoke runs
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "reason": getattr(exc, "reason", None),
        }

    source_after = {name: _sha256(path) for name, path in source_paths.items()}
    report["source_sha256_after"] = source_after
    report["source_hashes_unchanged"] = source_before == source_after
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "elapsed_seconds": report["elapsed_seconds"],
        "output": str(args.output.resolve()),
        "cache": report.get("cache", {}).get("sha256"),
        "source_hashes_unchanged": report["source_hashes_unchanged"],
    }, sort_keys=True), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

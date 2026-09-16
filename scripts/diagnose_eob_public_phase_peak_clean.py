"""Isolate public EOB phase peaks and unshifted clean-mode residuals.

This diagnostic consumes only the four previously saved black-box outputs from
``diagnose_eob_clean_hamiltonian_public.py``.  The phase in row two of the
returned post-peak state is differentiated by fixed local polynomial stencils
(degrees 4, 6, and 8).  Those stencils are refinement/cross-check diagnostics;
none is selected by a waveform residual.

The clean comparison uses a freshly evaluated SEOBNRv4 Hamiltonian, published
flux, and published (2,2) mode.  A public epoch is metadata: phase-peak times
remain in the returned initial-origin coordinate so they can be compared with
``-epoch``.  Shared full-trajectory samples are compared to the public mode in
absolute complex scale, with zero time and phase shifts.  The optional clean
NQC attachment is evaluated from the published targets and never fitted to the
public waveform.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC
from torchwave.approximants.eob.attachment import attach_seobnrv4_22
from torchwave.approximants.eob.events import orbital_frequency_peak
from torchwave.approximants.eob.nqc import nqc_22_factor
from torchwave.approximants.eob.published_waveform import (
    published_mode22_and_frequency,
)
from torchwave.approximants.eob.remnant import (
    eob_final_mass_fraction,
    hofmann_final_spin,
)
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/tmp")
OUT = REPO_ROOT / (
    "docs/reviews/2026-09-13-parity-review/verification/eob/"
    "eob-public-phase-peak-clean.json"
)
LABELS = ("dt_0p25M", "dt_0p5M", "dt_1M", "dt_1_4096s")
DERIVATIVE_DEGREES = (4, 6, 8)
TOTAL_MASS = 50.0
MSEC = TOTAL_MASS * MSUN_SEC
DISTANCE_MPC = 100.0
MODE_SCALE = MSEC * C_SI / (DISTANCE_MPC * MPC_SI)

# The wrapper is included because it defines the public black-box call and
# the raw-array schema.  The remaining files are the clean physics path used
# below, including transitive vector-field/mode dependencies.
DEPENDENCIES = (
    "scripts/diagnose_eob_public_phase_peak_clean.py",
    "scripts/diagnose_eob_clean_hamiltonian_public.py",
    "src/torchwave/constants.py",
    "src/torchwave/approximants/eob/attachment.py",
    "src/torchwave/approximants/eob/events.py",
    "src/torchwave/approximants/eob/hamiltonian.py",
    "src/torchwave/approximants/eob/integrator.py",
    "src/torchwave/approximants/eob/nqc.py",
    "src/torchwave/approximants/eob/published_waveform.py",
    "src/torchwave/approximants/eob/radiation_reaction.py",
    "src/torchwave/approximants/eob/remnant.py",
    "src/torchwave/approximants/eob/seobnrv4_amplitude22.py",
    "src/torchwave/approximants/eob/seobnrv4_flux.py",
    "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
    "src/torchwave/approximants/eob/seobnrv4_mode22.py",
    "src/torchwave/approximants/eob/factorized_modes.py",
    "src/torchwave/approximants/eob/merger_ringdown.py",
    "src/torchwave/approximants/eob/waveform.py",
    "src/torchwave/approximants/kerr_qnm_generated.py",
)


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    return float(value)


def _library_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    for module_name, distribution_name in (("lal", "lalsuite"), ("lalsimulation", "lalsuite")):
        try:
            module = __import__(module_name)
        except Exception:
            versions[module_name] = "unavailable"
            continue
        value = getattr(module, "__version__", None)
        if value is None:
            try:
                value = metadata.version(distribution_name)
            except metadata.PackageNotFoundError:
                value = "unknown"
        versions[module_name] = str(value)
    return versions


def _load_dataset(label: str) -> dict:
    path = DATA_DIR / f"torchwave-public-eob-{label}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; generate it through public_call in "
            "diagnose_eob_clean_hamiltonian_public.py"
        )
    with np.load(path) as data:
        # Copies prevent an NPZ-backed view from being mutated by Torch or a
        # future clean helper.
        state = np.asarray(data["state"], dtype=np.float64).copy()
        postpeak = np.asarray(data["postpeak_state"], dtype=np.float64).copy()
        mode = np.asarray(data["mode22"], dtype=np.complex128).copy()
        epoch_s = float(np.asarray(data["epoch_s"]).item())
        dt_s = float(np.asarray(data["dt_s"]).item())
    if state.ndim != 2 or state.shape[0] != 5:
        raise ValueError(f"{label}: expected state shape (5,N), got {state.shape}")
    if postpeak.ndim != 2 or postpeak.shape[0] != 5:
        raise ValueError(
            f"{label}: expected postpeak_state shape (5,N), got {postpeak.shape}"
        )
    if mode.ndim != 1 or mode.size == 0:
        raise ValueError(f"{label}: expected a nonempty one-dimensional mode")
    if not np.isfinite(state).all() or not np.isfinite(postpeak).all():
        raise ValueError(f"{label}: state contains non-finite values")
    if not np.isfinite(mode).all():
        raise ValueError(f"{label}: mode contains non-finite values")
    if not np.all(np.diff(state[0]) > 0) or not np.all(np.diff(postpeak[0]) > 0):
        raise ValueError(f"{label}: state times are not strictly increasing")
    if not np.isfinite(epoch_s) or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError(f"{label}: invalid epoch or dt metadata")
    dt_M = dt_s / MSEC
    state_grid = np.arange(state.shape[1], dtype=np.float64) * dt_M
    state_grid_error = float(np.max(np.abs(state[0] - state_grid)))
    return {
        "label": label,
        "path": path,
        "sha256": sha(path),
        "size_bytes": int(path.stat().st_size),
        "state": state,
        "postpeak_state": postpeak,
        "mode": mode,
        "epoch_s": epoch_s,
        "epoch_M": epoch_s / MSEC,
        "dt_s": dt_s,
        "dt_M": dt_M,
        "state_grid_error_M": state_grid_error,
    }


def _setup_clean_physics() -> dict:
    dtype = torch.float64
    zero = torch.tensor(0.0, dtype=dtype)
    nu = torch.tensor(0.24, dtype=dtype)
    delta = torch.tensor(0.2, dtype=dtype)
    ham = SEOBNRv4Hamiltonian(nu, zero, zero, delta=delta)
    flux = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8")
    mass_fraction = eob_final_mass_fraction(
        torch.tensor(0.6, dtype=dtype), torch.tensor(0.4, dtype=dtype), zero, zero
    )
    final_spin = hofmann_final_spin(
        torch.tensor(0.6, dtype=dtype), torch.tensor(0.4, dtype=dtype), zero, zero,
        fit_order=(3, 4),
    )
    qnm_real, qnm_damp = dimensionless_qnm(2, 2, 0, final_spin)
    sigma = torch.complex(qnm_real / mass_fraction, -qnm_damp / mass_fraction)
    return {
        "dtype": dtype,
        "zero": zero,
        "nu": nu,
        "ham": ham,
        "flux": flux,
        "sigma": sigma,
    }


def _local_phase_fit(time: np.ndarray, phase: np.ndarray, index: int, degree: int):
    """Fit phase in a fixed degree+1 contiguous stencil around ``index``."""

    count = degree + 1
    if time.size < count:
        raise ValueError("phase series is shorter than the requested stencil")
    start = max(0, min(int(index) - degree // 2, time.size - count))
    stop = start + count
    nodes = np.asarray(time[start:stop], dtype=np.float64)
    ordinates = np.asarray(phase[start:stop], dtype=np.float64)
    span = float(nodes[-1] - nodes[0])
    if not np.isfinite(span) or span <= 0:
        raise ValueError("phase stencil needs strictly increasing times")
    origin = float(time[index])
    local = (nodes - origin) / span
    coefficients = np.polynomial.polynomial.polyfit(local, ordinates, degree)
    residual = np.polynomial.polynomial.polyval(local, coefficients) - ordinates
    vandermonde = np.polynomial.polynomial.polyvander(local, degree)
    return {
        "coefficients": coefficients,
        "start": start,
        "stop": stop,
        "origin": origin,
        "span": span,
        "local": local,
        "residual_l2": float(np.linalg.norm(residual)),
        "residual_max_abs": float(np.max(np.abs(residual))),
        "condition_number": float(np.linalg.cond(vandermonde)),
    }


def _phase_omega(time: np.ndarray, phase: np.ndarray, degree: int) -> np.ndarray:
    """Differentiate the returned orbital phase with a fixed local polynomial."""

    result = np.empty_like(time, dtype=np.float64)
    for index in range(time.size):
        fit = _local_phase_fit(time, phase, index, degree)
        derivative = np.polynomial.polynomial.polyder(fit["coefficients"])
        result[index] = float(
            np.polynomial.polynomial.polyval(0.0, derivative) / fit["span"]
        )
    return result


def _phase_peak(time: np.ndarray, phase: np.ndarray, degree: int) -> dict:
    """Return a predetermined-stencil derivative peak and its diagnostics."""

    omega = _phase_omega(time, phase, degree)
    sample_index = int(np.argmax(omega))
    fit = _local_phase_fit(time, phase, sample_index, degree)
    first = np.polynomial.polynomial.polyder(fit["coefficients"])
    second = np.polynomial.polynomial.polyder(first)
    third = np.polynomial.polynomial.polyder(second)
    candidates = []
    for root in np.polynomial.polynomial.polyroots(second):
        if abs(float(root.imag)) > 1e-8:
            continue
        z = float(root.real)
        if fit["local"][0] - 1e-12 <= z <= fit["local"][-1] + 1e-12:
            curvature = float(
                np.polynomial.polynomial.polyval(z, third) / fit["span"] ** 3
            )
            if curvature < 0:
                value = float(
                    np.polynomial.polynomial.polyval(z, first) / fit["span"]
                )
                candidates.append((value, z, curvature))
    # A missing local maximum is a preserved diagnostic failure, rather than a
    # fallback to a different stencil or a sample selected by residual size.
    if not candidates:
        return {
            "resolved": False,
            "degree": degree,
            "sample_index": sample_index,
            "sample_time_M_initial_origin": float(time[sample_index]),
            "sample_omega": float(omega[sample_index]),
            "candidate_count": 0,
            "stencil_start_index": int(fit["start"]),
            "stencil_end_index": int(fit["stop"] - 1),
            "stencil_span_M": fit["span"],
            "phase_fit_residual_l2": fit["residual_l2"],
            "phase_fit_residual_max_abs": fit["residual_max_abs"],
            "condition_number": fit["condition_number"],
        }
    value, local_root, curvature = max(candidates, key=lambda item: item[0])
    refined_time = fit["origin"] + local_root * fit["span"]
    right_spacing = (
        float(time[sample_index + 1] - time[sample_index])
        if sample_index + 1 < time.size
        else None
    )
    return {
        "resolved": True,
        "degree": degree,
        "sample_index": sample_index,
        "sample_time_M_initial_origin": float(time[sample_index]),
        "sample_omega": float(omega[sample_index]),
        "refined_time_M_initial_origin": float(refined_time),
        "refined_omega": value,
        "fractional_bin_offset": (
            float((refined_time - fit["origin"]) / right_spacing)
            if right_spacing
            else None
        ),
        "candidate_count": len(candidates),
        "curvature_domega_dt_M2": curvature,
        "stencil_start_index": int(fit["start"]),
        "stencil_end_index": int(fit["stop"] - 1),
        "stencil_span_M": fit["span"],
        "phase_fit_residual_l2": fit["residual_l2"],
        "phase_fit_residual_max_abs": fit["residual_max_abs"],
        "condition_number": fit["condition_number"],
    }


def _clean_event(time: np.ndarray, state: np.ndarray, physics: dict) -> dict:
    time_t = torch.from_numpy(np.asarray(time, dtype=np.float64).copy())
    state_t = torch.from_numpy(np.asarray(state[1:].T, dtype=np.float64).copy())
    with torch.inference_mode():
        clean_omega_t = physics["flux"].from_hamiltonian(state_t, physics["ham"])["omega"]
        event = orbital_frequency_peak(time_t, clean_omega_t)
    clean_omega = clean_omega_t.detach().cpu().numpy().astype(np.float64)
    sample_index = int(event.sample_index.detach().cpu().item())
    return {
        "time_M_initial_origin": _scalar(event.time),
        "frequency": _scalar(event.frequency),
        "curvature": _scalar(event.curvature),
        "sample_index": sample_index,
        "sample_time_M_initial_origin": float(time[sample_index]),
        "sample_omega": float(clean_omega[sample_index]),
        "omega": clean_omega,
    }


def _clean_omega(state: np.ndarray, physics: dict) -> np.ndarray:
    """Evaluate the clean published orbital frequency on a copied state grid."""

    state_t = torch.from_numpy(np.asarray(state[1:].T, dtype=np.float64).copy())
    with torch.inference_mode():
        omega_t = physics["flux"].from_hamiltonian(state_t, physics["ham"])["omega"]
    return omega_t.detach().cpu().numpy().astype(np.float64)


def _clean_mode(state: np.ndarray, physics: dict) -> tuple[np.ndarray, np.ndarray]:
    state_t = torch.from_numpy(np.asarray(state[1:].T, dtype=np.float64).copy())
    with torch.inference_mode():
        mode, frequency = published_mode22_and_frequency(
            state_t,
            physics["ham"],
            physics["flux"],
            spin_v6="none",
            include_pan_v8=True,
            include_positive_chi_540=True,
        )
    return (
        mode.detach().cpu().numpy().astype(np.complex128),
        frequency.detach().cpu().numpy().astype(np.float64),
    )


def _metric(clean: np.ndarray, public: np.ndarray, time: np.ndarray, label: str, lo: float, hi: float) -> dict:
    mask = (time >= lo) & (time < hi) & np.isfinite(clean.real) & np.isfinite(public.real)
    if not bool(mask.any()):
        return {
            "available": False,
            "label": label,
            "requested_support_M": [float(lo), float(hi)],
            "reason": "no shared samples in requested support",
            "time_shift_applied_M": 0.0,
            "phase_shift_applied_rad": 0.0,
        }
    c = clean[mask]
    p = public[mask]
    residual = c - p
    pnorm = float(np.linalg.norm(p))
    cnorm = float(np.linalg.norm(c))
    nz = np.abs(p) > np.finfo(np.float64).tiny
    phase = np.angle(c[nz] * np.conj(p[nz]))
    amp_ratio = np.abs(c[nz]) / np.abs(p[nz])
    return {
        "available": True,
        "label": label,
        "requested_support_M": [float(lo), float(hi)],
        "actual_support_M": [float(time[mask][0]), float(time[mask][-1])],
        "sample_count": int(mask.sum()),
        "time_shift_applied_M": 0.0,
        "phase_shift_applied_rad": 0.0,
        "clean_complex_l2": cnorm,
        "public_complex_l2": pnorm,
        "raw_complex_l2": float(np.linalg.norm(residual)),
        "raw_relative_l2_over_public": float(np.linalg.norm(residual) / pnorm) if pnorm else None,
        "raw_relative_l2_over_clean": float(np.linalg.norm(residual) / cnorm) if cnorm else None,
        "max_abs": float(np.max(np.abs(residual))),
        "max_abs_over_public_peak": float(np.max(np.abs(residual)) / np.max(np.abs(p))) if np.max(np.abs(p)) else None,
        "phase_rms_rad": float(np.sqrt(np.mean(phase**2))) if phase.size else None,
        "phase_max_abs_rad": float(np.max(np.abs(phase))) if phase.size else None,
        "amplitude_ratio_mean_clean_over_public": float(np.mean(amp_ratio)) if amp_ratio.size else None,
        "amplitude_ratio_min_clean_over_public": float(np.min(amp_ratio)) if amp_ratio.size else None,
        "amplitude_ratio_max_clean_over_public": float(np.max(amp_ratio)) if amp_ratio.size else None,
    }


def _shared_mode_comparison(
    data: dict,
    clean_mode: np.ndarray,
    match_time: float,
    *,
    mode_label: str,
    clean_excludes_nqc: bool,
) -> dict:
    state = data["state"]
    public = data["mode"]
    count = min(state.shape[1], public.size, clean_mode.size)
    time = state[0, :count].copy()
    clean = clean_mode[:count]
    public = public[:count]
    expected = np.arange(count, dtype=np.float64) * data["dt_M"]
    grid_error = float(np.max(np.abs(time - expected)))
    # Fixed bands make cross-grid changes interpretable and do not choose any
    # band or stencil by the size of its residual.
    bands = (
        ("0_to_1000M", 0.0, 1000.0),
        ("1000_to_3000M", 1000.0, 3000.0),
        ("3000_to_4500M", 3000.0, 4500.0),
        ("4500_to_match_minus_100M", 4500.0, match_time - 100.0),
        ("match_minus_100_to_matchM", match_time - 100.0, match_time),
        ("all_pre_match", 0.0, match_time),
    )
    checkpoints = {}
    for requested in (0.0, 1000.0, 3000.0, 4500.0, 4900.0, match_time - 100.0):
        index = int(np.argmin(np.abs(time - requested)))
        c, p = clean[index], public[index]
        residual = c - p
        phase = float(np.angle(c * np.conj(p))) if abs(p) > np.finfo(float).tiny else None
        checkpoints[str(requested)] = {
            "requested_time_M": float(requested),
            "sample_index": index,
            "sample_time_M": float(time[index]),
            "abs_residual": float(abs(residual)),
            "abs_residual_over_public": float(abs(residual) / abs(p)) if abs(p) else None,
            "phase_error_rad": phase,
            "amplitude_ratio_clean_over_public": float(abs(c) / abs(p)) if abs(p) else None,
        }
    return {
        "comparison": (
            f"fresh clean published {mode_label} versus public mode at identical "
            "initial-origin output indices; public epoch omitted from this "
            "shared-grid comparison"
        ),
        "mode_scale_100Mpc": MODE_SCALE,
        "shared_sample_count": int(count),
        "state_mode_grid_max_error_M": grid_error,
        "clean_mode_variant": mode_label,
        "clean_base_excludes_nqc": clean_excludes_nqc,
        "clean_nqc_coefficients_fitted_to_public": False,
        "public_mode_nqc_status": "public API output may include its own attachment/NQC",
        "bands": {
            name: _metric(clean, public, time, name, lo, hi)
            for name, lo, hi in bands
        },
        "checkpoints": checkpoints,
    }


def _nqc_summary(attached, base_mode: np.ndarray, time: np.ndarray, match_time: float) -> dict:
    corrected = attached.corrected_inspiral.detach().cpu().numpy().astype(np.complex128)
    coefficients = attached.calibration.coefficients
    amplitude = coefficients.amplitude.detach().cpu().numpy().astype(np.float64)
    phase = coefficients.phase.detach().cpu().numpy().astype(np.float64)
    mask = (time <= match_time) & (np.abs(base_mode) > np.finfo(np.float64).tiny)
    factor = corrected[mask] / base_mode[mask]
    return {
        "coefficient_source": "clean published NQC targets; no public-waveform fit",
        "amplitude_coefficients_a1_a2_a3": amplitude.tolist(),
        "phase_coefficients_b1_b2": phase.tolist(),
        "pre_match_sample_count": int(mask.sum()),
        "pre_match_factor_amplitude_range": [
            float(np.min(np.abs(factor))), float(np.max(np.abs(factor)))
        ] if factor.size else None,
        "pre_match_factor_phase_range_rad": [
            float(np.min(np.angle(factor))), float(np.max(np.angle(factor)))
        ] if factor.size else None,
    }


def _postmatch_residual(clean_mode: np.ndarray, public_mode: np.ndarray, grid: np.ndarray, match_time: float, epoch_M: float) -> dict:
    count = min(clean_mode.size, public_mode.size, grid.size)
    clean_mode = clean_mode[:count]
    public_mode = public_mode[:count]
    grid = grid[:count]
    mask = grid >= match_time
    if not bool(mask.any()):
        return {
            "available": False,
            "reason": "output grid ends before clean NQC match",
            "time_shift_applied_M": 0.0,
            "phase_shift_applied_rad": 0.0,
        }
    c, p = clean_mode[mask], public_mode[mask]
    residual = c - p
    nz = np.abs(p) > np.finfo(np.float64).tiny
    phase = np.angle(c[nz] * np.conj(p[nz]))
    return {
        "available": True,
        "comparison": "same relative output indices after keeping public epoch separate",
        "time_shift_applied_M": 0.0,
        "phase_shift_applied_rad": 0.0,
        "sample_count": int(mask.sum()),
        "support_M_initial_origin": [float(grid[mask][0]), float(grid[mask][-1])],
        "support_M_public_epoch_coordinate": [
            float(epoch_M + grid[mask][0]), float(epoch_M + grid[mask][-1])
        ],
        "raw_complex_l2": float(np.linalg.norm(residual)),
        "raw_relative_l2_over_public": float(np.linalg.norm(residual) / np.linalg.norm(p)) if np.linalg.norm(p) else None,
        "max_abs": float(np.max(np.abs(residual))),
        "max_abs_over_public_peak": float(np.max(np.abs(residual)) / np.max(np.abs(p))) if np.max(np.abs(p)) else None,
        "phase_rms_rad": float(np.sqrt(np.mean(phase**2))) if phase.size else None,
        "phase_max_abs_rad": float(np.max(np.abs(phase))) if phase.size else None,
        "amplitude_ratio_mean_clean_over_public": float(np.mean(np.abs(c[nz]) / np.abs(p[nz]))) if nz.any() else None,
    }


def _evaluate(data: dict, physics: dict) -> dict:
    post = data["postpeak_state"]
    post_time = post[0].copy()
    post_clean_event = _clean_event(post_time, post, physics)
    phase = post[2].copy()
    phase_peaks = {
        str(degree): _phase_peak(post_time, phase, degree)
        for degree in DERIVATIVE_DEGREES
    }
    minus_epoch = -data["epoch_M"]
    for record in phase_peaks.values():
        record["minus_epoch_M"] = minus_epoch
        if record["resolved"]:
            record["difference_refined_to_clean_event_M"] = (
                record["refined_time_M_initial_origin"]
                - post_clean_event["time_M_initial_origin"]
            )
            record["difference_refined_to_minus_epoch_M"] = (
                record["refined_time_M_initial_origin"] - minus_epoch
            )
            record["difference_sample_to_clean_event_M"] = (
                record["sample_time_M_initial_origin"]
                - post_clean_event["time_M_initial_origin"]
            )
            record["difference_sample_to_minus_epoch_M"] = (
                record["sample_time_M_initial_origin"] - minus_epoch
            )
        else:
            record["difference_refined_to_clean_event_M"] = None
            record["difference_refined_to_minus_epoch_M"] = None
            record["difference_sample_to_clean_event_M"] = None
            record["difference_sample_to_minus_epoch_M"] = None
    phase_sample_indices = {record["sample_index"] for record in phase_peaks.values()}
    phase_refined_times = [
        record["refined_time_M_initial_origin"]
        for record in phase_peaks.values()
        if record["resolved"]
    ]

    full_clean_mode, _ = _clean_mode(data["state"], physics)
    # Attachment uses the same relative output grid contract as the previous
    # clean diagnostic.  Its returned epoch is not sent into attachment.
    post_clean_mode, post_clean_frequency = _clean_mode(post, physics)
    post_state_t = torch.from_numpy(post[1:].T.copy())
    post_time_t = torch.from_numpy(post_time.copy())
    clean_omega_t = torch.from_numpy(post_clean_event["omega"].copy())
    output_grid = np.arange(data["mode"].size, dtype=np.float64) * data["dt_M"]
    attached = attach_seobnrv4_22(
        post_time_t,
        post_state_t,
        clean_omega_t,
        torch.from_numpy(post_clean_mode.copy()),
        torch.from_numpy(post_clean_frequency.copy()),
        physics["nu"],
        physics["zero"],
        physics["sigma"],
        torch.from_numpy(output_grid.copy()),
    )
    match_time = _scalar(attached.match_time)
    attached_mode = attached.mode.detach().cpu().numpy().astype(np.complex128) * MODE_SCALE
    clean_nqc = _nqc_summary(attached, post_clean_mode, post_time, match_time)
    residual = _postmatch_residual(
        attached_mode, data["mode"], output_grid, match_time, data["epoch_M"]
    )
    amplitude_index = int(np.argmax(np.abs(data["mode"])))
    base_comparison = _shared_mode_comparison(
        data,
        full_clean_mode * MODE_SCALE,
        match_time,
        mode_label="base mode (NQC excluded)",
        clean_excludes_nqc=True,
    )
    # The public output contains the public model's own NQC behavior over its
    # returned support.  Apply the clean attachment-derived factor as a fixed
    # diagnostic only; its coefficients never see the public mode.  The factor
    # is evaluated on the full clean state grid so the early comparison is
    # explicit about the NQC contribution rather than calling base-v-public a
    # carrier-only test.
    full_omega = _clean_omega(data["state"], physics)
    full_state_t = torch.from_numpy(data["state"][1:].T.copy())
    full_omega_t = torch.from_numpy(full_omega.copy())
    with torch.inference_mode():
        full_factor_t = nqc_22_factor(
            full_state_t[..., 0],
            full_state_t[..., 2],
            full_omega_t,
            attached.calibration.coefficients,
            sample_dims=1,
        )
    full_factor = full_factor_t.detach().cpu().numpy().astype(np.complex128)
    clean_nqc_mode = full_clean_mode * full_factor
    nqc_comparison = _shared_mode_comparison(
        data,
        clean_nqc_mode * MODE_SCALE,
        match_time,
        mode_label="base mode with fixed clean attachment-derived NQC",
        clean_excludes_nqc=False,
    )
    nqc_comparison["nqc_factor_amplitude_range"] = [
        float(np.min(np.abs(full_factor))), float(np.max(np.abs(full_factor)))
    ]
    nqc_comparison["nqc_factor_phase_range_rad"] = [
        float(np.min(np.angle(full_factor))), float(np.max(np.angle(full_factor)))
    ]
    return {
        "npz_path": str(data["path"]),
        "npz_sha256": data["sha256"],
        "npz_size_bytes": data["size_bytes"],
        "raw_arrays": {
            "keys": ["state", "postpeak_state", "mode22", "epoch_s", "dt_s"],
            "state_shape": list(data["state"].shape),
            "postpeak_state_shape": list(data["postpeak_state"].shape),
            "mode22_shape": list(data["mode"].shape),
            "loaded_arrays_copied": True,
        },
        "dt_s": data["dt_s"],
        "dt_M": data["dt_M"],
        "state_grid_error_M": data["state_grid_error_M"],
        "epoch_s": data["epoch_s"],
        "epoch_M": data["epoch_M"],
        "minus_epoch_M": minus_epoch,
        "postpeak_time_support_M_initial_origin": [
            float(post_time[0]), float(post_time[-1])
        ],
        "postpeak_spacing_M": {
            "minimum": float(np.min(np.diff(post_time))),
            "median": float(np.median(np.diff(post_time))),
            "maximum": float(np.max(np.diff(post_time))),
        },
        "clean_hamiltonian_omega_peak": {
            key: value
            for key, value in post_clean_event.items()
            if key != "omega"
        },
        "returned_phase_omega_peak": {
            "phase_row": 2,
            "phase_time_coordinate": "postpeak_state[0] in initial-origin M coordinates",
            "phase_offset_ignored_by_derivative": True,
            "phase_increasing": bool(np.all(np.diff(phase) > 0)),
            "phase_increment_range": [
                float(np.min(np.diff(phase))), float(np.max(np.diff(phase)))
            ],
            "derivative_orders": list(DERIVATIVE_DEGREES),
            "records": phase_peaks,
            "all_sample_peak_indices_equal": len(phase_sample_indices) == 1,
            "common_sample_peak_index": next(iter(phase_sample_indices)) if len(phase_sample_indices) == 1 else None,
            "refined_time_range_M": [min(phase_refined_times), max(phase_refined_times)] if phase_refined_times else None,
            "refined_time_range_span_M": (max(phase_refined_times) - min(phase_refined_times)) if phase_refined_times else None,
        },
        "public_amplitude_maximum_sample": {
            "sample_index": amplitude_index,
            "relative_output_time_M": float(output_grid[amplitude_index]),
            "public_epoch_coordinate_M": float(data["epoch_M"] + output_grid[amplitude_index]),
            "amplitude": float(abs(data["mode"][amplitude_index])),
            "coordinate_definition": "epoch_M + arange(mode_length) * dt_M",
        },
        "clean_nqc_match": {
            "time_M_initial_origin": match_time,
            "time_M_public_epoch_coordinate": float(data["epoch_M"] + match_time),
            "minus_clean_hamiltonian_event_M": float(match_time - post_clean_event["time_M_initial_origin"]),
        },
        "known_clean_nqc_correction_before_match": clean_nqc,
        "base_inspiral_mode_vs_public_unshifted": base_comparison,
        "nqc_corrected_inspiral_mode_vs_public_unshifted": nqc_comparison,
        "postmatch_unshifted_complex_residual": residual,
    }


def _range_summary(results: dict[str, dict]) -> dict:
    scalar_getters = {
        "epoch_M": lambda row: row["epoch_M"],
        "minus_epoch_M": lambda row: row["minus_epoch_M"],
        "clean_event_time_M": lambda row: row["clean_hamiltonian_omega_peak"]["time_M_initial_origin"],
        "clean_match_time_M": lambda row: row["clean_nqc_match"]["time_M_initial_origin"],
        "postmatch_raw_relative_l2": lambda row: row["postmatch_unshifted_complex_residual"].get("raw_relative_l2_over_public"),
        "base_mode_raw_relative_l2_0_to_1000": lambda row: row["base_inspiral_mode_vs_public_unshifted"]["bands"]["0_to_1000M"].get("raw_relative_l2_over_public"),
        "base_mode_raw_relative_l2_all_pre_match": lambda row: row["base_inspiral_mode_vs_public_unshifted"]["bands"]["all_pre_match"].get("raw_relative_l2_over_public"),
        "base_mode_raw_relative_l2_match_minus_100_to_match": lambda row: row["base_inspiral_mode_vs_public_unshifted"]["bands"]["match_minus_100_to_matchM"].get("raw_relative_l2_over_public"),
        "nqc_corrected_mode_raw_relative_l2_0_to_1000": lambda row: row["nqc_corrected_inspiral_mode_vs_public_unshifted"]["bands"]["0_to_1000M"].get("raw_relative_l2_over_public"),
        "nqc_corrected_mode_raw_relative_l2_all_pre_match": lambda row: row["nqc_corrected_inspiral_mode_vs_public_unshifted"]["bands"]["all_pre_match"].get("raw_relative_l2_over_public"),
        "nqc_corrected_mode_raw_relative_l2_match_minus_100_to_match": lambda row: row["nqc_corrected_inspiral_mode_vs_public_unshifted"]["bands"]["match_minus_100_to_matchM"].get("raw_relative_l2_over_public"),
    }
    summary = {"scalar_ranges": {}, "phase_ranges_by_degree": {}, "pairwise": {}}
    for name, getter in scalar_getters.items():
        values = [getter(results[label]) for label in LABELS]
        values = [float(value) for value in values if value is not None]
        summary["scalar_ranges"][name] = {
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "max_minus_min": max(values) - min(values) if values else None,
        }
    for degree in DERIVATIVE_DEGREES:
        times = [
            results[label]["returned_phase_omega_peak"]["records"][str(degree)].get(
                "refined_time_M_initial_origin"
            )
            for label in LABELS
        ]
        times = [float(value) for value in times if value is not None]
        deltas = [
            results[label]["returned_phase_omega_peak"]["records"][str(degree)].get(
                "difference_refined_to_minus_epoch_M"
            )
            for label in LABELS
        ]
        deltas = [float(value) for value in deltas if value is not None]
        summary["phase_ranges_by_degree"][str(degree)] = {
            "refined_time_M_initial_origin": {
                "minimum": min(times) if times else None,
                "maximum": max(times) if times else None,
                "max_minus_min": max(times) - min(times) if times else None,
            },
            "refined_minus_minus_epoch_M": {
                "minimum": min(deltas) if deltas else None,
                "maximum": max(deltas) if deltas else None,
                "max_minus_min": max(deltas) - min(deltas) if deltas else None,
            },
        }
    for left, right in itertools.combinations(LABELS, 2):
        key = f"{left}_minus_{right}"
        summary["pairwise"][key] = {
            "epoch_M": float(results[left]["epoch_M"] - results[right]["epoch_M"]),
            "clean_event_time_M": float(
                results[left]["clean_hamiltonian_omega_peak"]["time_M_initial_origin"]
                - results[right]["clean_hamiltonian_omega_peak"]["time_M_initial_origin"]
            ),
            "phase_degree6_refined_time_M": float(
                results[left]["returned_phase_omega_peak"]["records"]["6"]["refined_time_M_initial_origin"]
                - results[right]["returned_phase_omega_peak"]["records"]["6"]["refined_time_M_initial_origin"]
            ),
        }
    return summary


def main() -> None:
    physics = _setup_clean_physics()
    results = {label: _evaluate(_load_dataset(label), physics) for label in LABELS}
    report = {
        "provenance": (
            "Bounded clean-room isolation using the four existing public "
            "lalsimulation SimIMRSpinAlignedEOBModes datasets for 30+20 Msun, "
            "nonspinning, fstart=20 Hz, distance=100 Mpc. Public postpeak row 2 "
            "is treated as orbital phase and differentiated independently with "
            "fixed local polynomial orders 4/6/8. Clean Hamiltonian/flux/event, "
            "published base mode, and known-target NQC attachment are evaluated "
            "afresh in float64. No source waveform, public coefficient, time "
            "offset, phase offset, or residual-selected stencil is used."
        ),
        "primary_reference": (
            "Bohé et al., Improved effective-one-body model for coalescing "
            "nonprecessing binary black holes, arXiv:1611.03703 (equation context)."
        ),
        "public_parameters": {
            "approximant_api": "lalsimulation.SimIMRSpinAlignedEOBModes",
            "approximant_version": 4,
            "m1_msun": 30.0,
            "m2_msun": 20.0,
            "chi1": 0.0,
            "chi2": 0.0,
            "fstart_hz": 20.0,
            "distance_mpc": DISTANCE_MPC,
            "mode": [2, 2],
            "spacings_M": {"dt_0p25M": 0.25, "dt_0p5M": 0.5, "dt_1M": 1.0, "dt_1_4096s": 1.0 / 4096.0 / MSEC},
        },
        "clean_parameters": {
            "nu": 0.24,
            "delta": 0.2,
            "chi": 0.0,
            "spin_truncation": "generic-leading-ell8",
            "spin_v6": "none",
            "include_pan_v8": True,
            "include_positive_chi_540": True,
            "final_spin_fit_order": [3, 4],
            "mode_scale_100Mpc": MODE_SCALE,
            "phase_derivative_orders": list(DERIVATIVE_DEGREES),
        },
        "coordinate_contract": {
            "returned_state_phase": "postpeak_state[2], differentiated against postpeak_state[0] in initial-origin M coordinates",
            "public_epoch_comparison": "compare phase peak and clean event in initial-origin M coordinates with -epoch_M",
            "attachment_output_time": "arange(mode22_length) * dt_s / Msec",
            "public_mode_time": "epoch_s + arange(mode22_length) * dt_s",
            "base_mode_comparison": "same-index output grid, absolute complex scale, zero time/phase shifts",
        },
        "mass_assumption": {"total_mass_msun": TOTAL_MASS, "Msec": MSEC, "MTSUN_s": MSUN_SEC},
        "libraries": _library_versions(),
        "python_executable": sys.executable,
        "host": platform.node(),
        "dependency_sha256": {path: sha(REPO_ROOT / path) for path in DEPENDENCIES},
        "results": results,
        "sampling_dependence": _range_summary(results),
        "interpretation": {
            "phase_peak_vs_epoch": (
                "The returned phase-derivative peak follows the clean initial-origin "
                "omega event within a few 1e-3 M on every grid. It therefore does "
                "not reproduce the 0.01707037956 M common-epoch displacement: the "
                "epoch discrepancy is not encoded as a different returned phase peak."
            ),
            "base_inspiral_mode_residual": (
                "The public complex mode contains its own NQC/attachment behavior over "
                "the returned support, so base-v-public residuals are NQC-confounded "
                "and do not by themselves identify a carrier discrepancy. The report "
                "also evaluates a fixed clean attachment-derived NQC factor before the "
                "match; this combined comparison is informative about the requested "
                "clean model contract but is not a public-NQC coefficient fit."
            ),
            "runtime_recommendation": (
                "No additional runtime change is justified by this isolation. Keep "
                "the relative attachment grid arange(n)*dt/Msec and retain the public "
                "epoch as separate metadata; do not substitute -epoch for the clean "
                "event or fit a time/phase shift. A future change should target the "
                "clean carrier/factorized-mode contract and be validated independently."
            ),
        },
        "limitations": [
            "The public phase row schema is used as the documented black-box observation from the paired wrapper; no LAL or PyCBC source is inspected.",
            "The public mode includes its own NQC/attachment behavior. The base-mode comparison therefore cannot isolate the carrier; the separate NQC-corrected comparison applies only clean attachment-derived coefficients and never infers coefficients from public data.",
            "The postmatch residual is available only where the clean attached output and public mode have the same relative output indices; it is raw complex subtraction with zero shifts.",
            "Degree 4/6/8 phase stencils are predetermined numerical refinement diagnostics and are not selected by parity, residual, or agreement with the epoch.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Localize clean SEOB attachment residuals on identical public trajectories.

This is a bounded component diagnostic.  It consumes the permitted public
post-peak state/mode fixtures and the clean TorchWave equations, without
evolving a trajectory or importing a reference implementation.  The public
mode is used as an observable on its supplied output grid.  For source-knot
comparisons it is represented by independently interpolated magnitude and
unwrapped positive phase; no complex public samples are interpolated.

The report separates the clean NQC matching system (targets, sampled basis
derivatives, solve and residual), the corrected source series, the discrete
floor boundary, and the attached ringdown.  Public NQC coefficients from the
prior black-box observable report are included as diagnostic observations
only; they are never used in the clean calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time as wallclock
from pathlib import Path

import numpy as np
import torch

from torchwave.approximants.eob.nqc import nqc_22_basis, seobnrv4_nqc_targets
from torchwave.approximants.eob.numerical_attachment import (
    attach_seobnrv4_22_numerical_compatibility,
    natural_cubic_spline,
    seobnrv4_ringdown_support_samples,
)
from torchwave.approximants.eob.published_waveform import (
    published_mode22_and_frequency,
)
from torchwave.approximants.eob.remnant import (
    eob_final_mass_fraction,
    hofmann_final_spin,
)
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.kerr_qnm_table_convention import (
    signed_qnm_220_table_convention,
)
from torchwave.constants import C_SI


DTYPE = torch.float64
REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = Path("/tmp")
REPORT_DEFAULT = (
    REPO_ROOT
    / "docs/reviews/2026-09-13-parity-review/verification"
    / "eob-attachment-localization-clean/report.json"
)
LABELS = ("dt_0p5M", "dt_1_4096s")
TOTAL_MASS_MSUN = 50.0
MTSUN_S = 4.925490947641267e-6
MASS_SECONDS = TOTAL_MASS_MSUN * MTSUN_S
DISTANCE_MPC = 100.0
MPC_SI = 3.0856775814913673e22
MODE_SCALE = MASS_SECONDS * C_SI / (DISTANCE_MPC * MPC_SI)

SOURCE_FILES = (
    "scripts/diagnose_eob_attachment_localization_clean.py",
    "src/torchwave/approximants/eob/numerical_attachment.py",
    "src/torchwave/approximants/eob/published_waveform.py",
    "src/torchwave/approximants/eob/nqc.py",
    "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
    "src/torchwave/approximants/eob/seobnrv4_flux.py",
    "src/torchwave/approximants/eob/remnant.py",
    "src/torchwave/approximants/kerr_qnm_table_convention.py",
)
INPUT_REPORTS = (
    "docs/reviews/2026-09-13-parity-review/verification/eob-numerical-root-review/public-nqc-substitution.json",
    "docs/reviews/2026-09-13-parity-review/verification/eob-numerical-root-review/public-derivative-attachment-local.json",
)


def _float(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    return float(value)


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).tolist()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(paths) -> dict[str, str]:
    return {str(path): _sha256(REPO_ROOT / path) for path in paths}


def _load_public(label: str) -> dict:
    path = PUBLIC_DATA / f"torchwave-public-eob-{label}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing permitted public fixture: {path}")
    with np.load(path, allow_pickle=False) as data:
        post = np.asarray(data["postpeak_state"], dtype=np.float64).copy()
        public_mode = np.asarray(data["mode22"], dtype=np.complex128).copy()
        dt_s = float(np.asarray(data["dt_s"]).item())
        epoch_s = float(np.asarray(data["epoch_s"]).item())
    if post.ndim != 2 or post.shape[0] != 5 or post.shape[1] < 4:
        raise ValueError(f"{label}: postpeak_state must have shape (5,N>=4)")
    if public_mode.ndim != 1 or public_mode.size < 2:
        raise ValueError(f"{label}: mode22 must be a nonempty one-dimensional array")
    if not np.isfinite(post).all() or not np.isfinite(public_mode).all():
        raise ValueError(f"{label}: fixture contains non-finite values")
    if not np.all(np.diff(post[0]) > 0):
        raise ValueError(f"{label}: state time must be strictly increasing")
    if not np.isfinite(dt_s) or dt_s <= 0 or not np.isfinite(epoch_s):
        raise ValueError(f"{label}: invalid epoch or output spacing")
    return {
        "path": path,
        "sha256": _sha256(path),
        "postpeak_state": post,
        "public_mode": public_mode,
        "dt_s": dt_s,
        "epoch_s": epoch_s,
    }


def _positive_phase(mode: np.ndarray) -> np.ndarray:
    """Return a lifted ``phi`` for ``h=A exp(-i phi)`` on the public grid."""
    phase = np.empty(mode.size, dtype=np.float64)
    phase[0] = -np.angle(mode[0])
    phase[1:] = phase[0] - np.cumsum(np.angle(mode[1:] * np.conj(mode[:-1])))
    return phase


def _positive_phase_tensor(mode: torch.Tensor) -> torch.Tensor:
    """Return the same lifted positive phase for a clean complex series."""
    phase = torch.empty_like(mode.real)
    phase[0] = -torch.angle(mode[0])
    if mode.shape[-1] > 1:
        phase[1:] = phase[0] - torch.angle(
            mode[1:] * mode[:-1].conj()
        ).cumsum(dim=-1)
    return phase


def _clean_corrected_on_grid(
    source_time: torch.Tensor,
    source_mode: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a clean sampled mode on an external grid.

    This check deliberately constructs independent natural cubic splines for
    the sampled amplitude and lifted positive phase.  It therefore resamples
    only the clean source, leaving the public output samples untouched for a
    direct same-time comparison.
    """
    amplitude_spline = natural_cubic_spline(source_time, source_mode.abs())
    phase_spline = natural_cubic_spline(
        source_time, _positive_phase_tensor(source_mode)
    )
    amplitude = amplitude_spline.evaluate(query)
    phase = phase_spline.evaluate(query)
    return amplitude * torch.exp(-1j * phase)


def _public_complex_from_amp_phase(
    query: np.ndarray,
    output_time: np.ndarray,
    public_amplitude: np.ndarray,
    public_phase: np.ndarray,
) -> np.ndarray:
    """Interpolate public magnitude and lifted phase, then re-project."""
    amplitude = np.interp(query, output_time, public_amplitude)
    phase = np.interp(query, output_time, public_phase)
    return amplitude * np.exp(-1j * phase)


def _metrics(clean: np.ndarray, reference: np.ndarray) -> dict:
    if clean.size == 0:
        return {"available": False, "sample_count": 0}
    clean = np.asarray(clean, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    ref_norm = float(np.linalg.norm(reference))
    amp_norm = float(np.linalg.norm(np.abs(reference)))
    phase_error = np.angle(clean * np.conj(reference))
    return {
        "available": True,
        "sample_count": int(clean.size),
        "complex_relative_l2": float(np.linalg.norm(clean - reference) / ref_norm)
        if ref_norm
        else None,
        "amplitude_relative_l2": float(
            np.linalg.norm(np.abs(clean) - np.abs(reference)) / amp_norm
        )
        if amp_norm
        else None,
        "phase_rms_rad": float(np.sqrt(np.mean(phase_error**2))),
        "phase_max_abs_rad": float(np.max(np.abs(phase_error))),
        "phase_mean_rad": float(np.mean(phase_error)),
    }


def _scalar_boundary_metrics(
    clean_amplitude: float,
    clean_phase: float,
    public_at_boundary: np.ndarray,
) -> dict:
    clean = clean_amplitude * np.exp(-1j * clean_phase)
    return {
        "clean_amplitude": clean_amplitude,
        "clean_phase_lifted_rad": clean_phase,
        "public_interpolated_amplitude": float(abs(public_at_boundary)),
        "public_interpolated_phase_lifted_rad": float(-np.angle(public_at_boundary)),
        "wrapped_phase_error_rad": float(
            np.angle(clean * np.conj(public_at_boundary))
        ),
        "complex_relative_error": float(
            abs(clean - public_at_boundary) / abs(public_at_boundary)
        ),
        "relative_amplitude_error": float(
            clean_amplitude / abs(public_at_boundary) - 1.0
        ),
    }


def _clean_matching_system(
    time: torch.Tensor,
    state: torch.Tensor,
    orbital_frequency: torch.Tensor,
    base_mode: torch.Tensor,
    match_time: torch.Tensor,
    nu: torch.Tensor,
    chi: torch.Tensor,
) -> dict:
    """Expose the sampled full-series derivative matrices used by the solve."""
    amplitude = base_mode.abs()
    q, p = nqc_22_basis(state[:, 0], state[:, 2], orbital_frequency)
    products = amplitude.unsqueeze(-1) * q
    combined = torch.cat((amplitude.unsqueeze(-1), products, p), dim=-1)
    combined_spline = natural_cubic_spline(time, combined)
    query = match_time.reshape(1)
    value = combined_spline.evaluate(query)[0]
    first = combined_spline.evaluate(query, derivative=1)[0]
    second = combined_spline.evaluate(query, derivative=2)[0]
    phase = torch.empty_like(time)
    phase[0] = -torch.angle(base_mode[0])
    phase[1:] = phase[0] - torch.angle(
        base_mode[1:] * base_mode[:-1].conj()
    ).cumsum(dim=-1)
    phase_spline = natural_cubic_spline(time, phase)
    phase_first = phase_spline.evaluate(query, derivative=1)[0]
    phase_second = phase_spline.evaluate(query, derivative=2)[0]
    targets = seobnrv4_nqc_targets(nu, chi)
    target_vector = torch.stack(tuple(targets))
    amplitude_matrix = torch.stack((value[1:4], first[1:4], second[1:4]))
    amplitude_rhs = torch.stack((targets.amplitude, targets.amplitude_derivative,
                                 targets.amplitude_second_derivative)) - torch.stack(
                                     (value[0], first[0], second[0])
                                 )
    phase_matrix = torch.stack((first[4:6], second[4:6]))
    phase_rhs = torch.stack((phase_first - targets.frequency,
                             phase_second - targets.frequency_derivative))
    return {
        "target_vector": _array(target_vector),
        "target_names": [
            "amplitude", "amplitude_derivative", "amplitude_second_derivative",
            "frequency", "frequency_derivative",
        ],
        "base_value_at_match": _array(value),
        "base_first_derivative_at_match": _array(first),
        "base_second_derivative_at_match": _array(second),
        "waveform_phase_first_derivative_at_match": _float(phase_first),
        "waveform_phase_second_derivative_at_match": _float(phase_second),
        "amplitude_matrix": _array(amplitude_matrix),
        "amplitude_rhs": _array(amplitude_rhs),
        "phase_matrix": _array(phase_matrix),
        "phase_rhs": _array(phase_rhs),
    }


def _prior_public_coefficients(label: str) -> dict | None:
    path = REPO_ROOT / INPUT_REPORTS[0]
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for case in data.get("cases", []):
        if case.get("label") != label:
            continue
        variants = case.get("variants", {})
        amplitude = variants.get("public_amplitude", {}).get("coefficients", {}).get(
            "amplitude"
        )
        phase = variants.get("public_phase", {}).get("coefficients", {}).get("phase")
        if amplitude is None or phase is None:
            return None
        return {
            "amplitude": amplitude,
            "phase": phase,
            "source": str(INPUT_REPORTS[0]),
            "use": "observable diagnostic only; not substituted into clean attachment",
        }
    return None


def _fixture(label: str, ham, flux, nu, chi, sigma) -> dict:
    fixture = _load_public(label)
    post = fixture["postpeak_state"]
    public_mode = fixture["public_mode"]
    source_time_np = post[0]
    source_time = torch.from_numpy(source_time_np)
    state = torch.from_numpy(post[1:].T)
    orbital_frequency = flux.from_hamiltonian(state, ham)["omega"]
    base_mode, base_frequency = published_mode22_and_frequency(
        state,
        ham,
        flux,
        spin_v6="none",
        include_pan_v8=True,
        include_positive_chi_540=True,
    )
    output_time = torch.arange(
        public_mode.size, dtype=DTYPE
    ) * (fixture["dt_s"] / MASS_SECONDS)
    attached = attach_seobnrv4_22_numerical_compatibility(
        source_time,
        state,
        orbital_frequency,
        base_mode,
        base_frequency,
        nu,
        chi,
        sigma,
        output_time,
    )
    output_time_np = output_time.detach().numpy()
    public_amplitude = np.abs(public_mode)
    public_phase = _positive_phase(public_mode)
    output_dt = float(output_time_np[1] - output_time_np[0])
    clean_base = base_mode.detach().numpy() * MODE_SCALE
    clean_corrected = attached.corrected_inspiral.detach().numpy() * MODE_SCALE
    clean_attached = attached.mode.detach().numpy() * MODE_SCALE
    boundary_time = _float(attached.boundary_time)
    match_time = _float(attached.match_time)
    source_mask = (source_time_np >= output_time_np[0]) & (source_time_np < boundary_time)
    # Compare at public output times without interpolating the public mode.
    # The immediate interval is one public output spacing before the clean
    # floor boundary; all earlier overlap is reported separately.
    common_output_mask = (
        (output_time_np >= source_time_np[0])
        & (output_time_np < boundary_time)
    )
    immediate_boundary_start = boundary_time - output_dt
    immediate_output_mask = common_output_mask & (
        output_time_np >= immediate_boundary_start
    )
    preboundary_output_mask = common_output_mask & ~immediate_output_mask
    common_query = output_time[common_output_mask]
    clean_corrected_common = (
        _clean_corrected_on_grid(
            source_time, attached.corrected_inspiral, common_query
        ).detach().numpy()
        * MODE_SCALE
    )
    clean_base_common = (
        _clean_corrected_on_grid(source_time, base_mode, common_query)
        .detach().numpy()
        * MODE_SCALE
    )
    public_common = public_mode[common_output_mask]
    common_positions = np.flatnonzero(common_output_mask)
    immediate_positions = np.flatnonzero(immediate_output_mask)
    preboundary_positions = np.flatnonzero(preboundary_output_mask)
    # ``common_query`` is ordered exactly like public_common, so these masks
    # select the corresponding clean spline values without touching public
    # amplitudes, phases, or complex samples.
    common_immediate_mask = immediate_output_mask[common_positions]
    common_preboundary_mask = ~common_immediate_mask
    post_mask = (output_time_np >= boundary_time) & (np.abs(public_mode) > 0)

    system = _clean_matching_system(
        source_time, state, orbital_frequency, base_mode,
        attached.match_time, nu, chi,
    )
    calibration = attached.calibration
    coeff = calibration.coefficients
    coeff_clean = {
        "amplitude": _array(coeff.amplitude),
        "phase": _array(coeff.phase),
    }
    public_coeff = _prior_public_coefficients(label)
    if public_coeff is not None:
        coeff_delta = {
            "amplitude_clean_minus_public": [
                float(a - b) for a, b in zip(coeff_clean["amplitude"], public_coeff["amplitude"])
            ],
            "phase_clean_minus_public": [
                float(a - b) for a, b in zip(coeff_clean["phase"], public_coeff["phase"])
            ],
        }
    else:
        coeff_delta = None

    boundary_index = int(_float(attached.boundary_index))
    boundary_mode = (
        _float(attached.boundary_amplitude) * MODE_SCALE
        * np.exp(-1j * _float(attached.boundary_phase))
    )
    boundary_public = _public_complex_from_amp_phase(
        np.asarray([boundary_time]), output_time_np, public_amplitude, public_phase
    )[0]
    next_index = int(np.searchsorted(output_time_np, boundary_time, side="right"))
    if 1 <= next_index < public_mode.size:
        public_backward_frequency = float(
            -np.angle(public_mode[next_index] * np.conj(public_mode[next_index - 1]))
            / (output_time_np[next_index] - output_time_np[next_index - 1])
        )
        next_output = {
            "index": next_index,
            "time_M": float(output_time_np[next_index]),
            "amplitude": float(public_amplitude[next_index]),
            "backward_frequency": public_backward_frequency,
            "boundary_to_next_output_M": float(output_time_np[next_index] - boundary_time),
            "clean_boundary_to_public_next_output": _metrics(
                np.asarray([boundary_mode]), np.asarray([public_mode[next_index]])
            ),
        }
    else:
        next_output = {"available": False}

    windows = {}
    for name, lo, hi in (
        ("first_1M", 0.0, 1.0),
        ("first_5M", 0.0, 5.0),
        ("first_10M", 0.0, 10.0),
        ("first_50M", 0.0, 50.0),
        ("first_100M", 0.0, 100.0),
        ("late_50_to_100M", 50.0, 100.0),
        ("late_100M_to_end", 100.0, float("inf")),
    ):
        mask = (
            post_mask
            & (output_time_np - boundary_time >= lo)
            & (output_time_np - boundary_time < hi)
        )
        windows[name] = _metrics(clean_attached[mask], public_mode[mask])

    matrix_report = system
    matrix_report.update(
        {
            "clean_coefficients": coeff_clean,
            "clean_sampling_condition": _float(calibration.sampling_condition),
            "clean_amplitude_condition": _float(calibration.amplitude_condition),
            "clean_phase_condition": _float(calibration.phase_condition),
            "clean_amplitude_solve_residual": _array(calibration.amplitude_residual),
            "clean_phase_solve_residual": _array(calibration.phase_residual),
            "public_coefficients": public_coeff,
            "clean_minus_public_coefficient_delta": coeff_delta,
        }
    )

    clean_phase_at_match = _float(attached.match_phase)
    match_mode = (
        attached.nqc.evaluate(attached.match_time).detach().cpu().item() * MODE_SCALE
    )
    match_public = _public_complex_from_amp_phase(
        np.asarray([match_time]), output_time_np, public_amplitude, public_phase
    )[0]
    damping = -_float(sigma.imag)
    support_count = int(
        seobnrv4_ringdown_support_samples(
            torch.tensor(damping, dtype=DTYPE), attached.internal_step
        )
    )
    return {
        "fixture_sha256": fixture["sha256"],
        "source_state_shape": list(post.shape),
        "public_mode_length": int(public_mode.size),
        "epoch_s": fixture["epoch_s"],
        "output_dt_M": float(output_time_np[1] - output_time_np[0]),
        "source_time_M": [float(source_time_np[0]), float(source_time_np[-1])],
        "output_time_M": [float(output_time_np[0]), float(output_time_np[-1])],
        "peak": {
            "time_M": _float(attached.peak.time),
            "frequency": _float(attached.peak.frequency),
            "curvature": _float(attached.peak.curvature),
            "sample_index": int(_float(attached.peak.sample_index)),
        },
        "match": {
            "time_M": match_time,
            "phase_lifted_rad": clean_phase_at_match,
            "clean_match_mode_vs_public_amp_phase": _metrics(
                np.asarray([match_mode]), np.asarray([match_public])
            ),
        },
        "boundary": {
            "time_M": boundary_time,
            "input_sample_index": boundary_index,
            "input_step_M": float(source_time_np[boundary_index] - source_time_np[boundary_index - 1]),
            "phase_lifted_rad": _float(attached.boundary_phase),
            "frequency_clean_backward": _float(attached.boundary_frequency),
            "amplitude_clean_scaled": _float(attached.boundary_amplitude) * MODE_SCALE,
            "clean_vs_public_interpolated": _scalar_boundary_metrics(
                _float(attached.boundary_amplitude) * MODE_SCALE,
                _float(attached.boundary_phase),
                boundary_public,
            ),
            "public_next_output_observable": next_output,
        },
        "ringdown": {
            "damping": damping,
            "internal_step_M": _float(attached.internal_step),
            "support_count_high_samples": support_count,
            "windows": windows,
        },
        "source_same_trajectory": {
            "comparison": "two checks are reported: public amplitude/phase interpolation at clean state knots, and clean amplitude/phase spline interpolation at public output times",
            "sample_count": int(source_mask.sum()),
            "base_mode_before_nqc": _metrics(
                clean_base[source_mask],
                _public_complex_from_amp_phase(
                    source_time_np[source_mask], output_time_np,
                    public_amplitude, public_phase,
                ),
            ),
            "corrected_mode_after_nqc": _metrics(
                clean_corrected[source_mask],
                _public_complex_from_amp_phase(
                    source_time_np[source_mask], output_time_np,
                    public_amplitude, public_phase,
                ),
            ),
            "direct_public_grid": {
                "comparison": (
                    "clean corrected and base source amplitudes and lifted positive phases are independently natural-cubic-splined to public output times; public complex samples are compared directly"
                ),
                "output_time_M": [
                    float(output_time_np[common_output_mask][0]),
                    float(output_time_np[common_output_mask][-1]),
                ] if common_output_mask.any() else None,
                "output_dt_M": output_dt,
                "sample_count_all_preboundary": int(common_output_mask.sum()),
                "sample_count_preboundary_excluding_immediate": int(
                    preboundary_output_mask.sum()
                ),
                "sample_count_immediate_boundary_interval": int(
                    immediate_output_mask.sum()
                ),
                "immediate_boundary_interval_M": [
                    float(immediate_boundary_start), float(boundary_time)
                ],
                "base_mode_before_nqc": _metrics(
                    clean_base_common, public_common
                ),
                "corrected_mode_after_nqc": {
                    "all_preboundary": _metrics(
                        clean_corrected_common, public_common
                    ),
                    "preboundary_excluding_immediate": _metrics(
                        clean_corrected_common[common_preboundary_mask],
                        public_common[common_preboundary_mask],
                    ),
                    "immediate_boundary_interval": _metrics(
                        clean_corrected_common[common_immediate_mask],
                        public_common[common_immediate_mask],
                    ),
                },
            },
        },
        "nqc_matching_system": matrix_report,
        "interpretation": {
            "first_material_residual_stage": (
                "unresolved corrected source / boundary discrepancy; public-grid interpolation remains confounded"
            ),
            "evidence": (
                "NQC removes the order-unity uncorrected carrier mismatch. The direct public-grid check resamples only the clean corrected source, but its remaining preboundary discrepancy and the one-spacing boundary interval do not by themselves distinguish a source/NQC convention from a floor-boundary convention. The amplitude/phase-interpolated boundary value is therefore diagnostic evidence rather than a first-stage attribution."
            ),
            "public_binary_coefficients_are_diagnostic": True,
            "required_next_observable": [
                "Reference corrected h22 amplitude and phase at the clean floor boundary knot and its preceding knot, or the corresponding continuous attachment values; the public output grid alone cannot resolve this.",
                "Reference NQC matching-system target and sampled basis derivative rows at the same clean match time if the source coefficient residual is to be separated from the boundary convention.",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args()
    started = wallclock.perf_counter()
    source_paths = SOURCE_FILES + INPUT_REPORTS
    before = _hashes(source_paths)

    zero = torch.tensor(0.0, dtype=DTYPE)
    nu = torch.tensor(0.24, dtype=DTYPE)
    ham = SEOBNRv4Hamiltonian(nu, zero, zero, delta=torch.tensor(0.2, dtype=DTYPE))
    flux = SEOBNRv4PublishedFlux(spin_truncation="generic-leading-ell8")
    mass1 = torch.tensor(0.6, dtype=DTYPE)
    mass2 = torch.tensor(0.4, dtype=DTYPE)
    mass_fraction = eob_final_mass_fraction(mass1, mass2, zero, zero)
    final_spin = hofmann_final_spin(mass1, mass2, zero, zero, fit_order=(3, 4))
    qnm_frequency, qnm_damping = signed_qnm_220_table_convention(final_spin)
    sigma = torch.complex(qnm_frequency, -qnm_damping) / mass_fraction

    results = {
        label: _fixture(label, ham, flux, nu, zero, sigma) for label in LABELS
    }
    after = _hashes(source_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "bounded_same_trajectory_attachment_localization",
        "host": platform.node(),
        "cwd": str(Path.cwd()),
        "command": " ".join(sys.argv),
        "pid": os.getpid(),
        "elapsed_seconds": wallclock.perf_counter() - started,
        "source_hashes_before": before,
        "source_hashes_after": after,
        "source_hashes_stable": before == after,
        "provenance": {
            "implementation": "MIT-only clean TorchWave Hamiltonian, flux, published mode, NQC, QNM and numerical attachment helpers",
            "trajectory": "identical public postpeak state arrays are used for all stages; no trajectory evolution",
            "public_comparison": "public mode samples are compared directly on their output grid; source-knot values use independently interpolated public amplitude and lifted positive phase, while the added preboundary output-grid check splines only the clean source amplitude and lifted positive phase",
            "reference_implementation_imported": False,
            "fitted_coefficients": False,
            "forbidden_source_or_cache_access": False,
        },
        "parameters": {
            "total_mass_msun": TOTAL_MASS_MSUN,
            "distance_mpc": DISTANCE_MPC,
            "m1_fraction": 0.6,
            "m2_fraction": 0.4,
            "nu": _float(nu),
            "chi": 0.0,
            "spin_v6": "none",
            "include_pan_v8": True,
            "include_positive_chi_540": True,
            "spin_truncation": "generic-leading-ell8",
        },
        "prior_evidence": {
            "public_nqc_substitution_report": str(INPUT_REPORTS[0]),
            "public_derivative_attachment_report": str(INPUT_REPORTS[1]),
            "known_public_phase_only_dt_0p5M_raw_relative_l2": 3.297968270513276e-6,
            "known_public_phase_only_dt_1_4096s_raw_relative_l2": 3.322373789622722e-6,
            "interpretation": "public coefficient substitution is retained as attribution evidence only; no public coefficient enters this clean rerun",
        },
        "results": results,
        "limitations": [
            "The 1/4096 s output spacing is 0.991335M for this 50 Msun fixture and is coarser than the 0.5M output spacing; neither public grid contains the clean continuous floor boundary time in general.",
            "The direct preboundary output-grid check removes public resampling but still uses a clean amplitude/phase spline, so it cannot by itself identify a unique source or boundary convention.",
            "A smaller post-boundary residual does not identify a unique production convention; direct boundary phase/frequency observables are required for further attribution.",
            "This is not a complete waveform, calibration, or acceptance gate.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

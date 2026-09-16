#!/usr/bin/env python3
"""Run the paired clean TEOB postmerger composition diagnostic.

This is a bounded, diagnostic-only driver.  It evolves the clean
nonspinning TEOB vacuum profile from the exact floating-point radius recorded
by the public binary fixture, evaluates the clean TEOB-specific
``TEOBRadiationClean.mode`` carrier (whose TEOB tail coordinate factor is
applied internally once), solves the explicit 2017 NQC seam, and appends the
clean UIB2016 ringdown ansatz.  The current horizon absorption setting is the
default baseline; a separately selected leading-order horizon probe is also
available for equation-level attribution.

The report keeps raw trajectory samples, NQC corrected charts, assembly
events, and two public-grid comparisons.  The latter use an independent
natural-cubic evaluator for signed amplitude and lifted phase, so a result
can be inspected without relying on the assembler's output interpolation.
No alignment, fitting, support truncation, or production dispatch is used.
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
import time as wall_time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torchwave.approximants.eob.teob_adaptive_continuation_clean import (
    DEFAULT_ATOL,
    DEFAULT_CONTINUATION_DURATION,
    DEFAULT_FIXED_STEP_CAP,
    DEFAULT_MAX_ACCEPTED_STEPS,
    DEFAULT_MAX_DURATION,
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_MAX_REJECTED_STEPS,
    DEFAULT_RTOL,
    continue_teob_adaptive,
)
from torchwave.approximants.eob.teob_merger_nqc_clean import (
    PUBLISHED_2017_NQC_PROVENANCE,
    assemble_teob_merger_nqc,
    phase_from_mode,
    published_2017_nqc_targets,
)
from torchwave.approximants.eob.teob_postadiabatic_clean import alternating8pa
from torchwave.approximants.eob.teob_radiation_clean import (
    TEOBRadiationClean,
    non_keplerian_radius,
)
from torchwave.approximants.eob.teob_ringdown_assembly_clean import (
    RingdownAssemblyResult,
    assemble_teob_ringdown_22,
)
from torchwave.approximants.eob.teob_trajectory import (
    conservative_trajectory_quantities,
)
from torchwave.approximants.eob.teob_uib2016_remnant import (
    uib2016_nonspinning_remnant,
)
from torchwave.approximants.eob.teob_waveform_state_clean import (
    teob_waveform_state_mode,
)
from torchwave.approximants.teobresums import (
    TEOBResumSHamiltonian,
    TEOBResumSRadiationReaction,
)


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "verification/teob-paired-driver-clean"
DEFAULT_PUBLIC = Path("/tmp/torchwave-public-teob-q1p5-r20-dt0p5M.npz")
DEFAULT_PUBLIC_META = DEFAULT_PUBLIC.with_suffix(".json")
DEFAULT_OUTPUT = VERIFY / "diagnostic.json"
DEFAULT_CACHE = VERIFY / "paired-physical-output.npz"
DEFAULT_LOG = VERIFY / "paired-driver.log"
DEFAULT_LO_OUTPUT = VERIFY / "diagnostic-lo-horizon.json"
DEFAULT_LO_CACHE = VERIFY / "paired-physical-output-lo-horizon.npz"
DEFAULT_LO_LOG = VERIFY / "paired-driver-lo-horizon.log"

EXPECTED_PUBLIC_SHA256 = (
    "09eaa8b00288c94947b2bb4b40ada7780dc51ac98491a48cbb8f60ed50ab686c"
)
PUBLIC_RADIUS = 19.999999999999996
MASS1_MSUN = 30.0
MASS2_MSUN = 20.0
NU_VALUE = 0.24
DELTA_VALUE = 0.2
R_MIN = 14.0
DR = 0.1
PA_ORDER = 8
OUTPUT_STEP = 0.5
DISTANCE_MPC = 100.0
M_MODE = 2
LO_HORIZON_SOURCE = ROOT / "scripts/diagnose_teob_pa_residual_clean.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _vector(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1)]
    return [float(item) for item in np.asarray(value).reshape(-1)]


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return _numpy(value).tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _source_paths() -> dict[str, Path]:
    """Return only the clean source files used by this diagnostic."""

    return {
        "script": Path(__file__).resolve(),
        "postadiabatic": ROOT / "src/torchwave/approximants/eob/teob_postadiabatic_clean.py",
        "adaptive_continuation": ROOT / "src/torchwave/approximants/eob/teob_adaptive_continuation_clean.py",
        "adaptive_driver": ROOT / "src/torchwave/approximants/eob/adaptive_driver_clean.py",
        "vacuum_driver": ROOT / "src/torchwave/approximants/eob/teob_vacuum_driver_clean.py",
        "trajectory": ROOT / "src/torchwave/approximants/eob/teob_trajectory.py",
        "radiation": ROOT / "src/torchwave/approximants/eob/teob_radiation_clean.py",
        "waveform_state_adapter": ROOT / "src/torchwave/approximants/eob/teob_waveform_state_clean.py",
        "hamiltonian": ROOT / "src/torchwave/approximants/teobresums.py",
        "nqc": ROOT / "src/torchwave/approximants/eob/teob_merger_nqc_clean.py",
        "nqc_basis": ROOT / "src/torchwave/approximants/eob/teob_nqc_clean.py",
        "numerical_attachment": ROOT / "src/torchwave/approximants/eob/numerical_attachment.py",
        "ringdown_assembly": ROOT / "src/torchwave/approximants/eob/teob_ringdown_assembly_clean.py",
        "postmerger": ROOT / "src/torchwave/approximants/eob/teob_postmerger.py",
        "uib_remnant": ROOT / "src/torchwave/approximants/eob/teob_uib2016_remnant.py",
        "contract": ROOT / "docs/reviews/2026-09-13-parity-review/teob-merger-assembly-contract.md",
    }


def _hamiltonian() -> TEOBResumSHamiltonian:
    dtype = torch.float64
    zero = torch.tensor(0.0, dtype=dtype)
    return TEOBResumSHamiltonian(
        torch.tensor(NU_VALUE, dtype=dtype),
        zero,
        zero,
        zero,
        zero,
        delta=torch.tensor(DELTA_VALUE, dtype=dtype),
    )


class _LeadingOrderHorizonReaction:
    """Diagnostic-only zero-spin horizon replacement from Eqs. (98)-(99).

    The clean infinity-mode flux is retained and the current clean horizon
    term is replaced by the independently published leading-order factor
    ``(X1**4 + X2**4) * vphi**8``.  This object is selected only by the
    diagnostic command line and never enters production routing.
    """

    def __init__(self) -> None:
        self.include_horizon_absorption = True
        self._radiation = TEOBRadiationClean(8, residual_phase="zero-unpublished")

    def radiation_reaction_forces(
        self,
        omega: torch.Tensor,
        pr_star: torch.Tensor,
        pphi: torch.Tensor,
        *,
        r: torch.Tensor | None = None,
        hamiltonian: TEOBResumSHamiltonian | None = None,
        include_horizon_absorption: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if r is None or hamiltonian is None:
            raise ValueError("leading-order horizon diagnostic requires the full clean state")
        if not isinstance(include_horizon_absorption, bool):
            raise TypeError("include_horizon_absorption must be a bool")
        h_eff, h_real, h_circ, a_prime = hamiltonian.clean_radiation_inputs(
            r, pr_star, pphi,
        )
        infinity_flux = self._radiation.flux(
            r,
            omega,
            torch.zeros_like(omega),
            pr_star,
            pphi,
            h_eff,
            h_real,
            h_circ,
            a_prime,
            hamiltonian.m1,
            hamiltonian.m2,
            include_horizon=False,
        )
        flux = infinity_flux
        if include_horizon_absorption:
            r_omega = non_keplerian_radius(r, h_circ, a_prime)
            vphi = r_omega * omega
            total_mass = hamiltonian.m1 + hamiltonian.m2
            x1 = hamiltonian.m1 / total_mass
            x2 = hamiltonian.m2 / total_mass
            horizon_ratio = (x1**4 + x2**4) * vphi**8
            horizon_flux = (
                (32.0 / 5.0)
                * hamiltonian.nu**2
                * r_omega**4
                * omega**5
                * horizon_ratio
            )
            flux = infinity_flux + horizon_flux
        nu = hamiltonian.nu
        if omega.dim() > nu.dim():
            nu = nu.unsqueeze(-1)
        return torch.zeros_like(flux), -flux / nu


def _radiation_reaction(
    hamiltonian: TEOBResumSHamiltonian,
    policy: str,
) -> object:
    if policy == "current":
        return TEOBResumSRadiationReaction(
            hamiltonian.nu,
            hamiltonian.chi1,
            hamiltonian.chi2,
            include_horizon_absorption=True,
        )
    if policy == "leading-order":
        return _LeadingOrderHorizonReaction()
    raise ValueError(f"unknown horizon policy {policy!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizon-policy",
        choices=("current", "leading-order"),
        default="current",
        help="current clean horizon baseline or diagnostic-only published LO replacement",
    )
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--public-meta", type=Path, default=DEFAULT_PUBLIC_META)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    args = parser.parse_args()
    if args.horizon_policy == "current":
        args.output = args.output or DEFAULT_OUTPUT
        args.cache = args.cache or DEFAULT_CACHE
        args.log_path = args.log_path or DEFAULT_LOG
    else:
        args.output = args.output or DEFAULT_LO_OUTPUT
        args.cache = args.cache or DEFAULT_LO_CACHE
        args.log_path = args.log_path or DEFAULT_LO_LOG
    return args


def _load_public(path: Path, metadata_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    digest = _sha256(path)
    if digest != EXPECTED_PUBLIC_SHA256:
        raise RuntimeError(
            f"public fixture hash mismatch: expected {EXPECTED_PUBLIC_SHA256}, got {digest}"
        )
    with np.load(path, allow_pickle=False) as data:
        required = {"hp", "hc", "epoch", "delta_t", "msec"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"public fixture is missing {sorted(missing)}")
        hp = np.asarray(data["hp"], dtype=np.float64)
        hc = np.asarray(data["hc"], dtype=np.float64)
        epoch_seconds = float(data["epoch"])
        delta_t_seconds = float(data["delta_t"])
        msec_seconds = float(data["msec"])
    if hp.ndim != 1 or hc.shape != hp.shape:
        raise ValueError("public hp and hc must be one-dimensional and equal length")
    if not np.isfinite(hp).all() or not np.isfinite(hc).all():
        raise ValueError("public hp/hc contain nonfinite values")
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("sha256") != digest:
        raise RuntimeError("public metadata hash does not match fixture")
    radius = float(metadata.get("Kepler_radius_double"))
    if radius != PUBLIC_RADIUS:
        raise RuntimeError(
            f"public radius changed: expected {PUBLIC_RADIUS!r}, got {radius!r}"
        )
    delta_t_m = delta_t_seconds / msec_seconds
    if not math.isclose(delta_t_m, OUTPUT_STEP, rel_tol=0.0, abs_tol=1e-14):
        raise RuntimeError(f"public grid is not the contracted dt=0.5M grid: {delta_t_m}")
    public_time = np.arange(hp.size, dtype=np.float64) * delta_t_m
    public_mode = hp - 1j * hc
    return public_mode, public_time, {
        "path": str(path),
        "metadata_path": str(metadata_path),
        "sha256": digest,
        "metadata_sha256": _sha256(metadata_path),
        "samples": int(hp.size),
        "epoch_seconds": epoch_seconds,
        "epoch_initial_mass": epoch_seconds / msec_seconds,
        "delta_t_seconds": delta_t_seconds,
        "delta_t_initial_mass": delta_t_m,
        "msec_seconds": msec_seconds,
        "metadata": metadata,
    }


def _natural_cubic_numpy(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Evaluate a natural cubic spline with an independent Thomas solve."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or query.ndim != 1:
        raise ValueError("independent cubic evaluator expects one-dimensional arrays")
    if x.size < 2 or y.size != x.size:
        raise ValueError("independent cubic evaluator needs two or more matching knots")
    if np.any(np.diff(x) <= 0):
        raise ValueError("independent cubic knots must be strictly increasing")
    if np.any(query < x[0]) or np.any(query > x[-1]):
        raise ValueError("independent cubic query is outside its knot support")
    h = np.diff(x)
    second = np.zeros(x.size, dtype=np.float64)
    if x.size > 2:
        diagonal = 2.0 * (h[:-1] + h[1:])
        rhs = 6.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
        if rhs.size == 1:
            interior = rhs / diagonal
        else:
            upper = h[1:-1].copy()
            lower = h[1:-1].copy()
            diag = diagonal.copy()
            reduced_rhs = rhs.copy()
            for index in range(1, diag.size):
                factor = lower[index - 1] / diag[index - 1]
                diag[index] -= factor * upper[index - 1]
                reduced_rhs[index] -= factor * reduced_rhs[index - 1]
            interior = np.empty_like(rhs)
            interior[-1] = reduced_rhs[-1] / diag[-1]
            for index in range(interior.size - 2, -1, -1):
                interior[index] = (
                    reduced_rhs[index] - upper[index] * interior[index + 1]
                ) / diag[index]
        second[1:-1] = interior
    interval = np.searchsorted(x, query, side="right") - 1
    interval = np.clip(interval, 0, x.size - 2)
    dx = query - x[interval]
    hi = h[interval]
    a = y[interval]
    b = (y[interval + 1] - y[interval]) / hi - hi * (
        2.0 * second[interval] + second[interval + 1]
    ) / 6.0
    c = second[interval] / 2.0
    d = (second[interval + 1] - second[interval]) / (6.0 * hi)
    return a + dx * (b + dx * (c + dx * d))


def _complex_metrics(clean: np.ndarray, public: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = clean[mask] - public[mask]
    reference = public[mask]
    denominator = np.linalg.norm(reference)
    return {
        "samples": int(mask.sum()),
        "l2": float(np.linalg.norm(selected)),
        "relative_l2": float(np.linalg.norm(selected) / denominator) if denominator else math.nan,
        "rms": float(np.sqrt(np.mean(np.abs(selected) ** 2))) if selected.size else math.nan,
        "max_abs": float(np.max(np.abs(selected))) if selected.size else math.nan,
    }


def _amplitude_metrics(clean: np.ndarray, public: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    error = np.abs(clean[mask]) - np.abs(public[mask])
    reference = np.abs(public[mask])
    denominator = np.linalg.norm(reference)
    return {
        "samples": int(mask.sum()),
        "l2": float(np.linalg.norm(error)),
        "relative_l2": float(np.linalg.norm(error) / denominator) if denominator else math.nan,
        "rms": float(np.sqrt(np.mean(error * error))) if error.size else math.nan,
        "max_abs": float(np.max(np.abs(error))) if error.size else math.nan,
    }


def _phase_metrics(clean: np.ndarray, public: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = mask & (np.abs(clean) > 0.0) & (np.abs(public) > 0.0)
    if not np.any(valid):
        return {"samples": 0, "l2_radians": math.nan, "rms_radians": math.nan, "max_abs_radians": math.nan}
    angle = np.unwrap(np.angle(clean[valid] * np.conj(public[valid])))
    return {
        "samples": int(valid.sum()),
        "l2_radians": float(np.linalg.norm(angle)),
        "rms_radians": float(np.sqrt(np.mean(angle * angle))),
        "max_abs_radians": float(np.max(np.abs(angle))),
        "first_radians": float(angle[0]),
        "last_radians": float(angle[-1]),
    }


def _comparison_metrics(
    clean: np.ndarray,
    public: np.ndarray,
    output_time: np.ndarray,
    boundary_time: float,
) -> dict[str, Any]:
    if clean.shape != public.shape or output_time.shape != clean.shape:
        raise ValueError("comparison arrays must share the public grid")
    masks = {
        "full": np.ones(output_time.shape, dtype=bool),
        "pre_boundary": output_time < boundary_time,
        "post_boundary": output_time >= boundary_time,
    }
    return {
        region: {
            "raw_complex": _complex_metrics(clean, public, mask),
            "amplitude": _amplitude_metrics(clean, public, mask),
            "phase": _phase_metrics(clean, public, mask),
        }
        for region, mask in masks.items()
    }


def _events(
    result: RingdownAssemblyResult,
    grid: np.ndarray,
    final_mass: torch.Tensor,
) -> dict[str, Any]:
    events = result.events
    attachment_time = _float(events.attachment_time)
    output_index = int(np.searchsorted(grid, attachment_time, side="left"))
    output_index = min(max(output_index, 0), grid.size - 1)
    return {
        "peak_knot_index_auxiliary": int(events.peak_knot_index),
        "peak_time_initial_mass": _float(events.peak_time),
        "peak_time_final_mass": _float(events.peak_time / final_mass),
        "merger_time_initial_mass": _float(events.merger_time),
        "merger_time_final_mass": _float(events.merger_time / final_mass),
        "t_match_initial_mass": _float(events.match_time),
        "t_match_final_mass": _float(events.match_time_final_mass),
        "strictly_earlier_index_joined": int(events.strictly_earlier_index),
        "attachment_sampled_index_joined": int(events.attachment_index),
        "attachment_time_initial_mass": attachment_time,
        "attachment_time_final_mass": _float(events.attachment_time_final_mass),
        "output_boundary_index_public_grid": output_index,
        "output_boundary_time_initial_mass": float(grid[output_index]),
        "attachment_threshold_arithmetic": "t_peak/final_mass - 3/final_mass + 2/final_mass",
        "epoch_coordinate_start_initial_mass": _float(result.epoch),
        "joined_support_initial_mass": [_float(result.joined_time[0]), _float(result.joined_time[-1])],
        "extended_support_initial_mass": [_float(result.extended_time[0]), _float(result.extended_time[-1])],
        "output_support_initial_mass": [_float(result.time[0]), _float(result.time[-1])],
        "joined_samples": int(result.joined_time.numel()),
        "extended_samples": int(result.extended_time.numel()),
        "output_samples": int(result.time.numel()),
        "extension_samples": int(result.extension_samples),
        "auxiliary_spacing_initial_mass": _float(result.auxiliary_spacing),
        "ringdown_phase0": _float(result.ringdown_phase0),
        "auxiliary_amplitude_negative": bool(result.auxiliary_amplitude_negative),
    }


def _nqc_summary(nqc: Any) -> dict[str, Any]:
    return {
        "target_provenance": nqc.target_provenance,
        "targets": _vector(torch.stack(nqc.targets)),
        "amplitude_coefficients": _vector(nqc.coefficients.amplitude),
        "phase_coefficients": _vector(nqc.coefficients.phase),
        "matching_index_auxiliary": int(nqc.matching_index),
        "cut_start_index_original": int(nqc.cut_start_index),
        "cut_last_index_original": int(nqc.cut_last_index),
        "t_peak_grid_initial_mass": _float(nqc.t_peak_grid),
        "t_nqc_initial_mass": _float(nqc.t_nqc),
        "auxiliary_spacing_initial_mass": _float(nqc.auxiliary_spacing),
        "auxiliary_samples": int(nqc.auxiliary.time.numel()),
        "amplitude_condition": _float(nqc.match.amplitude_condition),
        "phase_condition": _float(nqc.match.phase_condition),
        "amplitude_residual": _vector(nqc.match.amplitude_residual),
        "phase_residual": _vector(nqc.match.phase_residual),
        "negative_auxiliary_corrected_amplitude_samples": int(nqc.auxiliary_negative_amplitude.sum()),
    }


def _assemble(
    *,
    time: torch.Tensor,
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    auxiliary_time: torch.Tensor,
    auxiliary_amplitude: torch.Tensor,
    auxiliary_phase: torch.Tensor,
    auxiliary_frequency: torch.Tensor,
    nu: torch.Tensor,
    final_mass: torch.Tensor,
    final_spin: torch.Tensor,
    matching_time: torch.Tensor,
    matching_state: torch.Tensor,
) -> RingdownAssemblyResult:
    return assemble_teob_ringdown_22(
        time,
        amplitude,
        phase,
        auxiliary_time,
        auxiliary_amplitude,
        auxiliary_phase,
        auxiliary_frequency,
        nu=nu,
        final_mass=final_mass,
        final_spin=final_spin,
        matching_time=matching_time,
        matching_state=matching_state,
        output_step=torch.as_tensor(OUTPUT_STEP, dtype=time.dtype, device=time.device),
    )


def _public_grid_branch(
    result: RingdownAssemblyResult,
    public_mode: np.ndarray,
    public_time: np.ndarray,
    projection_scale: float,
    final_mass: torch.Tensor,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Sample signed amplitude/lifted phase independently on the public grid."""

    extended_time = _numpy(result.extended_time)
    extended_amplitude = _numpy(result.extended_amplitude)
    extended_phase = _numpy(result.extended_phase)
    if public_time[-1] > extended_time[-1]:
        raise ValueError("assembled support does not cover the full public grid")
    independent_amplitude = _natural_cubic_numpy(
        extended_time, extended_amplitude, public_time,
    )
    independent_phase = _natural_cubic_numpy(
        extended_time, extended_phase, public_time,
    )
    independent_mode = projection_scale * independent_amplitude * np.exp(
        -1j * independent_phase
    )
    native_time = _numpy(result.time)
    native_mode = projection_scale * _numpy(result.mode)
    native_on_public = None
    if native_time.shape == public_time.shape and np.array_equal(native_time, public_time):
        native_on_public = native_mode
    boundary = _float(result.events.attachment_time)
    report: dict[str, Any] = {
        "events": _events(result, public_time, final_mass),
        "signed_amplitude_chart": {
            "extended_min": float(extended_amplitude.min()),
            "extended_max": float(extended_amplitude.max()),
            "extended_negative_samples": int(np.count_nonzero(extended_amplitude < 0.0)),
        },
        "native_assembly_grid": {
            "samples": int(native_time.size),
            "support_initial_mass": [float(native_time[0]), float(native_time[-1])],
            "matches_public_grid": native_on_public is not None,
            "max_abs_grid_difference_initial_mass": (
                float(np.max(np.abs(native_time - public_time)))
                if native_time.shape == public_time.shape else None
            ),
            "metrics_vs_public": (
                _comparison_metrics(native_on_public, public_mode, public_time, boundary)
                if native_on_public is not None else None
            ),
        },
        "independent_cubic_public_grid": {
            "samples": int(public_time.size),
            "support_initial_mass": [float(public_time[0]), float(public_time[-1])],
            "grid_step_initial_mass": OUTPUT_STEP,
            "sampling": "independent natural cubic on signed amplitude and lifted phase",
            "metrics_vs_public": _comparison_metrics(
                independent_mode, public_mode, public_time, boundary,
            ),
        },
    }
    return report, independent_mode, independent_amplitude, independent_phase


def _trajectory_prefix(
    *,
    time: torch.Tensor,
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    public_mode: np.ndarray,
    public_time: np.ndarray,
    projection_scale: float,
    boundary: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compare independent cubic trajectory charts on their native prefix."""

    x = _numpy(time)
    mask = public_time <= x[-1]
    prefix_time = public_time[mask]
    prefix_amplitude = _natural_cubic_numpy(x, _numpy(amplitude), prefix_time)
    prefix_phase = _natural_cubic_numpy(x, _numpy(phase), prefix_time)
    prefix_mode = projection_scale * prefix_amplitude * np.exp(-1j * prefix_phase)
    report = {
        "samples": int(prefix_time.size),
        "support_initial_mass": [float(prefix_time[0]), float(prefix_time[-1])],
        "full_public_support": bool(mask.all()),
        "native_endpoint_initial_mass": float(x[-1]),
        "sampling": "independent natural cubic on trajectory amplitude and lifted phase",
        "metrics_vs_public_prefix": _comparison_metrics(
            prefix_mode, public_mode[mask], prefix_time, boundary,
        ),
    }
    return report, prefix_time, prefix_mode, prefix_amplitude, prefix_phase


def _save_cache(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return _sha256(path)


def main() -> int:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    started = wall_time.perf_counter()
    source_paths = _source_paths()
    source_before = {name: _sha256(path) for name, path in source_paths.items()}
    command = shlex.join([sys.executable, *sys.argv])
    pid = os.getpid()
    report: dict[str, Any] = {
        "status": "running",
        "diagnostic_scope": "paired physical TEOB component composition; bounded postmerger baseline",
        "run_control": {
            "host": socket.gethostname(),
            "cwd": os.getcwd(),
            "command": command,
            "pid": pid,
            "python": sys.executable,
            "platform": platform.platform(),
            "device": "cpu",
            "started_unix": wall_time.time(),
            "log_path": str(args.log_path),
            "expected_next_check": "inspect diagnostic.json and cache SHA after the bounded process exits",
            "stop_command": f"kill {pid}",
        },
        "probe": {
            "nu": NU_VALUE,
            "delta": DELTA_VALUE,
            "chi1": 0.0,
            "chi2": 0.0,
            "lambda1": 0.0,
            "lambda2": 0.0,
            "r0": PUBLIC_RADIUS,
            "r_min": R_MIN,
            "dr": DR,
            "pa_order": PA_ORDER,
            "include_horizon_absorption": True,
            "horizon_policy": args.horizon_policy,
            "horizon_policy_description": (
                "current clean TEOBResumS horizon prescription"
                if args.horizon_policy == "current"
                else "diagnostic-only leading-order zero-spin Eq. (98)-(99) replacement"
            ),
            "fixed_step_cap": DEFAULT_FIXED_STEP_CAP,
            "continuation_duration": DEFAULT_CONTINUATION_DURATION,
            "atol": DEFAULT_ATOL,
            "rtol": DEFAULT_RTOL,
            "max_rejected_steps": DEFAULT_MAX_REJECTED_STEPS,
            "max_evaluations": DEFAULT_MAX_EVALUATIONS,
            "max_accepted_steps": DEFAULT_MAX_ACCEPTED_STEPS,
            "max_duration": DEFAULT_MAX_DURATION,
            "output_step_initial_mass": OUTPUT_STEP,
        },
        "source_sha256_before": source_before,
        "output_path": str(args.output),
        "cache_path": str(args.cache),
    }

    try:
        public_mode, public_time, public_data = _load_public(
            args.public, args.public_meta,
        )
        try:
            import lal  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment contract
            raise RuntimeError("the exact public projection requires installed lal constants") from exc
        projection_scale = -math.sqrt(5.0 / (4.0 * math.pi)) * (
            50.0 * lal.MTSUN_SI * lal.C_SI / (DISTANCE_MPC * 1e6 * lal.PC_SI)
        )
        report["public_fixture"] = {
            **public_data,
            "relative_support_initial_mass": [float(public_time[0]), float(public_time[-1])],
            "absolute_support_initial_mass": [
                float(public_data["epoch_initial_mass"] + public_time[0]),
                float(public_data["epoch_initial_mass"] + public_time[-1]),
            ],
            "projection_scale_h22_to_faceon_strain": projection_scale,
            "projection_scale_formula": "-sqrt(5/(4*pi))*(50*lal.MTSUN_SI*lal.C_SI)/(100e6*lal.PC_SI)",
        }

        hamiltonian = _hamiltonian()
        radiation_reaction = _radiation_reaction(hamiltonian, args.horizon_policy)
        pa = alternating8pa(
            hamiltonian,
            radiation_reaction,
            r0=PUBLIC_RADIUS,
            r_min=R_MIN,
            dr=DR,
            pa_order=PA_ORDER,
            include_horizon_absorption=True,
        )
        if not pa.complete:
            raise RuntimeError(f"PA profile did not complete: {pa.skip_reason}")
        continuation = continue_teob_adaptive(
            hamiltonian,
            pa,
            radiation_reaction,
            include_horizon_absorption=True,
            fixed_step_cap=DEFAULT_FIXED_STEP_CAP,
            continuation_duration=DEFAULT_CONTINUATION_DURATION,
            max_rejected_steps=DEFAULT_MAX_REJECTED_STEPS,
            max_evaluations=DEFAULT_MAX_EVALUATIONS,
            max_accepted_steps=DEFAULT_MAX_ACCEPTED_STEPS,
            max_duration=DEFAULT_MAX_DURATION,
        )
        if not continuation.complete:
            raise RuntimeError(f"continuation did not complete: {continuation.skip_reason}")
        trajectory_time = continuation.time
        state = continuation.state
        orbital_frequency = continuation.frequency
        with torch.no_grad():
            h_eff, h_real, h_circ, a_prime = hamiltonian.clean_radiation_inputs(
                state[:, 0], state[:, 2], state[:, 3],
            )
            # The state adapter calls TEOBRadiationClean.mode directly;
            # that mode applies exp(-i*m*H*Omega) exactly once.  No external
            # tail factor is multiplied here.
            mode = teob_waveform_state_mode(
                state,
                hamiltonian,
                2,
                2,
                Omega=orbital_frequency,
                residual_phase="zero-unpublished",
            )
            amplitude = mode.abs()
            phase = phase_from_mode(mode, state[:, 1])
            radial_acceleration = conservative_trajectory_quantities(
                hamiltonian, state,
            ).radial_acceleration
        nu = torch.tensor(NU_VALUE, dtype=torch.float64)
        targets = published_2017_nqc_targets(nu, torch.tensor(0.0, dtype=torch.float64))
        nqc = assemble_teob_merger_nqc(
            trajectory_time,
            state,
            amplitude=amplitude,
            phase=phase,
            orbital_frequency=orbital_frequency,
            radial_acceleration=radial_acceleration,
            nu=nu,
            fixed_step=torch.tensor(DEFAULT_FIXED_STEP_CAP, dtype=torch.float64),
            targets=targets,
            target_provenance=PUBLISHED_2017_NQC_PROVENANCE,
        )
        final_mass, final_spin = uib2016_nonspinning_remnant(
            MASS1_MSUN, MASS2_MSUN,
        )
        final_mass = final_mass.to(dtype=torch.float64)
        final_spin = final_spin.to(dtype=torch.float64)
        matching_state = nqc.auxiliary.state[nqc.matching_index]
        raw_assembly = _assemble(
            time=trajectory_time,
            amplitude=amplitude,
            phase=phase,
            auxiliary_time=nqc.auxiliary.time,
            auxiliary_amplitude=nqc.auxiliary.amplitude,
            auxiliary_phase=nqc.auxiliary.phase,
            auxiliary_frequency=nqc.auxiliary.orbital_frequency,
            nu=nu,
            final_mass=final_mass,
            final_spin=final_spin,
            matching_time=nqc.t_nqc,
            matching_state=matching_state,
        )
        nqc_assembly = _assemble(
            time=trajectory_time,
            amplitude=nqc.original.corrected_amplitude,
            phase=nqc.original.corrected_phase,
            auxiliary_time=nqc.auxiliary.time,
            auxiliary_amplitude=nqc.auxiliary.corrected_amplitude,
            auxiliary_phase=nqc.auxiliary.corrected_phase,
            auxiliary_frequency=nqc.auxiliary.orbital_frequency,
            nu=nu,
            final_mass=final_mass,
            final_spin=final_spin,
            matching_time=nqc.t_nqc,
            matching_state=matching_state,
        )

        raw_public, raw_output_mode, raw_output_amplitude, raw_output_phase = _public_grid_branch(
            raw_assembly, public_mode, public_time, projection_scale, final_mass,
        )
        nqc_public, nqc_output_mode, nqc_output_amplitude, nqc_output_phase = _public_grid_branch(
            nqc_assembly, public_mode, public_time, projection_scale, final_mass,
        )
        boundary = _float(nqc_assembly.events.attachment_time)
        trajectory_raw, trajectory_raw_time, trajectory_raw_mode, trajectory_raw_amplitude, trajectory_raw_phase = _trajectory_prefix(
            time=trajectory_time,
            amplitude=amplitude,
            phase=phase,
            public_mode=public_mode,
            public_time=public_time,
            projection_scale=projection_scale,
            boundary=boundary,
        )
        trajectory_nqc, trajectory_nqc_time, trajectory_nqc_mode, trajectory_nqc_amplitude, trajectory_nqc_phase = _trajectory_prefix(
            time=trajectory_time,
            amplitude=nqc.original.corrected_amplitude,
            phase=nqc.original.corrected_phase,
            public_mode=public_mode,
            public_time=public_time,
            projection_scale=projection_scale,
            boundary=boundary,
        )

        stats = continuation.statistics
        report.update({
            "pa": {
                "status": pa.status,
                "samples": int(pa.time.numel()),
                "intervals": int(pa.time.numel() - 1),
                "radius_support": [_float(pa.radius[0]), _float(pa.radius[-1])],
                "time_support_initial_mass": [_float(pa.time[0]), _float(pa.time[-1])],
                "phase_support_radians": [_float(pa.phase[0]), _float(pa.phase[-1])],
                "endpoint_state": _vector(pa.endpoint_state),
                "endpoint_time_initial_mass": _float(pa.endpoint_time),
                "endpoint_radius_exact_input": _float(pa.radius[0]),
            },
            "continuation": {
                "status": continuation.status,
                "samples": int(continuation.time.numel()),
                "adaptive_samples": int(continuation.adaptive_time.numel()),
                "fixed_samples": int(continuation.fixed_time.numel()),
                "support_initial_mass": [_float(continuation.time[0]), _float(continuation.time[-1])],
                "peak_event_index": continuation.peak_event_index,
                "peak_event_time_initial_mass": _float(continuation.peak_event_time),
                "peak_event_frequency": _float(continuation.peak_event_frequency),
                "endpoint_state": _vector(continuation.endpoint_state),
                "endpoint_time_initial_mass": _float(continuation.endpoint_time),
                "first_ode_step": _float(continuation.first_ode_step),
                "adaptive_next_step": _float(continuation.adaptive_next_step),
                "fixed_step": _float(continuation.fixed_step),
                "phase_seam_error": _float(continuation.phase_continuity),
                "statistics": {
                    "adaptive_accepted_steps": stats.adaptive_accepted_steps,
                    "adaptive_rejected_steps": stats.adaptive_rejected_steps,
                    "adaptive_evaluations": stats.adaptive_evaluations,
                    "fixed_accepted_steps": stats.fixed_accepted_steps,
                    "fixed_rejected_steps": stats.fixed_rejected_steps,
                    "fixed_evaluations": stats.fixed_evaluations,
                    "accepted_steps": stats.accepted_steps,
                    "rejected_steps": stats.rejected_steps,
                    "evaluations": stats.evaluations,
                    "termination": stats.termination,
                },
            },
            "mode": {
                "api": "torchwave.approximants.eob.teob_waveform_state_clean.teob_waveform_state_mode",
                "underlying_api": "torchwave.approximants.eob.teob_radiation_clean.TEOBRadiationClean.mode",
                "provider_configuration": {
                    "lmax": 8,
                    "ell": 2,
                    "m": 2,
                    "residual_phase": "zero-unpublished",
                },
                "samples": int(mode.numel()),
                "dtype": str(mode.dtype),
                "amplitude_range": [_float(amplitude.min()), _float(amplitude.max())],
                "phase_range_radians": [_float(phase.min()), _float(phase.max())],
                "tail_gauge": {
                    "factor": "exp(-i*m*Hreal*Omega)",
                    "m": M_MODE,
                    "applications": 1,
                    "application_location": "inside TEOBRadiationClean.mode via teob_waveform_state_mode; no external factor",
                    "phase_increment_range_radians": [
                        _float((M_MODE * h_real * orbital_frequency).min()),
                        _float((M_MODE * h_real * orbital_frequency).max()),
                    ],
                    "hreal_range": [_float(h_real.min()), _float(h_real.max())],
                },
            },
            "horizon_reaction": {
                "policy": args.horizon_policy,
                "implementation": (
                    "TEOBResumSRadiationReaction clean current horizon"
                    if args.horizon_policy == "current"
                    else "local diagnostic _LeadingOrderHorizonReaction"
                ),
                "leading_order_source": (
                    str(LO_HORIZON_SOURCE)
                    if args.horizon_policy == "leading-order" else None
                ),
                "leading_order_source_sha256": (
                    _sha256(LO_HORIZON_SOURCE)
                    if args.horizon_policy == "leading-order" else None
                ),
                "leading_order_formula": (
                    "F_H=(32/5)*nu**2*r_omega**4*Omega**5*"
                    "(X1**4+X2**4)*vphi**8; F_phi,H=-F_H/nu"
                    if args.horizon_policy == "leading-order" else None
                ),
                "radial_force_convention": "F_r*=0; conservative radial acceleration is r_ddot",
            },
            "conservative_radial_acceleration": {
                "samples": int(radial_acceleration.numel()),
                "range": [_float(radial_acceleration.min()), _float(radial_acceleration.max())],
                "l2": _float(torch.linalg.vector_norm(radial_acceleration)),
                "finite": bool(torch.isfinite(radial_acceleration).all()),
            },
            "nqc": _nqc_summary(nqc),
            "remnant": {
                "model": "uib2016_nonspinning_remnant",
                "mass1_msun": MASS1_MSUN,
                "mass2_msun": MASS2_MSUN,
                "final_mass_fraction": _float(final_mass),
                "final_spin": _float(final_spin),
            },
            "assemblies": {
                "raw": raw_public,
                "nqc_corrected": nqc_public,
            },
            "trajectory_prefix_comparisons": {
                "raw": trajectory_raw,
                "nqc_corrected": trajectory_nqc,
            },
            "comparison_conventions": {
                "public_mode": "h_plus - 1j*h_cross",
                "projection": "exact 50*lal.MTSUN_SI*lal.C_SI/(100e6*lal.PC_SI) with -sqrt(5/(4*pi))",
                "raw_complex": "same-index unaligned complex strain samples",
                "amplitude": "absolute strain amplitude difference; signed input chart retained separately",
                "phase": "unwrapped angle(clean*conj(public)) in sample order",
                "post_boundary": "public samples with relative time >= native attachment time",
                "support": "full public uniform dt=0.5M grid for assembled branches; trajectory charts report native prefix",
                "alignment": "none",
                "fitting": "none",
            },
            "limitations": [
                "The radius 19.999999999999996 is taken exactly from public metadata; it is not varied or fit.",
                (
                    "This freezes the current clean horizon prescription as the baseline."
                    if args.horizon_policy == "current"
                    else "This is a diagnostic-only LO horizon equation probe; it does not select a production prescription."
                ),
                "The independent cubic public-grid branches are assembled outputs, while raw trajectory comparisons end at the accepted continuation endpoint.",
                "No full inspiral validation, alignment, support truncation, or production routing is claimed.",
            ],
        })

        cache_arrays = {
            "public_time": public_time,
            "public_mode": public_mode,
            "time": _numpy(trajectory_time),
            "state": _numpy(state),
            "frequency": _numpy(orbital_frequency),
            "mode": _numpy(mode),
            "amplitude": _numpy(amplitude),
            "phase": _numpy(phase),
            "h_eff": _numpy(h_eff),
            "h_real": _numpy(h_real),
            "h_circ": _numpy(h_circ),
            "a_prime": _numpy(a_prime),
            "radial_acceleration": _numpy(radial_acceleration),
            "pa_radius": _numpy(pa.radius),
            "pa_time": _numpy(pa.time),
            "pa_phase": _numpy(pa.phase),
            "pa_pr_star": _numpy(pa.pr_star),
            "pa_p_phi": _numpy(pa.p_phi),
            "pa_frequency": _numpy(pa.omega),
            "adaptive_time": _numpy(continuation.adaptive_time),
            "adaptive_state": _numpy(continuation.adaptive_state),
            "adaptive_frequency": _numpy(continuation.adaptive_frequency),
            "fixed_time": _numpy(continuation.fixed_time),
            "fixed_state": _numpy(continuation.fixed_state),
            "fixed_frequency": _numpy(continuation.fixed_frequency),
            "nqc_corrected_amplitude": _numpy(nqc.original.corrected_amplitude),
            "nqc_corrected_phase": _numpy(nqc.original.corrected_phase),
            "nqc_aux_time": _numpy(nqc.auxiliary.time),
            "nqc_aux_state": _numpy(nqc.auxiliary.state),
            "nqc_aux_amplitude": _numpy(nqc.auxiliary.amplitude),
            "nqc_aux_phase": _numpy(nqc.auxiliary.phase),
            "nqc_aux_frequency": _numpy(nqc.auxiliary.orbital_frequency),
            "nqc_aux_radial_acceleration": _numpy(nqc.auxiliary.radial_acceleration),
            "nqc_aux_corrected_amplitude": _numpy(nqc.auxiliary.corrected_amplitude),
            "nqc_aux_corrected_phase": _numpy(nqc.auxiliary.corrected_phase),
            "nqc_aux_negative_amplitude": _numpy(nqc.auxiliary_negative_amplitude),
            "nqc_amplitude_coefficients": _numpy(nqc.coefficients.amplitude),
            "nqc_phase_coefficients": _numpy(nqc.coefficients.phase),
            "nqc_targets": _numpy(torch.stack(nqc.targets)),
            "raw_output_time": public_time,
            "raw_output_mode": raw_output_mode,
            "raw_output_amplitude": raw_output_amplitude,
            "raw_output_phase": raw_output_phase,
            "nqc_output_time": public_time,
            "nqc_output_mode": nqc_output_mode,
            "nqc_output_amplitude": nqc_output_amplitude,
            "nqc_output_phase": nqc_output_phase,
            "trajectory_raw_public_time": trajectory_raw_time,
            "trajectory_raw_public_mode": trajectory_raw_mode,
            "trajectory_raw_public_amplitude": trajectory_raw_amplitude,
            "trajectory_raw_public_phase": trajectory_raw_phase,
            "trajectory_nqc_public_time": trajectory_nqc_time,
            "trajectory_nqc_public_mode": trajectory_nqc_mode,
            "trajectory_nqc_public_amplitude": trajectory_nqc_amplitude,
            "trajectory_nqc_public_phase": trajectory_nqc_phase,
            "raw_native_output_time": _numpy(raw_assembly.time),
            "raw_native_output_mode": _numpy(raw_assembly.mode),
            "nqc_native_output_time": _numpy(nqc_assembly.time),
            "nqc_native_output_mode": _numpy(nqc_assembly.mode),
        }
        cache_sha = _save_cache(args.cache, cache_arrays)
        report["output_cache"] = {
            "path": str(args.cache),
            "sha256": cache_sha,
            "arrays": sorted(cache_arrays),
            "reusable_public_grid": "raw_output_* and nqc_output_* are projected full-support dt=0.5M arrays",
        }
        report["status"] = "complete"
    except Exception as exc:  # pragma: no cover - diagnostic failure is reported
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}

    source_after = {name: _sha256(path) for name, path in source_paths.items()}
    report["source_sha256_after"] = source_after
    report["source_hashes_unchanged"] = source_before == source_after
    report["elapsed_seconds"] = wall_time.perf_counter() - started
    report["runtime"] = {
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "python": sys.executable,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "cache": str(args.cache),
        "cache_sha256": report.get("output_cache", {}).get("sha256"),
        "elapsed_seconds": report["elapsed_seconds"],
        "source_hashes_unchanged": report["source_hashes_unchanged"],
    }, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

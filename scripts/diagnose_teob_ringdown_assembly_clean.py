#!/usr/bin/env python3
"""Diagnose clean TEOB ringdown assembly on the frozen physical NQC cache.

The diagnostic consumes the reviewed NQC cache and does not evolve a
trajectory.  The cached corrected arrays are the authoritative pre-gauge
input.  A second, diagnostic-only branch applies the published TEOB tail
coordinate factor ``exp(-i*m*Hreal*Omega)`` to the same trajectory samples,
reruns the clean explicit-target NQC seam, and sends those arrays through the
same ringdown assembler.  The two branches are reported independently.

The public comparison is deliberately raw and unaligned.  It uses the
face-on ``h_plus - i h_cross`` convention from the public binary fixture and
the exact 50-solar-mass/100-Mpc projection scale.  This is a component
diagnostic, not a trajectory or parameter-selection run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torchwave.approximants.eob.teob_merger_nqc_clean import (
    assemble_teob_merger_nqc,
)
from torchwave.approximants.eob.teob_ringdown_assembly_clean import (
    RingdownAssemblyResult,
    assemble_teob_ringdown_22,
)
from torchwave.approximants.eob.teob_uib2016_remnant import (
    uib2016_nonspinning_remnant,
)
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "verification/teob-ringdown-assembly-clean"
DEFAULT_NQC_CACHE = (
    ROOT
    / "docs/reviews/2026-09-13-parity-review/verification/teob-merger-nqc-clean"
    / "physical-input-output.npz"
)
DEFAULT_NQC_REPORT = DEFAULT_NQC_CACHE.with_name("diagnostic.json")
DEFAULT_PUBLIC = Path("/tmp/torchwave-public-teob-q1p5-r20-dt0p5M.npz")
DEFAULT_PUBLIC_META = DEFAULT_PUBLIC.with_suffix(".json")
DEFAULT_OUTPUT = VERIFY / "physical-diagnostic.json"
DEFAULT_OUTPUT_CACHE = VERIFY / "physical-assembly-output.npz"

EXPECTED_NQC_SHA256 = (
    "06bb6d88d9b148cc481c8b45a694ea0d132ec195f0c109fc83ba0e024421d334"
)
EXPECTED_PUBLIC_SHA256 = (
    "09eaa8b00288c94947b2bb4b40ada7780dc51ac98491a48cbb8f60ed50ab686c"
)
M_MODE = 2
MASS1_MSUN = 30.0
MASS2_MSUN = 20.0
DISTANCE_MPC = 100.0
OUTPUT_STEP_M = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_tensor(array: np.ndarray) -> torch.Tensor:
    # Clone read-only NPZ views before using them in Torch operations.
    return torch.from_numpy(np.asarray(array)).clone()


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
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _hamiltonian() -> TEOBResumSHamiltonian:
    dtype = torch.float64
    zero = torch.tensor(0.0, dtype=dtype)
    return TEOBResumSHamiltonian(
        torch.tensor(0.24, dtype=dtype), zero, zero, zero, zero,
    )


def _source_paths() -> dict[str, Path]:
    return {
        "script": Path(__file__).resolve(),
        "ringdown_assembly": ROOT / "src/torchwave/approximants/eob/teob_ringdown_assembly_clean.py",
        "postmerger": ROOT / "src/torchwave/approximants/eob/teob_postmerger.py",
        "uib_remnant": ROOT / "src/torchwave/approximants/eob/teob_uib2016_remnant.py",
        "nqc": ROOT / "src/torchwave/approximants/eob/teob_merger_nqc_clean.py",
        "hamiltonian": ROOT / "src/torchwave/approximants/teobresums.py",
        "assembly_contract": ROOT / "docs/reviews/2026-09-13-parity-review/teob-merger-assembly-contract.md",
    }


def _load_public(path: Path, metadata_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
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
        epoch = float(data["epoch"])
        delta_t = float(data["delta_t"])
        msec = float(data["msec"])
    if hp.ndim != 1 or hc.shape != hp.shape:
        raise ValueError("public hp and hc must be one-dimensional and equal length")
    if not np.isfinite(hp).all() or not np.isfinite(hc).all():
        raise ValueError("public hp/hc contain nonfinite values")
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("sha256") != digest:
        raise RuntimeError("public metadata hash does not match fixture")
    return (
        hp - 1j * hc,
        {
            "path": str(path),
            "metadata_path": str(metadata_path),
            "sha256": digest,
            "samples": int(hp.size),
            "epoch_seconds": epoch,
            "delta_t_seconds": delta_t,
            "msec_seconds": msec,
            "metadata": metadata,
        },
    )


def _load_nqc(
    cache_path: Path, report_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    digest = _sha256(cache_path)
    if digest != EXPECTED_NQC_SHA256:
        raise RuntimeError(
            f"NQC cache hash mismatch: expected {EXPECTED_NQC_SHA256}, got {digest}"
        )
    report = json.loads(report_path.read_text())
    if report.get("cache_sha256") != digest:
        raise RuntimeError("NQC diagnostic does not describe the supplied cache")
    required = (
        "time", "state", "mode", "amplitude", "phase", "orbital_frequency",
        "radial_acceleration", "corrected_amplitude", "corrected_phase",
        "aux_time", "aux_state", "aux_amplitude", "aux_phase",
        "aux_orbital_frequency", "aux_radial_acceleration",
        "aux_corrected_amplitude", "aux_corrected_phase",
    )
    with np.load(cache_path, allow_pickle=False) as data:
        missing = set(required).difference(data.files)
        if missing:
            raise ValueError(f"NQC cache is missing {sorted(missing)}")
        arrays = {name: _as_tensor(data[name]) for name in required}
    return arrays, {
        "path": str(cache_path),
        "report_path": str(report_path),
        "sha256": digest,
        "report": report,
    }


def _assemble(
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    auxiliary_amplitude: torch.Tensor,
    auxiliary_phase: torch.Tensor,
    auxiliary_orbital_frequency: torch.Tensor,
    *,
    time: torch.Tensor,
    auxiliary_time: torch.Tensor,
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
        auxiliary_orbital_frequency,
        nu=nu,
        final_mass=final_mass,
        final_spin=final_spin,
        matching_time=matching_time,
        matching_state=matching_state,
        output_step=torch.as_tensor(OUTPUT_STEP_M, dtype=time.dtype),
    )


def _phase_metrics(clean: np.ndarray, public: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & (np.abs(clean) > 0.0) & (np.abs(public) > 0.0)
    if not np.any(valid):
        return {"samples": 0, "l2": math.nan, "rms": math.nan, "max_abs": math.nan}
    # The ratio retains the unshifted carrier convention, while unwrap keeps
    # the reported accumulation continuous across the public phase wraps.
    ratio_angle = np.unwrap(np.angle(clean[valid] * np.conj(public[valid])))
    return {
        "samples": int(valid.sum()),
        "l2": float(np.linalg.norm(ratio_angle)),
        "rms": float(np.sqrt(np.mean(ratio_angle * ratio_angle))),
        "max_abs": float(np.max(np.abs(ratio_angle))),
        "first": float(ratio_angle[0]),
        "last": float(ratio_angle[-1]),
    }


def _amplitude_metrics(clean: np.ndarray, public: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    clean_amplitude = np.abs(clean)
    public_amplitude = np.abs(public)
    error = clean_amplitude[mask] - public_amplitude[mask]
    denominator = np.linalg.norm(public_amplitude[mask])
    return {
        "samples": int(mask.sum()),
        "l2": float(np.linalg.norm(error)),
        "relative_l2": float(np.linalg.norm(error) / denominator) if denominator else math.nan,
        "rms": float(np.sqrt(np.mean(error * error))),
        "max_abs": float(np.max(np.abs(error))) if error.size else math.nan,
    }


def _metrics(
    clean: np.ndarray,
    public: np.ndarray,
    output_time: np.ndarray,
    boundary_time: float,
) -> dict[str, Any]:
    if clean.shape != public.shape:
        raise ValueError(f"assembled/public shape mismatch: {clean.shape} vs {public.shape}")
    delta = clean - public
    post_mask = output_time >= boundary_time
    pre_mask = ~post_mask

    def complex_metrics(mask: np.ndarray) -> dict[str, float]:
        selected = delta[mask]
        public_selected = public[mask]
        denominator = np.linalg.norm(public_selected)
        return {
            "samples": int(mask.sum()),
            "l2": float(np.linalg.norm(selected)),
            "relative_l2": float(np.linalg.norm(selected) / denominator) if denominator else math.nan,
            "rms": float(np.sqrt(np.mean(np.abs(selected) ** 2))),
            "max_abs": float(np.max(np.abs(selected))) if selected.size else math.nan,
        }

    return {
        "full": complex_metrics(np.ones(output_time.shape, dtype=bool)),
        "pre_boundary": complex_metrics(pre_mask),
        "post_boundary": complex_metrics(post_mask),
        "amplitude_full": _amplitude_metrics(clean, public, np.ones(output_time.shape, dtype=bool)),
        "amplitude_post_boundary": _amplitude_metrics(clean, public, post_mask),
        "phase_full_radians": _phase_metrics(clean, public, np.ones(output_time.shape, dtype=bool)),
        "phase_post_boundary_radians": _phase_metrics(clean, public, post_mask),
    }


def _events(result: RingdownAssemblyResult, output_time: np.ndarray) -> dict[str, Any]:
    events = result.events
    attachment_time = _float(events.attachment_time)
    output_index = int(np.searchsorted(output_time, attachment_time, side="left"))
    output_index = min(max(output_index, 0), output_time.size - 1)
    return {
        "peak_knot_index_auxiliary": int(events.peak_knot_index),
        "peak_time_initial_mass": _float(events.peak_time),
        "peak_first_derivative": _float(events.peak_first_derivative),
        "peak_second_derivative": _float(events.peak_second_derivative),
        "merger_time_initial_mass": _float(events.merger_time),
        "match_time_initial_mass": _float(events.match_time),
        "match_time_final_mass": _float(events.match_time_final_mass),
        "strictly_earlier_index_joined": int(events.strictly_earlier_index),
        "attachment_index_joined": int(events.attachment_index),
        "attachment_time_initial_mass": attachment_time,
        "attachment_time_final_mass": _float(events.attachment_time_final_mass),
        "output_boundary_index": output_index,
        "output_boundary_time_initial_mass": float(output_time[output_index]),
        "epoch_coordinate_start_initial_mass": _float(result.epoch),
        "joined_support_initial_mass": [
            _float(result.joined_time[0]), _float(result.joined_time[-1]),
        ],
        "extended_support_initial_mass": [
            _float(result.extended_time[0]), _float(result.extended_time[-1]),
        ],
        "output_support_initial_mass": [
            _float(result.time[0]), _float(result.time[-1]),
        ],
        "joined_samples": int(result.joined_time.numel()),
        "extended_samples": int(result.extended_time.numel()),
        "output_samples": int(result.time.numel()),
        "extension_samples": int(result.extension_samples),
        "auxiliary_spacing_initial_mass": _float(result.auxiliary_spacing),
        "ringdown_phase0": _float(result.ringdown_phase0),
        "auxiliary_amplitude_negative": bool(result.auxiliary_amplitude_negative),
    }


def _nqc_branch_summary(result: Any) -> dict[str, Any]:
    return {
        "amplitude_coefficients": _vector(result.coefficients.amplitude),
        "phase_coefficients": _vector(result.coefficients.phase),
        "matching_index": int(result.matching_index),
        "t_peak_grid": _float(result.t_peak_grid),
        "t_nqc": _float(result.t_nqc),
        "auxiliary_spacing": _float(result.auxiliary_spacing),
        "auxiliary_samples": int(result.auxiliary.time.numel()),
        "negative_auxiliary_samples": int(result.auxiliary_negative_amplitude.sum()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_NQC_CACHE)
    parser.add_argument("--nqc-report", type=Path, default=DEFAULT_NQC_REPORT)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--public-meta", type=Path, default=DEFAULT_PUBLIC_META)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-cache", type=Path, default=DEFAULT_OUTPUT_CACHE)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    arrays, nqc_data = _load_nqc(args.cache, args.nqc_report)
    public_mode, public_data = _load_public(args.public, args.public_meta)
    nqc_report = nqc_data["report"]

    dtype = torch.float64
    time = arrays["time"].to(dtype=dtype)
    state = arrays["state"].to(dtype=dtype)
    amplitude = arrays["amplitude"].to(dtype=dtype)
    phase = arrays["phase"].to(dtype=dtype)
    orbital_frequency = arrays["orbital_frequency"].to(dtype=dtype)
    radial_acceleration = arrays["radial_acceleration"].to(dtype=dtype)
    auxiliary_time = arrays["aux_time"].to(dtype=dtype)
    auxiliary_state = arrays["aux_state"].to(dtype=dtype)
    auxiliary_orbital_frequency = arrays["aux_orbital_frequency"].to(dtype=dtype)

    nu_value = float(nqc_report["probe"]["nu"])
    fixed_step = float(nqc_report["probe"]["fixed_step_cap"])
    targets = torch.as_tensor(nqc_report["nqc"]["targets"], dtype=dtype)
    target_provenance = str(nqc_report["nqc"]["target_provenance"])
    nu = torch.tensor(nu_value, dtype=dtype)
    final_mass, final_spin = uib2016_nonspinning_remnant(MASS1_MSUN, MASS2_MSUN)
    final_mass = final_mass.to(dtype=dtype)
    final_spin = final_spin.to(dtype=dtype)

    # The authoritative branch consumes the frozen corrected arrays byte for
    # byte.  Its metadata is recovered from the cached auxiliary matching
    # knot only to keep the assembly hand-off explicit.
    cached_matching_index = int(nqc_report["nqc"]["matching_index"])
    cached_t_nqc = torch.tensor(float(nqc_report["nqc"]["t_nqc"]), dtype=dtype)
    cached_matching_state = auxiliary_state[cached_matching_index]
    authoritative = _assemble(
        arrays["corrected_amplitude"].to(dtype=dtype),
        arrays["corrected_phase"].to(dtype=dtype),
        arrays["aux_corrected_amplitude"].to(dtype=dtype),
        arrays["aux_corrected_phase"].to(dtype=dtype),
        auxiliary_orbital_frequency,
        time=time,
        auxiliary_time=auxiliary_time,
        nu=nu,
        final_mass=final_mass,
        final_spin=final_spin,
        matching_time=cached_t_nqc,
        matching_state=cached_matching_state,
    )

    # Recreate the published TEOB tail coordinate factor on exactly the
    # frozen full and auxiliary states.  This is diagnostic-only; no default
    # mode or production route is changed.  Hreal is evaluated by the clean
    # Hamiltonian API, and the same factor is applied once to the phase chart.
    hamiltonian = _hamiltonian()
    hreal = hamiltonian.real_hamiltonian(
        state[:, 0], state[:, 2], state[:, 3],
    )
    auxiliary_hreal = hamiltonian.real_hamiltonian(
        auxiliary_state[:, 0], auxiliary_state[:, 2], auxiliary_state[:, 3],
    )
    gauge_phase_increment = M_MODE * hreal * orbital_frequency
    gauge_phase = phase + gauge_phase_increment
    cached_mode = arrays["mode"].to(dtype=torch.complex128)
    gauge_mode_before_nqc = cached_mode * torch.exp(
        torch.complex(torch.zeros_like(gauge_phase_increment), -gauge_phase_increment)
    )
    gauge_mode_from_phase = torch.complex(amplitude, torch.zeros_like(amplitude)) * torch.exp(
        torch.complex(torch.zeros_like(gauge_phase), -gauge_phase)
    )
    gauge_mode_reconstruction_error = (gauge_mode_before_nqc - gauge_mode_from_phase).abs().max()
    gauge_nqc = assemble_teob_merger_nqc(
        time,
        state,
        amplitude=amplitude,
        phase=gauge_phase,
        orbital_frequency=orbital_frequency,
        radial_acceleration=radial_acceleration,
        nu=nu,
        fixed_step=torch.tensor(fixed_step, dtype=dtype),
        targets=targets,
        target_provenance=target_provenance,
    )
    # The NQC implementation independently interpolates the gauge phase on
    # its own auxiliary grid.  Use those returned arrays rather than the
    # precomputed auxiliary chart so the branch is a real rerun of the seam.
    gauge_matching_state = gauge_nqc.auxiliary.state[gauge_nqc.matching_index]
    gauge_assembled = _assemble(
        gauge_nqc.original.corrected_amplitude,
        gauge_nqc.original.corrected_phase,
        gauge_nqc.auxiliary.corrected_amplitude,
        gauge_nqc.auxiliary.corrected_phase,
        gauge_nqc.auxiliary.orbital_frequency,
        time=gauge_nqc.original.time,
        auxiliary_time=gauge_nqc.auxiliary.time,
        nu=nu,
        final_mass=final_mass,
        final_spin=final_spin,
        matching_time=gauge_nqc.t_nqc,
        matching_state=gauge_matching_state,
    )

    # Check the output grid against the public fixture's half-M sampling.
    output_time = _numpy(authoritative.time)
    msec = public_data["msec_seconds"]
    delta_t_m = public_data["delta_t_seconds"] / msec
    public_time = np.arange(public_mode.size, dtype=np.float64) * delta_t_m
    if output_time.shape != public_time.shape:
        raise ValueError("the contracted half-M output grid does not cover the public fixture")
    grid_error = float(np.max(np.abs(output_time - public_time)))

    # The face-on projection used by the public binary fixture.  Keep this
    # expression explicit so the mass and distance scale is auditable.
    try:
        import lal  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("the public projection requires the installed lal constants") from exc
    projection_scale = -math.sqrt(5.0 / (4.0 * math.pi)) * (
        50.0 * lal.MTSUN_SI * lal.C_SI / (100e6 * lal.PC_SI)
    )
    public_epoch_m = public_data["epoch_seconds"] / msec

    authoritative_mode = projection_scale * _numpy(authoritative.mode)
    gauge_mode = projection_scale * _numpy(gauge_assembled.mode)
    boundary_time = _float(authoritative.events.attachment_time)
    report = {
        "status": "complete",
        "diagnostic_scope": "component attribution; no trajectory rerun, alignment, fitting, or production routing",
        "inputs": {
            "nqc_cache": nqc_data,
            "public_fixture": public_data,
            "public_metadata": public_data["metadata"],
            "final_mass_fraction": _float(final_mass),
            "final_spin": _float(final_spin),
            "nu": nu_value,
            "fixed_step": fixed_step,
            "output_step_initial_mass": OUTPUT_STEP_M,
            "projection_scale_h22_to_faceon_strain": projection_scale,
            "projection_scale_formula": "-sqrt(5/(4*pi))*(50*lal.MTSUN_SI*lal.C_SI)/(100e6*lal.PC_SI)",
            "public_relative_time_initial_mass": [
                float(public_time[0]), float(public_time[-1]),
            ],
            "public_physical_time_initial_mass": [
                float(public_epoch_m + public_time[0]),
                float(public_epoch_m + public_time[-1]),
            ],
            "public_epoch_initial_mass": public_epoch_m,
            "output_grid_max_abs_difference_initial_mass": grid_error,
        },
        "frozen_trajectory_evidence": {
            "full_samples": int(time.numel()),
            "full_support_initial_mass": [_float(time[0]), _float(time[-1])],
            "full_initial_radius": _float(state[0, 0]),
            "auxiliary_samples": int(auxiliary_time.numel()),
            "auxiliary_support_initial_mass": [_float(auxiliary_time[0]), _float(auxiliary_time[-1])],
            "auxiliary_initial_radius": _float(auxiliary_state[0, 0]),
            "frozen_initial_radius_policy": "r0=20.0 exactly; cached PA chart has 61 knots",
            "public_initial_radius_policy": "public metadata reports inverse Kepler r0=19.999999999999996; public chart has 60 knots",
            "trajectory_difference_policy": "reported as a fixture limitation; no radius or knot adjustment",
        },
        "authoritative_pre_gauge": {
            "mode_semantics": "cached corrected mode, pre-TEOB-tail gauge; authoritative input",
            "events": _events(authoritative, output_time),
            "metrics_vs_public_unaligned": _metrics(authoritative_mode, public_mode, output_time, boundary_time),
        },
        "diagnostic_tail_gauge_rerun": {
            "mode_semantics": "same frozen trajectory, one exp(-i*m*Hreal*Omega) phase gauge, then clean NQC rerun",
            "tail_gauge": {
                "m": M_MODE,
                "factor": "exp(-i*m*Hreal*Omega)",
                "mode_factor_reconstruction_max_abs": _float(gauge_mode_reconstruction_error),
                "full_hreal_range": [_float(hreal.min()), _float(hreal.max())],
                "auxiliary_hreal_range": [_float(auxiliary_hreal.min()), _float(auxiliary_hreal.max())],
                "full_phase_increment_range": [
                    _float(gauge_phase_increment.min()),
                    _float(gauge_phase_increment.max()),
                ],
                "auxiliary_phase_increment_range": [
                    _float((M_MODE * auxiliary_hreal * auxiliary_orbital_frequency).min()),
                    _float((M_MODE * auxiliary_hreal * auxiliary_orbital_frequency).max()),
                ],
            },
            "nqc": _nqc_branch_summary(gauge_nqc),
            "coefficients_delta_vs_cached": {
                "amplitude": [
                    value - ref
                    for value, ref in zip(
                        _vector(gauge_nqc.coefficients.amplitude),
                        nqc_report["nqc"]["amplitude_coefficients"],
                    )
                ],
                "phase": [
                    value - ref
                    for value, ref in zip(
                        _vector(gauge_nqc.coefficients.phase),
                        nqc_report["nqc"]["phase_coefficients"],
                    )
                ],
            },
            "events": _events(gauge_assembled, _numpy(gauge_assembled.time)),
            "metrics_vs_public_unaligned": _metrics(
                gauge_mode, public_mode, output_time, _float(gauge_assembled.events.attachment_time),
            ),
        },
        "comparison_conventions": {
            "complex_public_mode": "h_plus - 1j*h_cross",
            "raw": "same-index complex samples on the absolute half-M grid; no time or phase alignment",
            "phase": "continuous angle of clean*conj(public), unwrapped in output order",
            "amplitude": "absolute strain amplitude difference",
            "post_boundary": "output samples with relative time >= native attachment time",
            "tail_gauge_application": "diagnostic branch only; frozen pre-gauge branch is not modified",
        },
        "source_hashes": {
            name: _sha256(path) for name, path in _source_paths().items()
        },
        "data_hashes": {
            "nqc_cache": _sha256(args.cache),
            "nqc_report": _sha256(args.nqc_report),
            "public_fixture": _sha256(args.public),
            "public_metadata": _sha256(args.public_meta),
        },
        "output_cache": {
            "path": str(args.output_cache),
            "sha256": None,
            "arrays": [
                "output_time", "public_mode", "authoritative_mode", "gauge_mode",
                "gauge_mode_before_nqc", "hreal", "auxiliary_hreal",
                "gauge_phase_increment",
            ],
        },
        "notes": [
            "The clean UIB2016 nonspinning remnant API supplies final mass fraction and spin.",
            "The ringdown ansatz is appended and sampled through the isolated assembly API; no runtime driver is changed.",
            "The tail-gauge branch is a component attribution diagnostic. Its residual ranking does not choose a derivative, gauge, or NQC policy.",
            "The frozen trajectory uses exact r0=20.0 and 61 PA knots, while the public binary metadata uses r0=19.999999999999996 and 60 knots; this difference is retained as-is.",
            "No full inspiral or trajectory validation is claimed; only the bounded cached postmerger support is compared.",
        ],
        "runtime": {
            "cwd": str(Path.cwd()),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.executable,
            "torch": torch.__version__,
        },
    }

    np.savez_compressed(
        args.output_cache,
        output_time=output_time,
        public_mode=public_mode,
        authoritative_mode=authoritative_mode,
        gauge_mode=gauge_mode,
        gauge_mode_before_nqc=_numpy(gauge_mode_before_nqc),
        hreal=_numpy(hreal),
        auxiliary_hreal=_numpy(auxiliary_hreal),
        gauge_phase_increment=_numpy(gauge_phase_increment),
    )
    report["output_cache"]["sha256"] = _sha256(args.output_cache)
    args.output.write_text(json.dumps(report, indent=2, default=_json_value) + "\n")
    print(json.dumps({
        "report": str(args.output),
        "output_cache": str(args.output_cache),
        "authoritative_post_raw_l2": report["authoritative_pre_gauge"]["metrics_vs_public_unaligned"]["post_boundary"]["l2"],
        "authoritative_post_amplitude_l2": report["authoritative_pre_gauge"]["metrics_vs_public_unaligned"]["amplitude_post_boundary"]["l2"],
        "gauge_post_raw_l2": report["diagnostic_tail_gauge_rerun"]["metrics_vs_public_unaligned"]["post_boundary"]["l2"],
        "gauge_post_amplitude_l2": report["diagnostic_tail_gauge_rerun"]["metrics_vs_public_unaligned"]["amplitude_post_boundary"]["l2"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare clean TEOB radial acceleration conventions on paired trajectories.

This diagnostic consumes an already paired physical trajectory cache.  It does
not evolve a trajectory or regenerate the TEOB carrier.  The cached amplitude,
lifted phase, orbital frequency, state and public grid are sent through the
explicit 2017 NQC seam and the clean UIB2016 ringdown assembler for each of
the three documented radial acceleration choices:

``exact``
    The existing conservative chain-rule ``r_ddot`` helper.
``strict_p0``
    The strict leading-radial-momentum approximation, with mobility and
    conservative force evaluated at ``p_r*=0``.
``frozen_mobility``
    The explicitly named compatibility hypothesis ``C_current * pdot_current``.

All waveform comparisons are same-index, on the complete 0.5M public grid,
and unaligned.  The cache is treated as read-only; source and cache hashes are
recorded before and after the run so a concurrent edit is visible.
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

from torchwave.approximants.eob.teob_merger_nqc_clean import (
    PUBLISHED_2017_NQC_PROVENANCE,
    assemble_teob_merger_nqc,
    published_2017_nqc_targets,
)
from torchwave.approximants.eob.teob_radial_acceleration_clean import (
    radial_acceleration_approximations,
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
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "verification/teob-paired-driver-clean/paired-physical-output.npz"
DEFAULT_LO_CACHE = ROOT / "verification/teob-paired-driver-clean/paired-physical-output-lo-horizon.npz"
DEFAULT_INVENTORY_GLOB = "verification/teob-paired-driver-clean/paired-physical-output*.npz"
VERIFY = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-radial-acceleration-paired-clean"
DEFAULT_OUTPUT = VERIFY / "diagnostic.json"
DEFAULT_OUTPUT_CACHE = VERIFY / "paired-radial-acceleration-output.npz"
DEFAULT_LOG = VERIFY / "probe.log"

NU = 0.24
DELTA = 0.2
MASS1_MSUN = 30.0
MASS2_MSUN = 20.0
OUTPUT_STEP = 0.5
M_MODE = 2
BRANCHES = ("exact", "frozen_mobility", "strict_p0")
EXPECTED_RADIATION_SHA256 = (
    "dca2c9791403467d85572f25a062cbc995d8228743d33b6b2e5dc28c13336bd6"
)

REQUIRED_CACHE_KEYS = {
    "public_time",
    "public_mode",
    "time",
    "state",
    "frequency",
    "mode",
    "amplitude",
    "phase",
    "radial_acceleration",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths() -> dict[str, Path]:
    return {
        "script": Path(__file__).resolve(),
        "radial_acceleration": ROOT / "src/torchwave/approximants/eob/teob_radial_acceleration_clean.py",
        "trajectory": ROOT / "src/torchwave/approximants/eob/teob_trajectory.py",
        "nqc": ROOT / "src/torchwave/approximants/eob/teob_merger_nqc_clean.py",
        "nqc_basis": ROOT / "src/torchwave/approximants/eob/teob_nqc_clean.py",
        "ringdown_assembly": ROOT / "src/torchwave/approximants/eob/teob_ringdown_assembly_clean.py",
        "postmerger": ROOT / "src/torchwave/approximants/eob/teob_postmerger.py",
        "uib_remnant": ROOT / "src/torchwave/approximants/eob/teob_uib2016_remnant.py",
        "hamiltonian": ROOT / "src/torchwave/approximants/teobresums.py",
        "radiation": ROOT / "src/torchwave/approximants/eob/teob_radiation_clean.py",
        "assembly_contract": ROOT / "docs/reviews/2026-09-13-parity-review/teob-merger-assembly-contract.md",
    }


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _vector(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1)]
    return [float(item) for item in np.asarray(value).reshape(-1)]


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


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


def _hamiltonian() -> TEOBResumSHamiltonian:
    dtype = torch.float64
    zero = torch.tensor(0.0, dtype=dtype)
    return TEOBResumSHamiltonian(
        torch.tensor(NU, dtype=dtype), zero, zero, zero, zero,
        delta=torch.tensor(DELTA, dtype=dtype),
    )


def _as_tensor(array: np.ndarray, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    # NPZ arrays can be read-only views.  Clone before Torch consumes them.
    return torch.from_numpy(np.asarray(array)).clone().to(dtype=dtype)


def _validate_cache_arrays(data: dict[str, np.ndarray], path: Path) -> None:
    missing = REQUIRED_CACHE_KEYS.difference(data)
    if missing:
        raise ValueError(f"{path} is missing required arrays {sorted(missing)}")
    public_time = data["public_time"]
    public_mode = data["public_mode"]
    time = data["time"]
    state = data["state"]
    frequency = data["frequency"]
    amplitude = data["amplitude"]
    phase = data["phase"]
    radial_acceleration = data["radial_acceleration"]
    if public_time.ndim != 1 or public_mode.ndim != 1 or public_time.shape != public_mode.shape:
        raise ValueError("public_time and public_mode must be equal one-dimensional arrays")
    if time.ndim != 1 or state.shape != (time.size, 4):
        raise ValueError("time/state must have shapes (N,) and (N,4)")
    for name, value in {
        "public_time": public_time,
        "public_mode": public_mode,
        "time": time,
        "state": state,
        "frequency": frequency,
        "amplitude": amplitude,
        "phase": phase,
        "radial_acceleration": radial_acceleration,
    }.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{path}:{name} contains nonfinite values")
    if np.any(np.diff(public_time) <= 0) or np.any(np.diff(time) <= 0):
        raise ValueError(f"{path}: time axes must be strictly increasing")
    if any(np.asarray(data[name]).shape != time.shape for name in (
        "frequency", "amplitude", "phase", "radial_acceleration",
    )):
        raise ValueError(f"{path}: trajectory scalar arrays have inconsistent shapes")
    if np.any(state[:, 0] <= 0) or np.any(frequency <= 0) or np.any(amplitude <= 0):
        raise ValueError(f"{path}: trajectory leaves the positive-radius/frequency/amplitude domain")


def _load_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = path.resolve()
    before = _sha256(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]).copy() for name in archive.files}
    _validate_cache_arrays(data, path)
    # A paired cache contains the public grid, the frozen carrier and the
    # state/chart on which the branch comparison is performed.
    carrier_reconstruction = data["amplitude"] * np.exp(-1j * data["phase"])
    carrier_error = float(np.max(np.abs(carrier_reconstruction - data["mode"])))
    step = np.diff(data["public_time"])
    trajectory_step = np.diff(data["time"])
    info = {
        "path": str(path),
        "sha256_before": before,
        "bytes": path.stat().st_size,
        "keys": sorted(data),
        "public_samples": int(data["public_time"].size),
        "public_support_initial_mass": [float(data["public_time"][0]), float(data["public_time"][-1])],
        "trajectory_samples": int(data["time"].size),
        "trajectory_support_initial_mass": [float(data["time"][0]), float(data["time"][-1])],
        "public_grid_step_initial_mass": float(np.median(step)),
        "public_grid_uniform_max_error_initial_mass": float(np.max(np.abs(step - np.median(step)))),
        "trajectory_grid_step_median_initial_mass": float(np.median(trajectory_step)),
        "carrier_mode_reconstruction_max_abs": carrier_error,
        "carrier_semantics": "cached TEOB state carrier; amplitude*exp(-1j*phase) reconstructs mode",
    }
    return data, info


def _infer_fixed_step(data: dict[str, np.ndarray], override: float | None) -> tuple[float, str]:
    if override is not None:
        if not math.isfinite(override) or override <= 0:
            raise ValueError("--fixed-step must be finite and positive")
        return float(override), "explicit command-line value"
    if "nqc_aux_time" in data:
        differences = np.diff(data["nqc_aux_time"])
        if differences.size and np.all(differences > 0):
            step = float(np.median(differences))
            if np.max(np.abs(differences - step)) <= 1.0e-10 * max(1.0, abs(step)):
                return step, "uniform spacing recovered from cached nqc_aux_time"
    raise ValueError("cannot infer NQC fixed_step; supply --fixed-step")


def _metrics(clean: np.ndarray, public: np.ndarray, output_time: np.ndarray,
             boundary: float) -> dict[str, Any]:
    if clean.shape != public.shape or output_time.shape != clean.shape:
        raise ValueError("clean, public and output_time must share the full public grid")
    masks = {
        "full": np.ones(output_time.shape, dtype=bool),
        "pre_boundary": output_time < boundary,
        "post_boundary": output_time >= boundary,
    }

    def complex_metrics(mask: np.ndarray) -> dict[str, float]:
        error = clean[mask] - public[mask]
        reference = public[mask]
        denominator = float(np.linalg.norm(reference))
        return {
            "samples": int(mask.sum()),
            "l2": float(np.linalg.norm(error)),
            "relative_l2": float(np.linalg.norm(error) / denominator) if denominator else math.nan,
            "rms": float(np.sqrt(np.mean(np.abs(error) ** 2))) if error.size else math.nan,
            "max_abs": float(np.max(np.abs(error))) if error.size else math.nan,
        }

    def amplitude_metrics(mask: np.ndarray) -> dict[str, float]:
        error = np.abs(clean[mask]) - np.abs(public[mask])
        reference = np.abs(public[mask])
        denominator = float(np.linalg.norm(reference))
        return {
            "samples": int(mask.sum()),
            "l2": float(np.linalg.norm(error)),
            "relative_l2": float(np.linalg.norm(error) / denominator) if denominator else math.nan,
            "rms": float(np.sqrt(np.mean(error * error))) if error.size else math.nan,
            "max_abs": float(np.max(np.abs(error))) if error.size else math.nan,
        }

    def phase_metrics(mask: np.ndarray) -> dict[str, float]:
        valid = mask & (np.abs(clean) > 0.0) & (np.abs(public) > 0.0)
        if not np.any(valid):
            return {"samples": 0, "l2_radians": math.nan, "rms_radians": math.nan, "max_abs_radians": math.nan}
        ratio_phase = np.unwrap(np.angle(clean[valid] * np.conj(public[valid])))
        return {
            "samples": int(valid.sum()),
            "l2_radians": float(np.linalg.norm(ratio_phase)),
            "rms_radians": float(np.sqrt(np.mean(ratio_phase * ratio_phase))),
            "max_abs_radians": float(np.max(np.abs(ratio_phase))),
            "first_radians": float(ratio_phase[0]),
            "last_radians": float(ratio_phase[-1]),
        }

    return {
        region: {
            "raw_complex": complex_metrics(mask),
            "amplitude": amplitude_metrics(mask),
            "phase": phase_metrics(mask),
        }
        for region, mask in masks.items()
    }


def _events(result: RingdownAssemblyResult, output_time: np.ndarray, final_mass: torch.Tensor) -> dict[str, Any]:
    events = result.events
    attachment = _float(events.attachment_time)
    output_index = int(np.searchsorted(output_time, attachment, side="left"))
    output_index = min(max(output_index, 0), output_time.size - 1)
    return {
        "peak_knot_index_auxiliary": int(events.peak_knot_index),
        "peak_time_initial_mass": _float(events.peak_time),
        "peak_time_final_mass": _float(events.peak_time / final_mass),
        "merger_time_initial_mass": _float(events.merger_time),
        "merger_time_final_mass": _float(events.merger_time / final_mass),
        "match_time_initial_mass": _float(events.match_time),
        "match_time_final_mass": _float(events.match_time_final_mass),
        "strictly_earlier_index_joined": int(events.strictly_earlier_index),
        "attachment_index_joined": int(events.attachment_index),
        "attachment_time_initial_mass": attachment,
        "attachment_time_final_mass": _float(events.attachment_time_final_mass),
        "output_boundary_index_public_grid": output_index,
        "output_boundary_time_initial_mass": float(output_time[output_index]),
        "joined_samples": int(result.joined_time.numel()),
        "extended_samples": int(result.extended_time.numel()),
        "output_samples": int(result.time.numel()),
        "extension_samples": int(result.extension_samples),
        "auxiliary_spacing_initial_mass": _float(result.auxiliary_spacing),
        "ringdown_phase0": _float(result.ringdown_phase0),
        "auxiliary_amplitude_negative": bool(result.auxiliary_amplitude_negative),
    }


def _nqc_summary(result: Any) -> dict[str, Any]:
    return {
        "target_provenance": result.target_provenance,
        "targets": _vector(torch.stack(result.targets)),
        "amplitude_coefficients": _vector(result.coefficients.amplitude),
        "phase_coefficients": _vector(result.coefficients.phase),
        "matching_index_auxiliary": int(result.matching_index),
        "cut_start_index_original": int(result.cut_start_index),
        "cut_last_index_original": int(result.cut_last_index),
        "t_peak_grid_initial_mass": _float(result.t_peak_grid),
        "t_nqc_initial_mass": _float(result.t_nqc),
        "auxiliary_spacing_initial_mass": _float(result.auxiliary_spacing),
        "auxiliary_samples": int(result.auxiliary.time.numel()),
        "amplitude_condition": _float(result.match.amplitude_condition),
        "phase_condition": _float(result.match.phase_condition),
        "amplitude_residual": _vector(result.match.amplitude_residual),
        "phase_residual": _vector(result.match.phase_residual),
        "negative_auxiliary_corrected_amplitude_samples": int(result.auxiliary_negative_amplitude.sum()),
    }


def _assemble(
    *,
    time: torch.Tensor,
    amplitude: torch.Tensor,
    phase: torch.Tensor,
    nqc: Any,
    corrected: bool,
    nu: torch.Tensor,
    final_mass: torch.Tensor,
    final_spin: torch.Tensor,
) -> RingdownAssemblyResult:
    if corrected:
        full_amplitude = nqc.original.corrected_amplitude
        full_phase = nqc.original.corrected_phase
        auxiliary_amplitude = nqc.auxiliary.corrected_amplitude
        auxiliary_phase = nqc.auxiliary.corrected_phase
    else:
        full_amplitude = amplitude
        full_phase = phase
        auxiliary_amplitude = nqc.auxiliary.amplitude
        auxiliary_phase = nqc.auxiliary.phase
    return assemble_teob_ringdown_22(
        time,
        full_amplitude,
        full_phase,
        nqc.auxiliary.time,
        auxiliary_amplitude,
        auxiliary_phase,
        nqc.auxiliary.orbital_frequency,
        nu=nu,
        final_mass=final_mass,
        final_spin=final_spin,
        matching_time=nqc.t_nqc,
        matching_state=nqc.auxiliary.state[nqc.matching_index],
        output_step=torch.as_tensor(OUTPUT_STEP, dtype=time.dtype, device=time.device),
    )


def _branch_radial_summary(acceleration: torch.Tensor, cached: torch.Tensor,
                           name: str) -> dict[str, Any]:
    difference = acceleration - cached
    return {
        "name": name,
        "samples": int(acceleration.numel()),
        "finite": bool(torch.isfinite(acceleration).all()),
        "range": [_float(acceleration.min()), _float(acceleration.max())],
        "l2": _float(torch.linalg.vector_norm(acceleration)),
        "cached_exact_difference": {
            "l2": _float(torch.linalg.vector_norm(difference)),
            "max_abs": _float(difference.abs().max()),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, action="append", default=None,
        help="paired physical cache; repeat to compare multiple frozen caches "
        "(default: current and lo-horizon)",
    )
    parser.add_argument(
        "--fixed-step", type=float, default=None,
        help="NQC fixed step in initial-mass units (otherwise recover cached 0.1M spacing)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-cache", type=Path, default=DEFAULT_OUTPUT_CACHE)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    return parser.parse_args()


def _inventory() -> list[dict[str, Any]]:
    paths = sorted((ROOT / DEFAULT_INVENTORY_GLOB).parent.glob(Path(DEFAULT_INVENTORY_GLOB).name))
    records = []
    for path in paths:
        try:
            records.append({
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "classification": (
                    "canonical-current" if path.name == "paired-physical-output.npz"
                    else "variant-or-historical-artifact"
                ),
            })
        except OSError as exc:
            records.append({"path": str(path.resolve()), "error": str(exc)})
    return records


def main() -> int:
    args = _parse_args()
    cache_paths = [path.resolve() for path in (args.cache or [DEFAULT_CACHE, DEFAULT_LO_CACHE])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    started = wall_time.perf_counter()
    source_paths = _source_paths()
    source_before = {name: _sha256(path) for name, path in source_paths.items()}
    command = shlex.join([sys.executable, *sys.argv])
    pid = os.getpid()
    report: dict[str, Any] = {
        "status": "running",
        "diagnostic_scope": "cached paired physical TEOB radial-acceleration/NQC/ringdown attribution",
        "run_control": {
            "host": socket.gethostname(),
            "cwd": os.getcwd(),
            "command": command,
            "pid": pid,
            "python": sys.executable,
            "environment": {
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": os.environ.get(
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD"
                ),
            },
            "platform": platform.platform(),
            "device": "cpu",
            "started_unix": wall_time.time(),
            "log_path": str(args.log_path),
            "expected_next_check": "inspect diagnostic.json and paired-radial-acceleration-output.npz after the bounded process exits",
            "stop_command": f"kill {pid}",
        },
        "probe": {
            "nu": NU,
            "delta": DELTA,
            "mass1_msun": MASS1_MSUN,
            "mass2_msun": MASS2_MSUN,
            "chi1": 0.0,
            "chi2": 0.0,
            "lambda1": 0.0,
            "lambda2": 0.0,
            "branches": list(BRANCHES),
            "public_grid_policy": "use cached public_time/public_mode at every sample; no alignment",
            "output_step_initial_mass": OUTPUT_STEP,
            "fixed_step_override": args.fixed_step,
            "radiation_source_sha256_expected": EXPECTED_RADIATION_SHA256,
        },
        "cache_inventory": _inventory(),
        "selected_caches": [str(path) for path in cache_paths],
        "source_sha256_before": source_before,
        "output_path": str(args.output),
        "output_cache_path": str(args.output_cache),
    }
    output_arrays: dict[str, np.ndarray] = {}

    try:
        report["probe"]["radiation_source_sha256_actual"] = source_before["radiation"]
        report["probe"]["radiation_source_sha256_matches_expected"] = (
            source_before["radiation"] == EXPECTED_RADIATION_SHA256
        )
        if source_before["radiation"] != EXPECTED_RADIATION_SHA256:
            raise RuntimeError(
                "paired-cache diagnostic requires the reviewed patched radiation "
                f"source {EXPECTED_RADIATION_SHA256}, found {source_before['radiation']}"
            )
        try:
            import lal  # type: ignore
        except ImportError as exc:  # pragma: no cover - configured environment supplies lal
            raise RuntimeError("the paired public projection requires installed lal constants") from exc
        projection_scale = -math.sqrt(5.0 / (4.0 * math.pi)) * (
            50.0 * lal.MTSUN_SI * lal.C_SI / (100e6 * lal.PC_SI)
        )
        report["probe"]["projection_scale_h22_to_faceon_strain"] = projection_scale
        report["probe"]["projection_scale_formula"] = (
            "-sqrt(5/(4*pi))*(50*lal.MTSUN_SI*lal.C_SI)/(100e6*lal.PC_SI)"
        )

        hamiltonian = _hamiltonian()
        nu = torch.tensor(NU, dtype=torch.float64)
        targets = published_2017_nqc_targets(
            nu, torch.tensor(0.0, dtype=torch.float64),
        )
        final_mass, final_spin = uib2016_nonspinning_remnant(MASS1_MSUN, MASS2_MSUN)
        final_mass = final_mass.to(dtype=torch.float64)
        final_spin = final_spin.to(dtype=torch.float64)
        report["remnant"] = {
            "model": "uib2016_nonspinning_remnant",
            "final_mass_fraction": _float(final_mass),
            "final_spin": _float(final_spin),
        }

        all_cache_reports: list[dict[str, Any]] = []
        for cache_index, cache_path in enumerate(cache_paths):
            data, cache_info = _load_cache(cache_path)
            fixed_step, fixed_step_source = _infer_fixed_step(data, args.fixed_step)
            cache_info["fixed_step_initial_mass"] = fixed_step
            cache_info["fixed_step_source"] = fixed_step_source
            time = _as_tensor(data["time"])
            state = _as_tensor(data["state"])
            amplitude = _as_tensor(data["amplitude"])
            phase = _as_tensor(data["phase"])
            frequency = _as_tensor(data["frequency"])
            public_time = np.asarray(data["public_time"], dtype=np.float64)
            public_mode = np.asarray(data["public_mode"], dtype=np.complex128)

            with torch.no_grad():
                terms = radial_acceleration_approximations(
                    hamiltonian, state[:, 0], state[:, 2], state[:, 3],
                )
                exact = conservative_trajectory_quantities(
                    hamiltonian, state,
                ).radial_acceleration
            cached_exact = _as_tensor(
                data["radial_acceleration"] if "radial_acceleration" in data
                else np.zeros(time.shape, dtype=np.float64),
            )
            branch_accelerations = {
                "exact": exact,
                "frozen_mobility": terms.frozen_mobility,
                "strict_p0": terms.strict_p0,
            }
            cache_report: dict[str, Any] = {
                "cache": cache_info,
                "state_carrier": {
                    "provider": "cached teob_waveform_state_mode / TEOBRadiationClean carrier",
                    "carrier_arrays": ["mode", "amplitude", "phase", "frequency", "state"],
                    "mode_is_regenerated": False,
                    "mode_convention": "h=A*exp(-1j*phase)",
                },
                "radial_branches": {
                    name: _branch_radial_summary(value, cached_exact, name)
                    for name, value in branch_accelerations.items()
                },
                "nqc_target_provenance": PUBLISHED_2017_NQC_PROVENANCE,
                "nqc_targets": _vector(torch.stack(targets)),
                "assemblies": {},
            }

            raw_projected: np.ndarray | None = None
            for branch_name in BRANCHES:
                with torch.no_grad():
                    nqc = assemble_teob_merger_nqc(
                        time,
                        state,
                        amplitude=amplitude,
                        phase=phase,
                        orbital_frequency=frequency,
                        radial_acceleration=branch_accelerations[branch_name],
                        nu=nu,
                        fixed_step=torch.tensor(fixed_step, dtype=torch.float64),
                        targets=targets,
                        target_provenance=PUBLISHED_2017_NQC_PROVENANCE,
                    )
                    raw_assembly = _assemble(
                        time=time,
                        amplitude=amplitude,
                        phase=phase,
                        nqc=nqc,
                        corrected=False,
                        nu=nu,
                        final_mass=final_mass,
                        final_spin=final_spin,
                    )
                    corrected_assembly = _assemble(
                        time=time,
                        amplitude=amplitude,
                        phase=phase,
                        nqc=nqc,
                        corrected=True,
                        nu=nu,
                        final_mass=final_mass,
                        final_spin=final_spin,
                    )

                output_time = _numpy(corrected_assembly.time)
                if output_time.shape != public_time.shape:
                    raise ValueError(
                        f"{cache_path}: assembly output has {output_time.size} samples, "
                        f"public cache has {public_time.size}"
                    )
                grid_error = float(np.max(np.abs(output_time - public_time)))
                if grid_error > 1.0e-10:
                    raise ValueError(f"{cache_path}: assembly/public grid mismatch {grid_error}")
                raw_projected_candidate = projection_scale * _numpy(raw_assembly.mode)
                corrected_projected = projection_scale * _numpy(corrected_assembly.mode)
                if raw_projected is None:
                    raw_projected = raw_projected_candidate
                else:
                    raw_branch_difference = float(np.max(np.abs(raw_projected_candidate - raw_projected)))
                    cache_report.setdefault("raw_branch_max_differences", {})[branch_name] = raw_branch_difference
                boundary = _float(corrected_assembly.events.attachment_time)
                branch_report = {
                    "radial_acceleration": cache_report["radial_branches"][branch_name],
                    "nqc": _nqc_summary(nqc),
                    "assembly": {
                        "events": _events(corrected_assembly, output_time, final_mass),
                        "output_grid_max_abs_difference_initial_mass": grid_error,
                        "raw_pre_nqc_metrics": _metrics(
                            raw_projected_candidate, public_mode, output_time,
                            _float(raw_assembly.events.attachment_time),
                        ),
                        "nqc_corrected_metrics": _metrics(
                            corrected_projected, public_mode, output_time, boundary,
                        ),
                        "mode_units": "dimensionless assembly mode multiplied by exact 50Msun/100Mpc face-on scale",
                    },
                }
                cache_report["assemblies"][branch_name] = branch_report

                cache_slug = cache_path.stem.replace("-", "_")
                prefix = f"cache{cache_index}_{cache_slug}_{branch_name}"
                output_arrays[f"{prefix}_radial_acceleration"] = _numpy(branch_accelerations[branch_name])
                output_arrays[f"{prefix}_nqc_corrected_amplitude"] = _numpy(nqc.corrected_amplitude)
                output_arrays[f"{prefix}_nqc_corrected_phase"] = _numpy(nqc.corrected_phase)
                output_arrays[f"{prefix}_nqc_output_mode"] = corrected_projected
                if branch_name == BRANCHES[0]:
                    output_arrays[f"cache{cache_index}_{cache_slug}_raw_output_mode"] = raw_projected_candidate
                    output_arrays[f"cache{cache_index}_{cache_slug}_output_time"] = output_time
                    output_arrays[f"cache{cache_index}_{cache_slug}_public_mode"] = public_mode
                    output_arrays[f"cache{cache_index}_{cache_slug}_public_time"] = public_time
            assert raw_projected is not None
            cache_report["public_grid"] = {
                "samples": int(public_time.size),
                "support_initial_mass": [float(public_time[0]), float(public_time[-1])],
                "step_initial_mass": float(np.median(np.diff(public_time))),
                "mode_convention": "h_plus - 1j*h_cross",
                "alignment": "none",
                "full_pre_post_regions": "full, output_time < attachment, output_time >= attachment",
            }
            all_cache_reports.append(cache_report)

        report["caches"] = all_cache_reports
        # The output cache is a reusable full-grid hand-off.  All selected
        # branch outputs are prefixed by stable cache index/name keys.
        output_arrays["projection_scale_h22_to_faceon_strain"] = np.asarray(projection_scale)
        np.savez_compressed(args.output_cache, **output_arrays)
        report["output_cache"] = {
            "path": str(args.output_cache),
            "sha256": _sha256(args.output_cache),
            "arrays": sorted(output_arrays),
            "sampling": "complete cached public grid; no alignment",
        }
        report["comparison_conventions"] = {
            "public_mode": "cached h_plus - 1j*h_cross",
            "waveform_mode": "A*exp(-1j*phase), then exact face-on projection",
            "raw_pre_nqc": "same frozen trajectory and uncorrected amplitude/phase through ringdown assembly",
            "nqc_corrected": "same frozen trajectory with branch-specific NQC correction through ringdown assembly",
            "metrics": "same-index complex, absolute-amplitude, and unwrapped phase errors",
            "regions": "full public grid, pre-attachment, post-attachment",
            "alignment": "none",
            "fitting": "none",
        }
        report["limitations"] = [
            "This is a cached-state attribution; no PA/adaptive/fixed evolution was rerun.",
            "The frozen_mobility branch is an explicit compatibility hypothesis and is not selected as a default.",
            "The paired cache carrier and public waveform are consumed as supplied; no radius, phase, or time alignment is applied.",
            "Historical or horizon-variant files are inventoried but are processed only when passed explicitly with --cache.",
        ]
        report["status"] = "complete"
    except Exception as exc:  # pragma: no cover - diagnostic failure is reported
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}

    source_after = {name: _sha256(path) for name, path in source_paths.items()}
    report["source_sha256_after"] = source_after
    report["source_hashes_unchanged"] = source_before == source_after
    selected_after = []
    for path in cache_paths:
        try:
            selected_after.append({"path": str(path), "sha256_after": _sha256(path)})
        except OSError as exc:
            selected_after.append({"path": str(path), "error": str(exc)})
    report["selected_cache_sha256_after"] = selected_after
    report["selected_cache_hashes_unchanged"] = all(
        item.get("sha256_after") == info.get("cache", {}).get("sha256_before")
        for item, info in zip(selected_after, report.get("caches", []))
    ) if len(selected_after) == len(report.get("caches", [])) else False
    report["elapsed_seconds"] = wall_time.perf_counter() - started
    report["runtime"] = {
        "host": socket.gethostname(),
        "cwd": str(Path.cwd()),
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
        "output_cache": str(args.output_cache),
        "elapsed_seconds": report["elapsed_seconds"],
        "source_hashes_unchanged": report["source_hashes_unchanged"],
        "selected_cache_hashes_unchanged": report["selected_cache_hashes_unchanged"],
    }, sort_keys=True))
    return 0 if report["status"] == "complete" and report["source_hashes_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

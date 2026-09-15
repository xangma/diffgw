#!/usr/bin/env python3
"""Compare grid and continuous NQC matching on cached TEOB trajectories.

This diagnostic uses no oracle-derived fitting. It reconstructs derived quantities and
the postmerger from accepted trajectory knots in the convergence cache.  The
public waveform is read only for scores; it never supplies fitted quantities.
The runtime NQC/ringdown APIs provide ``matching_method`` and ``t_peak``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shlex
import socket
import sys
import time as clock
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import diagnose_teob_paired_driver_clean as driver
from torchwave.approximants.eob import teob_merger_nqc_clean as nqc_module
from torchwave.approximants.eob import teob_ringdown_assembly_clean as ringdown_module
from torchwave.approximants.eob import continuous_matching as continuous_matching_module
from torchwave.approximants.eob import numerical_attachment as numerical_attachment_module
from torchwave.approximants.eob import teob_vacuum_driver_clean as vacuum_module
from torchwave.approximants.eob.teob_vacuum_driver_clean import TEOBVacuumRadiationDriver

DEFAULT_CACHE = ROOT / "docs/reviews/2026-09-15-no-contact/teob-ode-convergence-clean/convergence-arrays.npz"
DEFAULT_PRIOR_REPORT = ROOT / "docs/reviews/2026-09-15-no-contact/teob-ode-convergence-clean/report.json"
DEFAULT_OUTPUT = ROOT / "docs/reviews/2026-09-15-continuous-matching/report.json"
LEVELS = ("baseline", "tight_0p1", "tight_0p01")
SPACINGS = (0.05, 0.1, 0.2, 0.5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def npy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def scalar(value: Any) -> float:
    return float(torch.as_tensor(value).detach().cpu().item())


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError(f"metric shape mismatch: {candidate.shape} != {reference.shape}")
    delta = candidate - reference
    return {
        "samples": int(candidate.size),
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-300)),
        "max_abs": float(np.max(np.abs(delta))),
        "rms_abs": float(np.sqrt(np.mean(np.abs(delta) ** 2))),
    }


def _projection_scale() -> float:
    try:
        import lal  # type: ignore
    except ImportError as exc:  # pragma: no cover - reviewed environment
        raise RuntimeError("lal constants are required for the reviewed projection") from exc
    return -math.sqrt(5.0 / (4.0 * math.pi)) * (
        50.0 * lal.MTSUN_SI * lal.C_SI / (driver.DISTANCE_MPC * 1e6 * lal.PC_SI)
    )


def _native_frequency(hamiltonian: Any, radiation: Any, state: torch.Tensor) -> torch.Tensor:
    return TEOBVacuumRadiationDriver(hamiltonian, radiation).vector_field(state)[..., 1]


def _score(result: Any, public_time: np.ndarray, public_mode: np.ndarray,
           projection: float, final_mass: torch.Tensor) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # Score the assembler's native piecewise output.  Interpolating the
    # extended amplitude/phase arrays across the attachment would turn the
    # continuous boundary back into a spacing-dependent sampled boundary.
    native_time, native_mode = npy(result.time), projection * npy(result.mode)
    mask = (public_time >= native_time[0]) & (public_time <= native_time[-1])
    score_time = public_time[mask]
    if score_time.size < 2:
        raise ValueError("assembled support has insufficient overlap with public grid")
    indices = np.searchsorted(native_time, score_time, side="left")
    right = np.minimum(indices, native_time.size - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(
        np.abs(native_time[left] - score_time) <= np.abs(native_time[right] - score_time),
        left, right,
    )
    if not np.all(np.abs(native_time[nearest] - score_time) <= 1e-12):
        raise ValueError("native output grid differs from public grid; direct evaluation is required")
    mode = native_mode[nearest]
    event = driver._events(result, public_time, final_mass)
    event.update({
        "metrics_vs_public": metrics(mode, public_mode[mask]),
        "scored_public_samples": int(score_time.size),
        "full_public_support": bool(mask.all()),
        "scored_public_support_initial_mass": [float(score_time[0]), float(score_time[-1])],
        "signed_extended_negative_samples": int(np.count_nonzero(npy(result.extended_amplitude) < 0.0)),
    })
    return score_time, mode, event


def _common_metrics(left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    """Compare two outputs at exactly shared physical sample coordinates."""
    lt, lm = left
    rt, rm = right
    start, end = max(lt[0], rt[0]), min(lt[-1], rt[-1])
    mesh = lt[(lt >= start) & (lt <= end)]
    if mesh.size < 2:
        raise ValueError("outputs have insufficient common physical support")
    right_index = np.searchsorted(rt, mesh, side="left")
    right_index = np.minimum(right_index, rt.size - 1)
    left_index = np.maximum(right_index - 1, 0)
    nearest = np.where(
        np.abs(rt[left_index] - mesh) <= np.abs(rt[right_index] - mesh),
        left_index, right_index,
    )
    if not np.all(np.abs(rt[nearest] - mesh) <= 1e-12):
        raise ValueError("comparison outputs use genuinely different grids")
    result = metrics(lm[(lt >= start) & (lt <= end)], rm[nearest])
    result.update({
        "common_support_initial_mass": [float(mesh[0]), float(mesh[-1])],
        "common_public_samples": int(mesh.size),
    })
    return result


def _vector_drift(left: Any, right: Any) -> dict[str, Any]:
    return metrics(np.asarray(left, dtype=float).reshape(-1), np.asarray(right, dtype=float).reshape(-1))


def _assemble(nqc: Any, time: torch.Tensor, nu: torch.Tensor,
              final_mass: torch.Tensor, final_spin: torch.Tensor,
              matching_method: str, matching_state: torch.Tensor | None = None) -> Any:
    """Call the new ringdown API, forwarding the continuous peak explicitly."""
    kwargs: dict[str, Any] = {
        "nu": nu,
        "final_mass": final_mass,
        "final_spin": final_spin,
        "matching_time": nqc.t_nqc,
        "matching_state": matching_state,
        "matching_method": matching_method,
        "output_step": torch.as_tensor(driver.OUTPUT_STEP, dtype=time.dtype),
    }
    if matching_method == "continuous":
        kwargs["peak_time"] = nqc.t_peak
    return driver.assemble_teob_ringdown_22(
        time, nqc.original.corrected_amplitude, nqc.original.corrected_phase,
        nqc.auxiliary.time, nqc.auxiliary.corrected_amplitude,
        nqc.auxiliary.corrected_phase, nqc.auxiliary.orbital_frequency,
        **kwargs,
    )


def _run_case(level: str, data: dict[str, np.ndarray], public_time: np.ndarray,
              public_mode: np.ndarray, projection: float, spacing: float,
              matching_method: str) -> dict[str, Any]:
    start_clock = clock.perf_counter()
    dtype = torch.float64
    time_t = torch.as_tensor(data[f"{level}_trajectory_time"], dtype=dtype)
    state = torch.as_tensor(data[f"{level}_trajectory_state"].copy(), dtype=dtype)
    hamiltonian = driver._hamiltonian()
    radiation = driver._radiation_reaction(hamiltonian, "current")
    with torch.no_grad():
        frequency = _native_frequency(hamiltonian, radiation, state)
        mode = driver.teob_waveform_state_mode(
            state, hamiltonian, 2, 2, Omega=frequency, residual_phase="zero-unpublished",
        )
        amplitude, phase = mode.abs(), driver.phase_from_mode(mode, state[:, 1])
        acceleration = driver.conservative_trajectory_quantities(
            hamiltonian, state,
        ).radial_acceleration
        nu = torch.tensor(driver.NU_VALUE, dtype=dtype)
        targets = driver.published_2017_nqc_targets(nu, torch.tensor(0.0, dtype=dtype))
        nqc = driver.assemble_teob_merger_nqc(
            time_t, state, amplitude=amplitude, phase=phase,
            orbital_frequency=frequency, radial_acceleration=acceleration,
            nu=nu, fixed_step=torch.tensor(spacing, dtype=dtype), targets=targets,
            target_provenance=driver.PUBLISHED_2017_NQC_PROVENANCE,
            matching_method=matching_method,
        )
        final_mass, final_spin = driver.uib2016_nonspinning_remnant(
            driver.MASS1_MSUN, driver.MASS2_MSUN,
        )
        final_mass, final_spin = final_mass.to(dtype=dtype), final_spin.to(dtype=dtype)
        # Metadata must describe the physical full trajectory.  In continuous
        # mode evaluate the state spline at t_nqc; grid mode deliberately omits
        # matching_state because the ringdown assembler does not need it.
        if matching_method == "continuous":
            state_spline = numerical_attachment_module.natural_cubic_spline(
                time_t, state, scalar_output=False,
            )
            matching_state = state_spline.evaluate(nqc.t_nqc)
        else:
            matching_state = None
        result = _assemble(
            nqc, time_t, nu, final_mass, final_spin, matching_method, matching_state,
        )
    score_time, output, event = _score(result, public_time, public_mode, projection, final_mass)
    amplitude_coeff = npy(nqc.coefficients.amplitude)
    phase_coeff = npy(nqc.coefficients.phase)
    peak = scalar(nqc.t_peak)
    return {
        "key": f"{level}:{matching_method}:{spacing:g}",
        "level": level,
        "matching_method": matching_method,
        "spacing_M": spacing,
        "trajectory_samples": int(time_t.numel()),
        "score_time": score_time,
        "output": output,
        "nqc": {
            "matching_index_auxiliary": int(nqc.matching_index),
            "matching_time_M": scalar(nqc.t_nqc),
            "t_peak_M": peak,
            "t_peak_grid_M": scalar(nqc.t_peak_grid),
            "auxiliary_samples": int(nqc.auxiliary.time.numel()),
            "auxiliary_spacing_M": scalar(nqc.auxiliary_spacing),
            "amplitude_coefficients": amplitude_coeff.tolist(),
            "phase_coefficients": phase_coeff.tolist(),
            "amplitude_condition": scalar(nqc.match.amplitude_condition),
            "phase_condition": scalar(nqc.match.phase_condition),
            "amplitude_residual": npy(nqc.match.amplitude_residual).tolist(),
            "phase_residual": npy(nqc.match.phase_residual).tolist(),
        },
        "events": event,
        "elapsed_seconds": clock.perf_counter() - start_clock,
    }


def _public_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": case["key"],
        "level": case["level"],
        "matching_method": case["matching_method"],
        "spacing_M": case["spacing_M"],
        "trajectory_samples": case["trajectory_samples"],
        "elapsed_seconds": case["elapsed_seconds"],
        "nqc": case["nqc"],
        "events": case["events"],
    }


def _source_paths(cache_path: Path) -> list[Path]:
    return [
        cache_path, Path(__file__), Path(driver.__file__), Path(nqc_module.__file__),
        Path(ringdown_module.__file__), Path(vacuum_module.__file__),
        Path(continuous_matching_module.__file__), Path(numerical_attachment_module.__file__),
    ]


def _source_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256(path) for path in paths if path.exists()}


def run(cache_path: Path = DEFAULT_CACHE, prior_report: Path = DEFAULT_PRIOR_REPORT,
        output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    started = clock.perf_counter()
    source_paths = _source_paths(cache_path)
    source_before = _source_hashes(source_paths)
    data = dict(np.load(cache_path, allow_pickle=False))
    public_time, public_mode = data["public_time"], data["public_mode"]
    projection = _projection_scale()
    cases: dict[str, dict[str, Any]] = {}
    for level in LEVELS:
        for method, spacing in (("grid", 0.1), ("continuous", 0.1)):
            case = _run_case(level, data, public_time, public_mode, projection, spacing, method)
            cases[case["key"]] = case
    spacing_cases: dict[str, dict[str, Any]] = {}
    for level in LEVELS:
        for spacing in SPACINGS:
            case = _run_case(level, data, public_time, public_mode, projection, spacing, "continuous")
            spacing_cases[case["key"]] = case

    comparisons = []
    for level in LEVELS:
        grid = cases[f"{level}:grid:0.1"]
        cont = cases[f"{level}:continuous:0.1"]
        comparisons.append({
            "level": level,
            "grid_vs_continuous_output": _common_metrics(
                (grid["score_time"], grid["output"]), (cont["score_time"], cont["output"]),
            ),
            "grid_public_score": grid["events"]["metrics_vs_public"],
            "continuous_public_score": cont["events"]["metrics_vs_public"],
            "event_drift": {
                name: float(cont["events"][name] - grid["events"][name])
                for name in ("peak_time_initial_mass", "merger_time_initial_mass",
                             "t_match_initial_mass", "attachment_time_initial_mass")
            },
            "coefficient_drift": {
                "amplitude": _vector_drift(cont["nqc"]["amplitude_coefficients"], grid["nqc"]["amplitude_coefficients"]),
                "phase": _vector_drift(cont["nqc"]["phase_coefficients"], grid["nqc"]["phase_coefficients"]),
            },
            "peak_drift_M": float(cont["nqc"]["t_peak_M"] - grid["nqc"]["t_peak_M"]),
        })

    spacing_stability = []
    for level in LEVELS:
        reference = spacing_cases[f"{level}:continuous:0.1"]
        for spacing in SPACINGS:
            case = spacing_cases[f"{level}:continuous:{spacing:g}"]
            spacing_stability.append({
                "level": level,
                "spacing_M": spacing,
                "relative_to_spacing_0.1_output": _common_metrics(
                    (case["score_time"], case["output"]),
                    (reference["score_time"], reference["output"]),
                ),
                "public_score": case["events"]["metrics_vs_public"],
                "coefficient_drift": {
                    "amplitude": _vector_drift(case["nqc"]["amplitude_coefficients"], reference["nqc"]["amplitude_coefficients"]),
                    "phase": _vector_drift(case["nqc"]["phase_coefficients"], reference["nqc"]["phase_coefficients"]),
                },
                "peak_drift_M": float(case["nqc"]["t_peak_M"] - reference["nqc"]["t_peak_M"]),
            })

    level_convergence = []
    for method in ("grid", "continuous"):
        left = cases[f"tight_0p1:{method}:0.1"]
        right = cases[f"tight_0p01:{method}:0.1"]
        level_convergence.append({
            "matching_method": method,
            "coarse_level": "tight_0p1",
            "fine_level": "tight_0p01",
            "output": _common_metrics(
                (left["score_time"], left["output"]),
                (right["score_time"], right["output"]),
            ),
            "coarse_public_score": left["events"]["metrics_vs_public"],
            "fine_public_score": right["events"]["metrics_vs_public"],
            "event_drift_fine_minus_coarse": {
                name: float(right["events"][name] - left["events"][name])
                for name in ("peak_time_initial_mass", "merger_time_initial_mass",
                             "t_match_initial_mass", "attachment_time_initial_mass")
            },
            "coefficient_drift": {
                "amplitude": _vector_drift(right["nqc"]["amplitude_coefficients"], left["nqc"]["amplitude_coefficients"]),
                "phase": _vector_drift(right["nqc"]["phase_coefficients"], left["nqc"]["phase_coefficients"]),
            },
        })

    baseline = cases["baseline:grid:0.1"]
    baseline_saved = data["baseline_processed_mode"]
    if baseline["output"].shape != baseline_saved.shape:
        raise ValueError("baseline native output and saved processed mode have different shapes")
    baseline_reproduction = {
        "source": "baseline_processed_mode in convergence-arrays.npz",
        "grid_case": baseline["key"],
        "metrics": metrics(baseline["output"], baseline_saved),
        "sample_count": int(baseline_saved.size),
    }

    source_after = _source_hashes(source_paths)
    changed_sources = sorted(
        path for path in set(source_before) | set(source_after)
        if source_before.get(path) != source_after.get(path)
    )
    if prior_report.exists():
        prior_hash = sha256(prior_report)
    else:
        prior_hash = None
    report = {
        "schema": "teob-continuous-matching-v1",
        "purpose": "cached-trajectory grid versus continuous NQC matching diagnostic",
        "public_input_role": "public waveform samples supply scores only; native equations and existing published fits supply reconstruction quantities",
        "levels": list(LEVELS),
        "continuous_spacing_M": list(SPACINGS),
        "generated_utc": clock.strftime("%Y-%m-%dT%H:%M:%SZ", clock.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "command": shlex.join(sys.argv),
        "python": sys.executable,
        "cache": {"path": str(cache_path), "sha256": sha256(cache_path)},
        "prior_report": {"path": str(prior_report), "sha256": prior_hash},
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "source_sha256": source_after,
        "source_changed_during_run": changed_sources,
        "elapsed_seconds": clock.perf_counter() - started,
        "cases": [_public_case(case) for case in [*cases.values(), *spacing_cases.values()]],
        "grid_vs_continuous": comparisons,
        "tight_level_convergence": level_convergence,
        "baseline_saved_processed_reproduction": baseline_reproduction,
        "continuous_spacing_stability": spacing_stability,
        "claims": [
            "Diagnostic comparison only; no promotion or model-selection claim.",
            "Spacing comparisons are on their common physical support; tail duration varies with fixed extension sample count.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if changed_sources:
        raise RuntimeError("source changed during run: " + ", ".join(changed_sources))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--prior-report", type=Path, default=DEFAULT_PRIOR_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run(args.cache, args.prior_report, args.output)
    print(json.dumps({"output": str(args.output), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run one bounded TEOB post-PA ODE tolerance convergence experiment.

The PA result, physical inputs, radiation prescription, NQC targets, fixed
step cap, and event/assembly conventions are held fixed.  Only the adaptive
ODE ``atol`` and ``rtol`` are changed at three predeclared levels.  Each raw
trajectory mode and each processed NQC assembly is independently sampled on
the cached public 0.5M grid; no oracle, alignment, fit, or runtime edit is
used.
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

# A direct ``python scripts/...`` launch places ``scripts/`` on sys.path but
# not the repository root; make the clean diagnostic module import explicit.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import diagnose_teob_paired_driver_clean as driver

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs/reviews/2026-09-13-parity-review"
PUBLIC = driver.DEFAULT_PUBLIC
PUBLIC_META = driver.DEFAULT_PUBLIC_META
OUT = ROOT / "docs/reviews/2026-09-15-no-contact/teob-ode-convergence-clean"

# Chosen before comparison.  The fixed-step and all non-tolerance limits are
# deliberately identical across levels.
LEVELS: dict[str, tuple[float, float]] = {
    "baseline": (driver.DEFAULT_ATOL, driver.DEFAULT_RTOL),
    "tight_0p1": (driver.DEFAULT_ATOL * 0.1, driver.DEFAULT_RTOL * 0.1),
    "tight_0p01": (driver.DEFAULT_ATOL * 0.01, driver.DEFAULT_RTOL * 0.01),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def tensor_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def vector(value: Any) -> list[float]:
    return [float(x) for x in tensor_array(value).reshape(-1)]


def array_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise ValueError(f"comparison shape mismatch: {candidate.shape} != {reference.shape}")
    delta = candidate - reference
    denominator = float(np.linalg.norm(reference))
    return {
        "samples": int(candidate.size),
        "relative_l2": finite(np.linalg.norm(delta) / max(denominator, 1e-300)),
        "max_abs": finite(np.max(np.abs(delta))) if delta.size else None,
        "rms_abs": finite(np.sqrt(np.mean(np.abs(delta) ** 2))) if delta.size else None,
    }


def state_difference(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        return {"shape_candidate": list(candidate.shape), "shape_reference": list(reference.shape)}
    delta = candidate - reference
    scale = np.maximum(np.abs(reference), 1e-300)
    return {
        "shape": list(candidate.shape),
        "max_abs_by_component": np.max(np.abs(delta), axis=0).tolist(),
        "max_relative_by_component": np.max(np.abs(delta) / scale, axis=0).tolist(),
        "endpoint_abs_by_component": np.abs(delta[-1]).tolist(),
    }


def source_paths() -> dict[str, Path]:
    # These are the clean source/contract inputs used by the wrapper.  The
    # imported paired driver supplies only documented clean helpers.
    return {
        "diagnostic": Path(__file__).resolve(),
        "paired_driver": Path(driver.__file__).resolve(),
        "adaptive_continuation": ROOT / "src/torchwave/approximants/eob/teob_adaptive_continuation_clean.py",
        "adaptive_driver": ROOT / "src/torchwave/approximants/eob/adaptive_driver_clean.py",
        "vacuum_driver": ROOT / "src/torchwave/approximants/eob/teob_vacuum_driver_clean.py",
        "postadiabatic": ROOT / "src/torchwave/approximants/eob/teob_postadiabatic_clean.py",
        "waveform_state": ROOT / "src/torchwave/approximants/eob/teob_waveform_state_clean.py",
        "nqc": ROOT / "src/torchwave/approximants/eob/teob_merger_nqc_clean.py",
        "ringdown": ROOT / "src/torchwave/approximants/eob/teob_ringdown_assembly_clean.py",
        "contract": REVIEW / "teob-merger-assembly-contract.md",
    }


def run_level(
    label: str,
    atol: float,
    rtol: float,
    *,
    hamiltonian: Any,
    radiation_reaction: Any,
    pa: Any,
    public_mode: np.ndarray,
    public_time: np.ndarray,
    projection_scale: float,
) -> dict[str, Any]:
    continuation = driver.continue_teob_adaptive(
        hamiltonian,
        pa,
        radiation_reaction,
        include_horizon_absorption=True,
        fixed_step_cap=driver.DEFAULT_FIXED_STEP_CAP,
        continuation_duration=driver.DEFAULT_CONTINUATION_DURATION,
        atol=atol,
        rtol=rtol,
        max_rejected_steps=driver.DEFAULT_MAX_REJECTED_STEPS,
        max_evaluations=driver.DEFAULT_MAX_EVALUATIONS,
        max_accepted_steps=driver.DEFAULT_MAX_ACCEPTED_STEPS,
        max_duration=driver.DEFAULT_MAX_DURATION,
    )
    if not continuation.complete:
        raise RuntimeError(f"{label} continuation did not complete: {continuation.skip_reason}")
    trajectory_time = continuation.time
    state = continuation.state
    frequency = continuation.frequency
    with torch.no_grad():
        mode = driver.teob_waveform_state_mode(
            state, hamiltonian, 2, 2, Omega=frequency, residual_phase="zero-unpublished",
        )
        amplitude = mode.abs()
        phase = driver.phase_from_mode(mode, state[:, 1])
        radial_acceleration = driver.conservative_trajectory_quantities(
            hamiltonian, state,
        ).radial_acceleration
    nu = torch.tensor(driver.NU_VALUE, dtype=torch.float64)
    targets = driver.published_2017_nqc_targets(nu, torch.tensor(0.0, dtype=torch.float64))
    nqc = driver.assemble_teob_merger_nqc(
        trajectory_time, state, amplitude=amplitude, phase=phase,
        orbital_frequency=frequency, radial_acceleration=radial_acceleration,
        nu=nu, fixed_step=torch.tensor(driver.DEFAULT_FIXED_STEP_CAP, dtype=torch.float64),
        targets=targets, target_provenance=driver.PUBLISHED_2017_NQC_PROVENANCE,
    )
    final_mass, final_spin = driver.uib2016_nonspinning_remnant(
        driver.MASS1_MSUN, driver.MASS2_MSUN,
    )
    final_mass, final_spin = final_mass.to(dtype=torch.float64), final_spin.to(dtype=torch.float64)
    matching_state = nqc.auxiliary.state[nqc.matching_index]
    nqc_assembly = driver._assemble(
        time=trajectory_time,
        amplitude=nqc.original.corrected_amplitude,
        phase=nqc.original.corrected_phase,
        auxiliary_time=nqc.auxiliary.time,
        auxiliary_amplitude=nqc.auxiliary.corrected_amplitude,
        auxiliary_phase=nqc.auxiliary.corrected_phase,
        auxiliary_frequency=nqc.auxiliary.orbital_frequency,
        nu=nu, final_mass=final_mass, final_spin=final_spin,
        matching_time=nqc.t_nqc, matching_state=matching_state,
    )
    processed_report, processed_mode, _, _ = driver._public_grid_branch(
        nqc_assembly, public_mode, public_time, projection_scale, final_mass,
    )
    trajectory_report, trajectory_grid, trajectory_mode, _, _ = driver._trajectory_prefix(
        time=trajectory_time, amplitude=amplitude, phase=phase,
        public_mode=public_mode, public_time=public_time,
        projection_scale=projection_scale, boundary=float(nqc_assembly.events.attachment_time),
    )
    events = driver._events(nqc_assembly, public_time, final_mass)
    stats = continuation.statistics
    return {
        "label": label,
        "tolerances": {"atol": atol, "rtol": rtol},
        "continuation": {
            "status": continuation.status,
            "samples": int(trajectory_time.numel()),
            "adaptive_samples": int(continuation.adaptive_time.numel()),
            "fixed_samples": int(continuation.fixed_time.numel()),
            "support_initial_mass": [finite(trajectory_time[0]), finite(trajectory_time[-1])],
            "peak_event_index": continuation.peak_event_index,
            "peak_event_time_initial_mass": finite(continuation.peak_event_time),
            "peak_event_frequency": finite(continuation.peak_event_frequency),
            "endpoint_state": vector(continuation.endpoint_state),
            "endpoint_time_initial_mass": finite(continuation.endpoint_time),
            "first_ode_step": finite(continuation.first_ode_step),
            "adaptive_next_step": finite(continuation.adaptive_next_step),
            "fixed_step": finite(continuation.fixed_step),
            "phase_seam_error": finite(continuation.phase_continuity),
            "statistics": {
                "adaptive_accepted_steps": stats.adaptive_accepted_steps,
                "adaptive_rejected_steps": stats.adaptive_rejected_steps,
                "adaptive_evaluations": stats.adaptive_evaluations,
                "fixed_accepted_steps": stats.fixed_accepted_steps,
                "fixed_rejected_steps": stats.fixed_rejected_steps,
                "fixed_evaluations": stats.fixed_evaluations,
                "termination": stats.termination,
            },
        },
        "events": events,
        "trajectory_pre_nqc": {
            "native_support_initial_mass": [finite(trajectory_time[0]), finite(trajectory_time[-1])],
            "public_grid_samples": int(trajectory_grid.size),
            "public_grid_support_initial_mass": [finite(trajectory_grid[0]), finite(trajectory_grid[-1])],
            "metrics_vs_public": trajectory_report["metrics_vs_public_prefix"],
        },
        "processed_nqc": {
            "metrics_vs_public": processed_report["independent_cubic_public_grid"]["metrics_vs_public"],
            "output_samples": int(nqc_assembly.time.numel()),
            "output_support_initial_mass": [finite(nqc_assembly.time[0]), finite(nqc_assembly.time[-1])],
            "output_matches_public_grid": bool(np.array_equal(tensor_array(nqc_assembly.time), public_time)),
            "nqc_matching_index_auxiliary": int(nqc.matching_index),
            "nqc_auxiliary_samples": int(nqc.auxiliary.time.numel()),
        },
        "arrays": {
            "trajectory_time": tensor_array(trajectory_time),
            "trajectory_state": tensor_array(state),
            "public_pre_nqc_mode": trajectory_mode,
            "public_processed_mode": processed_mode,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pid = os.getpid()
    sources = source_paths()
    public_mode, public_time, public_meta = driver._load_public(PUBLIC, PUBLIC_META)
    try:
        import lal  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lal constants are required for the reviewed projection") from exc
    projection_scale = -math.sqrt(5.0 / (4.0 * math.pi)) * (
        50.0 * lal.MTSUN_SI * lal.C_SI / (driver.DISTANCE_MPC * 1e6 * lal.PC_SI)
    )
    torch.set_default_dtype(torch.float64)
    hamiltonian = driver._hamiltonian()
    radiation_reaction = driver._radiation_reaction(hamiltonian, "current")
    pa = driver.alternating8pa(
        hamiltonian, radiation_reaction, r0=driver.PUBLIC_RADIUS,
        r_min=driver.R_MIN, dr=driver.DR, pa_order=driver.PA_ORDER,
        include_horizon_absorption=True,
    )
    if not pa.complete:
        raise RuntimeError(f"PA profile did not complete: {pa.skip_reason}")
    pa_endpoint = tensor_array(pa.endpoint_state).copy()
    pa_time = tensor_array(pa.time).copy()
    levels: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {"public_time": public_time, "public_mode": public_mode}
    for label, (atol, rtol) in LEVELS.items():
        result = run_level(
            label, atol, rtol, hamiltonian=hamiltonian,
            radiation_reaction=radiation_reaction, pa=pa,
            public_mode=public_mode, public_time=public_time,
            projection_scale=projection_scale,
        )
        arrays[f"{label}_trajectory_time"] = result["arrays"]["trajectory_time"]
        arrays[f"{label}_trajectory_state"] = result["arrays"]["trajectory_state"]
        arrays[f"{label}_pre_nqc_mode"] = result["arrays"]["public_pre_nqc_mode"]
        arrays[f"{label}_processed_mode"] = result["arrays"]["public_processed_mode"]
        del result["arrays"]
        levels[label] = result

    baseline = levels["baseline"]
    baseline_state = arrays["baseline_trajectory_state"]
    baseline_pre = arrays["baseline_pre_nqc_mode"]
    baseline_processed = arrays["baseline_processed_mode"]
    for label, result in levels.items():
        result["difference_from_baseline"] = {
            "trajectory_state": state_difference(arrays[f"{label}_trajectory_state"], baseline_state),
            "pre_nqc_mode_on_public_grid": array_metrics(arrays[f"{label}_pre_nqc_mode"], baseline_pre),
            "processed_mode_on_public_grid": array_metrics(arrays[f"{label}_processed_mode"], baseline_processed),
            "event_time_initial_mass": finite(float(result["events"]["attachment_time_initial_mass"]) - float(baseline["events"]["attachment_time_initial_mass"])),
        }

    np.savez_compressed(args.output_dir / "convergence-arrays.npz", **arrays)
    report: dict[str, Any] = {
        "status": "complete",
        "experiment": "post-PA ODE tolerance convergence with fixed PA handoff",
        "run_control": {
            "host": socket.gethostname(), "cwd": os.getcwd(),
            "command": shlex.join([sys.executable, *sys.argv]), "pid": pid,
            "python": sys.executable, "platform": platform.platform(),
            "log_path": str(args.output_dir / "convergence.log"),
            "expected_next_check": "inspect report.json and convergence-arrays.npz",
            "stop_command": f"kill {pid}",
            "elapsed_seconds": finite(time.perf_counter() - started),
        },
        "predeclared_levels": {label: {"atol": a, "rtol": r} for label, (a, r) in LEVELS.items()},
        "fixed_controls": {
            "pa_profile_reused": True,
            "pa_samples": int(pa.time.numel()),
            "pa_endpoint_time_initial_mass": finite(pa.time[-1]),
            "pa_endpoint_state": pa_endpoint.tolist(),
            "pa_time_hash": hashlib.sha256(pa_time.tobytes()).hexdigest(),
            "pa_endpoint_state_hash": hashlib.sha256(pa_endpoint.tobytes()).hexdigest(),
            "horizon_policy": "current",
            "fixed_step_cap": driver.DEFAULT_FIXED_STEP_CAP,
            "continuation_duration": driver.DEFAULT_CONTINUATION_DURATION,
            "nqc_targets": "published_2017_nqc_targets reused unchanged",
            "event_and_assembly_conventions": "clean driver defaults unchanged",
            "public_grid": public_meta,
        },
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
        "levels": levels,
        "classification": {
            "criterion": "solver convergence would make tightened levels approach baseline in trajectory, pre-NQC mode, processed mode, and event grid; persistent differences with stable events indicate model/event mismatch",
            "result": "Tolerance sensitivity is reported per level; no runtime default is promoted. The experiment is diagnostic only.",
            "limits": [
                "All levels use the cached public 0.5M grid for waveform comparison; native accepted knots remain adaptive.",
                "The public binary has no solver-tolerance control, so convergence is tested only in the clean continuation.",
                "No source outside clean helpers, fitting, alignment, coefficient recovery, or shared status files was used.",
            ],
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output_dir / "report.json"),
                      "levels": {k: v["continuation"]["samples"] for k, v in levels.items()}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

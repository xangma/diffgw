#!/usr/bin/env python3
"""Re-run the clean adaptive SEOB 30+20, 20 Hz controller comparison.

This is an attribution diagnostic for the reviewed state-scaled RKF45
driver.  It preserves the supplied public and earlier native NPZ files and
does not alter a canonical waveform route.  The native result is serialized
under the report directory before raw and amplitude metrics are calculated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch

from torchwave.approximants.eob.adaptive_driver_clean import (
    reconstruct_seobnrv4_22_adaptive,
)
from torchwave.constants import C_SI, MPC_SI, MSUN_SEC


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / (
    "docs/reviews/2026-09-13-parity-review/verification/"
    "eob-adaptive-full20-clean"
)
PUBLIC_PATH = Path("/tmp/torchwave-public-eob-dt_0p5M.npz")
OLD_NATIVE_PATH = Path("/tmp/torchwave-native-eob-full20-controller.npz")
REVIEWED_CONTROLLER_REPORT = ROOT / (
    "docs/reviews/2026-09-13-parity-review/verification/"
    "eob-adaptive-root-review/full20-controller.json"
)

# These are the clean runtime modules used by the adaptive reconstruction.
# Hashing them before and after the run records whether a concurrent edit made
# the attribution non-reproducible; no source is imported from a third-party
# implementation.
RUNTIME_SOURCES = (
    "src/torchwave/approximants/eob/adaptive_driver_clean.py",
    "src/torchwave/approximants/eob/numerical_attachment.py",
    "src/torchwave/approximants/eob/published_waveform.py",
    "src/torchwave/approximants/eob/integrator.py",
    "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
    "src/torchwave/approximants/eob/seobnrv4_flux.py",
    "src/torchwave/approximants/eob/seobnrv4_mode22.py",
    "src/torchwave/approximants/eob/seobnrv4_amplitude22.py",
    "src/torchwave/approximants/eob/initial_conditions.py",
    "src/torchwave/approximants/eob/nqc.py",
    "src/torchwave/approximants/eob/remnant.py",
    "src/torchwave/approximants/eob/events.py",
    "src/torchwave/approximants/kerr_qnm_table_convention.py",
    "src/torchwave/constants.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hashes(paths) -> dict[str, str]:
    return {str(path): sha256(path) for path in paths if path.exists()}


def json_value(value):
    """Convert scalar tensors/NumPy values to JSON-safe values."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return json_value(value.item())
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, (float, int, bool, str)) or value is None:
        return value
    return str(value)


def scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def tensor_array(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def trajectory_counter_report(run) -> dict:
    stats = run.statistics
    valid = tensor_array(run.valid).astype(bool, copy=False)
    output_valid = tensor_array(run.output_valid).astype(bool, copy=False)
    return {
        "accepted_steps": json_value(stats.accepted_steps),
        "rejected_steps": json_value(stats.rejected_steps),
        "evaluations": json_value(stats.evaluations),
        "termination": list(stats.termination),
        "event_time_M": json_value(stats.event_time),
        "accepted_knot_count": int(valid.sum()),
        "accepted_storage_samples": int(run.time.shape[-1]),
        "output_valid_count": int(output_valid.sum()),
        "output_storage_samples": int(run.output_time.shape[-1]),
        "first_valid_time_M": float(tensor_array(run.time)[np.flatnonzero(valid)[0]])
        if valid.any() else None,
        "last_valid_time_M": float(tensor_array(run.time)[np.flatnonzero(valid)[-1]])
        if valid.any() else None,
        "output_step_M": json_value(run.output_step),
        "initial_step_M": json_value(run.initial_step),
        "next_step_M": json_value(run.next_step),
    }


def support_report(mode: np.ndarray, dt: float) -> dict:
    mask = np.abs(mode) > 0.0
    indices = np.flatnonzero(mask)
    return {
        "exact_nonzero_count": int(indices.size),
        "first_nonzero_index": int(indices[0]) if indices.size else None,
        "last_nonzero_index": int(indices[-1]) if indices.size else None,
        "first_nonzero_time_M": float(indices[0] * dt) if indices.size else None,
        "last_nonzero_time_M": float(indices[-1] * dt) if indices.size else None,
        "array_samples": int(mode.size),
        "exact_zero_suffix": bool(indices.size == 0 or not mask[indices[-1] + 1:].any()),
    }


def padded_metrics(native: np.ndarray, reference: np.ndarray) -> dict:
    size = max(native.size, reference.size)
    native_pad = np.pad(native, (0, size - native.size))
    reference_pad = np.pad(reference, (0, size - reference.size))
    raw_delta = native_pad - reference_pad
    amp_delta = np.abs(native_pad) - np.abs(reference_pad)
    ref_raw_norm = float(np.linalg.norm(reference_pad))
    ref_amp_norm = float(np.linalg.norm(np.abs(reference_pad)))
    return {
        "padded_samples": int(size),
        "native_samples": int(native.size),
        "reference_samples": int(reference.size),
        "raw_full_padded_l2": float(np.linalg.norm(raw_delta)),
        "raw_full_padded_relative_l2": float(
            np.linalg.norm(raw_delta) / ref_raw_norm
        ),
        "amplitude_full_padded_l2": float(np.linalg.norm(amp_delta)),
        "amplitude_full_padded_relative_l2": float(
            np.linalg.norm(amp_delta) / ref_amp_norm
        ),
        "raw_max_abs": float(np.max(np.abs(raw_delta))),
        "amplitude_max_abs": float(np.max(np.abs(amp_delta))),
        "native_norm": float(np.linalg.norm(native_pad)),
        "reference_norm": ref_raw_norm,
        "native_amplitude_norm": float(np.linalg.norm(np.abs(native_pad))),
        "reference_amplitude_norm": ref_amp_norm,
    }


def direct_metrics(current: np.ndarray, baseline: np.ndarray) -> dict:
    """Compare equal-shape rerun arrays without resampling or alignment."""
    if current.shape != baseline.shape:
        raise ValueError(
            f"direct comparison requires equal shapes, got {current.shape} and {baseline.shape}"
        )
    delta = current - baseline
    baseline_norm = float(np.linalg.norm(baseline))
    return {
        "shape": list(current.shape),
        "l2": float(np.linalg.norm(delta)),
        "relative_l2": float(np.linalg.norm(delta) / baseline_norm)
        if baseline_norm else (0.0 if not np.linalg.norm(delta) else float("inf")),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def save_cache(path: Path, result) -> str:
    trajectory = result.trajectory
    coarse = trajectory.coarse
    fine = trajectory.fine
    payload = {
        "time": tensor_array(result.time),
        # ``mode`` remains in the clean geometric convention.  The public
        # frozen mode is the same unprojected mode at 100 Mpc; the comparison
        # adds that documented SI conversion explicitly below.
        "mode": tensor_array(result.mode),
        "trajectory_time": tensor_array(trajectory.time),
        "trajectory_state": tensor_array(trajectory.state),
        "trajectory_frequency": tensor_array(trajectory.frequency),
        "trajectory_valid": tensor_array(trajectory.valid),
        "coarse_time": tensor_array(coarse.time),
        "coarse_state": tensor_array(coarse.state),
        "coarse_derivative": tensor_array(coarse.derivative),
        "coarse_valid": tensor_array(coarse.valid),
        "coarse_output_time": tensor_array(coarse.output_time),
        "coarse_output_state": tensor_array(coarse.output_state),
        "coarse_output_valid": tensor_array(coarse.output_valid),
        "fine_time": tensor_array(fine.time),
        "fine_state": tensor_array(fine.state),
        "fine_derivative": tensor_array(fine.derivative),
        "fine_valid": tensor_array(fine.valid),
        "fine_output_time": tensor_array(fine.output_time),
        "fine_output_state": tensor_array(fine.output_state),
        "fine_output_valid": tensor_array(fine.output_valid),
        "coarse_accepted_steps": tensor_array(coarse.statistics.accepted_steps),
        "coarse_rejected_steps": tensor_array(coarse.statistics.rejected_steps),
        "coarse_evaluations": tensor_array(coarse.statistics.evaluations),
        "coarse_event_time": tensor_array(coarse.statistics.event_time),
        "fine_accepted_steps": tensor_array(fine.statistics.accepted_steps),
        "fine_rejected_steps": tensor_array(fine.statistics.rejected_steps),
        "fine_evaluations": tensor_array(fine.statistics.evaluations),
        "fine_event_time": tensor_array(fine.statistics.event_time),
        "peak_time": tensor_array(trajectory.peak.time),
        "peak_frequency": tensor_array(trajectory.peak.frequency),
        "match_time": tensor_array(result.attachment.match_time),
        "boundary_time": tensor_array(result.attachment.boundary_time),
        "boundary_index": tensor_array(result.attachment.boundary_index),
        "final_mass_fraction": tensor_array(result.final_mass_fraction),
        "final_spin": tensor_array(result.final_spin),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    np.savez(temporary, **payload)
    # np.savez appends .npz when given a path without that suffix.  The
    # temporary name already ends in .tmp, so use the generated filename.
    generated = Path(str(temporary) + ".npz")
    generated.replace(path)
    return sha256(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report_dir = args.output_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    mass1 = 30.0
    mass2 = 20.0
    f_lower = 20.0
    dt_geom = 0.5
    omega_start = math.pi * (mass1 + mass2) * float(MSUN_SEC) * f_lower
    parameters = {
        "mass1": mass1,
        "mass2": mass2,
        "chi1": 0.0,
        "chi2": 0.0,
        "f_lower_Hz": f_lower,
        "omega_start": omega_start,
        "dt_geom": dt_geom,
        "atol": 1.0e-10,
        "rtol": 1.0e-9,
        "max_accepted_steps": 5000,
        "max_evaluations": 100000,
        "max_duration_M": 10000.0,
        "ringdown_duration_M": 150.0,
        "spin_truncation": "generic-leading-ell8",
        "spin_v6": "bohe_total_energy",
        "include_pan_v8": False,
        "include_positive_chi_540": False,
        "final_spin_fit_order": [3, 4],
        "controller": "gsl_state_scaled",
    }

    runtime_before = file_hashes(ROOT / path for path in RUNTIME_SOURCES)
    input_before = {
        str(PUBLIC_PATH): sha256(PUBLIC_PATH),
        str(OLD_NATIVE_PATH): sha256(OLD_NATIVE_PATH),
        str(REVIEWED_CONTROLLER_REPORT): sha256(REVIEWED_CONTROLLER_REPORT),
    }
    startup = {
        "timestamp_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "host": platform.node(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "command": shlex.join([sys.executable, *sys.argv]),
        "executable": sys.executable,
        "parameters": parameters,
        "runtime_source_sha256_before": runtime_before,
        "input_sha256_before": input_before,
        "report_directory": str(report_dir),
    }
    atomic_json(report_dir / "startup.json", startup)

    started = time.monotonic()
    result = reconstruct_seobnrv4_22_adaptive(
        torch.tensor(omega_start, dtype=torch.float64),
        mass1,
        mass2,
        0.0,
        0.0,
        spin_truncation=parameters["spin_truncation"],
        spin_v6=parameters["spin_v6"],
        include_pan_v8=parameters["include_pan_v8"],
        include_positive_chi_540=parameters["include_positive_chi_540"],
        final_spin_fit_order=(3, 4),
        dt_geom=dt_geom,
        num_steps=parameters["max_accepted_steps"],
        ringdown_duration=parameters["ringdown_duration_M"],
        atol=parameters["atol"],
        rtol=parameters["rtol"],
        max_rejected_steps=10000,
        max_evaluations=parameters["max_evaluations"],
        max_accepted_steps=parameters["max_accepted_steps"],
        max_duration=parameters["max_duration_M"],
    )
    elapsed = time.monotonic() - started

    cache_path = report_dir / "native-full20-controller-clean.npz"
    cache_hash = save_cache(cache_path, result)
    public = np.load(PUBLIC_PATH, allow_pickle=False)
    old_native = np.load(OLD_NATIVE_PATH, allow_pickle=False)
    reference_mode = np.asarray(public["mode22"])
    old_native_mode = np.asarray(old_native["mode"])
    native_mode_geometric = tensor_array(result.mode).reshape(-1)
    public_distance_mpc = 100.0
    physical_scale = (
        (mass1 + mass2) * float(MSUN_SEC) * float(C_SI)
        / (public_distance_mpc * float(MPC_SI))
    )
    native_mode = native_mode_geometric * physical_scale
    # Store both conventions in the diagnostic cache.  This rewrite is
    # atomic and does not touch either supplied /tmp baseline.
    cache_payload = dict(np.load(cache_path, allow_pickle=False))
    cache_payload["mode_100mpc"] = native_mode
    cache_payload["public_distance_mpc"] = np.asarray(public_distance_mpc)
    cache_payload["geometric_to_public_scale"] = np.asarray(physical_scale)
    temporary_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
    np.savez(temporary_cache, **cache_payload)
    Path(str(temporary_cache) + ".npz").replace(cache_path)
    cache_hash = sha256(cache_path)
    reference_dt = float(public["dt_s"].reshape(-1)[0]) / (
        (mass1 + mass2) * float(MSUN_SEC)
    )
    native_time = tensor_array(result.time).reshape(-1)
    native_dt = float(native_time[1] - native_time[0])
    runtime_after = file_hashes(ROOT / path for path in RUNTIME_SOURCES)
    input_after = {
        str(PUBLIC_PATH): sha256(PUBLIC_PATH),
        str(OLD_NATIVE_PATH): sha256(OLD_NATIVE_PATH),
        str(REVIEWED_CONTROLLER_REPORT): sha256(REVIEWED_CONTROLLER_REPORT),
    }

    coarse = result.trajectory.coarse
    fine = result.trajectory.fine
    report = {
        "provenance": (
            "Clean adaptive SEOBNRv4 public-equation reconstruction using the "
            "named state-scaled RKF45 controller; raw comparison only, with "
            "no alignment, fitting, gate, or canonical-route change."
        ),
        "host": platform.node(),
        "cwd": str(Path.cwd()),
        "parameters": parameters,
        "elapsed_seconds": elapsed,
        "reviewed_controller_report": str(REVIEWED_CONTROLLER_REPORT),
        "reviewed_controller_report_sha256": input_before[str(REVIEWED_CONTROLLER_REPORT)],
        "public_input": {
            "path": str(PUBLIC_PATH),
            "sha256_before": input_before[str(PUBLIC_PATH)],
            "sha256_after": input_after[str(PUBLIC_PATH)],
            "samples": int(reference_mode.size),
            "dt_M": reference_dt,
            "epoch_s": float(public["epoch_s"].reshape(-1)[0]),
        },
        "old_native_input": {
            "path": str(OLD_NATIVE_PATH),
            "sha256_before": input_before[str(OLD_NATIVE_PATH)],
            "sha256_after": input_after[str(OLD_NATIVE_PATH)],
            "samples": int(old_native["mode"].size),
        },
        "native_cache": {
            "path": str(cache_path),
            "sha256": cache_hash,
            "samples": int(native_mode.size),
            "dt_M": native_dt,
            "geometric_mode_field": "mode",
            "public_mode_field": "mode_100mpc",
            "public_distance_mpc": public_distance_mpc,
            "geometric_to_public_scale": physical_scale,
        },
        "mode_normalization": {
            "comparison": "native geometric mode multiplied by M_sec*C_SI/(100 Mpc) against public mode22",
            "mass_total_solar": mass1 + mass2,
            "distance_mpc": public_distance_mpc,
            "scale": physical_scale,
            "provenance": "clean published_eob22_to_fd distance conversion; no fitted normalization",
        },
        "comparison": padded_metrics(native_mode, reference_mode),
        "rerun_vs_old_native": {
            "mode_geometric": direct_metrics(native_mode_geometric, old_native_mode),
            "time": direct_metrics(native_time, np.asarray(old_native["time"])),
            "coarse_output_time": direct_metrics(
                tensor_array(result.trajectory.coarse.output_time),
                np.asarray(old_native["coarse_time"]),
            ),
            "coarse_output_state": direct_metrics(
                tensor_array(result.trajectory.coarse.output_state),
                np.asarray(old_native["coarse_state"]),
            ),
            "fine_output_time": direct_metrics(
                tensor_array(result.trajectory.fine.output_time),
                np.asarray(old_native["fine_time"]),
            ),
            "fine_output_state": direct_metrics(
                tensor_array(result.trajectory.fine.output_state),
                np.asarray(old_native["fine_state"]),
            ),
        },
        "support": {
            "native": support_report(native_mode, native_dt),
            "public": support_report(reference_mode, reference_dt),
            "last_nonzero_time_delta_M": (
                support_report(native_mode, native_dt)["last_nonzero_time_M"]
                - support_report(reference_mode, reference_dt)["last_nonzero_time_M"]
            ),
        },
        "adaptive": {
            "coarse": trajectory_counter_report(coarse),
            "fine": trajectory_counter_report(fine),
            "restart_index": int(result.trajectory.restart_index),
            "restart_time_M": scalar(result.trajectory.restart_time),
            "fine_refinement_factor": int(result.attachment.refinement_factor),
            "splice_samples": int(result.time.numel()),
            "splice_valid_samples": int(tensor_array(result.trajectory.valid).sum()),
            "peak_time_M": scalar(result.trajectory.peak.time),
            "peak_frequency_M_inverse": scalar(result.trajectory.peak.frequency),
            "match_time_M": scalar(result.attachment.match_time),
            "boundary_time_M": scalar(result.attachment.boundary_time),
            "boundary_index": scalar(result.attachment.boundary_index),
        },
        "runtime_source_sha256_before": runtime_before,
        "runtime_source_sha256_after": runtime_after,
        "runtime_sources_unchanged": runtime_before == runtime_after,
        "inputs_unchanged": input_before == input_after,
        "diagnostic_script_sha256": sha256(Path(__file__).resolve()),
        "validation": {
            "native_finite": bool(np.isfinite(native_mode).all()),
            "public_finite": bool(np.isfinite(reference_mode).all()),
            "native_time_uniform": bool(np.allclose(np.diff(native_time), native_dt)),
            "public_dt_matches_mass_conversion": bool(
                math.isclose(reference_dt, dt_geom, rel_tol=0.0, abs_tol=1e-14)
            ),
            "runtime_sources_unchanged": runtime_before == runtime_after,
            "inputs_unchanged": input_before == input_after,
        },
    }
    atomic_json(report_dir / "diagnostic.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

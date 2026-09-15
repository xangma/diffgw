#!/usr/bin/env python3
"""Run the bounded clean TEOB auxiliary-grid NQC assembly probe.

The probe reuses the reviewed scalar ``nu=0.24`` post-adiabatic profile and
adaptive/fixed continuation, evaluates the clean factorized ``(2,2)`` mode,
and assembles the isolated NQC result with the explicitly named 2017 target
provider.  It stops at the accepted frequency peak plus two mass units.  A
portable NPZ cache contains the sampled inputs and the complete NQC output so
the independent ringdown assembly can consume the same arrays without
rerunning the trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import sys
import time
from pathlib import Path

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
from torchwave.approximants.eob.teob_published_driver import teob22_mode
from torchwave.approximants.eob.teob_trajectory import (
    conservative_trajectory_quantities,
)
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-merger-nqc-clean"
DEFAULT_OUTPUT = VERIFY / "diagnostic.json"
DEFAULT_CACHE = VERIFY / "physical-input-output.npz"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _float(value: object):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _vector(value: object):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1)]
    return [float(item) for item in value]


def _hamiltonian() -> TEOBResumSHamiltonian:
    dtype = torch.float64
    zero = torch.tensor(0.0, dtype=dtype)
    return TEOBResumSHamiltonian(
        torch.tensor(0.24, dtype=dtype), zero, zero, zero, zero,
    )


def _source_paths() -> dict[str, Path]:
    return {
        "module": ROOT / "src/torchwave/approximants/eob/teob_merger_nqc_clean.py",
        "test": ROOT / "tests/test_teob_merger_nqc_clean.py",
        "script": ROOT / "scripts/diagnose_teob_merger_nqc_clean.py",
        "pa": ROOT / "src/torchwave/approximants/eob/teob_postadiabatic_clean.py",
        "adaptive_driver": ROOT / "src/torchwave/approximants/eob/adaptive_driver_clean.py",
        "continuation": ROOT / "src/torchwave/approximants/eob/teob_adaptive_continuation_clean.py",
        "vacuum_driver": ROOT / "src/torchwave/approximants/eob/teob_vacuum_driver_clean.py",
        "mode_driver": ROOT / "src/torchwave/approximants/eob/teob_published_driver.py",
        "factorized_modes": ROOT / "src/torchwave/approximants/eob/factorized_modes.py",
        "trajectory": ROOT / "src/torchwave/approximants/eob/teob_trajectory.py",
        "nqc_basis": ROOT / "src/torchwave/approximants/eob/teob_nqc_clean.py",
        "numerical_attachment": ROOT / "src/torchwave/approximants/eob/numerical_attachment.py",
        "hamiltonian": ROOT / "src/torchwave/approximants/teobresums.py",
        "contract": ROOT / "docs/reviews/2026-09-13-parity-review/teob-merger-assembly-contract.md",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"JSON report path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--cache", type=Path, default=DEFAULT_CACHE,
        help=f"NPZ sampled input/output cache (default: {DEFAULT_CACHE})",
    )
    parser.add_argument(
        "--fixed-step-cap", type=float, default=DEFAULT_FIXED_STEP_CAP,
        help="fixed continuation cap in M (default: 0.1)",
    )
    return parser.parse_args()


def _save_cache(path: Path, result, mode: torch.Tensor, amplitude: torch.Tensor,
                phase: torch.Tensor, radial_acceleration: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        # Inputs used by the isolated assembly.
        "time": _numpy(result.time),
        "state": _numpy(result.state),
        "mode": _numpy(mode),
        "amplitude": _numpy(amplitude),
        "phase": _numpy(phase),
        "orbital_frequency": _numpy(result.frequency),
        "radial_acceleration": _numpy(radial_acceleration),
        # Corrected original and auxiliary series for a later assembly stage.
        "corrected_amplitude": _numpy(result.nqc.original.corrected_amplitude),
        "corrected_phase": _numpy(result.nqc.original.corrected_phase),
        "aux_time": _numpy(result.nqc.auxiliary.time),
        "aux_state": _numpy(result.nqc.auxiliary.state),
        "aux_amplitude": _numpy(result.nqc.auxiliary.amplitude),
        "aux_phase": _numpy(result.nqc.auxiliary.phase),
        "aux_orbital_frequency": _numpy(result.nqc.auxiliary.orbital_frequency),
        "aux_radial_acceleration": _numpy(result.nqc.auxiliary.radial_acceleration),
        "aux_corrected_amplitude": _numpy(result.nqc.auxiliary.corrected_amplitude),
        "aux_corrected_phase": _numpy(result.nqc.auxiliary.corrected_phase),
        "aux_basis_amplitude": _numpy(result.nqc.auxiliary.basis_amplitude),
        "aux_basis_phase": _numpy(result.nqc.auxiliary.basis_phase),
        "auxiliary_negative_amplitude": _numpy(result.nqc.auxiliary_negative_amplitude),
        "amplitude_rw": _numpy(result.nqc.amplitude_rw),
        "auxiliary_amplitude_rw": _numpy(result.nqc.auxiliary_amplitude_rw),
        "amplitude_coefficients": _numpy(result.nqc.coefficients.amplitude),
        "phase_coefficients": _numpy(result.nqc.coefficients.phase),
        "amplitude_matrix": _numpy(result.nqc.match.amplitude_matrix),
        "phase_matrix": _numpy(result.nqc.match.phase_matrix),
        "targets": _numpy(torch.stack(result.nqc.targets)),
    }
    np.savez(path, **arrays)
    return _hash(path)


def main() -> int:
    args = _parse_args()
    torch.set_default_dtype(torch.float64)
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    started = time.perf_counter()
    source_paths = _source_paths()
    before = {key: _hash(path) for key, path in source_paths.items()}
    report: dict[str, object] = {
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "command": command,
        "pid": os.getpid(),
        "python": sys.executable,
        "device": "cpu",
        "started_unix": time.time(),
        "probe": {
            "nu": 0.24,
            "effective_spin": 0.0,
            "r0": 20.0,
            "r_min": 14.0,
            "dr": 0.1,
            "pa_order": 8,
            "fixed_step_cap": args.fixed_step_cap,
            "continuation_duration": DEFAULT_CONTINUATION_DURATION,
            "atol": DEFAULT_ATOL,
            "rtol": DEFAULT_RTOL,
            "max_rejected_steps": DEFAULT_MAX_REJECTED_STEPS,
            "max_evaluations": DEFAULT_MAX_EVALUATIONS,
            "max_accepted_steps": DEFAULT_MAX_ACCEPTED_STEPS,
            "max_duration": DEFAULT_MAX_DURATION,
        },
        "source_sha256_before": before,
        "cache_path": str(args.cache),
    }
    try:
        hamiltonian = _hamiltonian()
        pa = alternating8pa(
            hamiltonian, r0=20.0, r_min=14.0, dr=0.1, pa_order=8,
        )
        continuation = continue_teob_adaptive(
            hamiltonian,
            pa,
            fixed_step_cap=args.fixed_step_cap,
            continuation_duration=DEFAULT_CONTINUATION_DURATION,
            max_rejected_steps=DEFAULT_MAX_REJECTED_STEPS,
            max_evaluations=DEFAULT_MAX_EVALUATIONS,
            max_accepted_steps=DEFAULT_MAX_ACCEPTED_STEPS,
            max_duration=DEFAULT_MAX_DURATION,
        )
        time_series = continuation.time
        state = continuation.state
        orbital_frequency = continuation.frequency
        mode = teob22_mode(state, hamiltonian, omega=orbital_frequency)
        amplitude = mode.abs()
        phase = phase_from_mode(mode, state[:, 1])
        with torch.no_grad():
            radial_acceleration = conservative_trajectory_quantities(
                hamiltonian, state,
            ).radial_acceleration
        nu = torch.tensor(0.24, dtype=time_series.dtype, device=time_series.device)
        targets = published_2017_nqc_targets(nu, torch.tensor(0.0, dtype=nu.dtype))
        nqc = assemble_teob_merger_nqc(
            time_series,
            state,
            amplitude,
            phase,
            orbital_frequency,
            radial_acceleration,
            nu=nu,
            fixed_step=args.fixed_step_cap,
            targets=targets,
            target_provenance=PUBLISHED_2017_NQC_PROVENANCE,
        )
        continuation_stats = continuation.statistics
        report["continuation"] = {
            "status": continuation.status,
            "samples": int(continuation.time.numel()),
            "valid_samples": int(continuation.valid.sum()),
            "adaptive_samples": int(continuation.adaptive_time.numel()),
            "fixed_samples": int(continuation.fixed_time.numel()),
            "peak_event_index": continuation.peak_event_index,
            "peak_event_time": _float(continuation.peak_event_time),
            "peak_event_frequency": _float(continuation.peak_event_frequency),
            "endpoint_time": _float(continuation.endpoint_time),
            "endpoint_state": _vector(continuation.endpoint_state),
            "phase_seam_error": _float(continuation.phase_seam_error),
            "statistics": {
                "adaptive_accepted_steps": continuation_stats.adaptive_accepted_steps,
                "adaptive_rejected_steps": continuation_stats.adaptive_rejected_steps,
                "adaptive_evaluations": continuation_stats.adaptive_evaluations,
                "fixed_accepted_steps": continuation_stats.fixed_accepted_steps,
                "fixed_rejected_steps": continuation_stats.fixed_rejected_steps,
                "fixed_evaluations": continuation_stats.fixed_evaluations,
                "accepted_steps": continuation_stats.accepted_steps,
                "rejected_steps": continuation_stats.rejected_steps,
                "evaluations": continuation_stats.evaluations,
            },
        }
        report["mode"] = {
            "samples": int(mode.numel()),
            "amplitude_min": _float(amplitude.min()),
            "amplitude_max": _float(amplitude.max()),
            "phase_initial": _float(phase[0]),
            "phase_final": _float(phase[-1]),
        }
        report["nqc"] = {
            "target_provenance": nqc.target_provenance,
            "targets": _vector(torch.stack(nqc.targets)),
            "cut_start_index": nqc.cut_start_index,
            "cut_last_index": nqc.cut_last_index,
            "auxiliary_samples": int(nqc.auxiliary.time.numel()),
            "auxiliary_spacing": _float(nqc.auxiliary_spacing),
            "t_peak_grid": _float(nqc.t_peak_grid),
            "t_nqc": _float(nqc.t_nqc),
            "matching_index": nqc.matching_index,
            "amplitude_coefficients": _vector(nqc.coefficients.amplitude),
            "phase_coefficients": _vector(nqc.coefficients.phase),
            "amplitude_matrix": nqc.amplitude_matrix.tolist(),
            "phase_matrix": nqc.phase_matrix.tolist(),
            "amplitude_condition": _float(nqc.amplitude_condition),
            "phase_condition": _float(nqc.phase_condition),
            "amplitude_residual": _vector(nqc.amplitude_residual),
            "phase_residual": _vector(nqc.phase_residual),
            "negative_auxiliary_samples": int(nqc.auxiliary_negative_amplitude.sum()),
        }
        cache_hash = _save_cache(args.cache, type("Bundle", (), {
            "time": time_series,
            "state": state,
            "frequency": orbital_frequency,
            "nqc": nqc,
        })(), mode, amplitude, phase, radial_acceleration)
        report["cache_sha256"] = cache_hash
        report["cache_arrays"] = [
            "time", "state", "mode", "amplitude", "phase",
            "orbital_frequency", "radial_acceleration", "corrected_amplitude",
            "corrected_phase", "aux_time", "aux_state", "aux_amplitude",
            "aux_phase", "aux_orbital_frequency", "aux_radial_acceleration",
            "aux_corrected_amplitude", "aux_corrected_phase",
        ]
        report["status"] = "complete"
    except Exception as exc:  # pragma: no cover - failure is reported for a probe
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    after = {key: _hash(path) for key, path in source_paths.items()}
    report["source_sha256_after"] = after
    report["source_hashes_unchanged"] = before == after
    report["elapsed_seconds"] = time.perf_counter() - started
    rendered = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

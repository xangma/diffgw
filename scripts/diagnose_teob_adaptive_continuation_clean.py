#!/usr/bin/env python3
"""Run the bounded physical clean TEOB PA-to-peak continuation probe.

The probe stops after the accepted frequency peak and a two-M continuation.
It does not construct modes, a waveform, or a merger/ringdown segment.
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
from torchwave.approximants.eob.teob_postadiabatic_clean import alternating8pa
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _state(value):
    if value is None:
        return None
    return [float(item) for item in value.detach().cpu().reshape(-1)]


def _hamiltonian() -> TEOBResumSHamiltonian:
    dtype = torch.float64
    zero = torch.tensor(0.0, dtype=dtype)
    return TEOBResumSHamiltonian(
        torch.tensor(0.24, dtype=dtype), zero, zero, zero, zero,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed-step-cap", type=float, default=DEFAULT_FIXED_STEP_CAP,
        help="fixed continuation cap in M (default: 0.1)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="optional JSON output path; stdout is used by default",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        device = "cpu"
    else:  # pragma: no cover - the bounded probe is normally CPU-only
        device = "cuda"
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    started = time.perf_counter()
    source_paths = {
        "pa": ROOT / "src/torchwave/approximants/eob/teob_postadiabatic_clean.py",
        "adaptive_driver": ROOT / "src/torchwave/approximants/eob/adaptive_driver_clean.py",
        "continuation": ROOT / "src/torchwave/approximants/eob/teob_adaptive_continuation_clean.py",
        "vacuum_driver": ROOT / "src/torchwave/approximants/eob/teob_vacuum_driver_clean.py",
        "contract": ROOT / "docs/reviews/2026-09-13-parity-review/teob-adaptive-continuation-contract.md",
    }
    source_sha256_before = {key: _hash(path) for key, path in source_paths.items()}
    report = {
        "host": socket.gethostname(),
        "cwd": os.getcwd(),
        "command": command,
        "pid": os.getpid(),
        "python": sys.executable,
        "device": device,
        "started_unix": time.time(),
        "probe": {
            "nu": 0.24,
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
        "source_sha256_before": source_sha256_before,
    }
    try:
        hamiltonian = _hamiltonian()
        pa = alternating8pa(
            hamiltonian, r0=20.0, r_min=14.0, dr=0.1, pa_order=8,
        )
        result = continue_teob_adaptive(
            hamiltonian,
            pa,
            fixed_step_cap=args.fixed_step_cap,
            continuation_duration=DEFAULT_CONTINUATION_DURATION,
            max_rejected_steps=DEFAULT_MAX_REJECTED_STEPS,
            max_evaluations=DEFAULT_MAX_EVALUATIONS,
            max_accepted_steps=DEFAULT_MAX_ACCEPTED_STEPS,
            max_duration=DEFAULT_MAX_DURATION,
        )
        event_index = result.peak_event_index
        report["pa"] = {
            "status": pa.status,
            "samples": int(pa.time.numel()),
            "endpoint_time": _float(pa.endpoint_time),
            "endpoint_state": _state(pa.endpoint_state),
        }
        report["continuation"] = {
            "status": result.status,
            "samples": int(result.time.numel()),
            "valid_samples": int(result.valid.sum()),
            "adaptive_samples": int(result.adaptive_time.numel()),
            "fixed_samples": int(result.fixed_time.numel()),
            "first_ode_step": _float(result.first_ode_step),
            "adaptive_next_step": _float(result.adaptive_next_step),
            "fixed_step": _float(result.fixed_step),
            "peak_event_index": event_index,
            "peak_event_time": _float(result.peak_event_time),
            "peak_event_frequency": _float(result.peak_event_frequency),
            "peak_event_radius": _float(result.state[event_index, 0]) if event_index is not None else None,
            "endpoint_time": _float(result.endpoint_time),
            "endpoint_state": _state(result.endpoint_state),
            "target_time": _float(result.target_time),
            "phase_seam_error": _float(result.phase_seam_error),
            "statistics": {
                "adaptive_accepted_steps": result.statistics.adaptive_accepted_steps,
                "adaptive_rejected_steps": result.statistics.adaptive_rejected_steps,
                "adaptive_evaluations": result.statistics.adaptive_evaluations,
                "fixed_accepted_steps": result.statistics.fixed_accepted_steps,
                "fixed_rejected_steps": result.statistics.fixed_rejected_steps,
                "fixed_evaluations": result.statistics.fixed_evaluations,
                "accepted_steps": result.statistics.accepted_steps,
                "rejected_steps": result.statistics.rejected_steps,
                "evaluations": result.statistics.evaluations,
                "termination": result.statistics.termination,
            },
        }
        report["status"] = "complete"
    except Exception as exc:  # pragma: no cover - exercised only on a failed probe
        report["status"] = "failed"
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "reason": getattr(exc, "reason", None),
        }
    report["source_sha256_after"] = {
        key: _hash(path) for key, path in source_paths.items()
    }
    report["source_hashes_unchanged"] = (
        report["source_sha256_before"] == report["source_sha256_after"]
    )
    report["elapsed_seconds"] = time.perf_counter() - started
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

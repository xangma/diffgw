#!/usr/bin/env python3
"""Bounded diagnostics for the independent TEOB alternating PA profile.

The default run covers only ``r=20`` to ``r=14``.  It reports construction
time, balance residuals, PA order refinement, and radial-grid refinement; it
does not generate a waveform or continue through merger.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from torchwave.approximants.eob.teob_postadiabatic_clean import alternating8pa
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


def _hamiltonian() -> TEOBResumSHamiltonian:
    dtype = torch.float64
    nu = torch.tensor(0.24, dtype=dtype)
    zero = torch.zeros_like(nu)
    return TEOBResumSHamiltonian(nu, zero, zero, zero, zero)


def _run(hamiltonian, *, r0: float, r_min: float, dr: float, order: int) -> dict[str, object]:
    started = time.perf_counter()
    result = alternating8pa(
        hamiltonian,
        r0=r0,
        r_min=r_min,
        dr=dr,
        pa_order=order,
    )
    elapsed = time.perf_counter() - started
    torque_scale = result.torque.abs().clamp_min(1e-30)
    return {
        "status": result.status,
        "pa_order": order,
        "dr": dr,
        "r0": r0,
        "r_min": r_min,
        "points": int(result.radius.numel()),
        "intervals": max(int(result.radius.numel()) - 1, 0),
        "elapsed_seconds": elapsed,
        "endpoint_radius": float(result.radius[-1]),
        "endpoint_time": None if result.endpoint_time is None else float(result.endpoint_time),
        "endpoint_phase": None if result.endpoint_phase is None else float(result.endpoint_phase),
        "max_abs_radial_balance": float(result.radial_balance_residual.abs().max()),
        "max_relative_angular_balance": float(
            (result.angular_balance_residual.abs() / torque_scale).max()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0", type=float, default=20.0)
    parser.add_argument("--r-min", type=float, default=14.0)
    parser.add_argument("--dr", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    hamiltonian = _hamiltonian()
    order_runs = [
        _run(
            hamiltonian,
            r0=args.r0,
            r_min=args.r_min,
            dr=args.dr,
            order=order,
        )
        for order in (2, 4, 6, 8)
    ]
    grid_runs = [
        _run(
            hamiltonian,
            r0=args.r0,
            r_min=args.r_min,
            dr=spacing,
            order=8,
        )
        for spacing in (2.0 * args.dr, args.dr)
    ]
    payload = {
        "description": "bounded clean-room nonspinning TEOB alternating PA diagnostics",
        "system": {"nu": 0.24, "chi1": 0.0, "chi2": 0.0, "lambda1": 0.0, "lambda2": 0.0},
        "order_runs": order_runs,
        "grid_runs": grid_runs,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

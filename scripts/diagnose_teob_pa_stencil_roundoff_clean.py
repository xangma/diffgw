#!/usr/bin/env python3
"""Compare PA stencils with independently derived uniform-grid arithmetic.

This bounded, diagnostic-only probe keeps the clean PA implementation intact
and replaces its two numerical primitives temporarily:

* quartic derivatives use coefficients obtained from exact integer-offset
  polynomial moments on a fixed spacing ``-dr``;
* cubic interval integrals use coefficients obtained from exact normalized
  moments on the same fixed spacing.

The production path solves an actual-node Vandermonde system for every local
stencil.  Both paths are evaluated on the same ``r0=19.999999999999996`` to
``r_min=14`` grid, with the diagnostic leading-order horizon reaction used by
the reviewed PA endpoint.  The supplied black-box endpoint is reported as a
comparison point only.  No fitted values, waveform alignment, or runtime PA
edits are used.
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
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

import torchwave.approximants.eob.teob_postadiabatic_clean as pa
from scripts.diagnose_teob_paired_driver_clean import (
    _LeadingOrderHorizonReaction,
    _hamiltonian,
)


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-pa-stencil-roundoff-clean"
DEFAULT_OUTPUT = VERIFY / "report.json"
DEFAULT_LOG = VERIFY / "diagnostic.log"

R0 = 19.999999999999996
R_MIN = 14.0
DR = 0.1
FIXED_SPACING = -DR
ORDERS = (2, 4, 6, 8)

# Parent-provided black-box endpoint for the exact grid and diagnostic
# leading-order horizon branch.  It is deliberately not used to select or
# modify any coefficient.
REFERENCE_ENDPOINT = {
    "time_M": 9523.942662602482,
    "phase": 132.17031461096497,
    "p_phi": 4.215698565112572,
    "pr_star": -0.0012978461179493459,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths() -> dict[str, Path]:
    """Return only clean sources directly used by this diagnostic."""

    return {
        "script": Path(__file__).resolve(),
        "postadiabatic": ROOT / "src/torchwave/approximants/eob/teob_postadiabatic_clean.py",
        "hamiltonian": ROOT / "src/torchwave/approximants/teobresums.py",
        "paired_clean_helpers": ROOT / "scripts/diagnose_teob_paired_driver_clean.py",
        "contract": ROOT / "docs/reviews/2026-09-13-parity-review/teob-postadiabatic-contract.md",
    }


def _source_hashes() -> dict[str, str]:
    paths = _source_paths()
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"source paths missing: {missing}")
    return {name: _sha256(path) for name, path in paths.items()}


def _finite(value: Any) -> float:
    if isinstance(value, Tensor):
        value = value.detach().cpu().reshape(-1)[0].item()
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite diagnostic value: {value!r}")
    return value


def _solve_fraction(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    """Solve a small exact linear system by fraction Gaussian elimination."""

    n = len(rhs)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("fraction system must be square")
    augmented = [list(row) + [rhs[row_index]] for row_index, row in enumerate(matrix)]
    for column in range(n):
        pivot_row = next(
            (row for row in range(column, n) if augmented[row][column] != 0),
            None,
        )
        if pivot_row is None:
            raise ValueError("singular exact stencil system")
        if pivot_row != column:
            augmented[column], augmented[pivot_row] = augmented[pivot_row], augmented[column]
        pivot = augmented[column][column]
        augmented[column] = [entry / pivot for entry in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][n] for row in range(n)]


def _quartic_starts(n: int) -> list[int]:
    if n < 5:
        raise ValueError("quartic derivative needs at least five nodes")
    return [min(max(index - 2, 0), n - 5) for index in range(n)]


def _cubic_starts(n: int) -> list[int]:
    if n < 4:
        raise ValueError("cubic integral needs at least four nodes")
    starts: list[int] = []
    for interval in range(n - 1):
        if interval == 0:
            starts.append(0)
        elif interval == n - 2:
            starts.append(n - 4)
        else:
            starts.append(interval - 1)
    return starts


def _fraction_quartic_coefficients(n: int) -> tuple[tuple[Fraction, ...], ...]:
    """Return d/dq coefficients from exact integer-offset moments."""

    rows: list[tuple[Fraction, ...]] = []
    for target, start in enumerate(_quartic_starts(n)):
        offsets = [Fraction(node - target) for node in range(start, start + 5)]
        matrix = [
            [offset**power for offset in offsets]
            for power in range(5)
        ]
        rhs = [Fraction(0), Fraction(1), Fraction(0), Fraction(0), Fraction(0)]
        rows.append(tuple(_solve_fraction(matrix, rhs)))
    return tuple(rows)


def _fraction_cubic_coefficients(n: int) -> tuple[tuple[Fraction, ...], ...]:
    """Return normalized four-node interval coefficients from exact moments."""

    rows: list[tuple[Fraction, ...]] = []
    moments = [Fraction(1), Fraction(1, 2), Fraction(1, 3), Fraction(1, 4)]
    for interval, start in enumerate(_cubic_starts(n)):
        q_nodes = [Fraction(node - interval) for node in range(start, start + 4)]
        matrix = [
            [q_node**power for q_node in q_nodes]
            for power in range(4)
        ]
        rows.append(tuple(_solve_fraction(matrix, moments)))
    return tuple(rows)


def _fraction_moment_residuals(
    quartic: tuple[tuple[Fraction, ...], ...],
    cubic: tuple[tuple[Fraction, ...], ...],
    n: int,
) -> dict[str, float]:
    derivative_error = Fraction(0)
    for target, start in enumerate(_quartic_starts(n)):
        offsets = [Fraction(node - target) for node in range(start, start + 5)]
        weights = quartic[target]
        expected = [Fraction(0), Fraction(1), Fraction(0), Fraction(0), Fraction(0)]
        for power, target_moment in enumerate(expected):
            residual = sum(weight * offset**power for weight, offset in zip(weights, offsets))
            derivative_error = max(derivative_error, abs(residual - target_moment))
    integral_error = Fraction(0)
    moments = [Fraction(1), Fraction(1, 2), Fraction(1, 3), Fraction(1, 4)]
    for interval, start in enumerate(_cubic_starts(n)):
        q_nodes = [Fraction(node - interval) for node in range(start, start + 4)]
        for power, target_moment in enumerate(moments):
            residual = sum(weight * q_node**power for weight, q_node in zip(cubic[interval], q_nodes))
            integral_error = max(integral_error, abs(residual - target_moment))
    return {
        "quartic_derivative_max_abs_fraction": float(derivative_error),
        "cubic_integral_max_abs_fraction": float(integral_error),
    }


def _check_grid(values: Tensor, radius: Tensor) -> int:
    if values.ndim < 1:
        raise ValueError("values must have a grid axis")
    if radius.ndim != 1 or values.shape[-1] != radius.numel():
        raise ValueError("values and radius have incompatible grid axes")
    if radius.numel() < 5:
        raise ValueError("diagnostic grid has too few nodes")
    return int(radius.numel())


def _fixed_quartic_derivative(
    values: Tensor,
    radius: Tensor | float,
) -> Tensor:
    """Diagnostic derivative using exact fixed-spacing coefficients."""

    radius_t = torch.as_tensor(radius, dtype=values.dtype, device=values.device)
    if radius_t.ndim == 0:
        spacing = radius_t
        n = int(values.shape[-1])
    else:
        n = _check_grid(values, radius_t)
        spacing = torch.as_tensor(FIXED_SPACING, dtype=values.dtype, device=values.device)
    if not bool(spacing < 0):
        raise ValueError("fixed-spacing diagnostic expects descending spacing")
    coefficients = _fraction_quartic_coefficients(n)
    weights = torch.tensor(
        [[float(coefficient) / float(spacing) for coefficient in row] for row in coefficients],
        dtype=values.dtype,
        device=values.device,
    )
    indices = torch.tensor(
        [[start + offset for offset in range(5)] for start in _quartic_starts(n)],
        dtype=torch.long,
        device=values.device,
    )
    return (values[..., indices] * weights).sum(dim=-1)


def _fixed_cubic_interval_integral(values: Tensor, radius: Tensor) -> Tensor:
    """Diagnostic cubic integral using exact fixed-spacing coefficients."""

    n = _check_grid(values, radius)
    coefficients = _fraction_cubic_coefficients(n)
    spacing = torch.as_tensor(FIXED_SPACING, dtype=values.dtype, device=values.device)
    weights = torch.tensor(
        [[float(coefficient) * float(spacing) for coefficient in row] for row in coefficients],
        dtype=values.dtype,
        device=values.device,
    )
    indices = torch.tensor(
        [[start + offset for offset in range(4)] for start in _cubic_starts(n)],
        dtype=torch.long,
        device=values.device,
    )
    return (values[..., indices] * weights).sum(dim=-1)


def _polynomial_validation() -> dict[str, Any]:
    """Check the independently generated tables against degree limits."""

    n = 13
    q = torch.arange(n, dtype=torch.float64)
    radius = q * FIXED_SPACING
    derivative_coefficients = torch.tensor(
        [0.37, -0.21, 0.11, -0.031, 0.007], dtype=torch.float64
    )
    values = sum(derivative_coefficients[power] * q.pow(power) for power in range(5))
    expected_derivative = sum(
        power * derivative_coefficients[power] * q.pow(power - 1)
        for power in range(1, 5)
    ) / FIXED_SPACING
    derivative = _fixed_quartic_derivative(values, radius)

    integral_coefficients = torch.tensor(
        [0.43, -0.17, 0.052, -0.009], dtype=torch.float64
    )
    integral_values = sum(
        integral_coefficients[power] * q.pow(power) for power in range(4)
    )
    left = q[:-1]
    right = q[1:]
    expected_intervals = sum(
        integral_coefficients[power]
        * (right.pow(power + 1) - left.pow(power + 1))
        / float(power + 1)
        * FIXED_SPACING
        for power in range(4)
    )
    intervals = _fixed_cubic_interval_integral(integral_values, radius)

    derivative_error = torch.max(torch.abs(derivative - expected_derivative)).item()
    integral_error = torch.max(torch.abs(intervals - expected_intervals)).item()
    if derivative_error > 5e-12 or integral_error > 5e-12:
        raise AssertionError(
            f"fixed stencil polynomial check failed: derivative={derivative_error}, "
            f"integral={integral_error}"
        )
    return {
        "grid_points": n,
        "spacing": FIXED_SPACING,
        "quartic_degree": 4,
        "cubic_degree": 3,
        "max_abs_derivative_error": float(derivative_error),
        "max_abs_integral_error": float(integral_error),
        "fraction_moment_residuals": _fraction_moment_residuals(
            _fraction_quartic_coefficients(n),
            _fraction_cubic_coefficients(n),
            n,
        ),
    }


def _endpoint(result: Any) -> dict[str, float]:
    return {
        "radius": _finite(result.radius[-1]),
        "time_M": _finite(result.endpoint_time),
        "phase": _finite(result.endpoint_phase),
        "pr_star": _finite(result.pr_star[-1]),
        "p_phi": _finite(result.p_phi[-1]),
    }


def _endpoint_difference(left: dict[str, float], right: dict[str, float]) -> dict[str, dict[str, float]]:
    fields = sorted(set(left).intersection(right))
    return {
        field: {
            "signed": left[field] - right[field],
            "abs": abs(left[field] - right[field]),
        }
        for field in fields
    }


def _profile_differences(left: Any, right: Any) -> dict[str, dict[str, float]]:
    fields = ("time", "phase", "pr_star", "p_phi", "omega", "radial_velocity")
    result: dict[str, dict[str, float]] = {}
    for field in fields:
        lhs = np.asarray(getattr(left, field).detach().cpu(), dtype=np.float64)
        rhs = np.asarray(getattr(right, field).detach().cpu(), dtype=np.float64)
        difference = lhs - rhs
        result[field] = {
            "max_abs": float(np.max(np.abs(difference))),
            "rms": float(np.sqrt(np.mean(difference * difference))),
            "endpoint_abs": float(abs(difference[-1])),
        }
    return result


def _run_pa(order: int, *, fixed_spacing: bool) -> Any:
    """Run one scalar PA profile, restoring all monkeypatched names."""

    old_quartic: Callable[..., Any] = pa.quartic_derivative
    old_cubic: Callable[..., Any] = pa.cubic_interval_integral
    if fixed_spacing:
        pa.quartic_derivative = _fixed_quartic_derivative
        pa.cubic_interval_integral = _fixed_cubic_interval_integral
    try:
        return pa.alternating8pa(
            _hamiltonian(),
            _LeadingOrderHorizonReaction(),
            r0=R0,
            r_min=R_MIN,
            dr=DR,
            pa_order=order,
            include_horizon_absorption=True,
        )
    finally:
        pa.quartic_derivative = old_quartic
        pa.cubic_interval_integral = old_cubic


def _spacing_stats(radius: Tensor) -> dict[str, float]:
    spacing = np.diff(np.asarray(radius.detach().cpu(), dtype=np.float64))
    return {
        "count": int(spacing.size),
        "min": float(np.min(spacing)),
        "max": float(np.max(spacing)),
        "mean": float(np.mean(spacing)),
        "max_abs_deviation_from_nominal": float(np.max(np.abs(spacing - FIXED_SPACING))),
    }


class _RunLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger = _RunLog(args.log_path)
    started_unix = time.time()
    started_perf = time.perf_counter()
    source_before = _source_hashes()
    logger.write(
        "start "
        + json.dumps(
            {
                "host": socket.gethostname(),
                "cwd": str(Path.cwd()),
                "pid": os.getpid(),
                "command": " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv]),
                "python": sys.executable,
            },
            sort_keys=True,
        )
    )
    logger.write("validating independent Fraction-derived stencils")
    validation = _polynomial_validation()
    rows: list[dict[str, Any]] = []
    native_results: dict[int, Any] = {}
    for order in ORDERS:
        logger.write(f"running PA order {order}: actual-node Vandermonde")
        native = _run_pa(order, fixed_spacing=False)
        native_results[order] = native
        logger.write(f"running PA order {order}: fixed-spacing Fraction coefficients")
        fixed = _run_pa(order, fixed_spacing=True)
        native_endpoint = _endpoint(native)
        fixed_endpoint = _endpoint(fixed)
        rows.append(
            {
                "pa_order": order,
                "points": int(native.radius.numel()),
                "native_actual_node": native_endpoint,
                "fixed_spacing": fixed_endpoint,
                "fixed_minus_native": _endpoint_difference(fixed_endpoint, native_endpoint),
                "native_minus_reference": _endpoint_difference(native_endpoint, REFERENCE_ENDPOINT),
                "fixed_minus_reference": _endpoint_difference(fixed_endpoint, REFERENCE_ENDPOINT),
                "profile_fixed_minus_native": _profile_differences(fixed, native),
            }
        )
    native_order8 = native_results[8]
    spacing = _spacing_stats(native_order8.radius)
    source_after = _source_hashes()
    source_unchanged = source_before == source_after
    elapsed = time.perf_counter() - started_perf
    report = {
        "status": "complete" if source_unchanged else "invalidated_by_source_change",
        "scope": "PA stencil roundoff diagnostic only; no runtime edits",
        "method": {
            "native": "teob_postadiabatic_clean actual-node Vandermonde solves",
            "fixed_spacing": "independent exact Fraction moment solves, converted to float64 and applied at nominal spacing -dr",
            "quartic_moments": "sum(w_j q_j^p)=delta[p,1], p=0..4",
            "cubic_moments": "sum(w_j q_j^p)=1/(p+1), p=0..3, then multiply by descending spacing",
            "quartic_stencils": "five consecutive nodes; first and last five at boundaries, centered elsewhere",
            "cubic_stencils": "four consecutive nodes; first and last four at boundaries, i-1..i+2 in the interior",
            "fixed_spacing_value": FIXED_SPACING,
        },
        "inputs": {
            "nu": 0.24,
            "r0": R0,
            "r_min": R_MIN,
            "dr": DR,
            "pa_orders": list(ORDERS),
            "horizon_branch": "diagnostic leading-order zero-spin horizon reaction",
            "spin": [0.0, 0.0],
            "tides": [0.0, 0.0],
            "radius_spacing_statistics": spacing,
        },
        "reference_endpoint": {
            "values": REFERENCE_ENDPOINT,
            "provenance": "parent-provided black-box endpoint for the contracted grid and leading-order horizon branch",
            "used_for_runtime": False,
        },
        "independent_polynomial_validation": validation,
        "orders": rows,
        "conclusion": {
            "criterion": "compare fixed-spacing minus native with the supplied order-8 native/reference drift",
            "order8_native_reference_abs": {
                field: rows[-1]["native_minus_reference"][field]["abs"]
                for field in rows[-1]["native_minus_reference"]
            },
            "order8_fixed_native_abs": {
                field: rows[-1]["fixed_minus_native"][field]["abs"]
                for field in rows[-1]["fixed_minus_native"]
            },
            "interpretation": "fixed-spacing versus actual-node differences quantify stencil arithmetic sensitivity; they do not replace the published/native endpoint or tune it",
        },
        "run": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "pid": os.getpid(),
            "command": " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv]),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "started_unix": started_unix,
            "elapsed_seconds": elapsed,
            "log_path": str(args.log_path),
            "stop_command": "terminate the diagnostic process by PID if required",
        },
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
        "source_hashes_unchanged": source_unchanged,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.write(f"complete elapsed_seconds={elapsed:.6f} source_hashes_unchanged={source_unchanged}")
    logger.write(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

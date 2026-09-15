"""Independent numerical representation of the IMRPhenomD 220 QNM inputs.

The D implementation consumes dimensionless *cyclic* frequencies, ``M f``.
This module deliberately constructs its small compatibility representation from
TorchWave's independently generated Kerr eigenproblem evaluator:

``signed_qnm_220`` -> divide by ``2*pi`` -> round to six significant digits
-> natural cubic interpolation in the signed spin.

The ordinary knots are the exact rational grid ``a = k/500`` for
``k=-499,...,499`` (through ``|a|=.998``).  Two independently generated guard
knots at ``a=-.999`` and ``a=.999`` make the usable end of the rounded table
explicit.  No value is invented at ``|a|=1``: the independent generator only
supports ``|a| <= .99999``, and queries outside the certified compatibility
interior ``|a| <= .994`` use its high-precision roots.  The rounded cubic is
therefore a compatibility representation, while the generator remains the
endpoint diagnostic path until an extremal-root prescription is independently
supported.

No waveform output, external QNM table, or source ordinate is used here.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Tuple

import torch

from torchwave.approximants.kerr_qnm_generated import signed_qnm_220


ROUNDING_SIG_DIGITS = 6
SPIN_STEP = 1.0 / 500.0
INTERIOR_SPIN_MIN = -499.0 / 500.0
INTERIOR_SPIN_MAX = 499.0 / 500.0
# The last two ordinary knots are retained as guards.  The frozen D cases
# stay well inside this certificate range (the largest magnitude sampled by
# the audit is below .85).
COMPATIBILITY_SPIN_MIN = -497.0 / 500.0
COMPATIBILITY_SPIN_MAX = 497.0 / 500.0
GENERATED_SPIN_LIMIT = 0.99999
TABLE_SPIN_MIN = -0.999
TABLE_SPIN_MAX = 0.999
SUPPORTED_SPIN_MIN = -GENERATED_SPIN_LIMIT
SUPPORTED_SPIN_MAX = GENERATED_SPIN_LIMIT
ENDPOINT_POLICY = (
    "rounded_cubic_within_abs_spin_0.994;"
    " generated_fallback_to_0.99999;"
    " abs_spin_1_unsupported"
)


def _round_significant(value: float, digits: int = ROUNDING_SIG_DIGITS) -> float:
    """Round a finite scalar to decimal significant digits reproducibly."""

    if not math.isfinite(value):
        raise ValueError("QNM ordinate must be finite")
    if value == 0.0:
        return 0.0
    return float(format(value, f".{digits}g"))


def _knot_spins() -> Tuple[float, ...]:
    # Keeping the integer numerator explicit prevents an accumulated arange
    # step error from changing an interval at a knot boundary.
    return (
        -0.999,
        *(float(k) / 500.0 for k in range(-499, 500)),
        0.999,
    )


KNOT_SPINS: Tuple[float, ...] = _knot_spins()
N_KNOTS = len(KNOT_SPINS)


def _rounding_step(value: float, digits: int = ROUNDING_SIG_DIGITS) -> float:
    """Return the decimal half-step associated with significant rounding."""

    if value == 0.0:
        return 0.5 * 10.0 ** (1 - digits)
    exponent = math.floor(math.log10(abs(value)))
    return 0.5 * 10.0 ** (exponent - digits + 1)


def _independent_cyclic_ordinates() -> Tuple[Tuple[float, float], ...]:
    """Generate rounded cyclic ordinates from the independent runtime roots."""

    spin = torch.tensor(KNOT_SPINS, dtype=torch.float64)
    omega_real, omega_damping = signed_qnm_220(spin)
    cyclic = torch.stack((omega_real, omega_damping), dim=-1).detach().cpu()
    cyclic = cyclic / (2.0 * math.pi)
    return tuple(
        tuple(_round_significant(float(value)) for value in row)
        for row in cyclic
    )


# This tuple is the numerical representation consumed by the cubic.  It is
# generated from the independent evaluator above at import time rather than
# copied from a waveform implementation or an external frequency table.
ROUNDED_CYCLIC_ORDINATES: Tuple[Tuple[float, float], ...] = _independent_cyclic_ordinates()


def _natural_cubic_coefficients(
    x: Tuple[float, ...], y: Tuple[Tuple[float, float], ...]
) -> Tuple[Tuple[Tuple[float, float], ...], ...]:
    """Return ``(a,b,c,d)`` coefficients for each natural cubic interval.

    For interval ``i``, the value is
    ``a[i] + b[i] dx + c[i] dx**2 + d[i] dx**3`` with ``dx=x-x[i]``.
    The tridiagonal solve is performed once while importing the fixed,
    rounded representation; runtime evaluation remains tensor-native.
    """

    n = len(x)
    if n < 3 or len(y) != n:
        raise ValueError("natural cubic interpolation needs at least three rows")
    h = [x[i + 1] - x[i] for i in range(n - 1)]
    if any(step <= 0.0 for step in h):
        raise ValueError("cubic knots must be strictly increasing")

    # Solve for the natural-spline second derivatives independently in the
    # real and damping channels.  Endpoints have zero second derivative.
    moments = [[0.0] * n for _ in range(2)]
    for channel in range(2):
        lower = [h[i - 1] for i in range(1, n - 1)]
        diagonal = [2.0 * (h[i - 1] + h[i]) for i in range(1, n - 1)]
        upper = [h[i] for i in range(1, n - 1)]
        rhs = [
            6.0
            * (
                (y[i + 1][channel] - y[i][channel]) / h[i]
                - (y[i][channel] - y[i - 1][channel]) / h[i - 1]
            )
            for i in range(1, n - 1)
        ]
        for i in range(1, len(diagonal)):
            factor = lower[i] / diagonal[i - 1]
            diagonal[i] -= factor * upper[i - 1]
            rhs[i] -= factor * rhs[i - 1]
        solution = [0.0] * (n - 2)
        solution[-1] = rhs[-1] / diagonal[-1]
        for i in range(len(solution) - 2, -1, -1):
            solution[i] = (rhs[i] - upper[i] * solution[i + 1]) / diagonal[i]
        moments[channel][1:-1] = solution

    intervals = []
    for i, width in enumerate(h):
        row = []
        for channel in range(2):
            a = y[i][channel]
            b = (y[i + 1][channel] - y[i][channel]) / width - width * (
                2.0 * moments[channel][i] + moments[channel][i + 1]
            ) / 6.0
            c = moments[channel][i] / 2.0
            d = (moments[channel][i + 1] - moments[channel][i]) / (6.0 * width)
            row.append((a, b, c, d))
        # Store interval-major, coefficient-major, channel-minor values for a
        # single gather in the Torch evaluator.
        intervals.append(tuple(tuple(row[channel][degree] for channel in range(2))
                               for degree in range(4)))
    return tuple(intervals)


NATURAL_CUBIC_COEFFICIENTS = _natural_cubic_coefficients(
    KNOT_SPINS, ROUNDED_CYCLIC_ORDINATES
)


def _working_dtype(spin: torch.Tensor) -> torch.dtype:
    return torch.float32 if spin.device.type == "mps" else torch.float64


@lru_cache(maxsize=8)
def _cubic_tensors(device: torch.device):
    dtype = torch.float32 if device.type == "mps" else torch.float64
    # Constants must stay ordinary tensors so a first inference-mode call does
    # not poison a later autograd call.
    with torch.inference_mode(False):
        return (
            torch.tensor(KNOT_SPINS, dtype=dtype, device=device),
            torch.tensor(NATURAL_CUBIC_COEFFICIENTS, dtype=dtype, device=device),
        )


def _as_spin(final_spin) -> torch.Tensor:
    if isinstance(final_spin, torch.Tensor):
        if not final_spin.is_floating_point():
            raise TypeError("final_spin must have a floating dtype")
        return final_spin
    return torch.as_tensor(final_spin, dtype=torch.float64)


def _validate_spin(spin: torch.Tensor, working_spin: torch.Tensor) -> None:
    if bool(
        torch.any(
            ~torch.isfinite(working_spin)
            | (working_spin < SUPPORTED_SPIN_MIN)
            | (working_spin > SUPPORTED_SPIN_MAX)
        )
    ):
        raise ValueError(
            "D QNM independent representation supports "
            f"[{SUPPORTED_SPIN_MIN}, {SUPPORTED_SPIN_MAX}]; "
            "extremal |spin|=1 roots are not prescribed"
        )


def _evaluate_cubic(spin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the fixed natural cubic on a validated signed-spin tensor."""

    dtype = _working_dtype(spin)
    working_spin = spin.to(dtype)
    knots, coefficients = _cubic_tensors(spin.device)
    interval = (torch.searchsorted(knots, working_spin.contiguous(), right=True) - 1).clamp(
        min=0, max=N_KNOTS - 2
    )
    dx = working_spin - knots[interval]
    selected = coefficients[interval]
    value = selected[..., 3, :] * dx.unsqueeze(-1) + selected[..., 2, :]
    value = value * dx.unsqueeze(-1) + selected[..., 1, :]
    value = value * dx.unsqueeze(-1) + selected[..., 0, :]
    return value[..., 0].to(spin.dtype), value[..., 1].to(spin.dtype)


def independent_cyclic_qnm_220(final_spin) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return high-precision generated ``(M f, M f_damp)`` cyclic values.

    This helper is useful for diagnostics and endpoint controls.  It exposes
    the generated evaluator's native ±0.99999 domain without clamping or
    extrapolating to an unsupported extremal root.
    """

    spin = _as_spin(final_spin)
    working_spin = spin.to(_working_dtype(spin))
    _validate_spin(spin, working_spin)
    real, damping = signed_qnm_220(working_spin)
    scale = 1.0 / (2.0 * math.pi)
    return (real * scale).to(spin.dtype), (damping * scale).to(spin.dtype)


def evaluate_d_qnm_220(final_spin) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the rounded natural-cubic D 220 frequencies in cyclic units.

    The returned pair is ``(M f_RD, M f_damp)`` and preserves shape, dtype,
    device, and spin gradients.  Within ``[-.994, .994]`` this is the fixed
    rounded natural cubic.  Outside that compatibility interior the function
    returns high-precision independent generated roots, through the generator
    limit ``|a|=.99999``.  Exact extremal ``|a|=1`` requests fail explicitly.
    """

    spin = _as_spin(final_spin)
    working_spin = spin.to(_working_dtype(spin))
    _validate_spin(spin, working_spin)
    cubic = _evaluate_cubic(spin)
    generated = independent_cyclic_qnm_220(spin)
    compatibility = (working_spin >= COMPATIBILITY_SPIN_MIN) & (
        working_spin <= COMPATIBILITY_SPIN_MAX
    )
    return tuple(
        torch.where(compatibility, rounded, generated_value)
        for rounded, generated_value in zip(cubic, generated)
    )


def evaluate_d_qnm_220_generated(final_spin) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return generated high-precision cyclic roots for endpoint diagnostics."""

    return independent_cyclic_qnm_220(final_spin)


# Descriptive aliases make the unit convention explicit at call sites.
evaluate_kerr_qnm_220_d = evaluate_d_qnm_220
signed_qnm_220_d = evaluate_d_qnm_220


__all__ = [
    "ROUNDING_SIG_DIGITS",
    "SPIN_STEP",
    "INTERIOR_SPIN_MIN",
    "INTERIOR_SPIN_MAX",
    "COMPATIBILITY_SPIN_MIN",
    "COMPATIBILITY_SPIN_MAX",
    "GENERATED_SPIN_LIMIT",
    "ENDPOINT_POLICY",
    "KNOT_SPINS",
    "N_KNOTS",
    "ROUNDED_CYCLIC_ORDINATES",
    "NATURAL_CUBIC_COEFFICIENTS",
    "independent_cyclic_qnm_220",
    "evaluate_d_qnm_220",
    "evaluate_d_qnm_220_generated",
    "evaluate_kerr_qnm_220_d",
    "signed_qnm_220_d",
]

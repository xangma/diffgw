# SPDX-License-Identifier: MIT
"""Caller-supplied inverse-polynomial intermediate amplitude for XHM.

The paper discusses intermediate-amplitude joining in
`arXiv:2001.10914v2 <https://arxiv.org/abs/2001.10914v2>`_, Sec. V.1.1,
Eq. (37) and Table 4. This helper supplies generic algebra only and does not
assert an unreleased ``122022`` provider schema. Values are the dimensionless
inverse normalized amplitude ``P = A0 / A`` with constant ``A0`` across the
join, and low-level derivatives are with respect to
``u=(f-f_start)/(f_end-f_start)``. No calibration values or region policy are
included; this is scaffolding for an explicitly identified data provider.
"""
from __future__ import annotations

import torch
from math import comb

__all__ = [
    "solve_intermediate_inverse_polynomial",
    "solve_intermediate_normalized_amplitude",
    "evaluate_intermediate_inverse_polynomial",
    "evaluate_intermediate_amplitude",
]


def _real(value, name):
    raw = torch.as_tensor(value)
    if raw.is_complex() or raw.dtype == torch.bool:
        raise ValueError(f"{name} must be finite real values")
    tensor = torch.as_tensor(value, dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all().detach().cpu()):
        raise ValueError(f"{name} must be finite real values")
    return tensor


def solve_intermediate_inverse_polynomial(
    start_value,
    start_derivative,
    interior_values,
    end_value,
    end_derivative,
    *,
    interior_fractions=(1.0 / 3.0, 2.0 / 3.0),
):
    """Solve degree-five ``P(u)`` coefficients from six C1 constraints.

    ``interior_values`` has trailing length two.  All inputs broadcast over
    leading dimensions.  The returned coefficients have trailing order
    ``(c0,...,c5)`` for ``P(u)=sum(c[k]*u**k)``.
    """
    if len(interior_fractions) != 2 or not all(0.0 < float(x) < 1.0 for x in interior_fractions):
        raise ValueError("interior_fractions must contain two fractions strictly between 0 and 1")
    fractions = tuple(float(x) for x in interior_fractions)
    if fractions[0] == fractions[1]:
        raise ValueError("interior_fractions must be distinct")
    values = [_real(x, name) for x, name in (
        (start_value, "start_value"), (start_derivative, "start_derivative"),
        (end_value, "end_value"), (end_derivative, "end_derivative"),
    )]
    interior = _real(interior_values, "interior_values")
    if interior.shape[-1:] != (2,):
        raise ValueError("interior_values must have trailing shape (2,)")
    values.append(interior[..., 0]); values.append(interior[..., 1])
    broadcast = torch.broadcast_tensors(*values)
    s, ds, e, de, i1, i2 = broadcast
    if bool((s <= 0).any()) or bool((e <= 0).any()) or bool((i1 <= 0).any()) or bool((i2 <= 0).any()):
        raise ValueError("inverse normalized amplitudes must be positive")
    rows = []
    rhs = torch.stack((s, ds, i1, i2, e, de), dim=-1)
    zero = torch.zeros_like(s)
    one = torch.ones_like(s)
    rows.append(torch.stack((one, zero, zero, zero, zero, zero), dim=-1))
    rows.append(torch.stack((zero, one, zero, zero, zero, zero), dim=-1))
    for u in fractions:
        rows.append(torch.stack(tuple(torch.full_like(s, u ** k) for k in range(6)), dim=-1))
    rows.append(torch.stack(tuple(torch.ones_like(s) for _ in range(6)), dim=-1))
    rows.append(torch.stack((zero, one, 2*one, 3*one, 4*one, 5*one), dim=-1))
    matrix = torch.stack(rows, dim=-2)
    return torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)


def solve_intermediate_normalized_amplitude(
    start_amplitude,
    start_derivative_df,
    interior_amplitudes,
    end_amplitude,
    end_derivative_df,
    f_start,
    f_end,
    *,
    interior_fractions=(1.0 / 3.0, 2.0 / 3.0),
):
    """Convert caller supplied ``A/A0`` data into inverse-polynomial coefficients.

    Amplitudes are dimensionless ``v=A/A0`` and endpoint derivatives are with
    respect to physical frequency.  Since ``A0`` is constant, the conversion
    is ``P=1/v`` and ``dP/du=-(f_end-f_start)v'/v**2``.
    """
    lo, hi = _real(f_start, "f_start"), _real(f_end, "f_end")
    width = hi - lo
    if bool((width <= 0).any()):
        raise ValueError("require f_end > f_start")
    start = _real(start_amplitude, "start_amplitude")
    end = _real(end_amplitude, "end_amplitude")
    interior = _real(interior_amplitudes, "interior_amplitudes")
    ds = _real(start_derivative_df, "start_derivative_df")
    de = _real(end_derivative_df, "end_derivative_df")
    if interior.shape[-1:] != (2,):
        raise ValueError("interior_amplitudes must have trailing shape (2,)")
    if bool((start <= 0).any()) or bool((end <= 0).any()) or bool((interior <= 0).any()):
        raise ValueError("normalized amplitudes must be positive")
    return solve_intermediate_inverse_polynomial(
        1 / start,
        -width * ds / start.square(),
        1 / interior,
        1 / end,
        -width * de / end.square(),
        interior_fractions=interior_fractions,
    )


def _certify_positive(coefficients):
    """Conservatively certify positivity on ``[0, 1]`` using Bernstein subdivision."""
    power = coefficients.detach()
    degree = power.shape[-1] - 1
    bernstein = torch.stack(tuple(sum(
        power[..., i] * (comb(k, i) / comb(degree, i)) for i in range(k + 1)
    ) for k in range(degree + 1)), dim=-1)
    pending = [bernstein]
    for _depth in range(10):
        unresolved = []
        for segment in pending:
            if bool((segment > 1e-10).all().cpu()):
                continue
            work = segment
            left = [work[..., 0]]
            right = [work[..., -1]]
            for _ in range(degree):
                work = 0.5 * (work[..., :-1] + work[..., 1:])
                left.append(work[..., 0])
                right.append(work[..., -1])
            unresolved.extend((
                torch.stack(left, dim=-1),
                torch.stack(tuple(reversed(right)), dim=-1),
            ))
        if not unresolved:
            return
        pending = unresolved
    raise ValueError("unable to certify inverse amplitude denominator is positive on [0,1]")


def evaluate_intermediate_inverse_polynomial(u, coefficients, *, reject_poles=True):
    """Evaluate ``P(u)``; optionally reject a denominator pole on ``[0,1]``."""
    x = _real(u, "u")
    c = _real(coefficients, "coefficients")
    if c.shape[-1:] != (6,):
        raise ValueError("coefficients must have trailing shape (6,)")
    if bool(((x < 0) | (x > 1)).any()):
        raise ValueError("u must lie in the join interval [0,1]")
    if reject_poles:
        _certify_positive(c)
    shape = torch.broadcast_shapes(x.shape, c.shape[:-1])
    x = x.expand(shape)
    c = c.expand(shape + (6,))
    result = torch.zeros_like(x)
    for coefficient in reversed(tuple(c.unbind(-1))):
        result = result * x + coefficient
    return result


def evaluate_intermediate_amplitude(frequency, f_start, f_end, a0, coefficients, *, reject_poles=True):
    """Evaluate physical amplitude ``A0/P(u)`` for frequencies in the join."""
    f, lo, hi, scale = (_real(x, name) for x, name in (
        (frequency, "frequency"), (f_start, "f_start"), (f_end, "f_end"), (a0, "a0")))
    if bool((hi <= lo).any()) or bool((scale <= 0).any()):
        raise ValueError("require f_end > f_start and positive a0")
    c = _real(coefficients, "coefficients")
    if c.shape[-1:] != (6,):
        raise ValueError("coefficients must have trailing shape (6,)")
    if reject_poles:
        _certify_positive(c)
    shape = torch.broadcast_shapes(f.shape, lo.shape, hi.shape, scale.shape, c.shape[:-1])
    f, lo, hi, scale = (x.expand(shape) for x in (f, lo, hi, scale))
    c = c.expand(shape + (6,))
    u = (f - lo) / (hi - lo)
    if bool(((u < 0) | (u > 1)).any()):
        raise ValueError("frequency must lie in the join interval [f_start,f_end]")
    denominator = evaluate_intermediate_inverse_polynomial(u, c, reject_poles=False)
    if bool((denominator <= 0).any()):
        raise ValueError("inverse amplitude polynomial is non-positive at the requested frequency")
    return scale / denominator

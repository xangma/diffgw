"""Closed-form clean-room evaluation of the TEOBResumS ``P^1_5`` potential.

The published logarithmic PN series is the coefficient sequence used by
``torchwave.approximants.teobresums._teobresums_orbital_potential``.  Matching
``(1+n_1 u)/(1+d_1 u+...+d_5 u^5)`` through ``u^6`` gives a triangular
coefficient reduction to one rational coefficient, so this module avoids a
pointwise 5-by-5 solve.  The radial derivative is analytic, with
``d/dr = -u^2 d/du`` for ``u=1/r``.

This opt-in helper does not alter the default Hamiltonian or PA route.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from torchwave.constants import PI


def _inputs(u, nu) -> tuple[Tensor, Tensor]:
    u_t = torch.as_tensor(u)
    if u_t.is_complex():
        raise TypeError("u must be real")
    if not u_t.is_floating_point():
        u_t = u_t.to(dtype=torch.get_default_dtype())
    if u_t.dtype not in (torch.float32, torch.float64):
        raise TypeError("u must use float32 or float64")
    try:
        nu_probe = torch.as_tensor(nu, device=u_t.device)
    except (TypeError, RuntimeError) as exc:
        raise TypeError("nu must be real float32/float64") from exc
    if nu_probe.is_complex():
        raise TypeError("nu must be real float32/float64")
    try:
        nu_t = torch.as_tensor(nu, dtype=u_t.dtype, device=u_t.device)
    except (TypeError, RuntimeError) as exc:
        raise TypeError("nu must be real float32/float64") from exc
    if u_t.ndim > nu_t.ndim:
        nu_t = nu_t.unsqueeze(-1)
    u_t, nu_t = torch.broadcast_tensors(u_t, nu_t)
    if not bool(torch.isfinite(u_t).all()) or bool((u_t <= 0).any()):
        raise ValueError("u must be finite and positive")
    if not bool(torch.isfinite(nu_t).all()) or bool((nu_t < 0).any()) or bool((nu_t > 0.25).any()):
        raise ValueError("nu must be finite and lie in [0, 0.25]")
    return u_t, nu_t


def _series_coefficients(u: Tensor, nu: Tensor):
    """Return ``c3,c4,c5,c6`` and their ``u`` derivatives."""

    pi = torch.as_tensor(PI, dtype=u.dtype, device=u.device)
    gamma_e = torch.as_tensor(
        0.577215664901532860606512090082402431,
        dtype=u.dtype, device=u.device,
    )
    ln2 = torch.as_tensor(math.log(2.0), dtype=u.dtype, device=u.device)
    a4 = 94.0 / 3.0 - (41.0 / 32.0) * pi.square()
    a50 = (-4237.0 / 60.0 + (2275.0 / 512.0) * pi.square()
           + (256.0 / 5.0) * ln2 + (128.0 / 5.0) * gamma_e)
    a51 = (41.0 / 32.0) * pi.square() - 221.0 / 6.0
    a5c = a50 + a51 * nu
    a5log = torch.as_tensor(64.0 / 5.0, dtype=u.dtype, device=u.device)
    a6c = 3097.3 * nu.square() - 1330.6 * nu + 81.38
    a6log = -7004.0 / 105.0 - (144.0 / 5.0) * nu
    log_u = torch.log(u)
    c3 = 2.0 * nu
    c4 = nu * a4
    c5 = nu * (a5c + a5log * log_u)
    c6 = nu * (a6c + a6log * log_u)
    zero = torch.zeros_like(u)
    dc3 = zero
    dc4 = zero
    dc5 = nu * a5log / u
    dc6 = nu * a6log / u
    return c3, c4, c5, c6, dc3, dc4, dc5, dc6


def _closed_coefficients(u: Tensor, nu: Tensor):
    c3, c4, c5, c6, dc3, dc4, dc5, dc6 = _series_coefficients(u, nu)
    denominator = 32.0 - 12.0 * c3 - 4.0 * c4 - c5
    numerator = c6 - c3.square() + 8.0 * c3 + 4.0 * c4 + 2.0 * c5
    if bool((denominator == 0).any()):
        raise ValueError("P^1_5 coefficient reduction is singular")
    d1 = numerator / denominator
    d2 = 2.0 * d1
    d3 = 4.0 * d1 - c3
    d4 = (8.0 - c3) * d1 - 2.0 * c3 - c4
    d5 = (16.0 - 4.0 * c3 - c4) * d1 - 4.0 * c3 - 2.0 * c4 - c5

    ddenominator = -12.0 * dc3 - 4.0 * dc4 - dc5
    dnumerator = dc6 - 2.0 * c3 * dc3 + 8.0 * dc3 + 4.0 * dc4 + 2.0 * dc5
    dd1 = (dnumerator * denominator - numerator * ddenominator) / denominator.square()
    dd2 = 2.0 * dd1
    dd3 = 4.0 * dd1 - dc3
    dd4 = (8.0 - c3) * dd1 - dc3 * d1 - 2.0 * dc3 - dc4
    dd5 = ((16.0 - 4.0 * c3 - c4) * dd1
           + (-4.0 * dc3 - dc4) * d1 - 4.0 * dc3 - 2.0 * dc4 - dc5)
    return d1, d2, d3, d4, d5, dd1, dd2, dd3, dd4, dd5


def _horner(value_coefficients: tuple[Tensor, ...], derivative_coefficients: tuple[Tensor, ...], u: Tensor):
    value = value_coefficients[-1]
    derivative = derivative_coefficients[-1]
    for coefficient, coefficient_derivative in zip(
        reversed(value_coefficients[:-1]), reversed(derivative_coefficients[:-1])
    ):
        derivative = derivative * u + value + coefficient_derivative
        value = value * u + coefficient
    return value, derivative


def orbital_potential_p15_with_radial_derivative(u, nu) -> tuple[Tensor, Tensor]:
    """Return ``(A_orb^PN, dA_orb^PN/dr)`` for ``u=1/r``.

    The coefficient reduction is the formal ``P^1_5`` of the printed
    5PN-log series.  All logarithmic dependence is retained in the analytic
    derivative.  ``nu=0`` reduces algebraically to the exact Schwarzschild
    expression.
    """

    u, nu = _inputs(u, nu)
    d1, d2, d3, d4, d5, dd1, dd2, dd3, dd4, dd5 = _closed_coefficients(u, nu)
    numerator, numerator_du = _horner(
        (torch.ones_like(u), -2.0 + d1),
        (torch.zeros_like(u), dd1), u,
    )
    denominator, denominator_du = _horner(
        (torch.ones_like(u), d1, d2, d3, d4, d5),
        (torch.zeros_like(u), dd1, dd2, dd3, dd4, dd5), u,
    )
    if bool((denominator == 0).any()):
        raise ValueError("P^1_5 potential denominator is singular")
    potential = numerator / denominator
    dpotential_du = (numerator_du * denominator - numerator * denominator_du) / denominator.square()
    radial_derivative = -u.square() * dpotential_du
    return potential, radial_derivative


def orbital_potential_p15(u, nu) -> Tensor:
    """Return the closed ``P^1_5`` orbital potential at ``u=1/r``."""

    return orbital_potential_p15_with_radial_derivative(u, nu)[0]


def orbital_potential_p15_radial_derivative(u, nu) -> Tensor:
    """Return the analytic radial derivative ``dA/dr`` at ``u=1/r``."""

    return orbital_potential_p15_with_radial_derivative(u, nu)[1]


# Same input convention as the existing private series/Pade entry point.
_teobresums_orbital_potential_closed = orbital_potential_p15
teobresums_orbital_potential_closed = orbital_potential_p15


__all__ = [
    "orbital_potential_p15",
    "orbital_potential_p15_with_radial_derivative",
    "orbital_potential_p15_radial_derivative",
    "teobresums_orbital_potential_closed",
]

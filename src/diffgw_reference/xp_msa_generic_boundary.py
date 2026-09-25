"""High-precision scalar diagnostics for the generic MSA aligned boundary.

This module evaluates the spin cubic and initial elliptic phase from
Chatziioannou et al., arXiv:1703.03967v3, Eqs. (21)--(25) and Appendix B.
It also evaluates the observer/J-frame chart of Pratten et al.,
arXiv:2004.06503v2, Appendix C Eqs. (67)--(85).  It imports neither waveform
frontend and deliberately applies no clipping, artificial root separation,
or aligned-chart fallback.

The cubic diagnostic identifies the two upper ordered roots as the physical
turning points.  That identification is valid for the near-aligned reference
configuration tested here; it is not a prescription through a physical root
reordering such as the up-down instability.  The chart diagnostic reports
``phi_j`` as undefined when the in-plane component of J vanishes instead of
choosing a compatibility gauge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mpmath


SOURCE_CHATZIIOANNOU = "https://arxiv.org/abs/1703.03967v3"
SOURCE_PRATTEN = "https://arxiv.org/html/2004.06503v2"
MTSUN_SI = "4.925490947641267e-6"


@dataclass(frozen=True)
class GenericCubicPhase:
    """Published cubic and inverse-phase quantities at the reference point."""

    coefficients: tuple[Any, Any, Any]
    depressed_coefficients: tuple[Any, Any]
    acos_argument: Any
    roots: tuple[Any, Any, Any]
    physical_root_gap: Any
    total_spin_squared: Any
    phase_fraction: Any
    elliptic_parameter: Any
    initial_phase: Any
    triple_product: Any
    root_residuals: tuple[Any, Any, Any]


@dataclass(frozen=True)
class SourceChart:
    """Appendix-C angles, retaining the exact-aligned azimuth singularity."""

    alpha0: Any
    theta_jn: Any
    zeta: Any
    phi_j: Any | None
    phi_j_defined: bool
    angular_momentum: tuple[Any, Any, Any]


def _context(dps: int):
    if dps < 40:
        raise ValueError("dps must be at least 40 for the boundary diagnostic")
    ctx = mpmath.mp.clone()
    ctx.dps = dps
    return ctx


def _mp(ctx, value):
    """Convert through decimal text so binary floats gain no false precision."""
    return ctx.mpf(str(value))


def _vector(ctx, values):
    if len(values) != 3:
        raise ValueError("each vector must have three components")
    return tuple(_mp(ctx, value) for value in values)


def _dot(left, right):
    return sum(x * y for x, y in zip(left, right))


def _cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(ctx, value):
    return ctx.sqrt(_dot(value, value))


def _scale(value, factor):
    return tuple(factor * component for component in value)


def _add(left, right):
    return tuple(x + y for x, y in zip(left, right))


def generic_cubic_phase(
    mass1,
    mass2,
    spin1,
    spin2,
    *,
    f_ref,
    dps: int = 90,
) -> GenericCubicPhase:
    """Evaluate the reference spin cubic and Eq. (23) inverse phase.

    Masses are solar masses and spins are dimensionless Cartesian vectors in
    the reference-L frame.  Inputs may be decimal strings to preserve a tiny
    transverse perturbation exactly.  The zero triple-product branch takes
    the positive inverse phase, matching the limiting value approached from
    a positive triple product; that sign selection is a declared derived
    convention because Eq. (24) only fixes nonzero signs.
    """
    ctx = _context(dps)
    m1, m2, fr = (_mp(ctx, value) for value in (mass1, mass2, f_ref))
    chi1, chi2 = _vector(ctx, spin1), _vector(ctx, spin2)
    if m1 <= 0 or m2 <= 0 or fr <= 0:
        raise ValueError("masses and f_ref must be positive")
    if m2 > m1:
        m1, m2, chi1, chi2 = m2, m1, chi2, chi1

    total = m1 + m2
    n1, n2 = m1 / total, m2 / total
    q, eta, delta = n2 / n1, n1 * n2, n1 - n2
    s1, s2 = _scale(chi1, n1**2), _scale(chi2, n2**2)
    s1sq, s2sq = _dot(s1, s1), _dot(s2, s2)
    stotal = _add(s1, s2)
    spin_squared = _dot(stotal, stotal)
    xi = (1 + q) * s1[2] + (1 + 1 / q) * s2[2]
    velocity = (ctx.pi * total * _mp(ctx, MTSUN_SI) * fr) ** (ctx.mpf(1) / 3)
    orbital = eta / velocity
    angular_momentum = (stotal[0], stotal[1], orbital + stotal[2])
    j = _norm(ctx, angular_momentum)
    j2, l2 = j**2, orbital**2

    b = ((l2 + s1sq) * q + 2 * orbital * xi - 2 * j2 - s1sq - s2sq
         + (l2 + s2sq) / q)
    c = ((j2 - l2) ** 2 - 2 * orbital * xi * (j2 - l2)
         - 2 * (1 - q) / q * (s1sq - q * s2sq) * l2
         + 4 * eta * l2 * xi**2
         - 2 * delta * (s1sq - s2sq) * xi * orbital
         + 2 * (1 - q) / q * (q * s1sq - s2sq) * j2)
    d = ((1 - q) / q * (s2sq - q * s1sq) * (j2 - l2) ** 2
         + delta**2 / eta * (s1sq - s2sq) ** 2 * l2
         + 2 * delta * orbital * xi * (s1sq - s2sq) * (j2 - l2))
    p = c - b**2 / 3
    depressed_q = 2 * b**3 / 27 - b * c / 3 + d
    if p >= 0:
        raise ValueError("the trigonometric cubic requires p < 0")
    argument = 3 * depressed_q / (2 * p) * ctx.sqrt(-3 / p)
    if not (-1 <= argument <= 1):
        raise ValueError("the high-precision cubic argument is outside [-1, 1]")
    theta = ctx.acos(argument) / 3
    roots = tuple(sorted(
        -b / 3 + 2 * ctx.sqrt(-p / 3) * ctx.cos(theta - 2 * ctx.pi * k / 3)
        for k in range(3)
    ))
    s3, sminus, splus = roots
    gap = splus - sminus
    if gap <= 0 or splus == s3:
        raise ValueError("the reference point does not have distinct turning roots")
    triple = s1[0] * s2[1] - s1[1] * s2[0]
    fraction = (splus - spin_squared) / gap
    # A coplanar reference state is exactly a turning point by Eq. (24): its
    # triple product, and therefore dS^2/dt, vanishes.  Resolve that published
    # identity by selecting the nearest cubic root for evaluating F.  The raw
    # quotient is retained above, including any working-precision overshoot.
    evaluation_fraction = fraction
    if triple == 0:
        nearest = min(range(3), key=lambda index: abs(roots[index] - spin_squared))
        if nearest == 1:
            evaluation_fraction = ctx.mpf(1)
        elif nearest == 2:
            evaluation_fraction = ctx.mpf(0)
        else:
            raise ValueError("the coplanar state lies on the auxiliary cubic root")
    elif not (0 <= fraction <= 1):
        raise ValueError("the reference spin lies outside the selected turning roots")
    parameter = gap / (splus - s3)
    sign = -1 if triple < 0 else 1
    phase = sign * ctx.ellipf(ctx.asin(ctx.sqrt(evaluation_fraction)), parameter)
    residuals = tuple(root**3 + b * root**2 + c * root + d for root in roots)
    return GenericCubicPhase(
        coefficients=(b, c, d),
        depressed_coefficients=(p, depressed_q),
        acos_argument=argument,
        roots=roots,
        physical_root_gap=gap,
        total_spin_squared=spin_squared,
        phase_fraction=fraction,
        elliptic_parameter=parameter,
        initial_phase=phase,
        triple_product=triple,
        root_residuals=residuals,
    )


def source_chart(
    total_spin,
    *,
    orbital_l,
    inclination,
    phi_ref,
    dps: int = 90,
) -> SourceChart:
    """Evaluate Pratten Appendix-C source/J/observer geometry.

    ``total_spin`` and ``orbital_l`` are in any common angular-momentum
    normalization.  Supplying the orbital series is intentionally the
    caller's responsibility, which keeps the published chart separate from
    v223's compatibility-derived orbital-momentum coefficient.
    """
    ctx = _context(dps)
    spin = _vector(ctx, total_spin)
    orbital, iota, phi = (_mp(ctx, value) for value in
                           (orbital_l, inclination, phi_ref))
    if orbital <= 0:
        raise ValueError("orbital_l must be positive")
    angular_momentum = (spin[0], spin[1], orbital + spin[2])
    jnorm = _norm(ctx, angular_momentum)
    if jnorm == 0:
        raise ValueError("the J-frame is undefined at J=0")
    jhat = _scale(angular_momentum, 1 / jnorm)
    observer = (
        ctx.sin(iota) * ctx.sin(phi),
        ctx.sin(iota) * ctx.cos(phi),
        ctx.cos(iota),
    )
    observer_x = (
        -ctx.cos(iota) * ctx.sin(phi),
        -ctx.cos(iota) * ctx.cos(phi),
        ctx.sin(iota),
    )
    yraw = _cross(jhat, observer)
    ynorm = _norm(ctx, yraw)
    if ynorm == 0:
        raise ValueError("the observer/J tangent frame is undefined")
    yj = _scale(yraw, 1 / ynorm)
    xj = _cross(yj, jhat)
    alpha0 = ctx.atan2(yj[2], xj[2])
    theta = ctx.atan2(_norm(ctx, _cross(jhat, observer)), _dot(jhat, observer))
    x_sphere = _add(_scale(xj, ctx.cos(theta)), _scale(jhat, -ctx.sin(theta)))
    zeta = ctx.atan2(_dot(observer_x, yj), _dot(observer_x, x_sphere))
    jperp = ctx.sqrt(angular_momentum[0] ** 2 + angular_momentum[1] ** 2)
    phi_j = ctx.atan2(angular_momentum[1], angular_momentum[0]) if jperp != 0 else None
    return SourceChart(
        alpha0=alpha0,
        theta_jn=theta,
        zeta=zeta,
        phi_j=phi_j,
        phi_j_defined=phi_j is not None,
        angular_momentum=angular_momentum,
    )


__all__ = [
    "GenericCubicPhase",
    "SourceChart",
    "generic_cubic_phase",
    "source_chart",
]

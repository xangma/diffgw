"""Independent scalar reference for PhenomT numerical spin evolution.

The spin pieces follow Sturani, LIGO-T1500554-v23, Eqs. (3), (21), (22),
(24), and (25).  The orbital-axis constraint and carrier-frequency mapping
follow Estellés et al., arXiv:2105.05872v2, Eqs. (12)--(14).  The nonspinning
3PN orbital angular momentum is the deliberate PhenomTP prescription
described in Sec. II.4.1 of that paper.

This module imports no production backend.  It evaluates each PN power with
an isolated mpmath context so tests can distinguish a transcription error in
one component from an integration, frame, or binary-reference discrepancy.
All spins are divided by total mass squared and derivatives are with respect
to ``t/M``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import mpmath


SOURCE_PHENOMTP = "https://arxiv.org/abs/2105.05872v2"
SOURCE_SPIN_EVOLUTION = "https://dcc.ligo.org/public/0122/T1500554/023/dLdS.pdf"

Vector = tuple[mpmath.mpf, mpmath.mpf, mpmath.mpf]
SpinPair = tuple[Vector, Vector]


@dataclass(frozen=True)
class SpinEvolutionReference:
    """High-precision scalar RHS and its source-defined PN components."""

    powers: Mapping[str, SpinPair]
    ds1: Vector
    ds2: Vector
    orbital_angular_momentum: mpmath.mpf
    dlhat: Vector


def _vector(ctx, values) -> Vector:
    values = tuple(values)
    if len(values) != 3:
        raise ValueError("vectors must have exactly three components")
    return tuple(ctx.mpf(str(value)) for value in values)


def _add(*vectors: Vector) -> Vector:
    return tuple(sum(values) for values in zip(*vectors))


def _sub(left: Vector, right: Vector) -> Vector:
    return tuple(a - b for a, b in zip(left, right))


def _scale(value, vector: Vector) -> Vector:
    return tuple(value * item for item in vector)


def _dot(left: Vector, right: Vector):
    return sum(a * b for a, b in zip(left, right))


def _cross(left: Vector, right: Vector) -> Vector:
    ax, ay, az = left
    bx, by, bz = right
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def evaluate_spin_derivatives(
    v,
    lhat,
    spin1,
    spin2,
    mass1_fraction,
    mass2_fraction,
    *,
    quadrupole_kappa1=1,
    quadrupole_kappa2=1,
    include_n3lo_ss: bool = True,
    dps: int = 80,
) -> SpinEvolutionReference:
    """Evaluate the published scalar RHS and retain every PN-power piece.

    ``mass1_fraction + mass2_fraction`` must equal one.  Setting
    ``include_n3lo_ss=False`` removes only Sturani Eq. (24), the v8
    quadratic-spin term omitted by the production PhenomTP prescription.
    """
    if dps < 30:
        raise ValueError("dps must be at least 30")
    ctx = mpmath.mp.clone()
    ctx.dps = dps
    v = ctx.mpf(str(v))
    m1 = ctx.mpf(str(mass1_fraction))
    m2 = ctx.mpf(str(mass2_fraction))
    kappa1 = ctx.mpf(str(quadrupole_kappa1))
    kappa2 = ctx.mpf(str(quadrupole_kappa2))
    lhat = _vector(ctx, lhat)
    spin1 = _vector(ctx, spin1)
    spin2 = _vector(ctx, spin2)
    if v <= 0 or m1 <= 0 or m2 <= 0:
        raise ValueError("v and mass fractions must be positive")
    # This reference also accepts ordinary binary64 inputs from independent
    # ODE drivers.  Requiring those decimal strings to satisfy an 80-digit
    # normalization check would reject every evolved unit vector after its
    # first floating-point step, without adding mathematical protection.
    tolerance = max(ctx.mpf(10) ** (-(dps - 5)), ctx.mpf("1e-12"))
    if abs(m1 + m2 - 1) > tolerance:
        raise ValueError("mass fractions must sum to one")
    if abs(_dot(lhat, lhat) - 1) > tolerance:
        raise ValueError("lhat must be a unit vector")

    l_cross_s1 = _cross(lhat, spin1)
    l_cross_s2 = _cross(lhat, spin2)
    s1_cross_s2 = _cross(spin1, spin2)
    l1 = _dot(lhat, spin1)
    l2 = _dot(lhat, spin2)

    c1 = ctx.mpf(3) / 2 - m1 - m1**2 / 2
    c2 = ctx.mpf(3) / 2 - m2 - m2**2 / 2
    v5 = (_scale(v**5 * c1, l_cross_s1), _scale(v**5 * c2, l_cross_s2))

    v6_1 = _add(
        _scale(-ctx.mpf(1) / 2, s1_cross_s2),
        _scale(-ctx.mpf(3) / 2 * l2, l_cross_s1),
        _scale(ctx.mpf(3) / 2 * kappa1 * (1 - 1 / m1) * l1, l_cross_s1),
    )
    v6_2 = _add(
        _scale(ctx.mpf(1) / 2, s1_cross_s2),
        _scale(-ctx.mpf(3) / 2 * l1, l_cross_s2),
        _scale(ctx.mpf(3) / 2 * kappa2 * (1 - 1 / m2) * l2, l_cross_s2),
    )
    v6 = (_scale(v**6, v6_1), _scale(v**6, v6_2))

    q1 = ctx.mpf(9) / 8 - m1 / 2 + 7 * m1**2 / 12 - 7 * m1**3 / 6 - m1**4 / 24
    q2 = ctx.mpf(9) / 8 - m2 / 2 + 7 * m2**2 / 12 - 7 * m2**3 / 6 - m2**4 / 24
    v7 = (_scale(v**7 * q1, l_cross_s1), _scale(v**7 * q2, l_cross_s2))

    a1 = (9 / (4 * m1) - ctx.mpf(13) / 4 + 3 * m1 / 4 + m1**2 / 4
          + kappa1 * (-9 / (4 * m1) + 3 * m1 / 2 + 3 * m1**2 / 4))
    a2 = (9 / (4 * m2) - ctx.mpf(13) / 4 + 3 * m2 / 4 + m2**2 / 4
          + kappa2 * (-9 / (4 * m2) + 3 * m2 / 2 + 3 * m2**2 / 4))
    b1, b2 = m1 / 2 + m1**2 / 2, m2 / 2 + m2**2 / 2
    d1 = -ctx.mpf(3) / 2 - 3 * m1 / 2 - m1**2 / 2
    d2 = -ctx.mpf(3) / 2 - 3 * m2 / 2 - m2**2 / 2
    v8_1 = _cross(_add(_scale(a1 * l1, lhat), _scale(b1, spin2), _scale(d1 * l2, lhat)), spin1)
    v8_2 = _cross(_add(_scale(a2 * l2, lhat), _scale(b2, spin1), _scale(d2 * l1, lhat)), spin2)
    v8_scale = v**8 if include_n3lo_ss else ctx.mpf(0)
    v8 = (_scale(v8_scale, v8_1), _scale(v8_scale, v8_2))

    r1 = (ctx.mpf(27) / 16 - 51 * m1 / 8 + 181 * m1**2 / 16 - 23 * m1**3 / 6
          - 39 * m1**4 / 16 - 3 * m1**5 / 8 - m1**6 / 48)
    r2 = (ctx.mpf(27) / 16 - 51 * m2 / 8 + 181 * m2**2 / 16 - 23 * m2**3 / 6
          - 39 * m2**4 / 16 - 3 * m2**5 / 8 - m2**6 / 48)
    v9 = (_scale(v**9 * r1, l_cross_s1), _scale(v**9 * r2, l_cross_s2))

    powers = MappingProxyType({"v5": v5, "v6": v6, "v7": v7, "v8": v8, "v9": v9})
    ds1 = _add(*(piece[0] for piece in powers.values()))
    ds2 = _add(*(piece[1] for piece in powers.values()))
    eta = m1 * m2
    orbital = eta / v * (
        1 + (ctx.mpf(3) / 2 + eta / 6) * v**2
        + (ctx.mpf(27) / 8 - 19 * eta / 8 + eta**2 / 24) * v**4
        + (ctx.mpf(135) / 16 + (-ctx.mpf(6889) / 144 + 41 * ctx.pi**2 / 24) * eta
           + 31 * eta**2 / 24 + 7 * eta**3 / 1296) * v**6
    )
    total = _add(ds1, ds2)
    projected = _sub(total, _scale(_dot(lhat, total) / _dot(lhat, lhat), lhat))
    dlhat = _scale(-1 / orbital, projected)
    return SpinEvolutionReference(powers, ds1, ds2, orbital, dlhat)


__all__ = [
    "SOURCE_PHENOMTP",
    "SOURCE_SPIN_EVOLUTION",
    "SpinEvolutionReference",
    "evaluate_spin_derivatives",
]

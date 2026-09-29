"""Source-first, scalar six-mode IMRPhenomHM positive-frequency projection.

London et al. (2018), Eq. (1), supplies the spin -2 expansion. García-Quirós
et al. (2020), Eqs. (2), (4)-(9), fix the positive-frequency mode-pair
convention. The Wigner matrix is independent of both frontend polynomial
projection helpers; the intrinsic mode carrier is deliberately absent.
"""

from __future__ import annotations

from functools import lru_cache

import mpmath as mp
import sympy as sp
from sympy.physics.wigner import wigner_d_small


SPEC_VERSION = "hm.projection.london-eq1.v1"
MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
IOTA, AZIMUTH = sp.symbols("iota azimuth", real=True)


@lru_cache(maxsize=None)
def _small_d(l: int) -> sp.Matrix:
    return wigner_d_small(l, IOTA)


@lru_cache(maxsize=None)
def harmonic(l: int, m: int) -> sp.Expr:
    """Spin -2 Y_lm, normalized by Y_22(0,0)=sqrt(5/(4*pi))."""
    if (l, abs(m)) not in MODES:
        raise ValueError(f"unsupported HM harmonic {(l, m)}")
    return (
        sp.Integer(-1) ** m
        * sp.sqrt(sp.Rational(2 * l + 1, 4) / sp.pi)
        * _small_d(l)[l - m, l - 2]
        * sp.exp(sp.I * m * AZIMUTH)
    )


@lru_cache(maxsize=None)
def paper_coefficients(l: int, m: int) -> tuple[sp.Expr, sp.Expr]:
    """García-Quirós Eqs. (8)-(9) on the positive-frequency h_{l,-m}."""
    if (l, m) not in MODES:
        raise ValueError(f"unsupported HM mode {(l, m)}")
    negative = harmonic(l, -m)
    partner = (-1) ** l * sp.conjugate(harmonic(l, m))
    return (negative + partner) / 2, sp.I * (negative - partner) / 2


@lru_cache(maxsize=None)
def native_carrier_coefficients(l: int, m: int) -> tuple[sp.Expr, sp.Expr]:
    """Paper pair adapted to native P_lm=(-1)^l N_lm, phi0=-azimuth."""
    return tuple(
        (-1) ** l * expression.subs(AZIMUTH, -AZIMUTH)
        for expression in paper_coefficients(l, m)
    )


@lru_cache(maxsize=None)
def _numeric_coefficients(l: int, m: int):
    return tuple(
        sp.lambdify((IOTA, AZIMUTH), expression, modules="mpmath")
        for expression in native_carrier_coefficients(l, m)
    )


def evaluate(l: int, m: int, inclination: float, phi0: float) -> tuple[complex, complex]:
    """Evaluate plus/cross multipliers with 50-digit mpmath arithmetic."""
    with mp.workdps(50):
        angle = mp.mpf(str(inclination))
        azimuth = mp.mpf(str(phi0))
        return tuple(complex(function(angle, azimuth))
                     for function in _numeric_coefficients(l, m))

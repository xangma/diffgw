"""Scalar positive-frequency XHM mode-pair projection from the v2 paper.

No intrinsic carrier, calibration fit, or native frontend is imported here.
The Wigner matrix constructs the spin -2 harmonics independently of the
polynomial branches in the Torch and JAX projection helpers.
"""

from __future__ import annotations

from functools import lru_cache

import sympy as sp
from sympy.physics.wigner import wigner_d_small


SPEC_VERSION = "xhm.positive-frequency.mode-pair-projection.v1"
MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
IOTA, PHI0 = sp.symbols("iota phi0", real=True)


@lru_cache(maxsize=None)
def _wigner(l: int) -> sp.Matrix:
    return wigner_d_small(l, IOTA)


def harmonic(l: int, m: int) -> sp.Expr:
    """Spin -2 harmonic with Y_22=sqrt(5/64pi)(1+cos iota)^2 exp(2i phi0)."""
    if (l, abs(m)) not in MODES:
        raise ValueError(f"unsupported XHM harmonic {(l, m)}")
    return (sp.Integer(-1) ** m * sp.sqrt(sp.Rational(2 * l + 1, 4) / sp.pi)
            * _wigner(l)[l - m, l - 2] * sp.exp(sp.I * m * PHI0))


def coefficients(l: int, m: int) -> tuple[sp.Expr, sp.Expr]:
    """Paper Eqs. (8)-(9) multiplying the positive-frequency h_{l,-m}."""
    if (l, m) not in MODES:
        raise ValueError(f"unsupported XHM mode {(l, m)}")
    y_negative = harmonic(l, -m)
    partner = (-1) ** l * sp.conjugate(harmonic(l, m))
    return (y_negative + partner) / 2, sp.I * (y_negative - partner) / 2


def evaluate(l: int, m: int, inclination: float, phi0: float) -> tuple[complex, complex]:
    """Evaluate the two coefficients at a real viewing angle."""
    substitutions = {IOTA: sp.Rational(str(inclination)), PHI0: sp.Rational(str(phi0))}
    return tuple(complex(sp.N(expr.subs(substitutions), 40))
                 for expr in coefficients(l, m))

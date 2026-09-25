# SPDX-License-Identifier: MIT
"""Independent scalar reference for the XHM inspiral carrier rescaling.

This module implements only García-Quirós et al., arXiv:2001.10914v2,
Eq. (29) and Appendix D, Eqs. (61)--(63).  It does not include the
mode-dependent phase correction, calibrated linear term, or phase offset in
Eqs. (30)--(34), and therefore is not a complete XHM inspiral phase.
"""

from __future__ import annotations

import sympy as sp


SPEC_VERSION = "imrphenomxhm-inspiral-carrier/v1"
REFERENCE_VERSION = SPEC_VERSION
COMPONENT_IDS = {
    "phase_rescaling": "xhm.phase.inspiral.carrier-rescaling.v1",
    "coefficient_rescaling": "xhm.phase.inspiral.carrier-coefficients.v1",
}

X = sp.symbols("x", positive=True)
MODE_M = sp.symbols("m", integer=True, positive=True)
INDEX = sp.symbols("i", integer=True)
C22, D22 = sp.symbols("c22 d22", real=True)


def phase_term(x=X, index=INDEX, c22=C22, d22=D22) -> sp.Expr:
    """Return one Appendix-D 22-phase term from Eq. (61)."""

    x = sp.sympify(x)
    index = sp.sympify(index)
    return x ** ((index - 5) / sp.Integer(3)) * (
        sp.sympify(c22) + sp.sympify(d22) * sp.log(x)
    )


def rescaled_carrier_phase(carrier_phase_at_scaled_frequency, mode_m=MODE_M) -> sp.Expr:
    """Apply the Eq. (29) ``m/2`` phase rescaling."""

    return sp.Rational(1, 2) * sp.sympify(mode_m) * sp.sympify(
        carrier_phase_at_scaled_frequency
    )


def rescaled_coefficients(
    index=INDEX, mode_m=MODE_M, c22=C22, d22=D22
) -> tuple[sp.Expr, sp.Expr]:
    """Return ``(c_m^i, d_m^i)`` from Appendix-D Eq. (63)."""

    index = sp.sympify(index)
    ratio = sp.sympify(mode_m) / 2
    scale = ratio ** ((8 - index) / sp.Integer(3))
    return (
        sp.simplify(scale * (sp.sympify(c22) - sp.sympify(d22) * sp.log(ratio))),
        sp.simplify(scale * sp.sympify(d22)),
    )


def rescaled_term_from_coefficients(
    x=X, index=INDEX, mode_m=MODE_M, c22=C22, d22=D22
) -> sp.Expr:
    """Evaluate the phase term after applying the Eq. (63) coefficients."""

    cm, dm = rescaled_coefficients(index, mode_m, c22, d22)
    return phase_term(x, index, cm, dm)


def directly_rescaled_term(
    x=X, index=INDEX, mode_m=MODE_M, c22=C22, d22=D22
) -> sp.Expr:
    """Evaluate Eq. (29) directly at the rescaled ``22`` frequency."""

    ratio = sp.sympify(mode_m) / 2
    return sp.simplify(ratio * phase_term(sp.sympify(x) / ratio, index, c22, d22))


__all__ = [
    "COMPONENT_IDS",
    "REFERENCE_VERSION",
    "SPEC_VERSION",
    "directly_rescaled_term",
    "phase_term",
    "rescaled_carrier_phase",
    "rescaled_coefficients",
    "rescaled_term_from_coefficients",
]

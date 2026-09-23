"""Scalar references for the printed and observed XAS quartic conventions.

Pratten et al., arXiv:2001.11412v1, Eqs. (67), (71), (79)-(83), Table 2.
The observed convention is separately identified: it agrees with the pinned
LALSimulation binary but is not implied by the printed Eq. (71). The paper's
fifth-order option, parameter-space fits, and transition frequencies are
outside this component. No frontend is imported.
"""

from __future__ import annotations

import sympy as sp

SPEC_VERSION = "imrphenomxas-intermediate-amplitude-quartic/v1"
SOURCE = "https://arxiv.org/html/2001.11412v1"
SOURCES = {
    "pratten2020v1": {
        "url": SOURCE,
        "arxiv": "2001.11412v1",
        "equations": ("67", "71", "79", "80", "81", "82", "83", "Table 2"),
    }
}
COMPONENT_IDS = {
    "quartic": "phenomxas.amp.intermediate.quartic.v1",
    "newtonian_factor": "phenomxas.amp.intermediate.newtonian_factor.v1",
    "collocation": "phenomxas.amp.intermediate.collocation.v1",
}
COMPONENT_SOURCES = {
    COMPONENT_IDS["quartic"]: ("pratten2020v1", "71"),
    COMPONENT_IDS["newtonian_factor"]: ("pratten2020v1", "67"),
    COMPONENT_IDS["collocation"]: (
        "pratten2020v1", "79", "80", "81", "82", "83", "Table 2"
    ),
}
OBSERVED_CONVENTION = "phenomxas.amp.intermediate.constant_normalization.v1"
OBSERVED_EVIDENCE = {
    OBSERVED_CONVENTION: {
        "kind": "binary_oracle",
        "distribution": "LALSuite",
        "version": "7.21",
        "module": "LALSimulation",
        "module_version": "5.3.1",
        "api": "SimInspiralChooseFDWaveform",
        "approximant": "IMRPhenomXAS",
        "frontend_class": "XASCarrierState",
        "frontend_method": "eval_waveform",
        "test": "test_external_binary_intermediate_amplitude_fixture",
    }
}

F, F1, F3, K = sp.symbols("f f_1 f_3 K", positive=True)
C = sp.symbols("c_0:5", real=True)


def inverse_polynomial(f=F, f1=F1, f3=F3, coefficients=C):
    """Eq. (71) denominator in the conditioned coordinate t=(f-f1)/(f3-f1)."""
    t = (sp.sympify(f) - f1) / (f3 - f1)
    return sum(coefficient * t**degree for degree, coefficient in enumerate(coefficients))


def normalized_amplitude(f=F, f1=F1, f3=F3, coefficients=C):
    """Inverse-quartic shape 1/P: A/A0 in print, or A/K as observed."""
    return 1 / inverse_polynomial(f, f1, f3, coefficients)


def newtonian_factor(f=F, k=K):
    """Eq. (67), with its frequency-independent normalization supplied as k."""
    return sp.sympify(k) * sp.sympify(f) ** -sp.Rational(7, 6)


def paper_amplitude(f=F, f1=F1, f3=F3, k=K, coefficients=C):
    """Eq. (71): A0(f) times the inverse quartic."""
    return newtonian_factor(f, k) * normalized_amplitude(f, f1, f3, coefficients)


def observed_amplitude(f=F, f1=F1, f3=F3, k=K, coefficients=C):
    """Binary-checked XAS convention A=K/P, with K independent of frequency.

    This is an implementation/behavioral reference, not a derivation of the
    paper's printed Eq. (71). The coefficients retain the same inverse-quartic
    algebra, but the represented physical amplitude is a different family.
    """
    return sp.sympify(k) * normalized_amplitude(f, f1, f3, coefficients)


def collocation_coefficients(f1, f3, v1, d1, v2, v3, d3):
    """Solve the five Table-2 constraints on B=1/P and dB/df.

    Inputs are exact scalar values. f is dimensionless Mf, f3>f1>0, and all
    amplitudes are positive. Derivatives are with respect to f, not t. This
    B denotes the quantity interpolated by the inverse quartic. Its physical
    normalization must be supplied by the caller's chosen convention.
    """
    f1, f3, v1, d1, v2, v3, d3 = map(
        sp.sympify, (f1, f3, v1, d1, v2, v3, d3)
    )
    if f1.is_number and f3.is_number and not (0 < f1 < f3):
        raise ValueError("require 0 < f1 < f3")
    for value in (v1, v2, v3):
        if value.is_number and value <= 0:
            raise ValueError("collocation amplitudes must be positive")
    span = f3 - f1
    matrix = sp.Matrix([
        [1, 0, 0, 0, 0],
        [1, sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 8), sp.Rational(1, 16)],
        [1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 2, 3, 4],
    ])
    rhs = sp.Matrix([
        1 / v1,
        1 / v2,
        1 / v3,
        -span * d1 / v1**2,
        -span * d3 / v3**2,
    ])
    return tuple(matrix.LUsolve(rhs))


def physical_collocation_coefficients(f1, f3, a1, da1, a2, a3, da3, k):
    """Map paper amplitude values/slopes to normalized collocation data.

    Since A0=k*f^(-7/6), B=A/A0 and
    B'=(A'+7*A/(6*f))/A0. The midpoint is (f1+f3)/2.
    """
    f1, f3, a1, da1, a2, a3, da3, k = map(
        sp.sympify, (f1, f3, a1, da1, a2, a3, da3, k)
    )
    f2 = (f1 + f3) / 2
    b1 = a1 / newtonian_factor(f1, k)
    b2 = a2 / newtonian_factor(f2, k)
    b3 = a3 / newtonian_factor(f3, k)
    db1 = (da1 + sp.Rational(7, 6) * a1 / f1) / newtonian_factor(f1, k)
    db3 = (da3 + sp.Rational(7, 6) * a3 / f3) / newtonian_factor(f3, k)
    return collocation_coefficients(f1, f3, b1, db1, b2, b3, db3)


def observed_collocation_coefficients(f1, f3, a1, da1, a2, a3, da3, k):
    """Solve the observed convention from physical A and dA/d(Mf) nodes.

    Here B=A/K, so B'=A'/K. In the paper convention B=A/(K*f^-7/6),
    which requires the extra 7*A/(6*f) term above.
    """
    f1, f3, a1, da1, a2, a3, da3, k = map(
        sp.sympify, (f1, f3, a1, da1, a2, a3, da3, k)
    )
    if k.is_number and k <= 0:
        raise ValueError("constant normalization must be positive")
    return collocation_coefficients(
        f1, f3, a1 / k, da1 / k, a2 / k, a3 / k, da3 / k
    )

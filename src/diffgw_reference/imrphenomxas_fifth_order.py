"""Paper-literal fifth-order IMRPhenomXAS intermediate-amplitude reference.

Pratten et al., arXiv:2001.11412v1, Eqs. (67), (72)-(78), Table 1.
This is reference-only: the current Torch/JAX carriers use the quartic option.
The frequency nodes and collocation values are supplied by the caller; this
module neither imports a frontend nor assumes unpublished calibration fits.
"""

from __future__ import annotations

import sympy as sp

from .imrphenomxas_intermediate import F, K, newtonian_factor

SPEC_VERSION = "imrphenomxas-intermediate-amplitude-fifth-order/v1"
SOURCE = "https://arxiv.org/html/2001.11412v1"
SOURCES = {
    "pratten2020v1": {
        "url": SOURCE,
        "arxiv": "2001.11412v1",
        "equations": ("67", "72", "73", "74", "75", "76", "77", "78", "Table 1"),
    }
}
COMPONENT_IDS = {
    "fifth_order": "phenomxas.amp.intermediate.fifth_order.v1",
    "collocation": "phenomxas.amp.intermediate.fifth_order_collocation.v1",
}
COMPONENT_SOURCES = {
    COMPONENT_IDS["fifth_order"]: ("pratten2020v1", "67", "72"),
    COMPONENT_IDS["collocation"]: (
        "pratten2020v1", "73", "74", "75", "76", "77", "78", "Table 1"
    ),
}

F1, F4 = sp.symbols("f_1 f_4", positive=True)
C = sp.symbols("c_0:6", real=True)


def inverse_polynomial(f=F, f1=F1, f4=F4, coefficients=C):
    """Eq. (72) denominator in t=(f-f1)/(f4-f1), a change of basis."""
    t = (sp.sympify(f) - f1) / (f4 - f1)
    return sum(coefficient * t**degree for degree, coefficient in enumerate(coefficients))


def normalized_amplitude(f=F, f1=F1, f4=F4, coefficients=C):
    """Paper's A/A0 = 1/P, with A0 from Eq. (67)."""
    return 1 / inverse_polynomial(f, f1, f4, coefficients)


def paper_amplitude(f=F, f1=F1, f4=F4, k=K, coefficients=C):
    """Eq. (72) with A0=k*f**(-7/6) and frequency-independent k."""
    return newtonian_factor(f, k) * normalized_amplitude(f, f1, f4, coefficients)


def collocation_coefficients(f1, f2, f3, f4, v1, v2, v3, v4, d1, d4):
    """Solve Eqs. (73)-(78) for B=1/P and its endpoint dB/d(Mf).

    The caller supplies four ordered positive dimensionless frequencies, four
    positive B values, and the two endpoint slopes. No parameter-space fits or
    interpretation of Table 1's node notation are supplied here.
    """
    f1, f2, f3, f4, v1, v2, v3, v4, d1, d4 = map(
        sp.sympify, (f1, f2, f3, f4, v1, v2, v3, v4, d1, d4)
    )
    nodes = (f1, f2, f3, f4)
    if all(node.is_number for node in nodes) and not (0 < f1 < f2 < f3 < f4):
        raise ValueError("require 0 < f1 < f2 < f3 < f4")
    for value in (v1, v2, v3, v4):
        if value.is_number and value <= 0:
            raise ValueError("collocation amplitudes must be positive")
    span = f4 - f1
    t2, t3 = (f2 - f1) / span, (f3 - f1) / span
    values = (v1, v2, v3, v4)
    matrix = sp.Matrix([
        [node**degree for degree in range(6)] for node in (0, t2, t3, 1)
    ] + [
        [0, 1, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5],
    ])
    rhs = sp.Matrix([1 / value for value in values] + [
        -span * d1 / v1**2,
        -span * d4 / v4**2,
    ])
    return tuple(matrix.LUsolve(rhs))


def physical_collocation_coefficients(
    f1, f2, f3, f4, a1, a2, a3, a4, da1, da4, k
):
    """Convert physical paper amplitudes/slopes to B=A/A0 constraints.

    At either endpoint, B'=(A'+7*A/(6*f))/A0. The primes use dimensionless
    Mf; slopes supplied per hertz require a total-mass conversion upstream.
    """
    f1, f2, f3, f4, a1, a2, a3, a4, da1, da4, k = map(
        sp.sympify, (f1, f2, f3, f4, a1, a2, a3, a4, da1, da4, k)
    )
    if k.is_number and k <= 0:
        raise ValueError("Newtonian normalization must be positive")
    frequencies = (f1, f2, f3, f4)
    amplitudes = (a1, a2, a3, a4)
    b1, b2, b3, b4 = (
        amplitude / newtonian_factor(frequency, k)
        for frequency, amplitude in zip(frequencies, amplitudes)
    )
    db1 = (da1 + sp.Rational(7, 6) * a1 / f1) / newtonian_factor(f1, k)
    db4 = (da4 + sp.Rational(7, 6) * a4 / f4) / newtonian_factor(f4, k)
    return collocation_coefficients(
        f1, f2, f3, f4, b1, b2, b3, b4, db1, db4
    )

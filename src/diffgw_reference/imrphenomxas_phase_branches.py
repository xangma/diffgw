"""Scalar XAS intermediate and merger-ringdown phase equations.

Pratten et al., arXiv:2001.11412v1, Eqs. (90), (92)--(97).
Frequencies are dimensionless Mf. Collocation values and fitted parameters are
supplied inputs; no native frontend or LAL code is imported.
"""

from __future__ import annotations

import sympy as sp

SPEC_VERSION = "imrphenomxas-intermediate-ringdown-phase/v1"
SOURCE = "https://arxiv.org/html/2001.11412v1"
SOURCES = {"pratten2020v1": {"url": SOURCE, "arxiv": "2001.11412v1",
                             "equations": ("90", "92", "93", "94", "95", "96", "97")}}
COMPONENT_IDS = {
    "ringdown": "phenomxas.phase.ringdown.ansatz.v1",
    "ringdown_collocation": "phenomxas.phase.ringdown.collocation.v1",
    "intermediate_printed": "phenomxas.phase.intermediate.printed.v1",
    "intermediate_collocation": "phenomxas.phase.intermediate.collocation.v1",
    "intermediate_carrier": "phenomxas.phase.intermediate.carrier_sign.v1",
    "join": "phenomxas.phase.intermediate_ringdown.join.v1",
}
COMPONENT_SOURCES = {
    COMPONENT_IDS["ringdown"]: ("pratten2020v1", "96"),
    COMPONENT_IDS["ringdown_collocation"]: ("pratten2020v1", "96", "97"),
    COMPONENT_IDS["intermediate_printed"]: ("pratten2020v1", "90"),
    COMPONENT_IDS["intermediate_collocation"]: (
        "pratten2020v1", "90", "92", "93", "94", "95"),
    COMPONENT_IDS["intermediate_carrier"]: ("pratten2020v1", "90"),
    COMPONENT_IDS["join"]: ("pratten2020v1", "90", "96"),
}

F, ETA, FRD, FDAMP, APHI = sp.symbols("f eta f_RD f_damp a_phi", real=True)
C_RD, C1, C2, C4, C0 = sp.symbols("c_RD c_1 c_2 c_4 c_0", real=True)
B0, B1, B2, B3, B4 = sp.symbols("b_0 b_1 b_2 b_3 b_4", real=True)


def ringdown_nodes(f_low, f_rd, f_damp):
    """Eq. (97), including its fourth node at the ringdown frequency."""
    low, rd, damp = map(sp.sympify, (f_low, f_rd, f_damp))
    high = rd + sp.Rational(5, 4) * damp
    width = high - low
    return (low, low + (1 - 1 / sp.sqrt(2)) * width / 2,
            low + width / 2, rd, high)


def intermediate_nodes(f_low, f_high):
    """Five Gauss-Chebyshev nodes of Eq. (92)."""
    low, high = map(sp.sympify, (f_low, f_high))
    width = high - low
    return (low, low + (1 - 1 / sp.sqrt(2)) * width / 2,
            low + width / 2, low + (1 + 1 / sp.sqrt(2)) * width / 2, high)


def ringdown_derivative(f=F, eta=ETA, f_rd=FRD, f_damp=FDAMP,
                        coefficients=(C_RD, C1, C2, C4, C0), a_phi=APHI):
    """Eq. (96), with powers -1/3, -2, -4 and an explicit a_phi."""
    f, eta, rd, damp, a = map(sp.sympify, (f, eta, f_rd, f_damp, a_phi))
    c_rd, c1, c2, c4, c0 = map(sp.sympify, coefficients)
    return (c_rd + c1 * f**-sp.Rational(1, 3) + c2 / f**2
            + c4 / f**4 + c0 * a / ((f - rd)**2 + damp**2)) / eta


def ringdown_primitive(f=F, eta=ETA, f_rd=FRD, f_damp=FDAMP,
                       coefficients=(C_RD, C1, C2, C4, C0), a_phi=APHI):
    """Antiderivative of Eq. (96), with zero arbitrary additive constant."""
    f, eta, rd, damp, a = map(sp.sympify, (f, eta, f_rd, f_damp, a_phi))
    c_rd, c1, c2, c4, c0 = map(sp.sympify, coefficients)
    return (c_rd * f + sp.Rational(3, 2) * c1 * f**sp.Rational(2, 3)
            - c2 / f - c4 / (3 * f**3)
            + c0 * a / damp * sp.atan((f - rd) / damp)) / eta


def intermediate_derivative(f=F, eta=ETA, f_rd=FRD, f_damp=FDAMP,
                            coefficients=(B0, B1, B2, B3, B4), c0=C0,
                            a_phi=APHI, *, carrier_sign=False):
    """Eq. (90); carrier_sign selects the opposite native Lorentzian sign."""
    f, eta, rd, damp, a = map(sp.sympify, (f, eta, f_rd, f_damp, a_phi))
    b0, b1, b2, b3, b4 = map(sp.sympify, coefficients)
    sign = 1 if carrier_sign else -1
    return (b0 + b1 / f + b2 / f**2 + b3 / f**3 + b4 / f**4
            + sign * 4 * sp.sympify(c0) * a
            / ((f - rd)**2 + (2 * damp)**2)) / eta


def intermediate_primitive(f=F, eta=ETA, f_rd=FRD, f_damp=FDAMP,
                           coefficients=(B0, B1, B2, B3, B4), c0=C0,
                           a_phi=APHI, *, carrier_sign=False):
    """Antiderivative of Eq. (90), with zero arbitrary additive constant."""
    f, eta, rd, damp, a = map(sp.sympify, (f, eta, f_rd, f_damp, a_phi))
    b0, b1, b2, b3, b4 = map(sp.sympify, coefficients)
    sign = 1 if carrier_sign else -1
    return (b0 * f + b1 * sp.log(f) - b2 / f - b3 / (2 * f**2)
            - b4 / (3 * f**3)
            + sign * 2 * sp.sympify(c0) * a / damp
            * sp.atan((f - rd) / (2 * damp))) / eta


def ringdown_collocation_coefficients(nodes, scaled_derivatives, f_rd, f_damp,
                                      a_phi=1):
    """Recover Eq. (96) coefficients from five supplied eta*phi' values."""
    nodes = tuple(map(sp.sympify, nodes))
    values = tuple(map(sp.sympify, scaled_derivatives))
    if len(nodes) != 5 or len(values) != 5:
        raise ValueError("require five nodes and scaled phase derivatives")
    rd, damp, a = map(sp.sympify, (f_rd, f_damp, a_phi))
    matrix = sp.Matrix([[1, f**-sp.Rational(1, 3), f**-2, f**-4,
                         a / ((f - rd)**2 + damp**2)] for f in nodes])
    return tuple(matrix.LUsolve(sp.Matrix(values)))


def intermediate_collocation_coefficients(nodes, scaled_derivatives,
                                          f_rd, f_damp, c0, a_phi=1,
                                          *, carrier_sign=False):
    """Recover the five Eq. (90) polynomial coefficients given c0*a_phi."""
    nodes = tuple(map(sp.sympify, nodes))
    values = tuple(map(sp.sympify, scaled_derivatives))
    if len(nodes) != 5 or len(values) != 5:
        raise ValueError("require five nodes and scaled phase derivatives")
    rd, damp, a, c0 = map(sp.sympify, (f_rd, f_damp, a_phi, c0))
    sign = 1 if carrier_sign else -1
    matrix = sp.Matrix([[1, 1 / f, 1 / f**2, 1 / f**3, 1 / f**4]
                        for f in nodes])
    rhs = sp.Matrix([value - sign * 4 * c0 * a
                     / ((f - rd)**2 + (2 * damp)**2)
                     for f, value in zip(nodes, values)])
    return tuple(matrix.LUsolve(rhs))


def join_offsets(f_join, intermediate_phase, ringdown_phase):
    """Affine adjustment making the ringdown primitive C1 at f_join."""
    join = sp.sympify(f_join)
    slope = (sp.diff(intermediate_phase, F) - sp.diff(ringdown_phase, F)).subs(F, join)
    value = (intermediate_phase - ringdown_phase).subs(F, join) - slope * join
    return slope, value

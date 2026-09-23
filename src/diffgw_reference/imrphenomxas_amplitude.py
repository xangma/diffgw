"""Independent scalar XAS inspiral and ringdown amplitude equations.

Pratten et al., arXiv:2001.11412v1, Eqs. (52), (59), (67)-(70), (84),
and (119)-(126). Frequencies are dimensionless Mf. No frontend or LAL code
is imported. Fitted pseudo-PN residuals and ringdown parameters are inputs.
"""

from __future__ import annotations

import sympy as sp

SPEC_VERSION = "imrphenomxas-inspiral-ringdown-amplitude/v1"
SOURCE = "https://arxiv.org/html/2001.11412v1"
SOURCES = {
    "pratten2020v1": {
        "url": SOURCE,
        "arxiv": "2001.11412v1",
        "equations": ("52", "59", "67", "68", "69", "70", "84", "119", "120", "121", "122", "123", "124", "125", "126"),
    }
}
COMPONENT_IDS = {
    "pn": "phenomxas.amp.inspiral.pn.v1",
    "pseudo_pn": "phenomxas.amp.inspiral.pseudo_pn.v1",
    "collocation": "phenomxas.amp.inspiral.collocation.v1",
    "ringdown": "phenomxas.amp.ringdown.lorentzian.v1",
    "peak_printed": "phenomxas.amp.ringdown.peak_printed.v1",
    "peak_stationary": "phenomxas.amp.ringdown.peak_stationary.v1",
}
COMPONENT_SOURCES = {
    COMPONENT_IDS["pn"]: ("pratten2020v1", "67", "69", "119", "120", "121", "122", "123", "124", "125", "126"),
    COMPONENT_IDS["pseudo_pn"]: ("pratten2020v1", "68"),
    COMPONENT_IDS["collocation"]: ("pratten2020v1", "52", "68", "70"),
    COMPONENT_IDS["ringdown"]: ("pratten2020v1", "84"),
    COMPONENT_IDS["peak_printed"]: ("pratten2020v1", "59"),
    COMPONENT_IDS["peak_stationary"]: ("pratten2020v1", "84"),
}

F, ETA, CHI1, CHI2, DELTA = sp.symbols("f eta chi_1 chi_2 delta", real=True)
RHO = sp.symbols("rho_1:4", real=True)
FRD, FDAMP, SIGMA, LAM, AR = sp.symbols(
    "f_RD f_damp sigma lambda a_R", real=True
)
Q = sp.Rational


def pn_coefficients(eta=ETA, chi1=CHI1, chi2=CHI2, delta=DELTA):
    """Appendix A.1, Eqs. (120)-(126), in the paper's spin convention."""
    e, a, b, d = map(sp.sympify, (eta, chi1, chi2, delta))
    return (
        sp.Integer(1),
        sp.Integer(0),
        -Q(323, 224) + Q(451, 168) * e,
        a * (Q(27, 16) * (1 + d) - Q(11, 12) * e)
        + b * (Q(27, 16) * (1 - d) - Q(11, 12) * e),
        a**2 * (-Q(81, 64) * (1 + d) + Q(81, 32) * e)
        + b**2 * (-Q(81, 64) * (1 - d) + Q(81, 32) * e)
        + Q(105271, 24192) * e**2 - Q(1975055, 338688) * e
        - Q(27312085, 8128512) - Q(47, 16) * e * a * b,
        a**3 * (d * (Q(3, 16) - Q(3, 16) * e) - Q(9, 16) * e + Q(3, 16))
        + a * (d * (Q(287213, 32256) - Q(2083, 8064) * e)
               - Q(2227, 2016) * e**2 - Q(15569, 1344) * e + Q(287213, 32256))
        + b**3 * (d * (Q(3, 16) * e - Q(3, 16)) - Q(9, 16) * e + Q(3, 16))
        + b * (d * (Q(2083, 8064) * e - Q(287213, 32256))
               - Q(2227, 2016) * e**2 - Q(15569, 1344) * e + Q(287213, 32256))
        - Q(85, 64) * sp.pi + Q(85, 16) * sp.pi * e,
        a * (-Q(17, 12) * sp.pi * d
             + (-Q(133249, 8064) * e**2 - Q(319321, 32256) * e) * b
             + Q(5, 3) * sp.pi * e - Q(17, 12) * sp.pi)
        + a**2 * (d * (-Q(141359, 32256) * e - Q(49039, 14336))
                  + Q(163199, 16128) * e**2 + Q(158633, 64512) * e
                  - Q(49039, 14336))
        + b**2 * (d * (Q(141359, 32256) * e + Q(49039, 14336))
                  + Q(163199, 16128) * e**2 + Q(158633, 64512) * e
                  - Q(49039, 14336))
        + b * (Q(17, 12) * sp.pi * d + Q(5, 3) * sp.pi * e
               - Q(17, 12) * sp.pi)
        - Q(177520268561, 8583708672)
        + (Q(545384828789, 5007163392) - Q(205, 48) * sp.pi**2) * e
        - Q(3248849057, 178827264) * e**2
        + Q(34473079, 6386688) * e**3,
    )


def pn_shape(f=F, eta=ETA, chi1=CHI1, chi2=CHI2, delta=DELTA):
    """A_PN/A0, Eq. (69), with A_i from Appendix A.1."""
    x = sp.pi * sp.sympify(f)
    return sum(c * x**Q(i, 3) for i, c in enumerate(
        pn_coefficients(eta, chi1, chi2, delta)))


def inspiral_shape(f=F, eta=ETA, chi1=CHI1, chi2=CHI2, delta=DELTA, rho=RHO):
    """A_Ins/A0, Eqs. (68)-(69)."""
    x = sp.pi * sp.sympify(f)
    return pn_shape(f, eta, chi1, chi2, delta) + sum(
        sp.sympify(c) * x**Q(7 + i, 3) for i, c in enumerate(rho)
    )


def paper_newtonian_factor(f=F, eta=ETA):
    """Eq. (67), with the paper's dimensionless mode normalization."""
    return sp.sqrt(2 * sp.sympify(eta) / (3 * sp.pi**Q(1, 3))) * sp.sympify(f)**-Q(7, 6)


def paper_inspiral_amplitude(f=F, eta=ETA, chi1=CHI1, chi2=CHI2, delta=DELTA, rho=RHO):
    return paper_newtonian_factor(f, eta) * inspiral_shape(f, eta, chi1, chi2, delta, rho)


def paper_collocation_nodes(f_meco):
    """The printed Eq. (70) nodes, explicitly anchored at f_MECO."""
    f_meco = sp.sympify(f_meco)
    return tuple(k * f_meco for k in (Q(1, 2), Q(3, 4), sp.Integer(1)))


def carrier_transition_frequency(f_meco, f_isco):
    """Eq. (52), the endpoint supplied by current Torch/JAX carriers."""
    return sp.sympify(f_meco) + (sp.sympify(f_isco) - sp.sympify(f_meco)) / 4


def residual_collocation_coefficients(endpoint, residuals):
    """Solve Eq. (68) pseudo-PN terms at fractional nodes of the endpoint.

    Paper Eq. (70) sets endpoint=f_MECO. Current carriers instead supply
    f_T^A from Eq. (52); the algebraic solve is identical but nodes differ.
    """
    endpoint = sp.sympify(endpoint)
    residuals = tuple(map(sp.sympify, residuals))
    if len(residuals) != 3:
        raise ValueError("require three residuals")
    if endpoint.is_number and endpoint <= 0:
        raise ValueError("endpoint must be positive")
    nodes = paper_collocation_nodes(endpoint)
    powers = (Q(7, 3), Q(8, 3), sp.Integer(3))
    matrix = sp.Matrix([[(sp.pi * node)**power for power in powers] for node in nodes])
    return tuple(matrix.LUsolve(sp.Matrix(residuals)))


def ringdown_amplitude(f=F, f_rd=FRD, f_damp=FDAMP, sigma=SIGMA, lam=LAM, a_r=AR):
    """Printed Eq. (84), with no extra Newtonian factor."""
    width = sp.sympify(f_damp) * sp.sympify(sigma)
    shift = sp.sympify(f) - sp.sympify(f_rd)
    return sp.sympify(a_r) * width / (shift**2 + width**2) * sp.exp(-sp.sympify(lam) * shift / width)


def printed_peak(f_rd=FRD, f_damp=FDAMP, sigma=SIGMA, lam=LAM):
    """The paper's printed Eq. (59), retaining its sqrt(1-lambda)."""
    return sp.Abs(sp.sympify(f_rd) + sp.sympify(f_damp) * sp.sympify(sigma)
                  * (sp.sqrt(1 - sp.sympify(lam)) - 1) / sp.sympify(lam))


def stationary_ringdown_peak(f_rd=FRD, f_damp=FDAMP, sigma=SIGMA, lam=LAM):
    """Near-ringdown local maximum of Eq. (84) for 0<lambda<1.

    The sqrt(1-lambda**2) follows by differentiation, unlike printed Eq. (59).
    """
    return sp.sympify(f_rd) + sp.sympify(f_damp) * sp.sympify(sigma) * (
        sp.sqrt(1 - sp.sympify(lam)**2) - 1) / sp.sympify(lam)


def ringdown_scale_at_node(node, value, f_rd=FRD, f_damp=FDAMP, sigma=SIGMA, lam=LAM):
    """Eq. (84) solved for a_R from one supplied amplitude value."""
    unit = ringdown_amplitude(node, f_rd, f_damp, sigma, lam, 1)
    return sp.sympify(value) / unit


def carrier_ringdown_shape(f=F, f_rd=FRD, f_damp=FDAMP, sigma=SIGMA, lam=LAM, a_r=AR):
    """Observed full-carrier frequency dependence; not printed Eq. (84)."""
    return sp.sympify(f)**-Q(7, 6) * ringdown_amplitude(f, f_rd, f_damp, sigma, lam, a_r)

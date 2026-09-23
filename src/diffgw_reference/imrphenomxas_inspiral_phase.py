"""Scalar IMRPhenomXAS inspiral phase from Pratten et al. (2020).

The fitted collocation values are inputs. No frontend or external waveform code
is imported. ``f`` is dimensionless Mf and must be positive.
"""

from __future__ import annotations

import sympy as sp

SPEC_VERSION = "imrphenomxas-inspiral-phase/v1"
SOURCE = "https://arxiv.org/html/2001.11412v1"
SOURCES = {"pratten2020v1": {"url": SOURCE, "arxiv": "2001.11412v1",
                             "equations": ("85", "86", "87", "88", "129-137",
                                           "138", "VII.1")}}
COMPONENT_IDS = {
    "pn": "phenomxas.phase.inspiral.pn.v1",
    "spin_tail": "phenomxas.phase.inspiral.spin_tail_4pn.v1",
    "pseudo_pn": "phenomxas.phase.inspiral.pseudo_pn.v1",
    "collocation": "phenomxas.phase.inspiral.collocation.v1",
    "join": "phenomxas.phase.inspiral.join.v1",
}
COMPONENT_SOURCES = {
    COMPONENT_IDS["pn"]: ("pratten2020v1", "86", "129-137"),
    COMPONENT_IDS["spin_tail"]: ("pratten2020v1", "138"),
    COMPONENT_IDS["pseudo_pn"]: ("pratten2020v1", "85"),
    COMPONENT_IDS["collocation"]: ("pratten2020v1", "87", "88"),
    COMPONENT_IDS["join"]: ("pratten2020v1", "85", "VII.1"),
}

F, ETA, DELTA, CHI_S, CHI_A = sp.symbols("f eta delta chi_s chi_a", real=True)
SIGMA = sp.symbols("sigma_0:5", real=True)
R = sp.Rational


def pn_coefficients(eta=ETA, delta=DELTA, chi_s=CHI_S, chi_a=CHI_A):
    """Appendix A.2, Eqs. (130)-(138), in powers of v=(pi*f)^(1/3).

    Logarithms in the printed coefficients are log(pi*f)=3*log(v).
    The result maps powers 0..8 to frequency-dependent coefficients.
    """
    e, d, s, a = map(sp.sympify, (eta, delta, chi_s, chi_a))
    log_pi_f = sp.log(sp.pi * F)
    c5 = (R(38645, 756) - R(65, 9)*e)*sp.pi
    c5 += d*(-R(732985, 2268)-R(140, 9)*e)*a
    c5 += (-R(732985, 2268)+R(24260, 81)*e+R(340, 9)*e**2)*s
    c6 = (R(11583231236531, 4694215680)-R(6848, 21)*sp.EulerGamma
          -R(640, 3)*sp.pi**2-R(13696, 21)*sp.log(2)
          +(-R(15737765635, 3048192)+R(2255, 12)*sp.pi**2)*e
          +R(76055, 1728)*e**2-R(127825, 1296)*e**3
          +R(2270, 3)*sp.pi*d*a+(R(2270, 3)-520*e)*sp.pi*s
          +(R(75515, 288)-R(263245, 252)*e-480*e**2)*a**2
          +(R(75515, 288)-R(232415, 504)*e+R(1255, 9)*e**2)*s**2
          +(R(75515, 144)-R(8225, 18)*e)*d*a*s)
    c7 = (sp.pi*(R(77096675, 254016)+R(378515, 1512)*e-R(74045, 756)*e**2)
          +d*a*(-R(25150083775, 3048192)+R(26804935, 6048)*e-R(1985, 48)*e**2)
          +s*(-R(25150083775, 3048192)+R(10566655595, 762048)*e
              -R(1042165, 3024)*e**2+R(5345, 36)*e**3)
          +sp.pi*(2240*e-570)*a**2
          -1140*sp.pi*d*a*s
          +(R(14585, 8)-7270*e+80*e**2)*a**2*s
          +(R(14585, 24)-2380*e)*d*a**3
          +sp.pi*(40*e-570)*s**2
          +(R(14585, 8)-R(215, 2)*e)*d*a*s**2
          +(R(14585, 24)-R(475, 6)*e+R(100, 3)*e**2)*s**3)
    c8_base = sp.pi * (d*a*(R(233915, 168)-R(99185, 252)*e)
                         +s*(R(233915, 168)-R(3970375, 2268)*e
                             +R(19655, 189)*e**2))
    return {
        0: sp.S.One,
        1: sp.S.Zero,
        2: R(3715, 756)+R(55, 9)*e,
        3: -16*sp.pi+R(113, 3)*d*a+(R(113, 3)-R(76, 3)*e)*s,
        4: (R(15293365, 508032)+R(27145, 504)*e+R(3085, 72)*e**2
            +(200*e-R(405, 8))*a**2-R(405, 4)*d*a*s
            +(R(5, 2)*e-R(405, 8))*s**2),
        5: c5*(1+log_pi_f),
        6: c6-R(6848, 63)*log_pi_f,
        7: c7,
        8: c8_base*(1-log_pi_f),
    }


def pn_phase(f=F, eta=ETA, delta=DELTA, chi_s=CHI_S, chi_a=CHI_A,
             *, spin_tail=True):
    """Eq. (129): printed 3.5PN sum or canonical carrier with Eq. (138)."""
    f = sp.sympify(f)
    coeffs = pn_coefficients(eta, delta, chi_s, chi_a)
    v = (sp.pi*F)**R(1, 3)
    end = 9 if spin_tail else 8
    expr = 3*sum(coeffs[k]*v**k for k in range(end))/(128*eta*v**5)-sp.pi/4
    return expr.subs(F, f)


def pseudo_pn_phase(f=F, eta=ETA, sigma=SIGMA):
    """Four-coefficient default of Eq. (85), including phase gauge sigma_0."""
    f, eta = map(sp.sympify, (f, eta))
    s0, s1, s2, s3, s4 = map(sp.sympify, sigma)
    return (s0+s1*f+R(3, 4)*s2*f**R(4, 3)
            +R(3, 5)*s3*f**R(5, 3)+s4*f**2/2)/eta


def inspiral_phase(f=F, eta=ETA, delta=DELTA, chi_s=CHI_S,
                   chi_a=CHI_A, sigma=SIGMA, *, spin_tail=True):
    return (pn_phase(f, eta, delta, chi_s, chi_a, spin_tail=spin_tail)
            +pseudo_pn_phase(f, eta, sigma))


def printed_nodes(f_meco):
    """Eq. (88) literally: the two interior nodes omit f_L."""
    low, high = R(13, 5000), R(51, 50)*sp.sympify(f_meco)
    width = high-low
    return low, width/4, 3*width/4, high


def carrier_nodes(f_meco):
    """Native four-node choice with lower-frequency offset restored."""
    low, _, _, high = printed_nodes(f_meco)
    width = high-low
    return low, low+width/4, low+3*width/4, high


def collocation_coefficients(nodes, values):
    """Solve four supplied values of sum_j a_j*f^(j/3), j=0..3."""
    nodes, values = tuple(map(sp.sympify, nodes)), tuple(map(sp.sympify, values))
    if len(nodes) != 4 or len(values) != 4:
        raise ValueError("require four nodes and four values")
    matrix = sp.Matrix([[f**(R(j, 3)) for j in range(4)] for f in nodes])
    return tuple(matrix.LUsolve(sp.Matrix(values)))


def carrier_sigma_from_fits(nodes, values):
    """Native fitted-value normalization; fits are supplied, not derived here."""
    scale = -R(5, 128)/sp.pi**R(5, 3)
    return tuple(scale*a for a in collocation_coefficients(nodes, values))


def join_coefficients(f_join, eta, pn, sigma_2_to_4, middle):
    """Solve sigma_1 and sigma_0 so inspiral and middle meet in C1."""
    f_join, eta = map(sp.sympify, (f_join, eta))
    s2, s3, s4 = map(sp.sympify, sigma_2_to_4)
    slope = sp.diff(middle-pn, F).subs(F, f_join)
    s1 = eta*slope-s2*f_join**R(1, 3)-s3*f_join**R(2, 3)-s4*f_join
    raw = pn+pseudo_pn_phase(F, eta, (0, s1, s2, s3, s4))
    s0 = eta*(middle-raw).subs(F, f_join)
    return s0, s1

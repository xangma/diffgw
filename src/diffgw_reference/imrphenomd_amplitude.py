"""Independent, paper-precision IMRPhenomD scaled-amplitude specification.

Khan et al., arXiv:1508.07253v2, Eqs. (2)-(4), (19)-(26), (29)-(31),
(54)-(60), and Table 5. The seven Table-5 rows below contain *printed*
digits only. Remnant/QNM frequencies are explicit inputs, not independently
specified by this slice. No production module is imported.
"""

from __future__ import annotations

from collections.abc import Mapping

import sympy as sp

SPEC_VERSION = "imrphenomd-amplitude-joins-paper-precision/v1"
SOURCE = "https://arxiv.org/html/1508.07253v2"
SOURCES = {"khan2016": {"url": SOURCE, "arxiv": "1508.07253v2",
                         "equations": ("2", "3", "4", "19", "20", "21", "22", "23",
                                       "24", "25", "26", "29", "30", "31", "54-60", "Table 5")}}

U, ETA, DELTA, CHI_S, CHI_A, CHI_PN = sp.symbols(
    "u eta delta chi_s chi_a chi_PN", real=True)
F_RD, F_DAMP = sp.symbols("f_RD f_damp", positive=True)
GAMMA1, GAMMA2, GAMMA3 = sp.symbols("gamma_1 gamma_2 gamma_3", real=True)
RHO1, RHO2, RHO3 = sp.symbols("rho_1 rho_2 rho_3", real=True)
F1, F3, V1, V2, V3, D1, D3 = sp.symbols(
    "f_1 f_3 v_1 v_2 v_3 d_1 d_3", real=True)

# Independent transcription of the seven amplitude rows of Appendix C, Table 5.
# Order: lambda00, lambda10, lambda01, lambda11, lambda21, lambda02,
# lambda12, lambda22, lambda03, lambda13, lambda23.
PUBLISHED_ROWS = {
    "rho1": "3931.9 -17395.8 3132.38 343966. -1.21626e6 -70698. 1.38391e6 -3.96628e6 -60017.5 803515. -2.09171e6".split(),
    "rho2": "-40105.5 112253. 23561.7 -3.47618e6 1.13759e7 754313. -1.30848e7 3.64446e7 596227. -7.42779e6 1.8929e7".split(),
    "rho3": "83208.4 -191238. -210916. 8.71798e6 -2.69149e7 -1.98898e6 3.0888e7 -8.39087e7 -1.4535e6 1.70635e7 -4.27487e7".split(),
    "v2": "0.814984 2.57476 1.16102 -2.36278 6.77104 0.757078 -2.72569 7.11404 0.176693 -0.797869 2.11624".split(),
    "gamma1": "0.0069274 0.0302047 0.00630802 -0.120741 0.262716 0.00341518 -0.107793 0.27099 0.000737419 -0.0274962 0.0733151".split(),
    "gamma2": "1.01034 0.000899312 0.283949 -4.04975 13.2078 0.103963 -7.02506 24.7849 0.030932 -2.6924 9.60937".split(),
    "gamma3": "1.30816 -0.00553773 -0.0678292 -0.668983 3.40315 -0.0529658 -0.992379 4.82068 -0.00613414 -0.384293 1.75618".split(),
}
PUBLISHED_ROWS = {name: tuple(values) for name, values in PUBLISHED_ROWS.items()}
FIT_NAMES = tuple(PUBLISHED_ROWS)
COMPONENT_IDS = {
    **{name: f"phenomd.amp.fit.{name}.v1" for name in FIT_NAMES},
    "spin_coordinates": "phenomd.amp.spin_coordinates.v1",
    "pn": "phenomd.amp.pn.v1",
    "inspiral": "phenomd.amp.inspiral.v1",
    "merger_ringdown": "phenomd.amp.merger_ringdown.v1",
    "peak": "phenomd.amp.peak.v1",
    "intermediate": "phenomd.amp.intermediate.v1",
}
COMPONENT_SOURCES = {
    **{COMPONENT_IDS[name]: ("khan2016", "31", "Table 5") for name in FIT_NAMES},
    COMPONENT_IDS["spin_coordinates"]: ("khan2016", "2-4"),
    COMPONENT_IDS["pn"]: ("khan2016", "54-60"),
    COMPONENT_IDS["inspiral"]: ("khan2016", "29-30"),
    COMPONENT_IDS["merger_ringdown"]: ("khan2016", "19"),
    COMPONENT_IDS["peak"]: ("khan2016", "20", "project continuation for |gamma2| >= 1"),
    COMPONENT_IDS["intermediate"]: ("khan2016", "21-26", "Table 2"),
}


def exact(value):
    """Interpret a decimal literal as its exact printed rational value."""
    return sp.Rational(str(value))


def spin_coordinates(m1, m2, chi1, chi2):
    """Return eta, delta, chi_s, chi_a, and *unscaled* chi_PN."""
    m1, m2, chi1, chi2 = map(sp.sympify, (m1, m2, chi1, chi2))
    total = m1 + m2
    eta = m1 * m2 / total**2
    delta = (m1 - m2) / total
    chi_s, chi_a = (chi1 + chi2) / 2, (chi1 - chi2) / 2
    chi_eff = (m1 * chi1 + m2 * chi2) / total
    chi_pn = chi_eff - sp.Rational(38, 113) * eta * (chi1 + chi2)
    return eta, delta, chi_s, chi_a, chi_pn


def fit_expr(name, eta=ETA, chi_pn=CHI_PN, *, coefficients=None):
    """Eq. (31), with printed Table-5 row or explicitly supplied coefficients."""
    if coefficients is None:
        coefficients = PUBLISHED_ROWS[name]
    if len(coefficients) != 11:
        raise ValueError("a Table-5 fit needs eleven coefficients")
    c = tuple(map(exact, coefficients))
    x = sp.sympify(chi_pn) - 1
    eta = sp.sympify(eta)
    return (c[0] + c[1] * eta
            + x * (c[2] + c[3] * eta + c[4] * eta**2)
            + x**2 * (c[5] + c[6] * eta + c[7] * eta**2)
            + x**3 * (c[8] + c[9] * eta + c[10] * eta**2))


def pn_coefficients(eta=ETA, delta=DELTA, chi_s=CHI_S, chi_a=CHI_A):
    """Exact re-expanded PN coefficients A_0 through A_6, Eqs. (54)-(60)."""
    e, d, s, a = map(sp.sympify, (eta, delta, chi_s, chi_a))
    q = sp.Rational
    return (
        sp.S.One,
        sp.S.Zero,
        -q(323, 224) + q(451, 168) * e,
        q(27, 8) * d * a + (q(27, 8) - q(11, 6) * e) * s,
        -q(27312085, 8128512) - q(1975055, 338688) * e + q(105271, 24192) * e**2
        + (-q(81, 32) + 8 * e) * a**2 - q(81, 16) * d * a * s
        + (-q(81, 32) + q(17, 8) * e) * s**2,
        -q(85, 64) * sp.pi + q(85, 16) * sp.pi * e
        + d * (q(285197, 16128) - q(1579, 4032) * e) * a
        + (q(285197, 16128) - q(15317, 672) * e - q(2227, 1008) * e**2) * s,
        -q(177520268561, 8583708672)
        + (q(545384828789, 5007163392) - q(205, 48) * sp.pi**2) * e
        - q(3248849057, 178827264) * e**2 + q(34473079, 6386688) * e**3
        + (q(1614569, 64512) - q(1873643, 16128) * e + q(2167, 42) * e**2) * a**2
        + (q(31, 12) * sp.pi - q(7, 3) * sp.pi * e) * s
        + (q(1614569, 64512) - q(61391, 1344) * e + q(57451, 4032) * e**2) * s**2
        + d * a * (q(31, 12) * sp.pi
                   + (q(1614569, 32256) - q(165961, 2688) * e) * s),
    )


def inspiral_expr(u=U, eta=ETA, delta=DELTA, chi_s=CHI_S, chi_a=CHI_A,
                  rho1=RHO1, rho2=RHO2, rho3=RHO3):
    """Dimensionless A_ins/A0; the physical A0 contains f^(-7/6)."""
    u = sp.sympify(u)
    v = (sp.pi * u)**sp.Rational(1, 3)
    return (sum(c * v**i for i, c in enumerate(pn_coefficients(eta, delta, chi_s, chi_a)))
            + rho1 * u**sp.Rational(7, 3) + rho2 * u**sp.Rational(8, 3) + rho3 * u**3)


def merger_ringdown_expr(u=U, f_rd=F_RD, f_damp=F_DAMP,
                         gamma1=GAMMA1, gamma2=GAMMA2, gamma3=GAMMA3):
    """Dimensionless A_MR/A0, Eq. (19)."""
    width = gamma3 * f_damp
    offset = u - f_rd
    return gamma1 * width * sp.exp(-gamma2 * offset / width) / (offset**2 + width**2)


def peak_expr(f_rd=F_RD, f_damp=F_DAMP, gamma2=GAMMA2, gamma3=GAMMA3):
    """Eq. (20) where its square root is real and gamma2 is nonzero."""
    return sp.Abs(f_rd + f_damp * gamma3 * (sp.sqrt(1 - gamma2**2) - 1) / gamma2)


def peak_compatibility(f_rd, f_damp, gamma2, gamma3):
    """Frontend continuation for |gamma2| >= 1; not a paper extremum there."""
    g = exact(gamma2)
    offset = -g / (1 + sp.sqrt(1 - g**2)) if abs(g) < 1 else -1 / g
    return sp.Abs(exact(f_rd) + exact(f_damp) * exact(gamma3) * offset)


def intermediate_expr(u=U, f1=F1, f3=F3, v1=V1, v2=V2, v3=V3, d1=D1, d3=D3):
    """Unique quartic satisfying Eqs. (22)-(26), in a stable Hermite basis."""
    t = (u - f1) / (f3 - f1)
    h = f3 - f1
    cubic = (v1 * (2 * t**3 - 3 * t**2 + 1)
             + v3 * (-2 * t**3 + 3 * t**2)
             + h * d1 * (t**3 - 2 * t**2 + t)
             + h * d3 * (t**3 - t**2))
    midpoint = (v1 + v3) / 2 + h * (d1 - d3) / 8
    return cubic + 16 * (v2 - midpoint) * t**2 * (1 - t)**2


def amplitude_pieces(*, eta, delta, chi_s, chi_a, rho1, rho2, rho3,
                     gamma1, gamma2, gamma3, v2, f_rd, f_damp):
    """Bind calibration and remnant inputs; return pieces, joins, and slopes."""
    ins = inspiral_expr(eta=eta, delta=delta, chi_s=chi_s, chi_a=chi_a,
                        rho1=rho1, rho2=rho2, rho3=rho3)
    mr = merger_ringdown_expr(f_rd=f_rd, f_damp=f_damp,
                              gamma1=gamma1, gamma2=gamma2, gamma3=gamma3)
    f1 = sp.Rational(14, 1000)
    f3 = peak_compatibility(f_rd, f_damp, gamma2, gamma3)
    v1 = ins.subs(U, f1)
    v3 = mr.subs(U, f3)
    d1 = sp.diff(ins, U).subs(U, f1)
    d3 = sp.diff(mr, U).subs(U, f3)
    middle = intermediate_expr(f1=f1, f3=f3, v1=v1, v2=v2,
                               v3=v3, d1=d1, d3=d3)
    return {"inspiral": ins, "intermediate": middle, "merger_ringdown": mr,
            "f1": f1, "f3": f3, "v1": v1, "v2": v2, "v3": v3,
            "d1": d1, "d3": d3}


def evaluate(expression, substitutions: Mapping[sp.Symbol, object] | None = None, *, dps=80):
    """High-precision scalar evaluation without frontend constants/functions."""
    values = {key: exact(value) for key, value in (substitutions or {}).items()}
    return sp.N(sp.sympify(expression).subs(values), dps)

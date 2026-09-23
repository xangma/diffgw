"""Paper-precision IMRPhenomD intrinsic phase and C1 joins.

Khan et al., arXiv:1508.07253v2, Eqs. (13)-(16), (27)-(28), (31),
(37), (41)-(53), and Table 5. The PN coefficients and fitted rows use only
published digits. QNM frequencies are independent inputs. This module imports
no production waveform implementation or coefficients.
"""

from __future__ import annotations

import sympy as sp

from . import imrphenomd_amplitude as amplitude

SPEC_VERSION = "imrphenomd-phase-joins-paper-precision/v1"
SOURCE = "https://arxiv.org/html/1508.07253v2"
SOURCES = {"khan2016": {"url": SOURCE, "arxiv": "1508.07253v2",
                         "equations": ("13-16", "27-28", "31", "37", "41-53", "Table 5")}}

U, ETA, DELTA, CHI_S, CHI_A, CHI_PN = (
    amplitude.U, amplitude.ETA, amplitude.DELTA, amplitude.CHI_S,
    amplitude.CHI_A, amplitude.CHI_PN)
F_RD, F_DAMP = amplitude.F_RD, amplitude.F_DAMP

# Appendix C, Table 5; order is Eq. (31)'s lambda00, lambda10, lambda01,
# lambda11, lambda21, lambda02, lambda12, lambda22, lambda03, lambda13, lambda23.
PUBLISHED_ROWS = {
    "sigma1": "2096.55 1463.75 1312.55 18307.3 -43534.1 -833.289 32047.3 -108609. 452.251 8353.44 -44531.3".split(),
    "sigma2": "-10114.1 -44631. -6541.31 -266959. 686328. 3405.64 -437508. 1.63182e6 -7462.65 -114585. 674402.".split(),
    "sigma3": "22933.7 230960. 14961.1 1.19402e6 -3.10422e6 -3038.17 1.87203e6 -7.30915e6 42738.2 467502. -3.06485e6".split(),
    "sigma4": "-14621.7 -377813. -9608.68 -1.71089e6 4.33292e6 -22366.7 -2.50197e6 1.02745e7 -85360.3 -570025. 4.39684e6".split(),
    "beta1": "97.8975 -42.6597 153.484 -1417.06 2752.86 138.741 -1433.66 2857.74 41.0251 -423.681 850.359".split(),
    "beta2": "-3.2827 -9.05138 -12.4154 55.4716 -106.051 -11.953 76.807 -155.332 -3.41293 25.5724 -54.408".split(),
    "beta3": "-2.51564e-5 1.97503e-5 -1.83707e-5 2.18863e-5 8.25024e-5 7.15737e-6 -5.578e-5 1.91421e-4 5.44717e-6 -3.22061e-5 7.97402e-5".split(),
    "alpha1": "43.3151 638.633 -32.8577 2415.89 -5766.88 -61.8546 2953.97 -8986.29 -21.5714 981.216 -3239.57".split(),
    "alpha2": "-0.0702021 -0.162698 -0.187251 1.13831 -2.83342 -0.17138 1.71975 -4.53972 -0.0499834 0.606207 -1.68277".split(),
    "alpha3": "9.59881 -397.054 16.2021 -1574.83 3600.34 27.0924 -1786.48 5152.92 11.1757 -577.8 1808.73".split(),
    "alpha4": "-0.0298949 1.40221 -0.0735605 0.833701 0.224001 -0.0552029 0.566719 0.718693 -0.0155074 0.157503 0.210768".split(),
    "alpha5": "0.997441 -0.00788445 -0.0590469 1.39587 -4.51663 -0.0558534 1.75166 -5.99021 -0.0179453 0.59651 -2.06089".split(),
}
PUBLISHED_ROWS = {name: tuple(row) for name, row in PUBLISHED_ROWS.items()}
FIT_NAMES = tuple(PUBLISHED_ROWS)
MODEL_FIT_NAMES = tuple(name for name in FIT_NAMES if name != "beta1")
COMPONENT_IDS = {
    **{name: f"phenomd.phase.fit.{name}.v1" for name in FIT_NAMES},
    "pn": "phenomd.phase.pn.v1",
    "inspiral": "phenomd.phase.inspiral.v1",
    "intermediate": "phenomd.phase.intermediate.v1",
    "merger_ringdown": "phenomd.phase.merger_ringdown.v1",
    "joins": "phenomd.phase.joins.v1",
    "gauge": "phenomd.phase.frontend_gauge.v1",
}
COMPONENT_SOURCES = {
    **{COMPONENT_IDS[name]: ("khan2016", "31", "Table 5") for name in FIT_NAMES},
    COMPONENT_IDS["pn"]: ("khan2016", "27", "41-53"),
    COMPONENT_IDS["inspiral"]: ("khan2016", "28"),
    COMPONENT_IDS["intermediate"]: ("khan2016", "15-16"),
    COMPONENT_IDS["merger_ringdown"]: ("khan2016", "13-14"),
    COMPONENT_IDS["joins"]: ("khan2016", "37"),
    COMPONENT_IDS["gauge"]: ("project-convention", "raw fitted alpha1 at amplitude peak"),
}
_TORCH_PHASE = ("torchwave.approximants.imrphenomd", "IMRPhenomD.__call__")
_JAX_PHASE = ("jaxwave.approximants.imrphenomd", "IMRPhenomD.evaluate")
COMPONENT_BINDINGS = {
    **{COMPONENT_IDS[name]: {
        "reference": "fit_expr",
        "torch": ("torchwave.approximants.imrphenomd", "IMRPhenomD.eval_table_v"),
        "jax": ("jaxwave.approximants.imrphenomd", "IMRPhenomD.eval_table_v"),
        "tests": (("test_published_phase_rows_are_pinned_to_table_5",)
                  if name == "beta1" else
                  ("test_published_phase_rows_are_pinned_to_table_5",
                   "test_phase_fits_match_both_frontends_with_explicit_inherited_rows")),
    } for name in FIT_NAMES},
    COMPONENT_IDS["pn"]: {
        "reference": "pn_phase_coefficients",
        "torch": ("torchwave.approximants.taylorf2", "TaylorF2.phase_coefficients"),
        "jax": ("jaxwave.approximants.taylorf2", "TaylorF2.phase_coefficients"),
        "tests": ("test_paper_pn_coefficients_against_both_frontends",),
    },
    **{COMPONENT_IDS[name]: {
        "reference": function, "torch": _TORCH_PHASE, "jax": _JAX_PHASE,
        "tests": ("test_integrated_branches_and_c1_matching",
                  "test_unwrapped_coalescence_phase_across_regions"),
    } for name, function in (("inspiral", "inspiral_expr"),
                             ("intermediate", "intermediate_shape"),
                             ("merger_ringdown", "merger_ringdown_shape"),
                             ("joins", "phase_pieces"))},
    COMPONENT_IDS["gauge"]: {
        "reference": "peak_gauge_slope", "torch": _TORCH_PHASE,
        "jax": _JAX_PHASE,
        "tests": ("test_unwrapped_coalescence_phase_across_regions",
                  "test_reference_frequency_phase_anchor"),
    },
}

exact = amplitude.exact
spin_coordinates = amplitude.spin_coordinates
evaluate = amplitude.evaluate


def fit_expr(name, eta=ETA, chi_pn=CHI_PN, *, coefficients=None):
    """Published phase row in Eq. (31); optional explicit inherited row."""
    row = PUBLISHED_ROWS[name] if coefficients is None else coefficients
    return amplitude.fit_expr(name, eta, chi_pn, coefficients=row)


def pn_phase_coefficients(eta=ETA, delta=DELTA, chi_s=CHI_S, chi_a=CHI_A):
    """Appendix B Eqs. (46)-(53), split into the frontend's log basis.

    Return c0,c2,c3,c4,c5,c5_log,c6,c6_log,c7, where c5_log
    multiplies log(v) and c6_log multiplies log(4v), v=(pi*u)^(1/3).
    """
    e, d, s, a = map(sp.sympify, (eta, delta, chi_s, chi_a))
    r = sp.Rational
    c5 = (r(38645, 756) - r(65, 9)*e)*sp.pi
    c5 += d*(-r(732985, 2268)-r(140, 9)*e)*a
    c5 += (-r(732985, 2268)+r(24260, 81)*e+r(340, 9)*e**2)*s
    c6 = (r(11583231236531, 4694215680)-r(6848, 21)*sp.EulerGamma
          -r(640, 3)*sp.pi**2
          +(-r(15737765635, 3048192)+r(2255, 12)*sp.pi**2)*e
          +r(76055, 1728)*e**2-r(127825, 1296)*e**3
          +r(2270, 3)*sp.pi*d*a+(r(2270, 3)-520*e)*sp.pi*s)
    return (
        sp.S.One,
        r(3715, 756)+r(55, 9)*e,
        -16*sp.pi+r(113, 3)*d*a+(r(113, 3)-r(76, 3)*e)*s,
        r(15293365, 508032)+r(27145, 504)*e+r(3085, 72)*e**2
        +(-r(405, 8)+200*e)*a**2-r(405, 4)*d*a*s
        +(-r(405, 8)+r(5, 2)*e)*s**2,
        c5, 3*c5, c6, -r(6848, 21),
        (r(77096675, 254016)+r(378515, 1512)*e-r(74045, 756)*e**2)*sp.pi
        +d*(-r(25150083775, 3048192)+r(26804935, 6048)*e-r(1985, 48)*e**2)*a
        +(-r(25150083775, 3048192)+r(10566655595, 762048)*e
          -r(1042165, 3024)*e**2+r(5345, 36)*e**3)*s,
    )


def taylorf2_expr(u=U, eta=ETA, delta=DELTA, chi_s=CHI_S, chi_a=CHI_A):
    """Intrinsic TaylorF2 phase of Eq. (27); no time/phase gauge constants."""
    c0,c2,c3,c4,c5,c5log,c6,c6log,c7 = pn_phase_coefficients(eta,delta,chi_s,chi_a)
    v = (sp.pi*sp.sympify(u))**sp.Rational(1,3)
    series = (c0+c2*v**2+c3*v**3+c4*v**4
              +(c5+c5log*sp.log(v))*v**5
              +(c6+c6log*sp.log(4*v))*v**6+c7*v**7)
    return 3*series/(128*eta*v**5)


def inspiral_expr(u=U, eta=ETA, delta=DELTA, chi_s=CHI_S, chi_a=CHI_A,
                  sigma1=None, sigma2=None, sigma3=None, sigma4=None):
    """Eq. (28) with sigma0=0 (arbitrary global phase gauge)."""
    sigma = [sp.Symbol(f"sigma_{i}") if x is None else x
             for i,x in enumerate((sigma1,sigma2,sigma3,sigma4), 1)]
    u = sp.sympify(u)
    return (taylorf2_expr(u,eta,delta,chi_s,chi_a)
            +(sigma[0]*u+sp.Rational(3,4)*sigma[1]*u**sp.Rational(4,3)
              +sp.Rational(3,5)*sigma[2]*u**sp.Rational(5,3)
              +sigma[3]*u**2/2)/eta)


def intermediate_shape(u=U, beta2=None, beta3=None):
    """Non-affine integral in Eq. (16), before C1 matching."""
    b2 = sp.Symbol("beta_2") if beta2 is None else beta2
    b3 = sp.Symbol("beta_3") if beta3 is None else beta3
    return b2*sp.log(u)-b3*u**-3/3


def merger_ringdown_shape(u=U, f_rd=F_RD, f_damp=F_DAMP,
                          alpha2=None, alpha3=None, alpha4=None, alpha5=None):
    """Non-affine integral in Eq. (14), before C1 matching."""
    a2,a3,a4,a5 = (sp.Symbol(f"alpha_{i}") if x is None else x
                   for i,x in zip(range(2,6),(alpha2,alpha3,alpha4,alpha5)))
    return (-a2/u+sp.Rational(4,3)*a3*u**sp.Rational(3,4)
            +a4*sp.atan((u-a5*f_rd)/f_damp))


def phase_pieces(*, eta, delta, chi_s, chi_a, f_rd, f_damp,
                 sigma1, sigma2, sigma3, sigma4, beta2, beta3,
                 alpha2, alpha3, alpha4, alpha5):
    """Eq. (37): solve both affine joins from value and slope continuity."""
    ins = inspiral_expr(eta=eta,delta=delta,chi_s=chi_s,chi_a=chi_a,
                        sigma1=sigma1,sigma2=sigma2,sigma3=sigma3,sigma4=sigma4)
    f1, f2 = sp.Rational(18,1000), f_rd/2
    g_int = intermediate_shape(beta2=beta2,beta3=beta3)
    b1 = eta*sp.diff(ins,U).subs(U,f1)-sp.diff(g_int,U).subs(U,f1)
    b0 = eta*ins.subs(U,f1)-b1*f1-g_int.subs(U,f1)
    middle = (b0+b1*U+g_int)/eta
    g_mr = merger_ringdown_shape(f_rd=f_rd,f_damp=f_damp,alpha2=alpha2,
                                 alpha3=alpha3,alpha4=alpha4,alpha5=alpha5)
    a1 = eta*sp.diff(middle,U).subs(U,f2)-sp.diff(g_mr,U).subs(U,f2)
    a0 = eta*middle.subs(U,f2)-a1*f2-g_mr.subs(U,f2)
    mr = (a0+a1*U+g_mr)/eta
    return {"inspiral":ins,"intermediate":middle,"merger_ringdown":mr,
            "f1":f1,"f2":f2,"beta0":b0,"beta1":b1,"alpha0":a0,"alpha1":a1}


def peak_gauge_slope(*, eta, alpha1_raw, alpha2, alpha3, alpha4, alpha5,
                     f_peak, f_rd, f_damp):
    """Frontend global time gauge: Eq. (13) with raw fitted alpha1 at f_peak.

    This gauge is a project choice; Eq. (37)'s joined MR branch uses the
    continuity-determined alpha1 instead.
    """
    return (alpha1_raw+sp.diff(merger_ringdown_shape(
        f_rd=f_rd,f_damp=f_damp,alpha2=alpha2,alpha3=alpha3,
        alpha4=alpha4,alpha5=alpha5),U).subs(U,f_peak))/eta

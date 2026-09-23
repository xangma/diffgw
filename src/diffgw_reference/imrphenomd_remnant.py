"""Paper-precision IMRPhenomD remnant fits and conditional QNM conversion.

Husa et al., arXiv:1508.07250v2, Eqs. (17), (22)-(24). Decimal
coefficients below contain only digits printed in that paper. The Kerr 220
eigenfrequencies are numerical inputs, not analytic functions supplied by it.
No production module or QNM data is imported.
"""

from __future__ import annotations

import math

import sympy as sp

SPEC_VERSION = "imrphenomd-remnant-qnm-paper-precision/v1"
SOURCE = "https://arxiv.org/html/1508.07250v2"
SOURCES = {"husa2016": {"url": SOURCE, "arxiv": "1508.07250v2",
                        "equations": ("17", "22", "23", "24")},
           "berti_schwarzschild": {
               "url": "https://pages.jh.edu/eberti2/ringdown/s2l2.dat",
               "row": 0, "units": "2M*angular_frequency"}}
COMPONENT_IDS = {
    "coordinates": "phenomd.remnant.coordinates.v1",
    "spin": "phenomd.remnant.spin.v1",
    "energy": "phenomd.remnant.energy.v1",
    "mass": "phenomd.remnant.mass.v1",
    "qnm_scaling": "phenomd.qnm.total_mass_scaling.v1",
    "qnm_interpolation": "phenomd.qnm.natural_cubic_compatibility.v1",
}
COMPONENT_SOURCES = {
    COMPONENT_IDS["coordinates"]: ("husa2016", "Sec. III.1"),
    COMPONENT_IDS["spin"]: ("husa2016", "22"),
    COMPONENT_IDS["energy"]: ("husa2016", "23-24"),
    COMPONENT_IDS["mass"]: ("husa2016", "24", "mass conservation"),
    COMPONENT_IDS["qnm_scaling"]: ("husa2016", "17"),
    COMPONENT_IDS["qnm_interpolation"]: (
        "project numerical compatibility", "berti_schwarzschild author anchor"),
}
_TORCH_REMNANT = ("torchwave.approximants.imrphenomd", "IMRPhenomD.remnant_properties")
_JAX_REMNANT = ("jaxwave.approximants.imrphenomd", "IMRPhenomD.remnant_properties")
COMPONENT_BINDINGS = {
    COMPONENT_IDS["coordinates"]: {
        "reference": "coordinates", "torch": _TORCH_REMNANT, "jax": _JAX_REMNANT,
        "tests": ("test_paper_decimal_transcription_and_limiting_cases",
                  "test_remnant_against_published_and_inherited_equations"),
    },
    COMPONENT_IDS["spin"]: {
        "reference": "final_spin_expr", "torch": _TORCH_REMNANT, "jax": _JAX_REMNANT,
        "tests": ("test_remnant_against_published_and_inherited_equations",
                  "test_remnant_spin_and_qnm_spin_derivative"),
    },
    COMPONENT_IDS["energy"]: {
        "reference": "radiated_fraction_expr", "torch": _TORCH_REMNANT, "jax": _JAX_REMNANT,
        "tests": ("test_remnant_against_published_and_inherited_equations",
                  "test_remnant_spin_and_qnm_spin_derivative"),
    },
    COMPONENT_IDS["mass"]: {
        "reference": "final_mass_expr", "torch": _TORCH_REMNANT, "jax": _JAX_REMNANT,
        "tests": ("test_remnant_against_published_and_inherited_equations",
                  "test_remnant_spin_and_qnm_spin_derivative"),
    },
    COMPONENT_IDS["qnm_scaling"]: {
        "reference": "qnm_total_mass_expr", "torch": _TORCH_REMNANT,
        "jax": _JAX_REMNANT, "tests": ("test_qnm_interpolation_and_mass_units",
                                          "test_remnant_qnm_composition"),
    },
    COMPONENT_IDS["qnm_interpolation"]: {
        "reference": "natural_cubic_qnm",
        "torch": ("torchwave.approximants.kerr_qnm_d_sampling_clean", "evaluate_d_qnm_220"),
        "jax": ("jaxwave.approximants.imrphenomd", "evaluate_d_qnm_220"),
        "tests": ("test_qnm_interpolation_and_mass_units",
                  "test_qnm_schwarzschild_against_author_published_datum",
                  "test_qnm_data_against_strict_eigenproblem_holdouts"),
    },
}

M1, M2, CHI1, CHI2 = sp.symbols("m_1 m_2 chi_1 chi_2", real=True)
ETA, S, S_HAT = sp.symbols("eta S S_hat", real=True)
QNM_REAL, QNM_DAMP = sp.symbols("M_f_f_RD M_f_f_damp", positive=True)

# Eq. (22): eta, eta^2, eta^3, eta^4; then S, S^2, S^3, S^4
# in the eta and eta^2 rows. The 2*sqrt(3) limit is exact.
SPIN_ETA_PRINTED = (None, "-4.399", "9.397", "-13.181")
SPIN_S_ETA_PRINTED = ("-0.085", "0.102", "-1.355", "-0.868")
SPIN_S_ETA2_PRINTED = ("-5.837", "-2.097", "4.109", "2.064")
ENERGY_NS_PRINTED = ("0.0559745", "0.580951", "-0.960673", "3.35241")
ENERGY_NUM_PRINTED = ("-0.00303023", "-2.00661", "7.70506")
ENERGY_DEN_PRINTED = ("-0.67144", "-1.47569", "7.30468")

# Author-provided Schwarzschild gravitational l=2, n=0 numerical datum,
# s2l2.dat row 0: (2 M omega_R, 2 M |omega_I|). At a=0, m is degenerate.
AUTHOR_SCHWARZSCHILD_220_2M_OMEGA = (
    "0.7473433688360838", "0.1779246313778714")


def exact(value):
    """Interpret a decimal literal as an exact rational coefficient."""
    return sp.Rational(str(value))


def coordinates(m1=M1, m2=M2, chi1=CHI1, chi2=CHI2):
    """Return (M, eta, S, S_hat) from component masses and aligned spins."""
    m1, m2, chi1, chi2 = map(sp.sympify, (m1, m2, chi1, chi2))
    total = m1 + m2
    eta = m1 * m2 / total**2
    spin = (m1**2 * chi1 + m2**2 * chi2) / total**2
    return total, eta, spin, spin / (1 - 2 * eta)


def final_spin_expr(eta=ETA, spin=S, *, eta_coeffs=SPIN_ETA_PRINTED,
                    s_eta_coeffs=SPIN_S_ETA_PRINTED,
                    s_eta2_coeffs=SPIN_S_ETA2_PRINTED):
    """Eq. (22), with optional explicitly supplied inherited coefficient rows."""
    eta, spin = map(sp.sympify, (eta, spin))
    c1 = 2 * sp.sqrt(3) if eta_coeffs[0] is None else exact(eta_coeffs[0])
    return (spin + c1 * eta
            + sum(exact(c) * eta**i for i, c in enumerate(eta_coeffs[1:], 2))
            + eta * sum(exact(c) * spin**i for i, c in enumerate(s_eta_coeffs, 1))
            + eta**2 * sum(exact(c) * spin**i for i, c in enumerate(s_eta2_coeffs, 1)))


def radiated_fraction_expr(eta=ETA, s_hat=S_HAT, *, ns_coeffs=ENERGY_NS_PRINTED,
                           num_coeffs=ENERGY_NUM_PRINTED,
                           den_coeffs=ENERGY_DEN_PRINTED):
    """Eqs. (23)-(24), E_rad/M, with optional inherited coefficient rows."""
    eta, s_hat = map(sp.sympify, (eta, s_hat))
    ns = sum(exact(c) * eta**i for i, c in enumerate(ns_coeffs, 1))
    num = 1 + s_hat * sum(exact(c) * eta**i for i, c in enumerate(num_coeffs))
    den = 1 + s_hat * sum(exact(c) * eta**i for i, c in enumerate(den_coeffs))
    return ns * num / den


def final_mass_expr(total, radiated_fraction):
    """Mass conservation: M_f = M(1-E_rad/M)."""
    return sp.sympify(total) * (1 - sp.sympify(radiated_fraction))


def qnm_total_mass_expr(total, final_mass, qnm_final_mass=QNM_REAL):
    """Convert an input cyclic QNM ordinate M_f*f to total-mass units M*f."""
    return sp.sympify(total) * sp.sympify(qnm_final_mass) / sp.sympify(final_mass)


def natural_cubic_qnm(spins, cyclic_ordinates, query):
    """Independent natural spline of supplied numerical QNM data.

    The knots and ordinates are explicit inputs, not evidence for the Kerr
    eigenproblem. This project interpolation policy is separate from Eq. (22).
    """
    from scipy.interpolate import CubicSpline

    if len(spins) != len(cyclic_ordinates) or len(spins) < 3:
        raise ValueError("need at least three paired QNM knots")
    if not math.isfinite(float(query)) or not spins[0] <= query <= spins[-1]:
        raise ValueError("QNM spin lies outside the supplied knot domain")
    spline = CubicSpline(spins, cyclic_ordinates, axis=0, bc_type="natural")
    return spline(query)


def evaluate(expression, substitutions=None, *, dps=80):
    """High-precision scalar evaluation, independent of Torch and JAX."""
    values = {key: exact(value) for key, value in (substitutions or {}).items()}
    return sp.N(sp.sympify(expression).subs(values), dps)

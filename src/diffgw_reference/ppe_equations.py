"""Versioned, independent SymPy map of the supported ppE equations.

This is an optional mathematical reference, not a waveform carrier. It imports
neither native frontend nor production constants. The ``I`` symbols denote the
two integrals printed below; keeping them symbolic makes the z=0 ratio limit
explicit and permits independent high-precision quadrature in tests.
"""

from dataclasses import dataclass

import sympy as sp


SPEC_VERSION = "ppe-supported-equations/v2"
SOURCE_PPE = "Yunes--Pretorius, arXiv:0909.3328v2, Eqs. (1), (43), (47), (48)--(50), (52)"
SOURCE_DISPERSION = "Mirshekari--Yunes--Will, arXiv:1110.2720v2, Eqs. (14)--(16), (28)--(31)"

COMPONENT_IDS = {
    "fa": "ppe.inspiral.fa.v1",
    "deltapsi_ppe": "ppe.inspiral.deltapsi_ppe.v1",
    "deltapsi_disp": "ppe.dispersion.deltapsi_disp.v1",
}

f, m1, m2, alpha, a, beta, n, distance, z, wavelength, d0 = sp.symbols(
    "f m1 m2 alpha a beta n distance z wavelength d0", positive=True
)
# The ppE coefficients and exponents can have either sign; replace the
# positivity assumptions made solely for the physical inputs above.
alpha, a, beta, n = sp.symbols("alpha a beta n", real=True)
omega_m, omega_lambda, h0 = sp.symbols("omega_m omega_lambda h0", positive=True)
i_minus2, i_zero = sp.symbols("I_minus2 I_zero", positive=True)
pn_order = sp.symbols("pn_order", real=True)
integration_z = sp.symbols("integration_z", nonnegative=True)

# Exact SI decimal values, independent of diffgw_core and both frontends.
C = sp.Integer(299792458)
G = sp.Rational("6.67430e-11")
MSUN = sp.Rational("1.9884098706980507e30")
MPC = sp.Rational("3.085677581491367e22")

CHIRP_SECONDS = G * MSUN / C**3 * (m1 * m2)**sp.Rational(3, 5) / (m1 + m2)**sp.Rational(1, 5)
U = sp.pi * CHIRP_SECONDS * f
I_MINUS2 = sp.Integral(
    (1 + integration_z)**-2 / sp.sqrt(omega_m * (1 + integration_z)**3 + omega_lambda),
    (integration_z, 0, z),
)
I_ZERO = sp.Integral(
    1 / sp.sqrt(omega_m * (1 + integration_z)**3 + omega_lambda),
    (integration_z, 0, z),
)
D0_DIRECT = (1 + z) * C * MPC * i_minus2 / (1000 * h0)
DL_DIRECT = (1 + z) * C * MPC * i_zero / (1000 * h0)
D0_FROM_DL = distance * MPC * i_minus2 / i_zero
AMPLITUDE = 1 + alpha * U**a
PAPER_PHASE = beta * U**(n / 3)
DISPERSION = -sp.pi * d0 * C / (wavelength**2 * (1 + z) * f)
PHASE_ONLY = sp.exp(-sp.I * (PAPER_PHASE + DISPERSION))
FULL_MULTIPLIER = AMPLITUDE * PHASE_ONLY
PN_TO_NATIVE_EXPONENT = 2 * pn_order - 5


@dataclass(frozen=True)
class Equation:
    expression: sp.Basic
    source: str
    status: str = "paper"


EQUATIONS = {
    "ppe.inspiral.chirp_seconds/v1": Equation(CHIRP_SECONDS, SOURCE_PPE),
    "ppe.inspiral.u/v1": Equation(U, SOURCE_PPE),
    "ppe.inspiral.amplitude/v1": Equation(AMPLITUDE, SOURCE_PPE),
    "ppe.inspiral.phase/v1": Equation(PAPER_PHASE, SOURCE_PPE,
                                       "project n=3b mapping of paper phase"),
    "ppe.dispersion.integral_minus2/v1": Equation(I_MINUS2, SOURCE_DISPERSION),
    "ppe.dispersion.integral_zero/v1": Equation(I_ZERO, SOURCE_DISPERSION),
    "ppe.dispersion.d0_direct/v1": Equation(D0_DIRECT, SOURCE_DISPERSION),
    "ppe.dispersion.dl_direct/v1": Equation(DL_DIRECT, SOURCE_DISPERSION),
    "ppe.dispersion.d0_from_dl/v1": Equation(D0_FROM_DL, SOURCE_DISPERSION,
                                             "algebraic elimination of H0"),
    "ppe.dispersion.massive_graviton/v1": Equation(DISPERSION, SOURCE_DISPERSION),
    "ppe.selector.pn_order/v1": Equation(PN_TO_NATIVE_EXPONENT, SOURCE_PPE,
                                         "project PN-order convenience mapping"),
    "ppe.wrapper.carrier_phase_only/v1": Equation(PHASE_ONLY, SOURCE_PPE,
                                                  "project negative-Fourier-phase adapter"),
    "ppe.wrapper.carrier/v2": Equation(FULL_MULTIPLIER, SOURCE_PPE,
                                      "project negative-Fourier-phase adapter"),
}

# These are carrier *selectors*, not distinct ppE equations. Their native GR
# amplitudes/phases are separate model specifications. Each receives the same
# multiplier for both polarizations on every requested positive frequency.
SUPPORTED_CARRIERS = ("TaylorF2", "IMRPhenomD", "IMRPhenomXAS")
DISTANCE_VARIANTS = ("supplied_d0", "flat_matter_lambda_from_dl")

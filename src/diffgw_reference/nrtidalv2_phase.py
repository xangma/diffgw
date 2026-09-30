"""Independent nonspinning NRTidalv2 frequency-domain tidal phase.

Dietrich et al., PRD 100, 044003 (2019), arXiv:1905.06011v2,
Eqs. (17)--(21); Wade et al., PRD 89, 103012 (2014),
arXiv:1402.5156v1, Eq. (5). No production waveform code is imported.
The four decimal fit inputs are exact as printed, not refitted values.
"""

from functools import lru_cache

import mpmath as mp
import sympy as sp


SPEC_VERSION = "nrtidalv2-nonspinning-tidal-phase/v1"
COMPONENT_IDS = {
    "lambda_tilde": "nrtidalv2.phase.lambda_tilde.v1",
    "pade": "nrtidalv2.phase.pade.v1",
    "phase": "nrtidalv2.phase.tidal.v1",
    "frequency_derivative": "nrtidalv2.phase.frequency_derivative.v1",
}
SOURCES = {
    "dietrich2019": "https://arxiv.org/pdf/1905.06011v2",
    "wade2014": "https://arxiv.org/pdf/1402.5156v1",
}

F, M_SEC, DELTA, LAMBDA1, LAMBDA2 = sp.symbols(
    "f M_sec delta Lambda1 Lambda2", real=True
)
X = sp.symbols("x", positive=True)
ETA = (1 - DELTA**2) / 4

# Dietrich et al. Eqs. (20)--(21). The free fit decimals are kept exact.
C1 = sp.Rational(3115, 1248)
C32 = -sp.pi
C2 = sp.Rational(28024205, 3302208)
C52 = -sp.Rational(4283, 1092) * sp.pi
N52 = sp.Rational("90.550822")
N3 = sp.Rational("-60.253578")
D1 = sp.Rational("-15.111208")
D2 = sp.Rational("8.0641096")

# Dietrich et al. Eq. (19): constraints from the PN-expanded Padé phase.
N1 = C1 + D1
N32 = (C1 * C32 - C52 - C32 * D1 + N52) / C1
N2 = C2 + C1 * D1 + D2
D32 = -(C52 + C32 * D1 - N52) / C1
PADE_COEFFICIENTS = {
    "n_1": N1, "n_3over2": N32, "n_2": N2,
    "n_5over2": N52, "n_3": N3,
    "d_1": D1, "d_3over2": D32, "d_2": D2,
}


def lambda_tilde_expr(delta=DELTA, lambda1=LAMBDA1, lambda2=LAMBDA2):
    """Wade et al. Eq. (5), extended to either body labelling with signed delta."""
    eta = (1 - delta**2) / 4
    return sp.Rational(8, 13) * (
        (1 + 7 * eta - 31 * eta**2) * (lambda1 + lambda2)
        + delta * (1 + 9 * eta - 11 * eta**2) * (lambda1 - lambda2)
    )


def pade_expr(x=X):
    """Dietrich et al. Eq. (18), with Eq. (19) coefficient constraints."""
    numerator = 1 + N1*x + N32*x**sp.Rational(3, 2) + N2*x**2 \
        + N52*x**sp.Rational(5, 2) + N3*x**3
    denominator = 1 + D1*x + D32*x**sp.Rational(3, 2) + D2*x**2
    return numerator / denominator


def phase_expr(f=F, m_sec=M_SEC, delta=DELTA,
               lambda1=LAMBDA1, lambda2=LAMBDA2):
    """Additive psi_T in radians; waveform multiplier is exp(-i*psi_T)."""
    x = (sp.pi * m_sec * f)**sp.Rational(2, 3)
    eta = (1 - delta**2) / 4
    kappa_eff = sp.Rational(3, 16) * lambda_tilde_expr(delta, lambda1, lambda2)
    return -kappa_eff * sp.Rational(39, 16) / eta * x**sp.Rational(5, 2) * pade_expr(x)


@lru_cache(maxsize=2)
def _evaluator(derivative):
    expression = phase_expr()
    if derivative:
        expression = sp.diff(expression, F)
    return sp.lambdify((F, M_SEC, DELTA, LAMBDA1, LAMBDA2), expression, "mpmath")


def evaluate(f, m_sec, delta, lambda1, lambda2, *, derivative=False, dps=80):
    """Evaluate a scalar phase or d(phase)/df at high precision."""
    values = tuple(mp.mpf(str(v)) for v in (f, m_sec, delta, lambda1, lambda2))
    if values[0] <= 0 or values[1] <= 0 or abs(values[2]) >= 1:
        raise ValueError("f, M_sec must be positive and |delta| < 1")
    if values[3] < 0 or values[4] < 0:
        raise ValueError("dimensionless tidal deformabilities must be nonnegative")
    with mp.workdps(dps):
        return +_evaluator(derivative)(*values)

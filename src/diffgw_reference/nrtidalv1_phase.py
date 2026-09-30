"""Paper-literal scalar NRTidal v1 frequency-domain tidal phase.

Dietrich, Bernuzzi and Tichy, arXiv:1706.02969v2, Eqs. (2), (7).
The fitted decimal inputs are kept at their published precision. This module
does not import either native frontend or a binary waveform oracle.
"""

from functools import lru_cache

import mpmath as mp
import sympy as sp


SPEC_VERSION = "nrtidalv1-nonspinning-tidal-phase/v1"
COMPONENT_IDS = {
    "kappa": "nrtidalv1.phase.kappa_eff.v1",
    "pade": "nrtidalv1.phase.pade.v1",
    "phase": "nrtidalv1.phase.tidal.v1",
    "frequency_derivative": "nrtidalv1.phase.frequency_derivative.v1",
}
SOURCE = "https://arxiv.org/pdf/1706.02969v2"
ORIGINAL_SOURCE = "https://arxiv.org/html/1706.02969v1"

F, M_SEC, DELTA, LAMBDA1, LAMBDA2 = sp.symbols(
    "f M_sec delta Lambda1 Lambda2", real=True
)
X = sp.symbols("x", positive=True)
N1 = sp.Rational("-17.428")
N32 = sp.Rational("31.867")
N2 = sp.Rational("-26.414")
N52 = sp.Rational("62.362")
D32 = sp.Rational("36.089")
C1 = sp.Rational(3115, 1248)
D1 = N1 - C1  # Eq. (7): impose the analytic 1PN limit before evaluation.


def kappa_eff_expr(delta=DELTA, lambda1=LAMBDA1, lambda2=LAMBDA2):
    """Eq. (2), using Lambda_A = (2/3) k2_A / C_A**5."""
    xa = (1 + delta) / 2
    xb = (1 - delta) / 2
    return sp.Rational(3, 13) * (
        (xa + 12 * xb) * xa**4 * lambda1
        + (xb + 12 * xa) * xb**4 * lambda2
    )


def kappa2_original_printed_expr(delta=DELTA, lambda1=LAMBDA1,
                                 lambda2=LAMBDA2):
    """Literal arXiv:1706.02969v1 Eq. (2), for source-revision audit only.

    Its A term prints X_A/X_B, unlike the revised effective coupling and
    the later standard kappa_2^T definition. It is not a native carrier option.
    """
    xa = (1 + delta) / 2
    xb = (1 - delta) / 2
    return 3 * (xa**6 * lambda1 / xb + xb**6 * lambda2 / xa)


def pade_expr(x=X, *, carrier_rounded=False):
    """Eq. (7); optional rounded d1 identifies the current native carrier."""
    d1 = sp.Rational("-19.924") if carrier_rounded else D1
    numerator = 1 + N1*x + N32*x**sp.Rational(3, 2) + N2*x**2 \
        + N52*x**sp.Rational(5, 2)
    denominator = 1 + d1*x + D32*x**sp.Rational(3, 2)
    return numerator / denominator


def phase_expr(f=F, m_sec=M_SEC, delta=DELTA,
               lambda1=LAMBDA1, lambda2=LAMBDA2, *, carrier_rounded=False):
    """Additive Eq. (7) phase in radians; waveform factor is exp(-i psi_T)."""
    x = (sp.pi * m_sec * f)**sp.Rational(2, 3)
    eta = (1 - delta**2) / 4
    return (-kappa_eff_expr(delta, lambda1, lambda2)
            * sp.Rational(39, 16) / eta * x**sp.Rational(5, 2)
            * pade_expr(x, carrier_rounded=carrier_rounded))


@lru_cache(maxsize=4)
def _evaluator(derivative, carrier_rounded):
    expression = phase_expr(carrier_rounded=carrier_rounded)
    if derivative:
        expression = sp.diff(expression, F)
    return sp.lambdify((F, M_SEC, DELTA, LAMBDA1, LAMBDA2),
                       expression, "mpmath")


def evaluate(f, m_sec, delta, lambda1, lambda2, *, derivative=False,
             carrier_rounded=False, dps=80):
    """Evaluate the paper phase or explicit carrier-rounded diagnostic."""
    values = tuple(mp.mpf(str(v)) for v in (f, m_sec, delta, lambda1, lambda2))
    if values[0] <= 0 or values[1] <= 0 or abs(values[2]) >= 1:
        raise ValueError("f, M_sec must be positive and |delta| < 1")
    if values[3] < 0 or values[4] < 0:
        raise ValueError("dimensionless tidal deformabilities must be nonnegative")
    with mp.workdps(dps):
        return +_evaluator(derivative, carrier_rounded)(*values)

"""Independent scalar NRTidalv2 amplitude and merger-taper reference.

Dietrich et al. (2019), arXiv:1905.06011v2, Eqs. (24)--(25);
Dietrich et al. (2018), arXiv:1804.02235, Eq. (11). Inputs use SI seconds
for mass and distance, Hz for frequency, and dimensionless Lambda values.
No native waveform or LAL implementation is imported.
"""

from functools import lru_cache

import mpmath as mp
import sympy as sp

from diffgw_reference.nrtidalv2_phase import lambda_tilde_expr


SPEC_VERSION = "nrtidalv2-nonspinning-tidal-amplitude-taper/v1"
F, M_SEC, D_SEC, DELTA, LAMBDA1, LAMBDA2 = sp.symbols(
    "f M_sec D_sec delta Lambda1 Lambda2", real=True
)
ETA = (1 - DELTA**2) / 4
X = (sp.pi * M_SEC * F)**sp.Rational(2, 3)
KAPPA = sp.Rational(3, 16) * lambda_tilde_expr(DELTA, LAMBDA1, LAMBDA2)

# Eq. (24), including its physical Newtonian amplitude normalization.
AMPLITUDE = (
    -sp.sqrt(5 * sp.pi * ETA / 24) * 9 * M_SEC**2 / D_SEC
    * KAPPA * X**sp.Rational(13, 4)
    * (1 + sp.Rational(449, 108)*X
       + sp.Rational(22672, 9)*X**sp.Rational(289, 100))
    / (1 + sp.Rational("13477.8")*X**4)
)

# Dietrich et al. (2018), Eq. (11), with the paper's stated substitution
# kappa_2^T -> kappa_eff^T = 3 Lambda_tilde / 16.
XA = (1 + sp.Abs(DELTA)) / 2
XB = (1 - sp.Abs(DELTA)) / 2
MERGER_FREQUENCY = (
    sp.Rational("0.3586") * sp.sqrt(XB / XA)
    * (1 + sp.Rational("0.03354")*KAPPA
       + sp.Rational("0.00004315")*KAPPA**2)
    / (1 + sp.Rational("0.07542")*KAPPA
       + sp.Rational("0.0002236")*KAPPA**2)
    / (2 * sp.pi * M_SEC)
)


@lru_cache(maxsize=2)
def _evaluator(name):
    expression = AMPLITUDE if name == "amplitude" else MERGER_FREQUENCY
    symbols = ((F, M_SEC, D_SEC, DELTA, LAMBDA1, LAMBDA2)
               if name == "amplitude" else
               (M_SEC, DELTA, LAMBDA1, LAMBDA2))
    return sp.lambdify(symbols, expression, "mpmath")


def amplitude(f, m_sec, d_sec, delta, lambda1, lambda2, *, dps=80):
    """Signed, face-on additive Fourier amplitude from Eq. (24)."""
    values = tuple(mp.mpf(str(v)) for v in
                   (f, m_sec, d_sec, delta, lambda1, lambda2))
    if min(values[:3]) <= 0 or abs(values[3]) >= 1 or min(values[4:]) < 0:
        raise ValueError("f, M_sec, D_sec > 0, |delta| < 1, Lambda_i >= 0")
    with mp.workdps(dps):
        return +_evaluator("amplitude")(*values)


def merger_frequency(m_sec, delta, lambda1, lambda2, *, dps=80):
    """GW frequency in Hz from the cited merger fit."""
    values = tuple(mp.mpf(str(v)) for v in
                   (m_sec, delta, lambda1, lambda2))
    if values[0] <= 0 or abs(values[1]) >= 1 or min(values[2:]) < 0:
        raise ValueError("M_sec > 0, |delta| < 1, Lambda_i >= 0")
    with mp.workdps(dps):
        return +_evaluator("merger")(*values)


def taper(f, f_merger, *, dps=80):
    """Descending Planck window of Eq. (25), ending at 1.2 f_merger."""
    f, f_merger = mp.mpf(str(f)), mp.mpf(str(f_merger))
    if f_merger <= 0:
        raise ValueError("f_merger must be positive")
    if f <= f_merger:
        return mp.mpf(1)
    if f >= mp.mpf("1.2") * f_merger:
        return mp.mpf(0)
    with mp.workdps(dps):
        z = (f - f_merger) / (mp.mpf("0.2") * f_merger)
        sigma = 1/z + 1/(z - 1)
        return +(1 / (1 + mp.exp(-sigma)))

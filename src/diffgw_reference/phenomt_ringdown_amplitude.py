"""Independent scalar THM (2,2) ringdown amplitude on supplied inputs.

Estellés et al., arXiv:2012.11923v1, Eqs. (26), (27c)–(27e).
``d2`` defaults to the printed QNM relation. An explicit ``d2`` represents
the native carrier's bounded-rate branch and is not Eq. (27d) fidelity.
No native waveform, fit, or QNM implementation is imported.
"""

from __future__ import annotations

import mpmath as mp


SPEC_VERSION = "phenomt-ringdown-amplitude/v1"
COMPONENT_ID = "T-RD-22-AMP-v1"
SOURCE = "Estelles et al., arXiv:2012.11923v1, Eqs. (26), (27c)-(27e)"


def _inputs(u, alpha1, alpha2, peak, d3, d2):
    time, a1, a2, hp, shape = (mp.mpf(str(x)) for x in (u, alpha1, alpha2, peak, d3))
    rate = (a2 - a1) / 2 if d2 is None else mp.mpf(str(d2))
    if not all(map(mp.isfinite, (time, a1, a2, hp, shape, rate))):
        raise ValueError("ringdown inputs must be finite")
    if time < 0 or a1 <= 0 or a2 <= a1 or hp <= 0 or rate <= 0:
        raise ValueError("requires u>=0, 0<alpha1<alpha2, H_peak>0, and d2>0")
    return time, a1, a2, hp, shape, rate


def coefficients(alpha1, alpha2, peak, d3, *, d2=None, dps=80):
    """Return ``(d1,d2,d3,d4)`` from Eq. (27c)–(27e)."""
    with mp.workdps(dps):
        _, a1, _, hp, shape, rate = _inputs(0, alpha1, alpha2, peak, d3, d2)
        d1 = hp * a1 * mp.cosh(shape) ** 2 / rate
        d4 = hp - d1 * mp.tanh(shape)
        return d1, rate, shape, d4


def amplitude(u, alpha1, alpha2, peak, d3, *, d2=None, dps=80):
    """Dimensionless positive-mode amplitude, Eq. (26), for ``u>=0``."""
    with mp.workdps(dps):
        time, a1, a2, hp, shape, rate = _inputs(u, alpha1, alpha2, peak, d3, d2)
        d1, _, _, d4 = coefficients(a1, a2, hp, shape, d2=rate, dps=dps)
        return mp.exp(-a1 * time) * (d1 * mp.tanh(rate * time + shape) + d4)


def derivative(u, alpha1, alpha2, peak, d3, *, d2=None, dps=80):
    """Analytic derivative ``dH/du`` of Eq. (26)."""
    with mp.workdps(dps):
        time, a1, a2, hp, shape, rate = _inputs(u, alpha1, alpha2, peak, d3, d2)
        d1, _, _, d4 = coefficients(a1, a2, hp, shape, d2=rate, dps=dps)
        arg = rate * time + shape
        body = d1 * mp.tanh(arg) + d4
        return mp.exp(-a1 * time) * (d1 * rate / mp.cosh(arg) ** 2 - a1 * body)

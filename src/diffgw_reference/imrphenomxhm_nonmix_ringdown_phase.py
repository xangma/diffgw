# SPDX-License-Identifier: MIT
"""Independent scalar XHM non-mixing ringdown phase, Eqs. (43)--(44).

García-Quirós et al., arXiv:2001.10914v2, Sec. VI.2.1. All coefficients,
QNM frequencies, and intermediate boundary data are caller supplied.
"""

import mpmath as mp


def _inputs(join, ring, damp):
    join, ring, damp = map(mp.mpf, (join, ring, damp))
    if min(join, ring, damp) <= 0:
        raise ValueError("join, ring, and damping frequencies must be positive")
    return join, ring, damp


def attachment_shifts(join, phase_at_join, slope_at_join, ring, damp, a2, a_lambda):
    """Unique linear and constant shifts yielding a C1 join."""
    j, r, d = _inputs(join, ring, damp)
    p, g, a2, a_lambda = map(mp.mpf, (phase_at_join, slope_at_join, a2, a_lambda))
    time_shift = g - a2 * r**2 / j**2 - a_lambda * d / (d**2 + (j - r)**2)
    phase_shift = p + a2 * r**2 / j - a_lambda * mp.atan((j - r) / d) - time_shift * j
    return time_shift, phase_shift


def phase(frequency, ring, damp, a2, a_lambda, time_shift, phase_shift):
    """Antiderivative with explicitly supplied attachment shifts."""
    f, r, d = _inputs(frequency, ring, damp)
    a2, a_lambda, time_shift, phase_shift = map(mp.mpf, (a2, a_lambda, time_shift, phase_shift))
    return -a2 * r**2 / f + a_lambda * mp.atan((f - r) / d) + time_shift * f + phase_shift


def derivative(frequency, ring, damp, a2, a_lambda, time_shift):
    f, r, d = _inputs(frequency, ring, damp)
    a2, a_lambda, time_shift = map(mp.mpf, (a2, a_lambda, time_shift))
    return a2 * r**2 / f**2 + a_lambda * d / (d**2 + (f - r)**2) + time_shift


def second_derivative(frequency, ring, damp, a2, a_lambda):
    f, r, d = _inputs(frequency, ring, damp)
    a2, a_lambda = map(mp.mpf, (a2, a_lambda))
    return -2 * a2 * r**2 / f**3 - 2 * a_lambda * d * (f - r) / (d**2 + (f - r)**2)**2

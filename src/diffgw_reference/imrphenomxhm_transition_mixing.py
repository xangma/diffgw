# SPDX-License-Identifier: MIT
"""Paper-era XHM (3,2) phase join and spherical/spheroidal mixing.

Only the algebra printed in Garcia-Quiros et al., arXiv:2001.10914v2,
Eqs. (5.4), (5.6), (6.6)--(6.8), and (A1)--(A3) is implemented.  All
collocation targets, QNM frequencies, amplitudes, and mixing coefficients
are caller data.  In particular, this is not a release-122022 fit provider.
"""

from __future__ import annotations

import mpmath as mp


SPEC_VERSION = "imrphenomxhm-transition-mixing/v1"
COMPONENT_IDS = {
    "intermediate_phase_join": "xhm.phase.32.intermediate-ringdown-join.v1",
    "spherical_spheroidal_mixing": "xhm.mixing.32.spherical-spheroidal.v1",
}


def spherical_32_from_spheroidal(h22, h32_s, alpha222, alpha232, alpha322, alpha332):
    """Eliminate spheroidal 22 from the two-mode projection, Eq. (A3)."""

    a222 = mp.mpc(alpha222)
    if not a222:
        raise ValueError("alpha222 must be nonzero")
    ratio = mp.mpc(alpha322) / a222
    return ratio * mp.mpc(h22) + (mp.mpc(alpha332) - ratio * mp.mpc(alpha232)) * mp.mpc(h32_s)


def mixed_phase_derivatives(frequency, h22, h32_s, coefficients):
    """Return local ``(phase, dphase/df, d2phase/df2)`` of mixed ``h32``.

    ``h22`` and ``h32_s`` are scalar complex callables of frequency.  With
    ``h=A exp(-i phase)``, logarithmic differentiation avoids unwrapping the
    principal value of ``arg(h)`` and remains valid for any nonzero amplitude.
    """

    f = mp.mpf(frequency)
    if f <= 0:
        raise ValueError("frequency must be positive")
    if len(coefficients) != 4:
        raise ValueError("four mixing coefficients are required")

    def mixed(x):
        return spherical_32_from_spheroidal(h22(x), h32_s(x), *coefficients)

    value = mixed(f)
    if not value:
        raise ValueError("mixed mode must be nonzero at the evaluation frequency")
    first = mp.diff(mixed, f) / value
    second = mp.diff(mixed, f, 2) / value - first**2
    return -mp.arg(value), -mp.im(first), -mp.im(second)


def ringdown_spheroidal_phase(frequency, f_ring32, f_damp32, coefficients, *, offset=0):
    """Integrated Eq. (6.7) for ``(alpha2, alpha4, alpha_lambda, dphi)``."""

    f, ring, damp = map(mp.mpf, (frequency, f_ring32, f_damp32))
    if min(f, ring, damp) <= 0:
        raise ValueError("frequencies must be positive")
    a2, a4, alam, dphi = map(mp.mpf, coefficients)
    return -a2 / f - a4 / (3 * f**3) + alam * mp.atan((f - ring) / damp) + dphi * f + mp.mpf(offset)


def ringdown_spheroidal_phase_derivative(frequency, f_ring32, f_damp32, coefficients):
    """The (3,2) spheroidal ringdown phase derivative, Eq. (6.6)."""

    f, ring, damp = map(mp.mpf, (frequency, f_ring32, f_damp32))
    if min(f, ring, damp) <= 0:
        raise ValueError("frequencies must be positive")
    a2, a4, alam, dphi = map(mp.mpf, coefficients)
    return a2 / f**2 + a4 / f**4 + alam * damp / (damp**2 + (f - ring)**2) + dphi


def _intermediate_row(f, ring, damp):
    lorentz = damp / (damp**2 + (f - ring)**2)
    return [lorentz, mp.mpf(1), *[f**-k for k in range(1, 5)]]


def intermediate_phase_coefficients(nodes, targets, f_join, dphi_join, d2phi_join, f_ring32, f_damp32):
    """Solve the six Eq. (5.6) collocation and ringdown-join conditions.

    Coefficients follow Eq. (5.4): ``(a_lambda, a0, a1, a2, a3, a4)``.
    ``dphi_join`` and ``d2phi_join`` belong to the *spherical* ringdown
    phase, after the Appendix-A basis transform.
    """

    if len(nodes) != 4 or len(targets) != 4:
        raise ValueError("four intermediate nodes and targets are required")
    fs = tuple(map(mp.mpf, nodes))
    values = tuple(map(mp.mpf, targets))
    join, ring, damp = map(mp.mpf, (f_join, f_ring32, f_damp32))
    if min(*fs, join, ring, damp) <= 0 or len(set(fs + (join,))) != 5:
        raise ValueError("positive, distinct nodes and join are required")
    rows = [_intermediate_row(f, ring, damp) for f in (*fs, join)]
    derivative_lorentz = -2 * damp * (join - ring) / (damp**2 + (join - ring)**2)**2
    rows.append([derivative_lorentz, mp.mpf(0), *[-k * join**(-k - 1) for k in range(1, 5)]])
    right = mp.matrix((*values, mp.mpf(dphi_join), mp.mpf(d2phi_join)))
    solution = mp.lu_solve(mp.matrix(rows), right)
    return tuple(solution)


def intermediate_phase_derivative(frequency, f_ring32, f_damp32, coefficients):
    """Evaluate Eq. (5.4), with its terms ordered as above."""

    f, ring, damp = map(mp.mpf, (frequency, f_ring32, f_damp32))
    if min(f, ring, damp) <= 0:
        raise ValueError("frequencies must be positive")
    alam, a0, a1, a2, a3, a4 = map(mp.mpf, coefficients)
    return alam * damp / (damp**2 + (f - ring)**2) + a0 + a1 / f + a2 / f**2 + a3 / f**3 + a4 / f**4


def intermediate_phase(frequency, f_ring32, f_damp32, coefficients, *, offset=0):
    """Primitive of Eq. (5.4); choose ``offset`` for value continuity."""

    f, ring, damp = map(mp.mpf, (frequency, f_ring32, f_damp32))
    if min(f, ring, damp) <= 0:
        raise ValueError("frequencies must be positive")
    alam, a0, a1, a2, a3, a4 = map(mp.mpf, coefficients)
    return (alam * mp.atan((f - ring) / damp) + a0 * f + a1 * mp.log(f)
            - a2 / f - a3 / (2 * f**2) - a4 / (3 * f**3) + mp.mpf(offset))

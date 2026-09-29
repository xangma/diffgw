# SPDX-License-Identifier: MIT
"""Independent scalar XHM non-mixing intermediate-phase construction.

García-Quirós et al., arXiv:2001.10914v2, Eqs. (38)--(39), Table 7.
The supplied targets are external calibration inputs, not derived here.
"""

from __future__ import annotations

import mpmath as mp


SPEC_VERSION = "imrphenomxhm-intermediate-phase/v1"
REFERENCE_VERSION = SPEC_VERSION
COMPONENT_IDS = {
    "nonmix_collocation": "xhm.phase.intermediate.nonmix-collocation.v1",
}


def default_nodes(eta, f_ins, f_ring):
    """Paper Table-7 default nodes C1, C2, E2, E3, E5."""
    eta, f_ins, f_ring = map(mp.mpf, (eta, f_ins, f_ring))
    if not 0 < eta <= mp.mpf("0.25") or not 0 < f_ins < f_ring:
        raise ValueError("expected 0 < eta <= 1/4 and 0 < f_ins < f_ring")
    start = (1 + mp.mpf("0.001") * (mp.mpf("0.25") / eta - 1)) * f_ins
    end = f_ring
    return (
        start,
        ((mp.sqrt(3) + 2) * start - (mp.sqrt(3) - 2) * end) / 4,
        (end + 3 * start) / 4,
        (end + start) / 2,
        (7 * end + start) / 8,
    )


def _basis(frequency, f_ring, f_damp):
    f, r, d = map(mp.mpf, (frequency, f_ring, f_damp))
    if f <= 0 or d <= 0:
        raise ValueError("positive frequency and damping frequency required")
    return (d / (d * d + (f - r) ** 2), 1, 1 / f, 1 / f**2, 1 / f**4)


def coefficients(nodes, targets, f_ring, f_damp):
    """Solve Eq. (39) without a backend or production coefficient dependency."""
    if len(nodes) != 5 or len(targets) != 5:
        raise ValueError("five nodes and targets are required")
    matrix = mp.matrix([_basis(f, f_ring, f_damp) for f in nodes])
    rhs = mp.matrix([mp.mpf(value) for value in targets])
    return tuple(mp.lu_solve(matrix, rhs))


def phase_derivative(frequency, coeffs, f_ring, f_damp):
    return sum(c * basis for c, basis in zip(coeffs, _basis(frequency, f_ring, f_damp)))


def phase(frequency, coeffs, f_ring, f_damp):
    """Antiderivative with arbitrary additive constant chosen as zero."""
    f, r, d = map(mp.mpf, (frequency, f_ring, f_damp))
    if f <= 0 or d <= 0:
        raise ValueError("positive frequency and damping frequency required")
    lam, a0, a1, a2, a4 = coeffs
    return lam * mp.atan((f - r) / d) + a0 * f + a1 * mp.log(f) - a2 / f - a4 / (3 * f**3)


__all__ = [
    "COMPONENT_IDS",
    "REFERENCE_VERSION",
    "SPEC_VERSION",
    "coefficients",
    "default_nodes",
    "phase",
    "phase_derivative",
]

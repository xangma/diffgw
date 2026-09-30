# SPDX-License-Identifier: MIT
"""Independent scalar XHM intermediate-amplitude construction.

García-Quirós et al., arXiv:2001.10914v2, Sec. V.1.1, Eq. (37).
The endpoint and interior amplitudes are supplied calibration data.  This
module solves the paper's polynomial in physical frequency with mpmath; the
native backends use a normalized coordinate for numerical conditioning.
"""

from __future__ import annotations

import mpmath as mp


SPEC_VERSION = "imrphenomxhm-intermediate-amplitude/v1"
REFERENCE_VERSION = SPEC_VERSION
COMPONENT_IDS = {"intermediate_inverse_polynomial": "xhm.amplitude.intermediate.inverse-polynomial.v1"}


def coefficients(f_start, f_end, start_amplitude, start_derivative_df,
                 interior_amplitudes, end_amplitude, end_derivative_df):
    """Return the six physical-frequency coefficients of ``A0/A(f)``.

    Amplitudes are normalized by the same constant ``A0``.  The two
    collocation frequencies divide the join interval into thirds.
    """
    lo, hi = mp.mpf(f_start), mp.mpf(f_end)
    if not lo < hi or len(interior_amplitudes) != 2:
        raise ValueError("ordered endpoints and two interior amplitudes required")
    v0, v1, v2, ve = map(mp.mpf, (start_amplitude, *interior_amplitudes, end_amplitude))
    if min(v0, v1, v2, ve) <= 0:
        raise ValueError("amplitudes must be positive")
    nodes = (lo, lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3, hi)

    def value_row(f):
        return [f ** k for k in range(6)]

    def derivative_row(f):
        return [mp.mpf(0)] + [k * f ** (k - 1) for k in range(1, 6)]

    matrix = mp.matrix((value_row(nodes[0]), derivative_row(nodes[0]),
                        value_row(nodes[1]), value_row(nodes[2]),
                        value_row(nodes[3]), derivative_row(nodes[3])))
    rhs = mp.matrix((1 / v0, -mp.mpf(start_derivative_df) / v0**2,
                     1 / v1, 1 / v2, 1 / ve,
                     -mp.mpf(end_derivative_df) / ve**2))
    return tuple(mp.lu_solve(matrix, rhs))


def inverse_polynomial(frequency, coeffs):
    f = mp.mpf(frequency)
    return sum(c * f**k for k, c in enumerate(coeffs))


def amplitude(frequency, a0, coeffs):
    return mp.mpf(a0) / inverse_polynomial(frequency, coeffs)


def amplitude_derivative_df(frequency, a0, coeffs):
    f = mp.mpf(frequency)
    p = inverse_polynomial(f, coeffs)
    dp = sum(k * coeffs[k] * f**(k - 1) for k in range(1, 6))
    return -mp.mpf(a0) * dp / p**2


__all__ = ["COMPONENT_IDS", "REFERENCE_VERSION", "SPEC_VERSION", "coefficients",
           "inverse_polynomial", "amplitude", "amplitude_derivative_df"]

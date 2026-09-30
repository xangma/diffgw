# SPDX-License-Identifier: MIT
"""Independent scalar transcription of García-Quirós et al. Eq. (42).

The caller supplies the calibrated parameters and QNM frequencies. No
production frontend, fit surface, or binary implementation is imported.
"""

import mpmath as mp


def amplitude_ratio(frequency, ring, damp, sigma, a_lambda, lambda_fit):
    """Dimensionless A_RD^(lm) / A_0^22 at positive frequency."""
    f, r, d, s, a, lam = map(mp.mpf, (frequency, ring, damp, sigma, a_lambda, lambda_fit))
    if f <= 0 or d <= 0 or s <= 0 or a == 0:
        raise ValueError("frequency, damping, sigma must be positive; a_lambda nonzero")
    width = d * s
    offset = f - r
    return abs(a) * width * mp.exp(-lam * offset / width) / (mp.root(f, 12) * (offset * offset + width * width))


def derivatives(frequency, ring, damp, sigma, a_lambda, lambda_fit):
    """Analytic derivatives in f, a_lambda, and lambda, in that order."""
    f, r, d, s, a, lam = map(mp.mpf, (frequency, ring, damp, sigma, a_lambda, lambda_fit))
    width = d * s
    offset = f - r
    value = amplitude_ratio(f, r, d, s, a, lam)
    return (
        value * (-1 / (12 * f) - 2 * offset / (offset * offset + width * width) - lam / width),
        value / a,
        -offset * value / width,
    )

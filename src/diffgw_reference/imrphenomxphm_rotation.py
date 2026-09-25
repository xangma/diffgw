"""Scalar IMRPhenomXPHM twist-up reference from arXiv:2004.06503v2.

This executable reference covers only the precession-specific layer: Wigner
rotation, positive-frequency polarization assembly, observer-basis rotation,
and per-mode SPA frequency routing.  The supplied coprecessing carrier is a
declared dependency; no IMRPhenomXHM fits are reproduced here.
"""

import cmath
import math
from typing import Tuple


def wigner_small_d(l: int, m: int, mp: int, beta: float) -> float:
    """Return the exact finite-sum Wigner ``d^l_{m,mp}(beta)`` scalar."""
    if l < 0 or abs(m) > l or abs(mp) > l:
        raise ValueError("Wigner indices require l >= 0 and |m|, |mp| <= l.")
    c = math.cos(0.5 * beta)
    s = math.sin(0.5 * beta)
    numerator = math.sqrt(
        math.factorial(l + m)
        * math.factorial(l - m)
        * math.factorial(l + mp)
        * math.factorial(l - mp)
    )
    value = 0.0
    for k in range(max(0, m - mp), min(l + m, l - mp) + 1):
        denominator = (
            math.factorial(l + m - k)
            * math.factorial(k)
            * math.factorial(l - mp - k)
            * math.factorial(k - m + mp)
        )
        value += (
            (-1) ** k
            * numerator
            / denominator
            * c ** (2 * l + m - mp - 2 * k)
            * s ** (2 * k - m + mp)
        )
    return value


def spa_twist_frequency(f_mode: float, mp: int) -> float:
    """Map a mode frequency to the equivalent 22 angle frequency (Eq. 96)."""
    if mp <= 0:
        raise ValueError("The positive-frequency twist-up requires mp > 0.")
    return 2.0 * f_mode / mp


def polarizations_to_primed(hp: complex, hc: complex, zeta: float) -> Tuple[complex, complex]:
    """Map the unprimed N-frame components to the primed spherical basis.

    This is the forward map printed in Pratten et al. Appendix D,
    Eqs. (93)--(94).
    """
    c = math.cos(2.0 * zeta)
    s = math.sin(2.0 * zeta)
    return c * hp - s * hc, s * hp + c * hc


def polarizations_from_primed(
    hp_prime: complex,
    hc_prime: complex,
    zeta: float,
) -> Tuple[complex, complex]:
    """Map primed spherical-basis components into the unprimed N-frame.

    The twist-up sum produces ``(hp_prime, hc_prime)`` in the spherical basis
    used by the harmonics.  XPHM returns the LAL observer basis, so this is the
    inverse of the paper's forward map in Eqs. (93)--(94).
    """
    c = math.cos(2.0 * zeta)
    s = math.sin(2.0 * zeta)
    return c * hp_prime + s * hc_prime, -s * hp_prime + c * hc_prime


def rotate_polarizations(hp: complex, hc: complex, zeta: float) -> Tuple[complex, complex]:
    """Compatibility name for the primed-to-observer inverse basis map."""
    return polarizations_from_primed(hp, hc, zeta)


def twist_up_mode(
    l: int,
    mp: int,
    carrier: complex,
    alpha: float,
    beta: float,
    gamma: float,
    theta_jn: float,
    zeta: float,
) -> Tuple[complex, complex]:
    """Project one positive-m coprecessing carrier into ``(h_plus,h_cross)``.

    The sum is Pratten et al. Eqs. (110)--(113), with ``gamma=-epsilon``.
    ``i**mp`` and the normalization factor adapt DiffGW's declared XHM
    unprojected-mode convention to the paper's ``h^L_{l,-mp}`` convention.
    """
    if l not in (2, 3, 4) or mp <= 0 or mp > l:
        raise ValueError("XPHM rotation supports l in {2,3,4} and 0 < mp <= l.")
    sign_l = -1.0 if l % 2 else 1.0
    scale = sign_l * math.sqrt((2 * l + 1) / (16.0 * math.pi))
    plus_transfer = 0j
    cross_transfer = 0j
    for m in range(-l, l + 1):
        harmonic_d = wigner_small_d(l, m, 2, theta_jn)
        a_minus = (
            cmath.exp(-1j * m * alpha)
            * wigner_small_d(l, m, -mp, beta)
            * harmonic_d
        )
        a_plus_star = (
            cmath.exp(1j * m * alpha)
            * wigner_small_d(l, m, mp, beta)
            * harmonic_d
        )
        plus_transfer += a_minus + sign_l * a_plus_star
        cross_transfer += a_minus - sign_l * a_plus_star

    adapted_carrier = (1j ** mp) * carrier
    phase = cmath.exp(1j * mp * gamma)
    hp_raw = scale * phase * adapted_carrier * plus_transfer
    hc_raw = 1j * scale * phase * adapted_carrier * cross_transfer
    return polarizations_from_primed(hp_raw, hc_raw, zeta)


__all__ = [
    "polarizations_from_primed",
    "polarizations_to_primed",
    "rotate_polarizations",
    "spa_twist_frequency",
    "twist_up_mode",
    "wigner_small_d",
]

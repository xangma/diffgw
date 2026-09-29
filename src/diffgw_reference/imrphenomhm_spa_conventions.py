"""Independent quadratic-chirp SPA convention reference, with supplied C_lm.

This diagnostic derives F_minus[conj(h_lm)] directly from a damped Gaussian.
It imports neither production waveforms nor numerical-relativity data. The
optional leading-PN phases are transcribed from the named paper, with an
explicit common TD strain sign. See the derivation in
docs/math/imrphenomhm-spa-conventions.md.
"""

from __future__ import annotations

from math import gcd

import mpmath as mp
import sympy as sp


SPEC_VERSION = "imrphenomhm-spa-conventions/v1"
COMPONENT = "hm.conventions.quadratic-chirp-spa.v1"
COMPONENT_SOURCES = {
    COMPONENT: (
        ("https://arxiv.org/abs/0802.1249v3",
         "Eqs. (9.3), (9.4a,b,d,e,h,i): leading TD coefficient phases"),
        ("https://arxiv.org/abs/1708.00404v4",
         "Eq. (7): declared London low-frequency offset baseline"),
        ("https://arxiv.org/abs/1904.04831v2",
         "App. C, Eqs. (C.21)-(C.24): spherical-basis strain convention; "
         "App. A.3.1: strain-file sign versions"),
    ),
}


def _td_sign(common_td_sign):
    if isinstance(common_td_sign, bool) or common_td_sign not in (-1, 1):
        raise ValueError("common_td_sign must be -1 or +1")
    return common_td_sign


def leading_td_coefficient_phase(ell: int, m: int, *, common_td_sign) -> mp.mpc:
    """Unit leading TD C_lm from Blanchet et al. Eqs. (9.3)-(9.4).

    Positive frequency, positive unequal-mass difference, and positive leading
    real amplitude factors are assumed. At equal mass odd modes vanish and
    their phase is undefined. The common sign is declared, never data-fitted.
    """
    sign = _td_sign(common_td_sign)
    phases = {(2, 2): 1, (2, 1): mp.j, (3, 3): -mp.j,
              (3, 2): 1, (4, 4): -1, (4, 3): -mp.j}
    try:
        return mp.mpc(sign * phases[(ell, m)])
    except KeyError as exc:
        raise ValueError("leading coefficient phase is unavailable for this mode") from exc


def london_orbital_phase_correction(m: int, *, common_td_sign) -> mp.mpf:
    """22-preserving phase difference from the declared London-form baseline.

    The baseline phasor is exp(-i*Delta_lm), with
    Delta_lm=pi*(3*l+(l+m)%2-2)/2. Choosing the
    principal coefficient angle for C_22 fixes the orbital branch. This
    diagnostic does not alter any native implementation.
    """
    primitive_powers(m)
    sign = _td_sign(common_td_sign)
    constant = mp.pi / 4 if sign == 1 else -3 * mp.pi / 4
    return (1 - mp.mpf(m) / 2) * constant


def primitive_powers(m: int) -> tuple[int, int]:
    """Smallest positive powers cancelling an m:2 orbital phase."""
    if isinstance(m, bool) or not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    g = gcd(2, m)
    return 2 // g, m // g


def symbolic_spa_phase(m: int, f, phi0, omega, chirp_rate) -> sp.Expr:
    """Stationary phase of exp[i m phi(t) - 2 pi i f t], k>0."""
    primitive_powers(m)
    t = (2 * sp.pi * f / m - omega) / chirp_rate
    phi = phi0 + omega * t + chirp_rate * t**2 / 2
    return m * phi - 2 * sp.pi * f * t + sp.pi / 4


def symbolic_gaussian_carrier(m: int, f, phi0, omega, chirp_rate,
                              epsilon, coefficient=1, amplitude=1) -> sp.Expr:
    """Exact full-line transform with real positive A, k, epsilon.

    h=C A exp[-i m(phi0+omega*t+k*t**2/2)] exp[-epsilon*t**2/2].
    Conjugation makes the constant phase +i*m*phi0. The principal square
    root approaches exp(+i*pi/4)/sqrt(m*k) as epsilon decreases to zero.
    """
    primitive_powers(m)
    a = epsilon - sp.I * m * chirp_rate
    delta = m * omega - 2 * sp.pi * f
    return (sp.conjugate(coefficient) * amplitude * sp.exp(sp.I * m * phi0)
            * sp.sqrt(2 * sp.pi / a) * sp.exp(-delta**2 / (2 * a)))


def symbolic_relative_angle(m: int, coefficient_angle, carrier_angle,
                            common_phase=0) -> sp.Expr:
    """Primitive SPA phase after cancelling matched orbital dynamics.

    Coefficient angles refer to TD C_lm and C_22. A common phase is applied
    to the extracted FD carriers; a TD strain rotation has the opposite sign.
    """
    a, b = primitive_powers(m)
    return (-a * coefficient_angle + b * carrier_angle
            + (a - b) * (sp.pi / 4 + common_phase))


def _parameters(m, f, phi0, omega, chirp_rate, amplitude):
    primitive_powers(m)
    values = tuple(mp.mpf(x) for x in (f, phi0, omega, chirp_rate, amplitude))
    if not all(mp.isfinite(x) for x in values):
        raise ValueError("chirp parameters must be finite")
    f, phi0, omega, k, amplitude = values
    if f <= 0 or k <= 0 or amplitude <= 0:
        raise ValueError("f, chirp_rate, and amplitude must be positive")
    return f, phi0, omega, k, amplitude


def gaussian_carrier(m: int, f, phi0, omega, chirp_rate, epsilon,
                     coefficient=1, amplitude=1) -> mp.mpc:
    """Arbitrary-precision exact transform for positive Gaussian damping."""
    f, phi0, omega, k, amplitude = _parameters(
        m, f, phi0, omega, chirp_rate, amplitude,
    )
    epsilon = mp.mpf(epsilon)
    if not mp.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    a = epsilon - mp.j * m * k
    delta = m * omega - 2 * mp.pi * f
    return mp.mpc(mp.conj(mp.mpc(coefficient)) * amplitude
                  * mp.exp(mp.j * m * phi0) * mp.sqrt(2 * mp.pi / a)
                  * mp.exp(-delta**2 / (2 * a)))


def spa_carrier(m: int, f, phi0, omega, chirp_rate,
                coefficient=1, amplitude=1) -> mp.mpc:
    """Zero-damping SPA, exact as an oscillatory quadratic-chirp integral."""
    f, phi0, omega, k, amplitude = _parameters(
        m, f, phi0, omega, chirp_rate, amplitude,
    )
    t = (2 * mp.pi * f / m - omega) / k
    phi = phi0 + omega * t + k * t**2 / 2
    phase = m * phi - 2 * mp.pi * f * t + mp.pi / 4
    return mp.mpc(mp.conj(mp.mpc(coefficient)) * amplitude
                  * mp.sqrt(2 * mp.pi / (m * k)) * mp.exp(mp.j * phase))


def unit_phase(value) -> mp.mpc:
    """Normalize a nonzero complex value without choosing an angle branch."""
    value = mp.mpc(value)
    if not mp.isfinite(value) or not abs(value):
        raise ValueError("unit phase requires a finite nonzero mode")
    return mp.mpc(value / abs(value))


def primitive_relative_phase(mode, carrier_at_2f_over_m, m: int) -> mp.mpc:
    """u_lm(f)^(2/g) conj[u_22(2f/m)]^(m/g), g=gcd(2,m)."""
    a, b = primitive_powers(m)
    return mp.mpc(unit_phase(mode)**a
                  * mp.conj(unit_phase(carrier_at_2f_over_m))**b)


def spa_relative_phase(coefficient, coefficient22, m: int) -> mp.mpc:
    """Coefficient and Maslov phases surviving the primitive SPA invariant."""
    a, b = primitive_powers(m)
    return mp.mpc(mp.conj(unit_phase(coefficient))**a
                  * unit_phase(coefficient22)**b
                  * mp.exp(mp.j * (a - b) * mp.pi / 4))

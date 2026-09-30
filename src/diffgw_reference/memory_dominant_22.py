"""Independent dominant-(2, ±2) nonlinear-memory scalar reference.

Favata, arXiv:1003.3486v1, Sec. 6.2, Eq. (21) in the PDF (the equation
rendered ``10rsu`` in arXiv HTML), gives the full plus pattern. Talbot et al.,
arXiv:1807.00990v2, Eqs. (1)-(9), fix the strain-mode and SI conventions.
The negative-m oscillatory mode is assumed to be the equatorial-symmetry mirror.
This is not the arbitrary-mode coupling or an oscillatory waveform generator.
"""

from __future__ import annotations

import mpmath as mp

SOURCE_SPEC_VERSION = "memory.dominant_22/v1"
SPEC_VERSION = "memory-dominant-22-direct-class/v1"
COMPONENT_IDS = {
    "dominant_pair": "memory.dominant_22.v1",
    "sampled_flux": "memory.sampled_flux.v1",
}
FAVATA_SOURCE = "https://arxiv.org/pdf/1003.3486v1"
TALBOT_SOURCE = "https://arxiv.org/pdf/1807.00990v2"
C_SI = mp.mpf("299792458")


def _mp(value):
    if isinstance(value, (mp.mpf, mp.mpc)):
        return value
    if isinstance(value, complex):
        return mp.mpc(str(value.real), str(value.imag))
    return mp.mpf(str(value))


def spin_minus_two_y20(inclination):
    x = mp.cos(_mp(inclination))
    return mp.sqrt(15 / (32 * mp.pi)) * (1 - x * x)


def h20_coefficient():
    """Project Favata's full angular pattern onto normalized _{-2}Y_20.

    h20 = (R/c) * coefficient * integral |dot h22|² dt.
    The projection integral is 2π/(192π) ∫Y20(1-x²)(17+x²) dx.
    """
    return mp.sqrt(5 / (6 * mp.pi)) / 7


def h20_coefficient_by_projection():
    def y20(x):
        return mp.sqrt(15 / (32 * mp.pi)) * (1 - x * x)

    return mp.quad(lambda x: y20(x) * (1 - x * x) * (17 + x * x), [-1, 1]) / 96


def h20_from_flux_integral(flux_integral, distance_m):
    """Return h20 for ∫|dot h22|² dt in SI seconds and strain at distance_m."""
    distance = _mp(distance_m)
    if distance <= 0:
        raise ValueError("distance_m must be positive")
    return distance * h20_coefficient() * _mp(flux_integral) / C_SI


def h20_linear_mode(time_s, start_s, slope, distance_m):
    """Exact reference for h22(t)=slope*(t-start_s), t>=start_s."""
    dt = _mp(time_s) - _mp(start_s)
    if dt < 0:
        raise ValueError("time_s must not precede start_s")
    s = _mp(slope)
    return h20_from_flux_integral(abs(s) ** 2 * dt, distance_m)


def dh20_d_real_slope(time_s, start_s, slope_real, slope_imag, distance_m):
    """Analytic derivative for the real part of a complex linear-mode slope."""
    dt = _mp(time_s) - _mp(start_s)
    return 2 * _mp(slope_real) * dt * _mp(distance_m) * h20_coefficient() / C_SI


def plus_from_h20(h20, inclination):
    """Full dominant-22 plus pattern, including its l=4,m=0 contribution."""
    x = mp.cos(_mp(inclination))
    return _mp(h20) * spin_minus_two_y20(inclination) * mp.mpf(7) * (17 + x * x) / 120


def plus_from_flux_integral(flux_integral, distance_m, inclination):
    """Favata Eq. (21) evaluated directly, independently of h20 projection."""
    x = mp.cos(_mp(inclination))
    return _mp(distance_m) * _mp(flux_integral) * (1 - x * x) * (17 + x * x) / (192 * mp.pi * C_SI)


def sampled_h20(times, h22, distance_m):
    """Independent scalar reference for the native unequal-grid sampling rule.

    Endpoint secants and the quadratic-interpolant interior derivative are
    integrated with adjacent trapezoids. This tests the numerical adapter,
    while the continuous linear-mode and projection gates test the paper law.
    """
    t = tuple(map(_mp, times))
    y = tuple(map(_mp, h22))
    if len(t) < 2 or len(t) != len(y) or any(b <= a for a, b in zip(t, t[1:])):
        raise ValueError("times must be increasing and match h22 with at least two samples")
    if _mp(distance_m) <= 0:
        raise ValueError("distance_m must be positive")
    secants = tuple((y[i + 1] - y[i]) / (t[i + 1] - t[i]) for i in range(len(t) - 1))
    slopes = [secants[0]]
    for i in range(1, len(t) - 1):
        left, right = t[i] - t[i - 1], t[i + 1] - t[i]
        slopes.append((right * secants[i - 1] + left * secants[i]) / (left + right))
    slopes.append(secants[-1])
    flux = tuple(abs(slope) ** 2 for slope in slopes)
    result = [mp.mpf(0)]
    for i in range(len(t) - 1):
        increment = (flux[i] + flux[i + 1]) * (t[i + 1] - t[i]) / 2
        result.append(result[-1] + increment)
    return tuple(h20_from_flux_integral(value, distance_m) for value in result)

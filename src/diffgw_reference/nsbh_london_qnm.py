"""Scalar London--Fauchon-Jones 220 QNM fit printed in arXiv:1810.03550v2."""

from dataclasses import dataclass

import mpmath


@dataclass(frozen=True)
class LondonQNM220:
    frequency_hz: mpmath.mpf
    quality_factor: mpmath.mpf
    omega: mpmath.mpc


def london_qnm_220(final_mass_solar, final_spin, *, dps=90):
    """Evaluate London Eq. (22), then Thompson Eqs. (16)--(19)."""
    ctx = mpmath.mp.clone()
    ctx.dps = dps
    mass = ctx.mpf(str(final_mass_solar))
    spin = ctx.mpf(str(final_spin))
    if not ctx.isfinite(mass) or not ctx.isfinite(spin) or mass <= 0 or not -1 < spin < 1:
        raise ValueError("Expected finite positive final mass and -1 < final spin < 1")

    kappa = ctx.sqrt(ctx.log(2 - spin) / ctx.log(3))
    amplitude_phase = (
        ("1.5578", "2.9031"),
        ("1.9510", "5.9210"),
        ("2.0997", "2.7606"),
        ("1.4109", "5.9143"),
        ("0.4106", "2.7952"),
    )
    omega = ctx.one + sum(
        ctx.mpf(amplitude) * ctx.exp(ctx.j * ctx.mpf(phase)) * kappa**power
        for power, (amplitude, phase) in enumerate(amplitude_phase, start=1)
    )
    if omega.real <= 0 or omega.imag <= 0:
        raise ValueError("Expected positive 220 frequency and damping rate")

    # CODATA 2018 G, IAU 2015 nominal solar mass parameter, exact SI c.
    solar_seconds = ctx.mpf("6.67430e-11") * ctx.mpf("1.9884098706980507e30") / ctx.mpf("299792458")**3
    return LondonQNM220(
        omega.real / (2 * ctx.pi * mass * solar_seconds),
        omega.real / (2 * omega.imag),
        omega,
    )

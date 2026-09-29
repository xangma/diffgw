"""Scalar dominant-inspiral time and phase adapter for IMRPhenomT.

THM Eqs. (8)--(11), T/TP Eq. (19), and the pinned MIT author-carrier time
origin. This module imports neither native frontend nor its constants.
"""

from __future__ import annotations

import math

import mpmath as mp

from .phenomt_inspiral_collocation import frequency as collocated_frequency
from .phenomt_inspiral_collocation import solve
from .phenomt_taylor_t3 import coefficients as pn_coefficients
from .phenomt_taylor_t3 import phase as pn_phase


def solar_mass_seconds(*, dps=80):
    """CODATA/IAU physical conversion used by the public seconds interface."""
    with mp.workdps(dps):
        c = mp.mpf("299792458")
        g = mp.mpf("6.67430e-11")
        solar_mass = mp.mpf("1.9884098706980507e30")
        return g * solar_mass / c**3


class InspiralCarrier:
    """Frozen fit inputs composed with independently specified PN/collocation."""

    def __init__(self, m1, m2, chi1, chi2, t0, fitted_frequencies, *, dps=80):
        self.dps = dps
        with mp.workdps(dps):
            self.mass = mp.mpf(str(m1)) + mp.mpf(str(m2))
            self.eta = mp.mpf(str(m1)) * mp.mpf(str(m2)) / self.mass**2
            self.delta = (mp.mpf(str(m1)) - mp.mpf(str(m2))) / self.mass
            self.chi1, self.chi2 = mp.mpf(str(chi1)), mp.mpf(str(chi2))
            self.pn = pn_coefficients(self.eta, self.delta, self.chi1, self.chi2, dps=dps)
            self.pseudo, _ = solve(self.eta, t0, fitted_frequencies, self.pn, dps=dps)
            self.mass_seconds = self.mass * solar_mass_seconds(dps=dps)

    def omega(self, theta):
        with mp.workdps(self.dps):
            return collocated_frequency(theta, self.pn, self.pseudo, dps=self.dps)

    def frequency_hz(self, theta):
        with mp.workdps(self.dps):
            return self.omega(theta) / (2 * mp.pi * self.mass_seconds)

    def time_seconds(self, theta):
        """Post-tEarly author-carrier map, relative to the 22 amplitude peak."""
        with mp.workdps(self.dps):
            x = mp.mpf(str(theta))
            return -5 * self.mass_seconds / (self.eta * x**8)

    def time_at_frequency(self, frequency_hz):
        """Independently invert the inspiral series by bisection in theta."""
        with mp.workdps(self.dps):
            target = 2 * mp.pi * mp.mpf(str(frequency_hz)) * self.mass_seconds
            lo, hi = mp.mpf("0.33"), mp.mpf("0.82")
            if not self.omega(lo) < target < self.omega(hi):
                raise ValueError("target outside the bounded inspiral interval")
            for _ in range(280):
                mid = (lo + hi) / 2
                if self.omega(mid) < target:
                    lo = mid
                else:
                    hi = mid
            return self.time_seconds((lo + hi) / 2)

    def phase(self, theta):
        """Integrated 22 phase, up to the carrier's global phase constant."""
        with mp.workdps(self.dps):
            x = mp.mpf(str(theta))
            base = pn_phase(x, self.eta, self.delta, self.chi1, self.chi2, dps=self.dps)
            correction = (10 / self.eta) * sum(
                value * x ** (j + 3) / (j + 3)
                for j, value in enumerate(self.pseudo)
            )
            return base + correction


def fixed_grid(delta_t, duration, t_pad, *, centered=True):
    """Project time-grid convention, with the origin at the 22 peak."""
    step, length, pad = map(float, (delta_t, duration, t_pad))
    if not (math.isfinite(step) and step > 0 and math.isfinite(length) and length > 0
            and math.isfinite(pad) and pad >= 0):
        raise ValueError("invalid fixed-grid parameters")
    count = math.ceil(length / step)
    offset = count * step - pad if centered else 0.0
    return tuple(k * step - offset for k in range(count))


def lower_frequency_mask(times, start_seconds, tc):
    return tuple(time >= float(start_seconds) + float(tc) for time in times)


def reference_frequency(f_lower, f_ref):
    return float(f_lower if f_ref is None or f_ref == 0 else f_ref)

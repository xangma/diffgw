"""Scalar supplied-input IMRPhenomT/THM (2,2) ringdown frequency.

The published THM Eq. (25) has a sign inconsistent with its Eq. (27a).
The peak-matched branch here is fixed by Eq. (27a) and the paper's merger
continuity condition (31b); see docs/math/phenomt-ringdown-frequency.md.
No native implementation or calibrated fit is imported.
"""

from __future__ import annotations

import mpmath as mp


def _inputs(t, omega_rd, omega_peak, c2, c3):
    u, wr, wp, rate, shape = (mp.mpf(str(x)) for x in (t, omega_rd, omega_peak, c2, c3))
    if u < 0 or rate <= 0 or shape <= 0 or wr <= wp:
        raise ValueError("requires t>=0, c2>0, c3>0, and omega_rd>omega_peak")
    return u, wr, wp, rate, shape


def c1_from_peak(omega_rd, omega_peak, c2, c3, *, dps=80):
    """Eq. (27a) at c4=0, using the 22 amplitude peak at t=0."""
    with mp.workdps(dps):
        _, wr, wp, rate, shape = _inputs(0, omega_rd, omega_peak, c2, c3)
        return (1 + shape) * (wr - wp) / (rate * shape)


def omega(t, omega_rd, omega_peak, c2, c3, *, dps=80):
    """Peak-matched c4=0 branch of the THM ringdown ansatz."""
    with mp.workdps(dps):
        u, wr, wp, rate, shape = _inputs(t, omega_rd, omega_peak, c2, c3)
        e = mp.exp(-rate * u)
        c1 = (1 + shape) * (wr - wp) / (rate * shape)
        return wr - c1 * rate * shape * e / (1 + shape * e)


def phase_increment(t, omega_rd, omega_peak, c2, c3, *, dps=80):
    """Analytical integral from the peak, with phase_increment(0)=0."""
    with mp.workdps(dps):
        u, wr, wp, rate, shape = _inputs(t, omega_rd, omega_peak, c2, c3)
        c1 = (1 + shape) * (wr - wp) / (rate * shape)
        return wr * u + c1 * mp.log((1 + shape * mp.exp(-rate * u)) / (1 + shape))


def domega(t, omega_rd, omega_peak, c2, c3, *, dps=80):
    """Positive slope implied by the supplied c4=0 branch."""
    with mp.workdps(dps):
        u, wr, wp, rate, shape = _inputs(t, omega_rd, omega_peak, c2, c3)
        e = mp.exp(-rate * u)
        c1 = (1 + shape) * (wr - wp) / (rate * shape)
        return c1 * rate**2 * shape * e / (1 + shape * e)**2

"""Scalar Pv2 remnant-spin map inherited by NRTidalv2, not a full remnant fit."""

import mpmath as mp


SPEC_VERSION = "nrtidalv2-pv2-remnant-spin/v1"


def effective_chip(m1, m2, s1x, s1y, s2x, s2y, *, dps=80):
    """Pv2 technical note Eqs. (29)–(30), with m1 the heavier body."""
    with mp.workdps(dps):
        a, b, x1, y1, x2, y2 = map(
            lambda value: mp.mpf(str(value)), (m1, m2, s1x, s1y, s2x, s2y)
        )
        if not a >= b > 0:
            raise ValueError("ordered positive masses required")
        weight1 = 2 + 3 * b / (2 * a)
        weight2 = 2 + 3 * a / (2 * b)
        p1 = weight1 * a * a * mp.sqrt(x1 * x1 + y1 * y1)
        p2 = weight2 * b * b * mp.sqrt(x2 * x2 + y2 * y2)
        return mp.mpf(max(p1, p2) / (weight1 * a * a))


def final_spin(aligned_final_spin, m1, m2, chip, *, dps=80):
    """Pv2 note Eq. (46) with Eq. (51) and lambda=M_final/M_initial."""
    with mp.workdps(dps):
        af, a, b, cp = map(
            lambda value: mp.mpf(str(value)),
            (aligned_final_spin, m1, m2, chip),
        )
        if not a >= b > 0 or not 0 < af < 1 or not 0 <= cp <= mp.mpf("0.5"):
            raise ValueError("outside frozen positive-spin component domain")
        return mp.sqrt(af * af + (cp * (a / (a + b)) ** 2) ** 2)

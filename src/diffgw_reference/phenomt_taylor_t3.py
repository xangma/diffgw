"""Independent scalar 3.5PN TaylorT3 kernel for the PhenomT (2,2) carrier.

Estellés et al., arXiv:2004.08302v2, Appendix A, Eqs. (A5a)--(A5h).
The author-carrier choice includes the published-paper omission of one 3PN
nonspinning term, as documented in pinned MIT phenomxpy internals.py.
"""

from __future__ import annotations

import mpmath as mp


def coefficients(eta, delta, chi1, chi2, *, variant="author_carrier", dps=80):
    """Return c2,...,c7 of the orbital TaylorT3 series, including log(2) in c6.

    The mass variables in Appendix A are fractions of total mass. ``delta``
    is nonnegative because the native carriers order m1 >= m2.
    """
    if variant not in ("author_carrier", "paper_literal"):
        raise ValueError("variant must be 'author_carrier' or 'paper_literal'")
    with mp.workdps(dps):
        e, d, s1, s2 = (mp.mpf(str(x)) for x in (eta, delta, chi1, chi2))
        if not (0 < e <= mp.mpf("0.25") and 0 <= d < 1 and abs(s1) <= 1 and abs(s2) <= 1):
            raise ValueError("expected physical eta, delta and aligned spins")
        def r(n, den):
            return mp.mpf(n) / den
        m1, m2 = (1 + d) / 2, (1 - d) / 2
        weighted_spin = m1 * s1 + m2 * s2
        weighted_spin2 = m1 * s1**2 + m2 * s2**2
        weighted_spin3 = m1 * s1**3 + m2 * s2**3
        spin_sum = s1 + s2
        spin2_sum = s1**2 + s2**2
        spin3_sum = s1**3 + s2**3
        spin_diff = s1 - s2
        pi = mp.pi

        c2 = r(743, 2688) + r(11, 32) * e
        c3 = -r(3, 10) * pi + r(113, 160) * weighted_spin - r(19, 80) * e * spin_sum
        c4 = (r(1855099, 14450688) - r(243, 1024) * weighted_spin2
              + r(56975, 258048) * e
              + r(3, 1024) * e * (81 * spin2_sum - 158 * s1 * s2)
              + r(371, 2048) * e**2)
        c5 = (-r(7729, 21504) * pi + r(146597, 64512) * weighted_spin
              + r(13, 256) * pi * e - r(1213, 1152) * e * spin_sum
              + r(7, 128) * e * d * spin_diff - r(17, 128) * e**2 * spin_sum)
        c6 = (-r(720817631400877, 288412611379200)
              + r(107, 280) * mp.euler + r(53, 200) * pi**2
              - r(6127, 6400) * pi * weighted_spin
              - r(16928263, 68812800) * weighted_spin2
              + r(25302017977, 4161798144) * e
              - r(451, 2048) * pi**2 * e
              + r(1051, 3200) * pi * e * spin_sum
              + r(23281001, 68812800) * e * spin2_sum
              - r(377345, 1376256) * e * s1 * s2
              + r(453767, 4915200) * e * d * (s1**2 - s2**2)
              - r(30913, 1835008) * e**2
              + r(335129, 2457600) * e**2 * spin2_sum
              - r(488071, 1228800) * e**2 * s1 * s2
              + r(107, 280) * mp.log(2))
        if variant == "author_carrier":
            c6 += r(235925, 1769472) * e**3
        c7 = (-r(188516689, 433520640) * pi
              + r(6579635551, 650280960) * weighted_spin
              + r(3663, 5120) * pi * weighted_spin2
              - r(67493, 81920) * weighted_spin3
              - r(97765, 258048) * pi * e
              - r(1496368361, 185794560) * e * spin_sum
              - r(3663, 5120) * pi * e * spin2_sum
              + r(3537, 2560) * pi * e * s1 * s2
              + r(206917, 163840) * e * spin3_sum
              - r(192709, 81920) * e * s1 * s2 * weighted_spin
              - r(28633921, 12386304) * e * d * spin_diff
              + r(71931, 163840) * e * d * (s1**3 - s2**3)
              + r(141769, 1290240) * pi * e**2
              - r(840149, 15482880) * e**2 * spin_sum
              - r(2219, 40960) * e**2 * spin3_sum
              + r(1343, 40960) * e**2 * s1 * s2 * spin_sum
              + r(2591, 46080) * e**2 * d * spin_diff
              - r(12029, 92160) * e**3 * spin_sum)
        return c2, c3, c4, c5, c6, c7


def frequency(theta, eta, delta, chi1, chi2, *, variant="author_carrier", dps=80):
    """Dimensionless (2,2) angular frequency, twice THM Eq. (8)."""
    with mp.workdps(dps):
        x = mp.mpf(str(theta))
        if x <= 0:
            raise ValueError("theta must be positive")
        c2, c3, c4, c5, c6, c7 = coefficients(
            eta, delta, chi1, chi2, variant=variant, dps=dps
        )
        return x**3 / 4 * (1 + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5
                           + (c6 + mp.mpf(107)/280 * mp.log(x))*x**6 + c7*x**7)


def phase(theta, eta, delta, chi1, chi2, *, variant="author_carrier", dps=80):
    """Carrier-origin (2,2) TaylorT3 phase primitive, up to global phase."""
    with mp.workdps(dps):
        x, e = mp.mpf(str(theta)), mp.mpf(str(eta))
        if x <= 0:
            raise ValueError("theta must be positive")
        c2, c3, c4, c5, c6, c7 = coefficients(
            eta, delta, chi1, chi2, variant=variant, dps=dps
        )
        return 10/e * (-x**-5/5 - c2*x**-3/3 - c3*x**-2/2 - c4/x
                       + c5*(mp.log(x) - mp.log(5)/8) + c6*x
                       + mp.mpf(107)/280*x*(mp.log(x)-1) + c7*x**2/2)

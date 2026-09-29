"""Scalar Thompson et al. (2020) compactness and disruption equations.

Appendix B Eq. (32) and Appendix A Eqs. (8)--(10) of
https://arxiv.org/pdf/2002.08383v1. No production waveform is imported.
"""

from dataclasses import dataclass

import mpmath


@dataclass(frozen=True)
class Disruption:
    compactness: mpmath.mpf
    xi: mpmath.mpf
    radius_sec: mpmath.mpf
    frequency_hz: mpmath.mpf


def _context(dps: int):
    ctx = mpmath.mp.clone()
    ctx.dps = dps
    return ctx


def _number(ctx, value):
    return ctx.mpf(value) if isinstance(value, mpmath.mpf) else ctx.mpf(str(value))


def compactness(lambda_ns: float, *, dps: int = 70):
    """Evaluate the published piecewise Lambda-to-C fit without native clamps."""
    ctx = _context(dps)
    lam = ctx.mpf(str(lambda_ns))
    if lam < 0:
        raise ValueError("Lambda must be nonnegative")
    a0, a1, a2 = (ctx.mpf(s) for s in ("0.360", "-0.0355", "0.000705"))
    if lam <= 1:
        return ctx.mpf("0.5") + (3 * a0 - a1 - ctx.mpf("1.5")) * lam**2 + (a1 - 2 * a0 + 1) * lam**3
    log_lam = ctx.log(lam)
    return a0 + a1 * log_lam + a2 * log_lam**2


def tidal_polynomial(xi, q, chi, compactness_value, *, dps: int = 70):
    """Evaluate Thompson Eq. (10) at positive xi."""
    ctx = _context(dps)
    x, mass_ratio, spin, c = (_number(ctx, value) for value in (xi, q, chi, compactness_value))
    if x < 0 or mass_ratio <= 0 or c <= 0:
        raise ValueError("Expected xi >= 0, q > 0, and C > 0")
    mu = mass_ratio * c
    return (
        x**5
        - 3 * mu * x**4
        + 2 * spin * ctx.sqrt(mu**3 * x**7)
        - 3 * mass_ratio * x**2
        + 6 * mass_ratio * mu * x
        - 3 * mass_ratio * mu**2 * spin**2
    )


def largest_tidal_root(q: float, chi: float, compactness_value, *, dps: int = 70):
    """Bracket the rightmost positive Eq. (10) root in y=sqrt(xi), then bisect.

    The monic degree-ten polynomial's Cauchy bound is a strict upper bound
    on every complex root modulus. Descending from it finds the rightmost
    sign-changing positive root. A separate test enumerates all roots on the
    frozen grid to detect a missed tangency or skipped narrow interval.
    """
    ctx = _context(dps)
    mass_ratio, spin, c = (_number(ctx, value) for value in (q, chi, compactness_value))
    if mass_ratio <= 0 or c <= 0:
        raise ValueError("Expected q > 0 and C > 0")
    mu = mass_ratio * c
    spin_coefficient = 2 * spin * mu ** ctx.mpf("1.5")

    def polynomial_y(y):
        return (
            y**10
            - 3 * mu * y**8
            + spin_coefficient * y**7
            - 3 * mass_ratio * y**4
            + 6 * mass_ratio * mu * y**2
            - 3 * mass_ratio * mu**2 * spin**2
        )

    bound = 1 + max(
        3 * mu, abs(spin_coefficient), 3 * mass_ratio, 6 * mass_ratio * mu, 3 * mass_ratio * mu**2 * spin**2
    )
    hi = bound
    if polynomial_y(hi) <= 0:
        raise ArithmeticError("Cauchy upper bound failed")
    lo = hi * ctx.mpf("0.95")
    while lo > ctx.mpf("1e-20") and polynomial_y(lo) >= 0:
        hi, lo = lo, lo * ctx.mpf("0.95")
    if polynomial_y(lo) >= 0:
        raise ArithmeticError("No positive sign-changing tidal root was bracketed")
    for _ in range(4 * dps):
        mid = (lo + hi) / 2
        if polynomial_y(mid) < 0:
            lo = mid
        else:
            hi = mid
    xi = ((lo + hi) / 2) ** 2
    if abs(polynomial_y(ctx.sqrt(xi))) > ctx.mpf("1e-" + str(dps // 2)):
        raise ArithmeticError("Tidal root did not converge")
    return xi


def disruption(q: float, chi: float, lambda_ns: float, mass2_solar: float = 1.4, *, dps: int = 70) -> Disruption:
    """Return the paper's compactness, root, radius, and signed frequency."""
    ctx = _context(dps)
    c = compactness(lambda_ns, dps=dps)
    xi = largest_tidal_root(q, chi, c, dps=dps)
    mass2_sec = (
        ctx.mpf(str(mass2_solar))
        * ctx.mpf("6.67430e-11")
        * ctx.mpf("1.9884098706980507e30")
        / ctx.mpf("299792458") ** 3
    )
    mass1_sec = ctx.mpf(str(q)) * mass2_sec
    radius = xi * mass2_sec * (1 - 2 * c) / c
    denominator = ctx.pi * (ctx.mpf(str(chi)) * mass1_sec + ctx.sqrt(radius**3 / mass1_sec))
    frequency = 1 / denominator if denominator != 0 else ctx.inf
    return Disruption(c, xi, radius, frequency)

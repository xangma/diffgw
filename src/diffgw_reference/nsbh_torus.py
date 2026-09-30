"""Scalar Thompson et al. (2020) Appendix A Eqs. (11)--(12).

The printed torus fit and the nonnegative native-carrier adapter are separate.
This module has no production-backend or LAL dependency.
"""

from dataclasses import dataclass

import mpmath

from diffgw_reference.nsbh_disruption import compactness, largest_tidal_root


@dataclass(frozen=True)
class Torus:
    isco_radius: mpmath.mpf
    raw_ratio: mpmath.mpf
    carrier_ratio: mpmath.mpf


def isco_radius(chi: float, *, dps: int = 70):
    """Kerr unit-mass ISCO radius from Thompson Appendix A Eq. (12)."""
    ctx = mpmath.mp.clone()
    ctx.dps = dps
    spin = ctx.mpf(str(chi))
    if abs(spin) > 1:
        raise ValueError("Kerr spin must be within [-1, 1]")
    z1 = 1 + (1 - spin**2) ** (ctx.mpf(1) / 3) * (
        (1 + spin) ** (ctx.mpf(1) / 3) + (1 - spin) ** (ctx.mpf(1) / 3)
    )
    z2 = ctx.sqrt(3 * spin**2 + z1**2)
    return 3 + z2 - ctx.sign(spin) * ctx.sqrt((3 - z1) * (3 + z1 + 2 * z2))


def torus(q: float, chi: float, lambda_ns: float, *, dps: int = 70) -> Torus:
    """Return raw Eq. (11) and its physical-mass clipping adapter."""
    ctx = mpmath.mp.clone()
    ctx.dps = dps
    mass_ratio = ctx.mpf(str(q))
    if mass_ratio < 1:
        raise ValueError("NSBH mass ratio must be at least one")
    c = compactness(lambda_ns, dps=dps)
    xi = largest_tidal_root(q, chi, c, dps=dps)
    r_isco = isco_radius(chi, dps=dps)
    raw = ctx.mpf("0.296") * xi * (1 - 2 * c) - ctx.mpf("0.171") * mass_ratio * c * r_isco
    return Torus(r_isco, raw, max(ctx.mpf(0), raw))

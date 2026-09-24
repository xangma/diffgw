"""Scalar physical opening-angle limit for an MSA one-spin/aligned row.

Pratten et al., arXiv:2004.06503v2, Appendix C Eqs. (67)-(70), give the
source-frame angular-momentum geometry. Chatziioannou et al.,
arXiv:1703.03967v3, Eqs. (41), (45), and (171), give the MSA ``J`` evolution,
leading spin average, and opening-angle geometry used by XP.
This module evaluates their zero-nutation, constant-``|S|`` specialization;
it does not encode a version-specific split-root or Euler-angle chart.

All angular momenta are divided by total mass squared. ``v_ref`` is the
dimensionless reference velocity and ``l_beta`` is the chosen positive orbital
angular-momentum magnitude at the target frequency, e.g. a published PN
expression supplied by the caller. Using a PN ``l_beta`` in place of the
Newtonian ``eta/v`` is an explicit choice, not an assertion about Eq. (41)'s
original PN order. No production backend is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


SOURCE_PRATTEN = "https://arxiv.org/html/2004.06503v2"
SOURCE_CHATZIIOANNOU = "https://arxiv.org/abs/1703.03967v3"


@dataclass(frozen=True)
class RegularizedBeta:
    eta: float
    spin_squared: float
    spin_perpendicular: float
    s_average: float
    j_ref_squared: float
    c1: float
    j_beta: float
    cos_beta: float
    beta: float


def one_spin_aligned_beta(
    mass1: float,
    mass2: float,
    chi1: tuple[float, float, float],
    chi2: tuple[float, float, float],
    *,
    v_ref: float,
    l_beta: float,
) -> RegularizedBeta:
    """Return the continuous physical opening angle for a zero-nutation row.

    At least one spin must be parallel to the reference orbital axis. With
    ``S=(m1/M)^2 chi1+(m2/M)^2 chi2``, ``L0=eta/v_ref`` and
    ``J0=|L0 zhat+S|``, the physical invariants are

    ``Sav=|S|^2``, ``c1=v_ref*(J0^2-L0^2-Sav)/2=eta*S_z``.

    Hence ``J_beta^2=l_beta^2+2*c1*l_beta/eta+Sav`` and the geometric
    opening-angle identity is

    ``cos(beta)=(J_beta^2+l_beta^2-|S|^2)/(2*J_beta*l_beta)``.

    We evaluate the algebraically identical ``atan2(|S_perp|,l_beta+S_z)``
    and ``(l_beta+S_z)/hypot(l_beta+S_z,|S_perp|)`` to retain the tiny-angle
    and reversed-``J`` limits without cancellation. At ``J_beta=0`` the
    opening angle is undefined.
    """
    if len(chi1) != 3 or len(chi2) != 3:
        raise ValueError("each spin must have three components")
    values = (mass1, mass2, *chi1, *chi2, v_ref, l_beta)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all inputs must be finite")
    if mass1 <= 0 or mass2 <= 0 or v_ref <= 0 or l_beta <= 0:
        raise ValueError("masses, v_ref, and l_beta must be positive")
    if not ((chi1[0] == chi1[1] == 0) or (chi2[0] == chi2[1] == 0)):
        raise ValueError("the constant-spin limit requires one aligned spin")

    total = mass1 + mass2
    w1, w2 = (mass1 / total) ** 2, (mass2 / total) ** 2
    eta = mass1 * mass2 / total**2
    sx = w1 * chi1[0] + w2 * chi2[0]
    sy = w1 * chi1[1] + w2 * chi2[1]
    sz = w1 * chi1[2] + w2 * chi2[2]
    s_perp = math.hypot(sx, sy)
    s_squared = s_perp**2 + sz**2
    l_ref = eta / v_ref
    j_ref_squared = (l_ref + sz) ** 2 + s_perp**2
    # This is exactly v_ref*(J0^2-L0^2-Sav)/2 after substituting L0=eta/v_ref.
    # The reduced form avoids cancellation for very small transverse spins.
    c1 = eta * sz
    j_parallel = l_beta + sz
    j_beta = math.hypot(j_parallel, s_perp)
    if j_beta == 0:
        raise ValueError("the opening angle is undefined when J_beta is zero")
    return RegularizedBeta(
        eta=eta,
        spin_squared=s_squared,
        spin_perpendicular=s_perp,
        s_average=s_squared,
        j_ref_squared=j_ref_squared,
        c1=c1,
        j_beta=j_beta,
        cos_beta=j_parallel / j_beta,
        beta=math.atan2(s_perp, j_parallel),
    )

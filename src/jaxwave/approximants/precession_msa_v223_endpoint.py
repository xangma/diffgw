"""Nonpublic JAX primitives for the aligned native-MSA v223 endpoint.

The secular coefficients and anchored primitives follow Chatziioannou et al.,
arXiv:1703.03967v3, Appendices D and F.  The remnant identity follows
Pratten et al., arXiv:2004.06503v2, Sec. IV D.  The split spin roots and the
total-mass-normalized remnant are explicitly *v223 compatibility* choices;
they are not the strict physical beta=0 limit.  Observer-frame transport and
the finite-beta Eqs. (11)--(13) projection are deliberately outside this module.
"""

from __future__ import annotations

import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC

from .imrphenom_xas import XASParameterFits
from .precession_msa import msa_phase_primitives


def _opening_l_3pn(v, eta, delta, spin_l, sigma_l):
    """Nonconserved-spin circular L, with the v223 opening-beta 259/72 term.

    The nonspinning 3PN term is Pratten et al. Appendix G; the spin terms are
    Blanchet, Buonanno and Faye, arXiv:gr-qc/0605140v4, Eq. (6.9).
    """
    l3 = (
        135.0 / 16.0
        + (41.0 * jnp.pi**2 / 24.0 - 6889.0 / 144.0) * eta
        + 31.0 * eta**2 / 24.0
        + 7.0 * eta**3 / 1296.0
    )
    return eta / v * (
        1.0
        + (1.5 + eta / 6.0) * v**2
        + (-23.0 * spin_l / 6.0 - 1.5 * delta * sigma_l) * v**3
        + (27.0 / 8.0 - 19.0 * eta / 8.0 + eta**2 / 24.0) * v**4
        + ((-77.0 / 8.0 + 259.0 * eta / 72.0) * spin_l
           + delta * (-33.0 / 8.0 + 7.0 * eta / 4.0) * sigma_l) * v**5
        + l3 * v**6
    )


def _secular_coefficients(eta, xi, c1, s1sq, s2sq, delta, sp, sm, g):
    """Appendix-D secular alpha and zeta coefficients at split roots."""
    g0, _g1, g2, g3, g4, _g5 = jnp.moveaxis(g, -1, 0)
    gap = sp - sm
    cp, cm = sp * eta**2 - c1**2, sm * eta**2 - c1**2
    # Native repeated-root continuation: the strict equation branch does not
    # take abs or floor this product.  Keep this distinction explicit.
    root = jnp.sqrt(jnp.maximum(jnp.abs(cp * cm), 1.0e-32))
    ad = (-3.0 * (s1sq - s2sq) * eta * delta
          + 3.0 * c1 / eta * (c1 - 2.0 * xi * eta**2)) / (4.0 * root)
    cd = 3.0 * gap**2 / (128.0 * eta * root)
    hd = c1 / eta**2 * (1.0 - (cp + cm) / (2.0 * root))
    fd = (cp + cm) / (8.0 * eta**4) * (1.0 - (cp + cm) / (2.0 * root))
    oz0 = 0.5 + 0.75 / eta + ad
    oz1 = -0.75 * xi / eta - ad * xi - ad * hd
    oz2 = ad * hd * xi + cd - ad * fd + ad * hd**2
    oz3 = (ad * fd - cd - ad * hd**2) * (xi + hd) + ad * fd * hd
    oz4 = (cd + ad * hd**2 - 2.0 * ad * fd) * (hd * xi + hd**2 - fd) - ad * fd**2
    gd = 3.0 * gap**2 / (64.0 * eta**3) * (c1 - eta**2 * xi) / root
    ze = (
        oz0,
        oz1 + c1 / eta**2 * oz0,
        oz2 + c1 / eta**2 * oz1,
        oz3 + c1 / eta**2 * oz2 + gd,
        oz4 + c1 / eta**2 * oz3 - gd * xi - gd * hd,
    )
    alpha = jnp.stack((
        3.0 * g0 * oz0,
        3.0 * g0 * oz1,
        3.0 * (g0 * oz2 + g2 * oz0),
        3.0 * (g0 * oz3 + g2 * oz1 + g3 * oz0),
        3.0 * (g0 * oz4 + g2 * oz2 + g3 * oz1 + g4 * oz0),
        jnp.zeros_like(oz0),  # v223 order=5
    ), axis=-1)
    zeta = jnp.stack((
        -g0 * ze[0],
        -1.5 * g0 * ze[1],
        -3.0 * (g0 * ze[2] + g2 * ze[0]),
        3.0 * (g0 * ze[3] + g2 * ze[1] + g3 * ze[0]),
        3.0 * (g0 * ze[4] + g2 * ze[2] + g3 * ze[1] + g4 * ze[0]),
        jnp.zeros_like(oz0),  # v223 order=5
    ), axis=-1)
    return alpha, zeta


def aligned_msa_v223_endpoint(frequency, mass1, mass2, spin1z, spin2z, f_ref):
    """Return intrinsic aligned v223 alpha, gamma, cos(beta), and remnant.

    Requires JAX x64; masses are solar masses and frequencies are Hz.  Inputs
    must be physical (positive masses and frequencies, spin magnitudes <= 1).
    The caller must supply exactly aligned spins.  Parameter axes precede the
    frequency axis.  This is a research primitive and does not change the
    public XP dispatcher or implement observer-azimuth transport.  Individual
    mean-spin projections are ill conditioned at equal mass under the v223
    ``1e-16`` denominator continuation; their sum and the remnant remain
    well defined.
    """
    f = jnp.atleast_1d(jnp.asarray(frequency, dtype=jnp.float64))
    m1, m2, chi1, chi2, fr = jnp.broadcast_arrays(*(
        jnp.asarray(x, dtype=jnp.float64)
        for x in (mass1, mass2, spin1z, spin2z, f_ref)
    ))
    swap = m2 > m1
    m1, m2, chi1, chi2 = (
        jnp.where(swap, m2, m1), jnp.where(swap, m1, m2),
        jnp.where(swap, chi2, chi1), jnp.where(swap, chi1, chi2),
    )
    total = m1 + m2
    n1, n2 = m1 / total, m2 / total
    eta, q, delta = n1 * n2, n2 / n1, n1 - n2
    s1, s2 = n1**2 * chi1, n2**2 * chi2
    st = s1 + s2
    s1sq, s2sq = s1**2, s2**2
    xi = (1.0 + q) * s1 + (1.0 + 1.0 / q) * s2
    v0 = (jnp.pi * total * MSUN_SEC * fr) ** (1.0 / 3.0)
    l0 = eta / v0
    j0sq = (l0 + st) ** 2
    # v223's aligned/one-spin root continuation, distinct from the exact
    # repeated-root solution (0, S^2, S^2).
    sm = st**2
    sp = sm + 1.0e-9
    sav = 0.5 * (sm + sp)
    c1 = 0.5 * v0 * (j0sq - l0**2 - sav)

    # Initial-spin flux couplings, rather than the MSA-averaged couplings.
    beta3 = (113.0 / 12.0 + 25.0 * q / 4.0) * s1 + (113.0 / 12.0 + 25.0 / (4.0 * q)) * s2
    beta5 = ((31319.0 / 1008.0 - 1159.0 * eta / 24.0 + q * (809.0 / 84.0 - 281.0 * eta / 8.0)) * s1
             + (31319.0 / 1008.0 - 1159.0 * eta / 24.0 + (809.0 / 84.0 - 281.0 * eta / 8.0) / q) * s2)
    s12 = s1 * s2
    sigma4 = ((247.0 * s12 - 721.0 * s12) / (48.0 * eta)
              + ((233.0 * s1sq - 719.0 * s1sq) / n1**2
                 + (233.0 * s2sq - 719.0 * s2sq) / n2**2) / 96.0)
    a0 = 96.0 * eta / 5.0
    a2 = -743.0 / 336.0 - 11.0 * eta / 4.0
    a3 = 4.0 * jnp.pi - beta3
    a4 = 34103.0 / 18144.0 + 13661.0 * eta / 2016.0 + 59.0 * eta**2 / 18.0 - sigma4
    a5 = -4159.0 * jnp.pi / 672.0 - 189.0 * jnp.pi * eta / 8.0 - beta5
    g0 = 1.0 / a0
    g2, g3 = -a2 / a0, -a3 / a0
    g4, g5 = (a2**2 - a4) / a0, (2.0 * a2 * a3 - a5) / a0
    g = jnp.stack((g0, jnp.zeros_like(g0), g2, g3, g4, g5), axis=-1)
    ac, zc = _secular_coefficients(eta, xi, c1, s1sq, s2sq, delta, sp, sm, g)

    v = (jnp.pi * total[..., None] * MSUN_SEC * f) ** (1.0 / 3.0)
    eta_u, c1_u, sav_u = eta[..., None], c1[..., None], sav[..., None]
    primitives = msa_phase_primitives(v, eta_u, c1_u, sav_u)
    primitives0 = msa_phase_primitives(v0[..., None], eta_u, c1_u, sav_u)
    alpha = jnp.sum((primitives - primitives0) * ac[..., None, :], axis=-1)
    def zeta_primitives(x):
        return eta_u[..., None] * jnp.stack((
            x**-3, x**-2, x**-1, jnp.log(x), x, x**2,
        ), axis=-1)
    zeta = jnp.sum((zeta_primitives(v) - zeta_primitives(v0[..., None]))
                   * zc[..., None, :], axis=-1)
    gamma = -zeta

    spin_l = st
    sigma_l = n2 * chi2 - n1 * chi1
    lb = _opening_l_3pn(v, eta_u, delta[..., None], spin_l[..., None], sigma_l[..., None])
    jb = jnp.sqrt(lb**2 + 2.0 * c1_u * lb / eta_u + sav_u)
    cos_beta = (jb**2 + lb**2 - sp[..., None]) / (2.0 * jb * lb)

    # Pratten et al. Sec. IV D, with the documented native total-M^2 L_fin
    # and remnant normalization.  The 1e-16 terms are the v223 mean-coupling
    # continuation at equal mass, not a new fit.
    den = eta * (1.0 - q**2 + 1.0e-16)
    mean1 = (c1 * (1.0 + q) - q * eta * xi) / den
    mean2 = -q * (c1 * (1.0 + q) - eta * xi) / den
    a_parallel = XASParameterFits.fit_final_spin(eta, chi1, chi2)
    l_fin = a_parallel - st
    final_sq = sav + l_fin**2 + 2.0 * l_fin * (mean1 + mean2)
    final_spin = jnp.where(a_parallel < 0.0, -1.0, 1.0) * jnp.sqrt(jnp.maximum(final_sq, 0.0))
    return {
        "alpha": alpha, "gamma": gamma, "cos_beta": cos_beta,
        "final_spin": final_spin, "sav": sav, "c1": c1,
        "mean_s1_l": mean1, "mean_s2_l": mean2,
    }


__all__ = ["aligned_msa_v223_endpoint"]

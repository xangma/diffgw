"""Intrinsic native-MSA v223 limit when at least one spin is aligned.

The constant-spin reduction of Chatziioannou et al., arXiv:1703.03967v3,
Eqs. (41), (45), (65) and Appendices D/F gives the secular precession angles.
The finite separation of the two upper auxiliary roots is an explicitly named
v223 compatibility prescription, not physical nutation.  Pratten et al.,
arXiv:2004.06503v2, Eq. (17) instead fixes the physical opening angle.  This
module is an intrinsic diagnostic; it does not select an observer frame or
change public XP dispatch.
"""

from __future__ import annotations

import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC

from .imrphenom_xas import XASParameterFits
from .precession_msa import msa_phase_primitives
from .precession_msa_physical_beta import one_spin_aligned_beta
from .precession_msa_v223_endpoint import _opening_l_3pn, _secular_coefficients
from .precession_msa_v223_generic import _flux


def one_spin_msa_v223_angles(frequency, mass1, mass2, spin1, spin2, f_ref):
    """Return zero-nutation intrinsic angles, physical beta and MSA remnant.

    Masses are solar masses and frequency is in Hz.  Spin vectors have a final
    Cartesian axis of length three; at least one vector has exactly zero
    transverse components.  Parameter axes precede the frequency axis.  JAX
    x64 is required for the small auxiliary root separation.  This diagnostic
    assumes positive masses/frequencies and a nonsingular total-J chart; its
    ``valid`` mask also reports the physical beta domain.

    ``c1`` and ``sav`` describe the v223 auxiliary-root continuation used by
    the secular angles and remnant.  ``physical_c1`` and ``cos_beta`` are the
    unsplit constant-spin quantities; no auxiliary-root amplitude enters the
    physical opening angle.
    """
    f = jnp.atleast_1d(jnp.asarray(frequency, dtype=jnp.float64))
    chi1 = jnp.asarray(spin1, dtype=jnp.float64)
    chi2 = jnp.asarray(spin2, dtype=jnp.float64)
    if chi1.ndim == 0 or chi2.ndim == 0 or chi1.shape[-1] != 3 or chi2.shape[-1] != 3:
        raise ValueError("spin vectors must have a final axis of length three")
    m1, m2, fr, *components = jnp.broadcast_arrays(
        jnp.asarray(mass1, dtype=jnp.float64),
        jnp.asarray(mass2, dtype=jnp.float64),
        jnp.asarray(f_ref, dtype=jnp.float64),
        *jnp.moveaxis(chi1, -1, 0),
        *jnp.moveaxis(chi2, -1, 0),
    )
    chi1 = jnp.stack(components[:3], axis=-1)
    chi2 = jnp.stack(components[3:], axis=-1)
    swap = m2 > m1
    cm1, cm2 = jnp.where(swap, m2, m1), jnp.where(swap, m1, m2)
    cs1 = jnp.where(swap[..., None], chi2, chi1)
    cs2 = jnp.where(swap[..., None], chi1, chi2)

    total = cm1 + cm2
    n1, n2 = cm1 / total, cm2 / total
    eta, q, delta = n1 * n2, n2 / n1, n1 - n2
    s1, s2 = n1[..., None] ** 2 * cs1, n2[..., None] ** 2 * cs2
    st = s1 + s2
    spin_sq = jnp.sum(st**2, axis=-1)
    spin_l = st[..., 2]
    s1sq = jnp.sum(s1**2, axis=-1)
    s2sq = jnp.sum(s2**2, axis=-1)
    xi = (1.0 + q) * s1[..., 2] + (1.0 + 1.0 / q) * s2[..., 2]
    v0 = (jnp.pi * total * MSUN_SEC * fr) ** (1.0 / 3.0)
    l0 = eta / v0
    j0sq = l0**2 + 2.0 * l0 * spin_l + spin_sq

    # At fixed spin magnitude S_+^2=S_-^2=|S|^2 by Eq. (45).  The v223
    # secular/remnant continuation uses separated auxiliary roots.  Their
    # 1e-9 separation is never interpreted as a physical spin amplitude.
    sm = spin_sq
    sp = sm + 1.0e-9
    sav = 0.5 * (sm + sp)
    c1 = 0.5 * v0 * (j0sq - l0**2 - sav)
    flux = _flux(eta, q, s1, s2)
    ac, zc = _secular_coefficients(eta, xi, c1, s1sq, s2sq, delta, sp, sm, flux)

    v = (jnp.pi * total[..., None] * MSUN_SEC * f) ** (1.0 / 3.0)
    v0u = v0[..., None]
    etau, c1u, savu = eta[..., None], c1[..., None], sav[..., None]
    primitives = msa_phase_primitives(v, etau, c1u, savu)
    primitives0 = msa_phase_primitives(v0u, etau, c1u, savu)
    alpha = jnp.sum((primitives - primitives0) * ac[..., None, :], axis=-1)

    def zeta_primitives(x):
        return etau[..., None] * jnp.stack(
            (x**-3, x**-2, x**-1, jnp.log(x), x, x**2), axis=-1,
        )

    zeta = jnp.sum(
        (zeta_primitives(v) - zeta_primitives(v0u)) * zc[..., None, :],
        axis=-1,
    )
    gamma = -zeta

    sigma_l = n2 * cs2[..., 2] - n1 * cs1[..., 2]
    lb = _opening_l_3pn(
        v, etau, delta[..., None], spin_l[..., None], sigma_l[..., None],
    )
    physical = one_spin_aligned_beta(cm1, cm2, cs1, cs2, v_ref=v0, l_beta=lb)

    # Pratten et al., Sec. IV D remnant identity.  The split-root mean
    # projections and total-M^2 normalization here are named v223 choices.
    den = eta * (1.0 - q**2 + 1.0e-16)
    mean1 = (c1 * (1.0 + q) - q * eta * xi) / den
    mean2 = -q * (c1 * (1.0 + q) - eta * xi) / den
    a_parallel = XASParameterFits.fit_final_spin(eta, cs1[..., 2], cs2[..., 2])
    l_fin = a_parallel - spin_l
    final_sq = sav + l_fin**2 + 2.0 * l_fin * (mean1 + mean2)
    final_spin = jnp.where(a_parallel < 0.0, -1.0, 1.0) * jnp.sqrt(
        jnp.maximum(final_sq, 0.0)
    )
    return {
        "alpha": alpha, "gamma": gamma,
        "cos_beta": physical["cos_beta"], "beta": physical["beta"],
        "valid": physical["valid"], "final_spin": final_spin,
        "spin_squared": spin_sq, "sav": sav, "c1": c1,
        "physical_c1": physical["c1"],
        "mean_s1_l": mean1, "mean_s2_l": mean2,
        "secular_phi": alpha, "secular_zeta": zeta,
    }


__all__ = ["one_spin_msa_v223_angles"]

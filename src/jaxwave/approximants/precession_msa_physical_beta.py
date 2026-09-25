"""Paper-derived, zero-nutation MSA opening angle in JAX.

This is the constant-|S|, one-spin/aligned limit of the angular-momentum
geometry in Pratten et al., arXiv:2004.06503v2, Appendix C Eqs. (67)-(70),
and Chatziioannou et al., arXiv:1703.03967v3, Eqs. (41), (45), (171).
It is independent of the v223 split-root continuation and Euler chart.
The caller chooses a positive orbital angular momentum ``l_beta``; a PN
choice must be supplied explicitly. This module does not change public XP.
"""

from __future__ import annotations

import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC


def one_spin_aligned_beta(mass1, mass2, chi1, chi2, *, v_ref, l_beta):
    """Return physical zero-nutation invariants and opening angle.

    Matches ``diffgw_reference.xp_msa_regularized_beta.one_spin_aligned_beta``.
    ``chi1`` and ``chi2`` have a final Cartesian-component axis of length 3.
    Parameter axes broadcast; ``l_beta`` may add target-frequency axes after
    them. For a batch and a frequency grid, pass ``l_beta`` as ``(batch, freq)``.
    JAX x64 is needed to resolve very small transverse spins.

    Exactly one or both spins must be aligned with the reference orbital axis.
    ``valid`` marks that precondition, finite positive inputs, and nonzero J.
    The angle outputs are NaN where it fails; callers must reject such rows.
    At exact alignment the transverse-spin derivative of beta is undefined;
    the selected JAX subgradient is zero. Longitudinal derivatives remain finite.
    """
    s1 = jnp.asarray(chi1, dtype=jnp.float64)
    s2 = jnp.asarray(chi2, dtype=jnp.float64)
    if s1.ndim == 0 or s2.ndim == 0 or s1.shape[-1] != 3 or s2.shape[-1] != 3:
        raise ValueError("each spin must have a final axis of length 3")
    m1, m2, vr, s1x, s1y, s1z, s2x, s2y, s2z = jnp.broadcast_arrays(
        jnp.asarray(mass1, dtype=jnp.float64),
        jnp.asarray(mass2, dtype=jnp.float64),
        jnp.asarray(v_ref, dtype=jnp.float64),
        *jnp.moveaxis(s1, -1, 0),
        *jnp.moveaxis(s2, -1, 0),
    )
    total = m1 + m2
    eta = m1 * m2 / total**2
    w1, w2 = (m1 / total) ** 2, (m2 / total) ** 2
    sx, sy = w1 * s1x + w2 * s2x, w1 * s1y + w2 * s2y
    sz = w1 * s1z + w2 * s2z
    perp_sq = sx**2 + sy**2
    # A safe inactive branch avoids sqrt'(0) poisoning longitudinal gradients.
    s_perp = jnp.where(perp_sq > 0, jnp.sqrt(jnp.where(perp_sq > 0, perp_sq, 1.0)), 0.0)
    s_squared = perp_sq + sz**2
    l_ref = eta / vr
    j_ref_squared = (l_ref + sz) ** 2 + perp_sq
    c1 = eta * sz  # Reduced Eq. (41) invariant, without tiny-spin cancellation.

    lb = jnp.asarray(l_beta, dtype=jnp.float64)
    added_axes = max(lb.ndim - m1.ndim, 0)
    expand = (...,) + (None,) * added_axes
    sz_target, sp_target = sz[expand], s_perp[expand]
    j_parallel = lb + sz_target
    j_squared = j_parallel**2 + sp_target**2
    j_beta = jnp.where(j_squared > 0, jnp.sqrt(jnp.where(j_squared > 0, j_squared, 1.0)), 0.0)
    safe_j = jnp.where(j_beta > 0, j_beta, 1.0)
    source_valid = (
        jnp.isfinite(m1) & jnp.isfinite(m2) & jnp.isfinite(vr)
        & jnp.isfinite(s1x) & jnp.isfinite(s1y) & jnp.isfinite(s1z)
        & jnp.isfinite(s2x) & jnp.isfinite(s2y) & jnp.isfinite(s2z)
        & (m1 > 0) & (m2 > 0) & (vr > 0)
        & (((s1x == 0) & (s1y == 0)) | ((s2x == 0) & (s2y == 0)))
    )
    valid = source_valid[expand] & jnp.isfinite(lb) & (lb > 0) & (j_beta > 0)
    return {
        "eta": eta,
        "spin_squared": s_squared,
        "spin_perpendicular": s_perp,
        "s_average": s_squared,
        "j_ref_squared": j_ref_squared,
        "c1": c1,
        "j_beta": j_beta,
        "cos_beta": jnp.where(valid, j_parallel / safe_j, jnp.nan),
        "beta": jnp.where(valid, jnp.arctan2(sp_target, j_parallel), jnp.nan),
        "valid": valid,
    }


def one_spin_aligned_beta_newtonian_grid(frequency, mass1, mass2, chi1, chi2, *, f_ref):
    """Evaluate the paper's Newtonian ``L=eta/v`` on a frequency grid.

    Nonpositive bins use ``f_ref`` only to keep all arithmetic finite. Their
    ``positive_frequency`` mask tells the waveform caller to zero its output.
    This adapter does not assert that Newtonian L is XP's chosen PN L.
    Parameter axes precede the frequency axis; frequencies are in Hz and
    masses in solar masses. ``f_ref`` must be positive.
    """
    s1, s2 = jnp.asarray(chi1), jnp.asarray(chi2)
    if s1.ndim == 0 or s2.ndim == 0 or s1.shape[-1] != 3 or s2.shape[-1] != 3:
        raise ValueError("each spin must have a final axis of length 3")
    m1, m2, fr, *_ = jnp.broadcast_arrays(
        jnp.asarray(mass1, dtype=jnp.float64),
        jnp.asarray(mass2, dtype=jnp.float64),
        jnp.asarray(f_ref, dtype=jnp.float64),
        *jnp.moveaxis(s1, -1, 0),
        *jnp.moveaxis(s2, -1, 0),
    )
    f = jnp.atleast_1d(jnp.asarray(frequency, dtype=jnp.float64))
    positive = f > 0
    f_safe = jnp.where(positive, f, fr[..., None])
    total = m1 + m2
    eta = m1 * m2 / total**2
    v_ref = (jnp.pi * total * MSUN_SEC * fr) ** (1.0 / 3.0)
    v = (jnp.pi * total[..., None] * MSUN_SEC * f_safe) ** (1.0 / 3.0)
    result = one_spin_aligned_beta(
        m1, m2, chi1, chi2, v_ref=v_ref, l_beta=eta[..., None] / v,
    )
    return {**result, "positive_frequency": jnp.broadcast_to(positive, result["beta"].shape)}


__all__ = ["one_spin_aligned_beta", "one_spin_aligned_beta_newtonian_grid"]

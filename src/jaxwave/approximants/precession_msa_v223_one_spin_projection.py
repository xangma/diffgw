"""Source-derived v223 XP quadrupole for the zero-nutation one-spin branch.

Pratten et al., arXiv:2004.06503v2, Eqs. (8)--(13), (17), (67)--(85)
and Appendix C specify the opening angle, rotation and observer frame.
The explicitly labelled v223 source-frame and Table-IV chart choices match
the existing clean Torch adapter.  This module contains no LAL code.
"""

from __future__ import annotations

import jax.numpy as jnp

from .imrphenom_xas import IMRPhenomXAS
from .precession_msa_v223_generic_projection import (
    _principal_delta,
    generic_source_frame_geometry_v223,
)
from .precession_msa_v223_projection import (
    aligned_source_frame_geometry_v223,
    project_l2_quadrupole,
)
from .precession_msa_v223_zero_nutation import one_spin_msa_v223_angles


def one_spin_msa_v223_waveform(
    frequency, mass1, mass2, spin1, spin2, *,
    distance=100.0, inclination=0.0, phic=0.0, tc=0.0,
    f_ref=20.0, convention=1, phasing_convention=None,
    reference_frame=None, return_state=False,
):
    """Evaluate the physical one-spin XP boundary, parameter axes first.

    ``phic`` is the Appendix-C source phase ``public phic - public phi0``.
    The caller supplies positive evaluation frequencies (zero bins are masked
    by the public adapter).  Canonical mass order is the caller's responsibility.
    """
    if convention not in (0, 1):
        raise ValueError("convention must be 0 or 1")
    f = jnp.atleast_1d(jnp.asarray(frequency, dtype=jnp.float64)).reshape(-1)
    m1, m2, dist, iota, phi, time, fr = jnp.broadcast_arrays(*(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (mass1, mass2, distance, inclination, phic, tc, f_ref)
    ))
    chi1 = jnp.broadcast_to(jnp.asarray(spin1, dtype=jnp.float64), m1.shape + (3,))
    chi2 = jnp.broadcast_to(jnp.asarray(spin2, dtype=jnp.float64), m1.shape + (3,))
    intrinsic = one_spin_msa_v223_angles(f, m1, m2, chi1, chi2, fr)
    geometry = generic_source_frame_geometry_v223(m1, m2, chi1, chi2, iota, phi, fr)
    zero_geometry = generic_source_frame_geometry_v223(
        m1, m2, chi1, chi2, iota, jnp.zeros_like(phi), fr,
    )
    transverse_sq = jnp.sum(chi1[..., :2] ** 2 + chi2[..., :2] ** 2, axis=-1)
    near_aligned = transverse_sq <= 1.0e-28
    exact_aligned = transverse_sq == 0.0
    aligned = aligned_source_frame_geometry_v223(
        m1, m2, chi1[..., 2], chi2[..., 2], iota, phi, fr,
    )
    aligned_zero = aligned_source_frame_geometry_v223(
        m1, m2, chi1[..., 2], chi2[..., 2], iota, jnp.zeros_like(phi), fr,
    )
    for key in ("alpha0", "theta_jn", "zeta", "phi_j"):
        geometry[key] = jnp.where(exact_aligned, aligned[key], geometry[key])
        zero_geometry[key] = jnp.where(exact_aligned, aligned_zero[key], zero_geometry[key])
    alpha0 = geometry["alpha0"] + jnp.where(near_aligned, jnp.pi, 0.0)
    alpha0_zero = zero_geometry["alpha0"] + jnp.where(near_aligned, jnp.pi, 0.0)
    delta_alpha = _principal_delta(alpha0, alpha0_zero)
    delta_phi_j = _principal_delta(geometry["phi_j"], zero_geometry["phi_j"])
    alpha = intrinsic["alpha"] + alpha0_zero[..., None] + delta_alpha[..., None]
    gamma = intrinsic["gamma"]
    zeta = geometry["zeta"]
    if convention == 0:
        carrier_phase = -geometry["phi_j"]
    else:
        gamma = gamma - jnp.where(near_aligned, 0.0, delta_phi_j)[..., None]
        phi_anchor = jnp.where(near_aligned, geometry["phi_j"], zero_geometry["phi_j"])
        gamma = gamma + jnp.pi - phi_anchor[..., None]
        # The exact aligned endpoint uses its own v223 Euler chart.
        alpha = alpha - jnp.where(exact_aligned, phi_anchor, 0.0)[..., None]
        gamma = gamma + jnp.where(exact_aligned, phi_anchor, 0.0)[..., None]
        zeta = zeta - jnp.pi / 2.0
        carrier_phase = jnp.zeros_like(phi)
    carrier_plus, _ = IMRPhenomXAS.evaluate(
        f=f, mass1=m1, mass2=m2, chi1=chi1[..., 2], chi2=chi2[..., 2],
        distance=dist, inclination=jnp.zeros_like(iota), phi_c=carrier_phase,
        t_c=time, f_ref=fr, phasing_convention=phasing_convention,
        reference_frame=reference_frame,
        _final_spin_override=intrinsic["final_spin"],
    )
    hneg = 2.0 * carrier_plus / jnp.sqrt(5.0 / (4.0 * jnp.pi))
    hp, hc = project_l2_quadrupole(
        hneg, alpha, gamma, intrinsic["cos_beta"], geometry["theta_jn"], zeta,
    )
    if return_state:
        return hp, hc, dict(
            alpha=alpha, gamma=gamma, cos_beta=intrinsic["cos_beta"],
            theta_jn=geometry["theta_jn"], zeta=zeta,
            phi_j=geometry["phi_j"], carrier_phase=carrier_phase,
            final_spin=intrinsic["final_spin"], valid=intrinsic["valid"],
        )
    return hp, hc


__all__ = ["one_spin_msa_v223_waveform"]

"""Generic two-spin v223 source frame and quadrupole composition.

Pratten et al., arXiv:2004.06503v2, Eqs. (67)--(85) fix the source,
inertial-J and observer frames.  Their Eqs. (5)--(13) and Appendix A fix the
positive-frequency quadrupole rotation and polarization projection.  The
native-v223 source-frame 259/75 coefficient and Table-IV convention transport
are separately labelled compatibility prescriptions.  This module composes
the intrinsic JAX MSA angles with XAS for the scoped public XP route.

The separate public observer azimuth is fixed to phi0=0 here.  ``phic`` is the
source-frame reference phase passed to the clean Torch helper and LAL's
``phiRef`` argument; it is not that observer azimuth.
"""

from __future__ import annotations

import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC

from .imrphenom_xas import IMRPhenomXAS
from .precession_msa_v223_endpoint import _opening_l_3pn
from .precession_msa_v223_generic import generic_msa_v223_angles
from .precession_msa_v223_projection import project_l2_quadrupole


def generic_source_frame_geometry_v223(
    mass1, mass2, spin1, spin2, inclination, phic, f_ref,
):
    """Return the generic Appendix-C geometry in the native v223 L chart.

    Spin vectors are dimensionless Cartesian components in the source L0
    frame.  The reference orbital L uses the compatibility-derived 259/75
    coefficient, distinct from the published 259/72 opening-angle series.
    The generic branch requires nonzero J and a nondegenerate J/N frame.
    """
    m1, m2, iota, phi, fr = jnp.broadcast_arrays(*(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (mass1, mass2, inclination, phic, f_ref)
    ))
    chi1, chi2 = jnp.broadcast_arrays(
        jnp.asarray(spin1, dtype=jnp.float64),
        jnp.asarray(spin2, dtype=jnp.float64),
    )
    total = m1+m2
    n1, n2 = m1/total, m2/total
    eta, delta = n1*n2, n1-n2
    spin = n1[..., None]**2*chi1+n2[..., None]**2*chi2
    spin_l = spin[..., 2]
    sigma_l = n2*chi2[..., 2]-n1*chi1[..., 2]
    v = (jnp.pi*total*MSUN_SEC*fr)**(1/3)
    l_ref = (_opening_l_3pn(v, eta, delta, spin_l, sigma_l)
             -(259/1800)*eta**2*spin_l*v**4)
    j = spin+jnp.stack((jnp.zeros_like(l_ref), jnp.zeros_like(l_ref), l_ref), -1)
    jhat = j/jnp.linalg.norm(j, axis=-1, keepdims=True)
    observer = jnp.stack((jnp.sin(iota)*jnp.sin(phi),
                          jnp.sin(iota)*jnp.cos(phi), jnp.cos(iota)), -1)
    observer_x = jnp.stack((-jnp.cos(iota)*jnp.sin(phi),
                            -jnp.cos(iota)*jnp.cos(phi), jnp.sin(iota)), -1)
    y_raw = jnp.cross(jhat, observer)
    y_norm = jnp.linalg.norm(y_raw, axis=-1)
    tangent = jnp.cross(jhat, observer_x)
    tangent_norm = jnp.linalg.norm(tangent, axis=-1)
    eps = jnp.finfo(jnp.float64).eps
    yj = jnp.where((y_norm > eps)[..., None],
                   y_raw/jnp.where(y_norm > eps, y_norm, 1.0)[..., None],
                   tangent/jnp.where(tangent_norm > eps, tangent_norm, 1.0)[..., None])
    xj = jnp.cross(yj, jhat)
    alpha_r = jnp.hypot(yj[..., 2], xj[..., 2])
    alpha0 = jnp.arctan2(jnp.where(alpha_r > eps, yj[..., 2], 0.0),
                         jnp.where(alpha_r > eps, xj[..., 2], 1.0))
    j_perp = jnp.hypot(j[..., 0], j[..., 1])
    phi_j = jnp.where(j_perp > eps, jnp.arctan2(j[..., 1], j[..., 0]),
                      jnp.pi/2-phi)
    jn = jnp.clip(jnp.sum(jhat*observer, axis=-1), -1.0, 1.0)
    theta_jn = jnp.arctan2(jnp.linalg.norm(jnp.cross(jhat, observer), axis=-1), jn)
    x_sphere = jnp.cos(theta_jn)[..., None]*xj-jnp.sin(theta_jn)[..., None]*jhat
    zeta = jnp.arctan2(jnp.sum(observer_x*yj, axis=-1),
                       jnp.sum(observer_x*x_sphere, axis=-1))
    return dict(alpha0=alpha0, theta_jn=theta_jn, zeta=zeta, phi_j=phi_j,
                l_reference=l_ref, j=j)


def _principal_delta(current, initial):
    difference = current-initial
    return jnp.arctan2(jnp.sin(difference), jnp.cos(difference))


def generic_msa_v223_waveform(
    frequency, mass1, mass2, spin1, spin2, *,
    distance=100.0, inclination=0.0, phic=0.0, tc=0.0, f_ref=20.0,
    convention=1, phasing_convention=None, reference_frame=None,
    return_state=False,
):
    """Compose generic v223 source-frame XP at public observer phi0=0.

    ``convention`` is the clean Table-IV convention 0 or 1.  Parameter axes
    precede the frequency axis.  This diagnostic requires JAX x64, unequal
    masses, generic nondegenerate roots, and a regular J/N observer frame.
    """
    if convention not in (0, 1):
        raise ValueError("convention must be 0 or 1")
    requested_f = jnp.atleast_1d(jnp.asarray(frequency, dtype=jnp.float64))
    # The PN phase primitives contain inverse powers of v.  A nonpositive
    # output bin has zero physical positive-frequency strain, so evaluate its
    # inactive algebra at a positive bin before masking the completed strain.
    positive_min = jnp.min(jnp.where(requested_f > 0.0, requested_f, jnp.inf))
    surrogate = jnp.where(jnp.isfinite(positive_min), positive_min, 20.0)
    f = jnp.where(requested_f > 0.0, requested_f, surrogate)
    m1, m2, dist, iota, phi, time, fr = jnp.broadcast_arrays(*(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (mass1, mass2, distance, inclination, phic, tc, f_ref)
    ))
    chi1 = jnp.asarray(spin1, dtype=jnp.float64)
    chi2 = jnp.asarray(spin2, dtype=jnp.float64)
    intrinsic = generic_msa_v223_angles(f, m1, m2, chi1, chi2, fr)
    geometry = generic_source_frame_geometry_v223(m1, m2, chi1, chi2, iota, phi, fr)
    zero_geometry = generic_source_frame_geometry_v223(
        m1, m2, chi1, chi2, iota, jnp.zeros_like(phi), fr,
    )
    delta_alpha = _principal_delta(geometry["alpha0"], zero_geometry["alpha0"])
    delta_phi_j = _principal_delta(geometry["phi_j"], zero_geometry["phi_j"])
    alpha = intrinsic["alpha"]+delta_alpha[..., None]+zero_geometry["alpha0"][..., None]
    gamma = intrinsic["gamma"]
    zeta = geometry["zeta"]
    if convention == 0:
        carrier_phase = -geometry["phi_j"]
    else:
        gamma = (gamma-delta_phi_j[..., None]
                 +jnp.pi-zero_geometry["phi_j"][..., None])
        zeta = zeta-jnp.pi/2
        carrier_phase = jnp.zeros_like(phi)
    carrier_plus, _ = IMRPhenomXAS.evaluate(
        f=f, mass1=m1, mass2=m2, chi1=chi1[..., 2], chi2=chi2[..., 2],
        distance=dist, inclination=jnp.zeros_like(iota),
        phi_c=carrier_phase, t_c=time, f_ref=fr,
        phasing_convention=phasing_convention,
        reference_frame=reference_frame,
        _final_spin_override=intrinsic["final_spin"],
    )
    hneg = 2*carrier_plus/jnp.sqrt(5/(4*jnp.pi))
    hp, hc = project_l2_quadrupole(
        hneg, alpha, gamma, intrinsic["cos_beta"], geometry["theta_jn"], zeta,
    )
    hp = jnp.where(requested_f > 0.0, hp, 0.0j)
    hc = jnp.where(requested_f > 0.0, hc, 0.0j)
    if return_state:
        return hp, hc, dict(
            alpha=alpha, gamma=gamma, cos_beta=intrinsic["cos_beta"],
            theta_jn=geometry["theta_jn"], zeta=zeta,
            phi_j=geometry["phi_j"], alpha0=geometry["alpha0"],
            carrier_phase=carrier_phase, final_spin=intrinsic["final_spin"],
            delta_alpha=delta_alpha, delta_phi_j=delta_phi_j,
            l_reference=geometry["l_reference"], j=geometry["j"],
        )
    return hp, hc


__all__ = ["generic_msa_v223_waveform", "generic_source_frame_geometry_v223"]

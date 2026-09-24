"""Nonpublic aligned v223 observer geometry and finite-beta XP projection.

Pratten et al., arXiv:2004.06503v2, Appendix C fixes the observer basis;
Eqs. (8)--(13) and Appendix A specify the quadrupole rotation.  The reference orbital L,
aligned Euler chart, split-root beta, and convention-1 endpoint correction
below are separately labelled v223 compatibility choices.  This diagnostic
does not change public XP dispatch or assert a strict physical beta=0 limit.
"""

from __future__ import annotations

import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC

from .imrphenom_xas import IMRPhenomXAS
from .precession_msa_v223_endpoint import (
    _opening_l_3pn,
    aligned_msa_v223_endpoint,
)


def aligned_source_frame_geometry_v223(
    mass1, mass2, spin1z, spin2z, inclination, phi_ref, f_ref,
):
    """Appendix-C observer geometry for exactly aligned spins.

    Returns ``alpha0, theta_jn, zeta, phi_j``.  The source-frame reference L
    uses the inferred v223-compatible 259/75 spin coefficient, distinct from
    the 259/72 opening-beta L.  At J perpendicular = 0, the compatibility
    chart assigns ``phi_j = pi/2 - phi_ref``; ``alpha0`` is the geometric zero before
    the v223 waveform's separate +pi aligned-chart shift.
    """
    m1, m2, chi1, chi2, iota, phi, fr = jnp.broadcast_arrays(*(
        jnp.asarray(x, dtype=jnp.float64)
        for x in (mass1, mass2, spin1z, spin2z, inclination, phi_ref, f_ref)
    ))
    total = m1 + m2
    n1, n2 = m1 / total, m2 / total
    eta, delta = n1 * n2, n1 - n2
    spin_l = n1**2 * chi1 + n2**2 * chi2
    sigma_l = n2 * chi2 - n1 * chi1
    v = (jnp.pi * total * MSUN_SEC * fr) ** (1.0 / 3.0)
    # The inferred v223-compatible source-frame 259/75 coefficient differs
    # from the opening-beta 259/72 coefficient by -259 eta^2 S_l v^4 / 1800.
    l_ref = (_opening_l_3pn(v, eta, delta, spin_l, sigma_l)
             - (259.0 / 1800.0) * eta**2 * spin_l * v**4)
    jz = l_ref + spin_l
    j_up = jz >= 0.0
    return {
        "alpha0": jnp.zeros_like(iota),
        "theta_jn": jnp.where(j_up, iota, jnp.pi - iota),
        "zeta": jnp.where(j_up, jnp.pi, 0.0),
        "phi_j": jnp.pi / 2.0 - phi,
        "jz": jz,
    }


def _d_l2_end(m: int, mp: int, c, s):
    """The m'=+/-2 columns of the real spin-2 Wigner small-d matrix."""
    c2, s2 = c * c, s * s
    cs = c * s
    if mp == 2:
        return (s2 * s2, 2.0 * s2 * cs, jnp.sqrt(6.0) * c2 * s2,
                2.0 * c2 * cs, c2 * c2)[m + 2]
    if mp == -2:
        return (c2 * c2, -2.0 * c2 * cs, jnp.sqrt(6.0) * c2 * s2,
                -2.0 * s2 * cs, s2 * s2)[m + 2]
    raise ValueError("mprime must be +2 or -2")


def project_l2_quadrupole(hneg, alpha, gamma, cos_beta, theta_jn, zeta):
    """Project one positive-frequency ``h_{2,-2}`` via Eqs. (11)--(13).

    ``hneg`` has the carrier normalization before multiplying the spin-weighted
    harmonic.  Its final axis is frequency.  ``cos_beta`` retains the finite
    v223 opening beta; no physical beta=0 substitution is made.
    """
    hneg = jnp.asarray(hneg, dtype=jnp.complex128)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    cos_beta = jnp.asarray(cos_beta, dtype=jnp.float64)
    theta = jnp.asarray(theta_jn, dtype=jnp.float64)[..., None]
    zeta = jnp.asarray(zeta, dtype=jnp.float64)[..., None]
    cb = jnp.sqrt(jnp.maximum((1.0 + cos_beta) / 2.0, 0.0))
    sb = jnp.sqrt(jnp.maximum((1.0 - cos_beta) / 2.0, 0.0))
    ct, st = jnp.cos(theta / 2.0), jnp.sin(theta / 2.0)
    pref = jnp.exp(2.0j * gamma) * hneg
    hp = jnp.zeros_like(pref)
    hc = jnp.zeros_like(pref)
    for m in range(-2, 3):
        harmonic = jnp.sqrt(5.0 / (4.0 * jnp.pi)) * _d_l2_end(m, 2, ct, st)
        azimuth = jnp.exp(-1.0j * m * alpha) * harmonic
        minus = azimuth * _d_l2_end(m, -2, cb, sb)
        plus = azimuth * _d_l2_end(m, 2, cb, sb)
        hp = hp + 0.5 * pref * (minus + jnp.conj(plus))
        hc = hc + 0.5j * pref * (minus - jnp.conj(plus))
    c2, s2 = jnp.cos(2.0 * zeta), jnp.sin(2.0 * zeta)
    return c2 * hp + s2 * hc, c2 * hc - s2 * hp


def aligned_msa_v223_waveform(
    frequency, mass1, mass2, spin1z, spin2z, *,
    distance=100.0, inclination=0.0, phic=0.0, observer_azimuth=0.0,
    tc=0.0, f_ref=20.0, convention=0, return_state=False,
):
    """Compose an exactly aligned, nonpublic JAX v223 quadrupole waveform.

    ``phic`` is the clean helper's source-frame phase (the public XP ``phic``
    at zero ``phi0``).  ``observer_azimuth`` is the *separate* public-XP-like
    ``phi0`` coordinate.  At exact alignment, its action on the physical
    quadrupole is the spin-2 factor ``exp(-2i phi0)`` from the azimuth of the
    spin-weighted harmonic.  Here that strict SO(2) identity is imposed after
    the finite-beta diagnostic projection; it is not a claim that the
    current public nonzero-phi0 legacy route or LAL implements this endpoint.
    Convention 0/1 otherwise follows the clean v223 chart and Table IV.
    Parameter axes precede frequency.  JAX x64 is required.
    """
    if convention not in (0, 1):
        raise ValueError("convention must be 0 or 1")
    f = jnp.atleast_1d(jnp.asarray(frequency, dtype=jnp.float64))
    m1, m2, chi1, chi2, dist, iota, phi, phi0, tc, fr = jnp.broadcast_arrays(*(
        jnp.asarray(x, dtype=jnp.float64)
        for x in (mass1, mass2, spin1z, spin2z, distance,
                  inclination, phic, observer_azimuth, tc, f_ref)
    ))
    endpoint = aligned_msa_v223_endpoint(f, m1, m2, chi1, chi2, fr)
    geometry = aligned_source_frame_geometry_v223(m1, m2, chi1, chi2, iota, phi, fr)
    # At exact alignment alpha0(phi)-alpha0(0)=0; aligned convention-1 gamma
    # transport is omitted.  The explicit v223 endpoint chart then shifts
    # alpha by -phi_j and gamma by +phi_j before Table-IV assembly.
    alpha = endpoint["alpha"] + jnp.pi
    gamma = endpoint["gamma"]
    zeta = geometry["zeta"]
    if convention == 0:
        carrier_phase = -geometry["phi_j"]
    else:
        alpha = alpha - geometry["phi_j"][..., None]
        gamma = gamma + jnp.pi
        zeta = zeta - jnp.pi / 2.0
        carrier_phase = jnp.zeros_like(phi)
    carrier_plus, _ = IMRPhenomXAS.evaluate(
        f=f, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
        distance=dist, inclination=jnp.zeros_like(iota),
        phi_c=carrier_phase, t_c=tc, f_ref=fr,
        _final_spin_override=endpoint["final_spin"],
    )
    hneg = 2.0 * carrier_plus / jnp.sqrt(5.0 / (4.0 * jnp.pi))
    hp, hc = project_l2_quadrupole(
        hneg, alpha, gamma, endpoint["cos_beta"], geometry["theta_jn"], zeta,
    )
    # Exact alignment has axial SO(2) symmetry.  Apply the independent
    # observer azimuth to the completed spin-2 strain so the v223 auxiliary
    # beta floor cannot spuriously mix m != 2 in this physical group action.
    observer_rotation = jnp.exp(-2.0j * phi0)[..., None]
    hp, hc = observer_rotation * hp, observer_rotation * hc
    if return_state:
        return hp, hc, {
            "alpha": alpha, "gamma": gamma, "cos_beta": endpoint["cos_beta"],
            "theta_jn": geometry["theta_jn"], "zeta": zeta,
            "phi_j": geometry["phi_j"], "carrier_phase": carrier_phase,
            "final_spin": endpoint["final_spin"], "jz": geometry["jz"],
            "observer_azimuth": phi0,
        }
    return hp, hc


__all__ = [
    "aligned_msa_v223_waveform",
    "aligned_source_frame_geometry_v223",
    "project_l2_quadrupole",
]

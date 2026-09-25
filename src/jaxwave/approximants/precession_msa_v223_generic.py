"""Nonpublic intrinsic two-spin MSA v223 angle diagnostic.

The spin cubic, nutation phase, secular coefficients and oscillatory rational
integral follow Chatziioannou et al., arXiv:1703.03967v3, Eqs. (21), (24),
(51), (65) and Appendices B, D and F.  The opening angle and averaged-remnant
identity follow Pratten et al., arXiv:2004.06503v2, Sec. IV and Appendix G.

Compatibility choices are explicit: the flux uses initial spin contractions;
the quadratic nutation coefficient uses the v223 q² reweighting; the pole
integral uses algebraic pole labels, absolute residues, and the second-pole
sign; opening beta uses nonconserved-spin 3PN L; and the remnant uses total-M²
normalization.  These are implementation contracts, not claims that the
published equations themselves prescribe those continuations.  This module
does not change public waveform dispatch or observer-frame transport.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC
from .imrphenom_xas import XASParameterFits
from .precession_msa import elliptic_f, jacobi_sn_cn_dn, msa_phase_primitives
from .precession_msa_v223_endpoint import _opening_l_3pn, _secular_coefficients


def _cubic(q, eta, delta, s1sq, s2sq, l, j, xi):
    """Chatziioannou Appendix B cubic for instantaneous total spin squared."""
    j2, l2 = j**2, l**2
    b = (l2+s1sq)*q + 2*l*xi - 2*j2 - s1sq-s2sq + (l2+s2sq)/q
    c = ((j2-l2)**2 - 2*l*xi*(j2-l2)
         - 2*(1-q)/q*(s1sq-q*s2sq)*l2 + 4*eta*l2*xi**2
         - 2*delta*(s1sq-s2sq)*xi*l
         + 2*(1-q)/q*(q*s1sq-s2sq)*j2)
    d = ((1-q)/q*(s2sq-q*s1sq)*(j2-l2)**2
         + delta**2/eta*(s1sq-s2sq)**2*l2
         + 2*delta*l*xi*(s1sq-s2sq)*(j2-l2))
    return b, c, d


def _roots(b, c, d):
    """Real trigonometric roots with the native physical-root sign choice."""
    p = c-b**2/3
    p_safe = jnp.where(p < 0, p, -jnp.ones_like(p))
    r = 2*b**3/27 - b*c/3 + d
    arg = 3*r/(2*p_safe)*jnp.sqrt(-3/p_safe)
    clamped = jnp.clip(arg, -1.0, 1.0)
    # At a cubic endpoint, keep the exact root value and a finite inactive
    # one-sided derivative, as in the audited Torch diagnostic.
    acos_arg = jnp.where(jnp.abs(clamped) >= 1.0,
                         jax.lax.stop_gradient(clamped), clamped)
    theta = jnp.arccos(acos_arg)/3
    k = jnp.arange(3, dtype=b.dtype)
    candidates = -b[..., None]/3 + 2*jnp.sqrt(-p_safe[..., None]/3)*jnp.cos(
        theta[..., None]-2*jnp.pi*k/3
    )
    ordered = jnp.sort(candidates, axis=-1)
    physical = jnp.sort(jnp.abs(ordered[..., 1:]), axis=-1)
    return jnp.concatenate((ordered[..., :1], physical), axis=-1)


def _flux(eta, q, s1, s2):
    """Reference spin contractions in the Appendix A--C reciprocal flux."""
    s1l, s2l = s1[..., 2], s2[..., 2]
    s1sq, s2sq = jnp.sum(s1**2, -1), jnp.sum(s2**2, -1)
    s12 = jnp.sum(s1*s2, -1)
    x1, x2 = 1/(1+q), q/(1+q)
    beta3 = (113/12+25*q/4)*s1l + (113/12+25/(4*q))*s2l
    beta5 = ((31319/1008-1159*eta/24+q*(809/84-281*eta/8))*s1l
             +(31319/1008-1159*eta/24+(809/84-281*eta/8)/q)*s2l)
    sigma4 = ((247*s12-721*s1l*s2l)/(48*eta)
              +((233*s1sq-719*s1l**2)/x1**2
                +(233*s2sq-719*s2l**2)/x2**2)/96)
    a0 = 96*eta/5
    a2 = -743/336-11*eta/4
    a3 = 4*jnp.pi-beta3
    a4 = 34103/18144+13661*eta/2016+59*eta**2/18-sigma4
    a5 = -4159*jnp.pi/672-189*jnp.pi*eta/8-beta5
    g0 = 1/a0
    return jnp.stack((g0, jnp.zeros_like(g0), -a2/a0, -a3/a0,
                      (a2**2-a4)/a0, (2*a2*a3-a5)/a0), axis=-1)


def _psi_coefficients(eta, delta, q, xi, c1, sav, s1sq, s2sq, g):
    """Eq. (51) limiting-root phase with the named v223 quadratic choice."""
    g0, g2 = g[..., 0], g[..., 2]
    bm2, bm1 = eta*delta**2, 2*eta*xi-4*c1
    b0 = q*s1sq+s2sq/q-s1sq-s2sq-2*sav
    cm2 = 4*c1**2-4*eta*c1*xi+4*eta**3*xi**2-2*eta*delta**2*(s1sq+s2sq)
    dm2 = (4*(1-q)/q*(s2sq-q*s1sq)*c1**2
           +eta*delta**2*(s1sq-s2sq)**2
           +4*delta*eta*c1*xi*(s1sq-s2sq))
    disc = jnp.maximum(cm2**2-4*bm2*dm2, 1e-24)
    candidates = jnp.stack(((-cm2+jnp.sqrt(disc))/(2*bm2),
                            (-cm2-jnp.sqrt(disc))/(2*bm2)), axis=-1)
    rp, rn = jnp.max(candidates, -1), jnp.min(candidates, -1)
    r1 = bm1/(2*bm2)
    r2 = (b0+2*rp+rn)/(2*bm2)-bm1**2/(8*bm2**2)
    psi1 = 1.5*(r1-xi)
    independent = 3*(r2-xi*r1+g2/g0)
    psi2 = 3*g2/g0 + q**2*(independent-3*g2/g0)
    return psi1, psi2


def _oscillatory(psi, v, eta, xi, s1sq, s2sq, delta, roots, j, l):
    """Appendix B/F rational primitive with the audited v223 pole choice."""
    s3, sm, sp = jnp.moveaxis(roots, -1, 0)
    gap = sp-sm
    j2 = j**2
    c0 = .75*(1-xi*v)*v**2*(eta**3+4*eta**3*xi*v
         -2*eta*(j2-sp+2*(s1sq-s2sq)*delta)*v**2
         -4*eta*xi*(j2-sp)*v**3+(j2-sp)**2/eta*v**4)
    c2 = -1.5*eta*gap*(1+2*xi*v-(j2-sp)/eta**2*v**2)*(1-xi*v)*v**4
    c4 = .75/eta*gap**2*(1-xi*v)*v**6
    d0 = -(j2-(l+jnp.sqrt(sp))**2)*(j2-(l-jnp.sqrt(sp))**2)
    d2 = -2*gap*(j2+l**2-sp)
    d4 = -gap**2
    rate = 3/(4*jnp.sqrt(eta))*(1-xi*v)*v**6*jnp.sqrt(sp-s3)
    disc = d2**2-4*d0*d4
    root = jnp.sqrt(jnp.abs(disc))
    la, lb = (-d2+root)/(2*d0), (-d2-root)/(2*d0)
    r0 = c4/d4
    u = c0/d0-r0
    w = c2/d0+r0*(la+lb)
    ra = (w+u*la)/(la-lb)
    rb = (-w-u*lb)/(la-lb)
    def branch(lam):
        distance = jnp.abs(1-lam)
        k = jnp.sqrt(jnp.where(distance > 0, distance, 1.0))
        primitive = jnp.arctan2((1-k)*jnp.sin(psi)*jnp.cos(psi),
                                jnp.cos(psi)**2+k*jnp.sin(psi)**2)/k
        return jnp.where(distance > 0, primitive, 0.0)
    ba, bb = branch(la), branch(lb)
    sign_a = jnp.sign(la)
    sign_b = jnp.sign(d0*(d0+d2+d4))*jnp.sign(lb)
    factor_a = (j2+l**2-sp)/(2*j*l)+gap/(2*j*l*la)
    factor_b = (j2+l**2-sp)/(2*j*l)+gap/(2*j*l*lb)
    prefactor = j/jnp.abs(rate)
    return (prefactor*(sign_a*jnp.abs(ra)*ba+sign_b*jnp.abs(rb)*bb),
            prefactor*(sign_a*jnp.abs(ra)*factor_a*ba
                       +sign_b*jnp.abs(rb)*factor_b*bb))


def generic_msa_v223_angles(frequency, mass1, mass2, spin1, spin2, f_ref):
    """Return intrinsic generic v223 angles and MSA remnant, for two spins.

    Masses are solar masses; spins are dimensionless Cartesian vectors with
    last axis 3; parameter axes precede the frequency axis.  Requires JAX
    x64.  This first generic branch requires distinct, nondegenerate roots
    and unequal masses; aligned/equal-mass endpoints use the companion
    endpoint module.  It is a diagnostic primitive, not public dispatch.
    """
    f = jnp.atleast_1d(jnp.asarray(frequency, dtype=jnp.float64))
    m1, m2, fr = jnp.broadcast_arrays(*(
        jnp.asarray(x, dtype=jnp.float64) for x in (mass1, mass2, f_ref)
    ))
    chi1 = jnp.asarray(spin1, dtype=jnp.float64)
    chi2 = jnp.asarray(spin2, dtype=jnp.float64)
    swap = m2 > m1
    cm1, cm2 = jnp.where(swap, m2, m1), jnp.where(swap, m1, m2)
    cs1 = jnp.where(swap[..., None], chi2, chi1)
    cs2 = jnp.where(swap[..., None], chi1, chi2)
    total = cm1+cm2
    n1, n2 = cm1/total, cm2/total
    eta, q, delta = n1*n2, n2/n1, n1-n2
    s1, s2 = n1[..., None]**2*cs1, n2[..., None]**2*cs2
    s1sq, s2sq = jnp.sum(s1**2, -1), jnp.sum(s2**2, -1)
    st = s1+s2
    ssum2 = jnp.sum(st**2, -1)
    xi = (1+q)*s1[..., 2]+(1+1/q)*s2[..., 2]
    v0 = (jnp.pi*total*MSUN_SEC*fr)**(1/3)
    l0 = eta/v0
    j0 = jnp.sqrt(l0**2+2*l0*(s1[..., 2]+s2[..., 2])+ssum2)
    roots0 = _roots(*_cubic(q, eta, delta, s1sq, s2sq, l0, j0, xi))
    s30, sm0, sp0 = jnp.moveaxis(roots0, -1, 0)
    sav = (sp0+sm0)/2
    c1 = .5*v0*(j0**2-l0**2-sav)
    den = eta*(1-q**2+1e-16)
    mean1 = (c1*(1+q)-q*eta*xi)/den
    mean2 = -q*(c1*(1+q)-eta*xi)/den
    flux = _flux(eta, q, s1, s2)
    psi1, psi2 = _psi_coefficients(eta, delta, q, xi, c1, sav, s1sq, s2sq, flux)
    phase_parameter = (sp0-sm0)/(sp0-s30)
    fraction = jnp.clip((sp0-ssum2)/(sp0-sm0), 0.0, 1.0)
    triple = s1[..., 0]*s2[..., 1]-s1[..., 1]*s2[..., 0]
    psi0 = jnp.where(triple < 0, -1.0, 1.0)*elliptic_f(
        jnp.arcsin(jnp.sqrt(fraction)), phase_parameter
    )
    ac, zc = _secular_coefficients(
        eta, xi, c1, s1sq, s2sq, delta, sp0, sm0, flux
    )
    v = (jnp.pi*total[..., None]*MSUN_SEC*f)**(1/3)
    v0u = v0[..., None]
    etau, c1u, savu = eta[..., None], c1[..., None], sav[..., None]
    def raw_phase(x):
        return -3*flux[..., 0, None]*delta[..., None]/4*x**-3*(
            1+psi1[..., None]*x+psi2[..., None]*x**2
        )
    psi = psi0[..., None]+raw_phase(v)-raw_phase(v0u)
    secular_alpha = jnp.sum((msa_phase_primitives(v, etau, c1u, savu)
                            -msa_phase_primitives(v0u, etau, c1u, savu))
                            *ac[..., None, :], axis=-1)
    def zeta_primitives(x):
        return etau[..., None]*jnp.stack(
            (x**-3, x**-2, x**-1, jnp.log(x), x, x**2), -1
        )
    secular_zeta = jnp.sum((zeta_primitives(v)-zeta_primitives(v0u))
                            *zc[..., None, :], axis=-1)
    l = etau/v
    j = jnp.sqrt(l**2+2*c1u/v+savu)
    roots = _roots(*_cubic(q[..., None], etau, delta[..., None],
                           s1sq[..., None], s2sq[..., None], l, j,
                           xi[..., None]))
    s3, sm, sp = jnp.moveaxis(roots, -1, 0)
    parameter = (sp-sm)/(sp-s3)
    sn, _, _ = jacobi_sn_cn_dn(psi, parameter)
    spin_squared = sp+(sm-sp)*sn**2
    osc_alpha, osc_zeta = _oscillatory(
        psi, v, etau, xi[..., None], s1sq[..., None], s2sq[..., None],
        delta[..., None], roots, j, l,
    )
    osc0_alpha, osc0_zeta = _oscillatory(
        psi0, v0, eta, xi, s1sq, s2sq, delta, roots0, j0, l0,
    )
    alpha = secular_alpha+osc_alpha-osc0_alpha[..., None]
    gamma = -(secular_zeta+osc_zeta-osc0_zeta[..., None])
    spin_l = s1[..., 2]+s2[..., 2]
    sigma_l = n2*cs2[..., 2]-n1*cs1[..., 2]
    lb = _opening_l_3pn(v, etau, delta[..., None], spin_l[..., None],
                        sigma_l[..., None])
    jb = jnp.sqrt(lb**2+2*c1u*lb/etau+savu)
    cos_beta = (jb**2+lb**2-spin_squared)/(2*jb*lb)
    aligned = XASParameterFits.fit_final_spin(eta, cs1[..., 2], cs2[..., 2])
    lfin = aligned-spin_l
    final_sq = sav+lfin**2+2*lfin*(mean1+mean2)
    final_spin = jnp.where(aligned < 0, -1.0, 1.0)*jnp.sqrt(jnp.maximum(final_sq, 0.0))
    return dict(alpha=alpha, gamma=gamma, cos_beta=cos_beta,
                final_spin=final_spin, c1=c1, sav=sav,
                roots_reference=roots0, roots=roots, psi=psi,
                psi_initial=psi0, psi1=psi1, psi2=psi2,
                spin_squared=spin_squared,
                mean_s1_l=mean1, mean_s2_l=mean2,
                secular_phi=secular_alpha, secular_zeta=secular_zeta,
                oscillatory_phi=osc_alpha, oscillatory_zeta=osc_zeta,
                flux_coefficients=flux)


__all__ = ["generic_msa_v223_angles"]

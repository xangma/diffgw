"""Native JAX port of the clean-room Pv2 source-frame and N10 phase prescription.

Equations and compatibility conventions follow the independently derived
``torchwave.approximants.phenompv2_waveform_clean`` implementation. No PyTorch
code is imported or executed. The legacy IMRPhenomP model remains separate.
"""
import math

import jax
import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC
from .imrphenomd import IMRPhenomD, evaluate_d_qnm_220
from .imrphenomp import nnlo_angle_coefficients, evaluate_angles, _safe_in_plane_norm


def _norm(x):
    squared = jnp.sum(x * x)
    nonzero = squared > 0
    return jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, squared, 1.0)), 0.0)


def _atan2(y, x):
    defined = (x != 0) | (y != 0)
    return jnp.arctan2(jnp.where(defined, y, 0.0), jnp.where(defined, x, 1.0))


def _natural_cubic_derivative(x, y, query):
    """Derivative of the ten-node natural cubic, differentiable in x and y."""
    step = x[1] - x[0]
    count = x.size - 2
    matrix = 4.0 * jnp.eye(count) + jnp.diag(jnp.ones(count - 1), 1) + jnp.diag(jnp.ones(count - 1), -1)
    rhs = 6.0 / step**2 * (y[:-2] - 2 * y[1:-1] + y[2:])
    moments = jnp.concatenate((jnp.zeros(1), jnp.linalg.solve(matrix, rhs), jnp.zeros(1)))
    interval = jnp.clip(jnp.searchsorted(x, query, side="right") - 1, 0, x.size - 2)
    dx = query - x[interval]
    b = (y[interval + 1] - y[interval]) / step - step * (2 * moments[interval] + moments[interval + 1]) / 6
    return b + moments[interval] * dx + (moments[interval + 1] - moments[interval]) / (2 * step) * dx**2


def _L2(m1, m2, f):
    v = (math.pi * (m1 + m2) * MSUN_SEC * f)**(1 / 3)
    eta = m1 * m2 / (m1 + m2)**2
    return eta / v * (1 + (1.5 + eta / 6) * v**2 + (27 / 8 - 19 * eta / 8 - eta**2 / 24) * v**4)


def _geometry(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi, fref):
    n1, n2 = m1 / (m1 + m2), m2 / (m1 + m2)
    L = _L2(m1, m2, fref)
    S = jnp.stack((n1**2*s1x+n2**2*s2x, n1**2*s1y+n2**2*s2y, n1**2*s1z+n2**2*s2z))
    J = S + jnp.stack((jnp.zeros_like(L), jnp.zeros_like(L), L))
    j = J / _norm(J)
    n = jnp.stack((jnp.sin(iota)*jnp.sin(phi), jnp.sin(iota)*jnp.cos(phi), jnp.cos(iota)))
    ox = jnp.stack((-jnp.cos(iota)*jnp.sin(phi), -jnp.cos(iota)*jnp.cos(phi), jnp.sin(iota)))
    y = jnp.cross(j, n)
    yn = _norm(y)
    y = jnp.where(yn > 1e-14, y / jnp.where(yn > 1e-14, yn, 1.0), jnp.cross(j, ox))
    yn = _norm(y)
    y = y / jnp.where(yn > 1e-14, yn, 1.0)
    x = jnp.cross(y, j)
    alpha0 = _atan2(y[2], x[2])
    theta = _atan2(_norm(jnp.cross(j, n)), jnp.dot(j, n))
    xs = jnp.cos(theta)*x - jnp.sin(theta)*j
    zeta = _atan2(jnp.dot(ox, y), jnp.dot(ox, xs)) + math.pi / 2
    phi_j = jnp.where(_norm(J[:2]) > 1e-14, -_atan2(J[1], J[0]), math.pi/2 - phi)
    return alpha0, theta, zeta, phi_j


def _wigner_d(l, m, mp, beta):
    cosine, sine = jnp.cos(beta/2), jnp.sin(beta/2)
    fact = math.factorial
    pref = math.sqrt(fact(l+m)*fact(l-m)*fact(l+mp)*fact(l-mp))
    total = jnp.zeros_like(beta)
    for k in range(2*l+1):
        a, q, r, z = l+mp-k, k, m-mp+k, l-m-k
        if min(a, q, r, z) >= 0:
            total = total + (-1.)**(k+m-mp)*pref/(fact(a)*fact(q)*fact(r)*fact(z))*cosine**(2*l+mp-m-2*k)*sine**(m-mp+2*k)
    return total


def _project(h22, alpha, gamma, beta, theta, zeta):
    hp, hc = jnp.zeros_like(h22), jnp.zeros_like(h22)
    for m in range(-2, 3):
        y = math.sqrt(5/(4*math.pi))*_wigner_d(2, m, 2, theta)
        aminus = jnp.exp(-1j*m*alpha)*_wigner_d(2, m, -2, beta)*y
        aplus = jnp.exp(-1j*m*alpha)*_wigner_d(2, m, 2, beta)*y
        pref = jnp.exp(2j*gamma)*h22
        hp = hp + 0.5*pref*(aminus+jnp.conj(aplus))
        hc = hc + 0.5j*pref*(aminus-jnp.conj(aplus))
    c2, s2 = jnp.cos(2*zeta), jnp.sin(2*zeta)
    return c2*hp+s2*hc, c2*hc-s2*hp


def face_on_to_positive_22(face_on):
    """Invert the zero-angle Pv2 projection of face-on plus strain."""
    return 2 * face_on / math.sqrt(5 / (4 * math.pi))


def waveform(f, mass1, mass2, spin1x=0., spin1y=0., spin1z=0.,
             spin2x=0., spin2y=0., spin2z=0., distance=100., inclination=0.,
             phic=0., tc=0., f_ref=None, convention=0,
             phasing_convention=None, reference_frame=None, precession_version="v2"):
    """Generate native Pv2; parameter axes precede the flattened frequency grid.

    As with other JAX models, concrete calls validate values while transformed
    calls require valid physical parameters. Strings and conventions are static.
    """
    if convention not in (0, 1):
        raise ValueError("convention must be 0 or 1")
    if precession_version not in (None, "v2"):
        raise ValueError("clean IMRPhenomPv2 supports only precession_version='v2'")
    for option in (phasing_convention, reference_frame):
        if option not in (None, "coalescence", "reference_frequency"):
            raise ValueError("phasing convention must be 'coalescence' or 'reference_frequency'")
    carrier_phasing = reference_frame or phasing_convention or "coalescence"
    f = jnp.asarray(f, dtype=jnp.float64).reshape(-1)
    if not isinstance(f, jax.core.Tracer) and not bool(jnp.any(f > 0)):
        raise ValueError("frequency grid must contain a positive frequency")
    rf = jnp.min(jnp.where(f > 0, f, jnp.inf)) if f_ref is None else f_ref
    vals = jnp.broadcast_arrays(*[jnp.asarray(x, dtype=jnp.float64) for x in
        (mass1,mass2,spin1x,spin1y,spin1z,spin2x,spin2y,spin2z,distance,inclination,phic,tc,rf)])
    for value in (vals[0], vals[1], vals[8]):
        if not isinstance(value, jax.core.Tracer) and bool(jnp.any(value <= 0)):
            raise ValueError("masses and distance must be positive")
    shape = vals[0].shape

    def single(*a):
        m1,m2,x1,y1,z1,x2,y2,z2,d,inc,pc,t0,rf = a
        swap = m1 < m2
        m1,m2 = jnp.where(swap,m2,m1),jnp.where(swap,m1,m2)
        x1,x2 = jnp.where(swap,x2,x1),jnp.where(swap,x1,x2)
        y1,y2 = jnp.where(swap,y2,y1),jnp.where(swap,y1,y2)
        z1,z2 = jnp.where(swap,z2,z1),jnp.where(swap,z1,z2)
        M = (m1+m2)*MSUN_SEC
        ff = jnp.where(f > 0, f, 1.0)
        v, vr = (math.pi*M*ff)**(1/3), (math.pi*M*rf)**(1/3)
        eta, delta = m1*m2/(m1+m2)**2, (m1-m2)/(m1+m2)
        q = m1/m2
        chip = jnp.maximum((2+1.5/q)*m1*m1*_safe_in_plane_norm(x1,y1), (2+1.5*q)*m2*m2*_safe_in_plane_norm(x2,y2))/((2+1.5/q)*m1*m1)
        n1, n2 = m1/(m1+m2), m2/(m1+m2)
        chil = (m1*z1+m2*z2)/(m1+m2)
        coeffs = nnlo_angle_coefficients(eta, -delta, n1, chip, chil/n1)
        al, ep = evaluate_angles(v, vr, *coeffs)
        alpha0, theta, zeta, phi_j = _geometry(m1,m2,x1,y1,z1,x2,y2,z2,inc,pc,rf)
        beta = jnp.arctan2(chip*n1**2, jnp.abs(_L2(m1,m2,ff)+n1**2*z1+n2**2*z2))
        alpha, gamma = al+alpha0, -ep
        if convention == 1:
            gamma, zeta, carrier_phase = gamma+math.pi-phi_j, zeta-math.pi/2, jnp.zeros_like(phi_j)
        else:
            carrier_phase = phi_j

        def remnant_provider(a,b,za,zb):
            mf,afd,_,_ = IMRPhenomD.remnant_properties(a,b,za,zb)
            ap = jnp.sqrt(afd**2+(chip*(a/(a+b))**2)**2)
            ap = jnp.where(afd < 0, -ap, ap)
            wr,wd = evaluate_d_qnm_220(ap)
            return mf,ap,wr/(mf/(a+b)),wd/(mf/(a+b))

        def d_call(grid, phase, time, return_phase=False):
            return IMRPhenomD.evaluate(grid,m1,m2,z1,z2,distance=d,inclination=0.,phic=phase,tc=time,
                f_ref=rf,phasing_convention=carrier_phasing,remnant_provider=remnant_provider,return_phase=return_phase)

        carrier,_ = d_call(f,carrier_phase,t0)
        _,_,frd_dimless,_ = remnant_provider(m1,m2,z1,z2)
        frd = frd_dimless/M
        low, high = 0.8*frd, jnp.minimum(1.2*frd,0.2/M)
        nodes = low+(high-low)*jnp.arange(10)/9
        _,_,phase_samples = d_call(nodes,0.,0.,True)
        slope = _natural_cubic_derivative(nodes,phase_samples,frd)
        carrier = carrier*jnp.exp(1j*ff*slope)
        hp,hc = _project(face_on_to_positive_22(carrier),alpha,gamma,beta,theta,zeta)
        return jnp.where(f > 0,hp,0.),jnp.where(f > 0,hc,0.)

    hp,hc = jax.vmap(single)(*[value.reshape(-1) for value in vals])
    return hp.reshape(shape+(f.size,)),hc.reshape(shape+(f.size,))


class IMRPhenomPv2:
    """Clean source-frame Pv2, distinct from the legacy IMRPhenomP primitive."""
    evaluate = staticmethod(waveform)

    def __call__(self, *args, **kwargs):
        return waveform(*args, **kwargs)

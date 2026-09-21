"""PhenomT-driven N4LO spin evolution and minimal-rotation transport.

The ODE follows Estelles et al., arXiv:2105.05872, Secs. II.4.1 and II.5.
An orthogonal orbital tangent is transported with angular velocity L x Ldot;
this enforces minimal rotation without differentiating Euler-angle charts.
The intrinsic integration grid includes f_ref exactly and is independent of
requested waveform samples. Cubic Hermite interpolation supplies dense output.
"""
from __future__ import annotations
import math
import jax
import jax.numpy as jnp
from diffgw_core.constants import MSUN_SEC
from diffgw_core import phenomt_fits as fits
from .imrphenomt import IMRPhenomT
from ._phenomt_spin_evolution import spin_derivatives
from ._phenomt_precession import _canonicalize_components
from ._phenomt_precession import _source_geometry as _source_frame
from ._phenomt_precession import _orbital_angular_momentum as _reference_l

def _arrays(values):
    return [jnp.asarray(v, dtype=jnp.float64) for v in values]

_broadcast = jnp.broadcast_arrays
def _sum(x):
    return jnp.sum(x, axis=-1, keepdims=True)

def _norm(x):
    squared = _sum(x*x)
    positive = squared > 0
    # A finite derivative at zero is needed for absent companion spins.
    return _where(positive, _sqrt(_where(positive, squared, squared*0+1)), squared*0)


def _stack(xs):
    return jnp.stack(xs, axis=-1)

def _cat(xs):
    return jnp.concatenate(xs, axis=-1)

def _split(x):
    return jnp.split(x, 4, axis=-1)

_cross = jnp.cross
_where, _sqrt, _cos, _atan2, _abs = jnp.where, jnp.sqrt, jnp.cos, jnp.arctan2, jnp.abs
_zeros_like, _maximum, _minimum, _clip = jnp.zeros_like, jnp.maximum, jnp.minimum, jnp.clip
def _linspace(n, like):
    return jnp.linspace(0, 1, n, dtype=like.dtype)



def _check_reference(fr, fp):
    if not isinstance(fr, jax.core.Tracer) and not isinstance(fp, jax.core.Tracer) and bool(jnp.any(fr > fp)):
        raise ValueError("numerical precession requires f_ref at or below the carrier peak frequency")


def _scan(step, initial, data):
    _, out = jax.lax.scan(step, initial, tuple(jnp.moveaxis(x, -1, 0) for x in data))
    return jnp.moveaxis(out, 0, -2)


def _join_states(initial, nodes):
    return jnp.concatenate((initial[..., None, :], nodes), axis=-2)


def _join_grid(forward, backward):
    return jnp.concatenate((jnp.flip(forward, -1), backward[..., 1:]), axis=-1)


def _join_segments(forward, backward):
    return jnp.concatenate((jnp.flip(forward, -2), backward[..., 1:, :]), axis=-2)


def _search(grid, q):
    lead = jnp.broadcast_shapes(grid.shape[:-1], q.shape[:-1])
    grid = jnp.broadcast_to(grid, lead+(grid.shape[-1],))
    q = jnp.broadcast_to(q, lead+(q.shape[-1],))
    out = jax.vmap(lambda a, b: jnp.searchsorted(a, b, side="right"))(grid.reshape((-1, grid.shape[-1])), q.reshape((-1, q.shape[-1])))
    return jnp.clip(out.reshape(q.shape)-1, 0, grid.shape[-1]-2)


def _gather(x, i):
    lead = jnp.broadcast_shapes(x.shape[:-1], i.shape[:-1])
    return jnp.take_along_axis(jnp.broadcast_to(x, lead+(x.shape[-1],)), jnp.broadcast_to(i, lead+(i.shape[-1],)), axis=-1)


def _gather_state(x, i):
    lead = jnp.broadcast_shapes(x.shape[:-2], i.shape[:-1])
    ix = jnp.broadcast_to(i, lead+(i.shape[-1],))[..., None]
    return jnp.take_along_axis(jnp.broadcast_to(x, lead+x.shape[-2:]), ix, axis=-2)


def _unit(v):
    return v / _maximum(_norm(v), 1e-30)


def _normalize(state, spin_norms):
    l, s1, s2, e = _split(state)
    l = _unit(l)
    s1 = s1 * spin_norms[0] / _maximum(_norm(s1), 1e-30)
    s2 = s2 * spin_norms[1] / _maximum(_norm(s2), 1e-30)
    e = _unit(e - _sum(e * l) * l)
    return _cat((l, s1, s2, e))


def _rates(q, v, state, x1, x2):
    l, s1, s2, e = _split(state)
    dl, ds1, ds2 = spin_derivatives(v, l, s1, s2, x1, x2)
    de = _cross(_cross(l, dl), e)
    # q = (1 - (t/M)/20)^(3/8) approximately resolves precession phase.
    du_dq = -(160.0 / 3.0) * q ** (5.0 / 3.0)
    return _cat((dl, ds1, ds2, de)) * du_dq[..., None]


def _segment(qnodes, initial, x1, x2, carrier, norms):
    mid = (qnodes[..., 1:] + qnodes[..., :-1]) / 2
    # Frequency and all coefficients are computed outside the sequential ODE.
    def velocity(q):
        omega = IMRPhenomT._evaluate(20 * (1 - q ** (8.0 / 3.0)), carrier)[0]
        return _maximum(omega / 2, 1e-30) ** (1.0 / 3.0)
    vn, vm = velocity(qnodes), velocity(mid)

    def step(state, data):
        q0, q1, v0, vh, v1 = data
        dq = (q1 - q0)[..., None]
        qh = (q0 + q1) / 2
        k1 = _rates(q0, v0, state, x1, x2)
        k2 = _rates(qh, vh, state + dq * k1 / 2, x1, x2)
        k3 = _rates(qh, vh, state + dq * k2 / 2, x1, x2)
        k4 = _rates(q1, v1, state + dq * k3, x1, x2)
        result = _normalize(state + dq * (k1 + 2*k2 + 2*k3 + k4) / 6, norms)
        return result, result

    nodes = _scan(step, initial, (qnodes[..., :-1], qnodes[..., 1:], vn[..., :-1], vm, vn[..., 1:]))
    states = _join_states(initial, nodes)
    rates = _rates(qnodes, vn, states, x1[..., None], x2[..., None])
    return states, rates


def prepare_numerical_trajectory(mass1, mass2, spin1x, spin1y, spin1z,
                                 spin2x, spin2y, spin2z, f_ref=20.0,
                                 *, n_backward=2048, n_forward=1024):
    """Prepare PV300/FS4 once for a binary or parameter batch.

    The backwards integration covers frequencies down to min(1 Hz, f_ref/2).
    Earlier direct queries return NaN; public sampling requires f_lower>=1 Hz.
    Counts are static resolution choices, exposed privately for convergence tests.
    """
    vals = _arrays((mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, f_ref))
    m1, m2, a, b, c, d, e, f, fr = _broadcast(*vals)
    m1, m2, a, b, c, d, e, f = _canonicalize_components(m1, m2, a, b, c, d, e, f)
    fr = _where(fr > 0, fr, fr * 0 + 20)
    total = m1 + m2
    x1, x2 = m1 / total, m2 / total
    ms = total * MSUN_SEC
    carrier = IMRPhenomT._coeff(m1, m2, c, f)
    # The reference spins must be specified before the coalescence peak.
    fp = IMRPhenomT._evaluate(_zeros_like(m1)[..., None], carrier)[0][..., 0] / (2*math.pi*ms)
    _check_reference(fr, fp)
    tref = IMRPhenomT().time_at_frequency(_minimum(fr, fp), m1, m2, c, f) / ms
    tmin = IMRPhenomT().time_at_frequency(_minimum(fr/2, fr*0+1), m1, m2, c, f) / ms
    qr = (1 - _minimum(tref, tref*0)/20) ** (3.0 / 8.0)
    qmax = (1 - tmin/20) ** (3.0 / 8.0)
    qback = qr[..., None] + (qmax-qr)[..., None] * _linspace(n_backward+1, m1)
    qforward = qr[..., None] + (1-qr)[..., None] * _linspace(n_forward+1, m1)
    s1 = _stack((x1*x1*a, x1*x1*b, x1*x1*c))
    s2 = _stack((x2*x2*d, x2*x2*e, x2*x2*f))
    zero, one = m1*0, m1*0+1
    l0, e0 = _stack((zero, zero, one)), _stack((one, zero, zero))
    initial = _cat((l0, s1, s2, e0))
    norms = (_norm(s1), _norm(s2))
    backward, db = _segment(qback, initial, x1, x2, carrier, norms)
    forward, df = _segment(qforward, initial, x1, x2, carrier, norms)
    q = _join_grid(qforward, qback)
    states, rates = _join_segments(forward, backward), _join_segments(df, db)
    peak = forward[..., -1, :]
    lp, p1, p2, _ = _split(peak)
    z1, z2 = _sum(p1*lp)[..., 0] / x1**2, _sum(p2*lp)[..., 0] / x2**2
    eta, delta = x1*x2, x1-x2
    af_as = fits.IMRPhenomX_FinalSpin2017(eta, z1, z2, delta)
    sperp = p1+p2-_sum((p1+p2)*lp)*lp
    af = _where(af_as >= 0, 1.0, -1.0) * _sqrt(af_as**2 + _sum(sperp*sperp)[..., 0])
    af = _clip(af, -0.9999, 0.9999)
    # Use the same published source-frame angular momentum as the projector.
    lref = _reference_l(m1, m2, fr, c, f)
    jhat = _unit(s1+s2+lref[..., None]*l0)
    # A stable fixed J-frame basis. Chart offsets are fixed at f_ref below.
    trial = _where((_abs(jhat[..., 0]) < 0.9)[..., None], e0, _stack((zero, one, zero)))
    ex = _unit(trial-_sum(trial*jhat)*jhat)
    ey = _cross(jhat, ex)
    return dict(q=q, states=states, rates=rates, initial=initial, peak=peak,
                final_spin=af, jhat=jhat, ex=ex, ey=ey, f_ref=fr,
                m1=m1, m2=m2, z1=c, z2=f, ms=ms,
                reference_valid=fr <= fp)


def _interpolate(trajectory, times):
    q = (1-_minimum(times, times*0)/20/trajectory["ms"][..., None]) ** (3.0/8.0)
    grid = trajectory["q"]
    i = _search(grid, q)
    qa, qb = _gather(grid, i), _gather(grid, i+1)
    h = _maximum(qb-qa, 1e-30)
    w = (q-qa)/h
    y0, y1 = _gather_state(trajectory["states"], i), _gather_state(trajectory["states"], i+1)
    d0, d1 = _gather_state(trajectory["rates"], i), _gather_state(trajectory["rates"], i+1)
    result = ((2*w**3-3*w**2+1)[..., None]*y0 + (w**3-2*w**2+w)[..., None]*h[..., None]*d0
              + (-2*w**3+3*w**2)[..., None]*y1 + (w**3-w**2)[..., None]*h[..., None]*d1)
    valid = (q <= grid[..., -1:]) & trajectory["reference_valid"][..., None]
    return _where(valid[..., None], result, result*0+float("nan"))


def _angles(state, prepared):
    l, _, _, e = _split(state)
    l = _unit(l)
    e = _unit(e-_sum(e*l)*l)
    extra = state.ndim - prepared["jhat"].ndim
    def expand(v):
        return v.reshape(v.shape[:-1]+(1,)*extra+(3,))
    j, ex, ey = (expand(prepared[key]) for key in ("jhat", "ex", "ey"))
    lx, ly, lz = (_sum(l*v)[..., 0] for v in (ex, ey, j))
    rho2 = lx*lx+ly*ly
    regular = rho2 > 1e-28
    rho = _sqrt(_where(regular, rho2, rho2*0+1))
    ca, sa = _where(regular, lx/rho, rho*0+1), _where(regular, ly/rho, rho*0)
    alpha = _atan2(sa, ca)
    beta = _atan2(_where(regular, rho, rho*0), lz)
    etheta = lz[..., None]*(ca[..., None]*ex+sa[..., None]*ey)-_where(regular, rho, rho*0)[..., None]*j
    ephi = -sa[..., None]*ex+ca[..., None]*ey
    gamma = _atan2(_sum(e*ephi)[..., 0], _sum(e*etheta)[..., 0])
    return alpha, beta, gamma


def numerical_geometry(times, frequency, mass1, mass2, spin1x, spin1y, spin1z,
                       spin2x, spin2y, spin2z, inclination, phic, f_ref,
                       frequency_peak=None, final_spin=None, prepared=None,
                       frequency_at_time=None):
    """Return PV300 Euler angles and observer geometry at physical times."""
    del frequency, frequency_peak, frequency_at_time
    if prepared is None:
        prepared = prepare_numerical_trajectory(mass1, mass2, spin1x, spin1y, spin1z,
                                               spin2x, spin2y, spin2z, f_ref)
    a0, theta, zeta, phi = _source_frame(mass1, mass2, spin1x, spin1y, spin1z,
                                      spin2x, spin2y, spin2z, inclination, phic, prepared["f_ref"])
    state = _interpolate(prepared, times)
    a, b, g = _angles(state, prepared)
    ar, _, gr = _angles(prepared["initial"], prepared)
    ap, bp, gp = _angles(prepared["peak"], prepared)
    da, dg = a0-ar, math.pi-phi-gr
    a, g = a+da[..., None], g+dg[..., None]
    ap, gp = ap+da, gp+dg
    af = prepared["final_spin"] if final_spin is None else final_spin
    m1, m2 = prepared["m1"], prepared["m2"]
    eta, delta = m1*m2/(m1+m2)**2, (m1-m2)/(m1+m2)
    mf = fits.IMRPhenomX_FinalMass2017(eta, prepared["z1"], prepared["z2"], delta)
    slope = 2*math.pi/prepared["ms"]*(fits.IMRPhenomT_fringfit(af, 22)-fits.IMRPhenomT_fringfit(af, 21))/mf
    post = times >= 0
    a = _where(post, ap[..., None]+slope[..., None]*times, a)
    g = _where(post, gp[..., None]-_cos(bp)[..., None]*slope[..., None]*times, g)
    b = _where(post, bp[..., None], b)
    return a, b, g, theta, zeta

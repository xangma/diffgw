"""Clean-room IMRPhenomPv2 frequency-domain driver.

The implementation follows the public Pv2 prescription (Schmidt et al.,
arXiv:1408.1810; Bohé et al., LIGO-T1500602) and the published NNLO angle
series (Pratten et al., arXiv:2004.06503).  The aligned carrier is supplied by
the existing public IMRPhenomD callable; no Pv2 implementation is imported.
"""
from __future__ import annotations
import math
import torch

from torchwave.constants import MSUN_SEC
from .imrphenomd import IMRPhenomD
from .kerr_qnm_d_sampling_clean import evaluate_d_qnm_220
from .precession_nnlo_clean import raw_angles, effective_chip
from .precession_twisting_clean import project_twisted_positive_modes


# The sampling convention is stated in physical Hz.  ``f_RD`` is obtained
# from the dimensionless cyclic QNM ordinate returned by the remnant provider
# by dividing by the total mass in seconds.  Keeping the factors here rather
# than burying them in the spline helper makes the unit conversion auditable.
PHASE_SPLINE_SAMPLES = 10
PHASE_SPLINE_LOW_FACTOR = 0.8
PHASE_SPLINE_HIGH_FACTOR = 1.2
PHASE_SPLINE_CUTOFF_MF = 0.2


def _natural_cubic_derivative(nodes, values, query):
    """Evaluate the derivative of a natural cubic through ``values``.

    ``nodes`` are required to be strictly increasing and equally spaced.  The
    construction is entirely tensor-native: the natural boundary moments are
    solved from the eight-by-eight tridiagonal system for the ten-point Pv2
    sample.  In particular, no finite difference or host-side interpolation is
    used, so derivatives with respect to masses and spins remain available.

    The returned derivative has the units of ``values`` per unit of ``nodes``.
    The query is restricted to the closed interpolation interval, including the
    right endpoint.
    """
    x = torch.as_tensor(nodes)
    y = torch.as_tensor(values)
    q = torch.as_tensor(query, device=x.device)
    if x.ndim != 1 or y.ndim != 1 or x.numel() != y.numel():
        raise ValueError("natural cubic inputs must be matching one-dimensional tensors")
    if x.numel() < 3:
        raise ValueError("natural cubic interpolation needs at least three nodes")
    if x.device != y.device or q.device != x.device:
        raise ValueError("natural cubic inputs must share a device")
    if not (x.is_floating_point() and y.is_floating_point()):
        raise TypeError("natural cubic nodes and values must have floating dtypes")

    dtype = torch.promote_types(x.dtype, y.dtype)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("natural cubic nodes and values must be real floating tensors")
    # CPU linear algebra has no reliable half-precision solve.  The public
    # waveform path uses FP64, while this promotion keeps the standalone
    # helper useful for callers passing half tensors and returns their native
    # precision below.
    result_dtype = dtype
    solve_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    x = x.to(solve_dtype)
    y = y.to(solve_dtype)
    q = q.to(solve_dtype)
    if not (
        bool(torch.isfinite(x).all())
        and bool(torch.isfinite(y).all())
        and bool(torch.isfinite(q).all())
    ):
        raise ValueError("natural cubic inputs must be finite")
    widths = x[1:] - x[:-1]
    if bool(torch.any(widths <= 0.0)):
        raise ValueError("natural cubic nodes must be strictly increasing")
    step = widths[0]
    if not bool(torch.allclose(widths, step, rtol=1.0e-6, atol=1.0e-12)):
        raise ValueError("natural cubic nodes must be uniformly spaced")
    if bool(torch.any((q < x[0]) | (q > x[-1]))):
        raise ValueError("natural cubic query lies outside the interpolation interval")

    # For equally spaced nodes, the interior moment system is
    # M[i-1] + 4 M[i] + M[i+1] = 6 (y[i-1] - 2 y[i] + y[i+1]) / h**2.
    n_interior = x.numel() - 2
    matrix = 4.0 * torch.eye(n_interior, dtype=solve_dtype, device=x.device)
    if n_interior > 1:
        off = torch.ones(n_interior - 1, dtype=solve_dtype, device=x.device)
        matrix = matrix + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    rhs = (6.0 / step.square()) * (y[:-2] - 2.0 * y[1:-1] + y[2:])
    moments = torch.cat((torch.zeros(1, dtype=solve_dtype, device=x.device),
                         torch.linalg.solve(matrix, rhs),
                         torch.zeros(1, dtype=solve_dtype, device=x.device)))

    # ``right=True`` selects the interval beginning at an exact interior knot;
    # the final endpoint is clamped to the last interval and receives its
    # right-hand derivative.
    interval = (torch.searchsorted(x, q.reshape(-1), right=True) - 1).clamp(
        min=0, max=x.numel() - 2
    )
    dx = q.reshape(-1) - x[interval]
    h = step
    b = (y[interval + 1] - y[interval]) / h - h * (
        2.0 * moments[interval] + moments[interval + 1]
    ) / 6.0
    c = moments[interval] / 2.0
    d = (moments[interval + 1] - moments[interval]) / (6.0 * h)
    result = b + 2.0 * c * dx + 3.0 * d * dx.square()
    return result.reshape(q.shape).to(result_dtype)


def _phase_sample_bounds(frd, total_mass_seconds):
    """Return the physical-Hz endpoints of the Pv2 phase sample interval."""
    low = PHASE_SPLINE_LOW_FACTOR * frd
    high = torch.minimum(
        PHASE_SPLINE_HIGH_FACTOR * frd,
        torch.as_tensor(PHASE_SPLINE_CUTOFF_MF, dtype=frd.dtype, device=frd.device)
        / total_mass_seconds,
    )
    if bool(high <= low):
        raise ValueError("Pv2 phase-spline support is empty at the remnant frequency")
    return low, high


def _phase_time_shift(carrier_phase, frd, total_mass_seconds, d_call):
    """Measure the D carrier slope at ``f_RD`` using the Pv2 N10 spline."""
    low, high = _phase_sample_bounds(frd, total_mass_seconds)
    unit = torch.arange(
        PHASE_SPLINE_SAMPLES, dtype=frd.dtype, device=frd.device
    ) / (PHASE_SPLINE_SAMPLES - 1)
    sample_frequencies = low + (high - low) * unit
    _, _, phase_samples = d_call(sample_frequencies, carrier_phase=carrier_phase)
    return _natural_cubic_derivative(sample_frequencies, phase_samples, frd)


def _L2(m1, m2, f):
    """Pv2 2PN orbital angular momentum used by the source-frame map.

    Pratten prints 27/8 with a plus sign, while the technical note prints
    decimal 3.373 with a minus sign.  The mixed discrete choice below is an
    observed public-API compatibility choice, not a single published formula.
    """
    M = (m1 + m2) * MSUN_SEC
    v = (math.pi * M * f).pow(1 / 3)
    eta = m1 * m2 / (m1 + m2).square()
    return eta / v * (1 + (1.5 + eta / 6) * v.square() +
                      (27 / 8 - 19 * eta / 8 - eta.square() / 24) * v.pow(4))


def _geometry(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi, fref):
    """Observer/J basis of the Pv2 source-frame construction."""
    M = m1 + m2; n1, n2 = m1 / M, m2 / M
    L = _L2(m1, m2, fref)
    S = torch.stack((n1.square()*s1x+n2.square()*s2x,
                     n1.square()*s1y+n2.square()*s2y,
                     n1.square()*s1z+n2.square()*s2z))
    J = S + torch.stack((torch.zeros_like(L), torch.zeros_like(L), L))
    j = J / torch.linalg.vector_norm(J)
    n = torch.stack((torch.sin(iota)*torch.sin(phi), torch.sin(iota)*torch.cos(phi), torch.cos(iota)))
    ox = torch.stack((-torch.cos(iota)*torch.sin(phi), -torch.cos(iota)*torch.cos(phi), torch.sin(iota)))
    y = torch.linalg.cross(j, n); yn = torch.linalg.vector_norm(y)
    y = y / torch.where(yn > 1e-14, yn, torch.ones_like(yn))
    y = torch.where((yn > 1e-14), y, torch.linalg.cross(j, ox))
    y = y / torch.where(torch.linalg.vector_norm(y) > 1e-14, torch.linalg.vector_norm(y), torch.ones_like(yn))
    x = torch.linalg.cross(y, j)
    alpha0 = torch.atan2(y[2], x[2])
    theta = torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(j, n)), torch.dot(j, n))
    xs = torch.cos(theta)*x - torch.sin(theta)*j
    # LAL's public source-frame API reports the polarization gauge with the
    # spin-weighted (rather than geometric tangent) zero, i.e. +pi/2 here.
    zeta = torch.atan2(torch.dot(ox, y), torch.dot(ox, xs)) + math.pi / 2
    # Its ``phi_aligned`` is the passive azimuth of J.
    phi_j = -torch.atan2(J[1], J[0]) if torch.linalg.vector_norm(J[:2]) > 1e-14 else math.pi/2 - phi
    return alpha0, theta, zeta, phi_j


def waveform(frequency, mass1, mass2, spin1x=0., spin1y=0., spin1z=0.,
             spin2x=0., spin2y=0., spin2z=0., distance=100., inclination=0.,
             phic=0., tc=0., f_ref=None, convention=0,
             phasing_convention=None, reference_frame=None,
             precession_version="v2"):
    """Return ``(h_plus, h_cross)`` with parameter axes preceding frequency."""
    if convention not in (0, 1):
        raise ValueError("convention must be 0 or 1")
    if precession_version not in (None, "v2"):
        raise ValueError("clean IMRPhenomPv2 supports only precession_version='v2'")
    if phasing_convention not in (None, "coalescence", "reference_frequency"):
        raise ValueError("phasing_convention must be 'coalescence' or 'reference_frequency'")
    if reference_frame not in (None, "coalescence", "reference_frequency"):
        raise ValueError("reference_frame must be 'coalescence' or 'reference_frequency'")
    # IMRPhenomD defines reference_frame as an alias which takes precedence.
    carrier_phasing = reference_frame if reference_frame is not None else (
        phasing_convention if phasing_convention is not None else "coalescence"
    )
    fi = torch.as_tensor(frequency)
    fi = fi if fi.is_floating_point() else fi.to(torch.float64)
    f = fi.to(torch.float64).reshape(-1); pos = f[f > 0]
    if not pos.numel(): raise ValueError("frequency grid must contain a positive frequency")
    vals = [torch.as_tensor(x, dtype=torch.float64, device=f.device) for x in
            (mass1,mass2,spin1x,spin1y,spin1z,spin2x,spin2y,spin2z,distance,inclination,phic,tc)]
    fr = pos.min() if f_ref is None else torch.as_tensor(f_ref, dtype=torch.float64, device=f.device)
    vals.append(fr); vals = torch.broadcast_tensors(*vals)
    if torch.any(vals[0] <= 0) or torch.any(vals[1] <= 0) or torch.any(vals[8] <= 0):
        raise ValueError("masses and distance must be positive")
    shape = vals[0].shape; flat = [x.reshape(-1) for x in vals]; hp, hc = [], []
    for i in range(flat[0].numel()):
        a = [x[i] for x in flat]; m1,m2,x1,y1,z1,x2,y2,z2,d,inc,pc,tc,rf = a
        if bool(m1 < m2): m1,m2,x1,x2,y1,y2,z1,z2 = m2,m1,x2,x1,y2,y1,z2,z1
        M = (m1+m2)*MSUN_SEC; v=(math.pi*M*torch.where(f>0,f,torch.ones_like(f))).pow(1/3)
        vr=(math.pi*M*rf).pow(1/3); eta=m1*m2/(m1+m2).square(); delta=(m1-m2)/(m1+m2)
        chip=effective_chip(m1,m2,x1,y1,x2,y2)
        # Pv2's Eq. (32) is evaluated through the earlier Eq. (30) mapping:
        # chi_l = chi_eff / x1.  The technical note's shorthand z1 is a
        # distinct NNLO variant; public Pv2 black-box probing selects this
        # mass-weighted mapping, retained here without fitted corrections.
        chil=(m1*z1+m2*z2)/(m1+m2)
        al, ep = raw_angles(v,eta,-delta,m1/(m1+m2),chil/(m1/(m1+m2)),chip)
        al0,ep0 = raw_angles(vr,eta,-delta,m1/(m1+m2),chil/(m1/(m1+m2)),chip)
        g=_geometry(m1,m2,x1,y1,z1,x2,y2,z2,inc,pc,rf)
        # Pv2's cone evolves with frequency.  Keeping the full evaluation
        # grid here is essential; anchoring beta at f_ref freezes the twist
        # and produces a visibly different waveform.
        ff=torch.where(f>0, f, torch.ones_like(f))
        # Eqs. (24)-(25) select the positive-cosine branch: the parallel
        # denominator enters by magnitude, including for L+S_parallel < 0.
        beta=torch.atan2(chip*(m1/(m1+m2)).square(), torch.abs(_L2(m1,m2,ff)+ (m1/(m1+m2)).square()*z1+(m2/(m1+m2)).square()*z2))
        alpha=al-al0+g[0]; gamma=-ep+ep0
        if convention == 1: gamma=gamma+math.pi-g[3]; zeta=g[2]-math.pi/2; carrier_phase=torch.zeros_like(g[3])
        else: zeta=g[2]; carrier_phase=g[3]
        def remnant_provider(a,b,za,zb):
            mf,afd,frd,fdd=IMRPhenomD.remnant_properties(a,b,za,zb)
            n=a/(a+b); ap=torch.sqrt(afd.square()+(chip*n.square()).square())
            ap=torch.where(afd < 0, -ap, ap)
            # The independent D compatibility helper returns cyclic
            # frequencies in final-mass units.  Convert those to the total-M
            # cyclic convention consumed by the D callable explicitly.
            wr,wd=evaluate_d_qnm_220(ap)
            return mf,ap,wr/(mf/(a+b)),wd/(mf/(a+b))
        # Pv2 uses the D coalescence phase origin.  Its remnant-frequency time
        # shift is measured from the prescribed ten-point natural cubic
        # carrier-phase interpolant.  The carrier itself is evaluated on the
        # caller's physical-Hz grid.
        carrier,_=IMRPhenomD()(f,m1,m2,z1,z2,distance=d,inclination=torch.zeros_like(inc),phic=carrier_phase,tc=tc,f_ref=rf,phasing_convention=carrier_phasing,remnant_provider=remnant_provider)
        with torch.enable_grad():
            mf,af,frd_dimless,_=remnant_provider(m1,m2,z1,z2)
            frd=frd_dimless/M

            def d_call(sample_frequencies, carrier_phase=0.0):
                return IMRPhenomD()(
                    sample_frequencies,
                    m1,
                    m2,
                    z1,
                    z2,
                    distance=d,
                    inclination=torch.zeros_like(inc),
                    phic=carrier_phase,
                    tc=0.0,
                    f_ref=rf,
                    phasing_convention=carrier_phasing,
                    return_phase=True,
                    remnant_provider=remnant_provider,
                )

            k=_phase_time_shift(torch.zeros_like(inc),frd,M,d_call)
        carrier=carrier*torch.exp(1j*torch.where(f>0,f,torch.ones_like(f))*k)
        h22=2*carrier/math.sqrt(5/(4*math.pi))
        p,c=project_twisted_positive_modes({(2,2):h22},alpha,gamma,beta,g[1],zeta)
        hp.append(torch.where(f>0,p,torch.zeros_like(p))); hc.append(torch.where(f>0,c,torch.zeros_like(c)))
    out=torch.complex128 if fi.dtype==torch.float64 else torch.complex64
    return torch.stack(hp).reshape(shape+(f.numel(),)).to(out),torch.stack(hc).reshape(shape+(f.numel(),)).to(out)

__all__=["waveform"]

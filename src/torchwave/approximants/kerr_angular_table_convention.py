"""Opt-in public-compatible (2,2), s=-2 angular convention.

This is a clean-room diagnostic convention.  The absolute ``0.01`` series
termination rule (the first small term is included, then evaluation stops)
and its angular metadata were inferred from black-box public outputs, not
published calibration data.  The converged Leaver solution in
``kerr_angular`` remains separate and is not replaced by this module.
"""
import math
import numpy as np
import torch
from .kerr_qnm_generated import signed_qnm_220
from .kerr_angular import leaver_frobenius_coefficients

_K21_NODES = (-0.9956571630258081,-0.9739065285171717,-0.9301574913557082,-0.8650633666889845,-0.7808177265864169,-0.6794095682990244,-0.5627571346686047,-0.43339539412924716,-0.2943928627014602,-0.14887433898163122,0.,0.14887433898163122,0.2943928627014602,0.43339539412924716,0.5627571346686047,0.6794095682990244,0.7808177265864169,0.8650633666889845,0.9301574913557082,0.9739065285171717,0.9956571630258081)
_K21_WEIGHTS = (0.011694638867371862,0.03255816230796478,0.054755896574351974,0.07503967481091994,0.0931254545836976,0.10938715880229763,0.12349197626206586,0.13470921731147337,0.14277593857706,0.14773910490133854,0.14944555400291687,0.14773910490133854,0.14277593857706,0.13470921731147337,0.12349197626206586,0.10938715880229763,0.0931254545836976,0.07503967481091994,0.054755896574351974,0.03255816230796478,0.011694638867371862)
_G10_INDICES = (1,3,5,7,9,11,13,15,17,19)
_G10_WEIGHTS = (0.06667134430868814,0.1494513491505806,0.21908636251598204,0.26926671930999635,0.29552422471475287,0.29552422471475287,0.26926671930999635,0.21908636251598204,0.1494513491505806,0.06667134430868814)


def _raw(mu, c, coeff):
    z = (1 + mu) / 2
    active = torch.ones_like(mu, dtype=torch.bool)
    total = coeff[..., 0] * torch.ones_like(mu, dtype=coeff.dtype)
    for n in range(1, coeff.shape[-1]):
        term = coeff[..., n] * z**n
        # The terminating term is evaluated, then the public loop stops.
        use = active
        total = total + torch.where(use, term, torch.zeros_like(term))
        active = active & (torch.abs(term) >= 0.01)
    return torch.exp(c * mu) * (1 + mu) ** 2 * total


def _norm(c, coeff, tol=1e-6, max_intervals=1000):
    """Globally adaptive embedded K21/G10 norm (inferred numerical convention)."""
    nodes = torch.tensor(_K21_NODES, dtype=c.real.dtype, device=c.device)
    weights = torch.tensor(_K21_WEIGHTS, dtype=c.real.dtype, device=c.device)
    gweights = torch.tensor(_G10_WEIGHTS, dtype=c.real.dtype, device=c.device)
    out = []
    for ci, ai in zip(c.reshape(-1), coeff.reshape(-1, coeff.shape[-1])):
        def leaf(lo, hi):
            x=(hi-lo)/2*nodes+(hi+lo)/2; y=_raw(x,ci,ai).abs().square()
            k=(hi-lo)/2*(weights*y).sum(); g=(hi-lo)/2*(gweights*y[list(_G10_INDICES)]).sum()
            return [lo,hi,k,(k-g).abs()]
        leaves=[leaf(torch.tensor(-1.,dtype=ci.real.dtype,device=ci.device),
                     torch.tensor(1.,dtype=ci.real.dtype,device=ci.device))]
        while sum(float(q[3].detach()) for q in leaves) > max(tol, tol*abs(float(sum(q[2] for q in leaves).detach()))):
            if len(leaves)>=max_intervals: raise RuntimeError("K21/G10 norm exceeded interval limit")
            j=max(range(len(leaves)),key=lambda k:float(leaves[k][3].detach())); lo,hi,_,_=leaves.pop(j); mid=(lo+hi)/2
            leaves.extend((leaf(lo,mid),leaf(mid,hi)))
        total = sum(q[2] for q in leaves)
        out.append(torch.sqrt(2 * math.pi * total).real)
    return torch.stack(out).reshape(c.shape)


def spin_weighted_spheroidal_harmonic_table_compat(spin, theta, phi=0.0, *, n_terms=48):
    """Evaluate inferred public-compatible normalized ``s=-2, l=m=2`` shape."""
    if isinstance(spin, (float, int)):
        spin = torch.tensor(spin, dtype=torch.float64)
    else:
        spin = torch.as_tensor(spin)
    if not spin.is_floating_point():
        spin = spin.to(torch.float64)
    if (spin.abs() > 0.9996).any() or not torch.isfinite(spin).all():
        raise ValueError("spin must be finite and in [-0.9996,0.9996]")
    theta = torch.as_tensor(theta, dtype=spin.dtype, device=spin.device)
    phi = torch.as_tensor(phi, dtype=spin.dtype, device=spin.device)
    if not torch.isfinite(theta).all() or not torch.isfinite(phi).all():
        raise ValueError("theta and phi must be finite")
    real, damp = signed_qnm_220(spin.to(torch.float64))
    # The angular callable's public eigenvalue uses 2 M omega; our QNM helper
    # returns M omega, so c=(a)(2M omega)=spin*(M omega).
    c = spin.to(torch.float64) * torch.complex(real, -damp)
    _, coeff = leaver_frobenius_coefficients(2, c, n_terms=n_terms)
    bspin, btheta, bphi = torch.broadcast_tensors(spin, theta, phi)
    cb = c.expand(bspin.shape)
    ab = coeff.expand(bspin.shape + (coeff.shape[-1],))
    raw = _raw(torch.cos(btheta), cb, ab)
    norm = _norm(c, coeff)
    raw0 = _raw(torch.zeros_like(c), c, coeff)
    gauge = torch.conj(raw0) / raw0.abs()
    gauge = gauge.reshape(spin.shape).expand(bspin.shape)
    out = raw * gauge / norm.reshape(spin.shape).expand(bspin.shape) * torch.exp(2j * bphi)
    return out.to(torch.complex64 if spin.dtype == torch.float32 else torch.complex128)

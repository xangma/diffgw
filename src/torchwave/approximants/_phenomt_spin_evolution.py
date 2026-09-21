"""Orbit-averaged N4LO spin evolution used by numerical PhenomT precession.

All spins are dimensionless physical spins divided by ``M**2`` and the
returned derivatives are with respect to ``t/M``.  Coefficients are from
Sturani, LIGO-T1500554, Eqs. (3), (22), (24), and (25); the Eq. (24) ``b``
term multiplies the full companion spin vector.
"""
from __future__ import annotations
import math
import torch


def spin_derivatives(v, lhat, s1, s2, m1_fraction, m2_fraction):
    """Return ``(dLhat/d(t/M), dS1/d(t/M), dS2/d(t/M))`` through N4LO.

    Inputs broadcast over arbitrary leading dimensions; the final state axis
    must have length three.  ``m1_fraction + m2_fraction`` is one.
    """
    lhat, s1, s2 = torch.broadcast_tensors(lhat, s1, s2)
    lead = lhat.shape[:-1]
    v = torch.broadcast_to(torch.as_tensor(v, device=lhat.device, dtype=lhat.dtype), lead)[..., None]
    m1 = torch.broadcast_to(torch.as_tensor(m1_fraction, device=lhat.device, dtype=lhat.dtype), lead)[..., None]
    m2 = torch.broadcast_to(torch.as_tensor(m2_fraction, device=lhat.device, dtype=lhat.dtype), lead)[..., None]
    eta = m1 * m2
    l1 = (lhat * s1).sum(-1, keepdim=True)
    l2 = (lhat * s2).sum(-1, keepdim=True)
    c1 = 1.5 - m1 - 0.5*m1*m1
    c2 = 1.5 - m2 - 0.5*m2*m2
    ds1 = c1*v**5 * torch.linalg.cross(lhat, s1)
    ds2 = c2*v**5 * torch.linalg.cross(lhat, s2)
    s12 = torch.linalg.cross(s1, s2)
    # Orbit-averaged NLO spin-spin terms.
    ds1 = ds1 + v**6 * (-0.5*s12 - 1.5*l2*torch.linalg.cross(lhat,s1)
                         + 1.5*(1.0-1.0/m1)*l1*torch.linalg.cross(lhat,s1))
    ds2 = ds2 + v**6 * (0.5*s12 - 1.5*l1*torch.linalg.cross(lhat,s2)
                         + 1.5*(1.0-1.0/m2)*l2*torch.linalg.cross(lhat,s2))
    # NNLO spin-orbit, Sturani Eq. (22).
    q1 = 9/8 - m1/2 + 7*m1**2/12 - 7*m1**3/6 - m1**4/24
    q2 = 9/8 - m2/2 + 7*m2**2/12 - 7*m2**3/6 - m2**4/24
    ds1 = ds1 + q1*v**7*torch.linalg.cross(lhat,s1)
    ds2 = ds2 + q2*v**7*torch.linalg.cross(lhat,s2)
    # N3LO orbit-averaged quadratic spin, Sturani Eq. (24), kappa=1.
    a1 = -13/4 + 9*m1/4 + m1**2
    a2 = -13/4 + 9*m2/4 + m2**2
    b1 = m1/2 + m1**2/2
    b2 = m2/2 + m2**2/2
    d1 = -3/2 - 3*m1/2 - m1**2/2
    d2 = -3/2 - 3*m2/2 - m2**2/2
    ds1 = ds1 + v**8*torch.linalg.cross(a1*l1*lhat + b1*s2 + d1*l2*lhat, s1)
    ds2 = ds2 + v**8*torch.linalg.cross(a2*l2*lhat + b2*s1 + d2*l1*lhat, s2)
    # N4LO spin-orbit, Sturani Eq. (25).
    r1 = 27/16 - 51*m1/8 + 181*m1**2/16 - 23*m1**3/6 - 39*m1**4/16 - 3*m1**5/8 - m1**6/48
    r2 = 27/16 - 51*m2/8 + 181*m2**2/16 - 23*m2**3/6 - 39*m2**4/16 - 3*m2**5/8 - m2**6/48
    ds1 = ds1 + r1*v**9*torch.linalg.cross(lhat,s1)
    ds2 = ds2 + r2*v**9*torch.linalg.cross(lhat,s2)
    # TP follows arXiv:2105.05872 Sec. II.4.1: higher-order linear-in-spin
    # corrections to L are intentionally disabled because they caused
    # technical problems in the rotated modes. Use nonspin 3PN L with the
    # projected Eq. 13 constraint below; this is not full SpinTaylor Eq. 19.
    L = eta/torch.clamp(v, min=torch.finfo(v.dtype).tiny) * (
        1 + (3/2 + eta/6)*v**2
        + (27/8 - 19*eta/8 + eta**2/24)*v**4
        + (135/16 + (-6889/144 + 41*math.pi**2/24)*eta
           + 31*eta**2/24 + 7*eta**3/1296)*v**6)
    total = ds1 + ds2
    l2norm=(lhat*lhat).sum(-1,keepdim=True).clamp_min(torch.finfo(v.dtype).tiny)
    dl = -(total - lhat*(lhat*total).sum(-1, keepdim=True)/l2norm)/torch.clamp(L, min=torch.finfo(v.dtype).tiny)
    return dl, ds1, ds2


__all__ = ["spin_derivatives"]

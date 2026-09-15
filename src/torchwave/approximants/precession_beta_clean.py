"""Clean-room orbital-angular-momentum beta from Pratten et al. (2004.06503v2).

Implements the orbital-magnitude coefficient form of Appendix G Eq. (123),
inserted in Eq. (26), and Eq. (17).  The ``pn_version=102`` branch retains the
version-102 3PN non-spin and 2.5PN spin-orbit terms; ``pn_version='full'``
retains the displayed higher-order terms.  Masses are in solar masses (the frequency input is Hz) and spins are
dimensionless component spins; all returned quantities are dimensionless and
differentiable tensors.
"""
from __future__ import annotations
import math
import torch


def orbital_angular_momentum(m1, m2, f, s1z, s2z, pn_version=102, compat_102=False):
    """Return L/M² from Appendix G Eq. (26), with x=(pi M f)^(2/3).

    Version 102 retains the 3PN nonspin and 2.5PN spin-orbit terms (Table 3).
    ``pn_version='full'`` is available for the complete displayed Eq. 26.
    """
    m1 = torch.as_tensor(m1, dtype=torch.float64)
    m2 = torch.as_tensor(m2, dtype=torch.float64, device=m1.device)
    # The public angular APIs use double precision; retain graph connectivity
    # when f is a differentiable tensor while making scalar inputs explicit.
    f = torch.as_tensor(f, dtype=torch.float64, device=m1.device if m1.device.type != 'cpu' else None)
    dtype, device = f.dtype, f.device
    m1 = m1.to(dtype=dtype, device=device); m2 = m2.to(dtype=dtype, device=device)
    s1z = torch.as_tensor(s1z, dtype=dtype, device=device)
    s2z = torch.as_tensor(s2z, dtype=dtype, device=device)
    M = m1 + m2; eta = m1*m2/M**2; delta = (m1-m2)/M
    nu1, nu2 = m1/M, m2/M
    x = (math.pi*M*f*4.925490947641267e-6)**(2/3)
    euler = 0.5772156649015329
    Sl = nu1**2*s1z + nu2**2*s2z
    Sigma = nu2*s2z - nu1*s1z
    l0 = 1.
    l1 = eta/6 + 3/2
    l2 = eta**2/24 - 19*eta/8 + 27/8
    l3 = 7*eta**3/1296 + 31*eta**2/24 + (41*math.pi**2/24-6889/144)*eta + 135/16
    if compat_102:
        # Optional diagnostic branch: public version-102 black-box behavior
        # omits the printed +41*pi^2*eta/24 contribution. This is retained
        # only for compatibility experiments; it is not the clean default.
        l3 = l3 - (41*math.pi**2/24)*eta
    l4 = (-55*eta**4/31104 - 215*eta**3/1728
          + (356035/3456-2255*math.pi**2/576)*eta**2
          + eta*(-64/3*torch.log(16*x)-6455*math.pi**2/1536-128*euler/3+98869/5760)
          + 2835/128)
    ls15 = -35*Sl/6 - 5*delta*Sigma/2
    ls25 = (-77/8+427*eta/72)*Sl + delta*(-21/8+35*eta/12)*Sigma
    ls35 = (-405/16+1101*eta/16-29*eta**2/16)*Sl + delta*(-81/6+117*eta/4-15*eta**2/16)*Sigma
    terms=l0+l1*x+l2*x**2+l3*x**3+ls15*x**1.5+ls25*x**2.5
    if pn_version == "full":
        terms=terms+l4*x**4+ls35*x**3.5
    return eta/torch.sqrt(x) * terms


def cos_beta(m1, m2, f, s1x, s1y, s1z, s2x, s2y, s2z, pn_version=102, compat_102=False):
    """Return cos(beta) using Eq. (17), with Eq. (26)'s L."""
    vals = [m1,m2,f,s1x,s1y,s1z,s2x,s2y,s2z]
    dtype = torch.float64
    t = [torch.as_tensor(v, dtype=dtype) for v in vals]
    m1,m2,f,s1x,s1y,s1z,s2x,s2y,s2z=t
    M=m1+m2; n1=m1/M; n2=m2/M
    L=orbital_angular_momentum(m1,m2,f,s1z,s2z,pn_version=pn_version,compat_102=compat_102)
    spar=n1**2*s1z+n2**2*s2z
    # Version 102's Eq. 17 uses the single-effective-spin prescription:
    # S_perp/M² = x1² chi_p (Appendix G Eqs. 114--119).
    a1=2+1.5*m2/m1; a2=2+1.5*m1/m2
    chip=torch.maximum(a1*m1**2*torch.hypot(s1x,s1y),a2*m2**2*torch.hypot(s2x,s2y))/(a1*m1**2)
    sperp=n1**2*chip
    jpar=L+spar
    return jpar/torch.sqrt(jpar**2+sperp**2)


def beta(*args, **kwargs):
    """Return beta=acos(cos(beta)) on the principal [0,pi] branch."""
    return torch.acos(cos_beta(*args, **kwargs))

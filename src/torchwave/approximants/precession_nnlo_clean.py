"""Independent NNLO single-spin Euler-angle series from Pratten et al., arXiv:2004.06503v2.

Equations (120)--(122j), with dimensionless masses m_i/M and the paper's
chi_l and chi_p conventions (Eqs. 114--119, 20--22). Constants of integration
are intentionally excluded: frame anchoring belongs to Appendix C/Table IV.
"""
from __future__ import annotations
import math
import torch


def effective_chip(m1, m2, s1x, s1y, s2x, s2y):
    M=m1+m2; q=m1/m2; a1=2+1.5/q; a2=2+1.5*q
    sp1=m1*m1*torch.hypot(s1x,s1y); sp2=m2*m2*torch.hypot(s2x,s2y)
    return torch.maximum(a1*sp1,a2*sp2)/(a1*m1*m1)


def coefficients(eta, delta, m1, chi_l, chi_p):
    pi=math.pi
    a3=-5*delta/(64*m1)-35/192
    a2=-5*m1*chi_l*(3*delta+7*m1)/(128*eta)
    a1= -5*(824*eta+1103)/3072 -15*delta**2*eta/(256*m1**2)-15*delta*m1**3*chi_p**2/(128*eta**2)-5*delta*(980*eta+911)/(7168*m1)-35*m1**4*chi_p**2/(128*eta**2)
    a_1=5*(36*eta*(85568*eta+23817)+8024297)/9289728 +5*m1**2*(3*delta**2*chi_p**2+75*delta**2*chi_l**2-112*pi*chi_l)/(256*eta)-15*delta*m1**7*(chi_p**4-4*chi_p**2*chi_l**2)/(512*eta**4)+5*delta*m1**3*((812*eta-97)*chi_p**2+20328*eta*chi_l**2)/(14336*eta**2)-15*pi*delta*m1*chi_l/(16*eta)-35*m1**8*(chi_p**4-4*chi_p**2*chi_l**2)/(512*eta**4)+m1**4*(25*(92*eta+19)*chi_p**2+52640*eta*chi_l**2)/(6144*eta**2)+15*delta**3*eta**2/(1024*m1**3)+5*delta**2*eta*(784*eta+323)/(28672*m1**2)+5*delta*(504*eta*(7630*eta-159)+5579177)/(21676032*m1)
    al= -5/48*(7*pi-3*delta**2*chi_l)-5*delta*m1**5*chi_p**2*chi_l/(128*eta**3)+5*delta*(7168*eta+407)*m1*chi_l/(21504*eta)-35*m1**6*chi_p**2*chi_l/(384*eta**3)+5*(4072*eta+599)*m1**2*chi_l/(9216*eta)-5*pi*delta/(16*m1)
    e3=a3; e2=a2
    e1=-5*(824*eta+1103)/3072-15*delta**2*eta/(256*m1**2)-5*delta*(980*eta+911)/(7168*m1)
    e_1=5*(36*eta*(85568*eta+23817)+8024297)/9289728+5*m1**2*chi_l*(75*delta**2*chi_l-112*pi)/(256*eta)+1815*delta*m1**3*chi_l**2/(256*eta)-15*pi*delta*m1*chi_l/(16*eta)+1645*m1**4*chi_l**2/(192*eta)+15*delta**3*eta**2/(1024*m1**3)+5*delta**2*eta*(784*eta+323)/(28672*m1**2)+5*delta*(504*eta*(7630*eta-159)+5579177)/(21676032*m1)
    el=-5/48*(7*pi-3*delta**2*chi_l)+5*delta*(7168*eta+407)*m1*chi_l/(21504*eta)+5*(4072*eta+599)*m1**2*chi_l/(9216*eta)-5*pi*delta/(16*m1)
    if isinstance(eta, torch.Tensor):
        dtype, device = eta.dtype, eta.device
    else:
        dtype, device = torch.float64, None
    vals_a = (a3, a2, a1, a_1, al)
    vals_e = (e3, e2, e1, e_1, el)
    return (torch.stack(tuple(torch.as_tensor(x, dtype=dtype, device=device) for x in vals_a), dim=-1),
            torch.stack(tuple(torch.as_tensor(x, dtype=dtype, device=device) for x in vals_e), dim=-1))


def raw_angles(v, eta, delta, m1, chi_l, chi_p):
    """Return (alpha,epsilon), omitting integration constants; v=(pi M f)^(1/3)."""
    aa,ee=coefficients(eta,delta,m1,chi_l,chi_p)
    x=torch.as_tensor(v)
    basis=torch.stack((x**-3, x**-2, x**-1, x, torch.log(x**3)), dim=-1 if x.ndim == 0 else -2)
    if x.ndim == 0:
        return (aa * basis).sum(dim=-1), (ee * basis).sum(dim=-1)
    return (aa.unsqueeze(-1) * basis).sum(dim=-2), (ee.unsqueeze(-1) * basis).sum(dim=-2)

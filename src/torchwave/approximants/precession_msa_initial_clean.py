"""Clean-room MSA spin initializer from Chatziioannou et al. 1703.03967v3.

This module contains only the invariant geometry and the elliptic spin
oscillation.  It is intentionally independent of the production precession
implementations.
"""
from __future__ import annotations
import torch


def initial_invariants(mass1, mass2, spin1, spin2, orbital):
    """Return ordered ``(q, eta, delta_m, S1, S2, L, J, xi)`` in M=1 units."""
    m1, m2 = torch.broadcast_tensors(torch.as_tensor(mass1), torch.as_tensor(mass2))
    dtype = torch.promote_types(torch.promote_types(m1.dtype, m2.dtype), torch.float64)
    m1, m2 = m1.to(dtype), m2.to(dtype)
    s1, s2, l = (torch.as_tensor(x, dtype=dtype)
                 for x in (spin1, spin2, orbital))
    if bool(torch.any(~torch.isfinite(m1))) or bool(torch.any(~torch.isfinite(m2))) or bool(torch.any((m1 <= 0) | (m2 <= 0))):
        raise ValueError("masses must be positive and finite")
    swap = m2 > m1
    om1, om2, os1, os2 = m1, m2, s1, s2
    m1, m2 = torch.where(swap, om2, om1), torch.where(swap, om1, om2)
    s1 = torch.where(swap[..., None], os2, os1)
    s2 = torch.where(swap[..., None], os1, os2)
    total = m1 + m2
    m1, m2 = m1 / total, m2 / total
    # Input spin vectors are dimensionless chi; S_i=(m_i/M)^2 chi_i.
    s1, s2 = m1[..., None] ** 2 * s1, m2[..., None] ** 2 * s2
    S1, S2, L = torch.linalg.vector_norm(s1, dim=-1), torch.linalg.vector_norm(s2, dim=-1), torch.linalg.vector_norm(l, dim=-1)
    Jv = l + s1 + s2
    J = torch.linalg.vector_norm(Jv, dim=-1)
    lhat = l / L.unsqueeze(-1)
    xi = (1 + m2 / m1) * (s1 * lhat).sum(-1) + (1 + m1 / m2) * (s2 * lhat).sum(-1)
    return m2 / m1, m1 * m2, m1 - m2, S1, S2, L, J, xi


def cubic_coefficients(q, eta, delta_m, S1_2, S2_2, L, J, xi):
    """Direct Appendix B coefficients of the spin cubic (Eq. 21)."""
    q, eta, dm, s1, s2, L, J, xi = torch.broadcast_tensors(
        *(torch.as_tensor(x, dtype=torch.float64) for x in
          (q, eta, delta_m, S1_2, S2_2, L, J, xi)))
    B = (L**2 + s1) * q + 2*L*xi - 2*J**2 - s1 - s2 + (L**2+s2)/q
    C = (J**2-L**2)**2 - 2*L*xi*(J**2-L**2) \
        - 2*(1-q)/q*(s1-q*s2)*L**2 + 4*eta*L**2*xi**2 \
        - 2*dm*(s1-s2)*xi*L + 2*(1-q)/q*(q*s1-s2)*J**2
    D = (1-q)/q*(s2-q*s1)*(J**2-L**2)**2 \
        + dm**2/eta*(s1-s2)**2*L**2 \
        + 2*dm*L*xi*(s1-s2)*(J**2-L**2)
    return B, C, D


def ordered_roots(B, C, D):
    """Pure-Torch real roots of x³+B x²+C x+D, sorted ascending."""
    B, C, D = torch.broadcast_tensors(*(torch.as_tensor(x, dtype=torch.float64) for x in (B,C,D)))
    p = C - B**2/3
    if bool(torch.any(p >= 0)):
        raise ValueError("cubic does not have three distinct real roots")
    q = 2*B**3/27 - B*C/3 + D
    raw_arg = 3*q/(2*p) * torch.sqrt(-3/p)
    if bool(torch.any(raw_arg.abs() > 1 + 1e-10)):
        raise ValueError("cubic does not have three real roots")
    arg = raw_arg.clamp(-1, 1)
    theta = torch.acos(arg) / 3
    k = torch.arange(3, dtype=B.dtype, device=B.device)
    roots = -B[...,None]/3 + 2*torch.sqrt(-p[...,None]/3) * torch.cos(theta[...,None] - 2*torch.pi*k/3)
    return torch.sort(roots, dim=-1).values


def elliptic_K_agm(m, iterations=12):
    a, b = torch.ones_like(torch.as_tensor(m)), torch.sqrt(1 - torch.as_tensor(m))
    for _ in range(iterations):
        a, b = (a + b) / 2, torch.sqrt(a * b)
    return torch.pi / (2 * a)


def elliptic_E_quadrature(m, quadrature=48):
    """Complete E(m), evaluated by fixed quadrature for autograd support."""
    m = torch.as_tensor(m)
    import numpy as np
    n, w = np.polynomial.legendre.leggauss(quadrature)
    x = torch.as_tensor(n, dtype=m.dtype, device=m.device)
    weights = torch.as_tensor(w, dtype=m.dtype, device=m.device)
    theta = (x + 1) * (torch.pi / 4)
    return (weights * torch.sqrt(1 - m.unsqueeze(-1) * torch.sin(theta) ** 2)).sum(-1) * (torch.pi / 4)


def spin_average_exact(S_plus2, S_minus2, S3_2, quadrature=48):
    """Exact period average from Eq. (26), with the m=0 continuous limit."""
    p, n, s3 = torch.broadcast_tensors(torch.as_tensor(S_plus2),
                                        torch.as_tensor(S_minus2), torch.as_tensor(S3_2))
    m = (p - n) / (p - s3)
    K = elliptic_K_agm(m)
    E = elliptic_E_quadrature(m, quadrature)
    safe_m = torch.where(m.abs() < 1e-10, torch.ones_like(m), m)
    value = ((m - 1) * p + n + (E / K) * (p - n)) / safe_m
    return torch.where(m.abs() < 1e-10, (p + n) / 2, value)


def integration_constant_c1(J, L, v, S_average2):
    """Leading c1 from Eq. (18), J²=L²+2 c1/v+<S²>."""
    J, L, v, sav = torch.broadcast_tensors(*(torch.as_tensor(x) for x in (J, L, v, S_average2)))
    if bool(torch.any(v == 0)):
        raise ValueError("v must be nonzero")
    return 0.5 * v * (J * J - L * L - sav)


def averaged_spin_couplings(mean_s1l, mean_s2l, S_plus2, S_minus2, q, eta, v0):
    """Leading spin-projection averages quoted below Eq. (27)."""
    vals = torch.broadcast_tensors(*(torch.as_tensor(x) for x in
                                     (mean_s1l, mean_s2l, S_plus2, S_minus2, q, eta, v0)))
    s1l, s2l, p, n, q, eta, v0 = vals
    if bool(torch.any((q <= 0) | (q >= 1) | (eta <= 0) | (v0 < 0))):
        raise ValueError("averages require 0<q<1, eta>0 and v0>=0")
    correction = (p - n) ** 2 * v0 ** 2 / (32 * eta ** 2 * (1 - q) ** 2)
    return s1l ** 2 + correction, s2l ** 2 + q ** 2 * correction, s1l * s2l - q * correction


def jacobi_sn(u, m, iterations=16, quadrature=64):
    """Differentiable Jacobi sn via period reduction and fixed inversion of F."""
    u, m = torch.broadcast_tensors(torch.as_tensor(u), torch.as_tensor(m))
    import numpy as np
    n, w = np.polynomial.legendre.leggauss(quadrature)
    nodes = torch.as_tensor(n, dtype=u.dtype, device=u.device)
    weights = torch.as_tensor(w, dtype=u.dtype, device=u.device)
    K = elliptic_K_agm(m)
    # sn has period 4K and sn(2K-u)=sn(u); fold into the invertible [-K,K].
    r = torch.remainder(u + 2 * K, 4 * K) - 2 * K
    sign = torch.where(r >= 0, torch.ones_like(r), -torch.ones_like(r))
    # Use the selected one-sided sign so d|r|/dr is +1 at the origin.
    ar = sign * r
    folded = torch.where(ar > K, 2 * K - ar, ar)
    amp = sign * (folded / K * (torch.pi / 2)).clamp_max(torch.pi / 2 - 1e-12)
    for _ in range(iterations):
        z = (amp.unsqueeze(-1) * (nodes + 1) / 2)
        amp_sign = torch.where(amp >= 0, torch.ones_like(amp), -torch.ones_like(amp))
        integ = (weights * (1 - m.unsqueeze(-1) * torch.sin(z)**2).rsqrt()).sum(-1) * amp_sign * amp / 2
        integ = amp_sign * integ
        amp = amp - (integ - sign * folded) * torch.sqrt(1 - m * torch.sin(amp)**2)
    return torch.sin(amp)


def spin_oscillation(psi, S_plus2, S_minus2, S3_2):
    m = (S_plus2 - S_minus2) / (S_plus2 - S3_2)
    return S_plus2 + (S_minus2 - S_plus2) * jacobi_sn(psi, m) ** 2


def initial_phase(S2, S_plus2, S_minus2, S3_2, triple_product):
    """Select the Jacobi inverse branch using Eq. (24)'s triple-product sign."""
    m = (S_plus2 - S_minus2) / (S_plus2 - S3_2)
    y = ((S_plus2 - S2) / (S_plus2 - S_minus2)).clamp(0, 1).sqrt()
    amp = torch.asin(y)
    n, w = __import__('numpy').polynomial.legendre.leggauss(24)
    nodes = torch.as_tensor(n, dtype=amp.dtype, device=amp.device)
    weights = torch.as_tensor(w, dtype=amp.dtype, device=amp.device)
    z = amp.unsqueeze(-1) * (nodes + 1) / 2
    base = (weights * (1 - m.unsqueeze(-1) * torch.sin(z)**2).rsqrt()).sum(-1) * amp / 2
    return torch.where(triple_product < 0, -base, base)


def spin_average(S_plus2, S_minus2):
    return (S_plus2 + S_minus2) / 2


__all__ = ["initial_invariants", "cubic_coefficients", "ordered_roots",
           "elliptic_K_agm", "elliptic_E_quadrature",
           "spin_average_exact", "integration_constant_c1", "averaged_spin_couplings",
           "jacobi_sn", "spin_oscillation", "initial_phase", "spin_average"]

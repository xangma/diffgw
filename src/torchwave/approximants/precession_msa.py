"""NNLO and published two-spin multiple-scale precession angles.

The MSA branch includes elliptic nutation, secular phases and leading
oscillatory corrections. The numerical distinctions between published
implementation labels 220--224 are not specified by the equations; these
labels share the analytical prescription here, with their stated fallback
policy. See docs/reviews/2026-09-12-clean-room/pass-2-precession.md.

Mathematical references:
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503), Sec. IV & Appendix B
- Chatziioannou et al. (2017), Phys. Rev. D 95, 104004 (arXiv:1703.03967)
- Chatziioannou et al. (2013), Phys. Rev. D 88, 063011 (arXiv:1307.4418)

"""

from typing import Optional, Tuple
import math
import torch

from torchwave.constants import PI, EULER_GAMMA
from torchwave.approximants.imrphenomp import IMRPhenomP


MSA_FINAL_SPIN_DOMAIN_ERROR = (
    "MSA final_spin_mod=3 requires a valid nonaligned MSA remnant root; "
    "use final_spin_mod=0 for equal-mass or coalescing-root rows."
)


def resolve_precession_version(method: str, version: Optional[int]) -> int:
    """An explicit version takes precedence; method chooses an omitted version."""
    method_l = method.lower()
    if method_l not in ("nnlo", "msa", "spintaylor", "st", "spintaylort4", "pnr", "xpnr"):
        raise ValueError("precession_method must be 'nnlo', 'msa', 'spintaylor', or 'pnr'.")
    if version is None:
        if method_l in ("pnr", "xpnr"):
            return 330
        if method_l in ("spintaylor", "st", "spintaylort4"):
            return 320
        return 223 if method_l == "msa" else 102
    if version not in (101, 102, 103, 104, 220, 221, 222, 223, 224, 300, 310, 311, 320, 321, 330):
        raise ValueError(f"Unsupported precession version: {version}.")
    if version == 300:
        return 330
    return version


def compute_effective_chip(
    m1: torch.Tensor,
    m2: torch.Tensor,
    s1x: torch.Tensor,
    s1y: torch.Tensor,
    s2x: torch.Tensor,
    s2y: torch.Tensor,
) -> torch.Tensor:
    """Published effective in-plane spin, Eqs. (4.7)--(4.9), for m1 >= m2."""
    q_pratten = m1 / m2
    a1_coeff = 2.0 + 1.5 / q_pratten
    a2_coeff = 2.0 + 1.5 * q_pratten
    s1_perp = m1.square() * torch.linalg.vector_norm(torch.stack((s1x, s1y), dim=-1), dim=-1)
    s2_perp = m2.square() * torch.linalg.vector_norm(torch.stack((s2x, s2y), dim=-1), dim=-1)
    return torch.maximum(a1_coeff * s1_perp, a2_coeff * s2_perp) / (a1_coeff * m1.square())


def elliptic_f(phi: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
    """Real F(phi|m), for |phi| <= pi/2, by Carlson duplication.

    F = sin(phi) RF(cos(phi)^2, 1-m sin(phi)^2, 1). The duplication
    identity and symmetric Taylor expansion are DLMF 19.26 and 19.36.
    Fixed iteration counts preserve tensor broadcasting and autograd.
    """
    s = torch.sin(phi)
    x, y, z = torch.broadcast_tensors(torch.cos(phi).square(), 1-parameter*s*s, torch.ones_like(s))
    for _ in range(12):
        root_x, root_y, root_z = torch.sqrt(x), torch.sqrt(y), torch.sqrt(z)
        lam = root_x*root_y + root_x*root_z + root_y*root_z
        x, y, z = (x+lam)/4, (y+lam)/4, (z+lam)/4
    mean = (x+y+z)/3
    dx, dy, dz = (mean-x)/mean, (mean-y)/mean, (mean-z)/mean
    e2, e3 = dx*dy-dz*dz, dx*dy*dz
    rf = (1-e2/10+e3/14+e2*e2/24-3*e2*e3/44)/torch.sqrt(mean)
    return s*rf


def jacobi_sn_cn_dn(u: torch.Tensor, parameter: torch.Tensor):
    """Jacobi functions from the descending Landen transformation (DLMF 22.20)."""
    u, parameter = torch.broadcast_tensors(u, parameter)
    eps = torch.finfo(u.dtype).eps
    m = torch.clamp(parameter, 0., 1.-eps)
    a, b = torch.ones_like(m), torch.sqrt(1-m)
    ratios = []
    for _ in range(12):
        next_a = (a+b)/2
        ratios.append((a-b)/(2*next_a))
        a, b = next_a, torch.sqrt(a*b)
    period = 2*PI/a
    reduced = u-torch.round(u/period)*period
    phase = (2**12)*a*reduced
    for ratio in reversed(ratios):
        phase = (phase+torch.asin(torch.clamp(ratio*torch.sin(phase), -1., 1.)))/2
    sn, cn = torch.sin(phase), torch.cos(phase)
    dn = torch.sqrt(torch.clamp(1-m*sn*sn, min=eps))
    is_one = parameter == 1
    sech = 1/torch.cosh(torch.clamp(u, -80., 80.))
    return (torch.where(is_one, torch.tanh(u), sn),
            torch.where(is_one, sech, cn), torch.where(is_one, sech, dn))


def msa_phase_primitives(v, eta, c1, sav_sq):
    """Primitives with derivative J*v**(n-3), n=0..5.

    Chatziioannou (2017), D21--D26. D22 requires c1**2 in its logarithmic
    coefficient: differentiating the printed c1 expression fails Eq. 65.
    The MSA dynamics use the Newtonian L=eta/v, as in the paper's Eq. 4.
    """
    l = eta/v
    j = torch.sqrt(l*l+2*c1/v+sav_sq)
    sav = torch.sqrt(sav_sq)
    log1 = torch.log(torch.clamp(c1+j*eta+l*eta, min=torch.finfo(v.dtype).tiny))
    log2 = torch.log(torch.clamp(c1+j*sav*v+sav_sq*v, min=torch.finfo(v.dtype).tiny))
    invariant = c1*c1-eta*eta*sav_sq
    p0 = j/eta**4*(c1*c1/2-c1*eta*eta/(6*v)-sav_sq*eta*eta/3-eta**4/(3*v*v))-c1*invariant/(2*eta**5)*log1
    p1 = -j/(2*eta*eta)*(c1+eta*l)+invariant/(2*eta**3)*log1
    p2 = -j+sav*log2-c1/eta*log1
    p3 = j*v-eta*log1+c1/sav*log2
    p4 = j*v*(c1+v*sav_sq)/(2*sav_sq)-invariant/(2*sav**3)*log2
    p5 = -j*v*(c1*c1/(2*sav_sq**2)-c1*v/(6*sav_sq)-v*v/3-eta*eta/(3*sav_sq))+c1*invariant/(2*sav**5)*log2
    return torch.stack((p0, p1, p2, p3, p4, p5), dim=-1)


def msa_nutation_phase(v, v0, eta, delta, q, xi, c1, sav_sq, s1_sq, s2_sq, psi_ref):
    """Eq. 51, with C1/C2 obtained by expanding the defining cubic.

    Vieta's relations determine the two finite leading roots and the third
    root's v^-2, v^-1, v^0 terms. Expanding Eq. 50 then integrating gives
    the published 1PN phase, avoiding cancellation-prone nested C3 factors.
    """
    b2 = eta*delta*delta
    b1 = 2*eta*xi-4*c1
    b0 = s1_sq*(q-1)+s2_sq*(1/q-1)-2*sav_sq
    c2 = 4*c1*c1-4*eta*xi*c1+4*eta**3*xi*xi-2*b2*(s1_sq+s2_sq)
    d2 = 4*(1-q)/q*(s2_sq-q*s1_sq)*c1*c1+b2*(s1_sq-s2_sq)**2+4*delta*eta*xi*(s1_sq-s2_sq)*c1
    root_sum, root_product = -c2/b2, d2/b2
    finite_plus = (root_sum+torch.sqrt(torch.clamp(root_sum*root_sum-4*root_product, min=0.)))/2
    sqrt1 = b1/(2*b2)
    sqrt2 = (finite_plus+b0+root_sum)/(2*b2)-b1*b1/(8*b2*b2)
    psi1 = 1.5*(sqrt1-xi)
    psi2 = 3*(sqrt2-xi*sqrt1+743/336+11*eta/4)
    g0 = 5/(96*eta)
    def phase(x):
        return -3*g0*delta/4*(x**-3+psi1*x**-2+psi2/x)
    return psi_ref+phase(v)-phase(v0)


def msa_opening_angle(l, eta, c1, sav_sq, spin_sq):
    """MSA triangle, with the selected PN L in the averaged J(L) relation.

    Integrating Chatziioannou Eq. 40 with constant <S^2> gives
    J^2=L^2+2*c1*L/eta+<S^2>. Insert these J,L into Eq. 8 (Pratten 4.12).
    J remains independent of the instantaneous nutation phase. Keeping an
    instantaneous spin projection fixed while upgrading L would violate
    this precession-averaged relation.
    """
    j_sq = l*l+2*c1*l/eta+sav_sq
    numerator = j_sq+l*l-spin_sq
    transverse_sq = torch.clamp(4*j_sq*l*l-numerator*numerator, min=0.)
    return torch.atan2(torch.sqrt(transverse_sq), numerator)


def msa_oscillatory_angles(v, j, l, xi, eta, s1_sq, s2_sq, delta,
                           s3_sq, sm_sq, sp_sq, psi):
    """Zero-mean fast-time primitives of Eq. 30 and F1, at m=0.

    Direct real partial fractions avoid the ambiguous C_phi notation in
    Eq. F19. Their psi derivatives equal (Omega-<Omega>)/psi_dot, for both
    Euler angles, including the J factor in Eq. 30.
    """
    gap = sp_sq-sm_sq
    d0 = 4*j*j*l*l-(j*j+l*l-sp_sq)**2
    d2 = -2*gap*(j*j+l*l-sp_sq)
    d4 = -gap*gap
    c0 = 3/4*(1-xi*v)*v*v*(eta**3+4*eta**3*xi*v
        -2*eta*(j*j-sp_sq+2*(s1_sq-s2_sq)*delta)*v*v
        -4*eta*xi*(j*j-sp_sq)*v**3+(j*j-sp_sq)**2/eta*v**4)
    c2 = -1.5*eta*gap*(1+2*xi*v-(j*j-sp_sq)/eta**2*v*v)*(1-xi*v)*v**4
    quotient = -3/(4*eta)*(1-xi*v)*v**6  # c4/d4, regular at gap=0
    disc = torch.sqrt(torch.clamp(d2*d2-4*d0*d4, min=torch.finfo(v.dtype).tiny))
    # D(s)=d0(1-lambda_c*s)(1-lambda_d*s).
    lambda_c = (-d2+disc)/(2*d0)
    lambda_d = (-d2-disc)/(2*d0)
    rem0, rem1 = c0/d0-quotient, c2/d0-quotient*d2/d0
    residue_c = (-rem1-rem0*lambda_c)/(lambda_d-lambda_c)
    residue_d = rem0-residue_c
    sinp, cosp = torch.sin(psi), torch.cos(psi)
    def periodic_primitive(lam):
        root = torch.sqrt(torch.clamp(1-lam, min=torch.finfo(v.dtype).eps))
        # atan(k*tan(psi))-psi, continuously wrapped into (-pi/2,pi/2).
        return torch.atan2((root-1)*sinp*cosp, cosp*cosp+root*sinp*sinp)/root
    primitive_c, primitive_d = periodic_primitive(lambda_c), periodic_primitive(lambda_d)
    psi_dot = 3/(4*torch.sqrt(eta))*(1-xi*v)*v**6*torch.sqrt(torch.clamp(sp_sq-s3_sq, min=torch.finfo(v.dtype).tiny))
    alpha = j/psi_dot*(residue_c*primitive_c+residue_d*primitive_d)
    a_theta = (j*j+l*l-sp_sq)/(2*j*l)
    b_theta = gap/(2*j*l)
    # Btheta/lambda has a finite gap->0 limit. This function is used only
    # for nondegenerate roots; degenerate initializations select NNLO.
    epsilon = j/psi_dot*(residue_c*(a_theta+b_theta/lambda_c)*primitive_c
        +residue_d*(a_theta+b_theta/lambda_d)*primitive_d
        -b_theta*v**6/8*sinp*cosp*2)
    return alpha, epsilon


def compute_l_3pn(
    v: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    s_ell: torch.Tensor,
    sigma_ell: torch.Tensor,
) -> torch.Tensor:
    """3PN orbital angular momentum: Pratten (2021), G10a-d and G10f-g.

    Formula:
        L(v) = (eta / v) * [1 + L1*v^2 + L_15_SO*v^3 + L2*v^4 + L_25_SO*v^5 + L3*v^6]
    """
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3

    l1 = 1.5 + eta / 6.0
    l2 = 27.0 / 8.0 - (19.0 / 8.0) * eta + (eta * eta) / 24.0
    l3 = (
        135.0 / 16.0
        + (41.0 * (PI ** 2) / 24.0 - 6889.0 / 144.0) * eta
        + (31.0 / 24.0) * (eta * eta)
        + (7.0 / 1296.0) * (eta ** 3)
    )

    l_15_so = -(35.0 / 6.0) * s_ell - 2.5 * delta * sigma_ell
    l_25_so = (-77.0 / 8.0 + (427.0 / 72.0) * eta) * s_ell + delta * (-21.0 / 8.0 + (35.0 / 12.0) * eta) * sigma_ell
    poly = 1.0 + l1 * v2 + l_15_so * v3 + l2 * v4 + l_25_so * v5 + l3 * v6
    return (eta / v) * poly


def compute_l_4pn(v, eta, delta, s_ell, sigma_ell, chi1, chi2, all_spin=False):
    """Pratten G10, with the 3.5PN SO term from Bohe 1212.5520 Eq. 4.7.

    The original SO derivation has -81/16, correcting G10h's -81/6.
    Version 104 adds the leading quadratic/cubic spin terms G10i--j.
    """
    l4 = (-55*eta**4/31104-215*eta**3/1728
        +(356035/3456-2255*PI**2/576)*eta**2
        +eta*(-64/3*torch.log(16*v*v)-6455*PI**2/1536
              -128*EULER_GAMMA/3+98869/5760)+2835/128)
    so35 = (-405/16+1101*eta/16-29*eta**2/16)*s_ell
    so35 = so35+delta*(-81/16+117*eta/4-15*eta**2/16)*sigma_ell
    correction = l4*v**8+so35*v**7
    if all_spin:
        spin2 = (0.5+delta/2-eta)*chi1**2+2*eta*chi1*chi2+(0.5-delta/2-eta)*chi2**2
        spin3 = (chi1**3*(3*delta*eta/4-3*eta**2/2+3*eta/4)
            +chi1**2*chi2*(3*delta*eta/4+3*eta**2/2+3*eta/4)
            +chi1*chi2**2*(-3*delta*eta/4+3*eta**2/2+3*eta/4)
            +chi2**3*(-3*delta*eta/4-3*eta**2/2+3*eta/4))
        correction = correction+spin2*v**4+spin3*v**7
    return compute_l_3pn(v,eta,delta,s_ell,sigma_ell)+eta/v*correction


def solve_spin_roots(
    l0: torch.Tensor,
    j0_sq: torch.Tensor,
    s1_sq: torch.Tensor,
    s2_sq: torch.Tensor,
    xi: torch.Tensor,
    q_c: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the depressed cubic polynomial for spin magnitude roots S_3^2 <= S_-^2 <= S_+^2.

    Literature:
        Chatziioannou et al. (2017) Appendix A / Appendix B Eq. 21.
        q_c = m2 / m1 <= 1 (Chatziioannou convention where body 1 is primary).
    """
    l0_sq = l0 * l0
    j_minus_l = j0_sq - l0_sq
    inv_qc = 1.0 / q_c
    one_minus_qc_over_qc = (1.0 - q_c) * inv_qc

    # Cubic coefficients P(z) = z^3 + B z^2 + C z + D
    b_coeff = (l0_sq + s1_sq) * q_c + 2.0 * l0 * xi - 2.0 * j0_sq - s1_sq - s2_sq + (l0_sq + s2_sq) * inv_qc
    c_coeff = (
        j_minus_l ** 2
        - 2.0 * l0 * xi * j_minus_l
        - 2.0 * one_minus_qc_over_qc * (s1_sq - q_c * s2_sq) * l0_sq
        + 4.0 * eta * l0_sq * (xi ** 2)
        - 2.0 * delta * (s1_sq - s2_sq) * xi * l0
        + 2.0 * one_minus_qc_over_qc * (q_c * s1_sq - s2_sq) * j0_sq
    )
    d_coeff = (
        one_minus_qc_over_qc * (s2_sq - q_c * s1_sq) * (j_minus_l ** 2)
        + ((delta ** 2) / eta) * ((s1_sq - s2_sq) ** 2) * l0_sq
        + 2.0 * delta * l0 * xi * (s1_sq - s2_sq) * j_minus_l
    )

    # Depressed cubic substitution z = y - B/3
    p = c_coeff - (b_coeff ** 2) / 3.0
    qc_dep = (2.0 / 27.0) * (b_coeff ** 3) - (b_coeff * c_coeff) / 3.0 + d_coeff

    # Discriminant condition: -(p/3)^3 - (qc_dep/2)^2 >= 0
    p_third = -p / 3.0
    safe_p_third = torch.clamp(p_third, min=1e-18)
    r_val = torch.sqrt(safe_p_third)
    r3 = r_val ** 3

    cos_3theta = torch.clamp(-qc_dep / (2.0 * r3 + 1e-30), -1.0 + 1e-15, 1.0 - 1e-15)
    theta = torch.acos(cos_3theta) / 3.0

    z0 = 2.0 * r_val * torch.cos(theta) - b_coeff / 3.0
    z1 = 2.0 * r_val * torch.cos(theta - 2.0 * PI / 3.0) - b_coeff / 3.0
    z2 = 2.0 * r_val * torch.cos(theta - 4.0 * PI / 3.0) - b_coeff / 3.0

    roots_stack = torch.stack([z0, z1, z2], dim=-1)
    sorted_roots, _ = torch.sort(roots_stack, dim=-1)

    s3_sq = sorted_roots[..., 0]
    sm_sq = sorted_roots[..., 1]
    sp_sq = sorted_roots[..., 2]

    # Repeated roots cannot be distinguished from a cubic's roundoff-sized
    # discriminant. Use a scale-dependent error bound, not an absolute spin
    # or mass-ratio cutoff. Such initializations select the documented NNLO
    # fallback in versions 220/223.
    disc = (safe_p_third ** 3) - (qc_dep / 2.0) ** 2
    eps = torch.finfo(l0.dtype).eps
    root_scale = torch.maximum(torch.abs(b_coeff), torch.sqrt(torch.abs(c_coeff)))
    disc_scale = torch.maximum(safe_p_third**3, (qc_dep/2)**2)
    is_valid = (
        (disc > 32*eps*disc_scale)
        & (sp_sq >= sm_sq)
        & (sm_sq >= -32*eps*root_scale)
        & ((sp_sq - sm_sq) > 32*math.sqrt(eps)*root_scale)
        & ((1.0 / q_c - 1.0) > 32*math.sqrt(eps))
    )

    return s3_sq, sm_sq, sp_sq, is_valid



def evaluate_msa_dynamics(v, v0, eta, delta, q_c, s1, s2, l_pn,
                          anchor_reference=True, return_averages=False):
    """Published two-spin MSA solution and initialization validity.

    Conservative evolution uses L_N=eta/v. The opening angle uses the
    selected PN L in the precession-averaged J(L) solution and triangle.
    Setting anchor_reference=False retains the published phase primitives'
    zero integration constants, as required by Table IV options 5 and 7.
    """
    s1_sq, s2_sq = (s1*s1).sum(-1), (s2*s2).sum(-1)
    s0_sq = ((s1+s2)**2).sum(-1)
    xi = (1+q_c)*s1[..., 2]+(1+1/q_c)*s2[..., 2]
    l0 = eta/v0
    j0_sq = l0*l0+2*l0*(s1[..., 2]+s2[..., 2])+s0_sq
    _, sm_sq, sp_sq, roots_valid = solve_spin_roots(l0, j0_sq, s1_sq, s2_sq, xi, q_c, eta, delta)
    sav0_sq = (sp_sq+sm_sq)/2
    c10 = v0*(j0_sq-l0*l0-sav0_sq)/2
    cp0, cm0 = eta*eta*sp_sq-c10*c10, eta*eta*sm_sq-c10*c10
    # D6 and E1--E2 require real sqrt(cp*cm). The frozen-root PN
    # resummation can violate this even when the original cubic is real.
    # That is a failed MSA initialization, not a value to clamp to zero.
    valid = roots_valid & (s1_sq > 0) & (s2_sq > 0) & (cp0 > 0) & (cm0 > 0)
    # A failed initialization is an entire binary, never selected frequencies.
    # Evaluate a regular surrogate in masked rows to keep inactive gradients
    # finite before the caller selects the published NNLO fallback.
    def regular(x, substitute):
        return torch.where(valid, x, torch.full_like(x, substitute))
    q_c, eta, delta = regular(q_c, .5), regular(eta, 2/9), regular(delta, 1/3)
    v0 = regular(v0, .2)
    vec_mask = valid.unsqueeze(-1)
    s1 = torch.where(vec_mask, s1, s1.new_tensor([.12,.04,.08]))
    s2 = torch.where(vec_mask, s2, s2.new_tensor([-.02,.03,-.01]))
    s1_sq, s2_sq = (s1*s1).sum(-1), (s2*s2).sum(-1)
    s0_sq = ((s1+s2)**2).sum(-1)
    xi = (1+q_c)*s1[..., 2]+(1+1/q_c)*s2[..., 2]
    l0 = eta/v0
    j0_sq = l0*l0+2*l0*(s1[..., 2]+s2[..., 2])+s0_sq
    s3_sq, sm_sq, sp_sq, _ = solve_spin_roots(l0,j0_sq,s1_sq,s2_sq,xi,q_c,eta,delta)
    sav_sq = (sp_sq+sm_sq)/2
    c1 = v0*(j0_sq-l0*l0-sav_sq)/2
    m1_norm, m2_norm = 1/(1+q_c), q_c/(1+q_c)
    # Dissipative coefficients
    g0 = 5.0 / (96.0 * eta)
    a0 = 96.0 * eta / 5.0
    a2 = -743.0 / 336.0 - 2.75 * eta

    denom_spin = eta * (1.0 - q_c * q_c)
    s1_ell_avg = (c1 * (1.0 + q_c) - q_c * eta * xi) / denom_spin
    s2_ell_avg = -q_c * (c1 * (1.0 + q_c) - eta * xi) / denom_spin

    variance = (sp_sq-sm_sq)**2*v0*v0/(32*eta*eta*(1-q_c)**2)
    cross_avg = s1_ell_avg*s2_ell_avg-q_c*variance
    s1_ell_sq_avg = s1_ell_avg*s1_ell_avg+variance
    s2_ell_sq_avg = s2_ell_avg*s2_ell_avg+q_c*q_c*variance

    beta3 = (113.0 / 12.0 + 25.0 * q_c / 4.0) * s1_ell_avg + (113.0 / 12.0 + 25.0 / (4.0 * q_c)) * s2_ell_avg
    a3 = 4.0 * PI - beta3
    beta5 = (
        ((31319.0 / 1008.0 - (1159.0 / 24.0) * eta) + q_c * (809.0 / 84.0 - (281.0 / 8.0) * eta)) * s1_ell_avg
        + ((31319.0 / 1008.0 - (1159.0 / 24.0) * eta) + (1.0 / q_c) * (809.0 / 84.0 - (281.0 / 8.0) * eta)) * s2_ell_avg
    )
    sigma4 = (
        (247.0 / 48.0 * (sav_sq / 2.0 - (s1_sq + s2_sq) / 2.0) - (721.0 / 48.0) * cross_avg) / eta
        + (233.0 / 96.0 * s1_sq - (719.0 / 96.0) * s1_ell_sq_avg) / (m1_norm ** 2)
        + (233.0 / 96.0 * s2_sq - (719.0 / 96.0) * s2_ell_sq_avg) / (m2_norm ** 2)
    )
    a4 = 34103.0 / 18144.0 + (13661.0 / 2016.0) * eta + (59.0 / 18.0) * (eta ** 2) - sigma4
    a5 = -4159.0 / 672.0 * PI - (189.0 / 8.0) * PI * eta - beta5

    g2 = -a2 / a0
    g3 = -a3 / a0
    g4 = -(a4 - a2 * a2) / a0
    g5 = -(a5 - 2.0 * a3 * a2) / a0

    # Alpha coefficients
    rm = sp_sq - sm_sq
    cp = sp_sq * (eta ** 2) - c1 * c1
    cm = sm_sq * (eta ** 2) - c1 * c1
    sqrt_cp_cm = torch.sqrt(torch.clamp(cp * cm, min=1e-24))

    # D4's printed 3*eta/4 conflicts with B5; the leading expansion of
    # Eq. 30 requires 3/(4*eta).
    a1_om = 0.5 + 0.75 / eta
    a2_om = -3.0 / (4.0 * eta) * xi
    ad = (-3.0 * (s1_sq - s2_sq) * eta * delta + 3.0 * (c1 / eta) * (c1 - 2.0 * xi * (eta ** 2))) / (4.0 * sqrt_cp_cm)
    cd = (3.0 / 128.0) * (rm ** 2) / (eta * sqrt_cp_cm)
    hd = (c1 / (eta ** 2)) * (1.0 - (cp + cm) / (2.0 * sqrt_cp_cm))
    fd = ((cp + cm) / (8.0 * (eta ** 4))) * (1.0 - (cp + cm) / (2.0 * sqrt_cp_cm))

    om_z0 = a1_om + ad
    om_z1 = a2_om - ad * xi - ad * hd
    om_z2 = ad * hd * xi + cd - ad * fd + ad * (hd ** 2)
    om_z3 = (ad * fd - cd - ad * (hd ** 2)) * (xi + hd) + ad * fd * hd
    om_z4 = (cd + ad * (hd ** 2) - 2.0 * ad * fd) * (hd * xi + hd * hd - fd) - ad * (fd ** 2)
    om_z5 = (cd - ad * fd + ad * (hd ** 2)) * fd * (xi + 2.0 * hd) - (cd + ad * (hd ** 2) - 2.0 * ad * fd) * (hd ** 2) * (xi + hd) - ad * (fd ** 2) * hd

    avg_om_z0 = 3.0 * g0 * om_z0
    avg_om_z1 = 3.0 * g0 * om_z1
    avg_om_z2 = 3.0 * (g0 * om_z2 + g2 * om_z0)
    avg_om_z3 = 3.0 * (g0 * om_z3 + g2 * om_z1 + g3 * om_z0)
    avg_om_z4 = 3.0 * (g0 * om_z4 + g2 * om_z2 + g3 * om_z1 + g4 * om_z0)
    avg_om_z5 = 3.0 * (g0 * om_z5 + g2 * om_z3 + g3 * om_z2 + g4 * om_z1 + g5 * om_z0)

    # Gamma / Epsilon coefficients
    gd = (3.0 / 64.0) * (rm ** 2) / (eta ** 3) * (c1 - (eta ** 2) * xi) / sqrt_cp_cm
    om_zeta0 = om_z0
    om_zeta1 = om_z1 + (c1 / (eta ** 2)) * om_z0
    om_zeta2 = om_z2 + (c1 / (eta ** 2)) * om_z1
    om_zeta3 = om_z3 + (c1 / (eta ** 2)) * om_z2 + gd
    om_zeta4 = om_z4 + (c1 / (eta ** 2)) * om_z3 - gd * xi - gd * hd
    om_zeta5 = om_z5 + (c1 / (eta ** 2)) * om_z4 + gd * hd * xi + gd * (hd * hd - fd)

    avg_om_zeta0 = -g0 * om_zeta0
    avg_om_zeta1 = -1.5 * g0 * om_zeta1
    avg_om_zeta2 = -3.0 * (g0 * om_zeta2 + g2 * om_zeta0)
    avg_om_zeta3 = 3.0 * (g0 * om_zeta3 + g2 * om_zeta1 + g3 * om_zeta0)
    avg_om_zeta4 = 3.0 * (g0 * om_zeta4 + g2 * om_zeta2 + g3 * om_zeta1 + g4 * om_zeta0)
    avg_om_zeta5 = 1.5 * (g0 * om_zeta5 + g2 * om_zeta3 + g3 * om_zeta2 + g4 * om_zeta1 + g5 * om_zeta0)


    alpha_coeff = torch.stack((avg_om_z0,avg_om_z1,avg_om_z2,avg_om_z3,avg_om_z4,avg_om_z5),dim=-1)
    # Pratten Sec. IV B / Khan Sec. II E number the six frequency powers
    # f^(-4/3+k/3) by k=1,...,6. Their O(5) drops the sixth power: retain
    # coefficients n=0,...,4 in the zero-based Eq. 4.11 / Eq. 66 sum.
    alpha_coeff = alpha_coeff*alpha_coeff.new_tensor([1,1,1,1,1,0])
    alpha_primitives = msa_phase_primitives(v,eta,c1,sav_sq)
    if anchor_reference:
        alpha_primitives = alpha_primitives-msa_phase_primitives(v0,eta,c1,sav_sq)
    alpha_secular = (alpha_primitives*alpha_coeff).sum(-1)
    def epsilon_secular(x):
        # F9 multiplies log(v): it integrates a v^-1 derivative.
        return eta*(avg_om_zeta0*x**-3+avg_om_zeta1*x**-2+avg_om_zeta2/x
            +avg_om_zeta3*torch.log(x)+avg_om_zeta4*x)
    eps_secular = epsilon_secular(v)
    if anchor_reference:
        eps_secular = eps_secular-epsilon_secular(v0)
    initial_parameter = torch.clamp((sp_sq-sm_sq)/(sp_sq-s3_sq),0.,1.-torch.finfo(v.dtype).eps)
    initial_fraction = torch.clamp((sp_sq-s0_sq)/(sp_sq-sm_sq),0.,1.)
    # Eq. 28 fixes the branch of sn^-1 from the initial scalar triple product.
    orientation = s1[...,0]*s2[...,1]-s1[...,1]*s2[...,0]
    sign = torch.where(orientation < 0, -torch.ones_like(orientation), torch.ones_like(orientation))
    psi_ref = sign*elliptic_f(torch.asin(torch.sqrt(initial_fraction)),initial_parameter)
    psi = msa_nutation_phase(v,v0,eta,delta,q_c,xi,c1,sav_sq,s1_sq,s2_sq,psi_ref)
    l = eta/v
    j_sq = l*l+2*c1/v+sav_sq
    j = torch.sqrt(torch.clamp(j_sq,min=torch.finfo(v.dtype).tiny))
    s3, sm, sp, _ = solve_spin_roots(l,j_sq,s1_sq,s2_sq,xi,q_c,eta,delta)
    parameter = torch.clamp((sp-sm)/(sp-s3),0.,1.-torch.finfo(v.dtype).eps)
    sn, _, _ = jacobi_sn_cn_dn(psi,parameter)
    spin_sq = sp+(sm-sp)*sn*sn
    beta = msa_opening_angle(l_pn,eta,c1,sav_sq,spin_sq)
    alpha_osc, eps_osc = msa_oscillatory_angles(v,j,l,xi,eta,s1_sq,s2_sq,delta,s3,sm,sp,psi)
    alpha = alpha_secular+alpha_osc
    epsilon = eps_secular+eps_osc
    if anchor_reference:
        alpha_osc0, eps_osc0 = msa_oscillatory_angles(v0,torch.sqrt(j0_sq),l0,xi,eta,s1_sq,s2_sq,delta,s3_sq,sm_sq,sp_sq,psi_ref)
        alpha = alpha-alpha_osc0
        epsilon = epsilon-eps_osc0
    if return_averages:
        # These are the precession-averaged quantities entering Eq. (4.20)
        # of Pratten et al. (arXiv:2004.06503v2). Keep this opt-in so the
        # historical four-value low-level API remains unchanged.
        return alpha, beta, -epsilon, valid, sav_sq, s1_ell_avg, s2_ell_avg
    return alpha,beta,-epsilon,valid


def compute_msa_initialization_validity(v0, m1, m2, s1x, s1y, s1z,
                                        s2x, s2y, s2z):
    """Return the published two-spin MSA initialization-validity mask."""
    (v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z) = torch.broadcast_tensors(
        v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
    )
    total_mass = m1 + m2
    m1_norm, m2_norm = m1 / total_mass, m2 / total_mass
    eta = m1_norm * m2_norm
    delta = m1_norm - m2_norm
    q_c = m2_norm / m1_norm
    s1 = m1_norm.square().unsqueeze(-1) * torch.stack(
        (s1x, s1y, s1z), dim=-1
    )
    s2 = m2_norm.square().unsqueeze(-1) * torch.stack(
        (s2x, s2y, s2z), dim=-1
    )
    s1_sq = (s1 * s1).sum(dim=-1)
    s2_sq = (s2 * s2).sum(dim=-1)
    s0_sq = ((s1 + s2) ** 2).sum(dim=-1)
    xi = (1 + q_c) * s1[..., 2] + (1 + 1 / q_c) * s2[..., 2]
    l0 = eta / v0
    j0_sq = l0 * l0 + 2 * l0 * (s1[..., 2] + s2[..., 2]) + s0_sq
    _, sm_sq, sp_sq, roots_valid = solve_spin_roots(
        l0, j0_sq, s1_sq, s2_sq, xi, q_c, eta, delta
    )
    sav0_sq = (sp_sq + sm_sq) / 2
    c10 = v0 * (j0_sq - l0 * l0 - sav0_sq) / 2
    cp0 = eta * eta * sp_sq - c10 * c10
    cm0 = eta * eta * sm_sq - c10 * c10
    return roots_valid & (s1_sq > 0) & (s2_sq > 0) & (cp0 > 0) & (cm0 > 0)


def compute_msa_averaged_couplings(v0, m1, m2, s1x, s1y, s1z,
                                   s2x, s2y, s2z):
    """Return the MSA couplings needed by the precession-averaged remnant.

    The returned values are ``(<S^2>_pr, <S1.Lhat>_pr,
    <S2.Lhat>_pr, valid)`` in total-mass-normalized units.  With
    ``s_i = S_i/M^2`` and ``M=m1+m2``, ``<S^2>_pr`` is numerically
    ``<S^2>_pr/M^4`` while each projection is numerically
    ``<S_i.Lhat>_pr/M^2``.  The construction is the same 3PN MSA
    initialization used for the angles, evaluated at the reference
    frequency, and follows Eqs. (A9)--(A11) of Chatziioannou et al.
    (arXiv:1703.03967) as summarized by Pratten et al.
    (arXiv:2004.06503v2), Eq. (4.20).
    The returned couplings are physically usable only where ``valid`` is true;
    callers must choose their own documented behavior for invalid rows.
    """
    (v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z) = torch.broadcast_tensors(
        v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
    )
    total_mass = m1 + m2
    m1_norm, m2_norm = m1 / total_mass, m2 / total_mass
    eta = m1_norm * m2_norm
    delta = m1_norm - m2_norm
    q_c = m2_norm / m1_norm
    s1_sq_norm = m1_norm * m1_norm
    s2_sq_norm = m2_norm * m2_norm
    s1 = s1_sq_norm.unsqueeze(-1) * torch.stack((s1x, s1y, s1z), dim=-1)
    s2 = s2_sq_norm.unsqueeze(-1) * torch.stack((s2x, s2y, s2z), dim=-1)
    s_ell = s1[..., 2] + s2[..., 2]
    sigma_ell = m2_norm * s2z - m1_norm * s1z
    l0 = compute_l_3pn(v0, eta, delta, s_ell, sigma_ell)
    result = evaluate_msa_dynamics(
        v0, v0, eta, delta, q_c, s1, s2, l0,
        anchor_reference=True, return_averages=True,
    )
    _, _, _, valid, sav_sq, s1_ell_avg, s2_ell_avg = result
    return sav_sq, s1_ell_avg, s2_ell_avg, valid


def compute_msa_final_spin_squared(
    s_av_sq, s1_l_avg, s2_l_avg, l_fin
):
    """Evaluate the published MSA ``<S_fin^2>`` identity.

    All arguments use total-mass-normalized units: ``s_av_sq`` is
    ``<S_av^2>/M^4``, ``s1_l_avg`` and ``s2_l_avg`` are projections divided
    by ``M^2``, and ``l_fin`` is ``L_fin/M^2``.  The return value is
    ``<S_fin^2>/M^4``.  ``l_fin`` is explicit because the source defines it
    using ``M_fin`` in Eq. (4.18) but prints ``M`` in the FinalSpinMod3
    normalization of Eq. (4.29); this helper intentionally performs no normalization,
    clamping, or signed-Kerr conversion.
    """
    return s_av_sq + l_fin.square() + 2.0 * l_fin * (s1_l_avg + s2_l_avg)


def compute_msa_final_spin(
    v0: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
    s1x: torch.Tensor,
    s1y: torch.Tensor,
    s1z: torch.Tensor,
    s2x: torch.Tensor,
    s2y: torch.Tensor,
    s2z: torch.Tensor,
) -> torch.Tensor:
    """Return the signed Kerr parameter from the published MSA remnant path.

    This combines the carrier prescription in Eq. (31) with the averaged
    identity in Eq. (33) of Pratten et al. (arXiv:2004.06503v2), and applies
    the physical Kerr normalization ``|S_fin|/M_fin^2``.  The printed Eq. (42)
    uses an ambiguous mass symbol; this explicit source-mass normalization is
    chosen from Eq. (30), and is not a claim of LAL parity.  Inputs ``m1,m2`` are source masses
    (in any common units), spins are dimensionless, and all intermediate
    angular momenta are normalized by the *source* total mass squared.
    The aligned fit supplies ``a_parallel`` and the final-mass fraction; thus
    Eq. (31) gives ``L_fin/M^2 = (M_fin/M)^2*a_parallel-S_parallel/M^2``.

    The MSA averaged couplings are singular at equal masses in the published
    prescription.  Callers must handle those rows explicitly.
    The carrier ``a_parallel`` is the aligned-spin fit evaluated from the
    initial aligned projections; the ``S_parallel`` subtraction uses the
    averaged projections returned by Eq. (33), which is the precession-averaged
    interpretation of Eq. (31) used by this API.
    """
    from torchwave.approximants.imrphenom_xas import XASParameterFits

    (v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z) = torch.broadcast_tensors(
        v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
    )
    is_comp = hasattr(torch.compiler, "is_compiling") and torch.compiler.is_compiling()
    if not is_comp:
        if any(bool(torch.any(~torch.isfinite(x))) for x in
               (v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z)):
            raise ValueError("MSA final spin requires finite masses, spins, and v0.")
        if bool(torch.any((m1 <= 0) | (m2 <= 0) | (v0 <= 0))):
            raise ValueError("MSA final spin requires positive masses and v0.")
        if bool(torch.any(torch.stack((s1x, s1y, s1z), -1).square().sum(-1) > 1.0 + 1e-12)) or bool(torch.any(torch.stack((s2x, s2y, s2z), -1).square().sum(-1) > 1.0 + 1e-12)):
            raise ValueError("MSA final spin requires each dimensionless spin magnitude <= 1.")
    total_mass = m1 + m2
    eta = m1 * m2 / total_mass.square()
    aligned = (s1x == 0) & (s1y == 0) & (s2x == 0) & (s2y == 0)
    # The published MSA averages are singular at q=1, while the physical
    # aligned limit is supplied directly by the calibrated nonprecessing fit.
    a_aligned = XASParameterFits.fit_final_spin(eta, s1z, s2z)
    m_final_fit = 1.0 - XASParameterFits.fit_erad(eta, s1z, s2z)
    if not is_comp:
        if bool(torch.any(~torch.isfinite(a_aligned) | (torch.abs(a_aligned) > 1.0 + 1e-10))):
            raise ValueError("MSA final spin fit is outside the Kerr domain.")
        if bool(torch.any(~torch.isfinite(m_final_fit) | (m_final_fit <= 0))):
            raise ValueError("MSA final spin is outside the positive remnant-mass domain.")
        if bool(torch.all(aligned)):
            return a_aligned
    sav_sq, s1_l_avg, s2_l_avg, valid = compute_msa_averaged_couplings(
        v0, m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
    )
    unequal = m1 != m2
    # Only machine-level root degeneracies are assigned the analytic aligned
    # limit.  A genuinely failed nonaligned root is an unsupported point and
    # must be surfaced before interpreting any surrogate coupling values.
    inplane_sq = s1x.square() + s1y.square() + s2x.square() + s2y.square()
    eps = torch.finfo(v0.dtype).eps
    numerical_degeneracy = inplane_sq <= 10.0 * eps * (
        1.0 + s1z.square() + s2z.square()
    )
    unsupported = ((~unequal) & (~aligned)) | (
        (~valid) & (~numerical_degeneracy)
    )
    if is_comp:
        torch._assert(~torch.any(unsupported), MSA_FINAL_SPIN_DOMAIN_ERROR)
    elif bool(torch.any(unsupported)):
        raise ValueError(MSA_FINAL_SPIN_DOMAIN_ERROR)

    a_parallel = a_aligned
    m_final = m_final_fit
    s_parallel = s1_l_avg + s2_l_avg
    l_fin = m_final.square() * a_parallel - s_parallel
    s_fin_sq = compute_msa_final_spin_squared(sav_sq, s1_l_avg, s2_l_avg, l_fin)
    if not is_comp:
        if bool(torch.any(~torch.isfinite(s_fin_sq) | (s_fin_sq < -1e-12))):
            raise ValueError("MSA final spin produced an invalid remnant-spin square.")
    magnitude = torch.sqrt(torch.clamp(s_fin_sq, min=0.0)) / m_final.square()
    # Eq. (31) fixes the sign in the co-precessing frame by the aligned fit.
    signed = torch.where(a_parallel < 0.0, -magnitude, magnitude)
    if not is_comp:
        if bool(torch.any(~torch.isfinite(magnitude) | (magnitude > 1.0 + 1e-10))):
            raise ValueError("MSA final spin exceeds the physical Kerr bound.")
    return torch.where(numerical_degeneracy, a_aligned, signed)

def evaluate_msa_precession(
    v: torch.Tensor,
    v0: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
    s1x: torch.Tensor,
    s1y: torch.Tensor,
    s1z: torch.Tensor,
    s2x: torch.Tensor,
    s2y: torch.Tensor,
    s2z: torch.Tensor,
    iota: torch.Tensor,
    phi_ref: torch.Tensor,
    prec_version: Optional[int] = 223,
    convention: int = 1,
    return_carrier_phase: bool = False,
    observer_azimuth: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """Evaluate Euler angles, observer inclination in J, and basis rotation.

    Versions 101--104 select NNLO angles with 2PN, 3PN, 4PN or 4PN plus
    leading quadratic/cubic spin angular momentum. Versions 220--224 share
    the published two-spin MSA prescription; 221/222 reject degenerate
    initialization, while 220/223/224 use NNLO in those rows.
    Conventions 0, 1, 5, 6 and 7 use Pratten Table IV. Pv2 alpha is obtained
    from the published observer-to-J rotation in T1500602 Eqs. (37)--(40).
    With the same source-frame spin input it agrees modulo 2pi with C16.
    observer_azimuth adds an independent source-frame observer azimuth to C4;
    it does not change Table IV's nonprecessing carrier-phase argument.

    Returns:
        alpha: Azimuthal precession angle (same shape as v)
        beta: Opening angle of precession cone (same shape as v)
        gamma: Coprecessing angle gamma = -epsilon (same shape as v)
        theta_jn: Line of sight angle in J frame (scalar or batched)
        zeta: Polarization basis rotation angle (scalar or batched)
        carrier_phase: Optional sixth result, the Table IV nonprecessing
            phiRef argument, when return_carrier_phase=True.
    """
    # This low-level evaluator's declared default is the published MSA
    # prescription (223).  Preserve that same default when callers pass
    # ``None`` explicitly; wrapper dispatch resolves NNLO/MSA before entry.
    prec_version = resolve_precession_version("msa", prec_version)
    if convention not in (0, 1, 5, 6, 7):
        raise ValueError(f"Unsupported precession convention: {convention}.")
    total_mass = m1 + m2
    m1_norm = m1 / total_mass
    m2_norm = m2 / total_mass
    eta = m1 * m2 / (total_mass ** 2)
    delta = (m1 - m2) / total_mass
    q_pratten = m1 / m2
    q_c = m2 / m1  # Chatziioannou convention <= 1

    # Spin vectors in M=1 geometric units
    s1_norm2 = m1_norm * m1_norm
    s2_norm2 = m2_norm * m2_norm
    s1x_g, s1y_g, s1z_g = s1_norm2 * s1x, s1_norm2 * s1y, s1_norm2 * s1z
    s2x_g, s2y_g, s2z_g = s2_norm2 * s2x, s2_norm2 * s2y, s2_norm2 * s2z

    in_plane_sq = s1x * s1x + s1y * s1y + s2x * s2x + s2y * s2y
    is_aligned = in_plane_sq == 0

    s_ell = s1z_g + s2z_g
    sigma_ell = m2_norm * s2z - m1_norm * s1z

    # Angular momentum at v0
    use_3pn_l = prec_version in (102, 220, 221, 222, 223, 224)
    if use_3pn_l:
        l0 = compute_l_3pn(v0, eta, delta, s_ell, sigma_ell)
        l_dyn = compute_l_3pn(v, eta, delta, s_ell, sigma_ell)
    elif prec_version == 101:
        l0 = IMRPhenomP.orbital_angular_momentum(v0, eta)
        l_dyn = IMRPhenomP.orbital_angular_momentum(v, eta)
    else:  # 103/104
        l0 = compute_l_4pn(v0,eta,delta,s_ell,sigma_ell,s1z,s2z,prec_version == 104)
        l_dyn = compute_l_4pn(v,eta,delta,s_ell,sigma_ell,s1z,s2z,prec_version == 104)

    j0x = s1x_g + s2x_g
    j0y = s1y_g + s2y_g
    j0z = l0 + s1z_g + s2z_g
    use_msa = prec_version in (220, 221, 222, 223, 224)
    if use_msa:
        alpha_msa_dyn, beta_msa, gamma_msa_dyn, roots_valid = evaluate_msa_dynamics(
            v,v0,eta,delta,q_c,
            torch.stack((s1x_g,s1y_g,s1z_g),dim=-1),
            torch.stack((s2x_g,s2y_g,s2z_g),dim=-1),l_dyn,
            anchor_reference=convention not in (5,7),
        )
        if prec_version in (221,222) and not bool(torch.all(roots_valid)):
            raise ValueError("The MSA precession system is degenerate at the reference frequency.")
        fallback_to_nnlo = ~roots_valid
    else:
        fallback_to_nnlo = torch.ones_like(eta,dtype=torch.bool)
        alpha_msa_dyn = gamma_msa_dyn = beta_msa = torch.zeros_like(v)

    # NNLO fallback and remnant paths share the same exact-zero-safe effective
    # chi_p construction.
    chip = compute_effective_chip(m1, m2, s1x, s1y, s2x, s2y)
    chi_l = m1_norm * s1z + m2_norm * s2z

    (
        a_m3, a_m2, a_m1, a_1, a_log,
        e_m3, e_m2, e_m1, e_1, e_log,
    ) = IMRPhenomP.nnlo_angle_coefficients(
        eta=eta,
        delta=delta,
        m1_norm=m1_norm,
        chip=chip,
        chi_l=chi_l,
    )

    alpha_nnlo_dyn, epsilon_nnlo_dyn = IMRPhenomP.evaluate_nnlo_angles(
        v=v,
        v_ref=torch.as_tensor(v0, device=v.device, dtype=v.dtype),
        a_minus3=a_m3, a_minus2=a_m2, a_minus1=a_m1, a_1=a_1, a_log=a_log,
        e_minus3=e_m3, e_minus2=e_m2, e_minus1=e_m1, e_1=e_1, e_log=e_log,
        anchor_reference=convention not in (5,7),
    )
    sp_geom = chip * (m1_norm * m1_norm)
    sl_geom = (m1_norm * m1_norm) * s1z + (m2_norm * m2_norm) * s2z
    l_plus_sl = l_dyn + sl_geom
    # The NNLO PN expansion has cos(beta)=+1 at leading order. At beta=pi
    # its alpha=epsilon zero-cone limit does not cancel in the Wigner
    # rotation (Pratten Eq. 3.1/4.4; T1500602 Eqs. 22--24). Reject that
    # unsupported secular domain rather than inventing a sign resummation.
    # The wrappers return a separate exact carrier for convention-1 aligned
    # rows, so those rows do not require any analytical precessing rotation.
    requires_nnlo = fallback_to_nnlo & (~is_aligned if convention == 1 else torch.ones_like(is_aligned))
    if v.dim() > requires_nnlo.dim():
        requires_nnlo = requires_nnlo.unsqueeze(-1)
    unsupported_nnlo = (l_plus_sl <= 0) | (l0+sl_geom <= 0)
    if not (hasattr(torch.compiler, "is_compiling") and torch.compiler.is_compiling()):
        if bool(torch.any(requires_nnlo & unsupported_nnlo)):
            raise ValueError("NNLO precessing rotation is unsupported when L + S_parallel <= 0 "
                             "at the reference or an evaluated frequency; the secular expansion "
                             "does not support J anti-aligned with L.")
    # Eq. (4.4): atan2 also preserves the zero-cone and L+S_parallel<0 limits.
    beta_nnlo = torch.atan2(sp_geom, l_plus_sl)

    # Frame geometry at reference frequency f_ref
    # J0 components
    j0_perp = torch.linalg.vector_norm(torch.stack((j0x, j0y), dim=-1), dim=-1)
    has_j0_perp = j0_perp > 0
    theta_jl0 = torch.atan2(j0_perp, j0z)
    phi_jl0 = torch.atan2(
        torch.where(has_j0_perp, j0y, torch.zeros_like(j0y)),
        torch.where(has_j0_perp, j0x, torch.ones_like(j0x)),
    )

    # Line of sight in L0 frame
    sin_i = torch.sin(iota)
    cos_i = torch.cos(iota)
    source_phase = phi_ref if observer_azimuth is None else phi_ref-observer_azimuth
    sin_pref = torch.sin(source_phase)
    cos_pref = torch.cos(source_phase)

    nx_l0 = sin_i * sin_pref
    ny_l0 = sin_i * cos_pref
    nz_l0 = cos_i

    # Rotate N to J prime frame: N_Jprime = R_y(-theta_jl0) * R_z(-phi_jl0) * N_L0
    cos_tjl = torch.cos(theta_jl0)
    sin_tjl = torch.sin(theta_jl0)
    cos_pjl = torch.cos(phi_jl0)
    sin_pjl = torch.sin(phi_jl0)

    nx_rot_z = nx_l0 * cos_pjl + ny_l0 * sin_pjl
    ny_rot_z = -nx_l0 * sin_pjl + ny_l0 * cos_pjl
    nz_rot_z = nz_l0

    nx_jp = nx_rot_z * cos_tjl - nz_rot_z * sin_tjl
    ny_jp = ny_rot_z
    nz_jp = nx_rot_z * sin_tjl + nz_rot_z * cos_tjl

    has_n_perp = (nx_jp != 0) | (ny_jp != 0)
    kappa = torch.atan2(torch.where(has_n_perp, ny_jp, torch.zeros_like(ny_jp)),
                       torch.where(has_n_perp, nx_jp, torch.ones_like(nx_jp)))

    # C16 and the independent Pv2 observer-to-J construction. Schmidt
    # arXiv:1703.01076 Eq. (31) maps source spins into the Pv2 observer frame.
    alpha_0_xp = PI - kappa
    epsilon_0_xp = phi_jl0 - PI
    jx_n = -cos_i*(sin_pref*j0x+cos_pref*j0y)+sin_i*j0z
    jy_n = cos_pref*j0x-sin_pref*j0y
    j_perp_n_sq = jx_n*jx_n+jy_n*jy_n
    j_norm = torch.linalg.vector_norm(torch.stack((j0x,j0y,j0z),dim=-1),dim=-1)
    # Eq. (39), multiplied by the common positive |J|*|J_perp,N|.
    # Expanding the x component cancels its large Jz^2 terms analytically,
    # preserving the small-cone limit without subtracting nearly equal terms.
    lx_j_scaled = cos_i*(j0x*j0x+j0y*j0y)-sin_i*j0z*(sin_pref*j0x+cos_pref*j0y)
    ly_j_scaled = sin_i*j_norm*jy_n
    has_l_perp_j = has_j0_perp & (j_perp_n_sq > 0)
    alpha_0_pv2 = torch.atan2(
        torch.where(has_l_perp_j, ly_j_scaled, torch.sin(alpha_0_xp)),
        torch.where(has_l_perp_j, lx_j_scaled, torch.cos(alpha_0_xp)),
    )
    alpha_0 = alpha_0_pv2 if convention in (1,5) else alpha_0_xp
    epsilon_0 = epsilon_0_xp if convention in (1,6) else torch.zeros_like(phi_jl0)
    carrier_phase = (phi_jl0 if convention == 0 else
                     torch.zeros_like(phi_ref) if convention == 1 else phi_ref)

    # Dynamics are reference-relative in options 0/1/6 and raw in 5/7.
    alpha_msa_final = alpha_msa_dyn+alpha_0
    gamma_msa_final = gamma_msa_dyn-epsilon_0

    # The same Table IV offsets apply to either analytical prescription.
    alpha_nnlo_final = alpha_nnlo_dyn + alpha_0
    epsilon_nnlo_final = epsilon_nnlo_dyn + epsilon_0
    gamma_nnlo_final = -epsilon_nnlo_final

    # Select MSA vs NNLO fallback
    fb_mask = fallback_to_nnlo.unsqueeze(-1) if (v.dim() > fallback_to_nnlo.dim()) else fallback_to_nnlo
    alpha = torch.where(fb_mask, alpha_nnlo_final, alpha_msa_final)
    beta = torch.where(fb_mask, beta_nnlo, beta_msa)
    gamma = torch.where(fb_mask, gamma_nnlo_final, gamma_msa_final)

    # Polarization basis rotation zeta and theta_jn
    # J0 . N_L0 / |J0|
    theta_jn = torch.atan2(torch.linalg.vector_norm(torch.stack((nx_jp, ny_jp), dim=-1), dim=-1), nz_jp)

    # Polarization vector P_L0 = (-cos(iota)*sin(phi_ref), -cos(iota)*cos(phi_ref), sin(iota))
    px_l0 = -cos_i * sin_pref
    py_l0 = -cos_i * cos_pref
    pz_l0 = sin_i

    px_rot_z = px_l0 * cos_pjl + py_l0 * sin_pjl
    py_rot_z = -px_l0 * sin_pjl + py_l0 * cos_pjl
    pz_rot_z = pz_l0

    px_jp = px_rot_z * cos_tjl - pz_rot_z * sin_tjl
    py_jp = py_rot_z
    pz_jp = px_rot_z * sin_tjl + pz_rot_z * cos_tjl

    # Rotate by R_z(-kappa):
    cos_k = torch.cos(kappa)
    sin_k = torch.sin(kappa)
    xn_x = px_jp * cos_k + py_jp * sin_k
    xn_y = -px_jp * sin_k + py_jp * cos_k
    xn_z = pz_jp

    # x_N_prime = (cos(theta_jn), 0, -sin(theta_jn)), y_N_prime = (0, 1, 0)
    # zeta = atan2(x_N . y_N_prime, x_N . x_N_prime)
    cos_tjn_b = torch.cos(theta_jn)
    sin_tjn_b = torch.sin(theta_jn)
    xn_dot_xp = xn_x * cos_tjn_b - xn_z * sin_tjn_b
    xn_dot_yp = xn_y

    zeta = torch.atan2(xn_dot_yp, xn_dot_xp)

    # Keep beta's geometric pi branch. Exact convention-1 aligned carrier
    # selection belongs to the waveform wrappers, including batched rows.
    if return_carrier_phase:
        return alpha, beta, gamma, theta_jn, zeta, carrier_phase
    return alpha, beta, gamma, theta_jn, zeta

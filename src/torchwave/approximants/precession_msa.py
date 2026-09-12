"""Multiple Scale Analysis (MSA) Spin-Precession Prescription (Variant 223 & NNLO).

Strict Clean-Room Implementation based exclusively on published literature:
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503), Sec. IV & Appendix B
- Chatziioannou et al. (2017), Phys. Rev. D 95, 104004 (arXiv:1703.03967)
- Chatziioannou et al. (2013), Phys. Rev. D 88, 063011 (arXiv:1307.4418)

Clean-Room Boundary Compliance:
- Mathematical equations implemented solely from cited literature.
- No LAL / LALSimulation / PyCBC source inspection, porting, or coefficient copying.
- Native PyTorch tensor implementation with full autograd differentiability and batching.
"""

from typing import Dict, Optional, Tuple, Union
import math
import torch

from torchwave.constants import PI, MSUN_SEC
from torchwave.conversions import as_tensor
from torchwave.approximants.imrphenomp import IMRPhenomP


def compute_l_3pn(
    v: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    s_ell: torch.Tensor,
    sigma_ell: torch.Tensor,
) -> torch.Tensor:
    """Compute 3PN spinning orbital angular momentum L(v) per Pratten et al. (2021) Eq. 18 / Appendix C.

    Formula:
        L(v) = (eta / v) * [ 1 + L1 * v^2 + L_15_SO * v^3 + L2 * v^4 + L_25_SO * v^5 + L3 * v^6 + L_35_SO * v^7 ]
    """
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v6 * v

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
    l_35_so = (
        (-405.0 / 16.0 + (1101.0 / 16.0) * eta - (29.0 / 16.0) * (eta * eta)) * s_ell
        + delta * (-81.0 / 6.0 + (117.0 / 4.0) * eta - (15.0 / 16.0) * (eta * eta)) * sigma_ell
    )

    poly = 1.0 + l1 * v2 + l_15_so * v3 + l2 * v4 + l_25_so * v5 + l3 * v6 + l_35_so * v7
    return (eta / v) * poly


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

    # Physical validity gate
    disc = (safe_p_third ** 3) - (qc_dep / 2.0) ** 2
    is_valid = (
        (disc >= -1e-12)
        & (sp_sq >= sm_sq)
        & (sm_sq >= -1e-12)
        & ((sp_sq - sm_sq) >= 1e-9)
        & ((1.0 / q_c - 1.0) >= 1e-4)
    )

    return s3_sq, sm_sq, sp_sq, is_valid


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
    prec_version: int = 223,
    convention: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate full precessing Euler angles (alpha, beta, gamma), frame angle theta_JN, and polarization rotation zeta.

    Returns:
        alpha: Azimuthal precession angle (same shape as v)
        beta: Opening angle of precession cone (same shape as v)
        gamma: Coprecessing angle gamma = -epsilon (same shape as v)
        theta_jn: Line of sight angle in J frame (scalar or batched)
        zeta: Polarization basis rotation angle (scalar or batched)
    """
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

    s1_sq = s1x_g * s1x_g + s1y_g * s1y_g + s1z_g * s1z_g
    s2_sq = s2x_g * s2x_g + s2y_g * s2y_g + s2z_g * s2z_g
    in_plane_sq = s1x * s1x + s1y * s1y + s2x * s2x + s2y * s2y
    is_aligned = in_plane_sq < 1e-12

    s_ell = s1z_g + s2z_g
    sigma_ell = m2_norm * s1z - m1_norm * s2z

    # Angular momentum at v0
    use_3pn_l = prec_version in (103, 104, 223)
    if use_3pn_l:
        l0 = compute_l_3pn(v0, eta, delta, s_ell, sigma_ell)
        l_dyn = compute_l_3pn(v, eta, delta, s_ell, sigma_ell)
    elif prec_version == 102:
        l0 = IMRPhenomP.orbital_angular_momentum(v0, eta)
        l_dyn = IMRPhenomP.orbital_angular_momentum(v, eta)
    else:
        l0 = eta / v0
        l_dyn = eta / v

    j0x = s1x_g + s2x_g
    j0y = s1y_g + s2y_g
    j0z = l0 + s1z_g + s2z_g
    j0_sq = j0x * j0x + j0y * j0y + j0z * j0z
    j0_mag = torch.sqrt(torch.clamp(j0_sq, min=1e-24))

    xi = (1.0 + q_c) * s1z_g + (1.0 + 1.0 / q_c) * s2z_g

    # Solve depressed cubic roots
    s3_sq, sm_sq, sp_sq, roots_valid = solve_spin_roots(
        l0=l0,
        j0_sq=j0_sq,
        s1_sq=s1_sq,
        s2_sq=s2_sq,
        xi=xi,
        q_c=torch.clamp(q_c, max=1.0 - 1e-4),
        eta=eta,
        delta=torch.clamp(delta, min=1e-5),
    )

    use_msa = prec_version in (220, 223)
    fallback_to_nnlo = (~roots_valid) if use_msa else torch.ones_like(roots_valid, dtype=torch.bool)

    # MSA computation path (guarded against singular denominators)
    q_safe = torch.clamp(q_c, max=1.0 - 1e-4)
    delta_safe = torch.clamp(delta, min=1e-4)

    sav_sq = 0.5 * (sp_sq + sm_sq)
    c1 = 0.5 * v0 * (j0_sq - l0 * l0 - sav_sq)
    j_dyn = torch.sqrt(torch.clamp(l_dyn * l_dyn + 2.0 * c1 / v + sav_sq, min=1e-16))

    cos_beta_msa = torch.clamp((j_dyn * j_dyn + l_dyn * l_dyn - sav_sq) / (2.0 * j_dyn * l_dyn), -1.0 + 1e-12, 1.0 - 1e-12)
    beta_msa = torch.acos(cos_beta_msa)

    # Dissipative coefficients
    g0 = 5.0 / (96.0 * eta)
    a0 = 96.0 * eta / 5.0
    a2 = -743.0 / 336.0 - 2.75 * eta

    denom_spin = eta * (1.0 - q_safe * q_safe)
    s1_ell_avg = (c1 * (1.0 + q_safe) - q_safe * eta * xi) / denom_spin
    s2_ell_avg = -q_safe * (c1 * (1.0 + q_safe) - eta * xi) / denom_spin

    beta3 = (113.0 / 12.0 + 25.0 / (4.0 * q_safe)) * s1_ell_avg + (113.0 / 12.0 + 25.0 * q_safe / 4.0) * s2_ell_avg
    a3 = 4.0 * PI - beta3
    beta5 = (
        ((31319.0 / 1008.0 - (1159.0 / 24.0) * eta) + (1.0 / q_safe) * (809.0 / 84.0 - (281.0 / 8.0) * eta)) * s1_ell_avg
        + ((31319.0 / 1008.0 - (1159.0 / 24.0) * eta) + q_safe * (809.0 / 84.0 - (281.0 / 8.0) * eta)) * s2_ell_avg
    )
    sigma4 = (
        (247.0 / 48.0 * (sav_sq / 2.0 - (s1_sq + s2_sq) / 2.0) - (721.0 / 48.0) * s1_ell_avg * s2_ell_avg) / eta
        + (233.0 / 96.0 * s1_sq - (719.0 / 96.0) * (s1_ell_avg ** 2)) / (m1_norm ** 2)
        + (233.0 / 96.0 * s2_sq - (719.0 / 96.0) * (s2_ell_avg ** 2)) / (m2_norm ** 2)
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

    a1_om = 0.5 + 0.75 * eta
    a2_om = -3.0 / (4.0 * eta) * xi
    ad = (-3.0 * (s1_sq - s2_sq) * eta * delta_safe + 3.0 * (c1 / eta) * (c1 - 2.0 * xi * (eta ** 2))) / (4.0 * sqrt_cp_cm)
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

    sav = torch.sqrt(torch.clamp(sav_sq, min=1e-20))

    def evaluate_phi_z(v_grid: torch.Tensor, j_grid: torch.Tensor, l_grid: torch.Tensor):
        l1_log = torch.log(torch.clamp(c1 + j_grid * eta + l_grid * eta, min=1e-20))
        l2_log = torch.log(torch.clamp(c1 + j_grid * sav * v_grid + sav_sq * v_grid, min=1e-20))

        phi0 = (j_grid / (eta ** 4)) * (
            0.5 * (c1 ** 2)
            - c1 * (eta ** 2) / (6.0 * v_grid)
            - (sav_sq * (eta ** 2)) / 3.0
            - (eta ** 4) / (3.0 * (v_grid ** 2))
        ) - (c1 / (2.0 * eta)) * ((c1 ** 2) / (eta ** 4) - sav_sq / (eta ** 2)) * l1_log

        phi1 = -(j_grid / (2.0 * (eta ** 2))) * (c1 + eta * l_grid) + (1.0 / (2.0 * (eta ** 3))) * (c1 - (eta ** 2) * sav_sq) * l1_log
        phi2 = -j_grid + sav * l2_log - (c1 / eta) * l1_log
        phi3 = j_grid * v_grid - eta * l1_log + (c1 / sav) * l2_log
        phi4 = (j_grid / (2.0 * sav_sq)) * v_grid * (c1 + v_grid * sav_sq) - (1.0 / (2.0 * (sav ** 3))) * (c1 * c1 - (eta ** 2) * sav_sq) * l2_log
        phi5 = -j_grid * v_grid * (
            (c1 ** 2) / (2.0 * (sav ** 4))
            - (c1 * v_grid) / (6.0 * sav_sq)
            - (v_grid ** 2) / 3.0
            - (eta ** 2) / (3.0 * sav_sq)
        ) + (c1 / (2.0 * (sav ** 5))) * (c1 * c1 - (eta ** 2) * sav_sq) * l2_log

        return (
            avg_om_z0 * phi0
            + avg_om_z1 * phi1
            + avg_om_z2 * phi2
            + avg_om_z3 * phi3
            + avg_om_z4 * phi4
            + avg_om_z5 * phi5
        )

    alpha_msa_dyn = evaluate_phi_z(v, j_dyn, l_dyn)
    alpha_msa_0 = evaluate_phi_z(v0, j0_mag, l0)

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

    def evaluate_gamma(v_grid: torch.Tensor):
        return eta * (v_grid ** (-3.0)) * (
            avg_om_zeta0
            + avg_om_zeta1 * v_grid
            + avg_om_zeta2 * (v_grid ** 2)
            + avg_om_zeta3 * (v_grid ** 3)
            + avg_om_zeta4 * (v_grid ** 4)
            + avg_om_zeta5 * (v_grid ** 5)
        )

    gamma_msa_dyn = evaluate_gamma(v)
    gamma_msa_0 = evaluate_gamma(v0)
    epsilon_msa_dyn = -gamma_msa_dyn
    epsilon_msa_0 = -gamma_msa_0

    # NNLO Fallback Computation Path
    a1_coeff = 2.0 + 1.5 / q_pratten
    a2_coeff = 2.0 + 1.5 * q_pratten
    s1_perp = (m1 * m1) * torch.sqrt(torch.clamp(s1x * s1x + s1y * s1y, min=1e-20))
    s2_perp = (m2 * m2) * torch.sqrt(torch.clamp(s2x * s2x + s2y * s2y, min=1e-20))
    s_p = torch.maximum(a1_coeff * s1_perp, a2_coeff * s2_perp)
    chip = s_p / (a1_coeff * m1 * m1)
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
    )
    sp_geom = chip * a1_coeff * (m1_norm * m1_norm)
    sl_geom = (m1_norm * m1_norm) * s1z + (m2_norm * m2_norm) * s2z
    l_plus_sl = l_dyn + sl_geom
    cos_beta_nnlo = torch.clamp(
        l_plus_sl / torch.sqrt(l_plus_sl * l_plus_sl + sp_geom * sp_geom + 1e-24),
        -1.0 + 1e-12,
        1.0 - 1e-12,
    )
    beta_nnlo = torch.acos(cos_beta_nnlo)

    # Frame geometry at reference frequency f_ref
    # J0 components
    has_j0_perp = (j0x * j0x + j0y * j0y) > 1e-12
    theta_jl0 = torch.where(
        has_j0_perp,
        torch.acos(torch.clamp(j0z / j0_mag, -1.0 + 1e-12, 1.0 - 1e-12)),
        torch.zeros_like(j0z),
    )
    phi_jl0 = torch.where(has_j0_perp, torch.atan2(j0y, j0x), torch.zeros_like(j0y))

    # Line of sight in L0 frame
    sin_i = torch.sin(iota)
    cos_i = torch.cos(iota)
    sin_pref = torch.sin(phi_ref)
    cos_pref = torch.cos(phi_ref)

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

    kappa = torch.atan2(ny_jp, nx_jp)

    # Offsets (Table 4 Convention 1 Default)
    alpha_0_xp = PI - kappa
    epsilon_0_xp = phi_jl0 - PI
    alpha_0_pv2 = PI - phi_jl0

    if convention == 0:
        alpha_0 = torch.zeros_like(phi_jl0)
        epsilon_0 = torch.zeros_like(phi_jl0)
    elif convention == 5:
        alpha_0 = alpha_0_xp
        epsilon_0 = epsilon_0_xp
    elif convention == 6:
        alpha_0 = alpha_0_pv2
        epsilon_0 = torch.zeros_like(phi_jl0)
    elif convention == 7:
        alpha_0 = alpha_0_xp
        epsilon_0 = torch.zeros_like(phi_jl0)
    else:  # convention == 1 (default)
        alpha_0 = alpha_0_xp
        epsilon_0 = epsilon_0_xp

    # Offsets for MSA
    alpha_offset = alpha_msa_0 - alpha_0
    epsilon_offset = epsilon_msa_0 - epsilon_0

    alpha_msa_final = alpha_msa_dyn - alpha_offset
    epsilon_msa_final = epsilon_msa_dyn - epsilon_offset
    gamma_msa_final = -epsilon_msa_final

    # For NNLO, alpha_dyn is relative to v0, so add alpha_0 directly
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
    cos_tjn = torch.clamp((j0x * nx_l0 + j0y * ny_l0 + j0z * nz_l0) / j0_mag, -1.0 + 1e-12, 1.0 - 1e-12)
    theta_jn = torch.acos(cos_tjn)

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

    if convention in (1, 5, 7):
        zeta = torch.atan2(xn_dot_yp, xn_dot_xp)
    else:
        zeta = torch.zeros_like(theta_jn)
        theta_jn = iota

    # Exact non-precessing / aligned-spin limit collapse
    is_al_u = is_aligned.unsqueeze(-1) if (v.dim() > is_aligned.dim()) else is_aligned
    alpha = torch.where(is_al_u, torch.zeros_like(alpha), alpha)
    beta = torch.where(is_al_u, torch.zeros_like(beta), beta)
    gamma = torch.where(is_al_u, torch.zeros_like(gamma), gamma)
    theta_jn = torch.where(is_aligned, iota, theta_jn)
    zeta = torch.where(is_aligned, torch.zeros_like(zeta), zeta)

    return alpha, beta, gamma, theta_jn, zeta

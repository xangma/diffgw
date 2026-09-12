"""IMRPhenomNSBH Frequency-Domain Gravitational Waveform Approximant.

Strict Clean-Room Implementation based exclusively on published literature:
- Thompson et al. (2020), Phys. Rev. D 101, 124059 (arXiv:2002.08383)
- Pannarale et al. (2015), Phys. Rev. D 92, 084050 (arXiv:1509.00512)
- Pannarale et al. (2015), Phys. Rev. D 92, 081504 (arXiv:1509.06209)
- Zappa et al. (2019), Phys. Rev. Lett. 123, 041102 (arXiv:1903.11622)
- Jiménez-Forteza et al. (2017), Phys. Rev. D 95, 064024 (arXiv:1611.00332)
- London et al. (2018), Phys. Rev. Lett. 120, 161102 (arXiv:1708.00404)
- Santamaria et al. (2010), Phys. Rev. D 82, 064016 (arXiv:1005.3306)
- Dietrich et al. (2019), Phys. Rev. D 100, 044003 (arXiv:1905.06011)
"""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from torchwave.conversions import (
    as_tensor,
    component_masses_to_eta,
    component_masses_to_delta,
    kerr_isco_radius,
    kerr_isco_frequency,
    spin_combinations,
    tidal_combinations,
)
from torchwave.approximants.imrphenomd import IMRPhenomD, TaylorF2
from torchwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
from torchwave.approximants.nrtidal_unrounded_fits import NRTIDAL_V2_PHASE_COEFFS
from torchwave.approximants.imrphenomd_nrtidal import (
    N1_V2,
    N32_V2,
    N2_V2,
    N52_V2,
    N3_V2,
    D1_V2,
    D32_V2,
    D2_V2,
    _nrtidal_self_spin_components,
)


def ns_compactness_from_lambda(lambda_ns: torch.Tensor) -> torch.Tensor:
    """Compute neutron star compactness C = M_NS / R_NS from tidal deformability Lambda.

    Thompson et al. (2020) Eq. 20 (Appendix B, Eq. 791):
        C(Lambda) = a0 + a1 * ln(Lambda) + a2 * ln(Lambda)^2, Lambda > 1
        C(Lambda) = 0.5 + (3a0 - a1 - 1.5)*Lambda^2 + (a1 - 2a0 + 1)*Lambda^3, Lambda <= 1
    with a0 = 0.360, a1 = -0.0355, a2 = 0.000705.
    At Lambda = 0, C(0) = 0.5 (Schwarzschild black hole limit).
    """
    a0 = 0.360
    a1 = -0.0355
    a2 = 0.000705
    b2 = 3.0 * a0 - a1 - 1.5
    b3 = a1 - 2.0 * a0 + 1.0

    lam_clamped = torch.clamp(lambda_ns, min=0.0)
    lam_safe = torch.clamp(lam_clamped, min=1.0)
    ln_lam = torch.log(lam_safe)
    c_gt1 = a0 + a1 * ln_lam + a2 * (ln_lam ** 2.0)
    c_le1 = 0.5 + b2 * (lam_clamped ** 2.0) + b3 * (lam_clamped ** 3.0)

    c = torch.where(lambda_ns > 1.0, c_gt1, c_le1)
    return torch.clamp(c, min=0.01, max=0.5)


def solve_xi_tide(
    q: torch.Tensor,
    chi: torch.Tensor,
    c: torch.Tensor,
    n_iter: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve for relativistic disruption root xi_tide via Newton-Raphson.

    Polynomial from Thompson et al. (2020) Eqs. 7-9:
        P(xi) = xi^5 - 3*mu*xi^4 + 2*chi*sqrt(mu^3 * xi^7) - 3*q*xi^2 + 6*q*mu*xi - 3*q*mu^2 * chi^2 = 0
        P'(xi) = 5*xi^4 - 12*mu*xi^3 + 7*chi*sqrt(mu^3 * xi^5) - 6*q*xi + 6*q*mu
    with mu = q * C.

    Initial guess: xi_0 = (3q)^(1/3) + 3*mu, ensuring convergence to the largest positive root.

    Returns:
        xi_tide: Root tensor
        res: Final polynomial residual P(xi_tide)
    """
    mu = q * c
    # Starting guess positioned in the basin of the rightmost positive root
    xi = torch.pow(3.0 * q, 1.0 / 3.0) + 3.0 * mu

    for _ in range(n_iter):
        xi_safe = torch.clamp(xi, min=1e-8)
        sqrt_p = torch.sqrt(torch.clamp((mu ** 3.0) * (xi_safe ** 7.0), min=0.0))
        sqrt_dp = torch.sqrt(torch.clamp((mu ** 3.0) * (xi_safe ** 5.0), min=0.0))
        p = (
            xi ** 5.0
            - 3.0 * mu * (xi ** 4.0)
            + 2.0 * chi * sqrt_p
            - 3.0 * q * (xi ** 2.0)
            + 6.0 * q * mu * xi
            - 3.0 * q * (mu ** 2.0) * (chi ** 2.0)
        )
        dp = (
            5.0 * (xi ** 4.0)
            - 12.0 * mu * (xi ** 3.0)
            + 7.0 * chi * sqrt_dp
            - 6.0 * q * xi
            + 6.0 * q * mu
        )
        delta_step = p / torch.where(dp.abs() > 1e-12, dp, torch.full_like(dp, 1e-12))
        xi = xi - delta_step

    xi_safe = torch.clamp(xi, min=1e-8)
    sqrt_p = torch.sqrt(torch.clamp((mu ** 3.0) * (xi_safe ** 7.0), min=0.0))
    res = (
        xi ** 5.0
        - 3.0 * mu * (xi ** 4.0)
        + 2.0 * chi * sqrt_p
        - 3.0 * q * (xi ** 2.0)
        + 6.0 * q * mu * xi
        - 3.0 * q * (mu ** 2.0) * (chi ** 2.0)
    )
    return xi, res


def bbh_remnant_uib2016(
    mass1: torch.Tensor,
    mass2: torch.Tensor,
    chi1: torch.Tensor,
    chi2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute BBH remnant mass ratio M_f/M and spin chi_f using Jiménez-Forteza et al. (2017) v2 (FinalStateUIB2016)."""
    m = mass1 + mass2
    eta = component_masses_to_eta(mass1, mass2)
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2

    msq = m * m
    m1sq = mass1 * mass1
    m2sq = mass2 * mass2
    s1 = chi1 * m1sq / msq
    s2 = chi2 * m2sq / msq
    stot = s1 + s2
    shat = (chi1 * m1sq + chi2 * m2sq) / (m1sq + m2sq)
    shat2 = shat * shat
    shat3 = shat2 * shat

    chidiff = chi1 - chi2
    chidiff2 = chidiff * chidiff

    sqrt2 = math.sqrt(2.0)
    sqrt3 = math.sqrt(3.0)
    sqrt1m4eta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))

    # Erad v2 coefficients (Jiménez-Forteza et al. 2017 Table VII-X)
    b10 = 0.346
    b20 = 0.211
    b30 = 0.128
    b50 = -0.212
    a2 = 0.5609904135313374
    a3 = -0.84667563764404
    a4 = 3.145145224278187
    b1 = -0.2091189048177395
    b2 = -0.19709136361080587
    b3 = -0.1588185739358418
    b5 = 2.9852925538232014
    f20 = 4.271313308472851
    f30 = 31.08987570280556
    f50 = 1.5673498395263061
    f10 = 1.8083565298668276
    f21 = 0.0
    d10 = -0.09803730445895877
    d11 = -3.2283713377939134
    d20 = 0.01118530335431078
    d30 = -0.01978238971523653
    d31 = -4.91667749015812
    f11 = 15.738082204419655
    f31 = -243.6299258830685
    f51 = -0.5808669012986468

    num_erad = (
        ((1.0 - 2.0 / 3.0 * sqrt2) * eta + a2 * eta2 + a3 * eta3 + a4 * eta4)
        * (
            1.0
            + b10 * b1 * shat * (f10 + f11 * eta + (16.0 - 16.0 * f10 - 4.0 * f11) * eta2)
            + b20 * b2 * shat2 * (f20 + f21 * eta + (16.0 - 16.0 * f20 - 4.0 * f21) * eta2)
            + b30 * b3 * shat3 * (f30 + f31 * eta + (16.0 - 16.0 * f30 - 4.0 * f31) * eta2)
        )
    )
    den_erad = 1.0 + b50 * b5 * shat * (f50 + f51 * eta + (16.0 - 16.0 * f50 - 4.0 * f51) * eta2)
    erad = (
        num_erad / den_erad
        + d10 * sqrt1m4eta * eta2 * (1.0 + d11 * eta) * chidiff
        + d30 * shat * sqrt1m4eta * eta * (1.0 + d31 * eta) * chidiff
        + d20 * eta3 * chidiff2
    )
    mf_over_m_bbh = 1.0 - erad

    # Final spin v2 coefficients (Jiménez-Forteza et al. 2017 Table I-IV)
    a20_s = 5.24
    a30_s = 1.3
    a50_s = 2.88
    b10_s = -0.194
    b20_s = 0.0851
    b30_s = 0.00954
    b50_s = -0.579
    a2_s = 3.8326341618708577
    a3_s = -9.487364155598392
    a5_s = 2.5134875145648374
    b1_s = 1.0009563702914628
    b2_s = 0.7877509372255369
    b3_s = 0.6540138407185817
    b5_s = 0.8396665722805308
    f21_s = 8.77367320110712
    f31_s = 22.830033250479833
    f50_s = 1.8804718791591157
    f11_s = 4.409160174224525
    f52_s = 0.0
    d10_s = 0.3223660562764661
    d11_s = 9.332575956437443
    d20_s = -0.059808322561702126
    d30_s = 2.3170397514509933
    d31_s = -3.2624649875884852
    f12_s = 0.5118334706832706
    f22_s = -32.060648277652994
    f32_s = -153.83722669033995
    f51_s = -4.770246856212403

    num_spin = (
        b10_s * b1_s * shat * (f11_s * eta + f12_s * eta2 + (64.0 - 16.0 * f11_s - 4.0 * f12_s) * eta3)
        + b20_s * b2_s * shat2 * (f21_s * eta + f22_s * eta2 + (64.0 - 16.0 * f21_s - 4.0 * f22_s) * eta3)
        + b30_s * b3_s * shat3 * (f31_s * eta + f32_s * eta2 + (64.0 - 16.0 * f31_s - 4.0 * f32_s) * eta3)
    )
    den_spin = 1.0 + b50_s * b5_s * shat * (
        f50_s + f51_s * eta + f52_s * eta2 + (64.0 - 64.0 * f50_s - 16.0 * f51_s - 4.0 * f52_s) * eta3
    )
    lorb = (
        (2.0 * sqrt3 * eta + a20_s * a2_s * eta2 + a30_s * a3_s * eta3) / (1.0 + a50_s * a5_s * eta)
        + num_spin / den_spin
        + d10_s * sqrt1m4eta * eta2 * (1.0 + d11_s * eta) * chidiff
        + d30_s * shat * sqrt1m4eta * eta3 * (1.0 + d31_s * eta) * chidiff
        + d20_s * eta3 * chidiff2
    )
    chif_bbh = lorb + stot

    return mf_over_m_bbh, chif_bbh


def zappa_remnant_properties(
    mass1: torch.Tensor,
    mass2: torch.Tensor,
    chi1: torch.Tensor,
    chi2: torch.Tensor,
    lambda2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute NSBH remnant mass M_f and remnant spin chi_f (Zappa et al. 2019 Eq. 6 & Table IV)."""
    m = mass1 + mass2
    eta = component_masses_to_eta(mass1, mass2)
    delta = component_masses_to_delta(mass1, mass2)
    chi = ((1.0 + delta) / 2.0) * chi1 + ((1.0 - delta) / 2.0) * chi2

    mf_bbh_ratio, chif_bbh = bbh_remnant_uib2016(mass1, mass2, chi1, chi2)

    # Zappa Table IV fits for Mf/M (X_bullet)
    p1_m = (-1.83e-3 * chi + 2.39e-3) * eta + (4.29e-3 * chi + 9.80e-3) * (eta ** 2.0)
    p2_m = (2.34e-7 * chi - 8.28e-7) * eta + (-1.64e-6 * chi + 8.08e-6) * (eta ** 2.0)
    p3_m = (-2.01e-2 * chi + 1.32e-1) * eta + (6.51e-2 * chi - 1.43e-1) * (eta ** 2.0)

    mf_ratio = mf_bbh_ratio * (1.0 + p1_m * lambda2 + p2_m * (lambda2 ** 2.0)) / ((1.0 + (p3_m ** 2.0) * lambda2) ** 2.0)
    mf = torch.clamp(m * mf_ratio, max=m)

    # Zappa Table IV fits for chif (a_bullet)
    p1_s = (-5.44e-3 * chi + 7.91e-3) * eta + (2.33e-2 * chi + 2.48e-2) * (eta ** 2.0)
    p2_s = (-8.57e-7 * chi - 2.82e-6) * eta + (6.61e-6 * chi + 4.29e-5) * (eta ** 2.0)
    p3_s = (-3.04e-2 * chi + 2.55e-1) * eta + (1.48e-1 * chi - 4.28e-1) * (eta ** 2.0)

    chif = chif_bbh * (1.0 + p1_s * lambda2 + p2_s * (lambda2 ** 2.0)) / ((1.0 + (p3_s ** 2.0) * lambda2) ** 2.0)
    chif = torch.clamp(chif, -0.999, 0.999)

    return mf, chif


def london_ringdown_properties(
    m_final: torch.Tensor,
    chi_final: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Kerr ringdown frequency f_RD and quality factor Q (London et al. 2018 / Thompson et al. 2020)."""
    kappa = torch.sqrt(torch.clamp(torch.log(2.0 - chi_final) / math.log(3.0), min=0.0))
    re_w = (
        1.0
        + 1.5578 * math.cos(2.9031) * kappa
        + 1.9510 * math.cos(5.9210) * (kappa ** 2.0)
        + 2.0997 * math.cos(2.7606) * (kappa ** 3.0)
        + 1.4109 * math.cos(5.9143) * (kappa ** 4.0)
        + 0.4106 * math.cos(2.7952) * (kappa ** 5.0)
    )
    im_w = (
        1.5578 * math.sin(2.9031) * kappa
        + 1.9510 * math.sin(5.9210) * (kappa ** 2.0)
        + 2.0997 * math.sin(2.7606) * (kappa ** 3.0)
        + 1.4109 * math.sin(5.9143) * (kappa ** 4.0)
        + 0.4106 * math.sin(2.7952) * (kappa ** 5.0)
    )
    mf_sec = m_final * MSUN_SEC
    f_rd = re_w / (2.0 * PI * mf_sec)
    q_factor = re_w / (2.0 * torch.clamp(im_w.abs(), min=1e-8))
    return f_rd, q_factor


def nsbh_frequencies(
    mass1: torch.Tensor,
    mass2: torch.Tensor,
    chi1: torch.Tensor,
    lambda2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute tidal disruption frequency f_tide, Kerr ISCO frequency f_isco, ringdown f_rd, and compactness C.

    Thompson et al. (2020), Pannarale et al. (2015), Zappa et al. (2019).

    Returns:
        (f_tide, f_isco, f_rd, compactness) in Hz
    """
    q = mass1 / torch.clamp(mass2, min=1e-6)
    m1_sec = mass1 * MSUN_SEC
    m2_sec = mass2 * MSUN_SEC

    # 1. NS compactness
    c = ns_compactness_from_lambda(lambda2)

    # 2. Relativistic disruption root xi_tide & disruption frequency
    xi, _ = solve_xi_tide(q, chi1, c)
    r_tide = xi * m2_sec * torch.clamp(1.0 - 2.0 * c, min=0.0) / torch.clamp(c, min=1e-6)
    denom = PI * (chi1 * m1_sec + torch.sqrt(torch.clamp((r_tide ** 3.0) / m1_sec, min=0.0) + 1e-30))
    f_tide = 1.0 / denom

    # 3. Kerr ISCO frequency
    f_isco = kerr_isco_frequency(mass1, chi1)

    # 4. Ringdown frequency of remnant BH via Zappa et al. (2019) fits
    mf, chif = zappa_remnant_properties(mass1, mass2, chi1, torch.zeros_like(chi1), lambda2)
    f_rd, _ = london_ringdown_properties(mf, chif)

    return f_tide, f_isco, f_rd, c


class IMRPhenomNSBH(nn.Module):
    """IMRPhenomNSBH Aligned-Spin Frequency-Domain Waveform Approximant for NS-BH Mergers.

    Clean-room implementation reconstructing Thompson et al. (2020), Phys. Rev. D 101, 124059.
    """

    def __init__(self):
        super().__init__()
        self.bbh_model = IMRPhenomD()

    def forward(
        self,
        f: torch.Tensor,
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        chi1: Union[float, torch.Tensor] = 0.0,
        chi2: Union[float, torch.Tensor] = 0.0,
        lambda2: Union[float, torch.Tensor] = 0.0,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
        phasing_convention: Optional[str] = None,
        reference_frame: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate frequency-domain polarizations (h_plus, h_cross) for NSBH coalescences.

        Args:
            f: Frequency tensor in Hz
            mass1: Primary mass (BH) in solar masses
            mass2: Secondary mass (NS) in solar masses
            chi1: Dimensionless aligned spin of BH (-0.99 <= chi1 <= 0.99)
            chi2: Dimensionless aligned spin of NS (typically ~0)
            lambda2: Dimensionless tidal deformability of NS (Lambda >= 0)
            distance: Luminosity distance in Mpc
            inclination: Orbital inclination in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            f_ref: Reference frequency in Hz
        """
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1 = as_tensor(chi1, device=device, dtype=dtype)
        s2 = as_tensor(chi2, device=device, dtype=dtype)
        lam2 = as_tensor(lambda2, device=device, dtype=dtype)
        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_c = as_tensor(phic, device=device, dtype=dtype)
        t_c = as_tensor(tc, device=device, dtype=dtype)

        total_mass = m1 + m2
        q = m1 / torch.clamp(m2, min=1e-6)
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)
        chi = ((1.0 + delta) / 2.0) * s1 + ((1.0 - delta) / 2.0) * s2

        m_sec = total_mass * MSUN_SEC
        m1_sec = m1 * MSUN_SEC
        m2_sec = m2 * MSUN_SEC
        d_sec = (dist * MPC_SI) / C_SI

        is_batched = total_mass.dim() >= 1
        if f.dim() == 1 and is_batched:
            f_eval = f.unsqueeze(0)
        else:
            f_eval = f

        def unsqueeze_param(t: torch.Tensor) -> torch.Tensor:
            if is_batched and f_eval.dim() > t.dim():
                return t.unsqueeze(-1)
            return t

        f_safe = torch.clamp(f_eval, min=1e-8)

        # -------------------------------------------------------------
        # 1. Step 1: NS Compactness C(Lambda)
        # -------------------------------------------------------------
        c = ns_compactness_from_lambda(lam2)

        # -------------------------------------------------------------
        # 2. Step 2: Relativistic Tidal Disruption Root xi_tide & f_tide
        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # 2. Step 2: Relativistic Tidal Disruption Root xi_tide & f_tide
        # -------------------------------------------------------------
        xi_tide, _ = solve_xi_tide(q, s1, c)
        r_tide = xi_tide * m2_sec * torch.clamp(1.0 - 2.0 * c, min=0.0) / torch.clamp(c, min=1e-6)
        denom_ftide = PI * (s1 * m1_sec + torch.sqrt(torch.clamp((r_tide ** 3.0) / m1_sec, min=0.0) + 1e-30))
        f_tide = 1.0 / denom_ftide

        # -------------------------------------------------------------
        # 3. Step 3: Baryonic Torus Remnant Mass Ratio
        # -------------------------------------------------------------
        r_isco = kerr_isco_radius(s1)
        torus_ratio = torch.clamp(
            0.296 * xi_tide * (1.0 - 2.0 * c) - 0.171 * q * c * r_isco,
            min=0.0,
        )
        has_torus = torus_ratio > 0.0

        # -------------------------------------------------------------
        # 4. Step 4: Remnant Mass M_f and Remnant Spin chi_f
        # -------------------------------------------------------------
        mf_bbh_ratio, a_bbh = bbh_remnant_uib2016(m1, m2, s1, s2)
        mf_zappa, a_zappa = zappa_remnant_properties(m1, m2, s1, s2, lam2)

        m_ratio_bbh = 1.0 / mf_bbh_ratio
        omega_r_bbh, omega_i_bbh = evaluate_kerr_qnm_220(a_bbh)
        f_rd_bbh = m_ratio_bbh * omega_r_bbh / (2.0 * PI)
        f_damp_bbh = m_ratio_bbh * torch.abs(omega_i_bbh) / (2.0 * PI)

        m_ratio_zappa = total_mass / mf_zappa
        omega_r_zappa, omega_i_zappa = evaluate_kerr_qnm_220(a_zappa)
        f_rd_zappa = m_ratio_zappa * omega_r_zappa / (2.0 * PI)
        f_damp_zappa = m_ratio_zappa * torch.abs(omega_i_zappa) / (2.0 * PI)

        f_rd_tilde_bbh = f_rd_bbh / m_sec
        is_ftide_ge_frd = f_tide >= f_rd_tilde_bbh

        chi_f = torch.where(is_ftide_ge_frd & (~has_torus), a_bbh, a_zappa)
        f_rd_ph = torch.where(is_ftide_ge_frd & (~has_torus), f_rd_bbh, f_rd_zappa)
        f_damp_ph = torch.where(is_ftide_ge_frd & (~has_torus), f_damp_bbh, f_damp_zappa)

        mf = torch.where(is_ftide_ge_frd & (~has_torus), total_mass * mf_bbh_ratio, mf_zappa)

        # -------------------------------------------------------------
        # 5. Step 5: Kerr Ringdown Frequency f_RD and Quality Factor Q (London 2018)
        # -------------------------------------------------------------
        f_rd, q_factor = london_ringdown_properties(mf, chi_f)

        # -------------------------------------------------------------
        # 6. Step 6: Merger Classification and Joining Parameters
        # -------------------------------------------------------------
        lam_safe = torch.clamp(lam2, min=0.0)
        frd_gt1 = 0.99 * 0.98 * f_rd
        frd_le1 = (1.0 - 0.02 * lam_safe + 0.01 * (lam_safe ** 2.0)) * 0.98 * f_rd
        f_rd_tilde = torch.where(lam2 > 1.0, frd_gt1, frd_le1)

        x_nd = ((f_tide - f_rd_tilde) / f_rd_tilde) ** 2.0 - 0.571505 * c - 0.00508451 * chi
        x_prime_nd = ((f_tide - f_rd_tilde) / f_rd_tilde) ** 2.0 - 0.657424 * c - 0.0259977 * chi
        x_d = torus_ratio + 0.424912 * c + 0.363604 * torch.sqrt(eta) - 0.060559 * chi
        x_prime_d = (
            torus_ratio
            - 0.132754 * c
            + 0.576669 * torch.sqrt(eta)
            - 0.0603749 * chi
            - 0.0601185 * (chi ** 2.0)
            - 0.0729134 * (chi ** 3.0)
        )

        def omega_plus(val: torch.Tensor, f0: Union[float, torch.Tensor], d: Union[float, torch.Tensor]) -> torch.Tensor:
            d_val = as_tensor(d, device=device, dtype=dtype)
            f0_val = as_tensor(f0, device=device, dtype=dtype)
            arg = torch.clamp(4.0 * (val - f0_val) / torch.clamp(d_val.abs(), min=1e-8), -50.0, 50.0)
            return torch.sigmoid(2.0 * arg)

        def omega_minus(val: torch.Tensor, f0: Union[float, torch.Tensor], d: Union[float, torch.Tensor]) -> torch.Tensor:
            d_val = as_tensor(d, device=device, dtype=dtype)
            f0_val = as_tensor(f0, device=device, dtype=dtype)
            arg = torch.clamp(4.0 * (val - f0_val) / torch.clamp(d_val.abs(), min=1e-8), -50.0, 50.0)
            return torch.sigmoid(-2.0 * arg)

        eps_tide_high = omega_plus(x_nd, -0.0796251, 0.0801192)
        eps_tide = torch.where(is_ftide_ge_frd, eps_tide_high, torch.zeros_like(eps_tide_high))

        eps_ins_disrupt = 1.29971 - 1.61724 * x_d
        eps_ins = torch.where(is_ftide_ge_frd & (~has_torus), torch.ones_like(eps_ins_disrupt), eps_ins_disrupt)

        sig_tide_high = omega_minus(x_prime_nd, -0.206465, 0.226844)
        sig_tide_disrupt = 0.137722 - 0.293237 * x_prime_d
        sig_tide_mild_notorus = 0.5 * (sig_tide_high + sig_tide_disrupt)
        sigma_tide = torch.where(
            is_ftide_ge_frd,
            sig_tide_high,
            torch.where(has_torus, sig_tide_disrupt, sig_tide_mild_notorus),
        )

        f0_tilde = torch.where(
            is_ftide_ge_frd,
            torch.where(has_torus, eps_ins * f_rd_tilde, f_rd_tilde),
            torch.where(has_torus, eps_ins * f_tide, ((q - 1.0) * f_rd_tilde + eps_ins * f_tide) / q),
        )
        f1_tilde = torch.where(
            is_ftide_ge_frd,
            torch.where(has_torus, eps_ins * f_rd_tilde, f_rd_tilde),
            torch.where(has_torus, f_tide, ((q - 1.0) * f_rd_tilde + f_tide) / q),
        )
        f2_tilde = f_rd_tilde
        d_win = (0.015 + sigma_tide) * f_rd

        # -------------------------------------------------------------
        # 7. Step 7: Phenomenological Amplitude Assembly (Santamaria Table II)
        # -------------------------------------------------------------
        gamma1 = torch.clamp(
            4.149 * chi - 4.07 * (chi ** 2.0) - 87.52 * eta * chi - 48.97 * eta + 666.5 * (eta ** 2.0),
            min=0.0,
        )
        delta1 = torch.clamp(
            -0.05472 * chi + 0.02094 * (chi ** 2.0) + 0.3554 * eta * chi + 0.1151 * eta + 0.964 * (eta ** 2.0),
            min=0.0,
        )
        delta2 = torch.clamp(
            -1.235 * chi + 0.3423 * (chi ** 2.0) + 6.062 * eta * chi + 5.949 * eta - 10.69 * (eta ** 2.0),
            min=1e-4,
        )

        gamma1_prime = torch.where(
            lam2 > 1.0,
            1.25 * gamma1,
            (1.0 - 0.5 * lam_safe - 0.25 * (lam_safe ** 2.0)) * gamma1,
        )
        delta2_prime_gt1 = 0.81248 * omega_minus((f_tide - f_rd_tilde) / f_rd_tilde, 0.0188092, 0.338737)
        delta2_prime_le1 = delta2 - 2.0 * (delta2 - 0.81248) * lam_safe + (delta2 - 0.81248) * (lam_safe ** 2.0)
        delta2_prime = torch.where(lam2 > 1.0, delta2_prime_gt1, delta2_prime_le1)
        sigma_lorentz = delta2_prime * f_rd / q_factor

        # Broadcast scalar binary parameters to grid dimension
        m_sec_u = unsqueeze_param(m_sec)
        d_sec_u = unsqueeze_param(d_sec)
        eta_u = unsqueeze_param(eta)
        chi_u = unsqueeze_param(chi)

        gamma1_p_u = unsqueeze_param(gamma1_prime)
        delta1_u = unsqueeze_param(delta1)
        sigma_l_u = unsqueeze_param(sigma_lorentz)
        eps_tide_u = unsqueeze_param(eps_tide)
        f_rd_u = unsqueeze_param(f_rd)

        f0_tilde_u = unsqueeze_param(f0_tilde)
        f1_tilde_u = unsqueeze_param(f1_tilde)
        f2_tilde_u = unsqueeze_param(f2_tilde)
        d_win_u = unsqueeze_param(d_win)

        # 3.5PN SPA Inspiral Amplitude (PhenomC)
        v = torch.pow(PI * m_sec_u * f_safe, 1.0 / 3.0)
        v2 = v * v
        v3 = v2 * v
        v4 = v2 * v2
        v5 = v3 * v2
        v6 = v3 * v3
        v7 = v4 * v3
        v10 = v5 * v5

        chisum = 2.0 * chi_u
        chiprod = chi_u * chi_u
        chi2 = chi_u * chi_u
        eta2 = eta_u * eta_u
        gamma_e = 0.577215664901532860606512090082402431

        xdota_n = 64.0 * eta_u / 5.0
        xdota2 = -7.43 / 3.36 - 11.0 * eta_u / 4.0
        xdota3 = 4.0 * PI - 11.3 * chi_u / 1.2 + 19.0 * eta_u * chisum / 6.0
        xdota4 = 3.4103 / 1.8144 + 5.0 * chi2 + eta_u * (13.661 / 2.016 - chiprod / 8.0) + 5.9 * eta2 / 1.8
        xdota5 = (
            -PI * (41.59 / 6.72 + 189.0 * eta_u / 8.0)
            - chi_u * (31.571 / 1.008 - 116.5 * eta_u / 2.4)
            + chisum * (21.863 * eta_u / 1.008 - 79.0 * eta2 / 6.0)
            - 3.0 * chi_u * chi2 / 4.0
            + 9.0 * eta_u * chi_u * chiprod / 4.0
        )
        xdota6 = (
            164.47322263 / 1.39708800
            - 17.12 * gamma_e / 1.05
            + 16.0 * (PI ** 2) / 3.0
            - 8.56 * math.log(16.0) / 1.05
            + eta_u * (45.1 * (PI ** 2) / 4.8 - 561.98689 / 2.17728)
            + 5.41 * eta2 / 8.96
            - 5.605 * eta_u * eta2 / 2.592
            - 80.0 * PI * chi_u / 3.0
            + eta_u * chisum * (20.0 * PI / 3.0 - 113.5 * chi_u / 3.6)
            + chi2 * (64.153 / 1.008 - 45.7 * eta_u / 3.6)
            - chiprod * (7.87 * eta_u / 1.44 - 30.37 * eta2 / 1.44)
        )
        xdota6log = -856.0 / 105.0
        xdota7 = (
            -PI * (4.415 / 4.032 - 358.675 * eta_u / 6.048 - 91.495 * eta2 / 1.512)
            - chi_u * (252.9407 / 2.7216 - 845.827 * eta_u / 6.048 + 415.51 * eta2 / 8.64)
            + chisum * (
                158.0239 * eta_u / 5.4432
                - 451.597 * eta2 / 6.048
                + 20.45 * eta2 * eta_u / 4.32
                + 107.0 * eta_u * chi2 / 6.0
                - 5.0 * eta2 * chiprod / 24.0
            )
            + 12.0 * PI * chi2
            - chi2 * chi_u * (150.5 / 2.4 + eta_u / 8.0)
            + chi_u * chiprod * (10.1 * eta_u / 2.4 + 3.0 * eta2 / 8.0)
        )

        log_v2 = torch.log(torch.clamp(v2, min=1e-20))
        xdot = 1.0 + xdota2 * v2 + xdota3 * v3 + xdota4 * v4 + xdota5 * v5 + (xdota6 + xdota6log * log_v2) * v6 + xdota7 * v7
        xdot = xdot * (xdota_n * v10)
        omgdot = 1.5 * v * xdot
        ampfac = torch.sqrt(torch.abs(PI / torch.clamp(omgdot, min=1e-30)))

        an = 8.0 * eta_u * math.sqrt(PI / 5.0)
        a2_re = (-107.0 + 55.0 * eta_u) / 42.0
        a3_re = 2.0 * PI - 4.0 * chi_u / 3.0 + 2.0 * eta_u * chisum / 3.0
        a4_re = -2.173 / 1.512 - eta_u * (10.69 / 2.16 - 2.0 * chiprod) + 2.047 * eta2 / 1.512
        a5_re = -10.7 * PI / 2.1 + eta_u * (3.4 * PI / 2.1)
        a5imag = -24.0 * eta_u
        a6_re = (
            270.27409 / 6.46800
            - 8.56 * gamma_e / 1.05
            + 2.0 * (PI ** 2) / 3.0
            + eta_u * (4.1 * (PI ** 2) / 9.6 - 27.8185 / 3.3264)
            - 20.261 * eta2 / 2.772
            + 11.4635 * eta_u * eta2 / 9.9792
            - 4.28 * math.log(16.0) / 1.05
        )
        a6log = -428.0 / 105.0
        a6imag = 4.28 * PI / 1.05

        amp_pn_re = ampfac * an * v2 * (1.0 + a2_re * v2 + a3_re * v3 + a4_re * v4 + a5_re * v5 + (a6_re + a6log * log_v2) * v6)
        amp_pn_im = ampfac * an * v2 * (a5imag * v5 + a6imag * v6)
        a_pn_dimless = torch.sqrt(amp_pn_re * amp_pn_re + amp_pn_im * amp_pn_im)

        f_bar = m_sec_u * f_safe
        a_pm_dimless = gamma1_p_u * torch.pow(f_bar, 5.0 / 6.0)

        f_rd_bar = m_sec_u * f_rd_u
        sigma_bar = m_sec_u * sigma_l_u
        lorentz_bar = (sigma_bar ** 2.0) / ((f_bar - f_rd_bar) ** 2.0 + (sigma_bar ** 2.0) / 4.0)
        a_rd_dimless = eps_tide_u * delta1_u * lorentz_bar * torch.pow(f_bar, -7.0 / 6.0)

        w_ins = omega_minus(f_eval, f0_tilde_u, d_win_u)
        w_pm = omega_minus(f_eval, f1_tilde_u, d_win_u)
        w_rd = omega_plus(f_eval, f2_tilde_u, d_win_u)

        full_amp_dimless = a_pn_dimless * w_ins + a_pm_dimless * w_pm + a_rd_dimless * w_rd

        norm = ((m_sec_u ** 2.0) / d_sec_u) * (0.5 * math.sqrt(5.0 / (4.0 * PI)))
        full_amp = norm * full_amp_dimless
        full_amp = torch.where(f_eval > 0.0, full_amp, torch.zeros_like(full_amp))

        # -------------------------------------------------------------
        # 8. Step 8: Phase Model & Waveform Synthesis
        # -------------------------------------------------------------
        chi_s, chi_a, chi_pn, chi_hat = spin_combinations(m1, m2, s1, s2)
        alpha1_raw = IMRPhenomD.eval_table_v("alpha1", eta, chi_pn)
        alpha2 = IMRPhenomD.eval_table_v("alpha2", eta, chi_pn)
        alpha3 = IMRPhenomD.eval_table_v("alpha3", eta, chi_pn)
        alpha4 = IMRPhenomD.eval_table_v("alpha4", eta, chi_pn)
        alpha5 = IMRPhenomD.eval_table_v("alpha5", eta, chi_pn)
        sigma1 = IMRPhenomD.eval_table_v("sigma1", eta, chi_pn)
        sigma2 = IMRPhenomD.eval_table_v("sigma2", eta, chi_pn)
        sigma3 = IMRPhenomD.eval_table_v("sigma3", eta, chi_pn)
        sigma4 = IMRPhenomD.eval_table_v("sigma4", eta, chi_pn)
        beta2 = IMRPhenomD.eval_table_v("beta2", eta, chi_pn)
        beta3 = IMRPhenomD.eval_table_v("beta3", eta, chi_pn)

        f1_ph = 0.018
        f2_ph = 0.5 * f_rd_ph

        (tf2_0, tf2_2, tf2_3, tf2_4, tf2_5, tf2_5_log, tf2_6, tf2_6_log, tf2_7, *_) = TaylorF2.phase_coefficients(
            m1, m2, s1, s2, phenom_d_calibration=True
        )

        def eval_tf2(u):
            v_u = (PI * u) ** (1.0 / 3.0)
            v2_u = v_u * v_u; v3_u = v2_u * v_u; v4_u = v2_u * v2_u; v5_u = v4_u * v_u; v6_u = v3_u * v3_u; v7_u = v6_u * v_u
            ln_v_u = torch.log(v_u); ln_4v_u = torch.log(4.0 * v_u)
            pn_sum_u = tf2_0 + tf2_2*v2_u + tf2_3*v3_u + tf2_4*v4_u + (tf2_5 + tf2_5_log*ln_v_u)*v5_u + (tf2_6 + tf2_6_log*ln_4v_u)*v6_u + tf2_7*v7_u
            phi_tf2_u = (3.0 / (128.0 * eta * v5_u)) * pn_sum_u
            dv_du_u = (PI / 3.0) * (PI * u) ** (-2.0 / 3.0)
            d_pn_sum_dv_u = -5.0*tf2_0*(v_u**-6) - 3.0*tf2_2*(v_u**-4) - 2.0*tf2_3*(v_u**-3) - tf2_4*(v_u**-2) + tf2_5_log*(v_u**-1) + (tf2_6 + tf2_6_log*ln_4v_u + tf2_6_log) + 2.0*tf2_7*v_u
            dphi_tf2_u = (3.0 / (128.0 * eta)) * d_pn_sum_dv_u * dv_du_u
            return phi_tf2_u, dphi_tf2_u

        phi_tf2_f1, dphi_tf2_f1 = eval_tf2(as_tensor(f1_ph, device=device, dtype=dtype))
        dphi_ins_f1 = dphi_tf2_f1 + (1.0 / eta) * (sigma1 + sigma2*(f1_ph**(1/3)) + sigma3*(f1_ph**(2/3)) + sigma4*f1_ph)
        phi_ins_f1 = phi_tf2_f1 + (1.0 / eta) * (sigma1*f1_ph + (3/4)*sigma2*(f1_ph**(4/3)) + (3/5)*sigma3*(f1_ph**(5/3)) + 0.5*sigma4*(f1_ph**2))
        beta1 = eta * dphi_ins_f1 - beta2*(f1_ph**-1.0) - beta3*(f1_ph**-4.0)
        beta0 = eta * phi_ins_f1 - beta1*f1_ph - beta2*math.log(f1_ph) + (beta3/3.0)*(f1_ph**-3.0)

        dphi_int_f2 = (1.0 / eta) * (beta1 + beta2*(f2_ph**-1.0) + beta3*(f2_ph**-4.0))
        phi_int_f2 = (1.0 / eta) * (beta0 + beta1*f2_ph + beta2*torch.log(f2_ph) - (beta3/3.0)*(f2_ph**-3.0))
        atan_arg_f2 = (f2_ph - alpha5 * f_rd_ph) / f_damp_ph
        d_atan_f2 = (alpha4 / f_damp_ph) / (1.0 + atan_arg_f2 * atan_arg_f2)
        alpha1 = eta * dphi_int_f2 - alpha2*(f2_ph**-2.0) - alpha3*(f2_ph**-0.25) - d_atan_f2
        alpha0 = eta * phi_int_f2 - alpha1*f2_ph + alpha2*(f2_ph**-1.0) - (4/3)*alpha3*(f2_ph**0.75) - alpha4*torch.atan(atan_arg_f2)

        # Grid evaluation
        mf_grid = m_sec_u * f_safe
        v_g = (PI * mf_grid) ** (1.0 / 3.0)
        v2_g = v_g * v_g; v3_g = v2_g * v_g; v4_g = v2_g * v2_g; v5_g = v4_g * v_g; v6_g = v3_g * v3_g; v7_g = v6_g * v_g
        ln_v_g = torch.log(v_g); ln_4v_g = torch.log(4.0 * v_g)
        pn_sum_g = unsqueeze_param(tf2_0) + unsqueeze_param(tf2_2)*v2_g + unsqueeze_param(tf2_3)*v3_g + unsqueeze_param(tf2_4)*v4_g + (unsqueeze_param(tf2_5) + unsqueeze_param(tf2_5_log)*ln_v_g)*v5_g + (unsqueeze_param(tf2_6) + unsqueeze_param(tf2_6_log)*ln_4v_g)*v6_g + unsqueeze_param(tf2_7)*v7_g
        phi_tf2_grid = (3.0 / (128.0 * eta_u * v5_g)) * pn_sum_g

        s1_u = unsqueeze_param(sigma1); s2_u = unsqueeze_param(sigma2); s3_u = unsqueeze_param(sigma3); s4_u = unsqueeze_param(sigma4)
        phi_ins_grid = phi_tf2_grid + (1.0 / eta_u) * (s1_u*mf_grid + (3/4)*s2_u*(mf_grid**(4/3)) + (3/5)*s3_u*(mf_grid**(5/3)) + 0.5*s4_u*(mf_grid**2))

        b0_u = unsqueeze_param(beta0); b1_u = unsqueeze_param(beta1); b2_u = unsqueeze_param(beta2); b3_u = unsqueeze_param(beta3)
        phi_int_grid = (1.0 / eta_u) * (b0_u + b1_u*mf_grid + b2_u*torch.log(mf_grid) - (b3_u/3.0)*(mf_grid**-3.0))

        a0_u = unsqueeze_param(alpha0); a1_u = unsqueeze_param(alpha1); a2_u = unsqueeze_param(alpha2); a3_u = unsqueeze_param(alpha3); a4_u = unsqueeze_param(alpha4); a5_u = unsqueeze_param(alpha5)
        frd_u = unsqueeze_param(f_rd_ph); fdamp_u = unsqueeze_param(f_damp_ph)
        atan_arg_grid = (mf_grid - a5_u * frd_u) / fdamp_u
        phi_mr_grid = (1.0 / eta_u) * (a0_u + a1_u*mf_grid - a2_u*(mf_grid**-1.0) + (4/3)*a3_u*(mf_grid**0.75) + a4_u*torch.atan(atan_arg_grid))

        f1_u = unsqueeze_param(as_tensor(f1_ph, device=device, dtype=dtype))
        f2_u = unsqueeze_param(f2_ph)
        phi_phenomd = torch.where(mf_grid <= f1_u, phi_ins_grid, torch.where(mf_grid <= f2_u, phi_int_grid, phi_mr_grid))

        # Tidal phasing from NRTidalv2
        lam1 = torch.zeros_like(lam2)
        lam_tilde, _ = tidal_combinations(m1, m2, lam1, lam2)
        lam_tilde_u = unsqueeze_param(lam_tilde)
        m_omega = PI * m_sec_u * f_safe
        x_pn = torch.pow(m_omega, 2.0 / 3.0)
        c_fit = NRTIDAL_V2_PHASE_COEFFS
        num_p = 1.0 + c_fit["n_1"]*x_pn + c_fit["n_3over2"]*(x_pn**1.5) + c_fit["n_2"]*(x_pn**2) + c_fit["n_5over2"]*(x_pn**2.5) + c_fit["n_3"]*(x_pn**3)
        den_p = 1.0 + c_fit["d_1"]*x_pn + c_fit["d_3over2"]*(x_pn**1.5) + c_fit["d_2"]*(x_pn**2)
        phi_tidal = -(117.0 / (256.0 * eta_u)) * (x_pn**2.5) * (num_p / den_p) * lam_tilde_u

        # Reference phase precalculation
        ref_f = f_ref if f_ref is not None else 30.0
        fref_t = as_tensor(ref_f, device=device, dtype=dtype)
        mf_ref = m_sec * fref_t
        phi_tf2_ref, _ = eval_tf2(mf_ref)
        phi_0 = phi_tf2_ref + (1.0 / eta) * (sigma1*mf_ref + (3/4)*sigma2*(mf_ref**(4/3)) + (3/5)*sigma3*(mf_ref**(5/3)) + 0.5*sigma4*(mf_ref**2))
        phi_precalc = unsqueeze_param(2.0 * phi_c + phi_0)

        # Time shift t0 evaluated at f_final * M_sec
        fmax_eval = kwargs.get("f_final", kwargs.get("fhigh", torch.max(f_eval)))
        fmax_t = as_tensor(fmax_eval, device=device, dtype=dtype)
        f_eval_t0 = m_sec * fmax_t
        atan_eval = (f_eval_t0 - alpha5 * f_rd_ph) / f_damp_ph
        t0_phenom = (1.0 / eta) * (alpha1_raw + alpha2*(f_eval_t0**-2.0) + alpha3*(f_eval_t0**-0.25) + (alpha4 / f_damp_ph) / (1.0 + atan_eval**2))
        t0_total = t0_phenom + (2.0 * PI * t_c) / m_sec
        t0_u = unsqueeze_param(t0_total)
        mf_ref_u = unsqueeze_param(mf_ref)

        total_phase = phi_phenomd + phi_tidal - t0_u * (mf_grid - mf_ref_u) - phi_precalc

        # -------------------------------------------------------------
        # 8c. Polarizations Projection onto -2Y_2,\pm 2
        # -------------------------------------------------------------
        cos_inc = torch.cos(unsqueeze_param(inc))
        plus_factor = 0.5 * (1.0 + cos_inc * cos_inc)
        cross_factor = cos_inc

        h_plus = - plus_factor * full_amp * torch.complex(torch.cos(total_phase), -torch.sin(total_phase))
        h_cross = torch.complex(
            cross_factor * full_amp * torch.sin(total_phase),
            cross_factor * full_amp * torch.cos(total_phase),
        )

        # Enforce exact finite zeros for DC and non-positive frequency bins
        h_plus = torch.where(f_eval > 0.0, h_plus, torch.zeros_like(h_plus))
        h_cross = torch.where(f_eval > 0.0, h_cross, torch.zeros_like(h_cross))

        return h_plus, h_cross

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

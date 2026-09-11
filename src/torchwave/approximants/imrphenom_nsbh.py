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
    tidal_combinations,
)
from torchwave.approximants.imrphenomd import IMRPhenomD
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
        xi_tide, _ = solve_xi_tide(q, chi, c)
        r_tide = xi_tide * m2_sec * torch.clamp(1.0 - 2.0 * c, min=0.0) / torch.clamp(c, min=1e-6)
        denom_ftide = PI * (chi * m1_sec + torch.sqrt(torch.clamp((r_tide ** 3.0) / m1_sec, min=0.0) + 1e-30))
        f_tide = 1.0 / denom_ftide

        # -------------------------------------------------------------
        # 3. Step 3: Baryonic Torus Remnant Mass Ratio
        # -------------------------------------------------------------
        r_isco = kerr_isco_radius(chi)
        torus_ratio = torch.clamp(
            0.296 * xi_tide * (1.0 - 2.0 * c) - 0.171 * q * c * r_isco,
            min=0.0,
        )

        # -------------------------------------------------------------
        # 4. Step 4: Remnant Mass M_f and Remnant Spin chi_f (Zappa 2019)
        # -------------------------------------------------------------
        mf, chif = zappa_remnant_properties(m1, m2, s1, s2, lam2)

        # -------------------------------------------------------------
        # 5. Step 5: Kerr Ringdown Frequency f_RD and Quality Factor Q (London 2018)
        # -------------------------------------------------------------
        f_rd, q_factor = london_ringdown_properties(mf, chif)

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
            return 0.5 * (1.0 + torch.tanh(arg))

        def omega_minus(val: torch.Tensor, f0: Union[float, torch.Tensor], d: Union[float, torch.Tensor]) -> torch.Tensor:
            d_val = as_tensor(d, device=device, dtype=dtype)
            f0_val = as_tensor(f0, device=device, dtype=dtype)
            arg = torch.clamp(4.0 * (val - f0_val) / torch.clamp(d_val.abs(), min=1e-8), -50.0, 50.0)
            return 0.5 * (1.0 - torch.tanh(arg))

        is_ftide_ge_frd = f_tide >= f_rd
        has_torus = torus_ratio > 0.0

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
            (1.0 + 0.5 * lam_safe - 0.25 * (lam_safe ** 2.0)) * gamma1,
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
        s1_u = unsqueeze_param(s1)
        s2_u = unsqueeze_param(s2)

        gamma1_p_u = unsqueeze_param(gamma1_prime)
        delta1_u = unsqueeze_param(delta1)
        sigma_l_u = unsqueeze_param(sigma_lorentz)
        eps_tide_u = unsqueeze_param(eps_tide)
        f_rd_u = unsqueeze_param(f_rd)

        f0_tilde_u = unsqueeze_param(f0_tilde)
        f1_tilde_u = unsqueeze_param(f1_tilde)
        f2_tilde_u = unsqueeze_param(f2_tilde)
        d_win_u = unsqueeze_param(d_win)

        v = torch.pow(PI * m_sec_u * f_safe, 1.0 / 3.0)
        x = v * v

        euler_gamma = 0.5772156649015329
        a0 = torch.ones_like(eta_u)
        a1 = torch.zeros_like(eta_u)
        a2 = -743.0 / 336.0 - 11.0 * eta_u / 4.0
        a3 = 4.0 * PI - 113.0 * chi_u / 12.0 + 19.0 * eta_u * (s1_u + s2_u) / 6.0
        a4 = (
            34103.0 / 18144.0
            + 5.0 * (chi_u ** 2.0)
            + eta_u * (13661.0 / 2016.0 - s1_u * s2_u / 8.0)
            + 59.0 * (eta_u ** 2.0) / 18.0
        )
        a5 = (
            -PI * (4159.0 / 672.0 + 189.0 * eta_u / 8.0)
            - chi_u * (31571.0 / 1008.0 - 1165.0 * eta_u / 24.0)
            + (s1_u + s2_u) * (21863.0 * eta_u / 1008.0 - 79.0 * (eta_u ** 2.0) / 6.0)
            - 0.75 * (chi_u ** 3.0)
            + 2.25 * eta_u * chi_u * s1_u * s2_u
        )
        a6 = (
            16447322263.0 / 139708800.0
            - 1712.0 * euler_gamma / 105.0
            + 16.0 * (PI ** 2.0) / 3.0
            - (856.0 / 105.0) * torch.log(16.0 * x)
            + eta_u * (451.0 * (PI ** 2.0) / 48.0 - 56198689.0 / 217728.0)
            + 541.0 * (eta_u ** 2.0) / 896.0
            - 5605.0 * (eta_u ** 3.0) / 2592.0
            - 80.0 * PI * chi_u / 3.0
            + (20.0 * PI / 3.0 - 1135.0 * chi_u / 36.0) * eta_u * (s1_u + s2_u)
            + (64153.0 / 1008.0 - 457.0 * eta_u / 36.0) * (chi_u ** 2.0)
            - (787.0 * eta_u / 144.0 - 3037.0 * (eta_u ** 2.0) / 144.0) * s1_u * s2_u
        )
        a7 = (
            -PI * (4415.0 / 4032.0 - 358675.0 * eta_u / 6048.0 - 91495.0 * (eta_u ** 2.0) / 1512.0)
            - chi_u * (2529407.0 / 27216.0 - 845827.0 * eta_u / 6048.0 + 41551.0 * (eta_u ** 2.0) / 864.0)
            + (s1_u + s2_u) * (
                1580239.0 * eta_u / 54432.0
                - 451597.0 * (eta_u ** 2.0) / 6048.0
                + 2045.0 * (eta_u ** 3.0) / 432.0
                + 107.0 * eta_u * (chi_u ** 2.0) / 6.0
                - 5.0 * (eta_u ** 2.0) * s1_u * s2_u / 24.0
            )
            + 12.0 * PI * (chi_u ** 2.0)
            - (chi_u ** 3.0) * (1505.0 / 24.0 + eta_u / 8.0)
            + chi_u * s1_u * s2_u * (101.0 * eta_u / 24.0 + 3.0 * (eta_u ** 2.0) / 8.0)
        )

        cap_a0 = torch.ones_like(eta_u)
        cap_a1 = torch.zeros_like(eta_u)
        cap_a2 = -107.0 / 42.0 + 55.0 * eta_u / 42.0
        cap_a3 = 2.0 * PI - 4.0 * chi_u / 3.0 + 2.0 * eta_u * (s1_u + s2_u) / 3.0
        cap_a4 = -2173.0 / 1512.0 - eta_u * (1069.0 / 216.0 - 2.0 * s1_u * s2_u) + 2047.0 * (eta_u ** 2.0) / 1512.0
        cap_a5 = -107.0 * PI / 21.0 + 34.0 * PI * eta_u / 21.0
        cap_a6 = (
            27027409.0 / 646800.0
            - 856.0 * euler_gamma / 105.0
            + 2.0 * (PI ** 2.0) / 3.0
            + eta_u * (41.0 * (PI ** 2.0) / 96.0 - 278185.0 / 33264.0)
            - 20261.0 * (eta_u ** 2.0) / 2772.0
            + 114635.0 * (eta_u ** 3.0) / 99792.0
            - (428.0 / 105.0) * torch.log(16.0 * x)
        )

        sum_cap_a = (
            cap_a0
            + cap_a1 * (x ** 0.5)
            + cap_a2 * x
            + cap_a3 * (x ** 1.5)
            + cap_a4 * (x ** 2.0)
            + cap_a5 * (x ** 2.5)
            + cap_a6 * (x ** 3.0)
        )
        sum_a = (
            a0
            + a1 * (x ** 0.5)
            + a2 * x
            + a3 * (x ** 1.5)
            + a4 * (x ** 2.0)
            + a5 * (x ** 2.5)
            + a6 * (x ** 3.0)
            + a7 * (x ** 3.5)
        )
        sum_a_safe = torch.clamp(sum_a, min=1e-8)

        prefac_pn = (
            ((m_sec_u ** 2.0) / d_sec_u)
            * PI
            * torch.sqrt(2.0 * eta_u / 3.0)
            * torch.pow(PI * m_sec_u * f_safe, -7.0 / 6.0)
        )
        a_pn = prefac_pn * (sum_cap_a / torch.sqrt(sum_a_safe))

        a_pm = (
            ((m_sec_u ** 2.0) / d_sec_u)
            * gamma1_p_u
            * torch.pow(PI * m_sec_u * f_safe, 5.0 / 6.0)
            * torch.pow(PI * m_sec_u, -7.0 / 6.0)
        )

        lorentz = (sigma_l_u ** 2.0) / ((f_safe - f_rd_u) ** 2.0 + (sigma_l_u ** 2.0) / 4.0)
        a_rd = (
            ((m_sec_u ** 2.0) / d_sec_u)
            * eps_tide_u
            * delta1_u
            * lorentz
            * torch.pow(PI * m_sec_u * f_safe, -7.0 / 6.0)
        )

        w_ins = omega_minus(f_eval, f0_tilde_u, d_win_u)
        w_pm = omega_minus(f_eval, f1_tilde_u, d_win_u)
        w_rd = omega_plus(f_eval, f2_tilde_u, d_win_u)

        full_amp = a_pn * w_ins + a_pm * w_pm + a_rd * w_rd
        full_amp = torch.where(f_eval > 0.0, full_amp, torch.zeros_like(full_amp))

        # -------------------------------------------------------------
        # 8. Step 8: Phase Model & Waveform Synthesis
        # -------------------------------------------------------------
        # 8a. Base BBH phase from IMRPhenomD
        _, _, phi_bbh = self.bbh_model(
            f=f_eval,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            distance=dist,
            inclination=torch.zeros_like(inc),
            phic=phi_c,
            tc=t_c,
            f_ref=f_ref,
            phasing_convention=phasing_convention,
            reference_frame=reference_frame,
            return_phase=True,
        )

        # 8b. Tidal phasing from NRTidalv2 (Dietrich et al. 2019)
        lam1 = torch.zeros_like(lam2)
        lam_tilde, _ = tidal_combinations(m1, m2, lam1, lam2)
        lam_tilde_u = unsqueeze_param(lam_tilde)

        leading_factor = -(117.0 / 256.0) * (lam_tilde_u / eta_u)
        num_pade = (
            1.0
            + N1_V2 * x
            + N32_V2 * (x ** 1.5)
            + N2_V2 * (x ** 2.0)
            + N52_V2 * (x ** 2.5)
            + N3_V2 * (x ** 3.0)
        )
        den_pade = 1.0 + D1_V2 * x + D32_V2 * (x ** 1.5) + D2_V2 * (x ** 2.0)
        psi_tidal = leading_factor * (x ** 2.5) * (num_pade / den_pade)

        # Self-spin corrections
        s23, s35 = _nrtidal_self_spin_components(
            x=x,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            lambda1=lam1,
            lambda2=lam2,
        )

        conv = reference_frame if reference_frame is not None else (
            phasing_convention if phasing_convention is not None else (
                "reference_frequency" if f_ref is not None else "coalescence"
            )
        )
        conv = conv.lower()

        if conv in ("reference_frequency", "reffreq", "ref"):
            ref_f = f_ref if f_ref is not None else (f_eval.reshape(-1)[0] if f_eval.numel() > 0 else 20.0)
            fref_t = as_tensor(ref_f, device=device, dtype=dtype)
            fref_u = unsqueeze_param(fref_t)
            safe_fref = torch.clamp(fref_u, min=1e-8)
            x_ref = torch.pow(PI * m_sec_u * safe_fref, 2.0 / 3.0)
            s23_ref, _ = _nrtidal_self_spin_components(
                x=x_ref,
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                lambda1=lam1,
                lambda2=lam2,
            )
            total_delta_psi = psi_tidal + (s23 - s23_ref) + s35
        else:
            total_delta_psi = psi_tidal + s23 + s35

        total_phase = phi_bbh + total_delta_psi

        # -------------------------------------------------------------
        # 8c. Polarizations Projection onto -2Y_2,\pm 2
        # -------------------------------------------------------------
        cos_inc = torch.cos(unsqueeze_param(inc))
        plus_factor = 0.5 * (1.0 + cos_inc * cos_inc)
        cross_factor = cos_inc

        exp_minus_i_phase = torch.complex(torch.cos(total_phase), -torch.sin(total_phase))

        h_plus = (plus_factor * full_amp).to(exp_minus_i_phase.dtype) * exp_minus_i_phase
        h_cross = torch.complex(
            -cross_factor * full_amp * torch.sin(total_phase),
            -cross_factor * full_amp * torch.cos(total_phase),
        )

        # Enforce exact finite zeros for DC and non-positive frequency bins
        h_plus = torch.where(f_eval > 0.0, h_plus, torch.zeros_like(h_plus))
        h_cross = torch.where(f_eval > 0.0, h_cross, torch.zeros_like(h_cross))

        return h_plus, h_cross

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

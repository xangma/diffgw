"""IMRPhenomNSBH Frequency-Domain Gravitational Waveform Approximant.

Literature-based reconstruction of the following models:
- Thompson et al. (2020), Phys. Rev. D 101, 124059 (arXiv:2002.08383)
- Pannarale et al. (2015), Phys. Rev. D 92, 084050 (arXiv:1509.00512)
- Pannarale et al. (2015), Phys. Rev. D 92, 081504 (arXiv:1509.06209)
- Zappa et al. (2019), Phys. Rev. Lett. 123, 041102 (arXiv:1903.11622)
- Jiménez-Forteza et al. (2017), Phys. Rev. D 95, 064024 (arXiv:1611.00332)
- London and Fauchon-Jones (2019), arXiv:1810.03550, Eq. (22) for the 220 QNM fit
- Santamaria et al. (2010), Phys. Rev. D 82, 064016 (arXiv:1005.3306)
- Dietrich et al. (2019), Phys. Rev. D 100, 044003 (arXiv:1905.06011)

The bounded equation review in docs/reviews/2026-09-12-clean-room/nsbh.md
does not certify the origin of every inherited coefficient or phase convention.
"""

from typing import NamedTuple, Optional, Tuple, Union
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
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.nsbh_phase_baseline import NSBHPhaseBaseline
from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
from torchwave.approximants.nsbh_clean_phase import nrtidalv2_frequency_phase
from torchwave.approximants.nsbh_clean_amplitude import positive_tidal_frequency, delta2_prime_plus, gamma1_prime_small_lambda, outer_pn_center


# Coefficients transcribed from the author-released ``BHNSremnant.py``
# accompanying Zappa et al. (2019), arXiv:1903.11622:
# https://arxiv.org/pdf/1903.11622.pdf
# SHA-256 of the released file used for this transcription:
# f0043e0f2fc4b3babe664c527c5a20e8242b81c09bc023bfc3b016f1ff1bbcad
ZAPPA_MASS_COEFFS = (
    -1.83417425e-3, 2.39226041e-3, 4.29407902e-3, 9.79775571e-3,
    2.33868869e-7, -8.28090025e-7, -1.64315549e-6, 8.08340931e-6,
    -2.00726981e-2, 1.31986011e-1, 6.50754064e-2, -1.42749961e-1,
)
ZAPPA_SPIN_COEFFS = (
    -5.44187381e-3, 7.91165608e-3, 2.33362046e-2, 2.47764497e-2,
    -8.56844797e-7, -2.81727682e-6, 6.61290966e-6, 4.28979016e-5,
    -3.04174272e-2, 2.54889050e-1, 1.47549350e-1, -4.27905832e-1,
)


def ns_compactness_from_lambda(lambda_ns: torch.Tensor) -> torch.Tensor:
    """Compute neutron star compactness C = M_NS / R_NS from tidal deformability Lambda.

    Thompson et al. (2020), Appendix B, Eq. (32):
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

    Polynomial from Thompson et al. (2020), Appendix A, Eq. (10):
        P(xi) = xi^5 - 3*mu*xi^4 + 2*chi*sqrt(mu^3 * xi^7) - 3*q*xi^2 + 6*q*mu*xi - 3*q*mu^2 * chi^2 = 0
        P'(xi) = 5*xi^4 - 12*mu*xi^3 + 7*chi*sqrt(mu^3 * xi^5) - 6*q*xi + 6*q*mu
    with mu = q * C.

    Initial guess: xi_0 = (3q)^(1/3) + 3*mu. A fixed iteration count
    does not guarantee convergence; the returned residual permits checking it.

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
    """Compute the NSBH remnant mass and spin from the Zappa fit.

    Zappa et al. (2019), Eqs. (1), (4)-(6), define the fit as
    ``F(nu, a_BH, Lambda)``.  Their NS spin is neglected, so the tidal
    correction polynomials use the black-hole spin ``chi1`` directly.  The
    BBH baseline remains evaluated with both caller-supplied spins for API
    compatibility; the published NSBH calibration itself assumes ``chi2=0``.
    """
    m = mass1 + mass2
    eta = component_masses_to_eta(mass1, mass2)
    # Zappa's a_BH is the initial black-hole spin, not a mass-weighted
    # effective spin.  In the NSBH ordering used here, that is chi1.
    chi = chi1

    mf_bbh_ratio, chif_bbh = bbh_remnant_uib2016(mass1, mass2, chi1, chi2)

    def _zappa_factor(coeff):
        p1 = ((coeff[0] * chi + coeff[1]) + (coeff[2] * chi + coeff[3]) * eta) * eta
        p2 = ((coeff[4] * chi + coeff[5]) + (coeff[6] * chi + coeff[7]) * eta) * eta
        p3 = ((coeff[8] * chi + coeff[9]) + (coeff[10] * chi + coeff[11]) * eta) * eta
        return (1.0 + p1 * lambda2 + p2 * (lambda2 ** 2.0)) / ((1.0 + (p3 ** 2.0) * lambda2) ** 2.0)

    mass_factor = _zappa_factor(ZAPPA_MASS_COEFFS)
    spin_factor = _zappa_factor(ZAPPA_SPIN_COEFFS)

    # Zappa et al. enforce the BBH result in the extrapolating regions where
    # the fit is known to become unphysical.  This is a correction factor of
    # one, so the BBH remnant (rather than the total mass) remains the base.
    bbh_fallback = torch.logical_or(
        torch.logical_and(chi < 0.0, eta < 0.188), chi < -0.5
    )
    mass_factor = torch.where(mass_factor > 1.0, torch.ones_like(mass_factor), mass_factor)
    mass_factor = torch.where(bbh_fallback, torch.ones_like(mass_factor), mass_factor)
    # The same physical guard keeps the spin correction from increasing the
    # BBH value in the extrapolating small-tide region; public properties
    # reduce to the BBH spin there.
    spin_factor = torch.where(spin_factor > 1.0, torch.ones_like(spin_factor), spin_factor)
    spin_factor = torch.where(bbh_fallback, torch.ones_like(spin_factor), spin_factor)

    mf = m * mf_bbh_ratio * mass_factor
    chif = chif_bbh * spin_factor

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


def nsbh_branch_predicates(
    f_tide: torch.Tensor, f_rd: torch.Tensor, torus_ratio: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Table I's closed frequency boundary and strictly positive torus flag."""
    return f_tide >= f_rd, torus_ratio > 0.0


class NSBHBranchAmplitudeFits(NamedTuple):
    f_rd_tilde: torch.Tensor
    x_nd: torch.Tensor
    x_prime_nd: torch.Tensor
    x_d: torch.Tensor
    x_prime_d: torch.Tensor
    epsilon_tide: torch.Tensor
    epsilon_ins: torch.Tensor
    sigma_tide: torch.Tensor


class NSBHJoiningFrequencies(NamedTuple):
    f0: torch.Tensor
    f1: torch.Tensor
    f2: torch.Tensor


def nsbh_joining_frequencies(
    q: torch.Tensor,
    f_tide: torch.Tensor,
    f_rd_tilde: torch.Tensor,
    epsilon_ins: torch.Tensor,
    tide_at_or_above_ringdown: torch.Tensor,
    has_torus: torch.Tensor,
) -> NSBHJoiningFrequencies:
    """Table I joins with the documented low/no-torus carrier adapter."""
    f0 = torch.where(
        tide_at_or_above_ringdown,
        torch.where(has_torus, epsilon_ins * f_rd_tilde, f_rd_tilde),
        torch.where(
            has_torus,
            epsilon_ins * f_tide,
            outer_pn_center(((q - 1.0) * f_rd_tilde + epsilon_ins * f_tide) / q, epsilon_ins),
        ),
    )
    f1 = torch.where(
        tide_at_or_above_ringdown,
        torch.where(has_torus, epsilon_ins * f_rd_tilde, f_rd_tilde),
        torch.where(has_torus, f_tide, ((q - 1.0) * f_rd_tilde + f_tide) / q),
    )
    return NSBHJoiningFrequencies(f0, f1, f_rd_tilde)


def nsbh_branch_amplitude_fits(
    chi: torch.Tensor,
    eta: torch.Tensor,
    compactness: torch.Tensor,
    f_tide: torch.Tensor,
    f_rd: torch.Tensor,
    torus_ratio: torch.Tensor,
    lambda_ns: torch.Tensor,
    tide_at_or_above_ringdown: torch.Tensor,
    has_torus: torch.Tensor,
) -> NSBHBranchAmplitudeFits:
    """Thompson Appendix A Eqs. (20)--(24), (30), and Table I factors."""
    f_rd_tilde = 0.98 * f_rd * torch.where(
        lambda_ns > 1.0,
        0.99,
        1.0 - 0.02 * lambda_ns + 0.01 * lambda_ns.square(),
    )
    u = ((f_tide - f_rd_tilde) / f_rd_tilde).square()
    x_nd = u - 0.571505 * compactness - 0.00508451 * chi
    x_prime_nd = u - 0.657424 * compactness - 0.0259977 * chi
    x_d = torus_ratio + 0.424912 * compactness + 0.363604 * torch.sqrt(eta) - 0.060559 * chi
    x_prime_d = (
        torus_ratio
        - 0.132754 * compactness
        + 0.576669 * torch.sqrt(eta)
        - 0.0603749 * chi
        - 0.0601185 * chi.square()
        - 0.0729134 * chi.pow(3.0)
    )
    high_arg = torch.clamp(4.0 * (x_nd + 0.0796251) / 0.0801192, -50.0, 50.0)
    sigma_arg = torch.clamp(4.0 * (x_prime_nd + 0.206465) / 0.226844, -50.0, 50.0)
    epsilon_tide_high = torch.sigmoid(2.0 * high_arg)
    sigma_tide_high = torch.sigmoid(-2.0 * sigma_arg)
    epsilon_ins_disrupt = 1.29971 - 1.61724 * x_d
    sigma_tide_disrupt = 0.137722 - 0.293237 * x_prime_d
    epsilon_tide = torch.where(tide_at_or_above_ringdown, epsilon_tide_high, 0.0)
    epsilon_ins = torch.where(tide_at_or_above_ringdown & ~has_torus, 1.0, epsilon_ins_disrupt)
    sigma_tide = torch.where(
        tide_at_or_above_ringdown,
        sigma_tide_high,
        torch.where(has_torus, sigma_tide_disrupt, 0.5 * (sigma_tide_high + sigma_tide_disrupt)),
    )
    return NSBHBranchAmplitudeFits(
        f_rd_tilde, x_nd, x_prime_nd, x_d, x_prime_d, epsilon_tide, epsilon_ins, sigma_tide
    )


def nsbh_torus_mass_ratio(
    q: torch.Tensor,
    chi: torch.Tensor,
    compactness: torch.Tensor,
    xi_tide: torch.Tensor,
) -> torch.Tensor:
    """Physical torus fraction from Thompson Appendix A Eqs. (11)--(12)."""
    raw_ratio = (
        0.296 * xi_tide * (1.0 - 2.0 * compactness)
        - 0.171 * q * compactness * kerr_isco_radius(chi)
    )
    return torch.clamp(raw_ratio, min=0.0)


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
    f_tide = positive_tidal_frequency(denom)

    # 3. Kerr ISCO frequency
    f_isco = kerr_isco_frequency(mass1, chi1)

    # 4. Ringdown frequency of remnant BH via Zappa et al. (2019) fits
    mf, chif = zappa_remnant_properties(mass1, mass2, chi1, torch.zeros_like(chi1), lambda2)
    f_rd, _ = london_ringdown_properties(mf, chif)

    return f_tide, f_isco, f_rd, c


def _amplitude_window(
    frequency_hz: torch.Tensor,
    center_hz: torch.Tensor,
    width_mf: torch.Tensor,
    total_mass_sec: torch.Tensor,
    *,
    rising: bool,
) -> torch.Tensor:
    """Thompson Eq. (30), with the width expressed in total-mass units.

    Pannarale (2015), Sec. III and Eq. (13), defines d + sigma_tide
    in M*f, so the frequency difference must be multiplied by M.
    """
    offset_mf = total_mass_sec * (frequency_hz - center_hz)
    arg = 8.0 * offset_mf / torch.clamp(width_mf.abs(), min=1e-8)
    return torch.sigmoid(arg if rising else -arg)


def _legacy_phase_offsets(m1, m2, s1, s2, mf_ref, mf_anchor, remnant_mass, remnant_spin):
    """Retain the inherited NSBH affine gauge, not a published prescription.

    Its constant uses the inspiral phase at the reference frequency; its time
    origin uses the raw MR slope at the requested upper frequency, evaluated
    with the amplitude remnant. These offsets do not alter intrinsic curvature.
    """
    eta = component_masses_to_eta(m1, m2)
    _, _, chi_pn, _ = spin_combinations(m1, m2, s1, s2)
    sigma1, sigma2, sigma3, sigma4 = (
        IMRPhenomD.eval_table_v(name, eta, chi_pn)
        for name in ("sigma1", "sigma2", "sigma3", "sigma4")
    )
    alpha1, alpha2, alpha3, alpha4, alpha5 = (
        IMRPhenomD.eval_table_v(name, eta, chi_pn)
        for name in ("alpha1", "alpha2", "alpha3", "alpha4", "alpha5")
    )
    p0, p2, p3, p4, p5, p5log, p6, p6log, p7, *_ = TaylorF2.phase_coefficients(
        m1, m2, s1, s2, phenom_d_calibration=True,
    )
    v_ref = (PI * mf_ref).pow(1.0 / 3.0)
    pn_sum = (
        p0 + p2 * v_ref.square() + p3 * v_ref.pow(3) + p4 * v_ref.pow(4)
        + (p5 + p5log * torch.log(v_ref)) * v_ref.pow(5)
        + (p6 + p6log * torch.log(4.0 * v_ref)) * v_ref.pow(6)
        + p7 * v_ref.pow(7)
    )
    reference_phase = 3.0 * pn_sum / (128.0 * eta * v_ref.pow(5)) + (
        sigma1 * mf_ref + 0.75 * sigma2 * mf_ref.pow(4.0 / 3.0)
        + 0.6 * sigma3 * mf_ref.pow(5.0 / 3.0) + 0.5 * sigma4 * mf_ref.square()
    ) / eta
    omega_r, omega_i = evaluate_kerr_qnm_220(remnant_spin)
    mass_ratio = (m1 + m2) / remnant_mass
    f_rd = mass_ratio * omega_r / (2.0 * PI)
    f_damp = mass_ratio * omega_i.abs() / (2.0 * PI)
    atan_arg = (mf_anchor - alpha5 * f_rd) / f_damp
    time_slope = (
        alpha1 + alpha2 * mf_anchor.pow(-2.0) + alpha3 * mf_anchor.pow(-0.25)
        + (alpha4 / f_damp) / (1.0 + atan_arg.square())
    ) / eta
    return reference_phase, time_slope


class IMRPhenomNSBH(nn.Module):
    """IMRPhenomNSBH Aligned-Spin Frequency-Domain Waveform Approximant for NS-BH Mergers.

    Reconstruction of Thompson et al. (2020), Phys. Rev. D 101, 124059.
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
            phasing_convention: None or 'legacy' retains the inherited affine
                offsets, including its upper-frequency time anchor. Explicit
                'reference_frequency' or 'coalescence' uses the PhenomD gauge.
            reference_frame: Alias for phasing_convention
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
        f_tide = positive_tidal_frequency(denom_ftide)

        # -------------------------------------------------------------
        # 3. Step 3: Baryonic Torus Remnant Mass Ratio
        # -------------------------------------------------------------
        torus_ratio = nsbh_torus_mass_ratio(q, s1, c, xi_tide)

        # -------------------------------------------------------------
        # 4. Step 4: Remnant Mass M_f and Remnant Spin chi_f
        # -------------------------------------------------------------
        # Thompson et al. (2020), Appendix A, steps 4--6: use the
        # NSBH-specific remnant fit for every merger classification.  The
        # BBH limit is already recovered by Zappa et al.'s fit at Lambda=0;
        # selecting the BBH remnant conditionally for a non-disruptive system
        # would therefore change the source-defined f_tide/f_RD boundary at
        # finite Lambda.
        mf, chi_f = zappa_remnant_properties(m1, m2, s1, s2, lam2)

        # -------------------------------------------------------------
        # 5. Step 5: Kerr Ringdown Frequency f_RD and Quality Factor Q (London 2018)
        # -------------------------------------------------------------
        f_rd, q_factor = london_ringdown_properties(mf, chi_f)
        # Step 6 applies the table's f_tide >= f_RD (closed) boundary.
        is_ftide_ge_frd, has_torus = nsbh_branch_predicates(f_tide, f_rd, torus_ratio)

        # -------------------------------------------------------------
        # 6. Step 6: Merger Classification and Joining Parameters
        # -------------------------------------------------------------
        lam_safe = torch.clamp(lam2, min=0.0)
        # Eqs. (21)--(24) use the BH spin, rather than the PhenomC effective
        # spin used below for the BBH carrier amplitude.
        branch_fits = nsbh_branch_amplitude_fits(
            s1, eta, c, f_tide, f_rd, torus_ratio, lam2, is_ftide_ge_frd, has_torus
        )
        f_rd_tilde = branch_fits.f_rd_tilde
        eps_tide = branch_fits.epsilon_tide
        eps_ins = branch_fits.epsilon_ins
        sigma_tide = branch_fits.sigma_tide

        f0_tilde, f1_tilde, f2_tilde = nsbh_joining_frequencies(
            q, f_tide, f_rd_tilde, eps_ins, is_ftide_ge_frd, has_torus
        )
        d_win_mf = 0.015 + sigma_tide

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
            gamma1_prime_small_lambda(lam_safe) * gamma1,
        )
        delta2_prime_gt1 = delta2_prime_plus(f_tide, f_rd_tilde)
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
        d_win_mf_u = unsqueeze_param(d_win_mf)

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

        w_ins = _amplitude_window(f_eval, f0_tilde_u, d_win_mf_u, m_sec_u, rising=False)
        w_pm = _amplitude_window(f_eval, f1_tilde_u, d_win_mf_u, m_sec_u, rising=False)
        w_rd = _amplitude_window(f_eval, f2_tilde_u, d_win_mf_u, m_sec_u, rising=True)

        full_amp_dimless = a_pn_dimless * w_ins + a_pm_dimless * w_pm + a_rd_dimless * w_rd

        norm = ((m_sec_u ** 2.0) / d_sec_u) * (0.5 * math.sqrt(5.0 / (4.0 * PI)))
        full_amp = norm * full_amp_dimless
        full_amp = torch.where(f_eval > 0.0, full_amp, torch.zeros_like(full_amp))

        # -------------------------------------------------------------
        # 8. Step 8: Phase Model & Waveform Synthesis
        # -------------------------------------------------------------
        # Thompson et al. (2020), Sec. II.B: the phase baseline is PhenomD.
        # Its remnant and C1 joins therefore belong to the BBH phase model,
        # independently of the disruption-dependent amplitude remnant above.
        # The papers leave affine offsets free. Preserve the inherited default
        # gauge; explicit conventions opt into the grid-independent BBH gauge.
        # The public NSBH API uses zero as the sentinel for a reference at the
        # lower edge of the generated positive-frequency grid.  Keep the
        # inherited 30 Hz default for an omitted reference, but resolve an
        # explicit zero before evaluating the logarithmic phase expressions.
        ref_f = f_ref if f_ref is not None else 30.0
        ref_f = as_tensor(ref_f, device=device, dtype=dtype)
        if not torch.compiler.is_compiling():
            if torch.any(~torch.isfinite(ref_f)):
                raise ValueError("f_ref must contain only finite, non-negative values")
            if torch.any(ref_f < 0.0):
                raise ValueError("f_ref must be non-negative; zero selects the lowest positive frequency")
        first_positive = torch.amin(
            torch.where(
                f_eval > 0.0,
                f_eval,
                torch.full_like(f_eval, float("inf")),
            )
        )
        first_positive = torch.where(
            torch.isfinite(first_positive),
            first_positive,
            torch.full_like(first_positive, 30.0),
        )
        ref_f = torch.where(ref_f > 0.0, ref_f, first_positive)
        requested_frame = reference_frame if reference_frame is not None else phasing_convention
        use_legacy_gauge = requested_frame is None or requested_frame.lower() == "legacy"
        _, _, phi_phenomd = NSBHPhaseBaseline(chi_f)(
            f, m1, m2, s1, s2,
            distance=dist, inclination=inc,
            phic=0.0 if use_legacy_gauge else phi_c,
            tc=0.0 if use_legacy_gauge else t_c,
            f_ref=ref_f, window_type=None,
            phasing_convention="coalescence" if use_legacy_gauge else phasing_convention,
            reference_frame=None if use_legacy_gauge else reference_frame,
            return_phase=True,
        )
        if use_legacy_gauge:
            mf_ref = m_sec * as_tensor(ref_f, device=device, dtype=dtype)
            anchor_hz = kwargs.get("f_final", kwargs.get("fhigh", torch.max(f_eval)))
            mf_anchor = m_sec * as_tensor(anchor_hz, device=device, dtype=dtype)
            reference_phase, time_slope = _legacy_phase_offsets(
                m1, m2, s1, s2, mf_ref, mf_anchor,
                IMRPhenomD().remnant_properties(m1, m2, s1, s2)[0], chi_f,
            )
            # Remove PhenomD's affine offsets, retaining its intrinsic IMR
            # phase, then restore the pre-existing default NSBH offsets.
            slope_peak = NSBHPhaseBaseline(chi_f).compute_slope_peak(m1, m2, s1, s2)
            phi_phenomd = (
                phi_phenomd + PI / 4.0 + unsqueeze_param(slope_peak) * f_bar
                - unsqueeze_param(reference_phase + 2.0 * phi_c)
                - unsqueeze_param(time_slope + 2.0 * PI * t_c / m_sec)
                * (f_bar - unsqueeze_param(mf_ref))
            )

        # Tidal phasing from NRTidalv2
        lam1 = torch.zeros_like(lam2)
        lam_tilde, _ = tidal_combinations(m1, m2, lam1, lam2)
        lam_tilde_u = unsqueeze_param(lam_tilde)
        phi_tidal = nrtidalv2_frequency_phase(
            f_safe, m_sec_u, eta_u, lam_tilde_u,
        )

        # Dietrich et al. (2019), Eqs. (4), (17)-(18): add the tidal
        # phase to the BBH phase. The supplied reference phase belongs to the
        # baseline; the tidal contribution is not independently re-anchored.
        total_phase = phi_phenomd + phi_tidal

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

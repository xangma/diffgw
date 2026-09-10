"""Kinematic and parameter conversions for gravitational-wave astrophysics.

All conversions strictly implement peer-reviewed formulations:
- Symmetric mass ratio and PN velocity: Blanchet (2014), Living Rev. Relativity 17, 2 (arXiv:1310.1528)
- Spin combinations: Bohé et al. (2013), CQG 30, 135009 (arXiv:1303.7412); Khan et al. (2016), PRD 93, 044007 (arXiv:1508.07253)
- Tidal parameter combinations: Flanagan & Hinderer (2008), PRD 77, 021502; Vines et al. (2011), PRD 83, 084051; Wade et al. (2014), PRD 89, 103012.
"""

from typing import Tuple, Union
import math
import torch

from torchwave.constants import MSUN_SEC, PI


def as_tensor(
    x: Union[float, int, torch.Tensor],
    device: Union[str, torch.device, None] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Ensure input is converted to a torch.Tensor with specified dtype and device."""
    if isinstance(x, torch.Tensor):
        if device is not None and x.device != torch.device(device):
            x = x.to(device=device)
        if x.dtype != dtype and x.is_floating_point():
            x = x.to(dtype=dtype)
        return x
    return torch.tensor(x, dtype=dtype, device=device)


def component_masses_to_chirp_mass(
    mass1: torch.Tensor,
    mass2: torch.Tensor,
) -> torch.Tensor:
    """Calculate chirp mass M_chirp = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)."""
    total_mass = mass1 + mass2
    eta = component_masses_to_eta(mass1, mass2)
    return eta ** (3.0 / 5.0) * total_mass


def component_masses_to_eta(
    mass1: torch.Tensor,
    mass2: torch.Tensor,
) -> torch.Tensor:
    """Calculate symmetric mass ratio eta = (m1 * m2) / (m1 + m2)^2."""
    total_mass = mass1 + mass2
    return (mass1 * mass2) / (total_mass * total_mass)


def component_masses_to_delta(
    mass1: torch.Tensor,
    mass2: torch.Tensor,
) -> torch.Tensor:
    """Calculate mass difference ratio delta = (m1 - m2) / (m1 + m2) = sqrt(1 - 4*eta).

    Assumes m1 >= m2.
    """
    total_mass = mass1 + mass2
    return (mass1 - mass2) / total_mass


def spin_combinations(
    mass1: torch.Tensor,
    mass2: torch.Tensor,
    chi1: torch.Tensor,
    chi2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute symmetric, antisymmetric, PN effective spin, and reduced spin.

    References:
    - Bohé et al. (2013) CQG 30, 135009 (arXiv:1303.7412)
    - Khan et al. (2016) PRD 93, 044007 (arXiv:1508.07253, Eqs. 5-7)

    Returns:
        chi_s: Symmetric spin (chi1 + chi2) / 2
        chi_a: Antisymmetric spin (chi1 - chi2) / 2
        chi_pn: PN effective spin chi_s * (1 - 76/113 * eta) + delta * chi_a
        chi_hat: Reduced spin parameter chi_pn / (1 - 76/113 * eta)
    """
    eta = component_masses_to_eta(mass1, mass2)
    delta = component_masses_to_delta(mass1, mass2)

    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)

    eta_factor = 1.0 - (76.0 / 113.0) * eta
    chi_pn = chi_s * eta_factor + delta * chi_a
    chi_hat = chi_pn / eta_factor

    return chi_s, chi_a, chi_pn, chi_hat


def total_spin_for_remnant(
    mass1: torch.Tensor,
    mass2: torch.Tensor,
    chi1: torch.Tensor,
    chi2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute total dimensionless spin S and rescaled spin S_hat for remnant fits.

    Reference: Husa et al. (2016) PRD 93, 044006 (arXiv:1508.07250, Sec. III B)
    S = chi1 * (m1/M)^2 + chi2 * (m2/M)^2
    S_hat = S / (1 - 2*eta)
    """
    total_mass = mass1 + mass2
    eta = component_masses_to_eta(mass1, mass2)

    m1_ratio = mass1 / total_mass
    m2_ratio = mass2 / total_mass
    s = chi1 * (m1_ratio * m1_ratio) + chi2 * (m2_ratio * m2_ratio)
    s_hat = s / (1.0 - 2.0 * eta)

    return s, s_hat


def tidal_combinations(
    mass1: torch.Tensor,
    mass2: torch.Tensor,
    lambda1: torch.Tensor,
    lambda2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute leading-order and next-to-leading order dimensionless tidal deformabilities.

    References:
    - Flanagan & Hinderer (2008) PRD 77, 021502 (arXiv:0709.1915)
    - Vines, Flanagan, Hinderer (2011) PRD 83, 084051 (arXiv:1101.1673)
    - Wade et al. (2014) PRD 89, 103012 (arXiv:1402.5156, Eqs. 5-6)

    Returns:
        lambda_tilde: Leading order (5PN) tidal deformability parameter
        delta_lambda_tilde: Next-to-leading order (6PN) tidal deformability parameter
    """
    eta = component_masses_to_eta(mass1, mass2)
    delta = component_masses_to_delta(mass1, mass2)

    l_sum = lambda1 + lambda2
    l_diff = lambda1 - lambda2

    eta2 = eta * eta
    eta3 = eta2 * eta

    # Lambda_tilde (Eq. 5 of Wade et al. 2014)
    lambda_tilde = (8.0 / 13.0) * (
        (1.0 + 7.0 * eta - 31.0 * eta2) * l_sum
        + delta * (1.0 + 9.0 * eta - 11.0 * eta2) * l_diff
    )

    # delta_Lambda_tilde (Eq. 6 of Wade et al. 2014)
    delta_lambda_tilde = 0.5 * (
        delta * (1.0 - (13272.0 / 1319.0) * eta + (8944.0 / 1319.0) * eta2) * l_sum
        + (1.0 - (15910.0 / 1319.0) * eta + (32850.0 / 1319.0) * eta2 + (3380.0 / 1319.0) * eta3) * l_diff
    )

    return lambda_tilde, delta_lambda_tilde


def kerr_isco_radius(chi: torch.Tensor) -> torch.Tensor:
    """Compute equatorial innermost stable circular orbit (ISCO) radius in units of M.

    Formula from Bardeen, Press, Teukolsky (1972) Eq. 2.10.
    Includes Taylor expansion around chi=0 to preserve C1 autograd continuity without NaN gradients.

    Args:
        chi: Dimensionless Kerr spin parameter (-1 <= chi <= 1), where chi > 0 is prograde.
    """
    chi_clamped = torch.clamp(chi, -0.9999, 0.9999)
    abs_chi = torch.abs(chi_clamped)

    # General Bardeen et al. formula
    one_minus_chi2 = torch.clamp(1.0 - chi_clamped * chi_clamped, min=1e-8)
    term_cubed = (
        torch.pow(torch.clamp(1.0 + chi_clamped, min=1e-8), 1.0 / 3.0)
        + torch.pow(torch.clamp(1.0 - chi_clamped, min=1e-8), 1.0 / 3.0)
    )
    z1 = 1.0 + torch.pow(one_minus_chi2, 1.0 / 3.0) * term_cubed
    z2 = torch.sqrt(3.0 * chi_clamped * chi_clamped + z1 * z1)

    radicand = torch.clamp((3.0 - z1) * (3.0 + z1 + 2.0 * z2), min=0.0)
    safe_sqrt = torch.sqrt(radicand + 1e-18)
    r_general = 3.0 + z2 - torch.sign(chi_clamped) * safe_sqrt

    # Linear expansion around chi=0: r(chi) = 6 - 4*sqrt(2/3)*chi
    # Eliminates the sqrt(0) singular branch cut at chi=0 for autograd
    c1 = 4.0 * math.sqrt(2.0 / 3.0)
    r_taylor = 6.0 - c1 * chi_clamped

    return torch.where(abs_chi < 1e-4, r_taylor, r_general)


def kerr_isco_frequency(total_mass: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    """Compute gravitational-wave quadrupole frequency at equatorial Kerr ISCO in Hz.

    f_GW,ISCO = c^3 / (pi * G * M * ((r_ISCO/M)^(3/2) + chi))

    Args:
        total_mass: Total binary mass in solar masses
        chi: Effective dimensionless spin parameter (-1 <= chi <= 1)
    """
    m_sec = total_mass * MSUN_SEC
    r_isco = kerr_isco_radius(chi)
    omega = 1.0 / (torch.pow(r_isco, 1.5) + chi)
    return omega / (PI * m_sec)

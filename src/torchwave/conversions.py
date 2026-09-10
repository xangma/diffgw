"""Kinematic and parameter conversions for gravitational-wave astrophysics.

All conversions strictly implement peer-reviewed formulations:
- Symmetric mass ratio and PN velocity: Blanchet (2014), Living Rev. Relativity 17, 2 (arXiv:1310.1528)
- Spin combinations: Bohé et al. (2013), CQG 30, 135009 (arXiv:1303.7412); Khan et al. (2016), PRD 93, 044007 (arXiv:1508.07253)
- Tidal parameter combinations: Flanagan & Hinderer (2008), PRD 77, 021502; Vines et al. (2011), PRD 83, 084051; Wade et al. (2014), PRD 89, 103012.
"""

from typing import Tuple, Union
import torch


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

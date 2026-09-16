"""Kinematic and parameter conversions for gravitational-wave astrophysics.

Pure Python float implementations. Zero external framework dependencies.
All conversions strictly implement peer-reviewed formulations:
- Symmetric mass ratio and PN velocity: Blanchet (2014), Living Rev. Relativity 17, 2 (arXiv:1310.1528)
- Spin combinations: Bohé et al. (2013), CQG 30, 135009 (arXiv:1303.7412); Khan et al. (2016), PRD 93, 044007 (arXiv:1508.07253)
"""

import math
from typing import Tuple


def component_masses_to_chirp_mass(mass1: float, mass2: float) -> float:
    """Calculate chirp mass M_chirp = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)."""
    total_mass = mass1 + mass2
    eta = component_masses_to_eta(mass1, mass2)
    return (eta ** (3.0 / 5.0)) * total_mass


def component_masses_to_eta(mass1: float, mass2: float) -> float:
    """Calculate symmetric mass ratio eta = (m1 * m2) / (m1 + m2)^2."""
    total_mass = mass1 + mass2
    return (mass1 * mass2) / (total_mass * total_mass)


def component_masses_to_delta(mass1: float, mass2: float) -> float:
    """Calculate mass difference ratio delta = (m1 - m2) / (m1 + m2) = sqrt(1 - 4*eta).

    Assumes mass1 >= mass2.
    """
    total_mass = mass1 + mass2
    return (mass1 - mass2) / total_mass


def spin_combinations(
    mass1: float,
    mass2: float,
    chi1: float,
    chi2: float,
) -> Tuple[float, float, float, float]:
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
    mass1: float,
    mass2: float,
    chi1: float,
    chi2: float,
) -> Tuple[float, float]:
    """Compute total dimensionless spin S and rescaled spin S_hat for remnant fits."""
    m1_sq = mass1 * mass1
    m2_sq = mass2 * mass2
    m_tot = mass1 + mass2
    m_tot_sq = m_tot * m_tot

    s = (m1_sq * chi1 + m2_sq * chi2) / m_tot_sq
    eta = component_masses_to_eta(mass1, mass2)
    s_hat = s / (1.0 - 2.0 * eta)

    return s, s_hat

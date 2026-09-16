"""Kinematic and parameter conversions for gravitational-wave astrophysics in JAX.

All conversions strictly implement peer-reviewed formulations:
- Symmetric mass ratio and PN velocity: Blanchet (2014), Living Rev. Relativity 17, 2 (arXiv:1310.1528)
- Spin combinations: Bohé et al. (2013), CQG 30, 135009 (arXiv:1303.7412); Khan et al. (2016), PRD 93, 044007 (arXiv:1508.07253)
- Tidal combinations: Flanagan & Hinderer (2008), PRD 77, 021502; Vines et al. (2011), PRD 83, 084051; Wade et al. (2014), PRD 89, 103012.
"""

import math
from typing import Tuple, Union
import jax.numpy as jnp
from diffgw_core.constants import MSUN_SEC, PI


def as_array(x: Union[float, int, jnp.ndarray], dtype=jnp.float64) -> jnp.ndarray:
    """Ensure input is converted to a JAX Array with specified dtype."""
    return jnp.asarray(x, dtype=dtype)


def component_masses_to_chirp_mass(mass1: jnp.ndarray, mass2: jnp.ndarray) -> jnp.ndarray:
    """Calculate chirp mass M_chirp = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)."""
    total_mass = mass1 + mass2
    eta = component_masses_to_eta(mass1, mass2)
    return (eta ** (3.0 / 5.0)) * total_mass


def component_masses_to_eta(mass1: jnp.ndarray, mass2: jnp.ndarray) -> jnp.ndarray:
    """Calculate symmetric mass ratio eta = (m1 * m2) / (m1 + m2)^2."""
    total_mass = mass1 + mass2
    return (mass1 * mass2) / (total_mass * total_mass)


def component_masses_to_delta(mass1: jnp.ndarray, mass2: jnp.ndarray) -> jnp.ndarray:
    """Calculate mass difference ratio delta = (m1 - m2) / (m1 + m2) = sqrt(1 - 4*eta).

    Assumes mass1 >= mass2.
    """
    total_mass = mass1 + mass2
    return (mass1 - mass2) / total_mass


def spin_combinations(
    mass1: jnp.ndarray,
    mass2: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute symmetric, antisymmetric, PN effective spin, and reduced spin.

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
    mass1: jnp.ndarray,
    mass2: jnp.ndarray,
    chi1: jnp.ndarray,
    chi2: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute total dimensionless spin S and rescaled spin S_hat for remnant fits."""
    m1_sq = mass1 * mass1
    m2_sq = mass2 * mass2
    m_tot = mass1 + mass2
    m_tot_sq = m_tot * m_tot

    s = (m1_sq * chi1 + m2_sq * chi2) / m_tot_sq
    eta = component_masses_to_eta(mass1, mass2)
    s_hat = s / (1.0 - 2.0 * eta)

    return s, s_hat


def tidal_combinations(
    mass1: jnp.ndarray,
    mass2: jnp.ndarray,
    lambda1: jnp.ndarray,
    lambda2: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate dimensionless tidal deformability combinations Lambda_tilde and delta_Lambda_tilde."""
    total_mass = mass1 + mass2
    eta = component_masses_to_eta(mass1, mass2)
    m1 = mass1 / total_mass
    m2 = mass2 / total_mass
    delta = component_masses_to_delta(mass1, mass2)

    eta2 = eta * eta

    # Flanagan & Hinderer 2008, Favata 2014, Wade et al. 2014 Eq. (5)
    lambda_tilde = (8.0 / 13.0) * (
        (1.0 + 7.0 * eta - 31.0 * eta2) * (lambda1 + lambda2)
        + delta * (1.0 + 9.0 * eta - 11.0 * eta2) * (lambda1 - lambda2)
    )

    # Wade et al. 2014 Eq. (6)
    delta_lambda_tilde = 0.5 * (
        delta * (1.0 - 13272.0 * eta / 1319.0 + 8944.0 * eta2 / 1319.0) * (lambda1 + lambda2)
        + (1.0 - 15910.0 * eta / 1319.0 + 32850.0 * eta2 / 1319.0 + 3380.0 * (eta2 * eta) / 1319.0) * (lambda1 - lambda2)
    )

    return lambda_tilde, delta_lambda_tilde


def kerr_isco_radius(chi: jnp.ndarray) -> jnp.ndarray:
    """Compute equatorial innermost stable circular orbit (ISCO) radius in units of M.

    Formula from Bardeen, Press, Teukolsky (1972) Eq. 2.10.
    Includes Taylor expansion around chi=0 to preserve C1 autograd continuity without NaN gradients.

    Args:
        chi: Dimensionless Kerr spin parameter (-1 <= chi <= 1), where chi > 0 is prograde.
    """
    chi_arr = as_array(chi, dtype=jnp.float64)
    chi_clamped = jnp.clip(chi_arr, -0.9999, 0.9999)
    abs_chi = jnp.abs(chi_clamped)

    safe_chi = jnp.where(abs_chi < 1e-4, 0.5, chi_clamped)
    one_minus_chi2 = jnp.maximum(1.0 - safe_chi * safe_chi, 1e-8)
    term_cubed = (
        jnp.maximum(1.0 + safe_chi, 1e-8) ** (1.0 / 3.0)
        + jnp.maximum(1.0 - safe_chi, 1e-8) ** (1.0 / 3.0)
    )
    z1 = 1.0 + (one_minus_chi2 ** (1.0 / 3.0)) * term_cubed
    z2 = jnp.sqrt(3.0 * safe_chi * safe_chi + z1 * z1)

    radicand = jnp.maximum((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0)
    safe_sqrt = jnp.sqrt(radicand + 1e-18)
    r_general = 3.0 + z2 - jnp.sign(safe_chi) * safe_sqrt

    c1 = 4.0 * math.sqrt(2.0 / 3.0)
    r_taylor = 6.0 - c1 * chi_clamped

    return jnp.where(abs_chi < 1e-4, r_taylor, r_general)


def kerr_isco_frequency(total_mass_msun: jnp.ndarray, final_spin: jnp.ndarray) -> jnp.ndarray:
    """Compute gravitational-wave quadrupole frequency at equatorial Kerr ISCO in Hz.

    f_GW,ISCO = c^3 / (pi * G * M * ((r_ISCO/M)^(3/2) + chi))

    Args:
        total_mass_msun: Total binary mass in solar masses
        final_spin: Effective dimensionless spin parameter (-1 <= chi <= 1)
    """
    m_sec = total_mass_msun * MSUN_SEC
    r_isco = kerr_isco_radius(final_spin)
    omega = 1.0 / (r_isco ** 1.5 + final_spin)
    f_isco = omega / (PI * m_sec)
    return f_isco


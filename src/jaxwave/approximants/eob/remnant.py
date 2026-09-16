"""Published aligned-spin remnant inputs in JAX.

Mass: Ossokine et al. (2020), PRD 102, 044055, Eq. (3.5).
Spin: Hofmann, Barausse & Rezzolla, arXiv:1605.01938v2, Eq. (7),
Table 1 and Eq. (11).
"""

from typing import Tuple, Union
import math
import jax
import jax.numpy as jnp

from diffgw_core.remnant import _HOFMANN
from jaxwave.conversions import as_array


def _kerr_isco(spin: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """(r/M, E, L/M) for positive orbital L and signed Kerr spin."""
    spin = as_array(spin, dtype=jnp.float64)
    u = (1.0 + spin) ** (1.0 / 3.0)
    v = (1.0 - spin) ** (1.0 / 3.0)
    t = u * v
    s = u + v
    d = u**2 + t + v**2
    z1 = 1.0 + t * s
    z2 = jnp.sqrt(3.0 * spin**2 + z1**2)
    ratio = (s + 4.0 / (2.0 + s)) / (1.0 + t + t**2) - 4.0 / ((2.0 + s) * d**2)
    radius = 3.0 + z2 - spin * jnp.sqrt(ratio * (3.0 + z1 + 2.0 * z2))
    radius = jnp.where(spin == -1.0, 9.0, radius)
    radius = jnp.where(spin == 0.0, 6.0, radius)
    radius = jnp.where(spin == 1.0, 1.0, radius)
    energy = jnp.sqrt(1.0 - 2.0 / (3.0 * radius))
    angular_momentum = 2.0 / (3.0 * math.sqrt(3.0)) * (1.0 + 2.0 * jnp.sqrt(3.0 * radius - 2.0))
    return radius, energy, angular_momentum


def _parameters(
    mass1: Union[float, jnp.ndarray],
    mass2: Union[float, jnp.ndarray],
    chi1: Union[float, jnp.ndarray],
    chi2: Union[float, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    m1 = as_array(mass1, dtype=jnp.float64)
    m2 = as_array(mass2, dtype=jnp.float64)
    s1 = as_array(chi1, dtype=jnp.float64)
    s2 = as_array(chi2, dtype=jnp.float64)
    m1, m2, s1, s2 = jnp.broadcast_arrays(m1, m2, s1, s2)

    scale = jnp.maximum(m1, m2)
    m1_s, m2_s = m1 / scale, m2 / scale
    denom = m1_s + m2_s
    nu = (m1_s / denom) * (m2_s / denom)
    weighted_spin = (m1_s**2 * s1 + m2_s**2 * s2) / (m1_s**2 + m2_s**2)
    total_spin = (1.0 - 2.0 * nu) * weighted_spin
    return nu, total_spin, weighted_spin, s1, s2


def eob_final_mass_fraction(
    mass1: Union[float, jnp.ndarray],
    mass2: Union[float, jnp.ndarray],
    chi1: Union[float, jnp.ndarray],
    chi2: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Return M_final/(mass1+mass2) from Ossokine Eq. (3.5)."""
    nu, total_spin, weighted_spin, _, _ = _parameters(mass1, mass2, chi1, chi2)
    _, energy, _ = _kerr_isco(total_spin)
    equal_mass_radiation = 0.00258 - 0.0773 / (weighted_spin - 1.6939)
    return 1.0 - nu * (1.0 - 4.0 * nu) * (1.0 - energy) - 16.0 * nu**2 * equal_mass_radiation


def hofmann_final_spin(
    mass1: Union[float, jnp.ndarray],
    mass2: Union[float, jnp.ndarray],
    chi1: Union[float, jnp.ndarray],
    chi2: Union[float, jnp.ndarray],
    *,
    fit_order: Tuple[int, int] = (3, 4),
) -> jnp.ndarray:
    """Return the signed final-spin projection from Hofmann Eq. (7)."""
    if fit_order not in _HOFMANN:
        raise ValueError("fit_order must be one of (1,2), (3,3), or (3,4)")
    xi, coefficients = _HOFMANN[fit_order]
    nu, total_spin, _, s1, s2 = _parameters(mass1, mass2, chi1, chi2)
    effective_spin = total_spin + xi * nu * (s1 + s2)
    _, energy, angular_momentum = _kerr_isco(effective_spin)

    k00 = 16.0 * (
        0.68646
        - math.sqrt(3.0) / 2.0
        - sum(row[0] / 4.0 ** (2 + i) for i, row in enumerate(coefficients) if i)
    )
    polynomial = jnp.zeros_like(effective_spin)
    for i in range(len(coefficients) - 1, -1, -1):
        row_value = jnp.zeros_like(effective_spin)
        for j in range(len(coefficients[i]) - 1, -1, -1):
            coefficient = k00 if (i, j) == (0, 0) else coefficients[i][j]
            row_value = row_value * effective_spin + coefficient
        polynomial = polynomial * nu + row_value
    return (
        total_spin
        + nu * (angular_momentum - 2.0 * total_spin * (energy - 1.0))
        + nu**2 * polynomial
    )


__all__ = [
    "eob_final_mass_fraction",
    "hofmann_final_spin",
]

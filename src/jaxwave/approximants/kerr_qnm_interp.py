"""Signed-spin 220 frequencies from Kerr equation solves in JAX.

Literature-based reconstruction matching TorchWave independent Kerr QNM evaluation.
"""

from typing import Tuple, Union
import jax.numpy as jnp

from diffgw_core.kerr_qnm_220_data import (
    DEGREE,
    KNOTS_220,
    COEFFS_220,
    KNOTS_2M20,
    COEFFS_2M20,
)
from jaxwave.conversions import as_array

_KNOTS_220 = jnp.array(KNOTS_220, dtype=jnp.float64)
_COEFFS_220 = jnp.array(COEFFS_220, dtype=jnp.float64)
_KNOTS_2M20 = jnp.array(KNOTS_2M20, dtype=jnp.float64)
_COEFFS_2M20 = jnp.array(COEFFS_2M20, dtype=jnp.float64)


def _dimensionless_qnm(
    x: jnp.ndarray,
    knots: jnp.ndarray,
    coefficients: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    interval = jnp.clip(
        jnp.searchsorted(knots, x, side="right") - 1,
        DEGREE,
        len(coefficients) - 1,
    )
    values = [coefficients[interval - DEGREE + j] for j in range(DEGREE + 1)]
    for r in range(1, DEGREE + 1):
        for j in range(DEGREE, r - 1, -1):
            left = knots[interval - DEGREE + j]
            right = knots[interval + 1 + j - r]
            alpha = jnp.expand_dims((x - left) / (right - left), -1)
            values[j] = (1.0 - alpha) * values[j - 1] + alpha * values[j]
    result = values[DEGREE]
    return result[..., 0], -result[..., 1]


def evaluate_kerr_qnm_220(
    final_spin: Union[float, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return Re(M_f*omega) and -Im(M_f*omega) using independent B-splines in JAX.

    Negative spin uses the positive-spin m=-2 branch by Kerr symmetry. Inputs
    outside [-0.9999, 0.9999] retain endpoint-clamping behavior.
    """
    spin = as_array(final_spin, dtype=jnp.float64)
    spin_clamped = jnp.clip(spin, -0.9999, 0.9999)
    return _evaluate_kerr_qnm_220_at_spin(spin_clamped)


def _evaluate_kerr_qnm_220_at_spin(
    spin: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate generated roots for signed spins within the generator's domain."""
    pos_spin = jnp.where(spin >= 0.0, spin, 0.0)
    neg_spin = jnp.where(spin < 0.0, -spin, 0.0)
    pos_x = -jnp.log1p(-pos_spin)
    neg_x = -jnp.log1p(-neg_spin)
    p_re, p_im = _dimensionless_qnm(pos_x, _KNOTS_220, _COEFFS_220)
    n_re, n_im = _dimensionless_qnm(neg_x, _KNOTS_2M20, _COEFFS_2M20)
    re = jnp.where(spin >= 0.0, p_re, n_re)
    im = jnp.where(spin >= 0.0, p_im, n_im)
    return re, im


__all__ = ["evaluate_kerr_qnm_220"]

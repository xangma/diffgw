"""Opt-in six-significant-digit natural-spline QNM compatibility convention in JAX.

Port of torchwave.approximants.kerr_qnm_table_convention for JAX.
"""

from typing import Tuple, Union
import jax.numpy as jnp

from diffgw_core.kerr_qnm_table_convention_data import (
    KNOTS,
    COEFF_A,
    COEFF_B,
    COEFF_C,
    COEFF_D,
    Y_ROUNDED,
)
from jaxwave.conversions import as_array

_KNOTS = jnp.array(KNOTS, dtype=jnp.float64)
_COEFF_A = jnp.array(COEFF_A, dtype=jnp.float64)
_COEFF_B = jnp.array(COEFF_B, dtype=jnp.float64)
_COEFF_C = jnp.array(COEFF_C, dtype=jnp.float64)
_COEFF_D = jnp.array(COEFF_D, dtype=jnp.float64)
_Y_ROUNDED = jnp.array(Y_ROUNDED, dtype=jnp.float64)


def signed_qnm_220_table_convention(
    spin: Union[float, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the opt-in rounded-knot natural cubic QNM convention in JAX.

    Returns ``(M_final*omega_real, M_final*|omega_imag|)``.
    """
    x = as_array(spin, dtype=jnp.float64)
    orig_shape = x.shape
    work = jnp.reshape(x, -1)
    idx = jnp.clip(
        jnp.searchsorted(_KNOTS, work, side="right") - 1,
        0,
        len(_KNOTS) - 2,
    )
    dx = work - _KNOTS[idx]
    dx_col = dx[:, None]
    out = (
        _COEFF_A[idx]
        + _COEFF_B[idx] * dx_col
        + _COEFF_C[idx] * (dx_col**2)
        + _COEFF_D[idx] * (dx_col**3)
    )
    out = out.reshape(orig_shape + (2,))
    return out[..., 0], out[..., 1]


def rounded_knot_values() -> jnp.ndarray:
    """Return regenerated rounded values, for provenance and validation."""
    return _Y_ROUNDED


__all__ = [
    "signed_qnm_220_table_convention",
    "rounded_knot_values",
    "KNOTS",
]

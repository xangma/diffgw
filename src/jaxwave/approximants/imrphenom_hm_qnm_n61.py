"""JAX evaluation of the independent N=61 London higher-mode QNM fit.

Frozen coefficients, provenance metadata, and their payload hash live in
``diffgw_core.hm_qnm_n61_data`` and remain re-exported here for compatibility.
Evaluation uses JAX arrays and operations.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import jax.numpy as jnp
from jaxwave.conversions import as_array


from diffgw_core.hm_qnm_n61_data import (
    N61_ARTIFACT_FORMAT,
    N61_COEFFICIENT_PAYLOAD_SHA256,
    N61_COEFFICIENTS,
    N61_COUNT,
    N61_FIT_STRATEGY,
    N61_GENERATION_ARTIFACT,
    N61_GRID_DEFINITION,
    N61_SPIN_LIMIT,
)


_ZERO_DAMPED = frozenset(((2, 2), (3, 3), (4, 4), (4, 3)))


def _horner(coeffs: Tuple[float, ...] | list[float], x: jnp.ndarray) -> jnp.ndarray:
    result = coeffs[-1]
    for coeff in reversed(coeffs[:-1]):
        result = result * x + coeff
    return result


def evaluate_n61_qnm(
    mode: Tuple[int, int] | Tuple[int, int, int],
    final_spin: Union[float, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate one frozen N=61 fit in final-mass units in JAX.

    Returns ``(M_f omega_RD, M_f/tau)``.  The fitted complex frequency follows
    London's ``omega=omega_r+i/tau`` convention; the second component is the
    positive angular damping magnitude ``M_f/tau``.  The caller performs the
    ``1/(2*pi)`` conversion to cyclic frequency and the total-to-final-mass
    conversion.
    """
    lm = (int(mode[0]), int(mode[1]))
    if lm not in N61_COEFFICIENTS:
        raise ValueError(f"unsupported N=61 HM QNM mode {mode!r}")
    spin = as_array(final_spin, dtype=jnp.float64)

    # Equation (21) of London et al.
    kappa = jnp.log(2.0 - spin) / math.log(3.0)
    exponent = 1.0 / float(2 + lm[0] - abs(lm[1]))
    kappa = jnp.power(kappa, exponent)

    raw = N61_COEFFICIENTS[lm]
    real_coeffs = [c[0] for c in raw]
    imag_coeffs = [c[1] for c in raw]

    real = _horner(real_coeffs, kappa)
    imag = _horner(imag_coeffs, kappa)

    if lm in _ZERO_DAMPED:
        omega_real = float(lm[1]) / 2.0 + kappa * real
        omega_imag = kappa * imag
    else:
        omega_real = real
        omega_imag = imag
    return omega_real, jnp.abs(omega_imag)


__all__ = [
    "N61_COUNT",
    "N61_SPIN_LIMIT",
    "N61_GRID_DEFINITION",
    "N61_FIT_STRATEGY",
    "N61_ARTIFACT_FORMAT",
    "N61_COEFFICIENTS",
    "N61_COEFFICIENT_PAYLOAD_SHA256",
    "N61_GENERATION_ARTIFACT",
    "evaluate_n61_qnm",
]

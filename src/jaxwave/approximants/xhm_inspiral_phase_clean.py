# SPDX-License-Identifier: MIT
"""Paper-derived IMRPhenomXHM inspiral carrier phase primitives in JAX."""

from __future__ import annotations

import math

import jax.numpy as jnp


COMPONENT_VERSION = "xhm.phase.inspiral.carrier-rescaling.v1"


def _validate_mode(mode_m: int) -> None:
    if isinstance(mode_m, bool) or not isinstance(mode_m, int) or mode_m <= 0:
        raise ValueError("mode_m must be a positive integer")


def rescale_carrier_phase(carrier_phase_at_scaled_frequency, mode_m: int):
    """Apply XHM v2 Eq. (29) to a 22 phase evaluated at ``2 f / m``."""

    _validate_mode(mode_m)
    return (mode_m / 2.0) * jnp.asarray(carrier_phase_at_scaled_frequency)


def rescale_carrier_phase_coefficients(mode_m: int, indices, c22, d22):
    """Apply XHM v2 Appendix-D Eq. (63) to phase coefficients."""

    _validate_mode(mode_m)
    dtype = jnp.result_type(indices, c22, d22, jnp.float64)
    i, c, d = jnp.broadcast_arrays(
        jnp.asarray(indices, dtype=dtype),
        jnp.asarray(c22, dtype=dtype),
        jnp.asarray(d22, dtype=dtype),
    )
    ratio = mode_m / 2.0
    scale = jnp.power(jnp.asarray(ratio, dtype=c.dtype), (8.0 - i) / 3.0)
    return scale * (c - d * math.log(ratio)), scale * d


__all__ = [
    "COMPONENT_VERSION",
    "rescale_carrier_phase",
    "rescale_carrier_phase_coefficients",
]

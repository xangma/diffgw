# SPDX-License-Identifier: MIT
"""Conditional paper-era IMRPhenomXHM (3,2) four-target phase solve.

The caller supplies all four derivative targets from García-Quirós et al.,
arXiv:2001.10914v2, Eq. (48). This helper is not called by the production
waveform and contains no parameter-space fits.
"""

from __future__ import annotations

import jax.numpy as jnp


def xhm_32_spheroidal_collocation_frequencies(f_ring22, f_ring32, f_damp32):
    """Return the four Table-9 nodes in published order."""

    ring22, ring32, damp32 = jnp.broadcast_arrays(
        jnp.asarray(f_ring22), jnp.asarray(f_ring32), jnp.asarray(f_damp32),
    )
    return jnp.stack((ring22, ring32 - 1.5 * damp32,
                      ring32 - 0.5 * damp32, ring32 + 0.5 * damp32), axis=-1)


def xhm_32_spheroidal_phase_coefficients(
    f_ring22, f_ring32, f_damp32, derivative_targets,
):
    """Return Eq. (46) coefficients from caller-supplied Eq. (48) targets.

    Inputs must have positive, distinct collocation nodes and a nonsingular
    matrix. The order is ``(alpha2,alpha4,alpha_lambda,dphi_RD)``. The
    dimensionless solve limits avoidable column-scaling loss in FP64.
    """

    targets = jnp.asarray(derivative_targets)
    if targets.shape[-1:] != (4,):
        raise ValueError("derivative_targets must have four values on its last axis")
    ring22, ring32, damp32, _ = jnp.broadcast_arrays(
        jnp.asarray(f_ring22), jnp.asarray(f_ring32),
        jnp.asarray(f_damp32), targets[..., 0],
    )
    targets = jnp.broadcast_to(targets, ring22.shape + (4,))
    nodes = xhm_32_spheroidal_collocation_frequencies(ring22, ring32, damp32)
    x = nodes / ring32[..., None]
    delta = damp32 / ring32
    lorentz = delta[..., None] / (delta[..., None]**2 + (x - 1)**2)
    matrix = jnp.stack((x**-2, x**-4, lorentz, jnp.ones_like(x)), axis=-1)
    scaled = jnp.linalg.solve(matrix, targets[..., None])[..., 0]
    scales = jnp.stack((ring32**2, ring32**4, ring32, jnp.ones_like(ring32)), axis=-1)
    return scaled * scales

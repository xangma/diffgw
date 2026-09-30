"""JAX float64 conditioning diagnostic for the generic MSA spin cubic.

The raw inverse-cosine argument is returned before endpoint handling.  The
separately labelled compatibility roots use v223's current endpoint exclusion,
making float64 branch loss visible rather than treating that result as the
published cubic.  This module does not participate in waveform dispatch.
"""

from __future__ import annotations

import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC


def generic_cubic_boundary_diagnostic(mass1, mass2, spin1, spin2, f_ref):
    """Return raw cubic conditioning and compatibility roots in float64."""
    m1, m2, fr = jnp.broadcast_arrays(*(
        jnp.asarray(value, dtype=jnp.float64) for value in (mass1, mass2, f_ref)
    ))
    chi1 = jnp.asarray(spin1, dtype=jnp.float64)
    chi2 = jnp.asarray(spin2, dtype=jnp.float64)
    if chi1.shape[-1] != 3 or chi2.shape[-1] != 3:
        raise ValueError("spin vectors must end in dimension 3")
    swap = m2 > m1
    cm1, cm2 = jnp.where(swap, m2, m1), jnp.where(swap, m1, m2)
    cs1 = jnp.where(swap[..., None], chi2, chi1)
    cs2 = jnp.where(swap[..., None], chi1, chi2)
    total = cm1 + cm2
    n1, n2 = cm1 / total, cm2 / total
    eta, q, delta = n1 * n2, n2 / n1, n1 - n2
    s1, s2 = n1[..., None] ** 2 * cs1, n2[..., None] ** 2 * cs2
    s1sq, s2sq = jnp.sum(s1**2, -1), jnp.sum(s2**2, -1)
    spin = s1 + s2
    spin_squared = jnp.sum(spin**2, -1)
    xi = (1 + q) * s1[..., 2] + (1 + 1 / q) * s2[..., 2]
    velocity = (jnp.pi * total * MSUN_SEC * fr) ** (1 / 3)
    orbital = eta / velocity
    angular_momentum = spin + jnp.stack(
        (jnp.zeros_like(orbital), jnp.zeros_like(orbital), orbital), -1
    )
    j = jnp.linalg.norm(angular_momentum, axis=-1)
    j2, l2 = j**2, orbital**2
    b = ((l2 + s1sq) * q + 2 * orbital * xi - 2 * j2 - s1sq - s2sq
         + (l2 + s2sq) / q)
    c = ((j2 - l2) ** 2 - 2 * orbital * xi * (j2 - l2)
         - 2 * (1 - q) / q * (s1sq - q * s2sq) * l2
         + 4 * eta * l2 * xi**2
         - 2 * delta * (s1sq - s2sq) * xi * orbital
         + 2 * (1 - q) / q * (q * s1sq - s2sq) * j2)
    d = ((1 - q) / q * (s2sq - q * s1sq) * (j2 - l2) ** 2
         + delta**2 / eta * (s1sq - s2sq) ** 2 * l2
         + 2 * delta * orbital * xi * (s1sq - s2sq) * (j2 - l2))
    p = c - b**2 / 3
    depressed_q = 2 * b**3 / 27 - b * c / 3 + d
    raw_argument = 3 * depressed_q / (2 * p) * jnp.sqrt(-3 / p)
    argument_used = jnp.clip(raw_argument, -1.0 + 1e-15, 1.0 - 1e-15)
    theta = jnp.arccos(argument_used) / 3
    k = jnp.arange(3, dtype=jnp.float64)
    candidates = (-b[..., None] / 3 + 2 * jnp.sqrt(-p[..., None] / 3)
                  * jnp.cos(theta[..., None] - 2 * jnp.pi * k / 3))
    roots = jnp.sort(candidates, axis=-1)
    gap = roots[..., 2] - roots[..., 1]
    fraction = (roots[..., 2] - spin_squared) / gap
    return {
        "coefficients": jnp.stack((b, c, d), -1),
        "depressed_coefficients": jnp.stack((p, depressed_q), -1),
        "acos_argument": raw_argument,
        "acos_argument_used": argument_used,
        "compatibility_roots": roots,
        "compatibility_root_gap": gap,
        "total_spin_squared": spin_squared,
        "phase_fraction_unclipped": fraction,
    }


__all__ = ["generic_cubic_boundary_diagnostic"]

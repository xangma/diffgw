# SPDX-License-Identifier: MIT
"""JAX inverse-polynomial XHM intermediate amplitude for supplied targets.

Implements García-Quirós et al., arXiv:2001.10914v2, Eq. (37).  This
transformable algebra requires caller-validated positive amplitudes, ordered
join endpoints, and an in-interval frequency grid.  It contains no fitted
surfaces, mode vetoes, release-122022 mapping, or pole certificate.
"""

from __future__ import annotations

import jax.numpy as jnp


COMPONENT_VERSION = "xhm.amplitude.intermediate.inverse-polynomial.v1"


def solve_intermediate_inverse_polynomial(
    start_value, start_derivative, interior_values, end_value, end_derivative,
    *, interior_fractions=(1.0 / 3.0, 2.0 / 3.0),
):
    """Solve for coefficients of ``P(u)`` from two C1 joins and two nodes."""
    if (len(interior_fractions) != 2 or
            not all(0.0 < float(x) < 1.0 for x in interior_fractions) or
            float(interior_fractions[0]) == float(interior_fractions[1])):
        raise ValueError("two distinct interior fractions in (0,1) required")
    interior = jnp.asarray(interior_values)
    if interior.shape[-1:] != (2,):
        raise ValueError("interior_values must have trailing shape (2,)")
    dtype = jnp.result_type(start_value, start_derivative, interior,
                            end_value, end_derivative, jnp.float64)
    s, ds, e, de, i1, i2 = jnp.broadcast_arrays(*(
        jnp.asarray(x, dtype=dtype) for x in
        (start_value, start_derivative, end_value, end_derivative,
         interior[..., 0], interior[..., 1])
    ))
    zero, one = jnp.zeros_like(s), jnp.ones_like(s)
    def row(values):
        return jnp.stack(values, axis=-1)
    matrix = jnp.stack((
        row((one, zero, zero, zero, zero, zero)),
        row((zero, one, zero, zero, zero, zero)),
        *(row(tuple(one * u**k for k in range(6))) for u in interior_fractions),
        row((one, one, one, one, one, one)),
        row((zero, one, 2*one, 3*one, 4*one, 5*one)),
    ), axis=-2)
    rhs = jnp.stack((s, ds, i1, i2, e, de), axis=-1)
    return jnp.linalg.solve(matrix, rhs[..., None])[..., 0]


def solve_intermediate_normalized_amplitude(
    start_amplitude, start_derivative_df, interior_amplitudes,
    end_amplitude, end_derivative_df, f_start, f_end,
    *, interior_fractions=(1.0 / 3.0, 2.0 / 3.0),
):
    """Convert supplied ``v=A/A0`` and ``dv/df`` into coefficients of ``1/v``."""
    start, end = jnp.asarray(start_amplitude), jnp.asarray(end_amplitude)
    width = jnp.asarray(f_end) - jnp.asarray(f_start)
    return solve_intermediate_inverse_polynomial(
        1 / start, -width * jnp.asarray(start_derivative_df) / start**2,
        1 / jnp.asarray(interior_amplitudes), 1 / end,
        -width * jnp.asarray(end_derivative_df) / end**2,
        interior_fractions=interior_fractions,
    )


def evaluate_intermediate_inverse_polynomial(u, coefficients):
    """Evaluate caller-certified positive ``P(u)`` by Horner's rule."""
    c = jnp.asarray(coefficients)
    if c.shape[-1:] != (6,):
        raise ValueError("coefficients must have trailing shape (6,)")
    x = jnp.asarray(u)
    shape = jnp.broadcast_shapes(x.shape, c.shape[:-1])
    x = jnp.broadcast_to(x, shape)
    c = jnp.broadcast_to(c, shape + (6,))
    result = jnp.zeros_like(x, dtype=jnp.result_type(x, c))
    for k in range(5, -1, -1):
        result = result * x + c[..., k]
    return result


def evaluate_intermediate_amplitude(frequency, f_start, f_end, a0, coefficients):
    """Evaluate ``A0/P`` on a caller-validated join interval."""
    u = (jnp.asarray(frequency) - jnp.asarray(f_start)) / (
        jnp.asarray(f_end) - jnp.asarray(f_start))
    return jnp.asarray(a0) / evaluate_intermediate_inverse_polynomial(u, coefficients)


__all__ = ["COMPONENT_VERSION", "solve_intermediate_inverse_polynomial",
           "solve_intermediate_normalized_amplitude", "evaluate_intermediate_inverse_polynomial",
           "evaluate_intermediate_amplitude"]

"""Continuous event location on accepted samples in JAX."""

import jax.numpy as jnp

from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline
from jaxwave.conversions import as_array


def continuous_spline_peak(time: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """Return the unique interior maximum of a scalar natural cubic spline."""
    time = as_array(time, dtype=jnp.float64)
    values = as_array(values, dtype=jnp.float64)

    if time.ndim != 1 or values.shape != time.shape or time.size < 3:
        raise ValueError("continuous peak needs a scalar series with at least three knots")

    spline = natural_cubic_spline(time, values, scalar_output=True)
    h = time[1:] - time[:-1]
    a, b, c, d = spline.a[:, 0], spline.b[:, 0], spline.c[:, 0], spline.d[:, 0]

    B, C, D = b, 2.0 * c * h, 3.0 * d * (h**2)
    eps = jnp.finfo(time.dtype).eps
    scale = jnp.maximum(jnp.maximum(jnp.abs(B), jnp.abs(C)), jnp.abs(D))
    quadratic = jnp.abs(D) > 8.0 * eps * scale
    discriminant = C**2 - 4.0 * D * B
    sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, jnp.finfo(time.dtype).tiny))
    q = -0.5 * (C + jnp.where(C >= 0, sqrt_disc, -sqrt_disc))
    safe_D = jnp.where(quadratic, D, jnp.ones_like(D))
    safe_q = jnp.where(q != 0, q, jnp.ones_like(q))
    linear = (~quadratic) & (jnp.abs(C) > 8.0 * eps * scale)
    safe_C = jnp.where(linear, C, jnp.ones_like(C))

    roots = jnp.stack([q / safe_D, B / safe_q, -B / safe_C], axis=-1)
    valid_kind = jnp.stack(
        [
            quadratic & (discriminant > 0),
            quadratic & (discriminant > 0) & (q != 0),
            linear,
        ],
        axis=-1,
    )
    valid = valid_kind & (roots >= -32.0 * eps) & (roots <= 1.0 + 32.0 * eps)
    valid = valid & ((C[:, None] + 2.0 * D[:, None] * roots) < -8.0 * eps * scale[:, None])

    u = jnp.clip(jnp.where(valid, roots, jnp.zeros_like(roots)), 0.0, 1.0)
    dx = h[:, None] * u
    heights = a[:, None] + dx * (b[:, None] + dx * (c[:, None] + dx * d[:, None]))
    times = time[:-1, None] + dx
    valid = valid & (times > time[0]) & (times < time[-1])

    if not bool(jnp.any(valid)):
        raise ValueError("continuous peak needs a resolved interior maximum")

    heights_masked = jnp.where(valid, heights, -jnp.inf)
    index = int(jnp.argmax(heights_masked))
    peak = times.reshape(-1)[index]
    height = heights_masked.reshape(-1)[index]

    tolerance = 32.0 * eps * jnp.max(jnp.abs(values))
    if bool(jnp.maximum(values[0], values[-1]) >= height - tolerance):
        raise ValueError("continuous peak maximum is not strictly interior")

    return peak


__all__ = [
    "continuous_spline_peak",
]

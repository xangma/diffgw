"""Resolved trajectory events in JAX, independent of an EOB calibration prescription.

Invalid concrete trajectories raise; compiled invalid rows return NaN values
and a ``-1`` sample index so valid rows in the same batch remain usable.
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from jaxwave.conversions import as_array


class OrbitalFrequencyPeak(NamedTuple):
    time: jnp.ndarray
    frequency: jnp.ndarray
    curvature: jnp.ndarray
    sample_index: jnp.ndarray


def _require(condition, message):
    """Raise for concrete invalid trajectories; traced ones are masked below."""
    if not isinstance(condition, jax.core.Tracer) and not bool(jnp.all(condition)):
        raise ValueError(message)


def orbital_frequency_peak(
    time: jnp.ndarray,
    frequency: jnp.ndarray,
    *,
    valid: Optional[jnp.ndarray] = None,
) -> OrbitalFrequencyPeak:
    """Refine an interior maximum with a seven-sample degree-six interpolation."""
    time = as_array(time, dtype=jnp.float64)
    frequency = as_array(frequency, dtype=jnp.float64)

    if time.ndim < 1 or frequency.ndim < 1 or time.shape[-1] != frequency.shape[-1]:
        raise ValueError("time and frequency must share a last-axis sample count")
    time, frequency = jnp.broadcast_arrays(time, frequency)
    count_all = time.shape[-1]
    if count_all < 7:
        raise ValueError("a resolved peak requires at least seven samples")
    row_ok = jnp.all(jnp.isfinite(time) & jnp.isfinite(frequency), axis=-1)
    _require(row_ok, "time and frequency must be finite")
    increasing = jnp.all(jnp.diff(time, axis=-1) > 0, axis=-1)
    _require(increasing, "sample times must be strictly increasing")
    row_ok &= increasing

    if valid is None:
        valid = jnp.ones_like(frequency, dtype=bool)
    else:
        valid = as_array(valid, dtype=bool)
    valid = jnp.broadcast_to(valid, frequency.shape)
    prefix = valid[..., 0] & ~jnp.any(valid[..., 1:] & ~valid[..., :-1], axis=-1)
    _require(prefix, "valid samples must form contiguous nonempty prefixes")
    row_ok &= prefix

    positive = jnp.all((frequency > 0) | ~valid, axis=-1)
    _require(positive, "valid orbital frequency samples must be positive")
    row_ok &= positive
    count = valid.sum(-1)
    resolved = count >= 7
    _require(resolved, "a resolved peak requires at least seven valid samples")
    row_ok &= resolved

    index = jnp.where(valid, frequency, -jnp.inf).argmax(-1)
    interior = (index >= 1) & (index + 1 < count)
    _require(interior, "orbital-frequency peak is unresolved: need samples on both sides")
    row_ok &= interior

    safe_index = jnp.clip(index, 1, count_all - 2)
    start = jnp.maximum(safe_index - 3, 0)
    start = jnp.minimum(start, jnp.maximum(count - 7, 0))

    offsets = jnp.arange(7)
    indices = jnp.expand_dims(start, -1) + offsets
    t = jnp.take_along_axis(time, indices, axis=-1)
    y = jnp.take_along_axis(frequency, indices, axis=-1)

    center = safe_index - start
    origin = jnp.take_along_axis(t, jnp.expand_dims(center, -1), axis=-1).squeeze(-1)
    scale = (t[..., -1] - t[..., 0]) / 2.0
    x = (t - jnp.expand_dims(origin, -1)) / jnp.expand_dims(scale, -1)
    matrix = jnp.stack([x**k for k in range(7)], axis=-1)

    y_center = jnp.take_along_axis(y, jnp.expand_dims(center, -1), axis=-1)
    coef = jnp.linalg.solve(matrix, jnp.expand_dims(y - y_center, -1)).squeeze(-1)

    def derivatives(z):
        first = sum(k * coef[..., k] * (z**(k - 1)) for k in range(1, 7))
        second = sum(k * (k - 1) * coef[..., k] * (z**(k - 2)) for k in range(2, 7))
        return first, second

    left = jnp.take_along_axis(x, jnp.expand_dims(center - 1, -1), axis=-1).squeeze(-1)
    right = jnp.take_along_axis(x, jnp.expand_dims(center + 1, -1), axis=-1).squeeze(-1)
    bracketed = (derivatives(left)[0] > 0) & (derivatives(right)[0] < 0)
    _require(bracketed, "local orbital-frequency maximum has no derivative bracket")
    row_ok &= bracketed

    root = jnp.zeros_like(origin)
    for _ in range(24):
        first, second = derivatives(root)
        safe_second = jnp.where(second != 0, second, jnp.ones_like(second))
        newton = root - first / safe_second
        candidate = jnp.where(
            (second < 0) & (newton >= left) & (newton <= right),
            newton,
            (left + right) / 2.0,
        )
        slope, _ = derivatives(candidate)
        left = jnp.where(slope > 0, candidate, left)
        right = jnp.where(slope < 0, candidate, right)
        root = candidate

    value = (
        jnp.take_along_axis(y, jnp.expand_dims(center, -1), axis=-1).squeeze(-1)
        + sum(coef[..., k] * (root**k) for k in range(7))
    )
    slope, second = derivatives(root)
    tolerance = 128 * jnp.finfo(time.dtype).eps * jnp.sum(jnp.abs(coef), axis=-1)
    converged = (jnp.abs(slope) <= tolerance) & (second < 0)
    _require(converged, "orbital-frequency peak refinement did not converge to a maximum")
    row_ok &= converged
    return OrbitalFrequencyPeak(
        jnp.where(row_ok, origin + scale * root, jnp.nan),
        jnp.where(row_ok, value, jnp.nan),
        jnp.where(row_ok, second / (scale**2), jnp.nan),
        jnp.where(row_ok, index, -1),
    )


__all__ = [
    "OrbitalFrequencyPeak",
    "orbital_frequency_peak",
]

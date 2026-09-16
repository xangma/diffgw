"""Resolved trajectory events in JAX, independent of an EOB calibration prescription."""

from typing import NamedTuple, Optional
import jax.numpy as jnp

from jaxwave.conversions import as_array


class OrbitalFrequencyPeak(NamedTuple):
    time: jnp.ndarray
    frequency: jnp.ndarray
    curvature: jnp.ndarray
    sample_index: jnp.ndarray


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
    if not bool(jnp.all(jnp.isfinite(time)) and jnp.all(jnp.isfinite(frequency))):
        raise ValueError("time and frequency must be finite")
    if bool(jnp.any(jnp.diff(time, axis=-1) <= 0)):
        raise ValueError("sample times must be strictly increasing")

    if valid is None:
        valid = jnp.ones_like(frequency, dtype=bool)
    else:
        valid = as_array(valid, dtype=bool)
    valid = jnp.broadcast_to(valid, frequency.shape)

    if bool(jnp.any((frequency <= 0) & valid)):
        raise ValueError("valid orbital frequency samples must be positive")
    count = valid.sum(-1)
    if bool(jnp.any(count < 7)):
        raise ValueError("a resolved peak requires at least seven valid samples")

    index = jnp.where(valid, frequency, -jnp.inf).argmax(-1)
    if bool(jnp.any((index < 1) | (index + 1 >= count))):
        raise ValueError("orbital-frequency peak is unresolved: need samples on both sides")

    start = jnp.maximum(index - 3, 0)
    start = jnp.minimum(start, count - 7)

    offsets = jnp.arange(7)
    indices = jnp.expand_dims(start, -1) + offsets
    t = jnp.take_along_axis(time, indices, axis=-1)
    y = jnp.take_along_axis(frequency, indices, axis=-1)

    center = index - start
    origin = jnp.take_along_axis(t, jnp.expand_dims(center, -1), axis=-1).squeeze(-1)
    scale = (t[..., -1] - t[..., 0]) / 2.0
    x = (t - jnp.expand_dims(origin, -1)) / jnp.expand_dims(scale, -1)
    matrix = jnp.stack([x**k for k in range(7)], axis=-1)

    y_center = jnp.take_along_axis(y, jnp.expand_dims(center, -1), axis=-1)
    coef = jnp.linalg.solve(matrix, y - y_center)

    def derivatives(z):
        first = sum(k * coef[..., k] * (z**(k - 1)) for k in range(1, 7))
        second = sum(k * (k - 1) * coef[..., k] * (z**(k - 2)) for k in range(2, 7))
        return first, second

    left = jnp.take_along_axis(x, jnp.expand_dims(center - 1, -1), axis=-1).squeeze(-1)
    right = jnp.take_along_axis(x, jnp.expand_dims(center + 1, -1), axis=-1).squeeze(-1)

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
    _, second = derivatives(root)
    return OrbitalFrequencyPeak(origin + scale * root, value, second / (scale**2), index)


__all__ = [
    "OrbitalFrequencyPeak",
    "orbital_frequency_peak",
]

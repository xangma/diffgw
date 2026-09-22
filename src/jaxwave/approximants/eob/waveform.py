"""Shared batched interpolation and quadrupole Fourier projection helpers."""

import math
from typing import Tuple
import jax
import jax.numpy as jnp

from diffgw_core.constants import PI


def _searchsorted_batched(a: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Batched searchsorted supporting N-dimensional a and v."""
    if a.ndim == 1:
        return jnp.searchsorted(a, v, side="left")
    flat_a = a.reshape(-1, a.shape[-1])
    flat_v = v.reshape(-1, v.shape[-1])
    res = jax.vmap(lambda a_, v_: jnp.searchsorted(a_, v_, side="left"))(flat_a, flat_v)
    return res.reshape(v.shape)


def _interp1d(x_new: jnp.ndarray, x_old: jnp.ndarray, y_old: jnp.ndarray) -> jnp.ndarray:
    """Interpolate along the final axis, with zero outside the sample interval."""
    scalar_query = x_new.ndim == 0
    if scalar_query:
        x_new = x_new[None]
    batch_shape = jnp.broadcast_shapes(x_new.shape[:-1], x_old.shape[:-1], y_old.shape[:-1])
    x_old_b = jnp.broadcast_to(x_old, batch_shape + (x_old.shape[-1],))
    y_old_b = jnp.broadcast_to(y_old, batch_shape + (y_old.shape[-1],))
    x_new_b = jnp.broadcast_to(x_new, batch_shape + (x_new.shape[-1],))

    idx = _searchsorted_batched(x_old_b, x_new_b)
    idx = jnp.clip(idx, 1, x_old_b.shape[-1] - 1)

    x0 = jnp.take_along_axis(x_old_b, idx - 1, axis=-1)
    x1 = jnp.take_along_axis(x_old_b, idx, axis=-1)
    y0 = jnp.take_along_axis(y_old_b, idx - 1, axis=-1)
    y1 = jnp.take_along_axis(y_old_b, idx, axis=-1)

    slope = (y1 - y0) / (x1 - x0 + 1e-30)
    y_new = y0 + slope * (x_new_b - x0)

    mask_low = x_new_b < x_old_b[..., :1]
    mask_high = x_new_b > x_old_b[..., -1:]
    y_new = jnp.where(mask_low | mask_high, jnp.zeros_like(y_new), y_new)

    return y_new[..., 0] if scalar_query else y_new


def _quadrupole_fd(
    f: jnp.ndarray,
    h22: jnp.ndarray,
    dt_sec: jnp.ndarray,
    inclination: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Project paired spherical quadrupoles and Fourier transform in seconds."""
    cos_i = jnp.cos(inclination)
    hp_td = math.sqrt(5.0 / (16.0 * PI)) * (1.0 + cos_i**2) * h22.real
    hc_td = -math.sqrt(5.0 / (4.0 * PI)) * cos_i * h22.imag

    fft_len = 2 ** math.ceil(math.log2(h22.shape[0] * 4))
    hp_fd = jnp.fft.rfft(hp_td, n=fft_len, axis=0) * dt_sec
    hc_fd = jnp.fft.rfft(hc_td, n=fft_len, axis=0) * dt_sec

    k = jnp.arange(fft_len // 2 + 1, dtype=f.dtype)
    if h22.ndim > 1:
        fft_freqs = k / (fft_len * dt_sec[..., None])
        hp_fd = jnp.moveaxis(hp_fd, 0, -1)
        hc_fd = jnp.moveaxis(hc_fd, 0, -1)
    else:
        fft_freqs = k / (fft_len * dt_sec)

    return _interp1d(f, fft_freqs, hp_fd), _interp1d(f, fft_freqs, hc_fd)


__all__ = [
    '_interp1d',
    '_quadrupole_fd',
]

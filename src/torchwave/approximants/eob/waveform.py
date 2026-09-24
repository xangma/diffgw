"""Shared batched interpolation and quadrupole Fourier projection helpers."""

import torch
import math

from torchwave.constants import PI


def _interp1d(x_new: torch.Tensor, x_old: torch.Tensor, y_old: torch.Tensor) -> torch.Tensor:
    """Interpolate along the final axis, with zero outside the sample interval.

    A scalar query returns one value per table batch; otherwise the query's
    final axis is its sample axis.
    """
    scalar_query = x_new.ndim == 0
    if scalar_query:
        x_new = x_new.unsqueeze(-1)
    batch_shape = torch.broadcast_shapes(x_new.shape[:-1], x_old.shape[:-1], y_old.shape[:-1])
    x_old_b = x_old.expand(*batch_shape, x_old.shape[-1])
    y_old_b = y_old.expand(*batch_shape, y_old.shape[-1])
    x_new_b = x_new.expand(*batch_shape, x_new.shape[-1])

    idx = torch.searchsorted(x_old_b.contiguous(), x_new_b.contiguous(), right=False)
    idx = torch.clamp(idx, 1, x_old_b.shape[-1] - 1)

    x0 = torch.gather(x_old_b, -1, idx - 1)
    x1 = torch.gather(x_old_b, -1, idx)
    y0 = torch.gather(y_old_b, -1, idx - 1)
    y1 = torch.gather(y_old_b, -1, idx)

    slope = (y1 - y0) / (x1 - x0 + 1e-30)
    y_new = y0 + slope * (x_new_b - x0)

    mask_low = x_new_b < x_old_b[..., :1]
    mask_high = x_new_b > x_old_b[..., -1:]
    y_new = torch.where(mask_low | mask_high, torch.zeros_like(y_new), y_new)

    return y_new.squeeze(-1) if scalar_query else y_new


def _quadrupole_fd(f, h22, dt_sec, inclination):
    """Project paired spherical quadrupoles and Fourier transform in seconds.

    Time is axis zero, followed by any parameter batch axes. The
    transform uses exp(-2 pi i f t), with frequencies on the last output axis.
    """
    cos_i = torch.cos(inclination)
    hp_td = math.sqrt(5.0 / (16.0 * PI)) * (1.0 + cos_i**2) * h22.real
    hc_td = -math.sqrt(5.0 / (4.0 * PI)) * cos_i * h22.imag
    fft_len = 2 ** math.ceil(math.log2(h22.shape[0] * 4))
    hp_fd = torch.fft.rfft(hp_td, n=fft_len, dim=0) * dt_sec
    hc_fd = torch.fft.rfft(hc_td, n=fft_len, dim=0) * dt_sec
    k = torch.arange(fft_len // 2 + 1, device=f.device, dtype=f.dtype)
    if h22.ndim > 1:
        fft_freqs = k / (fft_len * dt_sec.unsqueeze(-1))
        hp_fd, hc_fd = hp_fd.movedim(0, -1), hc_fd.movedim(0, -1)
    else:
        fft_freqs = k / (fft_len * dt_sec)
    return _interp1d(f, fft_freqs, hp_fd), _interp1d(f, fft_freqs, hc_fd)

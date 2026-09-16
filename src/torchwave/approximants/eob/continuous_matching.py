"""Continuous event location on accepted samples, without a resampling grid."""

from __future__ import annotations

import torch
from torch import Tensor

from .numerical_attachment import natural_cubic_spline


def continuous_spline_peak(time: Tensor, values: Tensor) -> Tensor:
    """Return the unique interior maximum of a scalar natural cubic spline.

    Solve each interval's quadratic derivative in local coordinates, then
    select the largest maximum. Endpoints and tied, distinct maxima are
    rejected. The selected root remains differentiable with respect to knots
    and values; the interval/maximum selection is necessarily piecewise.
    """
    if time.ndim != 1 or values.shape != time.shape or time.numel() < 3:
        raise ValueError("continuous peak needs a scalar series with at least three knots")
    spline = natural_cubic_spline(time, values, scalar_output=True)
    h = time[1:] - time[:-1]
    a, b, c, d = (v[:, 0] for v in (spline.a, spline.b, spline.c, spline.d))
    # With u=(t-t_i)/h, the derivative is B+C*u+D*u**2.
    B, C, D = b, 2 * c * h, 3 * d * h.square()
    eps = torch.finfo(time.dtype).eps
    scale = torch.stack((B.abs(), C.abs(), D.abs())).amax(0)
    quadratic = D.abs() > 8 * eps * scale
    discriminant = C.square() - 4 * D * B
    sqrt_disc = discriminant.clamp_min(torch.finfo(time.dtype).tiny).sqrt()
    q = -0.5 * (C + torch.where(C >= 0, sqrt_disc, -sqrt_disc))
    safe_D = torch.where(quadratic, D, torch.ones_like(D))
    safe_q = torch.where(q != 0, q, torch.ones_like(q))
    linear = (~quadratic) & (C.abs() > 8 * eps * scale)
    safe_C = torch.where(linear, C, torch.ones_like(C))
    roots = torch.stack((q / safe_D, B / safe_q, -B / safe_C), -1)
    valid_kind = torch.stack((quadratic & (discriminant > 0),
                              quadratic & (discriminant > 0) & (q != 0), linear), -1)
    valid = valid_kind & (roots >= -32 * eps) & (roots <= 1 + 32 * eps)
    valid = valid & ((C[:, None] + 2 * D[:, None] * roots) < -8 * eps * scale[:, None])
    # Invalid roots must not inject infinities into an otherwise valid graph.
    u = torch.where(valid, roots, torch.zeros_like(roots)).clamp(0, 1)
    dx = h[:, None] * u
    heights = a[:, None] + dx * (b[:, None] + dx * (c[:, None] + dx * d[:, None]))
    times = time[:-1, None] + dx
    valid = valid & (times > time[0]) & (times < time[-1])
    if not bool(valid.any()):
        raise ValueError("continuous peak needs a resolved interior maximum")
    heights = heights.masked_fill(~valid, -torch.inf)
    index = int(heights.detach().reshape(-1).argmax())
    peak = times.reshape(-1)[index]
    height = heights.reshape(-1)[index]
    tolerance = 32 * eps * values.abs().amax()
    if bool(torch.maximum(values[0], values[-1]) >= height - tolerance):
        raise ValueError("continuous peak maximum is not strictly interior")
    distinct = (times - peak).abs() > 64 * eps * (time.abs().amax() + h.amax())
    if bool((valid & distinct & ((heights - height).abs() <= tolerance)).any()):
        raise ValueError("continuous peak has tied distinct maxima")
    return peak

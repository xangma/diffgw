# SPDX-License-Identifier: MIT
"""Paper-derived IMRPhenomXHM inspiral carrier phase primitives."""

from __future__ import annotations

import math

import torch


COMPONENT_VERSION = "xhm.phase.inspiral.carrier-rescaling.v1"


def _validate_mode(mode_m: int) -> None:
    if isinstance(mode_m, bool) or not isinstance(mode_m, int) or mode_m <= 0:
        raise ValueError("mode_m must be a positive integer")


def rescale_carrier_phase(
    carrier_phase_at_scaled_frequency: object, mode_m: int
) -> torch.Tensor:
    """Apply XHM v2 Eq. (29) to a 22 phase evaluated at ``2 f / m``."""

    _validate_mode(mode_m)
    phase = torch.as_tensor(carrier_phase_at_scaled_frequency)
    if not phase.is_floating_point():
        phase = phase.to(dtype=torch.get_default_dtype())
    return (mode_m / 2.0) * phase


def rescale_carrier_phase_coefficients(
    mode_m: int,
    indices: object,
    c22: object,
    d22: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply XHM v2 Appendix-D Eq. (63) to phase coefficients.

    ``indices``, ``c22``, and ``d22`` broadcast.  The returned tensors are
    ``(c_m, d_m)``; no mode-dependent correction or attachment term is added.
    """

    _validate_mode(mode_m)
    c = torch.as_tensor(c22)
    dtype = c.dtype if c.is_floating_point() else torch.get_default_dtype()
    c = c.to(dtype=dtype)
    d = torch.as_tensor(d22, dtype=dtype, device=c.device)
    i = torch.as_tensor(indices, dtype=dtype, device=c.device)
    i, c, d = torch.broadcast_tensors(i, c, d)
    ratio = mode_m / 2.0
    scale = torch.pow(torch.as_tensor(ratio, dtype=dtype, device=c.device), (8.0 - i) / 3.0)
    return scale * (c - d * math.log(ratio)), scale * d


__all__ = [
    "COMPONENT_VERSION",
    "rescale_carrier_phase",
    "rescale_carrier_phase_coefficients",
]

"""Shared physical projection of paired quadrupole modes onto real polarizations."""

# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC


def _real_tensor(value: Any, *, dtype: torch.dtype, device: torch.device,
                 name: str) -> Tensor:
    if isinstance(value, bool) or (
        isinstance(value, Tensor) and value.dtype == torch.bool
    ):
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, Tensor) and value.is_complex():
        raise TypeError(f"{name} must be real")
    try:
        result = torch.as_tensor(value, dtype=dtype, device=device)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not result.dtype.is_floating_point:
        result = result.to(dtype=dtype)
    return result


def _parameter_broadcast(
    values: Sequence[Tuple[str, Any]], *, dtype: torch.dtype,
    device: torch.device,
) -> Mapping[str, Tensor]:
    """Convert and broadcast intrinsic/extrinsic scalar or batch values."""

    names = [name for name, _ in values]
    tensors = [
        _real_tensor(value, dtype=dtype, device=device, name=name)
        for name, value in values
    ]
    try:
        broadcasted = torch.broadcast_tensors(*tensors)
    except RuntimeError as exc:
        raise ValueError(
            "masses, spins, distance, inclination, phic and tc must be "
            "broadcastable"
        ) from exc
    return dict(zip(names, broadcasted))


def project_22_pair_to_polarizations(
    h22: Tensor,
    h2m2: Optional[Tensor] = None,
    *,
    mass1: Any,
    mass2: Any,
    distance: Any = 100.0,
    inclination: Any = 0.0,
    phic: Any = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Project a paired nonprecessing 22 mode onto real ``(h_plus,h_cross)``.

    ``h22`` and ``h2m2`` are dimensionless geometric modes with a trailing
    time axis.  The physical strain scale is ``M_seconds*c/(D*Mpc)``.  The
    mode phase convention is explicit: ``m=2`` receives ``exp(-2 i phic)``
    and ``m=-2`` receives ``exp(+2 i phic)``.  When ``h2m2`` is omitted, the
    standard aligned-spin relation ``h2m2=conj(h22)`` is used.
    """

    if not isinstance(h22, Tensor) or h22.ndim < 1 or h22.shape[-1] == 0:
        raise TypeError("h22 must be a nonempty tensor with a trailing time axis")
    # ``torch.complex`` accepts floating inputs only.  Integer modes are not
    # physically meaningful, but converting them here keeps the projection's
    # dtype/device contract explicit instead of leaking a backend error.
    if not h22.dtype.is_floating_point and not h22.is_complex():
        h22 = h22.to(dtype=torch.float64)
    if not h22.is_complex():
        h22 = torch.complex(h22, torch.zeros_like(h22))
    if h2m2 is None:
        h2m2 = torch.conj(h22)
    if not isinstance(h2m2, Tensor) or h2m2.shape != h22.shape:
        raise ValueError("h22 and h2m2 must have equal shapes")
    if h2m2.device != h22.device:
        raise ValueError("h22 and h2m2 must share a device")
    if not bool(torch.isfinite(h22).all()) or not bool(torch.isfinite(h2m2).all()):
        raise ValueError("h22 and h2m2 must be finite")

    real_dtype = h22.real.dtype
    device = h22.device
    values = _parameter_broadcast(
        (("mass1", mass1), ("mass2", mass2), ("distance", distance),
         ("inclination", inclination), ("phic", phic)),
        dtype=real_dtype,
        device=device,
    )
    m1, m2 = values["mass1"], values["mass2"]
    dist, inc, phase = values["distance"], values["inclination"], values["phic"]
    if not bool(torch.isfinite(m1).all()) or not bool((m1 > 0).all()):
        raise ValueError("mass1 must be finite and positive")
    if not bool(torch.isfinite(m2).all()) or not bool((m2 > 0).all()):
        raise ValueError("mass2 must be finite and positive")
    if not bool(torch.isfinite(dist).all()) or not bool((dist > 0).all()):
        raise ValueError("distance must be finite and positive")
    if not bool(torch.isfinite(inc).all()):
        raise ValueError("inclination must be finite")
    if not bool(torch.isfinite(phase).all()):
        raise ValueError("phic must be finite")

    batch_shape = torch.broadcast_shapes(
        h22.shape[:-1], m1.shape, m2.shape, dist.shape, inc.shape, phase.shape
    )
    n_time = h22.shape[-1]
    h22 = h22.expand(batch_shape + (n_time,))
    h2m2 = h2m2.to(device=device, dtype=h22.dtype).expand(batch_shape + (n_time,))
    m1, m2, dist, inc, phase = (
        value.expand(batch_shape)
        for value in (m1, m2, dist, inc, phase)
    )

    # The two spin-weighted harmonics at azimuth zero are
    # sqrt(5/(64*pi))*(1 +/- cos(iota))**2.  Taking Re(H) and -Im(H)
    # reproduces the established single-mode projection when h2m2=conj(h22).
    cosine = torch.cos(inc).unsqueeze(-1)
    plus_weight = (1.0 + cosine).square()
    minus_weight = (1.0 - cosine).square()
    phase22 = torch.exp((-2j) * phase.unsqueeze(-1))
    phasem2 = torch.exp((2j) * phase.unsqueeze(-1))
    pair = plus_weight * h22 * phase22 + minus_weight * h2m2 * phasem2
    normalization = h22.real.new_tensor(math.sqrt(5.0 / (64.0 * math.pi)))
    pair = normalization * pair
    m_seconds = (m1 + m2) * h22.real.new_tensor(MSUN_SEC)
    scale = (
        m_seconds * h22.real.new_tensor(C_SI)
        / (dist * h22.real.new_tensor(MPC_SI))
    ).unsqueeze(-1)
    pair = pair * scale
    return pair.real, -pair.imag


# Short aliases make the projection helper discoverable without introducing
# another implementation path.
project_22_modes = project_22_pair_to_polarizations
project_quadrupole_pair = project_22_pair_to_polarizations


__all__ = [
    'project_22_pair_to_polarizations',
    'project_22_modes',
    'project_quadrupole_pair',
]

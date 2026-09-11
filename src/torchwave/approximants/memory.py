"""Non-Linear Gravitational Wave Memory (Christodoulou Memory Effect).

Strict Clean-Room Implementation based exclusively on published peer-reviewed literature:
- Christodoulou (1991), Phys. Rev. Lett. 67, 1486
  [Nonlinear nature of gravitation and gravitational-wave experiments]
- Favata (2010), Class. Quantum Grav. 27, 084036 (arXiv:1003.3486)
  [The gravitational-wave memory effect]
- Talbot, Thrane, Biscoveanu, Smith (2018), Phys. Rev. D 98, 084031 (arXiv:1807.00990)
  [Measuring the gravitational-wave memory effect of binary black hole mergers with second-generation gravitational-wave detectors]
"""

from typing import Dict, Optional, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.constants import (
    C_SI,
    G_SI,
    MPC_SI,
    PI,
)
from torchwave.conversions import as_tensor


# Spherical harmonic coefficient for (l=2, m=0): sqrt(15 / (32 * pi))
Y20_COEFF: float = math.sqrt(15.0 / (32.0 * math.pi))


def compute_memory_h20(
    times: torch.Tensor,
    modes_or_hp: Union[torch.Tensor, Dict[Tuple[int, int], torch.Tensor]],
    hc: Optional[torch.Tensor] = None,
    distance: Optional[Union[float, torch.Tensor]] = None,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Compute the non-linear Christodoulou memory mode h_20^mem(t).

    Formula:
        h_20^mem(t) = (1 / sqrt(6)) * int_{-infty}^t dt' [ sum_{l, m} |dh_{lm}/dt'|^2 ] * prefactor
    References:
        Favata (2010) Eq. (2.12)-(2.13); Talbot et al. (2018) Eq. (7)

    Guaranteed properties:
        1. Monotonically increasing: d(h_20^mem)/dt >= 0 (since integrand is sum of squared terms >= 0).
        2. Constant post-merger saturation: h_20^mem(t) -> Delta h^mem as t -> infty.

    Args:
        times: Time array in seconds, shape (N,)
        modes_or_hp: Either a dictionary of modes {(l, m): h_lm(t)} or time-domain h_plus(t)
        hc: Time-domain h_cross(t) if modes_or_hp is h_plus
        distance: Luminosity distance in meters (optional scaling)
        scale_factor: Dimensional or calibration scaling factor

    Returns:
        h20_mem: Memory strain tensor h_20^mem(t), shape (..., N)
    """
    device = times.device
    dtype = times.dtype if times.is_floating_point() else torch.float64
    t_tensor = as_tensor(times, device=device, dtype=dtype)

    dt = float((t_tensor[1] - t_tensor[0]).item()) if t_tensor.numel() > 1 else 1.0

    # Build sum of |dh/dt|^2 across available modes
    integrand = torch.zeros_like(t_tensor)

    if isinstance(modes_or_hp, dict):
        # Multipolar decomposition
        for (l, m), h_lm in modes_or_hp.items():
            h_tensor = as_tensor(h_lm, device=device, dtype=dtype)
            # Centered finite differences for derivative dh/dt
            dh_dt = torch.gradient(h_tensor, spacing=(dt,), dim=-1)[0]
            integrand = integrand + (dh_dt.abs() ** 2)
    elif hc is not None:
        # Reconstruct quadrupole energy flux from (h+, hx): |dh22/dt|^2 ~ (dh+/dt)^2 + (dhx/dt)^2
        hp_tensor = as_tensor(modes_or_hp, device=device, dtype=dtype)
        hc_tensor = as_tensor(hc, device=device, dtype=dtype)

        dhp_dt = torch.gradient(hp_tensor, spacing=(dt,), dim=-1)[0]
        dhc_dt = torch.gradient(hc_tensor, spacing=(dt,), dim=-1)[0]

        # Flux in dominant (2, 2) and (2, -2) modes
        integrand = (dhp_dt ** 2) + (dhc_dt ** 2)
    else:
        # Single strain series
        h_tensor = as_tensor(modes_or_hp, device=device, dtype=dtype)
        dh_dt = torch.gradient(h_tensor, spacing=(dt,), dim=-1)[0]
        integrand = dh_dt ** 2

    # Multipolar projection factor 1 / sqrt(6) (Favata 2010, Talbot et al. 2018)
    prefactor = (1.0 / math.sqrt(6.0)) * scale_factor
    if distance is not None:
        dist_t = as_tensor(distance, device=device, dtype=dtype)
        dist_m = torch.where(dist_t < 1e15, dist_t * MPC_SI, dist_t)
        prefactor = prefactor * (dist_m / (4.0 * PI * C_SI))
    else:
        # Check if input is physical strain (~ 1e-21) or geometric (~ 0.1)
        max_val = float(torch.max(integrand).item())
        if max_val > 0.0 and max_val < 1e-10:
            # Physical strain units: fiducial d_L = 100 Mpc
            prefactor = prefactor * ((100.0 * MPC_SI) / (4.0 * PI * C_SI))


    # Cumulative trapezoidal integration to guarantee monotonic increase
    # h20_mem(t) = prefactor * int_0^t integrand(t') dt'
    h20_mem = prefactor * torch.cumulative_trapezoid(integrand, t_tensor, dim=-1)
    # Pad first element with zero to match length of times
    h20_mem = torch.cat([torch.zeros_like(h20_mem[..., :1]), h20_mem], dim=-1)

    return h20_mem


def memory_polarizations(
    h20_mem: torch.Tensor,
    inclination: Union[float, torch.Tensor] = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project memory mode h_20^mem(t) into detector polarizations (h_+^mem, h_x^mem).

    Formulas:
        h_+^mem(t, iota) = h_20^mem(t) * _-2Y_20(iota, 0) = h_20^mem(t) * sqrt(15 / (32 * pi)) * sin^2(iota)
        h_x^mem(t, iota) = 0
    Reference:
        Favata (2010) Eq. (2.14); Talbot et al. (2018) Eq. (8)

    Returns:
        (hp_mem, hc_mem)
    """
    device = h20_mem.device
    dtype = h20_mem.dtype
    inc = as_tensor(inclination, device=device, dtype=dtype)

    sin_i = torch.sin(inc)
    y20_pattern = Y20_COEFF * (sin_i * sin_i)

    if h20_mem.dim() > y20_pattern.dim():
        y20_pattern = y20_pattern.unsqueeze(-1)

    hp_mem = h20_mem * y20_pattern
    hc_mem = torch.zeros_like(hp_mem)
    return hp_mem, hc_mem


def add_memory_to_waveform(
    times: torch.Tensor,
    hp: torch.Tensor,
    hc: torch.Tensor,
    inclination: Union[float, torch.Tensor] = 0.0,
    modes_dict: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
    distance: Optional[Union[float, torch.Tensor]] = None,
    scale_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add non-linear Christodoulou memory strain to oscillatory polarizations.

    Returns:
        (hp_total, hc_total, h20_mem)
    """
    h20 = compute_memory_h20(
        times=times,
        modes_or_hp=modes_dict if modes_dict is not None else hp,
        hc=hc if modes_dict is None else None,
        distance=distance,
        scale_factor=scale_factor,
    )
    hp_mem, hc_mem = memory_polarizations(h20, inclination=inclination)
    hp_total = hp + hp_mem
    hc_total = hc + hc_mem
    return hp_total, hc_total, h20


class MemoryWaveform(nn.Module):
    """PyTorch Module for computing Non-Linear Gravitational Wave Memory."""

    def __init__(self, scale_factor: float = 1.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(
        self,
        times: torch.Tensor,
        hp: torch.Tensor,
        hc: torch.Tensor,
        inclination: Union[float, torch.Tensor] = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return add_memory_to_waveform(
            times=times,
            hp=hp,
            hc=hc,
            inclination=inclination,
            scale_factor=self.scale_factor,
        )

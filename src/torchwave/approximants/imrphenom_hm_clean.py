"""Clean-room public helpers for the original IMRPhenomHM frequency maps.

The maps are Eq. (5) of London et al., arXiv:1708.00404.  Keeping them in a
small dependency-free module makes their continuity and limiting behaviour
testable without depending on model implementation details.
"""

from __future__ import annotations

import torch


def pn_hhat_lm(
    mode: tuple[int, int], v: torch.Tensor, eta: torch.Tensor | float,
    delta: torch.Tensor | float, chi_s: torch.Tensor | float = 0.0,
    chi_a: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Restricted PN ``H-hat_lm`` factors used by London Eq. (4).

    This is a clean transcription of Mishra et al., PRD 93, 084054 (2016),
    Eqs. (11)-(12), retaining leading ``l=|m|`` terms and all terms through
    absolute 1.5PN in ``f`` for ``|m|=l-1`` as prescribed by London et al.
    (arXiv:1708.00404, text following Eq. (4)).  ``v`` is the positive
    orbital PN parameter and ``delta=(m1-m2)/M``; no detector or distance
    normalization is included.
    """
    v = torch.as_tensor(v)
    eta = torch.as_tensor(eta, dtype=v.dtype, device=v.device)
    delta = torch.as_tensor(delta, dtype=v.dtype, device=v.device)
    chi_s = torch.as_tensor(chi_s, dtype=v.dtype, device=v.device)
    chi_a = torch.as_tensor(chi_a, dtype=v.dtype, device=v.device)
    if mode == (2, 2):
        return -torch.ones_like(v)
    if mode == (3, 3):
        return -(3.0 / 4.0) * torch.sqrt(v.new_tensor(5.0 / 7.0)) * delta * v
    if mode == (4, 4):
        return -(4.0 / 9.0) * torch.sqrt(v.new_tensor(10.0 / 7.0)) * (1.0 - 3.0 * eta) * v**2
    if mode == (2, 1):
        # Eq. (12b), through absolute V^3.
        return -(torch.sqrt(v.new_tensor(2.0)) / 3.0) * (delta * v - 1.5 * (chi_a + delta * chi_s) * v**2 + delta * (335.0 / 672.0 + 117.0 * eta / 56.0) * v**3)
    if mode == (3, 2):
        return -(1.0 / 3.0) * torch.sqrt(v.new_tensor(5.0 / 7.0)) * ((1.0 - 3.0 * eta) * v**2 + 4.0 * eta * chi_s * v**3)
    if mode == (4, 3):
        return -(3.0 / 4.0) * torch.sqrt(v.new_tensor(3.0 / 35.0)) * delta * (1.0 - 2.0 * eta) * v**3
    raise ValueError(f"unsupported HM mode {mode}")


def map_amplitude_frequency(
    f: torch.Tensor,
    f0: torch.Tensor | float,
    f_rd_lm: torch.Tensor | float,
    f_rd_22: torch.Tensor | float,
    m: int,
) -> torch.Tensor:
    """Return the mapped 22 amplitude frequency ``f^A_22(f)``.

    The three affine pieces are the inspiral, intermediate, and ringdown
    pieces in London et al. Eq. (5).  Inputs are geometric frequencies.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    f = torch.as_tensor(f)
    f0 = torch.as_tensor(f0, dtype=f.dtype, device=f.device)
    fr = torch.as_tensor(f_rd_lm, dtype=f.dtype, device=f.device)
    f22 = torch.as_tensor(f_rd_22, dtype=f.dtype, device=f.device)
    slope = (f22 - 2.0 * f0 / m) / (fr - f0)
    mid = slope * (f - f0) + 2.0 * f0 / m
    return torch.where(f <= f0, 2.0 * f / m, torch.where(f <= fr, mid, f - (fr - f22)))


def map_phase_frequency(
    f: torch.Tensor,
    f0: torch.Tensor | float,
    f_rd_lm: torch.Tensor | float,
    f_rd_22: torch.Tensor | float,
    m: int,
) -> torch.Tensor:
    """Return the mapped 22 phase frequency ``f^phi_22(f)``.

    London et al. Eq. (5) is used through merger.  The PhenomD application
    replaces its ringdown branch by the multiplicative map in Eq. (13).
    """
    if m <= 0:
        raise ValueError("m must be positive")
    f = torch.as_tensor(f)
    f0 = torch.as_tensor(f0, dtype=f.dtype, device=f.device)
    fr = torch.as_tensor(f_rd_lm, dtype=f.dtype, device=f.device)
    f22 = torch.as_tensor(f_rd_22, dtype=f.dtype, device=f.device)
    slope = (f22 - 2.0 * f0 / m) / (fr - f0)
    mid = slope * (f - f0) + 2.0 * f0 / m
    return torch.where(f <= f0, 2.0 * f / m, torch.where(f <= fr, mid, (f22 / fr) * f))

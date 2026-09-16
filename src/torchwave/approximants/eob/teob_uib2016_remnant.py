"""Nonspinning Jiménez--Forteza et al. (2017) TEOB remnant inputs.

The nonspinning branch is Eqs. (7) and (21) of Jiménez--Forteza et al.,
Phys. Rev. D 95, 064024 (2017), arXiv:1611.00332.  The decimal
coefficients below retain the full precision published in the author
supplement to IMRPhenomXAS (``FinalSpin2017`` and ``Erad2017``):

https://arxiv.org/abs/2001.11412

The functions deliberately expose only the zero-spin branch.  They do not
select a spin extension or a TEOB trajectory convention.  The returned mass
is the dimensionless ratio ``M_f / (m1 + m2)`` required by the standalone
TEOB postmerger component.
"""

from __future__ import annotations

from typing import Tuple

import torch


# Full-precision values in the author-published IMRPhenomXAS supplement,
# section ``Energy Radiated``.  They are the a2, a3 and a4 coefficients in
# Jiménez--Forteza Eq. (21).
_ERAD_A2 = 0.5609904135313374
_ERAD_A3 = -0.84667563764404
_ERAD_A4 = 3.145145224278187

# Full-precision values in the same supplement, ``FinalSpin2017``.  Eq. (7)
# writes these as 5.24*a2, 1.3*a3 and 2.88*a5 respectively; retaining the
# factorized form makes that published equation explicit.
_SPIN_A2 = 3.8326341618708577
_SPIN_A3 = -9.487364155598392
_SPIN_A5 = 2.5134875145648374


def _coerce(value: torch.Tensor | float, *others: torch.Tensor | float) -> torch.Tensor:
    values = (value, *others)
    tensors = [item for item in values if isinstance(item, torch.Tensor)]
    if any(item.is_complex() for item in tensors):
        raise ValueError("UIB2016 remnant inputs must be real")
    if (any(isinstance(item, bool) for item in values)
            or any(item.dtype == torch.bool for item in tensors)):
        raise TypeError("UIB2016 remnant inputs must be real numbers")
    if tensors:
        device = tensors[0].device
        if any(item.device != device for item in tensors):
            raise ValueError("UIB2016 remnant tensors must share a device")
        dtype = torch.float32
        for item in tensors:
            if item.dtype.is_floating_point:
                dtype = torch.promote_types(dtype, item.dtype)
            elif not item.dtype.is_floating_point and item.dtype != torch.bool:
                # Integer tensors are accepted as exact real inputs, then
                # evaluated in the default differentiable real precision.
                continue
    else:
        device = torch.device("cpu")
        dtype = torch.float64
    return torch.as_tensor(value, device=device, dtype=dtype)


def _eta(eta: torch.Tensor | float) -> torch.Tensor:
    out = _coerce(eta)
    if torch.any(~torch.isfinite(out)) or torch.any((out < 0) | (out > 0.25)):
        raise ValueError("symmetric mass ratio eta must satisfy 0 <= eta <= 1/4")
    return out


def _component_eta(mass1: torch.Tensor | float, mass2: torch.Tensor | float) -> torch.Tensor:
    m1, m2 = _coerce(mass1, mass2), _coerce(mass2, mass1)
    m1, m2 = torch.broadcast_tensors(m1, m2)
    if torch.any(~torch.isfinite(m1)) or torch.any(~torch.isfinite(m2)):
        raise ValueError("component masses must be finite")
    if torch.any((m1 <= 0) | (m2 <= 0)):
        raise ValueError("component masses must be positive")
    # Scaling avoids overflow for dimensional masses while preserving the
    # exact symmetric mass ratio and its autograd path.
    scale = torch.maximum(m1, m2)
    x1, x2 = m1 / scale, m2 / scale
    eta = (x1 * x2) / (x1 + x2).square()
    # The exact ratio is bounded by 1/4.  Roundoff in the scaled expression
    # can produce 1/4 + ulp for nearly equal masses, which would otherwise
    # reject a valid binary at the public-domain check below.
    return torch.minimum(eta, eta.new_tensor(0.25))


def uib2016_nonspinning_radiated_energy(eta: torch.Tensor | float) -> torch.Tensor:
    """Return ``E_rad/M`` from Jiménez--Forteza Eq. (21), ``chi1=chi2=0``."""
    eta = _eta(eta)
    return ((1.0 - 2.0 * torch.sqrt(eta.new_tensor(2.0)) / 3.0) * eta
            + eta.new_tensor(_ERAD_A2) * eta.square()
            + eta.new_tensor(_ERAD_A3) * eta.pow(3)
            + eta.new_tensor(_ERAD_A4) * eta.pow(4))


def uib2016_nonspinning_final_mass_fraction(eta: torch.Tensor | float) -> torch.Tensor:
    """Return ``M_f/M = 1-E_rad/M`` from Jiménez--Forteza Eq. (21)."""
    return 1.0 - uib2016_nonspinning_radiated_energy(eta)


def uib2016_nonspinning_final_spin(eta: torch.Tensor | float) -> torch.Tensor:
    """Return ``chi_f`` from Jiménez--Forteza Eq. (7), ``chi1=chi2=0``."""
    eta = _eta(eta)
    numerator = (
        2.0 * torch.sqrt(eta.new_tensor(3.0)) * eta
        + eta.new_tensor(5.24 * _SPIN_A2) * eta.square()
        + eta.new_tensor(1.3 * _SPIN_A3) * eta.pow(3)
    )
    return numerator / (1.0 + eta.new_tensor(2.88 * _SPIN_A5) * eta)


def uib2016_nonspinning_remnant(
    mass1: torch.Tensor | float,
    mass2: torch.Tensor | float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(M_f/M, chi_f)`` for positive, nonspinning component masses.

    ``mass1`` and ``mass2`` are broadcastable and may carry gradients.  The
    branch is intentionally nonspinning: no spin input is silently inferred.
    """
    eta = _component_eta(mass1, mass2)
    return (uib2016_nonspinning_final_mass_fraction(eta),
            uib2016_nonspinning_final_spin(eta))


__all__ = [
    "uib2016_nonspinning_radiated_energy",
    "uib2016_nonspinning_final_mass_fraction",
    "uib2016_nonspinning_final_spin",
    "uib2016_nonspinning_remnant",
]

"""Torch evaluation of the independent N=61 London higher-mode QNM fit.

Frozen coefficients, provenance metadata, and their payload hash live in
``diffgw_core.hm_qnm_n61_data`` and remain re-exported here for compatibility.
Evaluation uses Torch operations, preserving spin gradients, dtype, and device.
"""

from __future__ import annotations

import math

import torch


from diffgw_core.hm_qnm_n61_data import (
    N61_ARTIFACT_FORMAT,
    N61_COEFFICIENT_PAYLOAD_SHA256,
    N61_COEFFICIENTS,
    N61_COUNT,
    N61_FIT_STRATEGY,
    N61_GENERATION_ARTIFACT,
    N61_GRID_DEFINITION,
    N61_SPIN_LIMIT,
)


_ZERO_DAMPED = frozenset(((2, 2), (3, 3), (4, 4), (4, 3)))


def _as_real_tensor(value: torch.Tensor | float) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value
        return value.to(dtype=torch.get_default_dtype())
    return torch.as_tensor(value, dtype=torch.get_default_dtype())


def _horner(coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(x)
    for coefficient in reversed(coefficients.unbind(0)):
        result = result * x + coefficient
    return result


def evaluate_n61_qnm(
    mode: tuple[int, int] | tuple[int, int, int],
    final_spin: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one frozen N=61 fit in final-mass units.

    Returns ``(M_f omega_RD, M_f/tau)``.  The fitted complex frequency follows
    London's ``omega=omega_r+i/tau`` convention; the second component is the
    positive angular damping magnitude ``M_f/tau``.  The caller performs the
    ``1/(2*pi)`` conversion to cyclic frequency and the total-to-final-mass
    conversion.
    """
    lm = (int(mode[0]), int(mode[1]))
    if lm not in N61_COEFFICIENTS:
        raise ValueError(f"unsupported N=61 HM QNM mode {mode!r}")
    spin = _as_real_tensor(final_spin)
    dtype = spin.dtype if spin.is_floating_point() else torch.get_default_dtype()
    device = spin.device
    spin = spin.to(dtype=dtype)
    # Equation (21) of London et al.; all production remnant spins lie in the
    # Kerr domain.  No Python conversion is used, preserving autograd.
    kappa = torch.log(2.0 - spin) / math.log(3.0)
    exponent = 1.0 / float(2 + lm[0] - abs(lm[1]))
    kappa = torch.pow(kappa, exponent)

    raw = torch.as_tensor(N61_COEFFICIENTS[lm], dtype=dtype, device=device)
    real = _horner(raw[:, 0], kappa)
    imag = _horner(raw[:, 1], kappa)
    if lm in _ZERO_DAMPED:
        omega_real = float(lm[1]) / 2.0 + kappa * real
        omega_imag = kappa * imag
    else:
        omega_real = real
        omega_imag = imag
    return omega_real, torch.abs(omega_imag)


__all__ = [
    "N61_COUNT",
    "N61_SPIN_LIMIT",
    "N61_GRID_DEFINITION",
    "N61_FIT_STRATEGY",
    "N61_ARTIFACT_FORMAT",
    "N61_COEFFICIENTS",
    "N61_COEFFICIENT_PAYLOAD_SHA256",
    "N61_GENERATION_ARTIFACT",
    "evaluate_n61_qnm",
]

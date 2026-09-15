# SPDX-License-Identifier: MIT
"""Native Numerical Relativity (NR) Calibrated Fits for Higher Harmonics.

TorchWave Sovereign Calibration:
These parameters were independently fitted to public numerical relativity (NR)
simulations from the Simulating eXtreme Spacetimes (SXS) Collaboration
(CC-BY-4.0 attribution preserved, see THIRD_PARTY_NOTICES.md).

No LALSimulation C code, private lookup tables, or oracle residual fittings
were used in deriving these coefficients. The ringdown frequencies and damping
times are independently computed using TorchWave's native Kerr Leaver spectral solver.

References:
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914), Eq. (42).
- SXS Collaboration Gravitational Waveform Catalog (CC-BY-4.0).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union
import torch

# Coordinate for symmetric mass ratio expansion: x = 4*eta - 1 in [-1, 0]
# Frozen ringdown amplitude polynomial coefficients derived from SXS NR:
# Training simulations: SXS:BBH:0008 (q=1.5), SXS:BBH:0169 (q=2), SXS:BBH:0182 (q=4),
# SXS:BBH:0056 (q=5), SXS:BBH:0063 (q=8).
# Held-out validation: SXS:BBH:0030 (q=3), SXS:BBH:0166 (q=6).

SIGMA_21: float = 1.30
SIGMA_33: float = 1.30
SIGMA_44: float = 1.33
SIGMA_32: float = 1.33

SIGMA_DEFAULTS: Dict[Tuple[int, int], float] = {
    (2, 1): SIGMA_21,
    (3, 3): SIGMA_33,
    (4, 4): SIGMA_44,
    (3, 2): SIGMA_32,
}

# (2, 1) Odd-m: log(|a_lambda| / delta) = sum(c_k * x^k)
LOG_A_LAMBDA_OVER_DELTA_COEFFS_21 = (
    -5.599935203250517,
    0.5331787158448524,
    -0.5309772976182556,
)
LAMBDA_COEFFS_21 = (
    0.6928259803381669,
    0.16099600120013763,
    -0.005146334634954575,
)

# (3, 3) Odd-m: log(|a_lambda| / delta) = sum(c_k * x^k)
LOG_A_LAMBDA_OVER_DELTA_COEFFS_33 = (
    -4.512187355706956,
    0.971670037661788,
    -0.5697358841889637,
)
LAMBDA_COEFFS_33 = (
    0.7937140739566705,
    0.21857482324801583,
    0.19316055184011058,
)

# (4, 4) Even-m: log(a_lambda / sqrt(eta - 3*eta^2)) = sum(c_k * x^k)
LOG_A_OVER_SQRT_COEFFS_44 = (
    -6.150473388905861,
    -3.6679249666477856,
    -2.434764928491926,
)
LAMBDA_COEFFS_44 = (
    1.4321238189902044,
    2.691792040467306,
    2.757894912452159,
)

# (3, 2) Even-m: log(|a_lambda|) = sum(c_k * x^k)
LOG_A_COEFFS_32 = (
    -8.446811851780064,
    -2.0437178995570346,
    0.7145312789414549,
)
LAMBDA_COEFFS_32 = (
    1.079415126810816,
    1.9128275656286753,
    1.878118156188172,
)


def _eval_poly_x(coeffs: Tuple[float, ...], x: torch.Tensor) -> torch.Tensor:
    """Evaluate polynomial sum(coeffs[k] * x^k) with Horner's method."""
    result = torch.full_like(x, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        result = result * x + c
    return result


def eval_native_21_ringdown_params(
    eta: torch.Tensor,
    delta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute native SXS-calibrated (2, 1) ringdown parameters (|a_lambda|, lambda)."""
    x = 4.0 * eta - 1.0
    log_a_over_delta = _eval_poly_x(LOG_A_LAMBDA_OVER_DELTA_COEFFS_21, x)
    a_lambda = delta.abs() * torch.exp(log_a_over_delta)
    lambda_param = _eval_poly_x(LAMBDA_COEFFS_21, x)
    return a_lambda, lambda_param


def eval_native_33_ringdown_params(
    eta: torch.Tensor,
    delta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute native SXS-calibrated (3, 3) ringdown parameters (|a_lambda|, lambda)."""
    x = 4.0 * eta - 1.0
    log_a_over_delta = _eval_poly_x(LOG_A_LAMBDA_OVER_DELTA_COEFFS_33, x)
    a_lambda = delta.abs() * torch.exp(log_a_over_delta)
    lambda_param = _eval_poly_x(LAMBDA_COEFFS_33, x)
    return a_lambda, lambda_param


def eval_native_44_ringdown_params(
    eta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute native SXS-calibrated (4, 4) ringdown parameters (|a_lambda|, lambda)."""
    x = 4.0 * eta - 1.0
    sqrt_factor = torch.sqrt(torch.clamp(eta - 3.0 * eta**2, min=0.0))
    log_a_over_sqrt = _eval_poly_x(LOG_A_OVER_SQRT_COEFFS_44, x)
    a_lambda = sqrt_factor * torch.exp(log_a_over_sqrt)
    lambda_param = _eval_poly_x(LAMBDA_COEFFS_44, x)
    return a_lambda, lambda_param


def eval_native_32_ringdown_params(
    eta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute native SXS-calibrated (3, 2) ringdown parameters (|a_lambda|, lambda)."""
    x = 4.0 * eta - 1.0
    log_a = _eval_poly_x(LOG_A_COEFFS_32, x)
    a_lambda = torch.exp(log_a)
    lambda_param = _eval_poly_x(LAMBDA_COEFFS_32, x)
    return a_lambda, lambda_param


def eval_native_ringdown_params(
    l: int,
    m: int,
    eta: torch.Tensor,
    delta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch native SXS-calibrated ringdown parameters for mode (l, m).

    Args:
        l: Polar spherical harmonic index
        m: Azimuthal spherical harmonic index
        eta: Symmetric mass ratio
        delta: Normalized mass difference (m1 - m2) / (m1 + m2)

    Returns:
        Tuple of (a_lambda, lambda_param) tensors.
    """
    if (l, m) == (2, 1):
        return eval_native_21_ringdown_params(eta, delta)
    elif (l, m) == (3, 3):
        return eval_native_33_ringdown_params(eta, delta)
    elif (l, m) == (4, 4):
        return eval_native_44_ringdown_params(eta)
    elif (l, m) == (3, 2):
        return eval_native_32_ringdown_params(eta)
    else:
        raise ValueError(f"Native NR calibration not supported for mode ({l}, {m})")


def native_ringdown_amplitude(
    l: int,
    m: int,
    f: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    f_ring: Union[float, torch.Tensor],
    f_damp: Union[float, torch.Tensor],
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """Compute native SXS-calibrated ringdown amplitude profile A_RD(f) / A_0^22(f) for mode (l, m).

    Implements the modified Lorentzian ringdown ansatz (Eq. 42 of arXiv:2001.10914)
    with TorchWave's native SXS-calibrated polynomial surfaces.
    """
    if sigma is None:
        sigma = SIGMA_DEFAULTS.get((l, m), 1.30)
    a_lambda, lam = eval_native_ringdown_params(l, m, eta, delta)
    width = f_damp * sigma

    f_diff = f - f_ring
    lorentzian = (f ** (-1.0 / 12.0)) * width / (f_diff ** 2 + width ** 2)
    damping_factor = torch.exp(-f_diff * lam / width)
    return a_lambda * lorentzian * damping_factor


def native_ringdown_amplitude_33(
    f: torch.Tensor,
    eta: torch.Tensor,
    delta: torch.Tensor,
    f_ring_33: Union[float, torch.Tensor],
    f_damp_33: Union[float, torch.Tensor],
    sigma: float = SIGMA_33,
) -> torch.Tensor:
    """Backward-compatible wrapper for native_ringdown_amplitude on (3, 3)."""
    return native_ringdown_amplitude(3, 3, f, eta, delta, f_ring_33, f_damp_33, sigma)

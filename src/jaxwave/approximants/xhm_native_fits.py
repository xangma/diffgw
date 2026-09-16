# SPDX-License-Identifier: MIT
"""Native Numerical Relativity (NR) Calibrated Fits for Higher Harmonics in JAX.

TorchWave Sovereign Calibration ported to JAX:
These parameters were independently fitted to public numerical relativity (NR)
simulations from the Simulating eXtreme Spacetimes (SXS) Collaboration.

References:
- García-Quirós et al. (2020), Phys. Rev. D 102, 064002 (arXiv:2001.10914), Eq. (42).
- SXS Collaboration Gravitational Waveform Catalog (CC-BY-4.0).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union
import jax.numpy as jnp

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

# (2, 1) Phase parameters: alpha_2 = sum(c_k * x^k), alpha_lambda = sum(c_k * x^k)
ALPHA2_COEFFS_21 = (
    10.00000000,
    -30.00000000,
)
ALAMBDA_COEFFS_21 = (
    -4.38174292,
    -4.61940634,
)
T0_OFFSET_21: float = -4.5672

# (3, 3) Phase parameters: alpha_2 = sum(c_k * x^k), alpha_lambda = sum(c_k * x^k)
ALPHA2_COEFFS_33 = (
    37.15011758,
    -7.92405510,
)
ALAMBDA_COEFFS_33 = (
    -1.12971547,
    -0.07975304,
)
T0_OFFSET_33: float = -13.783

# (4, 4) Phase parameters: alpha_2 = sum(c_k * x^k), alpha_lambda = sum(c_k * x^k)
ALPHA2_COEFFS_44 = (
    63.16622136,
    3.62059925,
)
ALAMBDA_COEFFS_44 = (
    -0.43596746,
    5.89463054,
)
T0_OFFSET_44: float = -16.1609

# (3, 2) Phase parameters: alpha_2 = sum(c_k * x^k), alpha_lambda = sum(c_k * x^k)
ALPHA2_COEFFS_32 = (
    39.08863265,
    5.80140909,
)
ALAMBDA_COEFFS_32 = (
    -1.84034153,
    -1.40999930,
)
T0_OFFSET_32: float = -16.8168

T0_OFFSET_DEFAULTS: Dict[Tuple[int, int], float] = {
    (2, 1): T0_OFFSET_21,
    (3, 3): T0_OFFSET_33,
    (4, 4): T0_OFFSET_44,
    (3, 2): T0_OFFSET_32,
}

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


def _eval_poly_x(coeffs: Tuple[float, ...], x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate polynomial sum(coeffs[k] * x^k) with Horner's method."""
    result = jnp.full_like(x, coeffs[-1])
    for c in reversed(coeffs[:-1]):
        result = result * x + c
    return result


def eval_native_21_ringdown_params(
    eta: jnp.ndarray,
    delta: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute native SXS-calibrated (2, 1) ringdown parameters (|a_lambda|, lambda)."""
    x = 4.0 * eta - 1.0
    log_a_over_delta = _eval_poly_x(LOG_A_LAMBDA_OVER_DELTA_COEFFS_21, x)
    a_lambda = jnp.abs(delta) * jnp.exp(log_a_over_delta)
    lambda_param = _eval_poly_x(LAMBDA_COEFFS_21, x)
    return a_lambda, lambda_param


def eval_native_33_ringdown_params(
    eta: jnp.ndarray,
    delta: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute native SXS-calibrated (3, 3) ringdown parameters (|a_lambda|, lambda)."""
    x = 4.0 * eta - 1.0
    log_a_over_delta = _eval_poly_x(LOG_A_LAMBDA_OVER_DELTA_COEFFS_33, x)
    a_lambda = jnp.abs(delta) * jnp.exp(log_a_over_delta)
    lambda_param = _eval_poly_x(LAMBDA_COEFFS_33, x)
    return a_lambda, lambda_param


def eval_native_44_ringdown_params(
    eta: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute native SXS-calibrated (4, 4) ringdown parameters (|a_lambda|, lambda)."""
    x = 4.0 * eta - 1.0
    sqrt_factor = jnp.sqrt(jnp.maximum(eta - 3.0 * eta**2, 0.0))
    log_a_over_sqrt = _eval_poly_x(LOG_A_OVER_SQRT_COEFFS_44, x)
    a_lambda = sqrt_factor * jnp.exp(log_a_over_sqrt)
    lambda_param = _eval_poly_x(LAMBDA_COEFFS_44, x)
    return a_lambda, lambda_param


def eval_native_32_ringdown_params(
    eta: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute native SXS-calibrated (3, 2) ringdown parameters (|a_lambda|, lambda)."""
    x = 4.0 * eta - 1.0
    log_a = _eval_poly_x(LOG_A_COEFFS_32, x)
    a_lambda = jnp.exp(log_a)
    lambda_param = _eval_poly_x(LAMBDA_COEFFS_32, x)
    return a_lambda, lambda_param


def eval_native_ringdown_params(
    l: int,
    m: int,
    eta: jnp.ndarray,
    delta: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Dispatch native SXS-calibrated ringdown parameters for mode (l, m)."""
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
    f: jnp.ndarray,
    eta: jnp.ndarray,
    delta: jnp.ndarray,
    f_ring: Union[float, jnp.ndarray],
    f_damp: Union[float, jnp.ndarray],
    sigma: Optional[float] = None,
) -> jnp.ndarray:
    """Compute native SXS-calibrated ringdown amplitude profile A_RD(f) / A_0^22(f) for mode (l, m)."""
    if sigma is None:
        sigma = SIGMA_DEFAULTS.get((l, m), 1.30)
    a_lambda, lam = eval_native_ringdown_params(l, m, eta, delta)
    width = f_damp * sigma

    f_diff = f - f_ring
    lorentzian = (f ** (-1.0 / 12.0)) * width / (f_diff ** 2 + width ** 2)
    damping_factor = jnp.exp(-f_diff * lam / width)
    return a_lambda * lorentzian * damping_factor


def eval_native_21_phase_params(
    eta: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute native SXS-calibrated (2, 1) ringdown phase parameters (alpha_2, alpha_lambda)."""
    x = 4.0 * eta - 1.0
    alpha_2 = _eval_poly_x(ALPHA2_COEFFS_21, x)
    alpha_lambda = _eval_poly_x(ALAMBDA_COEFFS_21, x)
    return alpha_2, alpha_lambda


def eval_native_33_phase_params(
    eta: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute native SXS-calibrated (3, 3) ringdown phase parameters (alpha_2, alpha_lambda)."""
    x = 4.0 * eta - 1.0
    alpha_2 = _eval_poly_x(ALPHA2_COEFFS_33, x)
    alpha_lambda = _eval_poly_x(ALAMBDA_COEFFS_33, x)
    return alpha_2, alpha_lambda


def eval_native_44_phase_params(
    eta: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute native SXS-calibrated (4, 4) ringdown phase parameters (alpha_2, alpha_lambda)."""
    x = 4.0 * eta - 1.0
    alpha_2 = _eval_poly_x(ALPHA2_COEFFS_44, x)
    alpha_lambda = _eval_poly_x(ALAMBDA_COEFFS_44, x)
    return alpha_2, alpha_lambda


def eval_native_32_phase_params(
    eta: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute native SXS-calibrated (3, 2) ringdown phase parameters (alpha_2, alpha_lambda)."""
    x = 4.0 * eta - 1.0
    alpha_2 = _eval_poly_x(ALPHA2_COEFFS_32, x)
    alpha_lambda = _eval_poly_x(ALAMBDA_COEFFS_32, x)
    return alpha_2, alpha_lambda


def eval_native_phase_params(
    l: int,
    m: int,
    eta: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Dispatch native SXS-calibrated ringdown phase parameters for mode (l, m)."""
    if (l, m) == (2, 1):
        return eval_native_21_phase_params(eta)
    elif (l, m) == (3, 3):
        return eval_native_33_phase_params(eta)
    elif (l, m) == (4, 4):
        return eval_native_44_phase_params(eta)
    elif (l, m) == (3, 2):
        return eval_native_32_phase_params(eta)
    else:
        raise ValueError(f"Native NR phase calibration not supported for mode ({l}, {m})")


def native_ringdown_phase(
    l: int,
    m: int,
    f: jnp.ndarray,
    eta: jnp.ndarray,
    f_ring: Union[float, jnp.ndarray],
    f_damp: Union[float, jnp.ndarray],
    include_arrival_time: bool = True,
) -> jnp.ndarray:
    """Compute native SXS-calibrated ringdown phase profile phi_RD(f) for mode (l, m)."""
    alpha_2, alpha_lambda = eval_native_phase_params(l, m, eta)
    f_safe = jnp.maximum(f, 1e-12)
    inv_term = -(f_ring ** 2) / f_safe
    atan_term = jnp.arctan((f - f_ring) / f_damp)
    phi = alpha_2 * inv_term + alpha_lambda * atan_term
    if include_arrival_time:
        t0 = T0_OFFSET_DEFAULTS.get((l, m), 0.0)
        phi = phi + 2.0 * math.pi * f * t0
    return phi

"""Scalar reference equations for published PhenomXPNR angle components.

This module is independent of Torch, JAX, LALSuite, and the repository's
inherited calibration arrays. The caller supplies polynomial coefficients or
already-evaluated calibration polynomials. It implements only relations printed
in Hamilton et al., arXiv:2507.02604v1, Eqs. (3)--(8), (15)--(16), (22)--(24),
and Thompson et al., arXiv:2312.10025v1, Eqs. (25)--(35), (44), and
(50)--(52), including the stated fit guards.  The window helper records the
endpoint-completing shifted-cosine interpretation used by both backends; the
printed Eq. (38) omits the factor of pi needed to reach zero at its endpoint.

The single-spin map must receive spins evolved to the SpinTaylor stopping /
merger-ringdown transition frequency. Supplying reference-frequency spins is a
different numerical prescription.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


SOURCE_XPNR = "https://arxiv.org/html/2507.02604v1"
SOURCE_XO4A = "https://arxiv.org/html/2312.10025v1"


@dataclass(frozen=True)
class SingleSpinMap:
    """Effective single-spin coordinates evaluated at the transition."""

    chi: float
    cos_theta_ls: float
    chi_parallel: float
    chi_perpendicular: float
    chi_effective: float


@dataclass(frozen=True)
class FitCoordinates:
    """Coordinates after the published out-of-calibration freezing rule."""

    eta: float
    chi: float
    cos_theta_ls: float


def _finite(*values: float) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all inputs must be finite")


def single_spin_map_at_transition(
    mass1: float,
    mass2: float,
    spin1: tuple[float, float, float],
    spin2: tuple[float, float, float],
) -> SingleSpinMap:
    """Evaluate Hamilton et al. (2025), Eqs. (3)--(6).

    ``mass1`` is the larger mass and both spin vectors are dimensionless spins
    at the SpinTaylor stopping / merger-ringdown transition frequency. At the
    exactly zero-spin coordinate the paper's angle is undefined; the scalar
    reference uses ``cos_theta_ls=0`` as an explicit numerical extension.
    """
    if len(spin1) != 3 or len(spin2) != 3:
        raise ValueError("each spin must have three components")
    _finite(mass1, mass2, *spin1, *spin2)
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("masses must be positive")
    if mass1 < mass2:
        raise ValueError("mass1 must identify the larger black hole")

    total_mass = mass1 + mass2
    chi_effective = (mass1 * spin1[2] + mass2 * spin2[2]) / total_mass
    chi_parallel = total_mass * chi_effective / mass1
    sx = mass1**2 * spin1[0] + mass2**2 * spin2[0]
    sy = mass1**2 * spin1[1] + mass2**2 * spin2[1]
    chi_perpendicular = math.hypot(sx, sy) / mass1**2
    chi = math.hypot(chi_parallel, chi_perpendicular)
    cos_theta_ls = chi_parallel / chi if chi != 0.0 else 0.0
    return SingleSpinMap(
        chi=chi,
        cos_theta_ls=max(-1.0, min(1.0, cos_theta_ls)),
        chi_parallel=chi_parallel,
        chi_perpendicular=chi_perpendicular,
        chi_effective=chi_effective,
    )


def freeze_fit_coordinates(eta: float, chi: float, cos_theta_ls: float, q: float) -> FitCoordinates:
    """Apply Hamilton et al. (2025), Eq. (22), to the fit coordinates."""
    _finite(eta, chi, cos_theta_ls, q)
    if eta <= 0.0 or eta > 0.25:
        raise ValueError("eta must lie in (0, 0.25]")
    if chi < 0.0 or q < 1.0 or not -1.0 <= cos_theta_ls <= 1.0:
        raise ValueError("require chi >= 0, q >= 1, and cos_theta_ls in [-1, 1]")
    chi_boundary = 0.80 - 0.20 * math.exp(-(((q - 6.0) / 1.5) ** 8))
    return FitCoordinates(max(eta, 0.09), min(chi, chi_boundary), cos_theta_ls)


def evaluate_calibration_polynomial(
    coefficients: Sequence[Sequence[Sequence[float]]],
    eta: float,
    chi: float,
    cos_theta_ls: float,
) -> float:
    """Evaluate XO4a Eq. (35), sum lambda[p,q,r] eta^p chi^q cos(theta)^r."""
    _finite(eta, chi, cos_theta_ls)
    if len(coefficients) != 4 or any(len(row) != 4 for row in coefficients):
        raise ValueError("coefficient tensor must have shape (4, 4, 5)")
    if any(len(column) != 5 for row in coefficients for column in row):
        raise ValueError("coefficient tensor must have shape (4, 4, 5)")
    value = 0.0
    for p in range(4):
        for q_power in range(4):
            for r in range(5):
                coefficient = float(coefficients[p][q_power][r])
                _finite(coefficient)
                value += coefficient * eta**p * chi**q_power * cos_theta_ls**r
    return value


def alpha_parameters(
    chi: float,
    cos_theta_ls: float,
    lambda_a1: float,
    lambda_a2: float,
    lambda_a3: float,
    lambda_a4: float,
) -> tuple[float, float, float, float]:
    """Evaluate XO4a Eqs. (25)--(28) and the published alpha guards."""
    _finite(chi, cos_theta_ls, lambda_a1, lambda_a2, lambda_a3, lambda_a4)
    if chi < 0.0 or not -1.0 <= cos_theta_ls <= 1.0:
        raise ValueError("require chi >= 0 and cos_theta_ls in [-1, 1]")
    transverse_spin = chi * math.sqrt(max(0.0, 1.0 - cos_theta_ls**2))
    a1 = abs(transverse_spin * lambda_a1**2)
    a2 = min(-transverse_spin * lambda_a2**2, 0.0)
    a3 = max(lambda_a3**2, 1.0e-5)
    a2 = max(a2, -(math.pi**2) * math.sqrt(a3))
    return a1, a2, a3, lambda_a4**2


def beta_parameters(
    theta_final: float,
    chi: float,
    cos_theta_ls: float,
    lambda_b0: float,
    lambda_b1: float,
    lambda_b2: float,
    lambda_b3: float,
    lambda_b4: float,
    lambda_b5: float,
) -> tuple[float, float, float, float, float, float]:
    """Evaluate XO4a Eqs. (29)--(34) and the published B4 guard."""
    _finite(
        theta_final,
        chi,
        cos_theta_ls,
        lambda_b0,
        lambda_b1,
        lambda_b2,
        lambda_b3,
        lambda_b4,
        lambda_b5,
    )
    if chi < 0.0 or not -1.0 <= cos_theta_ls <= 1.0:
        raise ValueError("require chi >= 0 and cos_theta_ls in [-1, 1]")
    transverse_spin = chi * math.sqrt(max(0.0, 1.0 - cos_theta_ls**2))
    return (
        theta_final - transverse_spin * lambda_b0,
        transverse_spin * math.exp(lambda_b1),
        -transverse_spin * math.exp(lambda_b2),
        transverse_spin * (math.exp(lambda_b3) - 200.0),
        max(lambda_b4**2, 175.0),
        -(lambda_b5**2),
    )


def alpha_merger_ringdown(mf: float, a1: float, a2: float, a3: float, a4: float) -> float:
    """Evaluate Hamilton et al. (2025), Eq. (15), without its mean offset."""
    _finite(mf, a1, a2, a3, a4)
    if mf <= 0.0 or a3 <= 0.0:
        raise ValueError("mf and a3 must be positive")
    return -(a1 / mf + a2 * math.sqrt(a3) / (a3 + (mf - a4) ** 2))


def beta_merger_ringdown(
    mf: float,
    b0: float,
    b1: float,
    b2: float,
    b3: float,
    b4: float,
    b5: float,
) -> float:
    """Evaluate Hamilton et al. (2025), Eq. (16)."""
    _finite(mf, b0, b1, b2, b3, b4, b5)
    return b0 + (b1 + b2 * mf + b3 * mf**2) / (1.0 + b4 * (mf + b5) ** 2)


def beta_ringdown(theta_final: float, chi: float, cos_theta_ls: float, lambda_bf: float) -> float:
    """Evaluate XO4a Eq. (44)."""
    _finite(theta_final, chi, cos_theta_ls, lambda_bf)
    transverse_spin = chi * math.sqrt(max(0.0, 1.0 - cos_theta_ls**2))
    return theta_final - transverse_spin * lambda_bf


def angle_calibration_window(q: float, chi: float) -> float:
    """Evaluate the backend's endpoint-completing interpretation of Eqs. (38)--(40).

    The paper prints ``cos((x-x_b)/x_w)`` while the implementation uses the
    shifted cosine with a factor of pi, equivalently ``cos(pi/2 * t)**2``.
    This helper intentionally exposes that interpretation rather than calling
    it an exact transcription of the printed equation.
    """
    _finite(q, chi)
    if q < 1.0 or chi < 0.0:
        raise ValueError("require q >= 1 and chi >= 0")

    def window(value: float, boundary: float, width: float) -> float:
        if value <= boundary:
            return 1.0
        if value >= boundary + width:
            return 0.0
        return 0.5 * (math.cos(math.pi * (value - boundary) / width) + 1.0)

    return window(q, 8.5, 3.5) * window(chi, 0.85, 0.35)


def deformed_carrier_coefficient(lambda_k: float, chi: float, cos_theta_ls: float, u_k: float) -> float:
    """Evaluate Hamilton et al. (2025), Eq. (7)."""
    _finite(lambda_k, chi, cos_theta_ls, u_k)
    return lambda_k + chi * math.sqrt(max(0.0, 1.0 - cos_theta_ls**2)) * u_k


def effective_ringdown_angular_frequency(
    omega_lm: float,
    m: int,
    cos_beta_ringdown: float,
    omega_22: float,
    omega_21: float,
) -> float:
    """Evaluate Hamilton et al. (2025), Eq. (8)."""
    _finite(omega_lm, cos_beta_ringdown, omega_22, omega_21)
    if not -1.0 <= cos_beta_ringdown <= 1.0:
        raise ValueError("cos_beta_ringdown must lie in [-1, 1]")
    return omega_lm - m * (1.0 - abs(cos_beta_ringdown)) * (omega_22 - omega_21)


def higher_mode_frequency_map(
    mf: float,
    m: int,
    mf_pn: float,
    mf_mr: float,
    mf_ringdown_22: float,
    mf_ringdown_lm: float,
) -> float:
    """Evaluate XO4a Eqs. (50)--(52), with the XPNR endpoint maps."""
    _finite(mf, mf_pn, mf_mr, mf_ringdown_22, mf_ringdown_lm)
    if m == 0 or mf_mr <= mf_pn:
        raise ValueError("m must be nonzero and mf_mr must exceed mf_pn")
    if m == 2 and mf_ringdown_lm == mf_ringdown_22:
        return mf
    low_value = 2.0 * mf / m
    delta_ringdown = mf_ringdown_lm - mf_ringdown_22
    if mf <= mf_pn:
        return low_value
    if mf > mf_mr:
        return mf - delta_ringdown
    slope = (mf_mr - delta_ringdown - 2.0 * mf_pn / m) / (mf_mr - mf_pn)
    return slope * (mf - mf_pn) + 2.0 * mf_pn / m


__all__ = [
    "FitCoordinates",
    "SOURCE_XO4A",
    "SOURCE_XPNR",
    "SingleSpinMap",
    "alpha_merger_ringdown",
    "alpha_parameters",
    "angle_calibration_window",
    "beta_merger_ringdown",
    "beta_parameters",
    "beta_ringdown",
    "deformed_carrier_coefficient",
    "effective_ringdown_angular_frequency",
    "evaluate_calibration_polynomial",
    "freeze_fit_coordinates",
    "higher_mode_frequency_map",
    "single_spin_map_at_transition",
]

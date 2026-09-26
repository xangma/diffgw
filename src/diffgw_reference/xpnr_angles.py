"""Scalar reference equations for published PhenomXPNR angle components.

This module is independent of Torch, JAX, LALSuite, and the repository's
inherited calibration arrays. The caller supplies polynomial coefficients or
already-evaluated calibration polynomials. It implements only relations printed
in Hamilton et al., arXiv:2507.02604v1, Eqs. (3)--(8), (15)--(24),
Hamilton et al., arXiv:2107.08876v1, Eqs. (41)--(45), and Thompson et al.,
arXiv:2312.10025v1, Eqs. (25)--(35), (44), and (50)--(52), including the
stated fit guards and the corrected factor of two in the emission-direction
map.  The window helper records the
endpoint-completing shifted-cosine interpretation used by both backends; the
printed Eq. (38) omits the factor of pi needed to reach zero at its endpoint.

The single-spin map must receive spins and the Newtonian orbital direction
evolved to the SpinTaylor stopping / merger-ringdown transition frequency.
Supplying reference-frequency spins is a different numerical prescription.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


SOURCE_XPNR = "https://arxiv.org/html/2507.02604v1"
SOURCE_XO4A = "https://arxiv.org/html/2312.10025v1"
SOURCE_FINAL_TWIST = "https://arxiv.org/html/2107.08876v1"


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
    at the SpinTaylor stopping / merger-ringdown transition frequency, in a
    frame whose z axis is the instantaneous Newtonian orbital direction. At
    the exactly zero-spin coordinate the paper's angle is undefined; the
    scalar reference uses ``cos_theta_ls=0`` as an explicit numerical
    extension.
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


def single_spin_map_from_transition_vectors(
    mass1: float,
    mass2: float,
    spin1: tuple[float, float, float],
    spin2: tuple[float, float, float],
    orbital_direction: tuple[float, float, float],
) -> SingleSpinMap:
    """Evaluate XPNR Eqs. (3)--(6) from a terminal SpinTaylor state.

    The dimensionless spins and ``orbital_direction`` may be expressed in any
    common Cartesian frame.  The function applies the paper's definition
    ``S_i_perp = m_i^2 [chi_i - (chi_i . Lhat) Lhat]`` before constructing
    the instantaneous vector sum.  It intentionally does not choose the
    SpinTaylor stopping event: that numerical prescription is a separate,
    unresolved full-model gate.
    """
    if len(spin1) != 3 or len(spin2) != 3 or len(orbital_direction) != 3:
        raise ValueError("each spin and the orbital direction must have three components")
    _finite(mass1, mass2, *spin1, *spin2, *orbital_direction)
    if mass1 <= 0.0 or mass2 <= 0.0:
        raise ValueError("masses must be positive")
    if mass1 < mass2:
        raise ValueError("mass1 must identify the larger black hole")

    orbital_norm = math.sqrt(sum(component * component for component in orbital_direction))
    if orbital_norm == 0.0:
        raise ValueError("orbital direction must be nonzero")
    lhat = tuple(component / orbital_norm for component in orbital_direction)
    parallel1 = sum(component * axis for component, axis in zip(spin1, lhat))
    parallel2 = sum(component * axis for component, axis in zip(spin2, lhat))

    total_mass = mass1 + mass2
    chi_effective = (mass1 * parallel1 + mass2 * parallel2) / total_mass
    chi_parallel = total_mass * chi_effective / mass1
    transverse_sum = tuple(
        mass1**2 * (component1 - parallel1 * axis) + mass2**2 * (component2 - parallel2 * axis)
        for component1, component2, axis in zip(spin1, spin2, lhat)
    )
    chi_perpendicular = math.sqrt(sum(component * component for component in transverse_sum)) / mass1**2
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


def _ordered_positive_frequencies(mf: float, start: float, end: float) -> None:
    _finite(mf, start, end)
    if mf <= 0.0 or start <= 0.0 or end <= start:
        raise ValueError("require mf > 0 and 0 < start < end")


def _hermite(t: float) -> tuple[float, float, float, float]:
    return (2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2)


def alpha_attachment(
    mf: float, start: float, end: float,
    value_start: float, slope_start: float, value_end: float, slope_end: float,
) -> float:
    """XPNR Eq. (19), fixed by values and d(alpha)/d(Mf) at both joins.

    Multiplication by ``mf`` turns the published quadratic-plus-inverse
    expression into an ordinary cubic. Hermite interpolation of that cubic
    is algebraically identical to solving for its four published coefficients.
    Evaluate only on the connection interval.
    """
    _ordered_positive_frequencies(mf, start, end)
    _finite(value_start, slope_start, value_end, slope_end)
    width = end - start
    h00, h10, h01, h11 = _hermite((mf - start) / width)
    q0, q1 = start * value_start, end * value_end
    dq0, dq1 = value_start + start * slope_start, value_end + end * slope_end
    return (h00 * q0 + h10 * width * dq0 + h01 * q1 + h11 * width * dq1) / mf


def beta_connection(
    mf: float, start: float, end: float,
    value_start: float, slope_start: float, value_end: float, slope_end: float,
) -> float:
    """XPNR Eq. (20), the cubic joining PN beta to merger-ringdown beta."""
    _ordered_positive_frequencies(mf, start, end)
    _finite(value_start, slope_start, value_end, slope_end)
    width = end - start
    h00, h10, h01, h11 = _hermite((mf - start) / width)
    return h00 * value_start + h10 * width * slope_start + h01 * value_end + h11 * width * slope_end


def beta_feature_cutoff_frequency(feature_frequency: float) -> float:
    """Evaluate Final Twist Eq. (58), also used by XO4a Eq. (37)."""
    _finite(feature_frequency)
    if feature_frequency >= 0.06:
        return feature_frequency - 0.03
    return 0.6 * feature_frequency


def beta_turnover_connection_frequency(
    f_max: float,
    beta_second_at_max: float,
    beta_third_at_max: float,
    beta_slope_at_inflection: float,
) -> float:
    """Evaluate Final Twist Eq. (57) with XO4a's corrected Eq. (56).

    XO4a corrects the printed ``2.5e-4`` factor to ``25e-4``.  All
    derivatives are with respect to the dimensionless frequency ``Mf``.
    """
    _finite(f_max, beta_second_at_max, beta_third_at_max, beta_slope_at_inflection)
    if beta_third_at_max == 0.0:
        raise ValueError("third derivative at beta maximum must be nonzero")
    slope_connection = 25.0e-4 * beta_slope_at_inflection**2
    discriminant = beta_second_at_max**2 + 2.0 * beta_third_at_max * slope_connection
    if discriminant < 0.0:
        raise ValueError("beta connection-frequency discriminant must be nonnegative")
    return f_max + (
        -beta_second_at_max + math.sqrt(discriminant)
    ) / beta_third_at_max


def beta_lower_connection_frequency(
    f_max: float,
    f_inflection: float,
    f_minimum: float,
    beta_second_at_max: float,
    beta_third_at_max: float,
    beta_slope_at_inflection: float,
) -> float:
    """Select XO4a's lower beta connection from caller-supplied landmarks.

    This implements the three morphology branches described around XO4a
    Eqs. (36)--(37).  It does not locate or validate the merger-ringdown
    ansatz roots, nor apply the later attachment-fallback checks.
    """
    _finite(
        f_max,
        f_inflection,
        f_minimum,
        beta_second_at_max,
        beta_third_at_max,
        beta_slope_at_inflection,
    )
    if f_inflection <= f_max:
        return beta_feature_cutoff_frequency(f_minimum)
    f_low = beta_feature_cutoff_frequency(f_inflection)
    if f_max <= f_low:
        return f_low
    return beta_turnover_connection_frequency(
        f_max,
        beta_second_at_max,
        beta_third_at_max,
        beta_slope_at_inflection,
    )


def tapered_opening_angle(
    mf: float,
    connection: float,
    two_spin_angle: float,
    single_spin_angle: float,
) -> float:
    """Evaluate Hamilton et al. (2021), Eqs. (44)--(45).

    The caller supplies the two-spin and mapped single-spin opening angles at
    ``mf``.  This isolated equation does not choose the connection frequency
    or construct either input track.
    """
    _finite(mf, connection, two_spin_angle, single_spin_angle)
    if mf < 0.0 or connection <= 0.0:
        raise ValueError("require mf >= 0 and connection > 0")
    if mf > connection:
        return single_spin_angle
    taper = math.cos(math.pi * mf / (2.0 * connection)) ** 2
    return single_spin_angle + taper * (two_spin_angle - single_spin_angle)


def optimal_emission_direction_beta(
    v: float,
    iota: float,
    eta: float,
    delta: float,
    chi_perpendicular: float,
    chi_parallel: float,
) -> float:
    """Evaluate Final Twist Eqs. (41)--(43), with the XO4a correction.

    XO4a identifies a missing factor of two in the denominator of the 2021
    printed map.  ``v`` is the PN velocity and ``iota`` is the opening angle
    before the optimal-emission-direction rescaling. The spin components are
    those returned by ``single_spin_map_at_transition``; using them directly
    preserves the zero-spin limit without reconstructing a degenerate angle.
    """
    _finite(v, iota, eta, delta, chi_perpendicular, chi_parallel)
    if v < 0.0 or not 0.0 < eta <= 0.25:
        raise ValueError("require v >= 0 and eta in (0, 0.25]")
    if not 0.0 <= delta <= 1.0 or chi_perpendicular < 0.0:
        raise ValueError("require delta in [0, 1] and chi_perpendicular >= 0")
    if not 0.0 <= iota < math.pi:
        raise ValueError("require iota in [0, pi)")

    sin_iota = math.sin(iota)
    cos_iota = math.cos(iota)
    sin_half_iota = math.sin(0.5 * iota)
    cos_half_iota = math.cos(0.5 * iota)
    mass_spin = (6.0 + 6.0 * delta + 5.0 * eta) * chi_perpendicular
    aligned_spin = (1.0 + delta - eta) * chi_parallel

    c0 = 84.0 * sin_iota
    c2 = (110.0 * eta - 214.0) * sin_iota
    c3 = (
        -7.0 * mass_spin * (2.0 * cos_iota - 1.0)
        + 56.0 * (3.0 * math.pi - aligned_spin) * sin_iota
    )
    d0 = 84.0 * cos_half_iota
    d2 = (110.0 * eta - 214.0) * cos_half_iota
    d3 = (
        14.0 * mass_spin * sin_half_iota
        + 56.0 * cos_half_iota * (3.0 * math.pi - aligned_spin)
    )
    denominator = 2.0 * (d0 + d2 * v**2 + d3 * v**3)
    if denominator == 0.0:
        raise ValueError("emission-direction denominator must be nonzero")
    numerator = (c0 + c2 * v**2 + c3 * v**3) / cos_half_iota
    return 2.0 * math.atan(numerator / denominator)


def alpha_exterior_continuation(
    mf: float, start: float, end: float,
    value_start: float, value_end: float, slope_end: float,
) -> float:
    """XPNR Eq. (17), matched at two PN values and the latter derivative.

    The published expression is quadratic in ``u = (Mf)**-2``. Its three
    coefficients follow from two values and ``d(alpha)/d(Mf)`` at ``end``.
    """
    _ordered_positive_frequencies(mf, start, end)
    _finite(value_start, value_end, slope_end)
    u, u0, u1 = mf**-2, start**-2, end**-2
    derivative_u1 = -0.5 * end**3 * slope_end
    delta = u0 - u1
    curvature = (value_start - value_end - derivative_u1 * delta) / delta**2
    return value_end + derivative_u1 * (u - u1) + curvature * (u - u1) ** 2


def beta_exterior_continuation(
    mf: float, start: float, end: float,
    value_start: float, value_end: float, slope_end: float,
    beta_ringdown_value: float, kappa: float,
) -> float:
    """XPNR Eq. (18), matched at two PN values and the latter derivative.

    ``kappa = 2*pi*Delta f_damp`` in consistent inverse-Mf units. Multiplying
    ``beta-beta_RD`` by ``exp(kappa*(Mf-end))*(Mf)**2`` gives a quadratic in
    ``u = 1/Mf``. The exponential is centred at ``end`` for conditioning.
    """
    _ordered_positive_frequencies(mf, start, end)
    _finite(value_start, value_end, slope_end, beta_ringdown_value, kappa)
    u, u0, u1 = 1 / mf, 1 / start, 1 / end
    r0 = (value_start - beta_ringdown_value) * math.exp(kappa * (start - end)) * start**2
    r1 = (value_end - beta_ringdown_value) * end**2
    derivative_u1 = -end**4 * (
        slope_end + (value_end - beta_ringdown_value) * (kappa + 2 / end)
    )
    delta = u0 - u1
    curvature = (r0 - r1 - derivative_u1 * delta) / delta**2
    r = r1 + derivative_u1 * (u - u1) + curvature * (u - u1) ** 2
    return beta_ringdown_value + math.exp(-kappa * (mf - end)) * r / mf**2


def beta_final_rescaling(
    mf: float, connection: float, beta_value: float,
    beta_at_connection: float, beta_slope_at_connection: float,
    target_value: float, target_slope: float,
) -> float:
    """XPNR Eq. (21), ``k(f) beta(f)`` with a C1 connection.

    The quadratic ``k(f) = 1 + k1*f + k2*f²`` is determined by the value
    and derivative of the already-tapered, emission-direction beta at the
    merger-ringdown connection. The caller supplies that base beta and its
    connection data; this does not specify the preceding taper or rescaling.
    """
    _finite(mf, connection, beta_value, beta_at_connection,
            beta_slope_at_connection, target_value, target_slope)
    if mf < 0.0 or connection <= 0.0 or beta_at_connection == 0.0:
        raise ValueError("require mf >= 0, connection > 0, and nonzero base beta at connection")
    k_connection = target_value / beta_at_connection
    dk_connection = (target_slope - k_connection * beta_slope_at_connection) / beta_at_connection
    k2 = (dk_connection * connection - (k_connection - 1.0)) / connection**2
    k1 = dk_connection - 2.0 * k2 * connection
    return beta_value * (1.0 + k1 * mf + k2 * mf**2)


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
    "SOURCE_FINAL_TWIST",
    "SingleSpinMap",
    "alpha_attachment",
    "alpha_exterior_continuation",
    "alpha_merger_ringdown",
    "alpha_parameters",
    "angle_calibration_window",
    "beta_merger_ringdown",
    "beta_connection",
    "beta_exterior_continuation",
    "beta_final_rescaling",
    "beta_parameters",
    "beta_ringdown",
    "deformed_carrier_coefficient",
    "effective_ringdown_angular_frequency",
    "evaluate_calibration_polynomial",
    "freeze_fit_coordinates",
    "higher_mode_frequency_map",
    "optimal_emission_direction_beta",
    "single_spin_map_at_transition",
    "single_spin_map_from_transition_vectors",
    "tapered_opening_angle",
]

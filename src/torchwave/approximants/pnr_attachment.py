"""Differentiable published XPNR attachment preprocessing and angle joins.

These component kernels do not choose SpinTaylor stop events or change public
XPNR dispatch. Their boundary data must come from a separately qualified PN
trajectory and merger-ringdown ansatz.
"""

from __future__ import annotations

import torch


def _hermite(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2)


def alpha_attachment(mf, start, end, value_start, slope_start, value_end, slope_end):
    """Eq. (19): ``a0*f² + a1*f + a2 + a3/f`` with C1 joins."""
    width = end - start
    h00, h10, h01, h11 = _hermite((mf - start) / width)
    q0, q1 = start * value_start, end * value_end
    dq0, dq1 = value_start + start * slope_start, value_end + end * slope_end
    return (h00 * q0 + h10 * width * dq0 + h01 * q1 + h11 * width * dq1) / mf


def beta_connection(mf, start, end, value_start, slope_start, value_end, slope_end):
    """Eq. (20): cubic joining PN and merger-ringdown beta with C1 joins."""
    width = end - start
    h00, h10, h01, h11 = _hermite((mf - start) / width)
    return h00 * value_start + h10 * width * slope_start + h01 * value_end + h11 * width * slope_end


def beta_feature_cutoff_frequency(feature_frequency):
    """Final Twist Eq. (58), also used by XO4a Eq. (37)."""
    return torch.where(feature_frequency >= 0.06, feature_frequency - 0.03, 0.6 * feature_frequency)


def beta_turnover_connection_frequency(
    f_max, beta_second_at_max, beta_third_at_max, beta_slope_at_inflection,
):
    """Final Twist Eq. (57) with XO4a's corrected Eq. (56)."""
    slope_connection = 25e-4 * beta_slope_at_inflection**2
    discriminant = beta_second_at_max**2 + 2 * beta_third_at_max * slope_connection
    return f_max + (-beta_second_at_max + torch.sqrt(discriminant)) / beta_third_at_max


def beta_lower_connection_frequency(
    f_max, f_inflection, f_minimum,
    beta_second_at_max, beta_third_at_max, beta_slope_at_inflection,
):
    """XO4a morphology selection from caller-supplied beta landmarks."""
    f_low = beta_feature_cutoff_frequency(f_inflection)
    use_turnover = (f_inflection > f_max) & (f_max > f_low)
    safe_third = torch.where(use_turnover, beta_third_at_max, torch.ones_like(beta_third_at_max))
    turnover = beta_turnover_connection_frequency(
        f_max, beta_second_at_max, safe_third, beta_slope_at_inflection,
    )
    standard = torch.where(use_turnover, turnover, f_low)
    return torch.where(
        f_inflection > f_max,
        standard,
        beta_feature_cutoff_frequency(f_minimum),
    )


def tapered_opening_angle(mf, connection, two_spin_angle, single_spin_angle):
    """Final Twist Eqs. (44)--(45): taper two-spin opening-angle oscillations."""
    taper = torch.cos(torch.pi * mf / (2 * connection)) ** 2
    return torch.where(
        mf <= connection,
        single_spin_angle + taper * (two_spin_angle - single_spin_angle),
        single_spin_angle,
    )


def bounded_opening_angle(beta):
    """Final Twist Eq. (62), XO4a arctangent bound, with p=0.002."""
    center = torch.pi / 2
    p = 0.002
    scale = center ** (1 - p)
    offset = beta - center
    x = torch.abs(offset) / scale
    log_t = torch.log(torch.clamp(x, min=torch.finfo(beta.dtype).tiny)) / p
    t = torch.exp(torch.clamp(log_t, min=-35.0, max=35.0))
    small = offset * (torch.atan(t) / t) ** p
    large = torch.sign(offset) * scale * torch.atan(t) ** p
    return center + torch.where(log_t <= 0, small, large)


def optimal_emission_direction_beta(v, iota, eta, delta, chi_perpendicular, chi_parallel):
    """Final Twist Eqs. (41)--(43), including XO4a's denominator correction."""
    sin_iota = torch.sin(iota)
    cos_iota = torch.cos(iota)
    sin_half_iota = torch.sin(iota / 2)
    cos_half_iota = torch.cos(iota / 2)
    mass_spin = (6 + 6 * delta + 5 * eta) * chi_perpendicular
    aligned_spin = (1 + delta - eta) * chi_parallel
    c0 = 84 * sin_iota
    c2 = (110 * eta - 214) * sin_iota
    c3 = -7 * mass_spin * (2 * cos_iota - 1) + 56 * (3 * torch.pi - aligned_spin) * sin_iota
    d0 = 84 * cos_half_iota
    d2 = (110 * eta - 214) * cos_half_iota
    d3 = 14 * mass_spin * sin_half_iota + 56 * cos_half_iota * (3 * torch.pi - aligned_spin)
    numerator = (c0 + c2 * v**2 + c3 * v**3) / cos_half_iota
    denominator = 2 * (d0 + d2 * v**2 + d3 * v**3)
    return 2 * torch.atan(numerator / denominator)


def alpha_exterior_continuation(mf, start, end, value_start, value_end, slope_end):
    """Eq. (17): quadratic in ``1/f²`` from two PN values and one slope."""
    u, u0, u1 = mf**-2, start**-2, end**-2
    derivative_u1 = -0.5 * end**3 * slope_end
    delta = u0 - u1
    curvature = (value_start - value_end - derivative_u1 * delta) / delta**2
    return value_end + derivative_u1 * (u - u1) + curvature * (u - u1) ** 2


def beta_exterior_continuation(
    mf, start, end, value_start, value_end, slope_end, beta_ringdown_value, kappa,
):
    """Eq. (18): exponential/rational continuation from two PN values and slope."""
    u, u0, u1 = 1 / mf, 1 / start, 1 / end
    r0 = (value_start - beta_ringdown_value) * torch.exp(kappa * (start - end)) * start**2
    r1 = (value_end - beta_ringdown_value) * end**2
    derivative_u1 = -end**4 * (
        slope_end + (value_end - beta_ringdown_value) * (kappa + 2 / end)
    )
    delta = u0 - u1
    curvature = (r0 - r1 - derivative_u1 * delta) / delta**2
    r = r1 + derivative_u1 * (u - u1) + curvature * (u - u1) ** 2
    return beta_ringdown_value + torch.exp(-kappa * (mf - end)) * r / mf**2


def beta_final_rescaling(
    mf, connection, beta_value, beta_at_connection, beta_slope_at_connection,
    target_value, target_slope,
):
    """Eq. (21): quadratic low-frequency-unity beta rescaling with a C1 join."""
    k_connection = target_value / beta_at_connection
    dk_connection = (target_slope - k_connection * beta_slope_at_connection) / beta_at_connection
    k2 = (dk_connection * connection - (k_connection - 1)) / connection**2
    k1 = dk_connection - 2 * k2 * connection
    return beta_value * (1 + k1 * mf + k2 * mf**2)

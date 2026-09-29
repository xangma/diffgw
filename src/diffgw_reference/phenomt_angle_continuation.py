"""Independent scalar PhenomT angle-continuation mathematics.

This module implements the minimal-rotation condition from PhenomTP Eq. (5),
the MECO linear continuation from Eqs. (20a-b), and the ringdown prescription
from Eqs. (24a-c).  It uses adaptive SciPy quadrature and scalar finite
differences; it imports no production backend or angle helper.

The caller supplies raw analytical ``(alpha, beta, gamma)`` values.  This
separates the published continuation rules from the much larger PV223 MSA
formula and permits focused scalar tables around the reference, MECO, peak,
and ringdown boundaries.  A caller with an independently interpolated scalar
table may also supply its exact interpolation derivatives; otherwise the
module uses its own five-point scalar stencil.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import quad


RawAngles = Callable[[float], tuple[float, float, float]]
RawAngleDerivatives = Callable[[float], tuple[float, float, float]]


def scalar_derivative(
    function: Callable[[float], float],
    time: float,
    *,
    relative_step: float = 2e-6,
    absolute_step: float = 2e-8,
) -> float:
    """Differentiate a scalar with a symmetric five-point stencil."""
    scale = max(1.0, abs(float(time)))
    step = max(absolute_step, relative_step * scale)
    return float((
        function(time - 2 * step)
        - 8 * function(time - step)
        + 8 * function(time + step)
        - function(time + 2 * step)
    ) / (12 * step))


def meco_value_and_slope(
    raw_angles: RawAngles,
    meco_time: float,
    raw_derivatives: RawAngleDerivatives | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw ``alpha,beta`` and their independently differentiated MECO slopes."""
    value = np.asarray(raw_angles(float(meco_time))[:2], dtype=float)
    if raw_derivatives is None:
        slope = np.asarray([
            scalar_derivative(lambda time, index=index: raw_angles(time)[index], float(meco_time))
            for index in range(2)
        ])
    else:
        slope = np.asarray(raw_derivatives(float(meco_time))[:2], dtype=float)
    return value, slope


def continued_alpha_beta(
    time: float,
    raw_angles: RawAngles,
    meco_time: float,
    value: np.ndarray | None = None,
    slope: np.ndarray | None = None,
) -> tuple[float, float]:
    """Evaluate raw angles before MECO and their continuous linear extension after it."""
    if time < meco_time:
        alpha, beta, _ = raw_angles(float(time))
        return float(alpha), float(beta)
    if value is None or slope is None:
        value, slope = meco_value_and_slope(raw_angles, meco_time)
    result = value + slope * (float(time) - float(meco_time))
    return float(result[0]), float(result[1])


def minimal_rotation_gamma(
    times,
    alpha_beta: Callable[[float], tuple[float, float]],
    reference_time: float,
    gamma_reference: float,
    *,
    alpha_derivative: Callable[[float], float] | None = None,
    integration_breakpoints=None,
    epsabs: float = 2e-10,
    epsrel: float = 2e-10,
) -> np.ndarray:
    """Integrate ``gamma_dot=-alpha_dot*cos(beta)`` by adaptive quadrature."""
    reference_time = float(reference_time)

    def integrand(time):
        beta = float(alpha_beta(float(time))[1])
        alpha_dot = (
            scalar_derivative(lambda query: alpha_beta(query)[0], float(time))
            if alpha_derivative is None
            else float(alpha_derivative(float(time)))
        )
        return -alpha_dot * np.cos(beta)

    query = np.asarray(times, dtype=float)
    flat = query.reshape(-1)
    lower = min(reference_time, float(np.min(flat)))
    upper = max(reference_time, float(np.max(flat)))
    pieces = [flat, [reference_time]]
    if integration_breakpoints is not None:
        breakpoints = np.asarray(integration_breakpoints, dtype=float).reshape(-1)
        pieces.append(breakpoints[(breakpoints > lower) & (breakpoints < upper)])
    nodes = np.unique(np.concatenate(pieces))
    cumulative = np.zeros(len(nodes))
    for index in range(len(nodes) - 1):
        cumulative[index + 1] = cumulative[index] + quad(
            integrand,
            float(nodes[index]),
            float(nodes[index + 1]),
            epsabs=epsabs,
            epsrel=epsrel,
            limit=50,
        )[0]
    reference_index = int(np.searchsorted(nodes, reference_time))
    indices = np.searchsorted(nodes, flat)
    values = float(gamma_reference) + cumulative[indices] - cumulative[reference_index]
    return values.reshape(query.shape)


def apply_angle_option(
    times,
    raw_angles: RawAngles,
    precession_version: int,
    *,
    reference_time: float,
    meco_time: float,
    ringdown_alpha_slope: float,
    raw_derivatives: RawAngleDerivatives | None = None,
    integration_breakpoints=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct one PV223 suffix from independent scalar continuation rules."""
    times = np.asarray(times, dtype=float)
    flat = times.reshape(-1)
    raw = np.asarray([raw_angles(float(time)) for time in flat], dtype=float)
    suffix = int(precession_version) % 100
    if suffix == 0:
        return tuple(raw[:, index].reshape(times.shape) for index in range(3))
    if suffix not in (10, 11, 21):
        raise ValueError(f"unsupported PhenomT angle suffix {suffix}")

    peak = np.asarray(raw_angles(0.0), dtype=float)
    alpha, beta, gamma = raw.T.copy()
    if suffix == 21:
        meco_value, meco_slope = meco_value_and_slope(
            raw_angles, meco_time, raw_derivatives
        )
        continued = np.asarray([
            continued_alpha_beta(time, raw_angles, meco_time, meco_value, meco_slope)
            for time in flat
        ])
        alpha, beta = continued.T
        peak[:2] = meco_value - meco_slope * meco_time

    if suffix in (11, 21):
        gamma_reference = float(raw_angles(float(reference_time))[2])
        if suffix == 21:
            def alpha_beta(time):
                return continued_alpha_beta(time, raw_angles, meco_time, meco_value, meco_slope)

            def alpha_derivative(time):
                if time >= meco_time:
                    return float(meco_slope[0])
                if raw_derivatives is not None:
                    return float(raw_derivatives(time)[0])
                return scalar_derivative(lambda query: raw_angles(query)[0], time)
        else:
            def alpha_beta(time):
                return tuple(map(float, raw_angles(time)[:2]))

            alpha_derivative = (
                None if raw_derivatives is None
                else lambda time: float(raw_derivatives(time)[0])
            )
        prepeak_times = np.where(flat < 0, flat, 0.0)
        gamma = minimal_rotation_gamma(
            prepeak_times,
            alpha_beta,
            reference_time,
            gamma_reference,
            alpha_derivative=alpha_derivative,
            integration_breakpoints=integration_breakpoints,
        )
        peak[2] = float(minimal_rotation_gamma(
            np.asarray([0.0]), alpha_beta, reference_time, gamma_reference,
            alpha_derivative=alpha_derivative,
            integration_breakpoints=integration_breakpoints,
        )[0])

    post = flat >= 0
    alpha[post] = peak[0] + float(ringdown_alpha_slope) * flat[post]
    beta[post] = peak[1]
    gamma[post] = peak[2] - np.cos(peak[1]) * (alpha[post] - peak[0])
    return tuple(value.reshape(times.shape) for value in (alpha, beta, gamma))


__all__ = [
    "apply_angle_option",
    "continued_alpha_beta",
    "meco_value_and_slope",
    "minimal_rotation_gamma",
    "scalar_derivative",
]

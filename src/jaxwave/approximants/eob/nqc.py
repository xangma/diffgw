"""Standalone (2,2) NQC equations in JAX from Bohé et al., arXiv:1611.03703v3."""

from typing import NamedTuple
import jax.numpy as jnp

from jaxwave.conversions import as_array


class NQCCoefficients(NamedTuple):
    amplitude: jnp.ndarray  # (..., 3), a1, a2, a3
    phase: jnp.ndarray  # (..., 2), b1, b2


class NQCMatchingTargets(NamedTuple):
    amplitude: jnp.ndarray
    amplitude_derivative: jnp.ndarray
    amplitude_second_derivative: jnp.ndarray
    frequency: jnp.ndarray
    frequency_derivative: jnp.ndarray


class NQCCalibration(NamedTuple):
    coefficients: NQCCoefficients
    sampling_condition: jnp.ndarray
    amplitude_condition: jnp.ndarray
    phase_condition: jnp.ndarray
    amplitude_residual: jnp.ndarray
    phase_residual: jnp.ndarray


def _fit_domain(nu: jnp.ndarray, chi: jnp.ndarray):
    nu = as_array(nu, dtype=jnp.float64)
    chi = as_array(chi, dtype=jnp.float64)
    nu, chi = jnp.broadcast_arrays(nu, chi)
    if bool(jnp.any((nu < 0) | (nu > 0.25) | (jnp.abs(chi) > 1.0))):
        raise ValueError("published NQC fits require 0 <= nu <= 1/4 and |chi| <= 1")
    return nu, chi


def seobnrv4_nqc_parameters(m1, m2, chi1, chi2):
    """Return nu and Eq. (2.8)'s chi; invariant under mass scaling and exchange."""
    m1 = as_array(m1, dtype=jnp.float64)
    m2 = as_array(m2, dtype=jnp.float64)
    chi1 = as_array(chi1, dtype=jnp.float64)
    chi2 = as_array(chi2, dtype=jnp.float64)
    m1, m2, chi1, chi2 = jnp.broadcast_arrays(m1, m2, chi1, chi2)
    if bool(jnp.any((m1 <= 0) | (m2 <= 0) | (jnp.abs(chi1) > 1.0) | (jnp.abs(chi2) > 1.0))):
        raise ValueError("masses must be positive and component spins within [-1, 1]")
    x1, x2 = m1 / (m1 + m2), m2 / (m1 + m2)
    return x1 * x2, (x1**2 * chi1 + x2**2 * chi2) / (x1**2 + x2**2)


def _endpoint_quadratic(nu, test_particle, equal_mass, linear):
    lo, hi = 0.001, 0.25
    quadratic = (equal_mass - test_particle - linear * (hi - lo)) / (hi**2 - lo**2)
    return test_particle + linear * (nu - lo) + quadratic * (nu**2 - lo**2)


def seobnrv4_nqc_targets(nu, chi):
    """Published peak targets, Eqs. (A1)-(A12); first amplitude derivative is zero."""
    nu, chi = _fit_domain(nu, chi)
    amplitude_tpl = 1.452857 + chi * (0.166134 + chi * (0.027356 - 0.020073 * chi))
    amplitude_eq = 1.577458 + chi * (-0.007695 + chi * (0.021887 + 0.023268 * chi))
    amplitude_linear = -0.034424 + chi * (-1.218066 + chi * (-0.568373 + 0.401114 * chi))
    amplitude = nu * _endpoint_quadratic(nu, amplitude_tpl, amplitude_eq, amplitude_linear)

    shifted = chi - 1.0
    curvature_tpl = shifted * (0.00239561 + shifted * (-0.00019274 - 0.00029666 * shifted))
    curvature_eq = -0.00412651 + 0.00222400 * chi
    curvature = nu * _endpoint_quadratic(nu, curvature_tpl, curvature_eq, -0.00577654 + 0.00103086 * chi)

    a3 = 10.262073 + 4.0 * (10.262073 - 25.850378) * (nu - 0.25)
    a4 = 7.629922 + 4.0 * (7.629922 - 25.819795) * (nu - 0.25)
    frequency = 0.562679 + (-0.087062 + 0.001743 * chi) * jnp.log(a3 - a4 * chi)

    slope_tpl = -0.011210 + (0.004087 + 0.000633 * chi) * jnp.log(68.474666 - 58.301488 * chi)
    slope_eq = 0.011282 + 0.000287 * chi
    slope = _endpoint_quadratic(nu, slope_tpl, slope_eq, 0.015743 + 0.022442 * chi)
    return NQCMatchingTargets(amplitude, jnp.zeros_like(amplitude), curvature, frequency, slope)


def seobnrv4_nqc_time_offset(nu, chi):
    """Delta t22_peak/M, Eqs. (4.9), (4.15)."""
    nu, chi = _fit_domain(nu, chi)
    nonspinning = 716.044155 * (nu**3) - 13.087878 * (nu**2) - 45.883834 * nu - 2.504992
    return (
        nonspinning
        - 0.192775 * (chi**3) * (nu**2)
        + 19.053803 * (chi**3) * nu
        - 11.543497 * (chi**2)
        + 40.318332 * chi * nu
        - 13.006363 * chi
    )


_NQC_TIME_OFFSET_SIX_SIG = {
    "nu3": 716.044,
    "nu2": -13.0879,
    "nu1": -45.8838,
    "constant": -2.50499,
    "chi3_nu2": -0.192775,
    "chi3_nu": 19.0538,
    "chi2": -11.5435,
    "chi_nu": 40.3183,
    "chi": -13.0064,
}


def seobnrv4_nqc_time_offset_numerical_compatibility(nu, chi):
    """Evaluate NQC peak offset with six-significant-digit coefficients."""
    nu, chi = _fit_domain(nu, chi)
    c = _NQC_TIME_OFFSET_SIX_SIG
    nonspinning = c["nu3"] * (nu**3) + c["nu2"] * (nu**2) + c["nu1"] * nu + c["constant"]
    return (
        nonspinning
        + c["chi3_nu2"] * (chi**3) * (nu**2)
        + c["chi3_nu"] * (chi**3) * nu
        + c["chi2"] * (chi**2)
        + c["chi_nu"] * chi * nu
        + c["chi"] * chi
    )


def nqc_22_basis(r, pr_star, orbital_frequency):
    """Eq. (2.7)'s real amplitude and phase bases."""
    r = as_array(r, dtype=jnp.float64)
    pr_star = as_array(pr_star, dtype=jnp.float64)
    orbital_frequency = as_array(orbital_frequency, dtype=jnp.float64)
    r, pr_star, orbital_frequency = jnp.broadcast_arrays(r, pr_star, orbital_frequency)

    ratio = pr_star / (r * orbital_frequency)
    amplitude = jnp.stack(
        [ratio**2, (ratio**2) / r, (ratio**2) / (r**1.5)],
        axis=-1,
    )
    phase = jnp.stack([ratio, ratio * (pr_star**2)], axis=-1)
    return amplitude, phase


def nqc_22_factor(r, pr_star, orbital_frequency, coefficients, *, sample_dims=0):
    """Complex N22 factor."""
    amplitude, phase = nqc_22_basis(r, pr_star, orbital_frequency)
    a, b = coefficients.amplitude, coefficients.phase

    a = a.reshape(a.shape[:-1] + (1,) * sample_dims + (3,))
    b = b.reshape(b.shape[:-1] + (1,) * sample_dims + (2,))
    return (1.0 + (amplitude * a).sum(-1)) * jnp.exp(1j * (phase * b).sum(-1))


def _condition(matrix, maximum, name):
    singular = jnp.linalg.svd(matrix, compute_uv=False)
    condition = singular[..., 0] / singular[..., -1]
    return condition


def _matching_solve(matrix, rhs, maximum, name):
    row_scale = jnp.linalg.norm(matrix, axis=-1)
    scaled = matrix / jnp.expand_dims(row_scale, -1)
    column_scale = jnp.linalg.norm(scaled, axis=-2)
    scaled = scaled / jnp.expand_dims(column_scale, -2)

    condition = _condition(scaled, maximum, name)
    solution = (
        jnp.linalg.solve(scaled, jnp.expand_dims(rhs / row_scale, -1)).squeeze(-1)
        / column_scale
    )
    residual = (matrix @ jnp.expand_dims(solution, -1)).squeeze(-1) - rhs
    return solution, condition, residual


def calibrate_nqc_22(
    time,
    r,
    pr_star,
    orbital_frequency,
    amplitude,
    gw_frequency,
    match_time,
    targets,
    *,
    degree=4,
    max_condition=1.0e10,
):
    """Fit NQC coefficients from a trajectory stencil."""
    time = as_array(time, dtype=jnp.float64)
    r = as_array(r, dtype=jnp.float64)
    pr_star = as_array(pr_star, dtype=jnp.float64)
    orbital_frequency = as_array(orbital_frequency, dtype=jnp.float64)
    amplitude = as_array(amplitude, dtype=jnp.float64)
    gw_frequency = as_array(gw_frequency, dtype=jnp.float64)
    match_time = as_array(match_time, dtype=jnp.float64)

    count = time.shape[-1]
    samples = jnp.broadcast_arrays(time, r, pr_star, orbital_frequency, amplitude, gw_frequency)
    batch = jnp.broadcast_shapes(
        samples[0].shape[:-1],
        match_time.shape,
        *(as_array(target).shape for target in targets),
    )

    time, r, pr_star, orbital_frequency, amplitude, gw_frequency = (
        jnp.broadcast_to(s, batch + (count,)) for s in samples
    )
    match_time = jnp.broadcast_to(match_time, batch)
    targets = NQCMatchingTargets(*(jnp.broadcast_to(as_array(t), batch) for t in targets))

    q, p = nqc_22_basis(r, pr_star, orbital_frequency)
    span = time[..., -1] - time[..., 0]
    x = (time - jnp.expand_dims(match_time, -1)) / jnp.expand_dims(span, -1)
    vandermonde = jnp.stack([x**k for k in range(degree + 1)], axis=-1)
    sampling_condition = _condition(vandermonde, max_condition, "local sample polynomial")

    ordinates = jnp.concatenate(
        [
            jnp.expand_dims(amplitude, -1),
            jnp.expand_dims(amplitude, -1) * q,
            p,
            jnp.expand_dims(gw_frequency, -1),
        ],
        axis=-1,
    )
    polynomial, _, _, _ = jnp.linalg.lstsq(vandermonde, ordinates)

    value = polynomial[..., 0, :]
    first = polynomial[..., 1, :] / jnp.expand_dims(span, -1)
    second = 2.0 * polynomial[..., 2, :] / jnp.expand_dims(span**2, -1)

    amplitude_matrix = jnp.stack([value[..., 1:4], first[..., 1:4], second[..., 1:4]], axis=-2)
    amplitude_rhs = jnp.stack(targets[:3], axis=-1) - jnp.stack(
        [value[..., 0], first[..., 0], second[..., 0]], axis=-1
    )
    phase_matrix = jnp.stack([first[..., 4:6], second[..., 4:6]], axis=-2)
    phase_rhs = jnp.stack(
        [value[..., 6] - targets.frequency, first[..., 6] - targets.frequency_derivative],
        axis=-1,
    )

    a, amplitude_condition, amplitude_residual = _matching_solve(
        amplitude_matrix, amplitude_rhs, max_condition, "amplitude matching system"
    )
    b, phase_condition, phase_residual = _matching_solve(
        phase_matrix, phase_rhs, max_condition, "phase matching system"
    )

    return NQCCalibration(
        NQCCoefficients(a, b),
        sampling_condition,
        amplitude_condition,
        phase_condition,
        amplitude_residual,
        phase_residual,
    )


__all__ = [
    "NQCCoefficients",
    "NQCMatchingTargets",
    "NQCCalibration",
    "seobnrv4_nqc_parameters",
    "seobnrv4_nqc_targets",
    "seobnrv4_nqc_time_offset",
    "seobnrv4_nqc_time_offset_numerical_compatibility",
    "nqc_22_basis",
    "nqc_22_factor",
    "calibrate_nqc_22",
]

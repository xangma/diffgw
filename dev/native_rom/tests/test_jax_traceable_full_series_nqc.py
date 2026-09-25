"""Fixed-grid value and derivative checks for development-only full-series NQC."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dev.native_rom.eob.jax.numerical_attachment_clean import (
    calibrate_nqc_22_full_series,
    traceable_nqc_22_full_series_coefficients,
)
from jaxwave.approximants.eob.nqc import (
    NQCMatchingTargets,
    _seobnrv4_nqc_targets_unchecked,
    _seobnrv4_nqc_time_offset_numerical_compatibility_unchecked,
    seobnrv4_nqc_targets,
    seobnrv4_nqc_time_offset_numerical_compatibility,
)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("nu,chi", [(0.16, 0.0), (0.01, -0.8), (0.25, 0.95)])
def test_traceable_target_arithmetic_preserves_checked_public_fit(nu, chi):
    checked = seobnrv4_nqc_targets(nu, chi)
    traceable = jax.jit(lambda spin: _seobnrv4_nqc_targets_unchecked(nu, spin))(chi)
    for actual, reference in zip(traceable, checked):
        np.testing.assert_allclose(actual, reference, rtol=1e-14, atol=1e-14)
    offset = jax.jit(lambda spin:
        _seobnrv4_nqc_time_offset_numerical_compatibility_unchecked(nu, spin))(chi)
    np.testing.assert_allclose(
        offset, seobnrv4_nqc_time_offset_numerical_compatibility(nu, chi),
        rtol=1e-14, atol=1e-14)
    with pytest.raises(ValueError, match="published NQC fits require"):
        seobnrv4_nqc_targets(nu, 1.01)
    with pytest.raises(ValueError, match="published NQC fits require"):
        seobnrv4_nqc_time_offset_numerical_compatibility(nu, 1.01)


def test_full_series_nqc_fixed_chart_gradient_matches_checked_solve_and_fd():
    time = jnp.linspace(0.0, 8.0, 81)
    match = jnp.asarray(4.13)
    targets = NQCMatchingTargets(0.24, 0.009, -0.0004, 0.073, 0.0008)

    def inputs(perturbation):
        r = (12.0 - 0.5*time + 0.015*time**2) * (1 + 0.005*perturbation)
        pr_star = -0.02 - 0.005*time
        omega = (0.024 + 0.002*time + 0.00005*time**2) * (
            1 + 0.002*perturbation)
        amplitude = 0.20 + 0.004*time + 0.0002*time**2 + 0.01*perturbation
        phase = 0.7 + 0.06*time + 0.001*time**2 + 0.03*perturbation*time**2
        return r, pr_star, omega, amplitude*jnp.exp(-1j*phase)

    def fit(perturbation):
        r, pr_star, omega, mode = inputs(perturbation)
        return traceable_nqc_22_full_series_coefficients(
            time, r, pr_star, omega, mode, match, targets)

    r, pr_star, omega, mode = inputs(0.0)
    checked = calibrate_nqc_22_full_series(
        time, r, pr_star, omega, jnp.abs(mode), omega, match, targets,
        base_mode=mode)
    eager = fit(0.0)
    compiled = jax.jit(fit)(0.0)
    expected = (checked.coefficients.amplitude, checked.coefficients.phase,
                checked.amplitude_matching_matrix, checked.phase_matching_matrix,
                checked.calibration.amplitude_condition,
                checked.calibration.phase_condition,
                checked.calibration.amplitude_residual,
                checked.calibration.phase_residual)
    for actual, jitted, reference in zip(eager, compiled, expected):
        np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(jitted, actual, rtol=1e-10, atol=1e-10)

    derivative = jax.jit(jax.jacfwd(lambda x: fit(x)[:2]))(0.0)
    for step in (1e-4, 1e-5):
        plus, minus = fit(step), fit(-step)
        for tangent, high, low in zip(derivative, plus[:2], minus[:2]):
            np.testing.assert_allclose(
                tangent, (high - low)/(2*step), rtol=2e-5, atol=1e-5)
    assert all(np.isfinite(np.asarray(value)).all() for value in derivative)

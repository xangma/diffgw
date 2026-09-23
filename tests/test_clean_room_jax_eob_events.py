"""Analytic peak and validity checks for the JAX EOB event."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwave.approximants.eob.events import orbital_frequency_peak
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator


def test_degree_six_peak_and_masked_suffix():
    jax.config.update("jax_enable_x64", True)
    time = jnp.linspace(-1.5, 1.5, 31)
    center = jnp.asarray([0.17, -0.23])
    x = time[None, :] - center[:, None]
    frequency = 1.7 - .24*x**2 + .035*x**4 - .012*x**6
    full = orbital_frequency_peak(time, frequency)
    compiled = jax.jit(orbital_frequency_peak)(time, frequency)
    np.testing.assert_allclose(full.time, center, rtol=0, atol=2e-12)
    np.testing.assert_allclose(compiled.time, full.time, rtol=0, atol=2e-12)
    np.testing.assert_allclose(full.frequency, 1.7, rtol=0, atol=2e-12)
    np.testing.assert_allclose(full.curvature, -.48, rtol=0, atol=2e-11)

    valid = jnp.arange(time.size) < 25
    padded = frequency.at[:, 25:].set(0)
    masked = orbital_frequency_peak(time, padded, valid=valid)
    masked_compiled = jax.jit(orbital_frequency_peak)(time, padded, valid=valid)
    np.testing.assert_allclose(masked.time, center, rtol=0, atol=2e-12)
    np.testing.assert_allclose(masked_compiled.time, center, rtol=0, atol=2e-12)
    np.testing.assert_allclose(masked.frequency, 1.7, rtol=0, atol=2e-12)


def test_degree_six_peak_location_gradient():
    jax.config.update("jax_enable_x64", True)
    time = jnp.linspace(-1.5, 1.5, 31)

    def locate(center):
        x = time - center
        return orbital_frequency_peak(time, 1.7 - .24*x**2 + .035*x**4 - .012*x**6).time

    np.testing.assert_allclose(jax.grad(locate)(jnp.asarray(.17)), 1, rtol=0, atol=2e-11)
    np.testing.assert_allclose(jax.jit(jax.grad(locate))(jnp.asarray(.17)), 1, rtol=0, atol=2e-11)


def test_compiled_invalid_row_does_not_mask_valid_row():
    jax.config.update("jax_enable_x64", True)
    time = jnp.linspace(-1.5, 1.5, 31)
    valid_row = 1.7 - .24*(time-.17)**2 + .035*(time-.17)**4 - .012*(time-.17)**6
    unresolved_row = time+2
    result = jax.jit(orbital_frequency_peak)(
        time, jnp.stack((valid_row, unresolved_row)))
    np.testing.assert_allclose(result.time[0], .17, rtol=0, atol=2e-12)
    assert result.sample_index[0] >= 0
    assert np.isnan(result.time[1])
    assert result.sample_index[1] == -1


class AnalyticFlow:
    def __init__(self, center):
        self.center = center

    def equations_of_motion_conservative(self, r, pr_star, pphi):
        return jnp.ones_like(r), jnp.exp(-.1*(r-self.center)**2), jnp.zeros_like(r)


class NoRadiation:
    def radiation_reaction_forces(self, omega, **kwargs):
        return jnp.zeros_like(omega), jnp.zeros_like(omega)


def test_batched_integration_locates_analytic_peak_by_default():
    jax.config.update("jax_enable_x64", True)
    center = jnp.asarray([2.37, 4.28])
    initial = jnp.broadcast_to(jnp.asarray([1., 0., 0., 1.]), (2, 4))
    result = EOBTrajectoryIntegrator(AnalyticFlow(center), NoRadiation()).integrate_to_frequency_peak(
        initial, .1, 50)
    np.testing.assert_allclose(result.peak.time, center-1, rtol=0, atol=2e-6)
    assert result.valid.sum(0)[0] < result.valid.sum(0)[1]


@pytest.mark.parametrize("case", ["endpoint", "short", "gap", "duplicate"])
def test_unresolved_or_invalid_peak_raises(case):
    jax.config.update("jax_enable_x64", True)
    time = jnp.linspace(0, 1, 11)
    frequency = 2 - (time - .4)**2
    valid = None
    if case == "endpoint":
        frequency = time + 1
    elif case == "short":
        time, frequency = time[:4], frequency[:4]
    elif case == "gap":
        valid = jnp.ones_like(time, dtype=bool).at[1].set(False)
    else:
        time = time.at[3].set(time[2])
    with pytest.raises(ValueError):
        orbital_frequency_peak(time, frequency, valid=valid)
    if case != "short":
        compiled = jax.jit(orbital_frequency_peak)(time, frequency, valid=valid)
        assert np.isnan(compiled.time)
        assert compiled.sample_index == -1

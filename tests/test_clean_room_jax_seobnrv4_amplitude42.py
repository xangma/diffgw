"""Transform and domain checks for paper-based pre-NQC (4,2)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwave.approximants.eob.seobnrv4_amplitude42 import (
    seobnrv4_published_amplitude42 as amplitude,
)

jax.config.update("jax_enable_x64", True)


def inputs(omega, x1=.8, chi1=.45, chi2=-.2):
    frequency = jnp.asarray(omega)
    return dict(omega=frequency, omega_zero_radial=frequency*.97,
                heff_over_mu=.93, hreal_over_M=.985,
                x1=x1, x2=1-x1, chi1=chi1, chi2=chi2)


def test_batched_jit_exchange_and_equal_mass_unequal_spin():
    args = inputs(jnp.asarray([.02, .05]), x1=jnp.asarray([[.5], [.8]]))
    actual = amplitude(**args)
    assert actual.shape == (2, 2)
    np.testing.assert_allclose(actual, jax.jit(amplitude)(**args), rtol=3e-15, atol=0)
    assert np.all(np.asarray(actual) > 0)
    swapped = dict(args, x1=args["x2"], x2=args["x1"],
                   chi1=args["chi2"], chi2=args["chi1"])
    np.testing.assert_allclose(actual, amplitude(**swapped), rtol=5e-15, atol=0)


def test_float32_dtype_and_jit_gradient():
    result = amplitude(**inputs(jnp.asarray(.04, dtype=jnp.float32)))
    assert result.dtype == jnp.float32

    def scalar(q):
        return amplitude(q[0], omega_zero_radial=q[1], heff_over_mu=q[2],
                         hreal_over_M=q[3], x1=q[4], x2=1-q[4],
                         chi1=q[5], chi2=q[6])

    point = jnp.asarray([.04, .039, .93, .985, .8, .45, -.2])
    gradient = np.asarray(jax.jit(jax.grad(scalar))(point))
    assert np.all(np.isfinite(gradient))
    step = 1e-6
    for index in range(point.size):
        delta = np.zeros(point.size)
        delta[index] = step
        expected = (float(scalar(point+delta))-float(scalar(point-delta)))/(2*step)
        np.testing.assert_allclose(gradient[index], expected, rtol=3e-8, atol=2e-10)


def test_newtonian_limit_with_energy_source():
    omega = jnp.asarray(1e-9)
    x1, x2 = .7, .3
    nu = x1*x2
    leading = 8/63*math.sqrt(math.pi)*nu*omega**(4/3)*(1-3*nu)
    actual = amplitude(omega, omega_zero_radial=omega, heff_over_mu=1.,
                       hreal_over_M=1., x1=x1, x2=x2)
    np.testing.assert_allclose(actual/leading, 1., rtol=5e-6)


def test_invalid_eager_raises_and_compiled_row_is_masked():
    args = inputs(jnp.asarray([.04, .06]), x1=jnp.asarray([.7, 1.2]))
    with pytest.raises(ValueError):
        amplitude(**args)
    result = jax.jit(amplitude)(**args)
    assert np.isfinite(result[0]) and np.isnan(result[1])

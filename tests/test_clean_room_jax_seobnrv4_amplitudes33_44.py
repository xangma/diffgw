"""Transform and domain checks for paper-based pre-NQC (3,3)/(4,4)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwave.approximants.eob.seobnrv4_amplitudes33_44 import (
    seobnrv4_published_amplitudes33_44 as amplitudes,
)

jax.config.update("jax_enable_x64", True)


def inputs(omega, x1=.8, chi1=.45, chi2=-.2):
    frequency = jnp.asarray(omega)
    return dict(omega=frequency, omega_zero_radial=frequency*.97,
                heff_over_mu=.94, hreal_over_M=.985,
                x1=x1, x2=1-x1, chi1=chi1, chi2=chi2)


def test_batched_jit_exchange_and_equal_mass_unequal_spin():
    args = inputs(jnp.asarray([.02, .05]), x1=jnp.asarray([[.5], [.8]]))
    actual = amplitudes(**args)
    compiled = jax.jit(amplitudes)(**args)
    for result, transformed in zip(actual, compiled):
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, transformed, rtol=3e-15, atol=0)
    assert np.all(np.asarray(actual[0][0]) > 0)
    swapped = dict(args, x1=args["x2"], x2=args["x1"],
                   chi1=args["chi2"], chi2=args["chi1"])
    for result, exchanged in zip(actual, amplitudes(**swapped)):
        np.testing.assert_allclose(result, exchanged, rtol=5e-15, atol=0)


def test_float32_dtype_and_jit_gradient():
    result = amplitudes(**inputs(jnp.asarray(.04, dtype=jnp.float32)))
    assert all(value.dtype == jnp.float32 for value in result)

    def summed(q):
        a33, a44 = amplitudes(q[0], omega_zero_radial=q[1],
                              heff_over_mu=q[2], hreal_over_M=q[3],
                              x1=q[4], x2=1-q[4], chi1=q[5], chi2=q[6])
        return a33+a44

    point = jnp.asarray([.04, .039, .94, .985, .8, .45, -.2])
    gradient = np.asarray(jax.jit(jax.grad(summed))(point))
    assert np.all(np.isfinite(gradient))
    step = 1e-6
    for index in range(point.size):
        delta = np.zeros(point.size)
        delta[index] = step
        expected = (float(summed(point+delta))-float(summed(point-delta)))/(2*step)
        np.testing.assert_allclose(gradient[index], expected, rtol=3e-8, atol=2e-10)


def test_invalid_eager_raises_and_compiled_row_is_masked():
    args = inputs(jnp.asarray([.04, .06]), x1=jnp.asarray([.7, 1.2]))
    with pytest.raises(ValueError):
        amplitudes(**args)
    for value in jax.jit(amplitudes)(**args):
        assert np.isfinite(value[0]) and np.isnan(value[1])

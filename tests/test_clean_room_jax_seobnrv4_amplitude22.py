"""JAX transforms and domain checks for the paper-based pre-NQC 22 mode."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwave.approximants.eob.seobnrv4_amplitude22 import seobnrv4_published_amplitude22 as amplitude22

jax.config.update("jax_enable_x64", True)


def inputs(omega, x1=.8, chi1=.45, chi2=-.2):
    frequency = jnp.asarray(omega)
    return dict(omega=frequency,
                omega_zero_radial=frequency*.97,
                heff_over_mu=.94, hreal_over_M=.985,
                x1=x1, x2=1-x1, chi1=chi1, chi2=chi2)


def test_batched_exchange_endpoints_and_jit():
    args = inputs(jnp.array([.01, .04, .1]), x1=jnp.array([[.5], [.8], [1.]]))
    actual = amplitude22(**args)
    compiled = jax.jit(amplitude22)(**args)
    np.testing.assert_allclose(compiled, actual, rtol=3e-15, atol=0)
    assert actual.shape == (3, 3)
    assert np.all(np.asarray(actual[-1]) == 0)
    exchanged = dict(args, x1=args["x2"], x2=args["x1"],
                     chi1=args["chi2"], chi2=args["chi1"])
    np.testing.assert_allclose(amplitude22(**exchanged), actual, rtol=5e-15, atol=0)


def test_float32_preserves_dtype_and_published_spin_choice():
    args = inputs(jnp.asarray([.02, .05], dtype=jnp.float32))
    actual = amplitude22(**args)
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, jnp.asarray(amplitude22(**inputs(jnp.array([.02, .05]))),
                                               dtype=jnp.float32), rtol=2e-6, atol=0)
    retained = amplitude22(**args, retain_pan_cubic_spin=True)
    assert np.all(np.asarray(retained) != np.asarray(actual))
    with pytest.raises(TypeError):
        amplitude22(**args, retain_pan_cubic_spin=1)


def test_jit_gradients_agree_with_central_differences():
    p = jnp.array([.04, .039, .94, .985, .8, .45, -.2])

    def evaluate(q):
        return amplitude22(q[0], omega_zero_radial=q[1], heff_over_mu=q[2],
                           hreal_over_M=q[3], x1=q[4], x2=1-q[4],
                           chi1=q[5], chi2=q[6])

    gradient = np.asarray(jax.jit(jax.grad(evaluate))(p))
    assert np.all(np.isfinite(gradient)) and np.all(gradient != 0)
    step = 1e-6
    finite_difference = []
    for index in range(p.size):
        delta = np.zeros(p.size)
        delta[index] = step
        finite_difference.append((float(evaluate(p+delta))-float(evaluate(p-delta)))/(2*step))
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize("replacement", [
    dict(omega_zero_radial=0.), dict(hreal_over_M=-.01), dict(x1=.9),
    dict(chi1=1.01), dict(heff_over_mu=float("nan")),
])
def test_invalid_inputs_raise_eager_and_mask_under_jit(replacement):
    args = dict(inputs(.04), **replacement)
    with pytest.raises(ValueError):
        amplitude22(**args)
    assert np.isnan(jax.jit(amplitude22)(**args))

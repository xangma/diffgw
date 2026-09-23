"""JAX transforms and input semantics of the published mode-flux sum."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwave.approximants.eob.seobnrv4_flux import emitted_flux_from_modes

jax.config.update("jax_enable_x64", True)


def test_batched_complex_modes_match_analytic_sum_under_jit():
    omega = jnp.asarray([.01, .03])
    modes = {(2, 2): jnp.asarray([.3+.4j, .2-.1j]),
             (3, 3): jnp.asarray(.07-.02j)}
    actual = jax.jit(emitted_flux_from_modes)(omega, modes)
    expected = ((2*np.asarray(omega))**2*np.abs(np.asarray(modes[2, 2]))**2
                +(3*np.asarray(omega))**2*np.abs(np.asarray(modes[3, 3]))**2)/(8*np.pi)
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)
    assert actual.dtype == jnp.float64


def test_newtonian_22_flux_and_gradient():
    nu = .21

    def flux(omega):
        amplitude = 8*jnp.sqrt(jnp.pi/5)*nu*omega**(2/3)
        return emitted_flux_from_modes(omega, {(2, 2): amplitude})

    omega = jnp.asarray(.02)
    expected = 32/5*nu**2*omega**(10/3)
    np.testing.assert_allclose(jax.jit(flux)(omega), expected, rtol=3e-15, atol=0)
    np.testing.assert_allclose(jax.jit(jax.grad(flux))(omega),
                               (10/3)*expected/omega, rtol=3e-15, atol=0)


@pytest.mark.parametrize("modes", [{}, {(1, 1): .1}, {(2, 0): .1}, {(9, 1): .1}])
def test_bad_mode_mapping_is_rejected(modes):
    with pytest.raises(ValueError):
        emitted_flux_from_modes(jnp.asarray(.02), modes)


def test_invalid_compiled_row_is_masked():
    omega = jnp.asarray([.02, -.01])
    modes = {(2, 2): jnp.asarray([.3, .2])}
    with pytest.raises(ValueError):
        emitted_flux_from_modes(omega, modes)
    actual = jax.jit(emitted_flux_from_modes)(omega, modes)
    assert np.isfinite(actual[0]) and np.isnan(actual[1])

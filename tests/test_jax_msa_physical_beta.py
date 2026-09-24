"""Independent scalar-reference checks for the nonpublic physical MSA limit."""

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC
from diffgw_reference.xp_msa_regularized_beta import one_spin_aligned_beta as scalar_beta
from jaxwave.approximants.precession_msa_physical_beta import (
    one_spin_aligned_beta,
    one_spin_aligned_beta_newtonian_grid,
)


@pytest.mark.parametrize("masses", ((40.0, 20.0), (30.0, 30.0)))
@pytest.mark.parametrize("azimuth", (-2.0, 0.4, 2.1))
def test_exact_and_tiny_spin_match_scalar_reference(masses, azimuth):
    m1, m2 = masses
    spin2 = (0.0, 0.0, -0.2)
    exact = one_spin_aligned_beta(
        m1, m2, (0.0, 0.0, 0.4), spin2, v_ref=0.25, l_beta=0.7,
    )
    assert bool(exact["valid"])
    assert float(exact["beta"]) == 0.0
    assert float(exact["cos_beta"]) == 1.0
    for epsilon in (1e-3, 1e-6, 1e-9, 1e-12):
        spin1 = (epsilon * math.cos(azimuth), epsilon * math.sin(azimuth), 0.4)
        expected = scalar_beta(m1, m2, spin1, spin2, v_ref=0.25, l_beta=0.7)
        result = one_spin_aligned_beta(
            m1, m2, spin1, spin2, v_ref=0.25, l_beta=0.7,
        )
        assert bool(result["valid"])
        for key in ("eta", "spin_squared", "spin_perpendicular", "s_average",
                    "j_ref_squared", "c1", "j_beta", "cos_beta", "beta"):
            np.testing.assert_allclose(np.asarray(result[key]), getattr(expected, key),
                                       rtol=3e-15, atol=0.0, err_msg=key)


def test_reversed_j_and_undefined_zero_j():
    exact = one_spin_aligned_beta(30.0, 30.0, (0.0, 0.0, -0.8),
                                  (0.0, 0.0, 0.0), v_ref=0.2, l_beta=0.1)
    assert bool(exact["valid"])
    assert float(exact["beta"]) == math.pi
    assert float(exact["cos_beta"]) == -1.0
    tilted = one_spin_aligned_beta(30.0, 30.0, (1e-9, 0.0, -0.8),
                                   (0.0, 0.0, 0.0), v_ref=0.2, l_beta=0.1)
    np.testing.assert_allclose(float(tilted["beta"]),
                               scalar_beta(30.0, 30.0, (1e-9, 0.0, -0.8),
                                           (0.0, 0.0, 0.0), v_ref=0.2,
                                           l_beta=0.1).beta, rtol=0.0, atol=5e-16)
    zero = one_spin_aligned_beta(30.0, 30.0, (0.0, 0.0, -0.8),
                                 (0.0, 0.0, 0.0), v_ref=0.2, l_beta=0.2)
    assert not bool(zero["valid"])
    assert bool(jnp.isnan(zero["beta"]))


def test_jit_batch_and_frequency_zero_bins():
    m1 = jnp.array([40.0, 30.0])
    m2 = jnp.array([20.0, 30.0])
    s1 = jnp.array([[0.0, 0.0, 0.4], [1e-9, 0.0, 0.4]])
    s2 = jnp.array([[0.0, 0.0, -0.2], [0.0, 0.0, -0.2]])
    frequency = jnp.array([0.0, 25.0, 40.0, 80.0])
    evaluate = jax.jit(one_spin_aligned_beta_newtonian_grid)
    result = evaluate(frequency, m1, m2, s1, s2, f_ref=jnp.array([40.0, 40.0]))
    assert result["beta"].shape == (2, 4)
    assert np.isfinite(np.asarray(result["beta"])).all()
    assert np.isfinite(np.asarray(result["cos_beta"])).all()
    np.testing.assert_array_equal(np.asarray(result["positive_frequency"]),
                                  [[False, True, True, True]] * 2)
    np.testing.assert_allclose(np.asarray(result["beta"][:, 0]),
                               np.asarray(result["beta"][:, 2]), atol=0.0)
    for row, masses in enumerate(((40.0, 20.0), (30.0, 30.0))):
        for col, f in enumerate((25.0, 40.0, 80.0), start=1):
            total = sum(masses)
            eta = masses[0] * masses[1] / total**2
            v_ref = (math.pi * total * MSUN_SEC * 40.0) ** (1 / 3)
            v = (math.pi * total * MSUN_SEC * f) ** (1 / 3)
            expected = scalar_beta(*masses, tuple(np.asarray(s1[row])),
                                   tuple(np.asarray(s2[row])), v_ref=v_ref,
                                   l_beta=eta / v)
            np.testing.assert_allclose(float(result["beta"][row, col]),
                                       expected.beta, rtol=3e-15, atol=0.0)


def test_gradients_at_exact_and_tiny_transverse_spin():
    def beta_of_parameters(parameters):
        x, z, lb = parameters
        return one_spin_aligned_beta(
            40.0, 20.0, jnp.stack((x, 0.0, z)),
            jnp.array((0.0, 0.0, -0.2)), v_ref=0.25, l_beta=lb,
        )["beta"]

    exact_gradient = jax.jit(jax.grad(beta_of_parameters))(jnp.array([0.0, 0.4, 0.7]))
    assert np.isfinite(np.asarray(exact_gradient)).all()
    np.testing.assert_allclose(np.asarray(exact_gradient), [0.0, 0.0, 0.0], atol=0.0)

    tiny = jnp.array([1e-9, 0.4, 0.7])
    gradient = jax.jit(jax.grad(beta_of_parameters))(tiny)
    assert np.isfinite(np.asarray(gradient)).all()
    w1 = (40.0 / 60.0) ** 2
    sp = w1 * 1e-9
    jp = 0.7 + w1 * 0.4 - (20.0 / 60.0) ** 2 * 0.2
    j2 = jp**2 + sp**2
    np.testing.assert_allclose(np.asarray(gradient),
                               [w1 * jp / j2, -w1 * sp / j2, -sp / j2],
                               rtol=3e-15, atol=1e-20)

    def masked_loss(x):
        result = one_spin_aligned_beta_newtonian_grid(
            jnp.array([0.0, 25.0, 40.0]), 40.0, 20.0,
            jnp.stack((x, 0.0, 0.4)), jnp.array((0.0, 0.0, -0.2)), f_ref=40.0,
        )
        return jnp.sum(jnp.where(result["positive_frequency"], result["beta"], 0.0))

    assert math.isfinite(float(jax.jit(jax.grad(masked_loss))(1e-9)))

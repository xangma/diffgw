"""Parity tests: torchwave vs jaxwave for hybrid MECO."""

import math
import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants.phenom_meco import (
    hybrid_binding_energy as torch_energy,
    hybrid_meco_velocity as torch_meco_velocity,
    hybrid_meco_frequency as torch_meco_frequency,
)
from jaxwave.approximants.phenom_meco import (
    hybrid_binding_energy as jax_energy,
    hybrid_meco_velocity as jax_meco_velocity,
    hybrid_meco_frequency as jax_meco_frequency,
)


@pytest.mark.parametrize(
    "m1, m2, s1, s2",
    [
        (30.0, 30.0, 0.0, 0.0),
        (35.0, 20.0, 0.3, -0.2),
        (50.0, 10.0, -0.5, 0.4),
        (40.0, 25.0, 0.8, -0.7),
    ],
)
def test_phenom_meco_parity(m1, m2, s1, s2):
    """Verify exact parity between torch and jax for hybrid MECO velocity and frequency."""
    v_t = torch_meco_velocity(m1, m2, s1, s2).item()
    f_t = torch_meco_frequency(m1, m2, s1, s2).item()
    e_t = torch_energy(v_t, m1, m2, s1, s2).item()

    v_j = float(jax_meco_velocity(m1, m2, s1, s2))
    f_j = float(jax_meco_frequency(m1, m2, s1, s2))
    e_j = float(jax_energy(v_j, m1, m2, s1, s2))

    np.testing.assert_allclose(v_j, v_t, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(f_j, f_t, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(e_j, e_t, rtol=1e-12, atol=1e-15)


def test_phenom_meco_jax_jit():
    """Verify JIT compatibility of hybrid_meco_velocity and hybrid_meco_frequency."""
    jitted_v = jax.jit(jax_meco_velocity)
    jitted_f = jax.jit(jax_meco_frequency)

    v = jitted_v(30.0, 25.0, 0.2, -0.1)
    f = jitted_f(30.0, 25.0, 0.2, -0.1)

    assert jnp.isfinite(v)
    assert jnp.isfinite(f)
    assert v > 0.0
    assert f > 0.0

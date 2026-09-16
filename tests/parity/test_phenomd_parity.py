"""Cross-framework parity test: torchwave vs jaxwave for IMRPhenomD."""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants.imrphenomd import IMRPhenomD as TorchPhenomD
from jaxwave.approximants.imrphenomd import IMRPhenomD as JaxPhenomD


@pytest.mark.parametrize("m1,m2", [(30.0, 30.0), (36.0, 29.0), (70.0, 15.0)])
@pytest.mark.parametrize("chi1,chi2", [(0.0, 0.0), (0.4, -0.3), (0.7, 0.7)])
def test_phenomd_remnant_parity(m1, m2, chi1, chi2):
    # PyTorch
    m1_t = torch.tensor(m1, dtype=torch.float64)
    m2_t = torch.tensor(m2, dtype=torch.float64)
    c1_t = torch.tensor(chi1, dtype=torch.float64)
    c2_t = torch.tensor(chi2, dtype=torch.float64)
    t_mf, t_af, t_frd, t_fdamp = TorchPhenomD.remnant_properties(m1_t, m2_t, c1_t, c2_t)

    # JAX
    m1_j = jnp.array(m1, dtype=jnp.float64)
    m2_j = jnp.array(m2, dtype=jnp.float64)
    c1_j = jnp.array(chi1, dtype=jnp.float64)
    c2_j = jnp.array(chi2, dtype=jnp.float64)
    j_mf, j_af, j_frd, j_fdamp = JaxPhenomD.remnant_properties(m1_j, m2_j, c1_j, c2_j)

    np.testing.assert_allclose(float(t_mf), float(j_mf), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(float(t_af), float(j_af), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(float(t_frd), float(j_frd), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(float(t_fdamp), float(j_fdamp), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("m1,m2", [(30.0, 30.0), (36.0, 29.0), (60.0, 15.0)])
@pytest.mark.parametrize("chi1,chi2", [(0.0, 0.0), (0.3, -0.2)])
def test_phenomd_waveform_parity(m1, m2, chi1, chi2):
    f_np = np.linspace(20.0, 512.0, 500, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    # Torch evaluate
    hp_t, hc_t = TorchPhenomD.evaluate(
        f=f_t,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=100.0,
        inclination=0.4,
        phic=0.1,
        tc=0.02,
        f_ref=20.0,
    )

    # JAX evaluate
    hp_j, hc_j = JaxPhenomD.evaluate(
        f=f_j,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=100.0,
        inclination=0.4,
        phic=0.1,
        tc=0.02,
        f_ref=20.0,
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-12, atol=1e-14)


def test_phenomd_jax_jit():
    """Verify that jaxwave IMRPhenomD compiles with jax.jit and supports autodiff."""
    f_j = jnp.linspace(20.0, 512.0, 200, dtype=jnp.float64)

    @jax.jit
    def generate(m1, m2, s1, s2):
        hp, hc = JaxPhenomD.evaluate(
            f=f_j,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            f_ref=20.0,
        )
        return jnp.real(hp).sum()

    val = generate(35.0, 30.0, 0.2, -0.1)
    assert jnp.isfinite(val)

    # Test exact autodiff gradient via JAX
    grad_fn = jax.jit(jax.grad(generate, argnums=(0, 1, 2, 3)))
    dm1, dm2, ds1, ds2 = grad_fn(35.0, 30.0, 0.2, -0.1)
    assert jnp.isfinite(dm1)
    assert jnp.isfinite(dm2)
    assert jnp.isfinite(ds1)
    assert jnp.isfinite(ds2)

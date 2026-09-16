"""Cross-framework parity test: torchwave vs jaxwave for TaylorF2."""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants.taylorf2 import TaylorF2 as TorchTaylorF2
from jaxwave.approximants.taylorf2 import TaylorF2 as JaxTaylorF2


@pytest.mark.parametrize("m1,m2", [(30.0, 20.0), (1.4, 1.4), (50.0, 10.0)])
@pytest.mark.parametrize("chi1,chi2", [(0.0, 0.0), (0.4, -0.2), (0.8, 0.8)])
def test_taylorf2_phase_coefficients_parity(m1, m2, chi1, chi2):
    # PyTorch
    m1_t = torch.tensor(m1, dtype=torch.float64)
    m2_t = torch.tensor(m2, dtype=torch.float64)
    c1_t = torch.tensor(chi1, dtype=torch.float64)
    c2_t = torch.tensor(chi2, dtype=torch.float64)
    torch_coeffs = TorchTaylorF2.phase_coefficients(m1_t, m2_t, c1_t, c2_t)

    # JAX
    m1_j = jnp.array(m1, dtype=jnp.float64)
    m2_j = jnp.array(m2, dtype=jnp.float64)
    c1_j = jnp.array(chi1, dtype=jnp.float64)
    c2_j = jnp.array(chi2, dtype=jnp.float64)
    jax_coeffs = JaxTaylorF2.phase_coefficients(m1_j, m2_j, c1_j, c2_j)

    for tc, jc in zip(torch_coeffs, jax_coeffs):
        np.testing.assert_allclose(
            tc.detach().cpu().numpy(),
            np.asarray(jc),
            rtol=1e-14,
            atol=1e-15,
        )


@pytest.mark.parametrize("m1,m2", [(36.0, 29.0), (1.4, 1.3)])
@pytest.mark.parametrize("chi1,chi2", [(0.2, -0.1), (0.0, 0.0)])
def test_taylorf2_waveform_parity(m1, m2, chi1, chi2):
    f_np = np.linspace(20.0, 300.0, 500, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    # Torch evaluate
    hp_t, hc_t = TorchTaylorF2.evaluate(
        f=f_t,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=100.0,
        inclination=0.5,
        phic=0.2,
        tc=0.01,
        f_isco_cutoff=False,
    )

    # JAX evaluate
    hp_j, hc_j = JaxTaylorF2.evaluate(
        f=f_j,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=100.0,
        inclination=0.5,
        phic=0.2,
        tc=0.01,
        f_isco_cutoff=False,
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-13, atol=1e-15)


def test_taylorf2_jax_jit():
    """Verify that jaxwave TaylorF2 compiles cleanly with jax.jit."""
    f_j = jnp.linspace(20.0, 300.0, 200, dtype=jnp.float64)

    @jax.jit
    def generate(m1, m2, s1, s2):
        return JaxTaylorF2.evaluate(
            f=f_j,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            f_isco_cutoff=False,
        )

    hp, hc = generate(30.0, 20.0, 0.1, -0.1)
    assert hp.shape == f_j.shape
    assert hc.shape == f_j.shape

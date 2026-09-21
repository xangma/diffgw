"""Torch/JAX parity for native PhenomT MSA geometry."""

import numpy as np
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants._phenomt_msa import msa_geometry as torch_geometry
from jaxwave.approximants._phenomt_msa import msa_geometry as jax_geometry


jax.config.update("jax_enable_x64", True)


def test_phenomt_msa_geometry_parity_scalar_and_batch():
    t = np.linspace(-0.03, 0.02, 24)
    f = np.linspace(20.0, 120.0, 24)
    params = dict(
        mass1=torch.tensor([30.0, 20.0], dtype=torch.float64),
        mass2=torch.tensor([20.0, 30.0], dtype=torch.float64),
        spin1x=torch.tensor([0.2, 0.0], dtype=torch.float64),
        spin1y=torch.tensor([0.1, 0.0], dtype=torch.float64),
        spin1z=torch.tensor([0.4, 0.3], dtype=torch.float64),
        spin2x=torch.tensor([-0.1, 0.0], dtype=torch.float64),
        spin2y=torch.tensor([0.2, 0.0], dtype=torch.float64),
        spin2z=torch.tensor([-0.3, 0.2], dtype=torch.float64),
        inclination=torch.tensor([0.8, 0.5], dtype=torch.float64),
        phic=torch.tensor([0.2, 0.1], dtype=torch.float64),
        f_ref=torch.tensor([20.0, 20.0], dtype=torch.float64),
    )
    actual = torch_geometry(torch.as_tensor(t), torch.as_tensor(f), **params)
    expected = jax_geometry(torch.as_tensor(t).numpy(), torch.as_tensor(f).numpy(),
                            **{k: jnp.asarray(v.numpy()) for k, v in params.items()})
    for x, y in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(x.detach()), np.asarray(y), rtol=1e-11, atol=2e-13)

    jitted = jax.jit(lambda tt, ff: jax_geometry(tt, ff, **{
        k: jnp.asarray(v.numpy()) for k, v in params.items()}))(jnp.asarray(t), jnp.asarray(f))
    for x, y in zip(expected, jitted):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-12, atol=1e-13)

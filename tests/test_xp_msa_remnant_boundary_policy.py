"""Regression gates for the XP MSA-remnant boundary policy."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC
from jaxwave.approximants.imrphenom_xp import IMRPhenomXP as JAXXP
from jaxwave.approximants.precession_msa import (
    MSA_FINAL_SPIN_DOMAIN_ERROR as JAX_DOMAIN_ERROR,
    compute_msa_final_spin as jax_msa_final_spin,
)
from torchwave.approximants.imrphenom_xp import IMRPhenomXP as TorchXP
from torchwave.approximants.precession_msa import (
    MSA_FINAL_SPIN_DOMAIN_ERROR as TORCH_DOMAIN_ERROR,
    compute_msa_final_spin as torch_msa_final_spin,
)


F = np.array([25.0, 40.0, 80.0])
BASE = dict(
    mass1=25.0,
    spin1x=0.2,
    spin1y=0.1,
    spin1z=0.3,
    spin2x=-0.1,
    spin2y=0.05,
    spin2z=-0.2,
    distance=150.0,
    inclination=0.7,
    phic=0.3,
    f_ref=20.0,
    phi0=0.0,
    prec_version=223,
    convention=1,
)


def _relative_l2(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


@pytest.mark.parametrize("mass2", (25.0, 24.999999))
def test_explicit_mode3_rejects_equal_and_near_equal_invalid_roots(mass2):
    assert JAX_DOMAIN_ERROR == TORCH_DOMAIN_ERROR
    params = dict(BASE, mass2=mass2, final_spin_mod=3)
    with pytest.raises(ValueError, match="valid nonaligned MSA remnant root"):
        TorchXP()(torch.as_tensor(F, dtype=torch.float64), **params)
    with pytest.raises(
        jax.errors.JaxRuntimeError,
        match="valid nonaligned MSA remnant root",
    ):
        hp, hc = JAXXP.evaluate(jnp.asarray(F), **params)
        jax.block_until_ready((hp, hc))


def test_direct_jax_mode3_eager_error_is_value_error():
    total_mass = BASE["mass1"] + 25.0
    v0 = (np.pi * total_mass * MSUN_SEC * BASE["f_ref"]) ** (1.0 / 3.0)
    with pytest.raises(ValueError, match="valid nonaligned MSA remnant root"):
        jax_msa_final_spin(
            jnp.asarray(v0), jnp.asarray(BASE["mass1"]), jnp.asarray(25.0),
            jnp.asarray(BASE["spin1x"]), jnp.asarray(BASE["spin1y"]),
            jnp.asarray(BASE["spin1z"]), jnp.asarray(BASE["spin2x"]),
            jnp.asarray(BASE["spin2y"]), jnp.asarray(BASE["spin2z"]),
        )


def test_compiled_torch_mode3_invalid_root_uses_runtime_assertion():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    total_mass = BASE["mass1"] + 25.0
    v0 = (np.pi * total_mass * MSUN_SEC * BASE["f_ref"]) ** (1.0 / 3.0)
    values = (
        v0, BASE["mass1"], 25.0,
        BASE["spin1x"], BASE["spin1y"], BASE["spin1z"],
        BASE["spin2x"], BASE["spin2y"], BASE["spin2z"],
    )
    args = tuple(torch.tensor(value, dtype=torch.float64) for value in values)
    compiled = torch.compile(torch_msa_final_spin, backend="eager")
    with pytest.raises(Exception, match="valid nonaligned MSA remnant root"):
        compiled(*args)


def test_mode0_is_continuous_finite_and_differentiable_across_boundary():
    torch_rows = []
    jax_rows = []
    for mass2 in (25.0, 24.999999):
        params = dict(BASE, mass2=mass2, final_spin_mod=0)
        torch_row = TorchXP()(torch.as_tensor(F, dtype=torch.float64), **params)
        jax_row = JAXXP.evaluate(jnp.asarray(F), **params)
        for jax_value, torch_value in zip(jax_row, torch_row):
            assert np.all(np.isfinite(np.asarray(jax_value)))
            assert torch.isfinite(torch_value).all()
            assert _relative_l2(jax_value, torch_value.detach().numpy()) < 2e-10
        torch_rows.append(torch_row)
        jax_rows.append(jax_row)

    for exact, near in zip(torch_rows[0], torch_rows[1]):
        assert _relative_l2(near.detach().numpy(), exact.detach().numpy()) < 2e-6
    for exact, near in zip(jax_rows[0], jax_rows[1]):
        assert _relative_l2(near, exact) < 2e-6

    def jax_loss(mass2):
        hp, hc = JAXXP.evaluate(
            jnp.asarray(F), **dict(BASE, mass2=mass2, final_spin_mod=0)
        )
        return 1e44 * jnp.sum(jnp.abs(hp) ** 2 + jnp.abs(hc) ** 2)

    torch_mass2 = torch.tensor(24.999999, dtype=torch.float64, requires_grad=True)
    hp, hc = TorchXP()(
        torch.as_tensor(F, dtype=torch.float64),
        **dict(BASE, mass2=torch_mass2, final_spin_mod=0),
    )
    torch_gradient = torch.autograd.grad(
        1e44 * (hp.abs().square() + hc.abs().square()).sum(), torch_mass2
    )[0]
    jax_gradient = jax.grad(jax_loss)(jnp.asarray(24.999999))
    assert torch.isfinite(torch_gradient)
    assert np.isfinite(np.asarray(jax_gradient))
    np.testing.assert_allclose(
        np.asarray(jax_gradient), torch_gradient.detach().numpy(),
        rtol=2e-7, atol=1e-9,
    )


def test_jitted_invalid_mode3_surfaces_compiled_callback_limitation():
    def evaluate(mass2):
        return JAXXP.evaluate(
            jnp.asarray(F), **dict(BASE, mass2=mass2, final_spin_mod=3)
        )

    compiled = jax.jit(evaluate)
    with pytest.raises(
        jax.errors.JaxRuntimeError,
        match="valid nonaligned MSA remnant root",
    ):
        hp, hc = compiled(jnp.asarray(24.999999))
        jax.block_until_ready((hp, hc))

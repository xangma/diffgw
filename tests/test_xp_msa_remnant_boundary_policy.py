"""Regression gates for the XP MSA-remnant boundary policy."""

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC
from jaxwave.approximants.imrphenom_xas import IMRPhenomXAS as JAXXAS
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
    with pytest.raises(ValueError, match="valid nonaligned MSA remnant root"):
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


def _mixed_remnant_args():
    # Distinct roots, an aligned equal-mass limit, and a machine-level
    # near-aligned degeneracy are all supported mode-3 rows.
    rows = np.array([
        [35., 25., .2, .1, .3, -.1, .05, -.2],
        [25., 25., 0., 0., .3, 0., 0., -.2],
        [30., 20., 1e-10, 0., .35, -5e-11, 0., -.25],
    ])
    v0 = (np.pi * rows[:, :2].sum(axis=1) * MSUN_SEC * 20.) ** (1. / 3.)
    return (v0, *(rows[:, index] for index in range(rows.shape[1])))


def test_mode3_valid_mixed_batch_survives_jit_vmap_and_torch_compile():
    args = _mixed_remnant_args()
    torch_args = tuple(torch.as_tensor(value) for value in args)
    jax_args = tuple(jnp.asarray(value) for value in args)
    expected = torch_msa_final_spin(*torch_args)
    assert torch.isfinite(expected).all()
    compiled_torch = torch.compile(torch_msa_final_spin, backend="eager")
    torch.testing.assert_close(compiled_torch(*torch_args), expected, rtol=1e-12, atol=0.)
    for evaluate in (
        jax.jit(jax_msa_final_spin),
        jax.jit(jax.vmap(jax_msa_final_spin)),
    ):
        np.testing.assert_allclose(
            np.asarray(evaluate(*jax_args)), expected.numpy(), rtol=1e-12, atol=0.
        )


def test_mode3_invalid_row_rejects_mixed_batch_eager_and_compiled():
    args = [value.copy() for value in _mixed_remnant_args()]
    args[3][1] = .2  # Make the aligned equal-mass row nonaligned.
    torch_args = tuple(torch.as_tensor(value) for value in args)
    jax_args = tuple(jnp.asarray(value) for value in args)
    with pytest.raises(ValueError, match="valid nonaligned MSA remnant root"):
        torch_msa_final_spin(*torch_args)
    with pytest.raises(ValueError, match="valid nonaligned MSA remnant root"):
        jax_msa_final_spin(*jax_args)
    with pytest.raises(Exception, match="valid nonaligned MSA remnant root"):
        torch.compile(torch_msa_final_spin, backend="eager")(*torch_args)
    for evaluate in (
        jax.jit(jax_msa_final_spin),
        jax.jit(jax.vmap(jax_msa_final_spin)),
    ):
        with pytest.raises(jax.errors.JaxRuntimeError, match="valid nonaligned MSA remnant root"):
            jax.block_until_ready(evaluate(*jax_args))


def test_jitted_aligned_mode3_preserves_unanchored_xas_carrier():
    f = jnp.linspace(25., 300., 24)
    params = dict(
        mass1=40., mass2=20., spin1z=.4, spin2z=-.2,
        distance=100., inclination=.7, phic=.3, f_ref=None,
        prec_version=223, convention=1, final_spin_mod=3,
    )
    actual = jax.jit(lambda: JAXXP.evaluate(f, **params))()
    expected = jax.jit(lambda: JAXXAS.evaluate(
        f=f, mass1=40., mass2=20., chi1=.4, chi2=-.2,
        distance=100., inclination=.7, phi_c=.3, f_ref=None,
    ))()
    for left, right in zip(actual, expected):
        assert _relative_l2(left, right) < 1e-12


def test_public_mode3_mixed_batch_checks_only_selected_remnant_routes():
    # The one-spin equal-mass row has its own zero-nutation continuation;
    # its unused distinct-root remnant must not enforce a domain error.
    rows = dict(
        mass1=jnp.array([35., 25., 25.]), mass2=jnp.array([25., 25., 25.]),
        spin1x=jnp.array([.2, 0., .15]), spin1y=jnp.array([.1, 0., .07]),
        spin1z=jnp.array([.3, .3, .35]), spin2x=jnp.array([-.1, 0., 0.]),
        spin2y=jnp.array([.05, 0., 0.]), spin2z=jnp.array([-.2, -.2, -.1]),
    )
    params = dict(distance=150., inclination=.7, phic=.3, f_ref=20.,
                  prec_version=223, convention=1, final_spin_mod=3)

    def evaluate(spin2x):
        return JAXXP.evaluate(jnp.asarray(F), **dict(rows, spin2x=spin2x), **params)

    compiled = jax.jit(evaluate)
    actual = compiled(rows["spin2x"])
    for index in range(3):
        row = {name: value[index] for name, value in rows.items()}
        expected = JAXXP.evaluate(jnp.asarray(F), **row, **params)
        for left, right in zip(actual, expected):
            assert _relative_l2(left[index], right) < 2e-10

    # Activating the second in-plane spin selects the genuinely unsupported
    # equal-mass legacy prescription, even beside supported rows.
    invalid = rows["spin2x"].at[2].set(.01)
    with pytest.raises(ValueError, match="valid nonaligned MSA remnant root"):
        evaluate(invalid)
    with pytest.raises(jax.errors.JaxRuntimeError, match="valid nonaligned MSA remnant root"):
        jax.block_until_ready(compiled(invalid))

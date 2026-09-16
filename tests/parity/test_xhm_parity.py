"""Cross-framework parity test: torchwave vs jaxwave for IMRPhenomXHM."""

import math
import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants.imrphenom_xhm import (
    IMRPhenomXHM as TorchXHM,
    _pn_amplitude_ratio as torch_pn_amp_ratio,
    _published_parameter_amplitude_veto as torch_pub_veto,
    _ordered_amplitude_vetoes as torch_ord_veto,
)
from jaxwave.approximants.imrphenom_xhm import (
    IMRPhenomXHM as JaxXHM,
    _pn_amplitude_ratio as jax_pn_amp_ratio,
    _published_parameter_amplitude_veto as jax_pub_veto,
    _ordered_amplitude_vetoes as jax_ord_veto,
)
from jaxwave import IMRPhenomXHM as RootJaxXHM


def test_imrphenom_xhm_exports_and_pytree():
    """Verify IMRPhenomXHM exports, supported modes, and PyTree flatten/unflatten behavior."""
    assert JaxXHM is RootJaxXHM
    model = JaxXHM()
    assert model.supported_modes == ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
    assert model.modes == ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))

    leaves, treedef = jax.tree_util.tree_flatten(model)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, JaxXHM)
    assert reconstructed.modes == model.modes
    assert reconstructed.calibration == model.calibration

    # Test execution on reconstructed instance
    f = jnp.linspace(30.0, 300.0, 50)
    hp1, hc1 = model(f, mass1=35.0, mass2=20.0, chi1=0.3, chi2=-0.2)
    hp2, hc2 = reconstructed(f, mass1=35.0, mass2=20.0, chi1=0.3, chi2=-0.2)
    np.testing.assert_allclose(np.asarray(hp1), np.asarray(hp2), rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(np.asarray(hc1), np.asarray(hc2), rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("mode", [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)])
def test_individual_mode_parity(mode):
    """Verify mode-by-mode parity between torchwave and jaxwave."""
    f_np = np.linspace(25.0, 500.0, 100)
    f_torch = torch.tensor(f_np, dtype=torch.float64)
    f_jax = jnp.array(f_np)

    m1, m2 = 38.0, 19.0
    chi1, chi2 = 0.45, -0.35
    inc = 0.7
    dist = 150.0

    torch_model = TorchXHM(modes=[mode])
    jax_model = JaxXHM(modes=[mode])

    hp_t, hc_t = torch_model(f_torch, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, inclination=inc, distance=dist)
    hp_j, hc_j = jax_model(f_jax, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, inclination=inc, distance=dist)

    np.testing.assert_allclose(np.asarray(hp_j), hp_t.detach().numpy(), rtol=1e-10, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_j), hc_t.detach().numpy(), rtol=1e-10, atol=1e-30)


@pytest.mark.parametrize(
    "m1, m2, chi1, chi2, inc",
    [
        # Equal mass non-spinning
        (30.0, 30.0, 0.0, 0.0, 0.5),
        # Equal mass symmetric spin
        (25.0, 25.0, 0.5, 0.5, 0.8),
        # Unequal mass, aligned spin
        (40.0, 20.0, 0.3, -0.2, 0.6),
        # Moderate mass ratio, asymmetric spin
        (60.0, 15.0, 0.6, -0.5, 0.4),
        # High mass ratio
        (70.0, 10.0, 0.8, 0.1, 0.9),
    ],
)
def test_full_polarization_parity(m1, m2, chi1, chi2, inc):
    """Verify total h_plus and h_cross parity across multiple physical binary configurations."""
    f_np = np.linspace(20.0, 600.0, 120)
    f_torch = torch.tensor(f_np, dtype=torch.float64)
    f_jax = jnp.array(f_np)

    torch_model = TorchXHM()
    jax_model = JaxXHM()

    hp_t, hc_t = torch_model(f_torch, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, inclination=inc, distance=100.0)
    hp_j, hc_j = jax_model(f_jax, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, inclination=inc, distance=100.0)

    np.testing.assert_allclose(np.asarray(hp_j), hp_t.detach().numpy(), rtol=1e-10, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_j), hc_t.detach().numpy(), rtol=1e-10, atol=1e-30)


def test_equal_mass_symmetry():
    """Verify equal mass symmetry: odd-m modes vanish for equal masses and equal spins."""
    f = jnp.linspace(20.0, 400.0, 80)
    model = JaxXHM()
    hp, hc, modes_dict = model(
        f,
        mass1=30.0,
        mass2=30.0,
        chi1=0.3,
        chi2=0.3,
        inclination=0.7,
        return_modes=True,
    )
    # Odd modes (2,1) and (3,3) must be identically zero
    np.testing.assert_allclose(np.asarray(modes_dict[(2, 1)][0]), 0.0, atol=1e-30)
    np.testing.assert_allclose(np.asarray(modes_dict[(2, 1)][1]), 0.0, atol=1e-30)
    np.testing.assert_allclose(np.asarray(modes_dict[(3, 3)][0]), 0.0, atol=1e-30)
    np.testing.assert_allclose(np.asarray(modes_dict[(3, 3)][1]), 0.0, atol=1e-30)

    # Even modes (2,2), (3,2), (4,4) must be nonzero
    assert np.max(np.abs(np.asarray(modes_dict[(2, 2)][0]))) > 1e-25
    assert np.max(np.abs(np.asarray(modes_dict[(3, 2)][0]))) > 1e-27
    assert np.max(np.abs(np.asarray(modes_dict[(4, 4)][0]))) > 1e-27


def test_mode_32_mixing_parity():
    """Verify mode (3,2) spherical-spheroidal mode mixing matches torchwave."""
    f_np = np.linspace(20.0, 800.0, 150)
    f_torch = torch.tensor(f_np, dtype=torch.float64)
    f_jax = jnp.array(f_np)

    m1, m2 = 50.0, 12.0
    chi1, chi2 = 0.7, -0.4

    torch_model = TorchXHM(modes=[(3, 2)])
    jax_model = JaxXHM(modes=[(3, 2)])

    hp_t, hc_t = torch_model(f_torch, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, inclination=0.8)
    hp_j, hc_j = jax_model(f_jax, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, inclination=0.8)

    np.testing.assert_allclose(np.asarray(hp_j), hp_t.detach().numpy(), rtol=1e-10, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_j), hc_t.detach().numpy(), rtol=1e-10, atol=1e-30)


def test_return_modes_and_unprojected_modes():
    """Verify return_modes and return_unprojected_modes parity with torchwave."""
    f_np = np.linspace(30.0, 400.0, 60)
    f_torch = torch.tensor(f_np, dtype=torch.float64)
    f_jax = jnp.array(f_np)

    m1, m2 = 36.0, 18.0
    chi1, chi2 = 0.3, -0.1
    inc = 0.5

    # 1. return_modes
    torch_model = TorchXHM()
    jax_model = JaxXHM()

    _, _, modes_t = torch_model(f_torch, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, inclination=inc, return_modes=True)
    _, _, modes_j = jax_model(f_jax, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, inclination=inc, return_modes=True)

    for lm in ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4)):
        np.testing.assert_allclose(np.asarray(modes_j[lm][0]), modes_t[lm][0].detach().numpy(), rtol=1e-10, atol=1e-30)
        np.testing.assert_allclose(np.asarray(modes_j[lm][1]), modes_t[lm][1].detach().numpy(), rtol=1e-10, atol=1e-30)

    # 2. return_unprojected_modes
    _, _, unproj_t = torch_model(f_torch, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, return_unprojected_modes=True)
    _, _, unproj_j = jax_model(f_jax, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, return_unprojected_modes=True)

    for lm in ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4)):
        np.testing.assert_allclose(np.asarray(unproj_j[lm]), unproj_t[lm].detach().numpy(), rtol=1e-10, atol=1e-30)
        # Check negative-m symmetry
        l, m = lm
        sign_l = 1.0 if l % 2 == 0 else -1.0
        np.testing.assert_allclose(
            np.asarray(unproj_j[(l, -m)]),
            sign_l * np.conj(np.asarray(unproj_j[lm])),
            rtol=1e-14, atol=1e-30,
        )


def test_jax_jit_compatibility():
    """Verify jax.jit on evaluate function matches eager evaluation."""
    f = jnp.linspace(20.0, 500.0, 80)

    @jax.jit
    def jitted_eval(m1, m2, s1, s2):
        return JaxXHM.evaluate(f, mass1=m1, mass2=m2, chi1=s1, chi2=s2, inclination=0.5)

    hp_eager, hc_eager = JaxXHM.evaluate(f, mass1=40.0, mass2=20.0, chi1=0.2, chi2=-0.3, inclination=0.5)
    hp_jit, hc_jit = jitted_eval(40.0, 20.0, 0.2, -0.3)

    np.testing.assert_allclose(np.asarray(hp_jit), np.asarray(hp_eager), rtol=1e-14, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_jit), np.asarray(hc_eager), rtol=1e-14, atol=1e-30)


def test_autodiff_parity_with_torch():
    """Verify jax.grad matches torch autograd .backward() for parameter derivatives."""
    f_np = np.linspace(30.0, 300.0, 30)
    f_jax = jnp.array(f_np)
    f_torch = torch.tensor(f_np, dtype=torch.float64)

    m1_val, m2_val = 35.0, 20.0
    chi1_val, chi2_val = 0.3, -0.2

    # JAX grad w.r.t mass1
    def jax_loss(m1):
        hp, hc = JaxXHM.evaluate(f_jax, mass1=m1, mass2=m2_val, chi1=chi1_val, chi2=chi2_val)
        return jnp.sum(jnp.abs(hp)**2)

    g_jax = jax.grad(jax_loss)(m1_val)

    # Torch grad w.r.t mass1
    m1_torch = torch.tensor(m1_val, dtype=torch.float64, requires_grad=True)
    torch_model = TorchXHM()
    hp_t, hc_t = torch_model(f_torch, mass1=m1_torch, mass2=m2_val, chi1=chi1_val, chi2=chi2_val)
    loss_t = torch.sum(torch.abs(hp_t)**2)
    loss_t.backward()
    g_torch = m1_torch.grad.item()

    np.testing.assert_allclose(float(g_jax), g_torch, rtol=1e-6)


def test_batched_and_vmap():
    """Verify batched inputs and jax.vmap execution."""
    f = jnp.linspace(30.0, 400.0, 60)
    m1_batch = jnp.array([30.0, 45.0, 50.0])
    m2_batch = jnp.array([20.0, 15.0, 10.0])
    chi1_batch = jnp.array([0.2, -0.3, 0.5])
    chi2_batch = jnp.array([-0.1, 0.4, -0.2])

    # Direct batched call
    hp_batch, hc_batch = JaxXHM.evaluate(
        f,
        mass1=m1_batch,
        mass2=m2_batch,
        chi1=chi1_batch,
        chi2=chi2_batch,
    )
    assert hp_batch.shape == (3, 60)
    assert hc_batch.shape == (3, 60)

    # vmap call
    def single_eval(m1, m2, s1, s2):
        return JaxXHM.evaluate(f, mass1=m1, mass2=m2, chi1=s1, chi2=s2)

    hp_vmap, hc_vmap = jax.vmap(single_eval)(m1_batch, m2_batch, chi1_batch, chi2_batch)
    np.testing.assert_allclose(np.asarray(hp_batch), np.asarray(hp_vmap), rtol=1e-12, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_batch), np.asarray(hc_vmap), rtol=1e-12, atol=1e-30)

"""Cross-framework parity test: torchwave vs jaxwave for TaylorF2_Eccentric."""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants.taylorf2 import TaylorF2 as TorchTaylorF2
from torchwave.approximants.taylorf2_eccentric import TaylorF2_Eccentric as TorchTaylorF2_Eccentric
from jaxwave.approximants.taylorf2 import TaylorF2 as JaxTaylorF2
from jaxwave.approximants.taylorf2_eccentric import TaylorF2_Eccentric as JaxTaylorF2_Eccentric


@pytest.mark.parametrize("m1,m2", [(30.0, 20.0), (1.4, 1.4), (10.0, 5.0)])
@pytest.mark.parametrize("e0", [0.01, 0.05, 0.1, 0.2])
@pytest.mark.parametrize("ecc_order", [2, 4, 6])
@pytest.mark.parametrize("f0", [20.0, 30.0])
def test_eccentric_phase_correction_parity(m1, m2, e0, ecc_order, f0):
    """Verify parity of eccentric_phase_correction between torchwave and jaxwave."""
    f_np = np.linspace(20.0, 300.0, 100, dtype=np.float64)
    f_t = torch.tensor(f_np, dtype=torch.float64)
    f_j = jnp.asarray(f_np)

    m1_t = torch.tensor(m1, dtype=torch.float64)
    m2_t = torch.tensor(m2, dtype=torch.float64)
    e0_t = torch.tensor(e0, dtype=torch.float64)
    f0_t = torch.tensor(f0, dtype=torch.float64)

    dpsi_t = TorchTaylorF2_Eccentric.eccentric_phase_correction(
        f=f_t, mass1=m1_t, mass2=m2_t, e0=e0_t, f0=f0_t, ecc_order=ecc_order
    )

    dpsi_j = JaxTaylorF2_Eccentric.eccentric_phase_correction(
        f=f_j, mass1=m1, mass2=m2, e0=e0, f0=f0, ecc_order=ecc_order
    )

    np.testing.assert_allclose(
        dpsi_t.detach().cpu().numpy(),
        np.asarray(dpsi_j),
        rtol=1e-14,
        atol=1e-15,
    )


@pytest.mark.parametrize("m1,m2", [(30.0, 20.0), (1.4, 1.4)])
@pytest.mark.parametrize("chi1,chi2", [(0.0, 0.0), (0.1, -0.05)])
def test_eccentric_zero_limit(m1, m2, chi1, chi2):
    """Verify that e0 = 0 collapses to quasi-circular TaylorF2."""
    f_np = np.linspace(20.0, 300.0, 200, dtype=np.float64)
    f_j = jnp.asarray(f_np)
    f_t = torch.tensor(f_np, dtype=torch.float64)
    f_ref = 20.0
    dist = 100.0
    inc = 0.5

    # JAX TaylorF2 vs JAX TaylorF2_Eccentric (e0=0)
    hp_circ_j, hc_circ_j = JaxTaylorF2.evaluate(
        f=f_j,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=dist,
        inclination=inc,
        f_ref=f_ref,
        f_isco_cutoff=False,
    )

    hp_ecc_j, hc_ecc_j = JaxTaylorF2_Eccentric.evaluate(
        f=f_j,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        e0=0.0,
        distance=dist,
        inclination=inc,
        f_ref=f_ref,
        f0_ecc=f_ref,
        f_isco_cutoff=False,
    )

    max_amp_p = float(jnp.max(jnp.abs(hp_circ_j)))
    max_amp_c = float(jnp.max(jnp.abs(hc_circ_j)))

    diff_p = float(jnp.max(jnp.abs(hp_ecc_j - hp_circ_j)))
    diff_c = float(jnp.max(jnp.abs(hc_ecc_j - hc_circ_j)))

    assert diff_p / max_amp_p < 1e-12, f"h_plus rel diff {diff_p / max_amp_p} exceeds 1e-12"
    assert diff_c / max_amp_c < 1e-12, f"h_cross rel diff {diff_c / max_amp_c} exceeds 1e-12"

    # Also compare with PyTorch TaylorF2 at e0=0
    hp_t, hc_t = TorchTaylorF2.evaluate(
        f=f_t,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=dist,
        inclination=inc,
        f_ref=f_ref,
        f_isco_cutoff=False,
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_ecc_j), rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_ecc_j), rtol=1e-11, atol=1e-15)


@pytest.mark.parametrize("m1,m2", [(36.0, 29.0), (30.0, 20.0), (1.4, 1.3), (1.4, 1.4)])
@pytest.mark.parametrize("e0", [0.01, 0.1, 0.2])
@pytest.mark.parametrize("chi1,chi2", [(0.0, 0.0), (0.1, -0.05)])
def test_eccentric_waveform_parity(m1, m2, e0, chi1, chi2):
    """Verify full waveform parity comparing torchwave vs jaxwave across masses and e0."""
    f_np = np.linspace(20.0, 300.0, 250, dtype=np.float64)
    f_t = torch.tensor(f_np, dtype=torch.float64)
    f_j = jnp.asarray(f_np)
    dist = 100.0
    inc = 0.5
    f_ref = 20.0

    # Torch evaluate
    hp_t, hc_t = TorchTaylorF2_Eccentric()(
        f=f_t,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        e0=e0,
        distance=dist,
        inclination=inc,
        f_ref=f_ref,
        f0_ecc=f_ref,
        f_isco_cutoff=False,
    )

    # JAX evaluate
    hp_j, hc_j = JaxTaylorF2_Eccentric.evaluate(
        f=f_j,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        e0=e0,
        distance=dist,
        inclination=inc,
        f_ref=f_ref,
        f0_ecc=f_ref,
        f_isco_cutoff=False,
    )

    # Accumulated phase over hundreds of radians causes sub-ppb numerical drift
    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-11, atol=1e-15)


@pytest.mark.parametrize("e0", [0.0, 0.05, 0.15])
def test_eccentric_amplitude_correction_parity(e0):
    """Verify amplitude correction factor and waveform with include_amplitude_correction=True."""
    f_np = np.linspace(20.0, 300.0, 100, dtype=np.float64)
    f_t = torch.tensor(f_np, dtype=torch.float64)
    f_j = jnp.asarray(f_np)
    f0 = 20.0

    factor_t = TorchTaylorF2_Eccentric.eccentric_amplitude_factor(
        f=f_t, e0=torch.tensor(e0, dtype=torch.float64), f0=torch.tensor(f0, dtype=torch.float64)
    )
    factor_j = JaxTaylorF2_Eccentric.eccentric_amplitude_factor(
        f=f_j, e0=e0, f0=f0
    )

    np.testing.assert_allclose(factor_t.numpy(), np.asarray(factor_j), rtol=1e-14, atol=1e-15)

    # Full waveform parity with include_amplitude_correction=True
    hp_t, hc_t = TorchTaylorF2_Eccentric()(
        f=f_t,
        mass1=30.0,
        mass2=20.0,
        e0=e0,
        f_ref=f0,
        f0_ecc=f0,
        f_isco_cutoff=False,
        include_amplitude_correction=True,
    )
    hp_j, hc_j = JaxTaylorF2_Eccentric.evaluate(
        f=f_j,
        mass1=30.0,
        mass2=20.0,
        e0=e0,
        f_ref=f0,
        f0_ecc=f0,
        f_isco_cutoff=False,
        include_amplitude_correction=True,
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-11, atol=1e-15)


def test_eccentric_jax_jit():
    """Verify that jaxwave TaylorF2_Eccentric compiles cleanly with jax.jit."""
    f_j = jnp.linspace(20.0, 300.0, 200, dtype=jnp.float64)

    @jax.jit
    def generate(m1, m2, s1, s2, e):
        return JaxTaylorF2_Eccentric.evaluate(
            f=f_j,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            e0=e,
            f_isco_cutoff=False,
        )

    hp, hc = generate(30.0, 20.0, 0.1, -0.1, 0.05)
    assert hp.shape == f_j.shape
    assert hc.shape == f_j.shape
    assert jnp.all(jnp.isfinite(hp))
    assert jnp.all(jnp.isfinite(hc))


def test_eccentric_autodiff():
    """Verify jax.grad autodiff w.r.t mass1, mass2, e0 produces finite non-zero gradients matching PyTorch."""
    f_np = np.linspace(20.0, 150.0, 50, dtype=np.float64)
    f_t = torch.tensor(f_np, dtype=torch.float64)
    f_j = jnp.asarray(f_np)

    m1, m2, e0 = 1.5, 1.2, 0.05

    # PyTorch autodiff
    m1_t = torch.tensor(m1, dtype=torch.float64, requires_grad=True)
    m2_t = torch.tensor(m2, dtype=torch.float64, requires_grad=True)
    e0_t = torch.tensor(e0, dtype=torch.float64, requires_grad=True)

    hp_t, hc_t = TorchTaylorF2_Eccentric()(
        f=f_t, mass1=m1_t, mass2=m2_t, e0=e0_t, f_ref=20.0, f_isco_cutoff=False
    )
    loss_t = (hp_t.real).sum() * 1e22
    loss_t.backward()

    # JAX autodiff
    def loss_j(m1_arg, m2_arg, e0_arg):
        hp, hc = JaxTaylorF2_Eccentric.evaluate(
            f_j, m1_arg, m2_arg, e0=e0_arg, f_ref=20.0, f_isco_cutoff=False
        )
        return (jnp.real(hp)).sum() * 1e22

    grad_fn = jax.grad(loss_j, argnums=(0, 1, 2))
    g_m1, g_m2, g_e0 = grad_fn(m1, m2, e0)

    # Check finite and non-zero
    for name, g in [("m1", g_m1), ("m2", g_m2), ("e0", g_e0)]:
        assert jnp.isfinite(g), f"Gradient for {name} is not finite"
        assert abs(float(g)) > 0.0, f"Gradient for {name} is zero"

    # Cross-framework gradient parity
    np.testing.assert_allclose(
        [m1_t.grad.item(), m2_t.grad.item(), e0_t.grad.item()],
        [float(g_m1), float(g_m2), float(g_e0)],
        rtol=1e-7,
    )


def test_eccentric_vmap():
    """Verify jax.vmap compatibility over batched mass1, mass2, e0."""
    f_j = jnp.linspace(20.0, 300.0, 100, dtype=jnp.float64)
    m1_b = jnp.array([1.4, 1.6, 2.0], dtype=jnp.float64)
    m2_b = jnp.array([1.3, 1.4, 1.5], dtype=jnp.float64)
    e0_b = jnp.array([0.01, 0.05, 0.1], dtype=jnp.float64)

    vmapped_eval = jax.vmap(
        lambda m1, m2, e: JaxTaylorF2_Eccentric.evaluate(
            f=f_j, mass1=m1, mass2=m2, e0=e, f_ref=20.0, f_isco_cutoff=False
        )
    )
    hp_b, hc_b = vmapped_eval(m1_b, m2_b, e0_b)

    assert hp_b.shape == (3, 100)
    assert hc_b.shape == (3, 100)

    for i in range(3):
        hp_s, hc_s = JaxTaylorF2_Eccentric.evaluate(
            f=f_j, mass1=m1_b[i], mass2=m2_b[i], e0=e0_b[i], f_ref=20.0, f_isco_cutoff=False
        )
        np.testing.assert_allclose(np.asarray(hp_b[i]), np.asarray(hp_s), rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(np.asarray(hc_b[i]), np.asarray(hc_s), rtol=1e-15, atol=1e-15)

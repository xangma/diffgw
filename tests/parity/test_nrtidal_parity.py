"""Cross-framework parity test: torchwave vs jaxwave for NRTidal waveform models."""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants.imrphenomd_nrtidal import (
    IMRPhenomD_NRTidal as TorchNRTidal,
    IMRPhenomD_NRTidalv2 as TorchNRTidalv2,
    _bns_merger_frequency as torch_bns_merger_freq,
    _planck_taper_window as torch_planck_taper,
    _quadrupole_monopole_constants as torch_qm_constants,
    _nrtidal_self_spin_components as torch_self_spin,
)
from jaxwave.approximants.imrphenomd_nrtidal import (
    IMRPhenomD_NRTidal as JaxNRTidal,
    IMRPhenomD_NRTidalv2 as JaxNRTidalv2,
    _bns_merger_frequency as jax_bns_merger_freq,
    _planck_taper_window as jax_planck_taper,
    _quadrupole_monopole_constants as jax_qm_constants,
    _nrtidal_self_spin_components as jax_self_spin,
)


@pytest.mark.parametrize("m1,m2", [(1.4, 1.4), (1.5, 1.25), (1.6, 1.1)])
@pytest.mark.parametrize("l1,l2", [(400.0, 400.0), (300.0, 550.0), (100.0, 900.0)])
def test_bns_merger_frequency_parity(m1, m2, l1, l2):
    m1_t = torch.tensor(m1, dtype=torch.float64)
    m2_t = torch.tensor(m2, dtype=torch.float64)
    l1_t = torch.tensor(l1, dtype=torch.float64)
    l2_t = torch.tensor(l2, dtype=torch.float64)
    f_t = torch_bns_merger_freq(m1_t, m2_t, l1_t, l2_t)

    m1_j = jnp.array(m1, dtype=jnp.float64)
    m2_j = jnp.array(m2, dtype=jnp.float64)
    l1_j = jnp.array(l1, dtype=jnp.float64)
    l2_j = jnp.array(l2, dtype=jnp.float64)
    f_j = jax_bns_merger_freq(m1_j, m2_j, l1_j, l2_j)

    np.testing.assert_allclose(f_t.numpy(), np.asarray(f_j), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("lam", [0.0, 10.0, 200.0, 500.0, 1200.0])
def test_quadrupole_monopole_constants_parity(lam):
    lam_t = torch.tensor(lam, dtype=torch.float64)
    cq_t, coc_t = torch_qm_constants(lam_t)

    lam_j = jnp.array(lam, dtype=jnp.float64)
    cq_j, coc_j = jax_qm_constants(lam_j)

    np.testing.assert_allclose(cq_t.numpy(), np.asarray(cq_j), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(coc_t.numpy(), np.asarray(coc_j), rtol=1e-14, atol=1e-15)


def test_planck_taper_parity():
    f_np = np.linspace(500.0, 2500.0, 300, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    w_t = torch_planck_taper(f_t, torch.tensor(1200.0, dtype=torch.float64), torch.tensor(1500.0, dtype=torch.float64))
    w_j = jax_planck_taper(f_j, jnp.array(1200.0, dtype=jnp.float64), jnp.array(1500.0, dtype=jnp.float64))

    np.testing.assert_allclose(w_t.numpy(), np.asarray(w_j), rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize(
    "m1,m2,l1,l2",
    [
        (1.4, 1.4, 400.0, 400.0),
        (1.5, 1.25, 300.0, 550.0),
        (1.35, 1.35, 800.0, 800.0),
        (1.6, 1.1, 150.0, 700.0),
        (1.4, 1.3, 0.0, 0.0),
    ],
)
@pytest.mark.parametrize("chi1,chi2", [(0.0, 0.0), (0.03, -0.02)])
def test_nrtidal_v1_waveform_parity(m1, m2, l1, l2, chi1, chi2):
    f_np = np.linspace(20.0, 1500.0, 400, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    hp_t, hc_t = TorchNRTidal.evaluate(
        f=f_t,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=50.0,
        inclination=0.3,
        phic=0.2,
        tc=0.01,
        lambda1=l1,
        lambda2=l2,
        f_ref=20.0,
    )

    hp_j, hc_j = JaxNRTidal.evaluate(
        f=f_j,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=50.0,
        inclination=0.3,
        phic=0.2,
        tc=0.01,
        lambda1=l1,
        lambda2=l2,
        f_ref=20.0,
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "m1,m2,l1,l2",
    [
        (1.4, 1.4, 400.0, 400.0),
        (1.5, 1.25, 300.0, 550.0),
        (1.35, 1.35, 800.0, 800.0),
        (1.6, 1.1, 150.0, 700.0),
        (1.4, 1.3, 0.0, 0.0),
    ],
)
@pytest.mark.parametrize("chi1,chi2", [(0.0, 0.0), (0.04, -0.01)])
def test_nrtidal_v2_waveform_parity(m1, m2, l1, l2, chi1, chi2):
    f_np = np.linspace(20.0, 1500.0, 400, dtype=np.float64)
    f_t = torch.from_numpy(f_np)
    f_j = jnp.asarray(f_np)

    hp_t, hc_t = TorchNRTidalv2.evaluate(
        f=f_t,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=50.0,
        inclination=0.3,
        phic=0.2,
        tc=0.01,
        lambda1=l1,
        lambda2=l2,
        f_ref=20.0,
    )

    hp_j, hc_j = JaxNRTidalv2.evaluate(
        f=f_j,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=50.0,
        inclination=0.3,
        phic=0.2,
        tc=0.01,
        lambda1=l1,
        lambda2=l2,
        f_ref=20.0,
    )

    np.testing.assert_allclose(hp_t.numpy(), np.asarray(hp_j), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hc_t.numpy(), np.asarray(hc_j), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("model_cls", [JaxNRTidal, JaxNRTidalv2])
def test_nrtidal_jax_jit(model_cls):
    """Verify that jaxwave NRTidal compiles with jax.jit."""
    f_j = jnp.linspace(20.0, 1024.0, 200, dtype=jnp.float64)

    @jax.jit
    def generate(m1, m2, s1, s2, l1, l2):
        hp, hc = model_cls.evaluate(
            f=f_j,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            lambda1=l1,
            lambda2=l2,
            f_ref=20.0,
        )
        return jnp.real(hp).sum()

    val = generate(1.4, 1.35, 0.02, -0.01, 350.0, 420.0)
    assert jnp.isfinite(val)


@pytest.mark.parametrize("model_cls", [JaxNRTidal, JaxNRTidalv2])
def test_nrtidal_jax_autodiff(model_cls):
    """Verify that jaxwave NRTidal supports autodiff w.r.t masses, spins, and tidal deformabilities."""
    f_j = jnp.linspace(20.0, 1024.0, 200, dtype=jnp.float64)

    def loss_fn(m1, m2, s1, s2, l1, l2):
        hp, hc = model_cls.evaluate(
            f=f_j,
            mass1=m1,
            mass2=m2,
            chi1=s1,
            chi2=s2,
            lambda1=l1,
            lambda2=l2,
            f_ref=20.0,
        )
        return jnp.real(hp).sum() + jnp.imag(hc).sum()

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5)))
    grads = grad_fn(1.4, 1.35, 0.02, -0.01, 350.0, 420.0)

    for i, g in enumerate(grads):
        assert jnp.isfinite(g), f"Gradient w.r.t arg {i} is not finite: {g}"

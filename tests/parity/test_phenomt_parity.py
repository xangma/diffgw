"""Torch/JAX parity and transformability checks for native IMRPhenomT."""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants.imrphenomt import IMRPhenomT as TorchIMRPhenomT
from jaxwave.approximants.imrphenomt import IMRPhenomT as JaxIMRPhenomT


jax.config.update("jax_enable_x64", True)


def _params():
    return dict(mass1=32.0, mass2=24.0, chi1=0.35, chi2=-0.15, distance=180.0, inclination=0.7, phic=0.2)


def _complex_parity_metrics(torch_wave, jax_wave):
    """Return global complex relative L2 error and unweighted mismatch."""
    torch_complex = np.asarray(torch_wave[0].detach()) - 1j * np.asarray(torch_wave[1].detach())
    jax_complex = np.asarray(jax_wave[0]) - 1j * np.asarray(jax_wave[1])
    relative_l2 = np.linalg.norm(torch_complex - jax_complex) / np.linalg.norm(jax_complex)
    inner = np.real(np.vdot(torch_complex, jax_complex))
    norm_product = np.sqrt(np.vdot(torch_complex, torch_complex).real * np.vdot(jax_complex, jax_complex).real)
    mismatch = max(0.0, 1.0 - inner / norm_product)
    return relative_l2, mismatch


def test_phenomt_torch_jax_parity():
    times = -0.5 + torch.arange(2253, dtype=torch.float64) / 4096.0
    jtimes = jnp.asarray(times.numpy())
    kwargs = _params()
    hp_t, hc_t = TorchIMRPhenomT().time_domain(times, **kwargs)
    hp_j, hc_j = JaxIMRPhenomT().time_domain(jtimes, **kwargs)

    np.testing.assert_allclose(np.asarray(hp_j), hp_t.detach().numpy(), rtol=2e-11, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_j), hc_t.detach().numpy(), rtol=2e-11, atol=1e-30)


def test_phenomt_batched_parity():
    times_t = -0.5 + torch.arange(2253, dtype=torch.float64) / 4096.0
    times_j = jnp.asarray(times_t.numpy())
    params_t = dict(
        mass1=torch.tensor([32.0, 40.0], dtype=torch.float64),
        mass2=torch.tensor([24.0, 18.0], dtype=torch.float64),
        chi1=torch.tensor([0.35, -0.2], dtype=torch.float64),
        chi2=torch.tensor([-0.15, 0.1], dtype=torch.float64),
        distance=torch.tensor([180.0, 220.0], dtype=torch.float64),
        inclination=torch.tensor([0.7, 1.1], dtype=torch.float64),
        phic=torch.tensor([0.2, -0.3], dtype=torch.float64),
    )
    params_j = {key: jnp.asarray(value.detach().numpy()) for key, value in params_t.items()}
    hp_t, hc_t = TorchIMRPhenomT().time_domain(times_t, **params_t)
    hp_j, hc_j = JaxIMRPhenomT().time_domain(times_j, **params_j)
    assert hp_t.shape == hp_j.shape == (2, times_t.numel())
    np.testing.assert_allclose(np.asarray(hp_j), hp_t.detach().numpy(), rtol=2e-11, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_j), hc_t.detach().numpy(), rtol=2e-11, atol=1e-30)


def test_phenomt_jax_jit_vmap_and_gradient():
    times = jnp.arange(2253, dtype=jnp.float64) / 4096.0 - 0.5
    model = JaxIMRPhenomT()

    def one(mass1, mass2):
        hp, hc = model.time_domain(times, mass1, mass2, chi1=0.35, chi2=-0.15, distance=180.0)
        return hp, hc

    hp_jit, hc_jit = jax.jit(one)(32.0, 24.0)
    hp_ref, hc_ref = one(32.0, 24.0)
    np.testing.assert_allclose(np.asarray(hp_jit), np.asarray(hp_ref), rtol=1e-12, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_jit), np.asarray(hc_ref), rtol=1e-12, atol=1e-30)

    hp_v, hc_v = jax.vmap(one)(jnp.array([32.0, 40.0]), jnp.array([24.0, 18.0]))
    assert hp_v.shape == hc_v.shape == (2, times.size)
    hp_b, hc_b = model.time_domain(
        times,
        jnp.array([32.0, 40.0]),
        jnp.array([24.0, 18.0]),
        chi1=0.35,
        chi2=-0.15,
        distance=180.0,
    )
    np.testing.assert_allclose(np.asarray(hp_v), np.asarray(hp_b), rtol=2e-12, atol=1e-30)
    np.testing.assert_allclose(np.asarray(hc_v), np.asarray(hc_b), rtol=2e-12, atol=1e-30)

    def loss(mass1):
        hp, hc = one(mass1, 24.0)
        return jnp.sum(hp * hp + hc * hc)

    grad = jax.grad(loss)(32.0)
    assert np.isfinite(grad) and grad != 0.0


def test_jax_phenomt_join_continuity_and_phase_frequency_relation():
    """JAX's private amplitude joins and phase primitive are C1/consistent."""
    model = JaxIMRPhenomT()
    dtype = jnp.float64
    state = model._coeff(
        jnp.asarray(30.0, dtype=dtype),
        jnp.asarray(20.0, dtype=dtype),
        jnp.asarray(0.4, dtype=dtype),
        jnp.asarray(-0.2, dtype=dtype),
    )
    from jaxwave.approximants._phenomt_amplitude import _merger_basis, _pn

    pn, pseudo, merger, alpha, c1, c2, c3, c4, rd_alpha = state[8]

    def omega_cb(u):
        return model._omega_insp(u, state[1], state[2], state[5], state[6], state[7])

    def inspiral(u):
        x = (omega_cb(u) / 2) ** (2 / 3)
        return jnp.abs(_pn(x, pn, pseudo))

    def middle(u):
        return jnp.sum(_merger_basis(u, alpha) * merger, axis=-1)

    def ringdown(u):
        return jnp.exp(-rd_alpha * u) * (c1 * jnp.tanh(c2 * u + c3) + c4)

    np.testing.assert_allclose(
        np.asarray(inspiral(jnp.asarray([-150.0], dtype=dtype))),
        np.asarray(middle(jnp.asarray([-150.0], dtype=dtype))),
        rtol=2e-8,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        np.asarray(middle(jnp.asarray([0.0], dtype=dtype))),
        np.asarray(ringdown(jnp.asarray([0.0], dtype=dtype))),
        rtol=2e-8,
        atol=2e-10,
    )
    for boundary, left_branch, right_branch in ((-150.0, inspiral, middle), (0.0, middle, ringdown)):
        dleft = jax.grad(lambda u: left_branch(u).sum())(jnp.asarray([boundary], dtype=dtype))
        dright = jax.grad(lambda u: right_branch(u).sum())(jnp.asarray([boundary], dtype=dtype))
        np.testing.assert_allclose(np.asarray(dleft), np.asarray(dright), rtol=2e-6, atol=2e-9)

    h = 1.0e-4
    for u0 in (-1000.0, -200.0, -1.0, 0.1, 10.0):
        u = jnp.asarray([u0 - h, u0 + h], dtype=dtype)
        frequency, phase, _ = model._evaluate(u, state)
        dphase = (phase[1] - phase[0]) / (2 * h)
        np.testing.assert_allclose(np.asarray(dphase), np.asarray(frequency.mean()), rtol=2e-6, atol=2e-8)


def test_jax_time_frequency_roundtrip_and_root_gradient():
    model = JaxIMRPhenomT()
    for mass1, mass2, frequency, positive_ringdown in (
        (30.0, 20.0, 80.0, False),
        (10.0, 1.0, 40.0, False),
        (1.4, 1.4, 10.0, False),
        (30.0, 25.0, 300.0, True),
    ):
        t = model.time_at_frequency(jnp.asarray(frequency, dtype=jnp.float64), mass1, mass2)
        recovered = model.frequency(t, mass1, mass2)
        np.testing.assert_allclose(np.asarray(recovered), frequency, rtol=2e-7, atol=2e-8)
        assert np.isfinite(t)
        assert (float(t) > 0) is positive_ringdown
    gradient = jax.grad(lambda mass: model.time_at_frequency(10.0, mass, mass))(jnp.asarray(1.4, dtype=jnp.float64))
    assert np.isfinite(gradient) and gradient != 0


@pytest.mark.parametrize(
    "mass1,mass2,chi1,chi2",
    ((30.0, 30.0, 0.8, -0.8), (60.0, 6.0, 0.9, 0.1), (30.0, 30.0, -0.9, -0.9)),
)
def test_phenomt_edge_global_complex_parity(mass1, mass2, chi1, chi2):
    """Extreme spins and q=10 retain global FP64 parity through ringdown."""
    times = torch.arange(-0.5, 0.050001, 1.0 / 4096.0, dtype=torch.float64)
    kwargs = dict(mass1=mass1, mass2=mass2, chi1=chi1, chi2=chi2, distance=180.0, inclination=0.7, phic=0.2)
    torch_wave = TorchIMRPhenomT().time_domain(times, **kwargs)
    jax_wave = JaxIMRPhenomT().time_domain(jnp.asarray(times.numpy()), **kwargs)
    relative_l2, mismatch = _complex_parity_metrics(torch_wave, jax_wave)
    assert relative_l2 <= 1.0e-11
    assert mismatch <= 1.0e-14


def test_phenomt_q10_jit_global_parity():
    """The extreme mass-ratio waveform remains parity-safe under JIT."""
    times = torch.arange(-0.5, 0.050001, 1.0 / 4096.0, dtype=torch.float64)
    jtimes = jnp.asarray(times.numpy())
    model = JaxIMRPhenomT()

    def evaluate(mass1, mass2, chi1, chi2):
        return model.time_domain(jtimes, mass1, mass2, chi1=chi1, chi2=chi2, distance=180.0, inclination=0.7, phic=0.2)

    jax_wave = jax.jit(evaluate)(60.0, 6.0, 0.9, 0.1)
    torch_wave = TorchIMRPhenomT().time_domain(
        times, 60.0, 6.0, chi1=0.9, chi2=0.1, distance=180.0, inclination=0.7, phic=0.2
    )
    relative_l2, mismatch = _complex_parity_metrics(torch_wave, jax_wave)
    assert relative_l2 <= 1.0e-11
    assert mismatch <= 1.0e-14


@pytest.mark.parametrize("mass1,mass2,chi1,chi2", ((30.0, 30.0, 0.5, -0.3), (60.0, 6.0, 0.9, 0.1)))
def test_phenomt_phase_sensitive_intrinsic_gradient_parity(mass1, mass2, chi1, chi2):
    """JAX and Torch agree on all intrinsic gradients when phase is observed."""
    times = np.arange(-0.08, 0.045001, 1.0 / 2048.0, dtype=np.float64)
    weights = np.linspace(-1.0, 1.0, times.size)
    torch_times = torch.as_tensor(times)
    torch_weights = torch.as_tensor(weights)
    torch_values = torch.tensor([mass1, mass2, chi1, chi2], dtype=torch.float64, requires_grad=True)

    def torch_objective(values):
        hp, hc = TorchIMRPhenomT().time_domain(
            torch_times,
            values[0],
            values[1],
            chi1=values[2],
            chi2=values[3],
            distance=400.0,
            phic=0.23,
        )
        return (hp * (1.0 + torch_weights) + hc * (1.0 - 0.3 * torch_weights)).sum() * 1.0e21

    torch_objective(torch_values).backward()

    jax_times = jnp.asarray(times)
    jax_weights = jnp.asarray(weights)
    jax_model = JaxIMRPhenomT()

    def jax_objective(values):
        hp, hc = jax_model.time_domain(
            jax_times,
            values[0],
            values[1],
            chi1=values[2],
            chi2=values[3],
            distance=400.0,
            phic=0.23,
        )
        return jnp.sum(hp * (1.0 + jax_weights) + hc * (1.0 - 0.3 * jax_weights)) * 1.0e21

    jax_gradient = np.asarray(jax.grad(jax_objective)(jnp.asarray([mass1, mass2, chi1, chi2])))
    torch_gradient = torch_values.grad.detach().numpy()
    np.testing.assert_allclose(jax_gradient, torch_gradient, rtol=2.0e-9, atol=1.0e-8)


def test_jax_time_frequency_invalid_range_is_eager_error_or_jit_nan():
    model = JaxIMRPhenomT()
    with pytest.raises(ValueError, match="frequency must be finite and positive"):
        model.time_at_frequency(0.0, 30.0, 20.0)
    with pytest.raises(ValueError, match="above the supported"):
        model.time_at_frequency(1.0e8, 30.0, 20.0)
    bad = jax.jit(lambda f: model.time_at_frequency(f, 30.0, 20.0))(jnp.asarray(1.0e8))
    assert np.isnan(np.asarray(bad))

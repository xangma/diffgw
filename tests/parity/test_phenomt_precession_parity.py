"""Torch/JAX parity for the native NNLO dominant precessing model."""

import numpy as np
import pytest
import torch
import jax
import jax.numpy as jnp

from torchwave.approximants.imrphenomtp import IMRPhenomTP as TorchTP
from torchwave.approximants.imrphenomtphm import IMRPhenomTPHM as TorchTPHM
from jaxwave.approximants.imrphenomtp import IMRPhenomTP as JaxTP
from jaxwave.approximants.imrphenomtphm import IMRPhenomTPHM as JaxTPHM


CASES = (
    (30.0, 30.0, 0.4, -0.3, 0.2, -0.1, -0.12, 0.08),
    (40.0, 15.0, 0.3, -0.2, 0.12, -0.04, 0.03, 0.07),
    (60.0, 6.0, 0.9, 0.1, 0.2, 0.1, 0.1, 0.0),
    (30.0, 20.0, -0.9, -0.9, -0.15, 0.12, 0.08, -0.04),
)


def _run_torch(t, p):
    m1, m2, c1, c2, x1, y1, x2, y2 = p
    return TorchTP().time_domain(
        t,
        m1,
        m2,
        c1,
        c2,
        100.0,
        0.9,
        0.27,
        0.0,
        20.0,
        x1,
        y1,
        x2,
        y2,
    )


def _run_jax(t, p):
    m1, m2, c1, c2, x1, y1, x2, y2 = p
    return JaxTP().time_domain(
        t,
        m1,
        m2,
        c1,
        c2,
        100.0,
        0.9,
        0.27,
        0.0,
        20.0,
        x1,
        y1,
        x2,
        y2,
    )


@pytest.mark.parametrize("params", CASES)
def test_tp_global_complex_parity(params):
    t = np.linspace(-0.5, 0.05, 81)
    th = torch.tensor(t, dtype=torch.float64)
    hp_t, hc_t = _run_torch(th, params)
    hp_j, hc_j = _run_jax(jnp.asarray(t), params)
    actual = hp_t.detach().numpy() - 1j * hc_t.detach().numpy()
    reference = np.asarray(hp_j) - 1j * np.asarray(hc_j)
    rel = np.linalg.norm(actual - reference) / np.linalg.norm(actual)
    match = abs(np.vdot(actual, reference)) / (np.linalg.norm(actual) * np.linalg.norm(reference))
    assert rel <= 1e-11
    assert 1.0 - match <= 1e-14


def test_tp_jit_matches_eager_for_high_mass_ratio_case():
    t = jnp.linspace(-0.5, 0.05, 81)
    params = CASES[1]
    eager = _run_jax(t, params)
    compiled = jax.jit(lambda x: _run_jax(x, params))(t)
    for lhs, rhs in zip(eager, compiled):
        np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-11, atol=1e-30)


def test_tp_intrinsic_gradient_parity():
    t = jnp.linspace(-0.3, 0.02, 20)
    weight = jnp.linspace(0.3, 1.2, 20)
    initial = np.array([30.0, 20.0, 0.3, -0.2, 0.1, 0.05, -0.1, 0.08])

    def objective_jax(x):
        hp, hc = _run_jax(t, tuple(x))
        return jnp.sum((hp * (1.0 + weight) + hc * (1.0 - 0.2 * weight)) * 1e21)

    x = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    hp, hc = _run_torch(torch.tensor(np.asarray(t), dtype=torch.float64), tuple(x))
    wt = torch.tensor(np.asarray(weight), dtype=torch.float64)
    objective_torch = (hp * (1.0 + wt) + hc * (1.0 - 0.2 * wt)).sum() * 1e21
    grad_torch = torch.autograd.grad(objective_torch, x)[0].detach().numpy()
    grad_jax = np.asarray(jax.grad(objective_jax)(jnp.asarray(initial)))
    np.testing.assert_allclose(grad_torch, grad_jax, rtol=2e-9, atol=1e-10)


def _run_family(model_cls, t, p):
    m1, m2, c1, c2, x1, y1, x2, y2 = p
    return model_cls().time_domain(t, m1, m2, c1, c2, 100.0, 0.9, 0.27, 0.0, 20.0, x1, y1, x2, y2)


@pytest.mark.parametrize("params", CASES[0:3])
def test_thm_global_complex_parity_and_jit(params):
    t = np.linspace(-0.35, 0.02, 32)
    hp_t, hc_t = _run_family(TorchTPHM, torch.tensor(t, dtype=torch.float64), params)
    hp_j, hc_j = _run_family(JaxTPHM, jnp.asarray(t), params)
    actual = hp_t.detach().numpy() - 1j * hc_t.detach().numpy()
    reference = np.asarray(hp_j) - 1j * np.asarray(hc_j)
    rel = np.linalg.norm(actual - reference) / np.linalg.norm(actual)
    match = abs(np.vdot(actual, reference)) / (np.linalg.norm(actual) * np.linalg.norm(reference))
    assert rel <= 1e-11
    assert 1.0 - match <= 1e-14
    eager = _run_family(JaxTPHM, jnp.asarray(t), params)
    compiled = jax.jit(lambda x: _run_family(JaxTPHM, x, params))(jnp.asarray(t))
    for lhs, rhs in zip(eager, compiled):
        np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), rtol=1e-11, atol=1e-30)


def test_thm_intrinsic_gradient_parity():
    t = jnp.linspace(-0.25, 0.015, 16)
    weight = jnp.linspace(0.3, 1.2, 16)
    initial = np.array([30.0, 20.0, 0.3, -0.2, 0.1, 0.05, -0.1, 0.08])

    def objective_jax(x):
        hp, hc = _run_family(JaxTPHM, t, tuple(x))
        return jnp.sum((hp * (1.0 + weight) + hc * (1.0 - 0.2 * weight)) * 1e21)

    x = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    hp, hc = _run_family(TorchTPHM, torch.tensor(np.asarray(t), dtype=torch.float64), tuple(x))
    wt = torch.tensor(np.asarray(weight), dtype=torch.float64)
    objective_torch = (hp * (1.0 + wt) + hc * (1.0 - 0.2 * wt)).sum() * 1e21
    grad_torch = torch.autograd.grad(objective_torch, x)[0].detach().numpy()
    grad_jax = np.asarray(jax.grad(objective_jax)(jnp.asarray(initial)))
    np.testing.assert_allclose(grad_torch, grad_jax, rtol=2e-9, atol=1e-10)


@pytest.mark.parametrize("model_cls", (JaxTP, JaxTPHM))
def test_exact_alignment_jax_intrinsic_gradients_are_finite(model_cls):
    t = jnp.linspace(-0.25, 0.015, 18)
    weight = jnp.linspace(0.3, 1.2, 18)
    initial = jnp.array([30.0, 20.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.0])

    def objective(x):
        hp, hc = _run_family(model_cls, t, tuple(x))
        return jnp.sum((hp * (1.0 + weight) + hc * (1.0 - 0.2 * weight)) * 1e21)

    gradient = np.asarray(jax.grad(objective)(initial))
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient[:4]) > 0.0

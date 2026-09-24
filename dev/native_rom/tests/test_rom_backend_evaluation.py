"""Cross-backend waveform and autodiff holdouts for the dev-only ROM algebra."""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom.rom.backend_evaluation import evaluate_jax, evaluate_torch
from dev.native_rom.rom.reduced_interpolant import ReducedTensorInterpolant

jax.config.update("jax_enable_x64", True)


def _model():
    def nodes(low, high, n):
        return (low + high)/2 - (high-low)/2 * np.cos(np.pi*np.arange(n)/(n-1))

    grid = (nodes(1., 6., 4), nodes(-.8, .8, 4), nodes(-.8, .8, 4))
    frequency = np.linspace(.01, .15, 16)
    q, a, b, f = np.meshgrid(*grid, frequency, indexing="ij")
    log_amp = .01 + f*(q + a*b) + f**2*(q**2-a)
    phase = f*(q**2+a) + f**3*(b+q*a)
    return ReducedTensorInterpolant.fit(grid, frequency, log_amp, phase,
                                        ranks=(2, 2))


def test_torch_jax_batched_and_jitted_waveforms_match_independent_numpy():
    model = _model()
    q, a, b = np.array([2.1, 4.9, 1.]), np.array([-.3, .6, -.8]), np.array([.4, -.2, .8])
    expected = model.evaluate(q, a, b)
    torch_wave = evaluate_torch(model, torch.tensor(q), torch.tensor(a), torch.tensor(b))
    jax_wave = jax.jit(lambda x, y, z: evaluate_jax(model, x, y, z))(
        jnp.asarray(q), jnp.asarray(a), jnp.asarray(b))
    np.testing.assert_allclose(torch_wave.detach().numpy(), expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(np.asarray(jax_wave), expected, rtol=2e-12, atol=2e-12)


def test_torch_jax_gradients_match_analytic_interpolant_including_exact_node():
    model = _model()
    weight = np.linspace(.5, 1.5, len(model.frequency)) * np.exp(
        1j * np.linspace(.1, .7, len(model.frequency)))
    for values in ((2.1, -.3, .4), (1., -.8, .8)):
        _, jacobian = model.evaluate(*values, derivatives=True)
        expected = np.real(np.sum(jacobian * weight, axis=-1))

        variables = tuple(torch.tensor(x, dtype=torch.float64, requires_grad=True)
                          for x in values)
        torch_weight = torch.tensor(weight, dtype=torch.complex128)
        torch_loss = torch.real(torch.sum(evaluate_torch(model, *variables) * torch_weight))
        torch_grad = torch.autograd.grad(torch_loss, variables)
        np.testing.assert_allclose([g.item() for g in torch_grad], expected,
                                   rtol=2e-10, atol=2e-10)

        jax_weight = jnp.asarray(weight)
        jax_loss = lambda x, y, z: jnp.real(jnp.sum(
            evaluate_jax(model, x, y, z) * jax_weight))
        jax_grad = jax.grad(jax_loss, argnums=(0, 1, 2))(*map(jnp.asarray, values))
        np.testing.assert_allclose(np.asarray(jax_grad), expected,
                                   rtol=2e-10, atol=2e-10)


def test_backend_out_of_domain_signals_failure():
    model = _model()
    assert torch.isnan(evaluate_torch(model, 6.1, 0., 0.)).all()
    assert bool(jnp.isnan(evaluate_jax(model, 6.1, 0., 0.)).all())

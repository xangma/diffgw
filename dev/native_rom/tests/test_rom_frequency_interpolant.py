"""Withheld-frequency and derivative checks for the in-memory ROM spline."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from dev.native_rom.rom.frequency_interpolant import (
    FrequencyReducedInterpolant, evaluate_jax_frequency, evaluate_torch_frequency)
from dev.native_rom.rom.reduced_interpolant import ReducedTensorInterpolant

jax.config.update("jax_enable_x64", True)


def _nodes(low, high, count):
    return (low + high) / 2 - (high - low) / 2 * np.cos(
        np.pi * np.arange(count) / (count - 1))


def _analytic(f, q, a, b):
    amplitude = .01 + f * (q + a*b) + f**2 * (q**2-a)
    phase = f * (q**2+a) + f**3 * (b+q*a)
    return np.exp(amplitude + 1j * phase)


def _derivatives(f, q, a, b):
    amplitude = (f + 2*f**2*q, f*b-f**2, f*a,
                 q+a*b+2*f*(q**2-a))
    phase = (2*f*q+f**3*a, f+f**3*q, f**3,
             q**2+a+3*f**2*(b+q*a))
    return tuple(_analytic(f, q, a, b) * (da + 1j*dp)
                 for da, dp in zip(amplitude, phase))


def _model():
    grid = (_nodes(1., 6., 4), _nodes(-.8, .8, 4), _nodes(-.8, .8, 4))
    frequency = np.linspace(.01, .15, 16)
    q, a, b, f = np.meshgrid(*grid, frequency, indexing="ij")
    amplitude = .01 + f*(q+a*b) + f**2*(q**2-a)
    phase = f*(q**2+a) + f**3*(b+q*a)
    return FrequencyReducedInterpolant.from_model(
        ReducedTensorInterpolant.fit(grid, frequency, amplitude, phase,
                                     ranks=(2, 2)))


def test_withheld_frequencies_batch_and_parameter_jacobian():
    model = _model()
    f = np.array([.01, .02617, .07437, .15])
    q, a, b = np.array([2.3, 6.]), np.array([-.4, .8]), np.array([.2, -.8])
    waveform, jacobian = model.evaluate(f, q, a, b, derivatives=True)
    expected = _analytic(f[None, :], q[:, None], a[:, None], b[:, None])
    np.testing.assert_allclose(waveform, expected, rtol=2e-12, atol=2e-12)
    for axis, derivative in enumerate(_derivatives(
            f[None, :], q[:, None], a[:, None], b[:, None])[:3]):
        np.testing.assert_allclose(jacobian[:, axis], derivative,
                                   rtol=3e-11, atol=3e-11)
    np.testing.assert_allclose(model.evaluate(model.model.frequency, 2.3, -.4, .2),
                               model.model.evaluate(2.3, -.4, .2),
                               rtol=2e-12, atol=2e-12)


def test_torch_jax_off_grid_frequency_and_parameter_gradients():
    model = _model()
    for f, q, a, b in ((.07437, 2.3, -.4, .2),
                       (model.model.frequency[7], 1., -.8, .8)):
        expected = _analytic(f, q, a, b)
        parameter_and_frequency = _derivatives(f, q, a, b)
        analytic_gradient = (parameter_and_frequency[3],
                             *parameter_and_frequency[:3])
        variables = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                          for v in (f, q, a, b))
        torch_wave = evaluate_torch_frequency(model, *variables)
        torch_gradient = torch.autograd.grad(torch.real(torch_wave), variables)
        np.testing.assert_allclose(torch_wave.detach().numpy(), expected,
                                   rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose([x.item() for x in torch_gradient],
                                   np.real(analytic_gradient), rtol=3e-10,
                                   atol=3e-10)

        jax_wave = jax.jit(lambda *x: evaluate_jax_frequency(model, *x))(
            *map(jnp.asarray, (f, q, a, b)))
        jax_gradient = jax.jit(jax.grad(
            lambda *x: jnp.real(evaluate_jax_frequency(model, *x)),
            argnums=(0, 1, 2, 3)))(*map(jnp.asarray, (f, q, a, b)))
        np.testing.assert_allclose(np.asarray(jax_wave), expected,
                                   rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(np.asarray(jax_gradient),
                                   np.real(analytic_gradient), rtol=3e-10,
                                   atol=3e-10)


def test_backend_batches_and_domain_failures():
    model = _model()
    f = np.array([.02617, .07437, .15])
    q, a, b = np.array([2.3, 6.]), np.array([-.4, .8]), np.array([.2, -.8])
    expected = model.evaluate(f, q, a, b)
    torch_wave = evaluate_torch_frequency(model, torch.tensor(f), torch.tensor(q),
                                          torch.tensor(a), torch.tensor(b))
    jax_wave = jax.jit(lambda x, y, z, v: evaluate_jax_frequency(
        model, x, y, z, v))(jnp.asarray(f), jnp.asarray(q),
                           jnp.asarray(a), jnp.asarray(b))
    np.testing.assert_allclose(torch_wave.detach().numpy(), expected,
                               rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(np.asarray(jax_wave), expected,
                               rtol=2e-12, atol=2e-12)

    with pytest.raises(ValueError, match="frequency outside"):
        model.evaluate(.151, 2.3, -.4, .2)
    bad_f = np.array([.07437, .151])
    assert np.isnan(evaluate_torch_frequency(model, bad_f, 2.3, -.4, .2)
                    .detach().numpy()[1])
    assert np.isnan(np.asarray(evaluate_jax_frequency(model, bad_f, 2.3, -.4, .2))[1])
    bad_q = np.array([2.3, 6.01])
    torch_bad = evaluate_torch_frequency(model, f, bad_q, a, b).detach().numpy()
    jax_bad = np.asarray(evaluate_jax_frequency(model, f, bad_q, a, b))
    assert np.isfinite(torch_bad[0]).all() and np.isnan(torch_bad[1]).all()
    assert np.isfinite(jax_bad[0]).all() and np.isnan(jax_bad[1]).all()

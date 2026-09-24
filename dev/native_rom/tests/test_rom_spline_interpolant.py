"""Independent polynomial holdouts for the candidate tensor-spline ROM."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from scipy.interpolate import CubicSpline

from dev.native_rom.rom.frequency_interpolant import (
    FrequencyReducedInterpolant,
    evaluate_jax_frequency,
    evaluate_torch_frequency,
)
from dev.native_rom.rom.spline_interpolant import (
    SplineTensorInterpolant,
    evaluate_jax_spline,
    evaluate_torch_spline,
)

jax.config.update("jax_enable_x64", True)


def _fields(q, a, b, f):
    amp = .01 + f * (q + a*b) + f**2 * (q**2 - a)
    phi = f * (q**2 + a) + f**3 * (b + q*a)
    return amp, phi


def _model():
    # Uneven, non-Chebyshev nodes expose accidental reliance on the older
    # global polynomial interpolant.
    nodes = (np.array([1., 1.4, 2.3, 4., 7.]),
             np.array([-.8, -.5, 0., .3, .8]),
             np.array([-.8, -.2, .1, .5, .8]))
    f = np.linspace(.01, .3, 20)
    q, a, b, frequency = np.meshgrid(*nodes, f, indexing="ij")
    amp, phi = _fields(q, a, b, frequency)
    return SplineTensorInterpolant.fit(nodes, f, amp, phi, ranks=(3, 2))


def test_off_grid_and_boundary_waveforms_and_derivatives():
    model = _model()
    q = np.array([2.4, 5.1, 1., 7.])
    a = np.array([.2, -.36, -.8, .8])
    b = np.array([-.1, .31, .8, -.8])
    wave, jacobian = model.evaluate(q, a, b, derivatives=True)
    f = model.frequency[None, :]
    amp, phi = _fields(q[:, None], a[:, None], b[:, None], f)
    expected = np.exp(amp + 1j*phi)
    np.testing.assert_allclose(wave, expected, rtol=2e-12, atol=2e-12)
    damp = (f + 2*f**2*q[:, None], f*b[:, None] - f**2,
            f*a[:, None] + np.zeros_like(expected.real))
    dphi = (2*f*q[:, None] + f**3*a[:, None],
            f + f**3*q[:, None], f**3 + np.zeros_like(expected.real))
    for axis in range(3):
        np.testing.assert_allclose(
            jacobian[:, axis], expected*(damp[axis] + 1j*dphi[axis]),
            rtol=3e-11, atol=3e-11)
    assert model.log_amplitude.training_relative_error < 2e-13
    assert model.phase.training_relative_error < 2e-13


def test_nonpolynomial_parameter_interpolation_matches_nested_scipy_splines():
    model = _model()
    q, a, b, f = np.meshgrid(*model.nodes, model.frequency, indexing="ij")
    shape = np.sin(.7*q)*np.exp(.2*a) + b**4
    amp = shape*(1 + 2*f) + .02*f**2
    phi = shape*(.4 - f) + .03*f**3
    nonlinear = SplineTensorInterpolant.fit(model.nodes, model.frequency,
                                             amp, phi, ranks=(3, 3))
    point = (2.73, -.13, .27)

    def oracle(values):
        for grid, query in zip(model.nodes, point):
            values = CubicSpline(grid, values, axis=0)(query)
        return values

    expected = np.exp(oracle(amp) + 1j*oracle(phi))
    np.testing.assert_allclose(nonlinear.evaluate(*point), expected,
                               rtol=3e-12, atol=3e-12)


def test_torch_jax_waveforms_jit_and_gradients():
    model = _model()
    q, a, b = np.array([2.4, 5.1, 1.]), np.array([.2, -.36, -.8]), np.array([-.1, .31, .8])
    expected = model.evaluate(q, a, b)
    torch_wave = evaluate_torch_spline(model, torch.tensor(q), torch.tensor(a),
                                       torch.tensor(b))
    jax_wave = jax.jit(lambda x, y, z: evaluate_jax_spline(model, x, y, z))(
        jnp.asarray(q), jnp.asarray(a), jnp.asarray(b))
    np.testing.assert_allclose(torch_wave.detach().numpy(), expected,
                               rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(np.asarray(jax_wave), expected,
                               rtol=2e-12, atol=2e-12)

    weight = np.linspace(.5, 1.5, len(model.frequency)) * np.exp(
        1j*np.linspace(.1, .7, len(model.frequency)))
    for values in ((2.4, .2, -.1), (1., -.8, .8), (2.3, 0., .1)):
        _, jacobian = model.evaluate(*values, derivatives=True)
        target = np.real(np.sum(jacobian*weight, axis=-1))
        variables = tuple(torch.tensor(x, dtype=torch.float64, requires_grad=True)
                          for x in values)
        loss = torch.real(torch.sum(evaluate_torch_spline(model, *variables)
                                    * torch.tensor(weight)))
        actual = torch.autograd.grad(loss, variables)
        np.testing.assert_allclose([x.item() for x in actual], target,
                                   rtol=2e-10, atol=2e-10)
        def jax_loss(x, y, z):
            return jnp.real(jnp.sum(
                evaluate_jax_spline(model, x, y, z) * jnp.asarray(weight)))
        actual = jax.grad(jax_loss, argnums=(0, 1, 2))(*map(jnp.asarray, values))
        np.testing.assert_allclose(np.asarray(actual), target,
                                   rtol=2e-10, atol=2e-10)


def test_domains_and_bad_training_grids():
    model = _model()
    with pytest.raises(ValueError, match="outside"):
        model.evaluate(7.01, 0., 0.)
    with pytest.raises(ValueError, match="finite"):
        model.evaluate(np.nan, 0., 0.)
    assert torch.isnan(evaluate_torch_spline(model, 7.01, 0., 0.)).all()
    assert bool(jnp.isnan(evaluate_jax_spline(model, 7.01, 0., 0.)).all())

    bad = (np.linspace(1., 7., 3), *model.nodes[1:])
    shape = (3, 5, 5, len(model.frequency))
    with pytest.raises(ValueError, match="four"):
        SplineTensorInterpolant.fit(bad, model.frequency, np.zeros(shape),
                                    np.zeros(shape), ranks=(1, 1))


def test_arbitrary_frequency_spline_and_cross_backend_derivatives():
    model = _model()
    interpolant = FrequencyReducedInterpolant.from_model(model)
    frequency = np.array([.035, model.frequency[7], .217])
    values = (2.4, .2, -.1)
    amp, phi = _fields(*values, frequency)
    expected = np.exp(amp + 1j*phi)
    waveform, jacobian = interpolant.evaluate(frequency, *values,
                                               derivatives=True)
    np.testing.assert_allclose(waveform, expected, rtol=2e-12, atol=2e-12)
    damp_dq = frequency + 2*frequency**2*values[0]
    dphi_dq = 2*frequency*values[0] + frequency**3*values[1]
    np.testing.assert_allclose(jacobian[0], expected*(damp_dq + 1j*dphi_dq),
                               rtol=3e-11, atol=3e-11)

    torch_f = torch.tensor(frequency, requires_grad=True)
    torch_wave = evaluate_torch_frequency(interpolant, torch_f, *values)
    jax_wave = jax.jit(lambda f: evaluate_jax_frequency(interpolant, f, *values))(
        jnp.asarray(frequency))
    np.testing.assert_allclose(torch_wave.detach().numpy(), expected,
                               rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(np.asarray(jax_wave), expected,
                               rtol=2e-12, atol=2e-12)
    damp_df = values[0] + values[1]*values[2] + 2*frequency*(values[0]**2-values[1])
    dphi_df = values[0]**2 + values[1] + 3*frequency**2*(values[2]+values[0]*values[1])
    expected_grad = np.real(expected*(damp_df + 1j*dphi_df))
    torch_grad = torch.autograd.grad(torch_wave.real.sum(), torch_f)[0]
    jax_grad = jax.grad(lambda f: jnp.real(
        evaluate_jax_frequency(interpolant, f, *values)).sum())(
            jnp.asarray(frequency))
    np.testing.assert_allclose(torch_grad.detach().numpy(), expected_grad,
                               rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(np.asarray(jax_grad), expected_grad,
                               rtol=3e-10, atol=3e-10)

    torch_q = torch.tensor(values[0], dtype=torch.float64, requires_grad=True)
    q_loss = evaluate_torch_frequency(interpolant, frequency, torch_q,
                                       *values[1:]).real.sum()
    torch_q_grad = torch.autograd.grad(q_loss, torch_q)[0]
    jax_q_grad = jax.grad(lambda q: jnp.real(
        evaluate_jax_frequency(interpolant, frequency, q,
                               *values[1:])).sum())(jnp.asarray(values[0]))
    expected_q_grad = np.real(jacobian[0].sum())
    np.testing.assert_allclose(torch_q_grad.item(), expected_q_grad,
                               rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(np.asarray(jax_q_grad), expected_q_grad,
                               rtol=3e-10, atol=3e-10)


def test_symmetric_mass_ratio_and_linear_amplitude_contract():
    """Published coordinate/field choices retain physical q derivatives."""
    nu_nodes = np.array([.10, .14, .18, .22, .25])
    spin_nodes = np.array([-.8, -.4, 0., .4, .8])
    nodes = (nu_nodes, spin_nodes, spin_nodes)
    frequency = np.linspace(.01, .3, 20)
    nu, a, b, f = np.meshgrid(*nodes, frequency, indexing="ij")
    amplitude = 1 + f*(2*nu + a*b) + f**2*(nu**2 + .2*a)
    phase = f*(nu+a) + f**3*(b+nu*a)
    model = SplineTensorInterpolant.fit(
        nodes, frequency, amplitude, phase, ranks=(3, 2),
        first_coordinate="nu", amplitude_kind="linear")
    interpolant = FrequencyReducedInterpolant.from_model(model)
    qs = np.array([1., 2.4, 6.])
    spins1 = np.array([.2, -.3, .1])
    spins2 = np.array([-.1, .4, -.2])
    sample_f = np.array([.035, .147, .217])

    def expected(q, a, b, f):
        nu = q/(1+q)**2
        amp = 1 + f*(2*nu+a*b) + f**2*(nu**2+.2*a)
        phi = f*(nu+a) + f**3*(b+nu*a)
        wave = amp*np.exp(1j*phi)
        damp = 2*f + 2*f**2*nu
        dphi = f + f**3*a
        dq = np.exp(1j*phi)*(damp+1j*amp*dphi)*(1-q)/(1+q)**3
        return wave, dq

    for q, a, b in zip(qs, spins1, spins2):
        wave, dq = expected(q, a, b, frequency)
        actual, jac = model.evaluate(q, a, b, derivatives=True)
        np.testing.assert_allclose(actual, wave, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(jac[0], dq, rtol=3e-11, atol=3e-11)

    batched, batched_jac = interpolant.evaluate(
        sample_f, qs, spins1, spins2, derivatives=True)
    assert batched.shape == (3, 3) and batched_jac.shape == (3, 3, 3)
    for index, (q, a, b) in enumerate(zip(qs, spins1, spins2)):
        wave, dq = expected(q, a, b, sample_f)
        np.testing.assert_allclose(batched[index], wave, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(batched_jac[index, 0], dq, rtol=3e-11, atol=3e-11)
        wave, dq = expected(q, a, b, sample_f)
        actual, jac = interpolant.evaluate(sample_f, q, a, b, derivatives=True)
        np.testing.assert_allclose(actual, wave, rtol=3e-12, atol=3e-12)
        np.testing.assert_allclose(jac[0], dq, rtol=3e-11, atol=3e-11)

    q, a, b = 2.4, -.3, .4
    wave, dq = expected(q, a, b, sample_f)
    torch_q = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    torch_wave = evaluate_torch_frequency(interpolant, sample_f, torch_q, a, b)
    torch_grad = torch.autograd.grad(torch_wave.real.sum(), torch_q)[0]
    jax_wave = jax.jit(lambda x: evaluate_jax_frequency(
        interpolant, sample_f, x, a, b))(jnp.asarray(q))
    jax_grad = jax.grad(lambda x: jnp.real(evaluate_jax_frequency(
        interpolant, sample_f, x, a, b)).sum())(jnp.asarray(q))
    np.testing.assert_allclose(torch_wave.detach().numpy(), wave, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(np.asarray(jax_wave), wave, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(torch_grad.item(), dq.real.sum(), rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(np.asarray(jax_grad), dq.real.sum(), rtol=3e-10, atol=3e-10)
    nu = q/(1+q)**2
    amp = 1 + sample_f*(2*nu+a*b) + sample_f**2*(nu**2+.2*a)
    phi = sample_f*(nu+a) + sample_f**3*(b+nu*a)
    df = np.exp(1j*phi)*(
        2*nu+a*b+2*sample_f*(nu**2+.2*a)
        + 1j*amp*(nu+a+3*sample_f**2*(b+nu*a)))
    torch_f = torch.tensor(sample_f, dtype=torch.float64, requires_grad=True)
    torch_freq_grad = torch.autograd.grad(
        evaluate_torch_frequency(interpolant, torch_f, q, a, b).real.sum(),
        torch_f)[0]
    jax_freq_grad = jax.grad(lambda f: jnp.real(evaluate_jax_frequency(
        interpolant, f, q, a, b)).sum())(jnp.asarray(sample_f))
    np.testing.assert_allclose(torch_freq_grad.detach().numpy(), df.real,
                               rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(np.asarray(jax_freq_grad), df.real,
                               rtol=3e-10, atol=3e-10)
    torch_spline_q = torch.tensor(q, dtype=torch.float64, requires_grad=True)
    torch_spline_grad = torch.autograd.grad(
        evaluate_torch_spline(model, torch_spline_q, a, b).real.sum(),
        torch_spline_q)[0]
    _, full_dq = expected(q, a, b, frequency)
    jax_spline_grad = jax.grad(lambda x: jnp.real(evaluate_jax_spline(
        model, x, a, b)).sum())(jnp.asarray(q))
    np.testing.assert_allclose(torch_spline_grad.item(), full_dq.real.sum(),
                               rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(np.asarray(jax_spline_grad), full_dq.real.sum(),
                               rtol=3e-10, atol=3e-10)
    with pytest.raises(ValueError, match="at least one"):
        model.evaluate(.9, a, b)
    with pytest.raises(ValueError, match="at least one"):
        interpolant.evaluate(sample_f, .9, a, b)
    assert torch.isnan(evaluate_torch_spline(model, .9, a, b)).all()
    assert bool(jnp.isnan(evaluate_jax_spline(model, .9, a, b)).all())
    with pytest.raises(ValueError, match="positive"):
        SplineTensorInterpolant.fit(
            nodes, frequency, -amplitude, phase, ranks=(3, 2),
            first_coordinate="nu", amplitude_kind="linear")
    with pytest.raises(ValueError, match="symmetric-mass-ratio"):
        SplineTensorInterpolant.fit(
            (nu_nodes + .01, *nodes[1:]), frequency, amplitude, phase,
            ranks=(3, 2), first_coordinate="nu", amplitude_kind="linear")

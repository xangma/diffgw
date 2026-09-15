"""Equation and autodiff checks for the published TEOBResumS postmerger fit."""

import pytest
import torch

from torchwave.approximants.eob.teob_postmerger import (
    merger_amplitude,
    merger_frequency,
    nonspinning_ringdown_shape,
    nqc_targets,
    postmerger_22,
    qnm_fits,
)


def _time_derivative(value, time):
    """Differentiate a complex mode with respect to its time chart."""
    real_dot = torch.autograd.grad(value.real.sum(), time, create_graph=True)[0]
    imag_dot = torch.autograd.grad(value.imag.sum(), time, create_graph=True)[0]
    return torch.complex(real_dot, imag_dot)


def test_published_nonspinning_merger_and_qnm_limits():
    nu = torch.tensor([0.25, 0.16], dtype=torch.float64)
    amp = merger_amplitude(nu)
    freq = merger_frequency(nu)
    assert torch.allclose(amp, torch.tensor([1.577757875, 1.501293632], dtype=torch.float64))
    assert torch.allclose(freq, torch.tensor([0.3599296875, 0.3219874704], dtype=torch.float64), atol=1e-12)
    omega, alpha, alpha21 = qnm_fits(torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(torch.stack((omega, alpha, alpha21)),
                          torch.tensor([0.373672, 0.0889623, 0.184953], dtype=torch.float64))


def test_published_coefficients_preserve_float64_precision():
    chi = torch.tensor([0.0, 0.8], dtype=torch.float64)
    omega, alpha, alpha21 = qnm_fits(chi)
    expected = torch.tensor([0.373672, 0.0889623, 0.184953,
                             0.586395951020793, 0.075635607084509,
                             0.152348505886330],
                            dtype=torch.float64)
    assert torch.allclose(torch.stack((omega, alpha, alpha21), dim=1).flatten(),
                          expected, rtol=0.0, atol=2e-12)


def test_ringdown_is_continuous_at_merger_and_batched():
    dtype = torch.float64
    time = torch.tensor([0.0, 2.0, 4.0], dtype=dtype)
    out = postmerger_22(time, nu=0.25, final_mass=0.95, final_spin=0.69)
    expected_amp = 0.25 * merger_amplitude(torch.tensor(0.25, dtype=dtype))
    assert torch.allclose(out[0].real, expected_amp, atol=1e-12)
    assert torch.allclose(out[0].imag, torch.zeros((), dtype=dtype), atol=1e-12)
    assert out.shape == time.shape
    batch = postmerger_22(torch.stack((time, time + 0.5)),
                           nu=torch.tensor([0.25, 0.16], dtype=dtype),
                           final_mass=torch.tensor([0.95, 0.96], dtype=dtype),
                           final_spin=torch.tensor([0.69, 0.70], dtype=dtype))
    assert batch.shape == (2, 3)
    assert torch.isfinite(batch.real).all() and torch.isfinite(batch.imag).all()


def test_merger_frequency_and_zero_amplitude_rate_are_anchored():
    scalar_time = torch.tensor([1.75], dtype=torch.float64, requires_grad=True)
    scalar = postmerger_22(
        scalar_time,
        nu=0.20,
        final_mass=0.95,
        final_spin=0.68,
        phase0=0.31,
        merger_time=1.75,
    )
    scalar_dot = _time_derivative(scalar, scalar_time)
    scalar_frequency = -(scalar_dot / scalar).imag
    torch.testing.assert_close(
        scalar_frequency,
        merger_frequency(torch.tensor(0.20, dtype=torch.float64)).reshape(1),
        rtol=2e-14,
        atol=2e-15,
    )
    scalar_amplitude_dot = torch.autograd.grad(
        scalar.abs().sum(), scalar_time, create_graph=True
    )[0]
    torch.testing.assert_close(
        scalar_amplitude_dot,
        torch.zeros_like(scalar_amplitude_dot),
        rtol=0.0,
        atol=2e-14,
    )

    nu = torch.tensor([0.16, 0.24], dtype=torch.float64)
    final_mass = torch.tensor([0.94, 0.97], dtype=torch.float64)
    final_spin = torch.tensor([0.55, 0.72], dtype=torch.float64)
    merger_time = torch.tensor([2.25, 3.50], dtype=torch.float64)
    phase0 = torch.tensor([0.17, -0.41], dtype=torch.float64)
    selected_frequency = torch.tensor([0.37, 0.43], dtype=torch.float64)
    batch_time = merger_time[:, None].clone().detach().requires_grad_(True)
    batch = postmerger_22(
        batch_time,
        nu,
        final_mass,
        final_spin,
        phase0=phase0,
        merger_time=merger_time,
        merger_frequency_value=selected_frequency,
    )
    batch_dot = _time_derivative(batch, batch_time)
    batch_frequency = -(batch_dot / batch).imag[:, 0]
    torch.testing.assert_close(batch_frequency, selected_frequency, rtol=2e-14, atol=2e-15)
    batch_amplitude_dot = torch.autograd.grad(
        batch.abs().sum(), batch_time, create_graph=True
    )[0]
    torch.testing.assert_close(
        batch_amplitude_dot,
        torch.zeros_like(batch_amplitude_dot),
        rtol=0.0,
        atol=2e-14,
    )


def test_late_frequency_and_amplitude_rates_reach_qnm_limits():
    nu = torch.tensor([0.16, 0.24], dtype=torch.float64)
    final_mass = torch.tensor([0.94, 0.97], dtype=torch.float64)
    final_spin = torch.tensor([0.55, 0.72], dtype=torch.float64)
    merger_time = torch.tensor([1.25, 2.75], dtype=torch.float64)
    phase0 = torch.tensor([0.23, -0.19], dtype=torch.float64)
    omega1, alpha1, _ = qnm_fits(final_spin)
    late_tau = 300.0
    late_time = (merger_time + late_tau * final_mass)[:, None]
    late_time = late_time.clone().detach().requires_grad_(True)
    mode = postmerger_22(
        late_time,
        nu,
        final_mass,
        final_spin,
        phase0=phase0,
        merger_time=merger_time,
    )
    mode_dot = _time_derivative(mode, late_time)
    instantaneous_frequency = -(mode_dot / mode).imag[:, 0]
    amplitude_log_rate = torch.autograd.grad(
        mode.abs().log().sum(), late_time, create_graph=True
    )[0][:, 0]
    torch.testing.assert_close(
        instantaneous_frequency,
        omega1 / final_mass,
        rtol=2e-12,
        atol=2e-13,
    )
    torch.testing.assert_close(
        amplitude_log_rate,
        -alpha1 / final_mass,
        rtol=2e-12,
        atol=2e-13,
    )


def test_postmerger_parameters_and_time_have_finite_gradients():
    time = torch.tensor(
        [[2.0, 2.6, 3.4], [1.8, 2.3, 3.1]], dtype=torch.float64, requires_grad=True
    )
    nu = torch.tensor([0.18, 0.23], dtype=torch.float64, requires_grad=True)
    final_mass = torch.tensor([0.93, 0.96], dtype=torch.float64, requires_grad=True)
    final_spin = torch.tensor([0.48, 0.70], dtype=torch.float64, requires_grad=True)
    phase0 = torch.tensor([0.29, -0.22], dtype=torch.float64, requires_grad=True)
    merger_time = torch.tensor([1.5, 1.2], dtype=torch.float64, requires_grad=True)
    merger_frequency_value = torch.tensor([0.36, 0.41], dtype=torch.float64, requires_grad=True)
    mode = postmerger_22(
        time,
        nu,
        final_mass,
        final_spin,
        phase0=phase0,
        merger_time=merger_time,
        merger_frequency_value=merger_frequency_value,
    )
    weights = torch.tensor([[1.0, 1.3, 1.7], [0.8, 1.2, 1.9]], dtype=torch.float64)
    loss = (weights * mode.real).sum() + (0.5 * weights * mode.imag).sum()
    gradients = torch.autograd.grad(
        loss,
        (time, nu, final_mass, final_spin, phase0, merger_time, merger_frequency_value),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_ringdown_and_published_targets_are_differentiable():
    nu = torch.tensor([0.20, 0.24], dtype=torch.float64, requires_grad=True)
    out = postmerger_22(torch.tensor([1.0, 2.0], dtype=torch.float64), nu,
                        final_mass=0.95, final_spin=0.68)
    targets = nqc_targets(nu)
    loss = out.real.square().sum() + sum(x.square().sum() for x in targets.values())
    loss.backward()
    assert nu.grad is not None and torch.isfinite(nu.grad).all()


def test_nonspinning_shape_has_no_hidden_spin_dependence():
    nu = torch.tensor([0.01, 0.16, 0.25], dtype=torch.float64)
    shape = nonspinning_ringdown_shape(nu)
    assert torch.allclose(shape["c3A"], -0.561584 + 0.829868 * nu)
    assert torch.allclose(shape["c3phi"], 3.88838 + 0.455847 * nu)
    assert torch.allclose(shape["c4phi"], 1.49969 + 2.08223 * nu)


def test_nqc_amplitude_rate_matches_published_omega_times_orbital_rate():
    nu = torch.tensor(0.20, dtype=torch.float64)
    targets = nqc_targets(nu)
    omega = 0.286399 + 0.251240 * nu + 0.542717 * nu.square()
    orbital_rate = -((-0.00421428 - 0.0847947 * nu) / (1.0 + 16.1559 * nu))
    assert torch.allclose(targets["amplitude_rate"], omega * orbital_rate, atol=1e-14)


def test_postmerger_rejects_invalid_domains():
    time = torch.tensor([0.0, 1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="symmetric mass ratio"):
        merger_amplitude(0.0)
    with pytest.raises(ValueError, match="symmetric mass ratio"):
        merger_frequency(0.30)
    with pytest.raises(ValueError, match="final spin"):
        qnm_fits(1.0)
    with pytest.raises(ValueError, match="final_mass"):
        postmerger_22(time, 0.2, 0.0, 0.5)
    with pytest.raises(ValueError, match="final spin"):
        postmerger_22(time, 0.2, 0.95, 1.0)
    with pytest.raises(ValueError, match="phase0"):
        postmerger_22(time, 0.2, 0.95, 0.5, phase0=float("nan"))
    with pytest.raises(ValueError, match="merger_frequency_value"):
        postmerger_22(time, 0.2, 0.95, 0.5, merger_frequency_value=0.0)
    with pytest.raises(ValueError, match="time >= merger_time"):
        postmerger_22(time, 0.2, 0.95, 0.5, merger_time=0.5)

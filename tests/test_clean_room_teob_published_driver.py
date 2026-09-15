"""Primary-equation checks for the isolated TEOB factorized 22 driver."""
import torch

from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.teob_published_driver import (
    teob22_flux, teob22_mode, teob22_mode_and_frequency)


def setup():
    dtype = torch.float64
    nu = torch.tensor(.21, dtype=dtype)
    ham = SEOBNRv4Hamiltonian(nu, torch.tensor(0., dtype=dtype), torch.tensor(0., dtype=dtype))
    rr = SEOBNRv4PublishedFlux(spin_truncation="tabulated")
    state = torch.tensor([8., .2, -.02, 3.5], dtype=dtype)
    return state, ham, rr


def test_flux_is_exactly_22_and_excludes_21():
    state, ham, _ = setup()
    omega = ham.equations_of_motion_conservative(state[0], state[2], state[3])[1]
    mode22 = teob22_mode(state, ham, omega=omega)
    expected = (2 * omega).square() * mode22.abs().square() / (8 * torch.pi)
    torch.testing.assert_close(teob22_flux(state, ham, omega=omega), expected)


def test_directional_frequency_matches_complex_flow():
    state, ham, rr = setup()
    mode, frequency = teob22_mode_and_frequency(state, ham, rr)
    field = EOBTrajectoryIntegrator(ham, rr).vector_field(state)
    eps = 1e-5
    derivative = (teob22_mode(state + eps * field, ham) -
                  teob22_mode(state - eps * field, ham)) / (2 * eps)
    torch.testing.assert_close(frequency, -(derivative / mode).imag, rtol=2e-7, atol=1e-9)


def test_no_grad_returns_detached_and_preserves_batch_shape():
    state, ham, rr = setup()
    state = state.expand(2, 3, 4).clone()
    with torch.no_grad():
        mode, frequency = teob22_mode_and_frequency(state, ham, rr)
    assert mode.shape == (2, 3)
    assert frequency.shape == (2, 3)
    assert not mode.requires_grad and not frequency.requires_grad


def test_mode_and_frequency_retain_state_gradient():
    state, ham, rr = setup()
    state.requires_grad_(True)
    mode, frequency = teob22_mode_and_frequency(state, ham, rr)
    loss = mode.real.square() + mode.imag.square() + frequency.square()
    gradient, = torch.autograd.grad(loss, state)
    assert gradient.shape == state.shape
    assert bool(torch.isfinite(gradient).all())


def test_inference_constructed_inputs_are_supported():
    expected_state, expected_ham, expected_rr = setup()
    expected_mode, expected_frequency = teob22_mode_and_frequency(
        expected_state, expected_ham, expected_rr)
    with torch.inference_mode():
        state, ham, rr = setup()
        mode, frequency = teob22_mode_and_frequency(state, ham, rr)
    assert mode.shape == () and frequency.shape == ()
    assert not mode.requires_grad and not frequency.requires_grad
    torch.testing.assert_close(mode, expected_mode)
    torch.testing.assert_close(frequency, expected_frequency)
    assert bool(torch.isfinite(mode.real) & torch.isfinite(mode.imag))
    assert bool(torch.isfinite(frequency))

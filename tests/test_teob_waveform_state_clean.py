"""Contracts for the explicit clean TEOB state-to-mode adapter."""

import pytest
import torch

from torchwave.approximants.eob.teob_radiation_clean import TEOBRadiationClean
from torchwave.approximants.eob.teob_waveform_state_clean import (
    teob_state_mode,
    teob_waveform_state_mode,
)
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


DTYPE = torch.float64


def _hamiltonian(*, chi1=0.0, chi2=0.0, lambda1=0.0, lambda2=0.0,
                 lambda1_oct=None, lambda2_oct=None):
    def tensor(value):
        return torch.as_tensor(value, dtype=DTYPE)

    return TEOBResumSHamiltonian(
        tensor(0.22), tensor(chi1), tensor(chi2), tensor(lambda1), tensor(lambda2),
        None if lambda1_oct is None else tensor(lambda1_oct),
        None if lambda2_oct is None else tensor(lambda2_oct),
    )


def _state(*, r=10.0, phi=0.17, pr_star=-0.03, pphi=3.0, requires_grad=False):
    return torch.tensor([r, phi, pr_star, pphi], dtype=DTYPE, requires_grad=requires_grad)


def _explicit_mode(state, hamiltonian, ell, m, omega):
    """Build the direct clean-radiation call used as an adapter oracle."""
    r, phi, pr_star, pphi = state.unbind(-1)
    h_eff = hamiltonian.effective_hamiltonian(r, pr_star, pphi)
    h_real = hamiltonian.real_hamiltonian_from_eff(h_eff)
    h_circ = hamiltonian.real_hamiltonian(
        r, torch.zeros_like(pr_star), pphi,
    )
    a_prime = hamiltonian.metric_potential_derivative(r)
    total_mass = hamiltonian.m1 + hamiltonian.m2
    m1 = hamiltonian.m1 / total_mass
    m2 = hamiltonian.m2 / total_mass
    return TEOBRadiationClean(max(2, ell), residual_phase="zero-unpublished").mode(
        ell, m, r, omega, phi, pr_star, pphi, h_eff, h_real, h_circ, a_prime,
        m1, m2,
    )


def test_scalar_state_routes_directly_to_clean_radiation():
    ham = _hamiltonian()
    state = _state()
    omega = ham.equations_of_motion_conservative(
        state[0], state[2], state[3],
    )[1]

    result = teob_waveform_state_mode(state, ham, 2, 2, Omega=omega)
    expected = _explicit_mode(state, ham, 2, 2, omega)
    assert result.ndim == 0
    assert result.is_complex()
    assert torch.allclose(result, expected, rtol=1e-13, atol=0)


def test_omitted_and_explicit_orbital_frequency_agree():
    ham = _hamiltonian()
    state = _state()
    omega = ham.equations_of_motion_conservative(
        state[0], state[2], state[3],
    )[1]
    omitted = teob_waveform_state_mode(state, ham, 2, 2)
    explicit_upper = teob_waveform_state_mode(state, ham, 2, 2, Omega=omega)
    explicit_lower = teob_state_mode(state, ham, 2, 2, omega=omega)
    assert torch.equal(omitted, explicit_upper)
    assert torch.equal(omitted, explicit_lower)
    with pytest.raises(TypeError, match="only one"):
        teob_waveform_state_mode(state, ham, 2, 2, Omega=omega, omega=omega)


def test_batched_states_match_independent_scalar_calls():
    ham = _hamiltonian()
    states = torch.tensor(
        [
            [10.0, 0.17, -0.03, 3.00],
            [11.0, -0.20, -0.02, 3.10],
            [9.0, 0.70, 0.01, 2.80],
        ],
        dtype=DTYPE,
    )
    batched = teob_waveform_state_mode(states, ham, 3, 2)
    scalar = torch.stack([
        teob_waveform_state_mode(row, ham, 3, 2) for row in states
    ])
    assert batched.shape == (3,)
    assert torch.allclose(batched, scalar, rtol=1e-13, atol=0)


def test_state_gradients_propagate_through_hamiltonian_and_radiation():
    ham = _hamiltonian()
    state = _state(requires_grad=True)
    mode = teob_waveform_state_mode(state, ham, 2, 2)
    loss = mode.real.square() + mode.imag.square()
    gradient, = torch.autograd.grad(loss, state)
    assert torch.isfinite(gradient).all()
    assert torch.any(gradient != 0)


def test_negative_m_uses_clean_conjugate_symmetry():
    ham = _hamiltonian()
    state = _state()
    positive = teob_waveform_state_mode(state, ham, 3, 2)
    negative = teob_waveform_state_mode(state, ham, 3, -2)
    assert torch.allclose(negative, -positive.conj(), rtol=1e-13, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chi1": 0.1},
        {"chi2": -0.2},
        {"lambda1": 10.0},
        {"lambda2": 20.0},
        {"lambda1_oct": 5.0},
        {"lambda2_oct": 7.0},
    ],
)
def test_unsupported_spin_and_tide_sectors_are_explicitly_rejected(kwargs):
    with pytest.raises(ValueError, match="nonspinning vacuum"):
        _hamiltonian(**kwargs)

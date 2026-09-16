"""Focused checks for the Hamiltonian-to-mode phase/radiation seam."""

import pytest
import torch

from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.seobnrv4_mode22 import (
    seobnrv4_published_delta22,
    seobnrv4_published_mode22,
    seobnrv4_published_mode22_from_hamiltonian,
)
from torchwave.approximants.eob.nqc import nqc_22_factor


PHASE = dict(spin_v6="bohe_total_energy", include_pan_v8=False,
             include_positive_chi_540=False)


def setup():
    state = torch.tensor([[8.0, 0.2, -0.02, 3.5],
                          [7.9, 0.24, -0.021, 3.49]], dtype=torch.float64)
    ham = SEOBNRv4Hamiltonian(torch.tensor(.24, dtype=torch.float64),
                              torch.tensor(.1, dtype=torch.float64),
                              torch.tensor(-.2, dtype=torch.float64))
    return state, ham, SEOBNRv4PublishedFlux(spin_truncation="tabulated")


def direct_mode(state, ham, radiation, *, omega=None, nqc_coefficients=None,
                nqc_sample_dims=0):
    quantities = radiation.from_hamiltonian(state, ham, omega=omega)
    residual = seobnrv4_published_delta22(
        **{key: quantities[key] for key in
           ("omega", "hreal_over_M", "x1", "x2", "chi1", "chi2")},
        **PHASE,
    )
    mode = seobnrv4_published_mode22(
        quantities["omega"], state[..., 1], residual_phase=residual,
        **{key: quantities[key] for key in quantities if key != "omega"},
    )
    if nqc_coefficients is not None:
        mode = mode*nqc_22_factor(state[..., 0], state[..., 2],
                                  quantities["omega"], nqc_coefficients,
                                  sample_dims=nqc_sample_dims)
    return mode


def test_hamiltonian_adapter_matches_explicit_phase_and_mode_composition():
    state, ham, radiation = setup()
    actual = seobnrv4_published_mode22_from_hamiltonian(
        state, ham, radiation, **PHASE)
    torch.testing.assert_close(actual, direct_mode(state, ham, radiation))


def test_supplied_frequency_and_caller_nqc_coefficients_are_retained():
    state, ham, radiation = setup()
    omega = radiation.from_hamiltonian(state, ham)["omega"]*1.01
    coefficients = (torch.tensor([.03, -.02, .01], dtype=state.dtype),
                    torch.tensor([.04, -.01], dtype=state.dtype))
    actual = seobnrv4_published_mode22_from_hamiltonian(
        state, ham, radiation, omega=omega, nqc_coefficients=coefficients,
        **PHASE)
    expected = direct_mode(state, ham, radiation, omega=omega,
                           nqc_coefficients=coefficients)
    torch.testing.assert_close(actual, expected)


def test_trajectory_batch_uses_explicit_nqc_sample_axis():
    base_state, _, radiation = setup()
    state = base_state[:1].expand(2, 3, 4).clone()
    state[..., 1] += torch.tensor([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])
    state[..., 0] += torch.tensor([[0.0, 0.1, 0.2], [0.2, 0.1, 0.0]])
    ham = SEOBNRv4Hamiltonian(
        torch.tensor([.24, .20], dtype=state.dtype),
        torch.tensor([.10, .15], dtype=state.dtype),
        torch.tensor([-.20, -.10], dtype=state.dtype),
    )
    coefficients = (torch.tensor([[.03, -.02, .01], [.01, .02, -.03]],
                                 dtype=state.dtype),
                    torch.tensor([[.04, -.01], [-.02, .03]], dtype=state.dtype))
    actual = seobnrv4_published_mode22_from_hamiltonian(
        state, ham, radiation, nqc_coefficients=coefficients,
        nqc_sample_dims=1, **PHASE)
    expected = direct_mode(state, ham, radiation,
                           nqc_coefficients=coefficients, nqc_sample_dims=1)
    torch.testing.assert_close(actual, expected)
    # Each row has independent coefficients; shared coefficients would erase
    # this batch distinction and are therefore not silently inferred.
    assert not torch.allclose(actual[0], actual[1])


@pytest.mark.parametrize("missing", ["spin_v6", "include_pan_v8",
                                      "include_positive_chi_540"])
def test_phase_inheritance_choices_are_required_at_the_wiring_boundary(missing):
    state, ham, radiation = setup()
    kwargs = dict(PHASE)
    del kwargs[missing]
    with pytest.raises(TypeError):
        seobnrv4_published_mode22_from_hamiltonian(state, ham, radiation,
                                                    **kwargs)


def test_radiation_force_adapter_rejects_zero_angular_momentum():
    _, ham, radiation = setup()
    with pytest.raises(ValueError, match="p_phi>0"):
        radiation.radiation_reaction_forces(
            torch.tensor(.02), torch.tensor(0.), torch.tensor(0.),
            r=torch.tensor(8.), hamiltonian=ham)

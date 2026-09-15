"""Focused checks for the explicit UIB2016/postmerger adapter."""

import pytest
import torch

from torchwave.approximants.eob.teob_postmerger import postmerger_22
from torchwave.approximants.eob.teob_uib2016_postmerger import (
    uib2016_nonspinning_postmerger_22,
)
from torchwave.approximants.eob.teob_uib2016_remnant import uib2016_nonspinning_remnant


def test_adapter_passes_published_uib_remnant_into_postmerger():
    time = torch.tensor([0.0, 2.0, 8.0], dtype=torch.float64)
    mass1, mass2 = torch.tensor(2.0, dtype=time.dtype), torch.tensor(1.0, dtype=time.dtype)
    final_mass, final_spin = uib2016_nonspinning_remnant(mass1, mass2)
    expected = postmerger_22(time, (mass1 * mass2) / (mass1 + mass2).square(),
                             final_mass, final_spin)
    actual = uib2016_nonspinning_postmerger_22(time, mass1, mass2)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_adapter_supports_batched_masses_and_gradients():
    time = torch.stack((torch.linspace(0.0, 5.0, 6, dtype=torch.float64),
                        torch.linspace(0.0, 5.0, 6, dtype=torch.float64)))
    masses = torch.tensor([2.0, 5.0], dtype=torch.float64, requires_grad=True)
    mode = uib2016_nonspinning_postmerger_22(time, masses, masses.new_tensor([1.0, 2.0]))
    assert mode.shape == time.shape and torch.isfinite(mode.real).all()
    gradient = torch.autograd.grad(mode.real.square().sum() + mode.imag.square().sum(), masses)[0]
    assert torch.isfinite(gradient).all() and (gradient.abs() > 0).all()


def test_adapter_preserves_explicit_phase_and_merger_frequency():
    time = torch.tensor([3.0, 4.0], dtype=torch.float64)
    phase = torch.tensor(0.37, dtype=torch.float64)
    frequency = torch.tensor(0.42, dtype=torch.float64)
    actual = uib2016_nonspinning_postmerger_22(
        time, 2.0, 1.0, phase0=phase, merger_time=3.0,
        merger_frequency_value=frequency)
    reference = postmerger_22(
        time, 2.0 / 9.0, *uib2016_nonspinning_remnant(2.0, 1.0),
        phase0=phase, merger_time=3.0, merger_frequency_value=frequency)
    torch.testing.assert_close(actual, reference)


def test_adapter_rejects_invalid_component_masses():
    with pytest.raises(ValueError, match="component masses"):
        uib2016_nonspinning_postmerger_22(torch.tensor([0.0, 1.0]), 0.0, 1.0)


def test_adapter_accepts_nearly_equal_masses_at_eta_roundoff_boundary():
    time = torch.tensor([0.0, 1.0], dtype=torch.float64)
    mode = uib2016_nonspinning_postmerger_22(time, 1.0, 1.0 + 1e-8)
    assert torch.isfinite(mode).all()

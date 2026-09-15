import torch
import pytest

from torchwave.approximants.xhm_phase_clean import (
    xhm_32_spheroidal_collocation_frequencies,
    xhm_32_spheroidal_phase,
    xhm_32_spheroidal_phase_coefficients,
    xhm_32_spheroidal_phase_derivative,
)


def test_32_spheroidal_four_conditions_recover_coefficients():
    ring22, ring32, damp = 0.080, 0.095, 0.012
    expected = torch.tensor([0.031, -0.004, 0.72, -1.3], dtype=torch.float64)
    points = xhm_32_spheroidal_collocation_frequencies(ring22, ring32, damp)
    targets = xhm_32_spheroidal_phase_derivative(points, ring32, damp, expected)
    actual = xhm_32_spheroidal_phase_coefficients(ring22, ring32, damp, targets)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


def test_32_spheroidal_fourth_target_is_used_and_differentiable():
    ring22 = torch.tensor(0.080, dtype=torch.float64, requires_grad=True)
    ring32 = torch.tensor(0.095, dtype=torch.float64, requires_grad=True)
    damp = torch.tensor(0.012, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([1.2, 1.3, 1.4, 1.5], dtype=torch.float64)
    coeff = xhm_32_spheroidal_phase_coefficients(ring22, ring32, damp, targets)
    shifted = xhm_32_spheroidal_phase_coefficients(ring22, ring32, damp, targets + torch.tensor([0., 0., 0., 1.]))
    assert torch.abs(shifted - coeff).max() > 1e-8
    value = xhm_32_spheroidal_phase(0.12, ring32, damp, coeff)
    torch.autograd.grad(value, (ring22, ring32, damp))


def test_32_spheroidal_requires_four_targets():
    try:
        xhm_32_spheroidal_phase_coefficients(0.08, 0.095, 0.012, [1., 2., 3.])
    except ValueError as exc:
        assert "four" in str(exc)
    else:
        raise AssertionError("three-condition reduced system was accepted")


def test_32_spheroidal_phase_derivative_matches_integrated_phase_and_batches():
    ring22 = torch.tensor([0.080, 0.082], dtype=torch.float64)
    ring32 = torch.tensor([0.095, 0.097], dtype=torch.float64)
    damp = torch.tensor([0.012, 0.013], dtype=torch.float64)
    targets = torch.tensor([[1.2, 1.3, 1.4, 1.5], [1.1, 1.25, 1.35, 1.45]], dtype=torch.float64)
    coeff = xhm_32_spheroidal_phase_coefficients(ring22, ring32, damp, targets)
    frequency = torch.tensor([0.11, 0.12], dtype=torch.float64, requires_grad=True)
    phase = xhm_32_spheroidal_phase(frequency, ring32, damp, coeff)
    derivative = torch.stack([torch.autograd.grad(v, frequency, retain_graph=True)[0][i] for i, v in enumerate(phase)])
    expected = xhm_32_spheroidal_phase_derivative(frequency, ring32, damp, coeff)
    torch.testing.assert_close(derivative, expected, rtol=2e-12, atol=2e-12)
    assert coeff.shape == (2, 4)


def test_32_spheroidal_rejects_nonfinite_and_coincident_nodes():
    with pytest.raises(ValueError, match="finite"):
        xhm_32_spheroidal_phase_coefficients(0.08, 0.095, 0.012, [1., 2., float("nan"), 4.])
    with pytest.raises(ValueError, match="distinct"):
        xhm_32_spheroidal_phase_coefficients(0.08, 0.095, 0.01, [1., 2., 3., 4.])

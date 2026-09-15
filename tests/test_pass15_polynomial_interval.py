"""Clean-room checks for the XHM polynomial interval guard."""

import pytest
import torch

from torchwave.approximants.polynomial_interval import polynomial_has_nonpositive


def test_finds_narrow_negative_dip_between_the_old_eleven_nodes():
    # The negative interval is approximately [0.52684, 0.53316], so the
    # 0.0, 0.1, ..., 1.0 sampling grid does not evaluate it.
    coefficients = torch.tensor([0.28089, -1.06, 1.0], dtype=torch.float64)
    assert polynomial_has_nonpositive(coefficients, 0.0, 1.0).item()
    nodes = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    sampled = sum(coefficients[k] * nodes**k for k in range(coefficients.numel()))
    assert torch.all(sampled > 0)


def test_includes_tangent_root_and_rejects_positive_near_zero_minimum():
    tangent = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    positive = torch.tensor([1.0e-12, 0.0, 1.0], dtype=torch.float64)
    assert polynomial_has_nonpositive(tangent, -1.0, 1.0).item()
    assert not polynomial_has_nonpositive(positive, -1.0, 1.0).item()


def test_mixed_batch_broadcasts_bounds_and_handles_degree_degeneracy():
    coefficients = torch.tensor(
        [
            [2.0, 0.0, 0.0],             # positive constant
            [-3.0, 0.0, 0.0],            # negative constant
            [1.0, -2.0, 0.0],            # linear root at the right endpoint
            [1.0, 0.0, 1.0],             # positive quadratic
            [0.28089, -1.06, 1.0],       # narrow dip
        ],
        dtype=torch.float64,
    )
    result = polynomial_has_nonpositive(coefficients, 0.0, 1.0)
    expected = torch.tensor([False, True, True, False, True])
    torch.testing.assert_close(result, expected)

    # A batch-shaped lower bound broadcasts against a single polynomial.
    broadcast = polynomial_has_nonpositive(
        torch.tensor([1.0, -2.0], dtype=torch.float64),
        torch.tensor([0.0, 0.6]),
        torch.tensor([0.4, 1.0]),
    )
    torch.testing.assert_close(broadcast, torch.tensor([False, True]))


def test_scaled_interval_and_coefficients_remain_device_native():
    # The stationary point is at x=1e6; the positive offset is deliberately
    # much smaller than the polynomial's coefficient scale.
    center = 1.0e6
    coefficients = torch.tensor([center * center + 1.0e-3, -2.0 * center, 1.0], dtype=torch.float64)
    assert not polynomial_has_nonpositive(coefficients, center - 1.0, center + 1.0).item()

    # Rescaling the whole polynomial must not turn a positive or negative
    # decision into a unit-sized absolute-threshold decision.
    narrow_dip = torch.tensor([0.28089, -1.06, 1.0], dtype=torch.float64)
    assert polynomial_has_nonpositive(narrow_dip * 1.0e-20, 0.0, 1.0).item()
    assert polynomial_has_nonpositive(narrow_dip * 1.0e20, 0.0, 1.0).item()
    tiny_positive = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64) * 1.0e-20
    assert not polynomial_has_nonpositive(tiny_positive, -1.0, 1.0).item()


def test_decision_is_detached_and_nonfinite_inputs_are_conservative():
    coefficients = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64, requires_grad=True)
    decision = polynomial_has_nonpositive(coefficients, 0.0, 1.0)
    assert decision.dtype is torch.bool
    assert not decision.requires_grad
    assert polynomial_has_nonpositive(torch.tensor([float("nan")]), 0.0, 1.0).item()
    assert polynomial_has_nonpositive(torch.tensor([1.0]), float("-inf"), 1.0).item()


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_available_accelerator_matches_cpu(device):
    if not getattr(torch, f"{device}").is_available():
        pytest.skip(f"{device} unavailable")
    coefficients = torch.tensor([0.28089, -1.06, 1.0], dtype=torch.float32, device=device)
    result = polynomial_has_nonpositive(coefficients, 0.0, 1.0)
    assert result.device.type == device
    assert result.item()

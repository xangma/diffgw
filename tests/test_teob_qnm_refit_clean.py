"""Tests for the bounded independent TEOB QNM calibration experiment."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from refit_teob_qnm_independent_clean import (  # noqa: E402
    SPIN_MAX, denominator_poles, fit_rational, rational, validate_coefficients,
    _targets,
)


def test_synthetic_cubic_rational_recovery():
    spins = np.linspace(0.0, SPIN_MAX, 31)
    expected = np.array([1.7, -0.3, 0.12, 0.02, -0.2, 0.08, -0.01])
    recovered, result = fit_rational(spins, rational(spins, expected))
    assert result.success
    np.testing.assert_allclose(recovered, expected, rtol=2e-7, atol=2e-9)


def test_independent_generated_holdouts_are_small_and_positive():
    train = np.linspace(0.0, SPIN_MAX, 41)
    holdout = np.array([0.037, 0.173, 0.311, 0.527, 0.713, 0.887, 0.943])
    train_targets, holdout_targets = _targets(train), _targets(holdout)
    for name, values in train_targets.items():
        coefficients, _ = fit_rational(train, values)
        validate_coefficients(coefficients)
        assert np.max(np.abs(rational(holdout, coefficients) - holdout_targets[name])) < 3e-4


def test_pole_rejection_is_explicit():
    pole = np.array([1.0, 0, 0, 0, -1.0 / 0.5, 0, 0])
    assert np.any(np.isclose(denominator_poles(pole), 0.5, atol=1e-12))
    with pytest.raises(ValueError, match="pole"):
        validate_coefficients(pole)


def test_two_close_poles_are_rejected_without_grid_sampling():
    roots = (0.5000001, 0.5000002, 2.0)
    # denominator = product(1 - spin/root), so its constant is exactly one.
    c1 = -sum(1.0 / r for r in roots)
    c2 = sum(1.0 / (roots[i] * roots[j]) for i in range(3) for j in range(i + 1, 3))
    c3 = -np.prod([1.0 / r for r in roots])
    with pytest.raises(ValueError, match="pole"):
        validate_coefficients([1.0, 0.0, 0.0, 0.0, c1, c2, c3])


def test_independent_qnm_numeric_consistency_preserves_dtype():
    from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm
    spin = torch.tensor([0.2, 0.7], dtype=torch.float64)
    omega, alpha = dimensionless_qnm(2, 2, 0, spin)
    scalar = dimensionless_qnm(2, 2, 0, spin[0])
    assert omega.dtype == spin.dtype and alpha.dtype == spin.dtype
    torch.testing.assert_close(omega[0], scalar[0])
    torch.testing.assert_close(alpha[0], scalar[1])

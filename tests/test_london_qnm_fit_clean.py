"""Focused checks for the independent London--Fauchon-Jones reconstruction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_london_qnm_fit_clean.py"
SPEC = importlib.util.spec_from_file_location("torchwave_london_fit_clean", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fit
SPEC.loader.exec_module(fit)


def test_declared_grids_are_nested_and_check_nodes_are_distinct() -> None:
    assert fit.GRID_COUNTS == (65, 129, 257)
    finest_theta = fit.theta_grid(257)
    for count in fit.GRID_COUNTS:
        indices = fit.nested_indices(count)
        assert len(indices) == count
        np.testing.assert_allclose(fit.theta_grid(count), finest_theta[indices])
        assert fit.spin_grid(count)[0] == -fit.SPIN_LIMIT
        assert fit.spin_grid(count)[-1] == fit.SPIN_LIMIT

    training_theta = set(np.round(finest_theta, 14))
    assert not any(round(float(theta), 14) in training_theta for theta in fit.check_theta_grid())


def test_main_zero_damped_fit_recovers_declared_polynomial() -> None:
    mode = (2, 2, 0)
    spins = np.linspace(-0.8, 0.8, 19)
    expected = np.asarray(
        [0.11 + 0.02j, -0.31 + 0.07j, 0.17 - 0.04j, 0.05 + 0.03j, -0.02 + 0.01j],
        dtype=complex,
    )
    frequencies = fit.evaluate_coefficients(spins, mode, expected, "main")

    coefficients, quality = fit.fit_coefficients(spins, frequencies, mode, "main")

    np.testing.assert_allclose(coefficients, expected, rtol=0.0, atol=1e-12)
    assert quality["target"] == "(omega-m/2)/kappa"
    assert quality["training_residual"]["max_abs_frequency"] < 1e-12


def test_main_nonzero_damped_fit_uses_seven_powers() -> None:
    mode = (3, 2, 0)
    spins = np.linspace(-0.75, 0.75, 23)
    expected = np.asarray(
        [0.41 - 0.02j, -0.18 + 0.05j, 0.09 + 0.03j, -0.04 - 0.01j, 0.02 + 0.008j, -0.01 + 0.004j, 0.003 - 0.002j],
        dtype=complex,
    )
    frequencies = fit.evaluate_coefficients(spins, mode, expected, "main")

    coefficients, quality = fit.fit_coefficients(spins, frequencies, mode, "main")

    np.testing.assert_allclose(coefficients, expected, rtol=0.0, atol=1e-9)
    assert quality["powers"] == list(range(7))


def test_direct_constrained_weighted_alternative_is_reported_separately() -> None:
    mode = (3, 3, 0)
    spins = np.linspace(-0.7, 0.7, 17)
    expected = np.asarray([0.2 + 0.01j, -0.06 + 0.03j, 0.04 - 0.02j, 0.01 + 0.004j, -0.003 + 0.002j])
    frequencies = fit.evaluate_coefficients(spins, mode, expected, "direct_weighted")

    coefficients, quality = fit.fit_coefficients(spins, frequencies, mode, "direct_weighted")

    np.testing.assert_allclose(coefficients, expected, rtol=0.0, atol=1e-11)
    assert quality["powers"] == list(range(1, 6))
    assert quality["weight_definition"] == "residual multiplier kappa**(-1/2), sample weight 1/kappa"


def test_printed_phase_comparison_wraps_at_zero() -> None:
    base, printed = fit._printed_polar((2, 2, 0))
    assert base == 1.0
    reference = printed[0][0] * np.exp(1j * (printed[0][1] - 2 * np.pi))
    comparison = fit.compare_printed([reference] + [1.0 + 0.0j] * 4, (2, 2, 0))
    assert comparison["coefficients"][0]["within_phase_rounding_interval"]


def test_independent_solver_returns_certified_physical_root() -> None:
    solve = fit._load_solver()
    root = fit._solve_certified(solve, 0.0, (2, 2, 0))
    assert root.converged
    assert root.omega.real > 0.0
    assert root.omega.imag < 0.0
    assert root.metadata["estimated_frequency_error"] < 1e-9

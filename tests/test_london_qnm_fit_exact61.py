"""Focused checks for the exact independent N=61 London QNM reconstruction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_london_qnm_fit_exact61.py"
SPEC = importlib.util.spec_from_file_location("torchwave_london_fit_exact61", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fit
SPEC.loader.exec_module(fit)


def test_exact_grid_is_61_and_uses_declared_spin_map() -> None:
    theta = fit.exact_theta_grid()
    spins = fit.exact_spin_grid()
    assert len(theta) == fit.EXACT_GRID_COUNT == 61
    np.testing.assert_allclose(theta, np.linspace(-np.pi / 2, np.pi / 2, 61))
    np.testing.assert_allclose(spins, fit.SPIN_LIMIT * np.sin(theta))
    assert spins[0] == -fit.SPIN_LIMIT
    assert spins[-1] == fit.SPIN_LIMIT


def test_exact_grid_is_not_a_nested_slice_or_interpolated_grid() -> None:
    theta257 = np.linspace(-np.pi / 2, np.pi / 2, 257)
    theta61 = fit.exact_theta_grid()
    common_nodes = sum(np.any(np.isclose(value, theta257, atol=0.0, rtol=0.0)) for value in theta61)
    assert common_nodes < fit.EXACT_GRID_COUNT
    assert common_nodes == 5  # endpoints, center, and the two quarter points


def test_exact_fit_preserves_main_basis_and_reports_rounding_rows() -> None:
    helper = fit._load_helper()
    mode = (2, 2, 0)
    spins = np.linspace(-0.8, 0.8, 19)
    expected = np.asarray(
        [0.11 + 0.02j, -0.31 + 0.07j, 0.17 - 0.04j, 0.05 + 0.03j, -0.02 + 0.01j],
        dtype=complex,
    )
    frequencies = helper.evaluate_coefficients(spins, mode, expected, "main")
    rows = [
        {"theta": float(theta), "spin": float(spin), "omega_paper": complex(value)}
        for theta, spin, value in zip(np.linspace(-np.pi / 2, np.pi / 2, len(spins)), spins, frequencies)
    ]
    result = fit._fit_exact_rows(rows, rows, mode)
    np.testing.assert_allclose(result["coefficients_rectangular"], expected, rtol=0.0, atol=1e-12)
    assert result["fit"]["powers"] == [0, 1, 2, 3, 4]
    assert len(result["printed_comparison"]["coefficients"]) == 5
    assert result["independent_check_main"]["max_abs_frequency"] < 1e-12


def test_source_hashes_cover_exact_script_solver_helper_and_printed_reference() -> None:
    hashes = fit._source_hashes()
    assert set(hashes) == {
        "scripts/diagnose_london_qnm_fit_exact61.py",
        "scripts/diagnose_london_qnm_fit_clean.py",
        "scripts/kerr_leaver_solver.py",
        "src/torchwave/approximants/kerr_qnm_london_published.py",
    }
    assert all(len(value) == 64 for value in hashes.values())

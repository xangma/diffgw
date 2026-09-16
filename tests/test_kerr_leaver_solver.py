"""Equation/continuation checks for the independent generation-time solver.

No QNM table or waveform oracle supplies expected values in this module.
"""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "torchwave_independent_leaver", Path(__file__).resolve().parents[1]/"scripts/kerr_leaver_solver.py")
solver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = solver
SPEC.loader.exec_module(solver)


@pytest.mark.parametrize("ell,m", [(2, 2), (2, -2), (3, 2), (4, 4)])
def test_angular_zero_spin_has_exact_spherical_eigenvalue(ell, m):
    assert solver.angular_eigenvalue(0, ell, m) == ell*(ell+1)-2


@pytest.mark.parametrize("ell", [2, 3, 4])
def test_large_ell_wkb_continuation_labels_distinct_schwarzschild_overtones(ell):
    roots = [solver.solve(0, ell, ell, n) for n in range(3)]
    assert all(result.converged for result in roots)
    assert all(result.metadata["branch_origin"].startswith("large-ell WKB") for result in roots)
    frequencies = np.array([result.omega for result in roots])
    assert np.all(frequencies.real > 0)
    assert np.all(np.diff(frequencies.real) < 0)
    assert np.all(np.diff(frequencies.imag) < 0)
    assert np.min(abs(np.diff(frequencies))) > .1
    # All inversions must vanish for each root: inversion is a numerical
    # conditioning choice, not an independent overtone label.
    for result in roots:
        for inversion in range(4):
            value = solver.radial_residual(0, ell, inversion, result.omega,
                                           result.separation_constant, 2048)
            assert abs(value) < 3e-11


@pytest.mark.parametrize("a,ell,m,n", [(.2, 2, 2, 2), (.7, 3, 2, 1), (.99, 4, 4, 0)])
def test_spinning_roots_agree_after_independent_depth_and_basis_refinement(a, ell, m, n):
    ordinary = solver.solve(a, ell, m, n)
    strict = solver.solve(a, ell, m, n, omega_seed=ordinary.omega, strict=True)
    assert ordinary.converged and strict.converged
    assert abs(strict.omega-ordinary.omega) < 2e-12
    assert strict.metadata["angular_ell_max"] > ordinary.metadata["angular_ell_max"]
    assert strict.metadata["minimal_remainder_real_part"] < 0
    assert strict.metadata["estimated_frequency_error"] < 3e-13
    assert abs(solver.radial_residual(a, m, n, strict.omega,
                                     strict.separation_constant, 4*strict.cf_depth)) < 3e-10


def test_signed_spin_and_azimuthal_reflection_have_identical_positive_frequency_branch():
    negative_m = solver.solve(.7, 2, -2, 0)
    negative_spin = solver.solve(-.7, 2, 2, 0)
    assert negative_m.converged and negative_spin.converged
    assert abs(negative_m.omega-negative_spin.omega) < 2e-13
    assert abs(negative_m.separation_constant-negative_spin.separation_constant) < 3e-13


def test_near_extremal_corotating_overtones_remain_distinct():
    roots = [solver.solve(.99999, 2, 2, n) for n in range(3)]
    assert all(result.converged for result in roots)
    damping = [-result.omega.imag for result in roots]
    assert 0 < damping[0] < damping[1] < damping[2]
    assert min(np.diff(damping)) > 1000*max(
        result.metadata["estimated_frequency_error"] for result in roots)
    for n, result in enumerate(roots):
        refined = solver.solve(.99999, 2, 2, n, omega_seed=result.omega, strict=True)
        assert refined.converged
        assert abs(refined.omega-result.omega) < 1e-12


@pytest.mark.parametrize("m,n", [(-2, 0), (1, 1)])
def test_damped_endpoint_uses_extended_arithmetic_and_refines_consistently(m, n):
    pytest.importorskip("gmpy2")
    # The generator separately certifies the full spin path in both
    # directions. This test isolates endpoint arithmetic with a nearby,
    # freshly solved branch predictor to keep the regression suite bounded.
    predictor = solver.solve(.999, 2, m, n)
    ordinary = solver.solve(.99999, 2, m, n, omega_seed=predictor.omega,
                            angular_seed=predictor.separation_constant)
    refined = solver.solve(.99999, 2, m, n, omega_seed=ordinary.omega, strict=True)
    assert ordinary.converged and refined.converged
    assert ordinary.metadata["arithmetic_bits"] == 113
    assert refined.metadata["arithmetic_bits"] == 160
    assert abs(refined.omega-ordinary.omega) < 2e-13
    assert refined.angular_error < 1e-24
    assert refined.metadata["estimated_frequency_error"] < 1e-13


@pytest.mark.parametrize("a,ell,m,n", [(1., 2, 2, 0), (-1., 2, 2, 0),
    (float("nan"), 2, 2, 0), (.5, 1, 1, 0), (.5, 2, 3, 0), (.5, 2, 2, 3),
    (.5, 2.5, 2, 0)])
def test_invalid_eigenproblem_parameters_are_rejected(a, ell, m, n):
    with pytest.raises(ValueError):
        solver.solve(a, ell, m, n)


def test_nonfinite_initial_seed_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        solver.solve(.5, 2, 2, 0, omega_seed=complex(float("nan"), -.1))

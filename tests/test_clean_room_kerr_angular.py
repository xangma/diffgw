"""Equation, normalization, and gradient checks for the angular solver."""

import math
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from torchwave.approximants.kerr_angular import (
    spheroidal_coefficients,
    spin_weighted_spherical_harmonic,
    spin_weighted_spheroidal_harmonic,
)


@pytest.mark.parametrize("ell,m", [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (2, -2)])
def test_zero_oblateness_recovers_spherical_limit(ell, m):
    a, coefficients = spheroidal_coefficients(ell, m, torch.tensor(0j, dtype=torch.complex128))
    expected = torch.zeros_like(coefficients)
    expected[ell - max(2, abs(m))] = 1
    torch.testing.assert_close(a, a.new_tensor(ell * (ell + 1) - 2), atol=0, rtol=0)
    torch.testing.assert_close(coefficients, expected, atol=0, rtol=0)


def test_spherical_explicit_quadrupoles():
    theta = torch.linspace(0, math.pi, 41, dtype=torch.float64)
    c = torch.cos(theta)
    expected22 = math.sqrt(5 / (64 * math.pi)) * (1 + c).square()
    expected21 = math.sqrt(5 / (16 * math.pi)) * torch.sin(theta) * (1 + c)
    for m, expected in [(2, expected22), (1, expected21)]:
        torch.testing.assert_close(spin_weighted_spherical_harmonic(2, m, theta).real, expected, atol=4e-16, rtol=2e-15)


@pytest.mark.parametrize("ell,m,c", [(2, 2, .42-.055j), (3, 2, .8-.14j), (4, 4, 1.6-.02j), (2, -2, -.5+.07j)])
def test_angular_teukolsky_differential_equation(ell, m, c):
    # This evaluates the differential equation itself, rather than applying
    # the same spectral matrix used by the solver.
    x = torch.linspace(-.8, .8, 13, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(c, dtype=torch.complex128)
    a, _ = spheroidal_coefficients(ell, m, c)
    value = spin_weighted_spheroidal_harmonic(ell, m, c, torch.acos(x))
    def derivative(z):
        return torch.complex(
            torch.autograd.grad(z.real.sum(), x, create_graph=True, retain_graph=True)[0],
            torch.autograd.grad(z.imag.sum(), x, create_graph=True, retain_graph=True)[0],
        )
    first = derivative(value)
    second = derivative(first)
    residual = (1-x.square()) * second - 2*x*first + (
        c.square()*x.square() + 4*c*x - 2 + a - (m-2*x).square()/(1-x.square())
    ) * value
    assert residual.abs().max() < 2e-11


def test_sphere_normalization_and_azimuth():
    nodes, weights = np.polynomial.legendre.leggauss(80)
    theta = torch.acos(torch.from_numpy(nodes))
    value = spin_weighted_spheroidal_harmonic(3, 2, .7-.11j, theta)
    integral = 2 * math.pi * torch.sum(torch.from_numpy(weights) * value.abs().square())
    torch.testing.assert_close(integral, integral.new_tensor(1), rtol=2e-13, atol=2e-13)
    shifted = spin_weighted_spheroidal_harmonic(3, 2, .7-.11j, theta, .37)
    torch.testing.assert_close(shifted, value * np.exp(2j*.37), rtol=2e-14, atol=2e-14)


def test_small_oblateness_perturbation():
    # First-order perturbation of Eq. (51): A' = 2s <x> = -8/3.
    c = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    a, coefficients = spheroidal_coefficients(2, 2, c)
    derivative = torch.autograd.grad(a.real, c, retain_graph=True)[0]
    torch.testing.assert_close(derivative, c.new_tensor(-8/3), rtol=2e-14, atol=2e-14)
    derivative_c3 = torch.autograd.grad(coefficients[1].real, c)[0]
    expected = 4.0 * math.sqrt(5/63) / 6.0
    torch.testing.assert_close(derivative_c3, c.new_tensor(expected), rtol=2e-14, atol=2e-14)


def test_batched_complex_gradients_and_resolution():
    c = torch.tensor([.4-.07j, .82-.12j], dtype=torch.complex128, requires_grad=True)
    theta = torch.tensor([.8, 1.3], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda c, t: spin_weighted_spheroidal_harmonic(3, 2, c, t), (c, theta), atol=2e-7, rtol=2e-5)
    value = spin_weighted_spheroidal_harmonic(3, 2, c, theta)
    finer = spin_weighted_spheroidal_harmonic(3, 2, c, theta, ell_max=22)
    torch.testing.assert_close(value, finer, rtol=2e-12, atol=2e-12)
    for i in range(2):
        scalar = spin_weighted_spheroidal_harmonic(3, 2, c[i], theta[i])
        torch.testing.assert_close(value[i], scalar, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("c", [2.1, complex(float("nan"), 0)])
def test_rejects_unvalidated_oblateness(c):
    with pytest.raises(ValueError, match="finite"):
        spheroidal_coefficients(2, 2, c)


def test_fresh_author_eigenvalues_and_mixing_coefficients():
    external = os.environ.get("TORCHWAVE_EXTERNAL_VALIDATION_DIR")
    if not external:
        pytest.skip("Cook-derived validation data require explicit opt-in")
    fixture = json.loads((Path(external) / "kerr_angular_author_samples.json").read_text())
    for case in fixture["cases"]:
        c = case["spin"] * complex(*case["omega"])
        a, coefficients = spheroidal_coefficients(case["ell"], case["m"], c)
        expected = coefficients.new_tensor([complex(*z) for z in case["C"]])
        torch.testing.assert_close(a, a.new_tensor(complex(*case["A"])), atol=5e-13, rtol=0)
        torch.testing.assert_close(coefficients[:len(expected)], expected, atol=3e-14, rtol=0)

"""Independent equation checks for the Kerr angular solver, without QNM data."""

import math

import numpy as np
import pytest
import torch

from torchwave.approximants.kerr_angular import (
    _angular_matrix,
    spheroidal_coefficients,
    spin_weighted_spherical_harmonic,
    spin_weighted_spheroidal_harmonic,
    leaver_frobenius_coefficients,
    leaver_frobenius_harmonic,
)


MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (2, -2)]


@pytest.mark.parametrize("c", [0.0, 0.3, 0.7, 0.95])
def test_leaver_series_spherical_limit_and_truncation(c):
    theta = torch.linspace(0.03, math.pi - 0.03, 121, dtype=torch.float64)
    value = leaver_frobenius_harmonic(2, torch.tensor(c, dtype=torch.complex128), theta,
                                       n_terms=48, quadrature_order=256)
    if c == 0.0:
        expected = spin_weighted_spherical_harmonic(2, 2, theta)
        torch.testing.assert_close(value, expected, rtol=2e-13, atol=2e-13)
    coarse = leaver_frobenius_harmonic(2, torch.tensor(c, dtype=torch.complex128), theta,
                                       n_terms=32, quadrature_order=128)
    torch.testing.assert_close(value, coarse, rtol=3e-6, atol=3e-6)


def test_leaver_series_matches_independent_spherical_solver():
    c = torch.tensor([0.3, 0.7, 0.95], dtype=torch.complex128)
    leaver_a, _ = leaver_frobenius_coefficients(2, c, n_terms=48)
    spectral_a, _ = spheroidal_coefficients(2, 2, c, ell_max=18)
    torch.testing.assert_close(leaver_a, spectral_a, rtol=3e-10, atol=3e-10)


@pytest.mark.parametrize("c", [0.3 - 0.1j, 0.7 - 0.2j, 0.95 - 0.25j])
def test_leaver_complex_harmonic_fixed_phase_and_norm(c):
    theta = torch.linspace(0.04, math.pi - 0.04, 97, dtype=torch.float64)
    cc = torch.tensor(c, dtype=torch.complex128)
    value = leaver_frobenius_harmonic(2, cc, theta, n_terms=56, quadrature_order=512)
    reference = spin_weighted_spheroidal_harmonic(2, 2, cc, theta, ell_max=18)
    # The projection-defined phase is fixed before this comparison; no phase
    # optimization is performed. The remaining error is finite recurrence
    # truncation and native spectral tail at complex c.
    torch.testing.assert_close(value, reference, rtol=1e-10, atol=1e-10)
    nodes, weights = np.polynomial.legendre.leggauss(512)
    q = leaver_frobenius_harmonic(2, cc, torch.acos(torch.from_numpy(nodes)),
                                   n_terms=48, quadrature_order=512)
    norm = 2 * math.pi * torch.from_numpy(weights) @ q.abs().square()
    torch.testing.assert_close(norm, torch.tensor(1.0, dtype=torch.float64), rtol=2e-12, atol=2e-12)


def test_leaver_broadcasting_and_finite_difference_derivative():
    theta = torch.tensor([0.4, 1.1, 2.0], dtype=torch.float64)
    c = torch.tensor([0.3, 0.7, 0.95], dtype=torch.complex128)
    value = leaver_frobenius_harmonic(2, c, theta, n_terms=40)
    assert value.shape == theta.shape
    grid = leaver_frobenius_harmonic(2, c[:, None], theta, n_terms=40)
    assert grid.shape == (3, 3)
    h = 1e-5
    cp = torch.tensor(0.7 + h - 0.2j, dtype=torch.complex128)
    cm = torch.tensor(0.7 - h - 0.2j, dtype=torch.complex128)
    fd = (leaver_frobenius_coefficients(2, cp, n_terms=40)[0] -
          leaver_frobenius_coefficients(2, cm, n_terms=40)[0]) / (2 * h)
    cvar = torch.tensor(0.7 - 0.2j, dtype=torch.complex128, requires_grad=True)
    av = leaver_frobenius_coefficients(2, cvar, n_terms=40)[0]
    grad = torch.autograd.grad(av.real, cvar, create_graph=False)[0]
    torch.testing.assert_close(grad.real, fd.real, rtol=2e-4, atol=2e-5)


def test_harmonic_solves_original_c_shape_once(monkeypatch):
    import torchwave.approximants.kerr_angular as angular
    seen = []
    original = angular.leaver_frobenius_coefficients
    def spy(ell, c, **kwargs):
        seen.append(tuple(c.shape))
        return original(ell, c, **kwargs)
    monkeypatch.setattr(angular, "leaver_frobenius_coefficients", spy)
    angular.leaver_frobenius_harmonic(2, torch.tensor(0.7-0.2j), torch.ones(1, 3), n_terms=24)
    angular.leaver_frobenius_harmonic(2, torch.tensor([.3, .7])[:, None], torch.ones(1, 3), n_terms=24)
    assert seen == [(), (2, 1)]


def test_harmonic_value_gradients_with_phi_grid():
    c = torch.tensor(0.7 - 0.2j, dtype=torch.complex128, requires_grad=True)
    theta = torch.tensor([[.4, 1.1, 2.]], dtype=torch.float64)
    phi = torch.tensor([[0., .2, .4]], dtype=torch.float64, requires_grad=True)
    value = leaver_frobenius_harmonic(2, c, theta, phi, n_terms=40)
    dc, dphi = torch.autograd.grad(value.real.sum(), (c, phi))
    assert value.shape == phi.shape
    assert torch.isfinite(dc) and torch.isfinite(dphi).all()


@pytest.mark.parametrize("ell", [2, 3])
def test_leaver_a0_gauge_is_normalized_and_theta_independent(ell):
    c = torch.tensor(.7 - .2j, dtype=torch.complex128)
    theta = torch.tensor([.3, .8, 1.7], dtype=torch.float64)
    raw = leaver_frobenius_harmonic(ell, c, theta, phase_convention="a0_real_positive")
    fixed = leaver_frobenius_harmonic(ell, c, theta)
    ratio = raw / fixed
    torch.testing.assert_close(ratio, ratio[0].expand_as(ratio), rtol=2e-13, atol=2e-13)
    nodes, weights = np.polynomial.legendre.leggauss(256)
    q = leaver_frobenius_harmonic(ell, c, torch.acos(torch.from_numpy(nodes)), phase_convention="a0_real_positive")
    torch.testing.assert_close(2 * math.pi * torch.from_numpy(weights) @ q.abs().square(), torch.tensor(1., dtype=torch.float64), rtol=2e-12, atol=2e-12)
    _, coeff = leaver_frobenius_coefficients(ell, c)
    torch.testing.assert_close(coeff[0].real, torch.tensor(1., dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(coeff[0].imag, torch.tensor(0., dtype=torch.float64), atol=0, rtol=0)


def test_leaver_phase_option_default_and_validation():
    c = torch.tensor(.4 - .1j, dtype=torch.complex128)
    theta = torch.tensor(.9, dtype=torch.float64)
    torch.testing.assert_close(leaver_frobenius_harmonic(2, c, theta), leaver_frobenius_harmonic(2, c, theta, phase_convention="spherical_projection"), rtol=0, atol=0)
    with pytest.raises(ValueError, match="phase_convention"):
        leaver_frobenius_harmonic(2, c, theta, phase_convention="bad")


def test_leaver_a0_gauge_has_finite_gradient():
    c = torch.tensor(.6 - .15j, dtype=torch.complex128, requires_grad=True)
    value = leaver_frobenius_harmonic(2, c, torch.tensor(.7, dtype=torch.float64), phase_convention="a0_real_positive")
    grad = torch.autograd.grad(value.real, c)[0]
    assert torch.isfinite(grad)


def test_leaver_angular_ode_residual():
    x = torch.linspace(-0.8, 0.8, 17, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(0.7 - 0.2j, dtype=torch.complex128)
    y = leaver_frobenius_harmonic(2, c, torch.acos(x), n_terms=56)
    yp = torch.autograd.grad(y.real.sum(), x, create_graph=True)[0] + 1j * torch.autograd.grad(y.imag.sum(), x, create_graph=True)[0]
    ypp = torch.autograd.grad(yp.real.sum(), x, create_graph=True)[0] + 1j * torch.autograd.grad(yp.imag.sum(), x)[0]
    A = leaver_frobenius_coefficients(2, c, n_terms=56)[0]
    residual = (1-x*x)*ypp - 2*x*yp + (c*c*x*x + 4*c*x - 2 + A - (2-2*x)**2/(1-x*x))*y
    assert float(residual.abs().max().detach()) < 3e-6


@pytest.mark.parametrize("m", [-2, 1, 2, 3, 4])
def test_spectral_operator_matches_sphere_quadrature(m):
    # Project Eq. (51) by integration. In particular, this detects the
    # boundary contribution lost by squaring an already-truncated x matrix.
    ell_max = 9
    nodes, weights = np.polynomial.legendre.leggauss(64)
    x, w = torch.from_numpy(nodes), torch.from_numpy(weights)
    degrees = range(max(2, abs(m)), ell_max + 1)
    basis = torch.stack([
        spin_weighted_spherical_harmonic(j, m, torch.acos(x)).real
        for j in degrees
    ])
    moment1 = 2 * math.pi * ((basis * (w*x)) @ basis.T)
    moment2 = 2 * math.pi * ((basis * (w*x*x)) @ basis.T)
    a0 = torch.diag(torch.tensor([j*(j+1)-2 for j in degrees], dtype=x.dtype))
    c = torch.tensor(1.4 - .37j, dtype=torch.complex128)
    projected = a0 - c*c*moment2 - 4*c*moment1
    torch.testing.assert_close(_angular_matrix(m, c, ell_max), projected,
                               rtol=5e-13, atol=5e-13)


@pytest.mark.parametrize("ell,m", MODES)
def test_second_order_spherical_perturbation(ell, m):
    # Rayleigh--Schrodinger perturbation of Eq. (51). There is no imported
    # eigenvalue series: the x moments follow directly from Eq. (52).
    def offdiag(j):
        return math.sqrt((j*j-m*m)*(j*j-4)/(j*j*(4*j*j-1)))

    lower = offdiag(ell)
    upper = offdiag(ell+1)
    diagonal = 2*m/(ell*(ell+1))
    expected_first = -4*diagonal
    expected_second = -2*(lower**2+upper**2+diagonal**2)
    expected_second += 32*(lower**2/(2*ell)-upper**2/(2*(ell+1)))
    c = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    value, _ = spheroidal_coefficients(ell, m, c)
    first = torch.autograd.grad(value.real, c, create_graph=True)[0]
    second = torch.autograd.grad(first, c)[0]
    torch.testing.assert_close(first, c.new_tensor(expected_first), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(second, c.new_tensor(expected_second), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("ell,m", MODES)
def test_complex_symmetric_hellmann_feynman_derivative(ell, m):
    # For this complex symmetric operator the left eigenvector is C^T,
    # without conjugation. This independently checks both complex gradients.
    base = torch.tensor([.6-.2j, 1.3-.35j], dtype=torch.complex128)
    direction = torch.tensor([1.+0j, 0.+1j], dtype=torch.complex128)
    t = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    c = base+t*direction
    value, coefficients = spheroidal_coefficients(ell, m, c)
    ell_max = 18
    a0 = _angular_matrix(m, torch.zeros_like(base), ell_max)
    plus = _angular_matrix(m, torch.ones_like(base), ell_max)
    minus = _angular_matrix(m, -torch.ones_like(base), ell_max)
    linear, quadratic = (plus-minus)/2, (plus+minus)/2-a0
    derivative_matrix = linear+2*base[:, None, None]*quadratic
    expected = (
        coefficients*(derivative_matrix @ coefficients[..., None]).squeeze(-1)
    ).sum(-1)/(coefficients*coefficients).sum(-1)*direction
    observed = torch.complex(
        torch.autograd.grad(value.real.sum(), t, retain_graph=True)[0],
        torch.autograd.grad(value.imag.sum(), t)[0],
    )
    torch.testing.assert_close(observed, expected.sum(), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("ell,m", MODES)
def test_mirror_eigenvalue_and_phase_fixed_coefficients(ell, m):
    # Eqs. (48b,c,e,f), together with the real-positive spherical anchor.
    c = torch.tensor([.3-.5j, 1.5-.1j], dtype=torch.complex128)
    value, coefficients = spheroidal_coefficients(ell, m, c)
    mirror_value, mirror = spheroidal_coefficients(ell, -m, -c.conj())
    degrees = torch.arange(max(2, abs(m)), 19)
    parity = (-1.)**(degrees+ell)
    torch.testing.assert_close(mirror_value, value.conj(), rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(mirror, parity*coefficients.conj(), rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("ell,m", MODES)
def test_eight_step_branch_matches_fine_continuation(ell, m):
    # Exercise an equation-defined domain covering low-overtones' complex
    # oblateness, rather than reading frequency tables to choose examples.
    c = torch.tensor([0j, .2-.9j, .8-.6j, 1.9-.25j, -1.3-.45j],
                     dtype=torch.complex128)
    value, _ = spheroidal_coefficients(ell, m, c)
    previous = torch.full_like(c, ell*(ell+1)-2)
    with torch.no_grad():
        for step in range(1, 65):
            values = torch.linalg.eigvals(_angular_matrix(m, c*(step/64), 22))
            index = (values-previous[..., None]).abs().argmin(-1)
            previous = values.gather(-1, index[..., None]).squeeze(-1)
    torch.testing.assert_close(value, previous, rtol=2e-13, atol=2e-13)

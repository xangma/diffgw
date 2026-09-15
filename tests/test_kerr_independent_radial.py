"""Primary-equation and continuation checks; no imported QNM root fixtures."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch


_SPEC = importlib.util.spec_from_file_location(
    "independently_reviewed_kerr_solver",
    Path(__file__).resolve().parents[1] / "scripts/kerr_leaver_solver.py",
)
solver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(solver)


@pytest.mark.parametrize("a,m,omega", [
    (0., 2, .4-.1j), (.7, 1, .6-.2j), (-.8, 2, .3-.15j),
    (.99999, 4, 1.99-.003j),
])
def test_radial_coefficients_follow_direct_teukolsky_substitution(a, m, omega):
    # Cook Eq. (5a), with R=F(r)Y(y), y=(r-r+)/(r-r-). The logarithmic
    # derivatives below evaluate the original ODE, without using Eq. (27).
    s = -2
    b = math.sqrt((1-a)*(1+a))
    rp, rm = 1+b, 1-b
    sp = (2*omega*rp-m*a)/(2*b)
    sm = (2*omega*rm-m*a)/(2*b)
    xi, eta, alpha = -s-1j*sp, -s+1j*sm, 1-4j*omega
    separation = 6.3+.27j  # Arbitrary A: no eigenvalue is needed for this identity.
    d0, d1, d2, d3, d4, *_ = solver._radial_parameters(a, m, omega, separation)
    for radius in rp+np.array([.2, 1., 4., 12.]):
        delta = (radius-rp)*(radius-rm)
        delta_first = 2*(radius-1)
        y = (radius-rp)/(radius-rm)
        y_first = 2*b/(radius-rm)**2
        y_second = -4*b/(radius-rm)**3
        q = xi/(radius-rp)+(eta-alpha)/(radius-rm)+1j*omega
        q_first = -xi/(radius-rp)**2-(eta-alpha)/(radius-rm)**2
        k = (radius*radius+a*a)*omega-m*a
        radial_lambda = separation+a*a*omega*omega-2*m*a*omega
        potential = (k*k-2j*s*(radius-1)*k)/delta+4j*s*omega*radius-radial_lambda
        observed = np.array([
            delta*y_first*y_first,
            delta*(2*q*y_first+y_second)+(s+1)*delta_first*y_first,
            delta*(q_first+q*q)+(s+1)*delta_first*q+potential,
        ])
        expected = np.array([y*(1-y)**2, d0+d1*y+d2*y*y, d3+d4*y])
        np.testing.assert_allclose(observed, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("ell,n", [(2., 0), (3.7, 1), (20., 2)])
def test_wkb_seed_uses_tortoise_potential_curvature(ell, n):
    # Schutz--Will Eqs. (6)--(9), with the complex conjugation required by
    # this solver's exp(-i omega t) convention. Obtain curvature by autograd.
    degree = ell*(ell+1)
    radius = (3*(degree+3)+math.sqrt(9*(degree+3)**2-96*degree))/(2*degree)
    r = torch.tensor(radius, dtype=torch.float64, requires_grad=True)
    potential = (1-2/r)*(degree/r.square()-6/r.pow(3))
    first = torch.autograd.grad(potential, r, create_graph=True)[0]
    second = torch.autograd.grad(first, r)[0]
    curvature = (1-2/r).square()*second
    expected = complex(potential.detach(), -(n+.5)*torch.sqrt(-2*curvature).detach())
    assert abs(float(first.detach())) < 3e-14
    assert abs(solver._wkb_seed(ell, n)**2-expected) < 2e-14


@pytest.mark.parametrize("n", [0, 1, 2])
def test_schwarzschild_label_survives_independent_finer_wkb_homotopy(n):
    # Change both the asymptotic start and continuation spacing. The target
    # roots must agree; merely solving a different n inversion is insufficient.
    seed = solver._wkb_seed(16., n)
    independent = {}
    for ell in np.linspace(16., 2., 281):
        separation = ell*(ell+1)-2
        result = solver._correct(0., 2, 0, n, seed, separation, fixed_a=separation)
        assert result.converged
        assert abs(result.omega-seed) < .07
        seed = result.omega
        if min(abs(ell-j) for j in (2, 3, 4)) < 1e-12:
            independent[round(ell)] = result.omega
    production = solver._schwarzschild_roots(n)
    for ell, omega in independent.items():
        assert abs(omega-production[ell].omega) < 2e-12


@pytest.mark.parametrize("ell,n", [(ell, n) for ell in (2, 3, 4) for n in range(3)])
def test_fresh_schwarzschild_roots_satisfy_other_inversions(ell, n):
    result = solver._schwarzschild_roots(n)[ell]
    omega, separation = result.omega, complex(ell*(ell+1)-2)
    errors = []
    for inversion in range(5):
        value = solver.radial_residual(0., 0, inversion, omega, separation, depth=2048)
        step = 1e-6
        derivative = (
            solver.radial_residual(0., 0, inversion, omega+step, separation, depth=2048)
            -solver.radial_residual(0., 0, inversion, omega-step, separation, depth=2048)
        )/(2*step)
        assert abs(derivative) > 1e-5
        errors.append(abs(value/derivative))
    assert max(errors) < 2e-12


def test_schwarzschild_overtone_roots_are_distinct_and_damping_ordered():
    for ell in (2, 3, 4):
        roots = [solver._schwarzschild_roots(n)[ell].omega for n in range(3)]
        assert all(root.real > 0 and root.imag < 0 for root in roots)
        assert roots[0].imag > roots[1].imag > roots[2].imag
        assert min(abs(a-b) for i, a in enumerate(roots) for b in roots[i+1:]) > .1

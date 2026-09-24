"""Primary-equation checks; no waveform implementation or oracle is imported."""

import math

import numpy as np
import pytest
import torch
from numpy.polynomial import polynomial as poly

from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.constants import EULER_GAMMA


def tensor(value, **kwargs):
    if isinstance(value, torch.Tensor):
        value = value.detach().clone()
        return value.to(dtype=torch.float64).requires_grad_(kwargs.get("requires_grad", False))
    return torch.tensor(value, dtype=torch.float64, **kwargs)


def model(nu, chi1=0.0, chi2=0.0):
    return SEOBNRv4Hamiltonian(tensor(nu), tensor(chi1), tensor(chi2))


@pytest.mark.parametrize("nu,chi1,chi2", [(0.25, 0., 0.), (0.16, .7, -.3), (.02, -.8, .5)])
def test_radial_potential_recovers_published_4pn_series(nu, chi1, chi2):
    """Expand log as a formal power series; test the independent PN target.

    1202.0790 (7c), 1608.01907 (A2c), 1611.03703 (2.3).
    log(u) is held symbolic by evaluating two arbitrary values.
    """
    ham = model(nu, chi1, chi2)
    r = tensor(10.)
    coeff = [float(x) for x in ham.radial_coefficients(r)]
    a = float(ham.spin_combinations(r)[0])
    k = float(ham.calibration_parameters(r)[0])
    for log_u in [0., -3.7]:
        d0, *rest = coeff
        d1, d2, d3, d4, d5, d5_log = rest
        x = np.array([0., d1, d2, d3, d4, d5 + d5_log*log_u])
        log_series = sum((-1.)**(j+1)/j * poly.polypow(x, j)[:6] for j in range(1, 6))
        log_series[0] = d0 + 1.0/nu
        expanded = poly.polymul([1/(nu*k-1)**2, 2/(nu*k-1), a*a], nu*log_series)[:6]
        a5 = (2275*math.pi**2/512 + 128*EULER_GAMMA/5 - 4237/60
              + 256*math.log(2)/5 + (41*math.pi**2/32 - 221/6)*nu + 64*log_u/5)
        expected = [1., -2., a*a, 2*nu, (94/3 - 41*math.pi**2/32)*nu, a5*nu]
        np.testing.assert_allclose(expanded, expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("a", [-.9, 0., .9])
def test_exact_kerr_circular_energy_and_orbital_frequency(a):
    """Analytic equatorial Kerr geodesic constants, including retrograde spin."""
    ham = model(0., a)
    r = tensor([7., 12., 25.])
    denominator = r.pow(.75)*torch.sqrt(r.pow(1.5) - 3*r.sqrt() + 2*a)
    energy = (r.pow(1.5) - 2*r.sqrt() + a)/denominator
    angular_momentum = (r.square() - 2*a*r.sqrt() + a*a)/denominator
    metric = ham.deformed_kerr_potentials(r)
    torch.testing.assert_close(metric.delta_t, r.square()-2*r+a*a, atol=3e-13, rtol=1e-14)
    torch.testing.assert_close(metric.delta_r, metric.delta_t, atol=1e-13, rtol=1e-14)
    torch.testing.assert_close(ham.effective_hamiltonian(r, torch.zeros_like(r), angular_momentum), energy,
                               atol=5e-15, rtol=1e-14)
    dr, omega, dpr = ham.equations_of_motion_conservative(r, torch.zeros_like(r), angular_momentum)
    torch.testing.assert_close(omega, 1/(r.pow(1.5)+a), atol=1e-15, rtol=1e-13)
    torch.testing.assert_close(dr, torch.zeros_like(r), atol=1e-14, rtol=0.)
    torch.testing.assert_close(dpr, torch.zeros_like(r), atol=1e-14, rtol=0.)


def test_nonspinning_hamiltonian_and_tortoise_quartic_limit():
    ham = model(.24)
    r, pr, j = tensor([6., 9., 20.]), tensor([-.15, .1, -.03]), tensor([3., 4., 5.])
    metric = ham.deformed_kerr_potentials(r)
    a = metric.delta_t/r.square()
    expected = torch.sqrt(pr.square() + a*(1+j.square()/r.square()
                         + 2*.24*(4-3*.24)*pr.pow(4)/r.square()))
    torch.testing.assert_close(ham.effective_hamiltonian(r, pr, j), expected, atol=1e-15, rtol=1e-14)


def test_leading_spin_orbit_coupling_in_weak_field():
    """0912.3517 (5.53), (5.56): L*(2*S_Kerr+3*sigma_*/2)/r^3."""
    chi1 = tensor(0., requires_grad=True)
    ham = SEOBNRv4Hamiltonian(tensor(.16), chi1, tensor(0.))
    # At much larger radii, subtractive cancellation in the fitted potential's
    # spin derivative can dominate this tiny O(r^-5/2) coupling in float64.
    r, pr, j = tensor(1e4), tensor(.00316), tensor(94.9)
    derivative = torch.autograd.grad(ham.effective_hamiltonian(r, pr, j), chi1)[0]
    expected = j/r.pow(3)*(2*ham.m1.square() + 1.5*ham.nu)
    torch.testing.assert_close(derivative, expected, atol=0., rtol=3e-5)


def test_potential_radial_derivative_and_hamiltonian_gradients():
    nu, c1, c2 = tensor(.19, requires_grad=True), tensor(.6, requires_grad=True), tensor(-.4, requires_grad=True)
    r, pr, j = tensor(9., requires_grad=True), tensor(-.03, requires_grad=True), tensor(3.6, requires_grad=True)
    ham = SEOBNRv4Hamiltonian(nu, c1, c2)
    metric = ham.deformed_kerr_potentials(r)
    derivative = torch.autograd.grad(metric.delta_t, r, create_graph=True)[0]
    torch.testing.assert_close(derivative, metric.delta_t_prime, atol=5e-14, rtol=1e-14)
    assert torch.autograd.gradcheck(lambda n, a, b, x, p, l:
        SEOBNRv4Hamiltonian(n, a, b).effective_hamiltonian(x, p, l), (nu, c1, c2, r, pr, j),
        atol=2e-6, rtol=2e-5)


def test_batch_scalar_and_inference_mode_agree():
    r, pr, j = tensor([8., 12.]), tensor([-.03, .02]), tensor([3.4, 4.1])
    ham = model([.16, .24], [.7, -.3], [-.2, .6])
    batched = ham.equations_of_motion_conservative(r, pr, j)
    scalar = [model(n, a, b).equations_of_motion_conservative(x, p, l)
              for n, a, b, x, p, l in zip(ham.nu, ham.chi1, ham.chi2, r, pr, j)]
    for actual, expected in zip(batched, zip(*scalar)):
        torch.testing.assert_close(actual, torch.stack(expected), atol=1e-14, rtol=1e-13)
    with torch.inference_mode():
        inference_ham = model([.16, .24], [.7, -.3], [-.2, .6])
        result = inference_ham.equations_of_motion_conservative(r.clone(), pr.clone(), j.clone())
    for actual, expected in zip(result, batched):
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-13)


def test_shared_state_keeps_parameter_batch_forces_independent():
    hamiltonian_class = SEOBNRv4Hamiltonian
    nu, c1, c2 = tensor([.16, .24]), tensor([.7, -.3]), tensor([-.2, .6])
    r, pr, j = tensor(9., requires_grad=True), tensor(-.03), tensor(3.6)
    batched = hamiltonian_class(nu, c1, c2).equations_of_motion_conservative(r, pr, j)
    scalar = [hamiltonian_class(n, a, b).equations_of_motion_conservative(r, pr, j)
              for n, a, b in zip(nu, c1, c2)]
    for actual, expected in zip(batched, zip(*scalar)):
        expected = torch.stack(expected)
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-13)
        actual_derivative = torch.autograd.grad(actual.sum(), r, retain_graph=True)[0]
        expected_derivative = torch.autograd.grad(expected.sum(), r, retain_graph=True)[0]
        torch.testing.assert_close(actual_derivative, expected_derivative, atol=1e-14, rtol=1e-13)

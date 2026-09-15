"""Equation-derived hybrid-energy checks; no calibrated model outputs."""

import cmath
from decimal import Decimal, localcontext
import math

import pytest
from scipy.optimize import brentq
import torch

from torchwave.approximants.phenom_meco import (
    hybrid_binding_energy,
    hybrid_meco_frequency,
    hybrid_meco_velocity,
)


def real(value):
    return torch.tensor(value, dtype=torch.float64)


def equation_energy(v, m1, m2, a, b, pn_order=3.5):
    """Unsimplified E_PN/eta - truncated Kerr + exact Kerr.

    Cabero v3 Eqs. (5), (6), (23)--(26), with black-hole quadrupoles.
    Complex v supports an independent complex-step energy derivative.
    """
    x, y = m1 / (m1 + m2), m2 / (m1 + m2)
    eta, delta = x*y, x-y
    spin, sigma = x*x*a + y*y*b, y*b-x*a
    chi = x*a + y*b
    nonspin = (1 - (3/4+eta/12)*v**2
               - (27/8-19*eta/8+eta**2/24)*v**4
               - (675/64-(34445/576-205*math.pi**2/96)*eta
                  + 155*eta**2/96+35*eta**3/5184)*v**6)
    so = ((14*spin/3+2*delta*sigma)*v**3
          + ((11-61*eta/9)*spin+(3-10*eta/3)*delta*sigma)*v**5)
    if pn_order == 3.5:
        so += ((135/4-367*eta/4+29*eta**2/12)*spin
               + (27/4-39*eta+5*eta**2/4)*delta*sigma)*v**7
    ss2 = -(2*x*y*a*b+x*x*a*a+y*y*b*b)*v**4
    ss3 = -5*eta*a*b*(3+eta)/9
    for fraction, body_spin in ((x, a), (y, b)):
        ss3 += fraction**2 * body_spin**2 * (
            5*(9-3*eta-5*fraction**2)/9
            - 5*(3+3*eta+4*fraction**2)/6)
    pn = -v**2*(nonspin+so+ss2+ss3*v**6)/2
    truncated = (1-3*v**2/4+8*chi*v**3/3-(27/8+chi**2)*v**4
                 + 8*chi*v**5-(675/64+65*chi**2/18)*v**6)
    if pn_order == 3.5:
        truncated += 27*chi*v**7
    w = v**2/(1-chi*v**3)**(2/3)
    exact = (1-2*w+chi*w**1.5)/cmath.sqrt(1-3*w+2*chi*w**1.5)-1
    return pn + v**2*truncated/2 + exact


def reference_velocity(*parameters, pn_order=3.5):
    # Search using complex-step derivatives of the unsimplified paper energy.
    m1, m2, a, b = parameters
    chi = (m1*a+m2*b)/(m1+m2)
    photon_r = 2*(1+math.cos(2*math.acos(-chi)/3))
    upper = (1-1e-5)/(photon_r**1.5+chi)**(1/3)

    def slope(v):
        return equation_energy(v+1e-25j, *parameters, pn_order).imag/1e-25

    low = 1e-4*upper
    for index in range(1, 4097):
        high = upper*index/4096
        if slope(high) >= 0:
            return brentq(slope, low, high, xtol=5e-15, rtol=1e-14)
        low = high
    raise AssertionError("reference energy has no bracketed minimum")


def decimal_isco_velocity(spin):
    # Independent radial marginal-stability polynomial, with 70-digit roots.
    with localcontext() as context:
        context.prec = 70
        a = Decimal.from_float(spin)
        low, high = Decimal(1), Decimal(9)
        for _ in range(235):
            r = (low+high)/2
            if r*r-6*r+8*a*r.sqrt()-3*a*a > 0:
                high = r
            else:
                low = r
        r = (low+high)/2
        return float((r*r.sqrt()+a)**(-Decimal(1)/3))


@pytest.mark.parametrize("order", [3, 3.5])
def test_energy_matches_full_pn_minus_kerr_series(order):
    configurations = [(1., 1., 0., 0.), (3., 1., .8, -.7),
                      (1., 8., -.5, .9), (1e5, 1., .7, -.8)]
    for parameters in configurations:
        v = real([.08, .17, .29, .38])
        expected = real([equation_energy(z.item(), *parameters, order).real for z in v])
        torch.testing.assert_close(hybrid_binding_energy(v, *parameters, pn_order=order),
                                   expected, rtol=1e-13, atol=3e-16)
        complex_step = real([equation_energy(z.item()+1e-25j, *parameters, order).imag/1e-25
                             for z in v])
        torch.testing.assert_close(
            hybrid_binding_energy(v, *parameters, pn_order=order, derivative=1),
            complex_step, rtol=1e-12, atol=3e-15)


def test_newtonian_energy_survives_small_velocity_cancellation():
    v = real([1e-12, 1e-10, 1e-8])
    result = hybrid_binding_energy(v, 3., 1., .7, -.2)
    torch.testing.assert_close(result/(-v*v/2), torch.ones_like(v), rtol=5e-15, atol=0)


@pytest.mark.parametrize("order", [3, 3.5])
def test_analytic_velocity_derivatives_match_autograd(order):
    v = real([.2, .35, .5]).requires_grad_()
    energy = hybrid_binding_energy(v, 3., 1., .8, .4, pn_order=order)
    first = torch.autograd.grad(energy.sum(), v, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), v)[0]
    for derivative, expected in ((1, first), (2, second)):
        torch.testing.assert_close(
            hybrid_binding_energy(v, 3., 1., .8, .4, pn_order=order, derivative=derivative),
            expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("order", [3, 3.5])
def test_stationary_points_match_independent_complex_step_search(order):
    configurations = [(1., 1., -.9, .3), (1., 1., .8, .8),
                      (2., 1., .5, -.6), (5., 1., -.8, .9),
                      (100., 1., .9999, .9999), (10000., 1., -.9, .4)]
    for parameters in configurations:
        expected = reference_velocity(*parameters, pn_order=order)
        actual = hybrid_meco_velocity(*parameters, pn_order=order)
        torch.testing.assert_close(actual, real(expected), rtol=2e-12, atol=1e-13)


@pytest.mark.parametrize("order", [3, 3.5])
def test_test_particle_limit_is_kerr_isco_and_ignores_zero_mass_spin(order):
    spins = [-1., -.8, 0., .8, .99, .9999]
    actual = hybrid_meco_velocity(1., 0., real(spins), -.7, pn_order=order)
    expected = real([decimal_isco_velocity(spin) for spin in spins])
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-14)
    torch.testing.assert_close(actual, hybrid_meco_velocity(0., 1., .9, real(spins), pn_order=order),
                               rtol=0, atol=0)
    frequency = hybrid_meco_frequency(1., 0., 0., pn_order=order)
    torch.testing.assert_close(frequency, real(1/(6**1.5*math.pi)), rtol=3e-15, atol=0)


def test_finite_mass_correction_converges_to_test_particle_limit():
    spin = real([-.8, 0., .8])
    target = hybrid_meco_velocity(1., 0., spin, -.2)
    errors = [(hybrid_meco_velocity(1., epsilon, spin, -.2)-target).abs()
              for epsilon in (1e-3, 1e-4, 1e-5)]
    assert torch.all(errors[2] < errors[1]/8)
    assert torch.all(errors[1] < errors[0]/8)


@pytest.mark.parametrize("order", [3, 3.5])
def test_domain_minima_and_scan_convergence(order):
    # 1,210 configurations, independent component spins, q=1..10^12.
    mass1 = real([1., 1.2, 2., 4., 8., 16., 100., 1e3, 1e6, 1e12])[:, None, None]
    spins = real([-1., -.9, -.7, -.3, 0., .3, .7, .9, .99, .999, .9999])
    s1, s2 = spins[None, :, None], spins[None, None, :]
    coarse = hybrid_meco_velocity(mass1, 1., s1, s2, pn_order=order,
                                  scan_points=64, bisection_steps=24)
    fine = hybrid_meco_velocity(mass1, 1., s1, s2, pn_order=order,
                                scan_points=512, bisection_steps=72)
    torch.testing.assert_close(coarse, fine, rtol=5e-13, atol=3e-14)
    assert torch.all((fine > .3) & (fine < .8))
    curvature = hybrid_binding_energy(fine, mass1, 1., s1, s2, pn_order=order, derivative=2)
    assert torch.all(curvature > 0)
    for sign in (-1, 1):
        adjacent = fine + sign*1e-6
        slope = hybrid_binding_energy(adjacent, mass1, 1., s1, s2, pn_order=order, derivative=1)
        assert torch.all(sign*slope > 0)


def test_batching_exchange_symmetry_and_mass_scale():
    m1, m2, a, b = real([[1.], [3.]]), real([1., 2., 5.]), real(.6), real([-.8, 0., .4])
    baseline = hybrid_meco_frequency(m1, m2, a, b)
    assert baseline.shape == (2, 3)
    torch.testing.assert_close(baseline, hybrid_meco_frequency(m2, m1, b, a), rtol=0, atol=0)
    for scale in (1e-250, 7., 1e250):
        torch.testing.assert_close(baseline, hybrid_meco_frequency(scale*m1, scale*m2, a, b),
                                   rtol=3e-14, atol=0)


@pytest.mark.parametrize("order", [3, 3.5])
def test_parameter_first_and_second_derivatives(order):
    arguments = tuple(real(value).requires_grad_()
                      for value in ([3., 1., 1.], [1., 1., 4.], [.6, -.2, .9], [-.2, .5, -.7]))
    function = lambda *parameters: hybrid_meco_frequency(*parameters, pn_order=order)
    assert torch.autograd.gradcheck(function, arguments, atol=2e-8, rtol=2e-5)
    assert torch.autograd.gradgradcheck(function, arguments, atol=2e-8, rtol=2e-5)


def test_root_gradient_satisfies_implicit_stationarity_equation():
    arguments = tuple(real(value).requires_grad_() for value in (3., 1., .8, -.4))
    velocity = hybrid_meco_velocity(*arguments)
    gradients = torch.autograd.grad(velocity, arguments, retain_graph=True)
    slope = hybrid_binding_energy(velocity.detach(), *arguments, derivative=1)
    partials = torch.autograd.grad(slope, arguments)
    curvature = hybrid_binding_energy(velocity.detach(), *arguments, derivative=2)
    for gradient, partial in zip(gradients, partials):
        torch.testing.assert_close(gradient, -partial/curvature, rtol=3e-13, atol=3e-14)


def test_dtype_promotion_no_grad_and_inference_mode():
    a = torch.tensor([0., .9], dtype=torch.float32, requires_grad=True)
    result = hybrid_meco_frequency(3., 1., a, -.2)
    assert result.dtype == torch.float32
    assert torch.all(torch.isfinite(torch.autograd.grad(result.sum(), a)[0]))
    promoted = hybrid_meco_frequency(real(3.), 1., a, -.2)
    assert promoted.dtype == torch.float64
    torch.testing.assert_close(result.double(), promoted, rtol=3e-7, atol=0)
    for context in (torch.no_grad(), torch.inference_mode()):
        with context:
            torch.testing.assert_close(hybrid_meco_frequency(3., 1., a, -.2), result)
    assert hybrid_meco_frequency(3., 1.).dtype == torch.float64


@pytest.mark.parametrize("arguments", [(-1., 2.), (0., 0.), (math.inf, 1.),
                                       (1., 1., 1.01), (1., 1., -1.01),
                                       (1., 1., 1., 1.), (1., 1., math.nan)])
def test_invalid_physical_inputs_are_rejected(arguments):
    with pytest.raises(ValueError):
        hybrid_meco_velocity(*arguments)


@pytest.mark.parametrize("invalid", [True, 1j, torch.tensor(1., dtype=torch.float16)])
def test_invalid_input_types_are_rejected(invalid):
    with pytest.raises(TypeError):
        hybrid_meco_velocity(invalid, 1.)


@pytest.mark.parametrize("kwargs", [{"pn_order": 4}, {"pn_order": True},
                                    {"scan_points": 31}, {"scan_points": 64.5},
                                    {"bisection_steps": 19}])
def test_unsupported_numerical_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        hybrid_meco_velocity(3., 1., **kwargs)


@pytest.mark.parametrize("velocity", [0., -1., 1/math.sqrt(3), .7])
def test_energy_rejects_non_timelike_or_nonpositive_velocities(velocity):
    with pytest.raises(ValueError):
        hybrid_binding_energy(velocity, 1., 1.)


def test_energy_rejects_unresolved_float32_photon_boundary():
    spin = torch.tensor(.9999, dtype=torch.float32)
    radius = 2*(1+torch.cos((2/3)*torch.acos(-spin)))
    photon = (radius.pow(1.5)+spin).pow(-1/3)
    velocity = torch.nextafter(photon, torch.zeros_like(photon))
    with pytest.raises(ValueError, match="precision"):
        hybrid_binding_energy(velocity, 1., 0., spin)

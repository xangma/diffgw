"""Independent checks for the published TEOBResumS orbital potential."""

import math

import torch

from torchwave.approximants.teobresums import TEOBResumSHamiltonian


def t(value):
    return torch.as_tensor(value, dtype=torch.float64)


def test_teob_p1_5_test_mass_branch_is_schwarzschild():
    ham = TEOBResumSHamiltonian(t(0.0), t(0.0), t(0.0), t(0.0), t(0.0))
    r = t([5.0, 10.0, 30.0])
    a, b, xi = ham.point_mass_metric_potentials(r)
    expected_a = 1.0 - 2.0 / r
    torch.testing.assert_close(a, expected_a, rtol=0.0, atol=2e-15)
    torch.testing.assert_close(b, 1.0 / expected_a, rtol=0.0, atol=2e-15)
    torch.testing.assert_close(xi, expected_a, rtol=0.0, atol=2e-15)


def test_teob_p1_5_matches_printed_pn_series_in_weak_field():
    """P^1_5 reproduces the published 5PN-log coefficients at small u."""
    nu = t(0.23)
    ham = TEOBResumSHamiltonian(nu, t(0.0), t(0.0), t(0.0), t(0.0))
    u = t([1.0e-3, 2.0e-3])
    a = ham.point_mass_metric_potentials(1.0 / u)[0]
    gamma_e = 0.577215664901532860606512090082402431
    a4 = 94.0 / 3.0 - 41.0 * math.pi**2 / 32.0
    a5c = (-4237.0 / 60.0 + 2275.0 * math.pi**2 / 512.0
           + 256.0 * math.log(2.0) / 5.0 + 128.0 * gamma_e / 5.0
           + (41.0 * math.pi**2 / 32.0 - 221.0 / 6.0) * nu)
    a6c = 3097.3 * nu**2 - 1330.6 * nu + 81.38
    a5log = 64.0 / 5.0
    a6log = -7004.0 / 105.0 - 144.0 * nu / 5.0
    expected = (1.0 - 2.0 * u + 2.0 * nu * u**3 + nu * a4 * u**4
                + nu * (a5c + a5log * torch.log(u)) * u**5
                + nu * (a6c + a6log * torch.log(u)) * u**6)
    # The first omitted term is O(u^7 log u), below this tolerance here.
    torch.testing.assert_close(a, expected, rtol=0.0, atol=2e-13)


def test_teob_orbital_potential_preserves_coordinate_gradient():
    ham = TEOBResumSHamiltonian(t(0.21), t(0.0), t(0.0), t(0.0), t(0.0))
    r = t(8.0).requires_grad_(True)
    a = ham.point_mass_metric_potentials(r)[0]
    derivative = torch.autograd.grad(a, r)[0]
    assert torch.isfinite(derivative)
    assert derivative > 0.0

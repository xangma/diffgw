"""Independent metric/tortoise checks; these do not validate TEOB calibration.

Damour--Nagar 0704.3550v2 Eqs. (3), (6)-(11), and Nagar et al.
1806.01772v2 Eqs. (6), (10), (23). No target-library imports or outputs.
"""

import pytest
import torch

import torchwave.approximants.teobresums as teob_module
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.teobresums import (
    TEOBResumSRadiationReaction,
    TEOBResumSHamiltonian,
)
from torchwave.constants import MSUN_SEC, PI
from torchwave.waveforms import get_fd_waveform


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def hamiltonian(nu=.23, lambda1=0., lambda2=0.):
    nu, lambda1, lambda2 = map(tensor, (nu, lambda1, lambda2))
    return TEOBResumSHamiltonian(
        nu, torch.zeros_like(nu), torch.zeros_like(nu), lambda1, lambda2)


def test_vacuum_preserves_d_and_tortoise_mass_shell():
    """AB is the printed D; canonical radial momentum gives the mass shell."""
    nu = tensor([.12, .24])
    ham = hamiltonian(nu)
    r = tensor([[6., 10., 20.], [7., 12., 24.]])
    a, b, xi = ham.metric_potentials(r)
    a0, _, _ = ham.point_mass_metric_potentials(r)
    # The explicit P^0_3[D] in the footnote following Eq. (3).
    d = 1 / (1 + 6*nu[:, None]/r**2
             + 2*nu[:, None]*(26 - 3*nu[:, None])/r**3)
    torch.testing.assert_close(a, a0)
    assert torch.all(a > 0)
    torch.testing.assert_close(a*b, d, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(xi**2, a/b, rtol=2e-15, atol=2e-15)

    pr_star, j = tensor([-.2, .3, -.4]), tensor([3.1, 3.6, 4.2])
    pr = pr_star / xi  # p_r dr = p_r* dr*, Eq. (6).
    geodesic_mass_shell = a*(1 + pr**2/b + j**2/r**2)
    z3 = 2*nu[:, None]*(4 - 3*nu[:, None])
    # Eq. (7) uses the stated 3PN truncation for the quartic term.
    quartic = a*z3*pr_star**4/r**2
    actual = ham.effective_hamiltonian(r, pr_star, j).square()
    torch.testing.assert_close(actual, geodesic_mass_shell + quartic,
                               rtol=2e-15, atol=2e-15)


def test_vacuum_conservative_equations_with_numerical_radial_slope():
    """Check all three published Hamilton equations with nonzero p_r*."""
    ham = hamiltonian()
    r, p, j = tensor([7., 10., 16.]), tensor([-.13, .09, -.03]), tensor([2.9, 3.5, 4.2])
    a, _, xi = ham.metric_potentials(r)
    step = 1e-4
    da = (ham.metric_potentials(r + step)[0]
          - ham.metric_potentials(r - step)[0]) / (2*step)
    z3 = 2*ham.nu*(4 - 3*ham.nu)
    h = torch.sqrt(p**2 + a*(1 + (j**2 + z3*p**4)/r**2))
    energy = torch.sqrt(1 + 2*ham.nu*(h - 1))
    expected = (
        xi*(p + 2*a*z3*p**3/r**2)/(energy*h),
        a*j/(r**2*energy*h),
        -xi/(2*energy*h)*(da*(1 + (j**2 + z3*p**4)/r**2)
                          - 2*a*(j**2 + z3*p**4)/r**3),
    )
    actual = ham.equations_of_motion_conservative(r, p, j)
    for got, wanted in zip(actual, expected):
        torch.testing.assert_close(got, wanted, rtol=5e-8, atol=1e-10)


def test_zero_tides_use_calibrated_teob_point_mass_dynamics():
    """Zero tides retain the calibrated TEOBResumS orbital potential."""
    ham = hamiltonian(nu=.23, lambda1=0., lambda2=0.)
    r = tensor([6., 10., 20.])
    a, b, xi = ham.metric_potentials(r)
    a_pm, b_pm, xi_pm = ham.point_mass_metric_potentials(r)
    torch.testing.assert_close(a, a_pm, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(b, b_pm, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(xi, xi_pm, rtol=2e-15, atol=2e-15)

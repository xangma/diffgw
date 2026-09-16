"""Analytic checks for the London et al. clean-room HM maps."""

import torch

from torchwave.approximants.imrphenom_hm_clean import map_phase_frequency, pn_hhat_lm


def test_phenomd_phase_map_joins_are_continuous():
    f0, fr, f22, m = 0.021, 0.0589, 0.0596, 2
    f = torch.tensor([f0, fr], dtype=torch.float64)
    y = map_phase_frequency(f, f0, fr, f22, m)
    assert torch.allclose(y, torch.tensor([2 * f0 / m, f22], dtype=torch.float64))
    eps = 1e-8
    left = map_phase_frequency(torch.tensor([fr - eps]), f0, fr, f22, m)
    right = map_phase_frequency(torch.tensor([fr + eps]), f0, fr, f22, m)
    assert torch.allclose(left, right, atol=2e-8)


def test_phenomd_phase_map_piecewise_derivatives():
    f0, fr, f22, m = 0.021, 0.0589, 0.0596, 3
    eps = 1e-7
    def slope(x):
        y = map_phase_frequency(torch.tensor([x-eps, x+eps], dtype=torch.float64), f0, fr, f22, m)
        return ((y[1] - y[0]) / (2 * eps)).item()
    expected_mid = (f22 - 2 * f0 / m) / (fr - f0)
    assert abs(slope(0.03) - expected_mid) < 1e-7
    assert abs(slope(0.08) - f22 / fr) < 1e-7


def test_pn_hhat_leading_scaling_and_exchange_symmetry():
    v = torch.tensor([0.01, 0.02], dtype=torch.float64)
    eta, delta = 0.22, 0.3
    h33 = pn_hhat_lm((3, 3), v, eta, delta)
    h44 = pn_hhat_lm((4, 4), v, eta, delta)
    assert torch.allclose(h33[1] / h33[0], v[1] / v[0])
    assert torch.allclose(h44[1] / h44[0], (v[1] / v[0]) ** 2)
    # Odd-m leading mass-asymmetry factors vanish for equal masses.
    assert torch.allclose(pn_hhat_lm((2, 1), v, eta, 0.0, chi_s=0.0, chi_a=0.0), torch.zeros_like(v))
    assert torch.allclose(pn_hhat_lm((3, 3), v, eta, 0.0), torch.zeros_like(v))
    assert torch.allclose(pn_hhat_lm((4, 3), v, eta, 0.0), torch.zeros_like(v))
    # Exchange m1<->m2 changes delta and chi_a signs, preserving eta, chi_s.
    a = pn_hhat_lm((4, 3), v, eta, delta, 0.1, 0.2)
    b = pn_hhat_lm((4, 3), v, eta, -delta, 0.1, -0.2)
    assert torch.allclose(a, -b)
    # Equal masses retain the antisymmetric-spin 21 contribution.
    assert torch.any(pn_hhat_lm((2, 1), v, eta, 0.0, chi_a=0.2) != 0)
    # Eq. (12d): the 32 spin term shares the displayed prefactor.
    h32 = pn_hhat_lm((3, 2), v, eta, delta, chi_s=0.2)
    expected = -(1 / 3) * torch.sqrt(torch.tensor(5 / 7, dtype=v.dtype)) * ((1 - 3 * eta) * v**2 + 4 * eta * 0.2 * v**3)
    assert torch.allclose(h32, expected)

from torchwave.approximants.imrphenom_hm_phase_clean import (
    mapped_phase_c1, phenomd_mr_phase, phenomd_mr_phase_derivative,
)


def test_clean_phase_map_is_c1_at_both_joins():
    dtype = torch.float64
    f0, fr, f22 = 0.021, 0.0589, 0.0596
    def carrier(x):
        return 0.7 + 2.3 * x + 4.1 * x**2
    eps = 1e-7
    for join in (f0, fr):
        left = mapped_phase_c1(torch.tensor([join-eps], dtype=dtype), f0, fr, f22, 3, carrier, ell=3)
        right = mapped_phase_c1(torch.tensor([join+eps], dtype=dtype), f0, fr, f22, 3, carrier, ell=3)
        assert torch.allclose(left, right, atol=1e-6)
        yjoin = mapped_phase_c1(torch.tensor([join], dtype=dtype), f0, fr, f22, 3, carrier, ell=3)
        dl = (yjoin - left) / eps
        dr = (right - yjoin) / eps
        assert torch.allclose(dl, dr, atol=2e-5)


def test_clean_mr_phase_derivative_matches_autograd():
    f = torch.tensor([0.08, 0.1], dtype=torch.float64, requires_grad=True)
    alpha = [0.2, 1.1, 0.03, -0.4, 0.7, 1.3]
    y = phenomd_mr_phase(f, 0.22, alpha, 0.06, 0.07, 0.01)
    analytic = phenomd_mr_phase_derivative(f, 0.22, alpha, 0.06, 0.07, 0.01)
    got = torch.autograd.grad(y.sum(), f)[0]
    assert torch.allclose(got, analytic, atol=1e-12, rtol=1e-12)


def test_eq14_delta_derivative_and_c1_attach():
    from torchwave.approximants.imrphenom_hm_phase_clean import (
        phenomd_eq14_phase_delta, phenomd_eq14_phase_delta_derivative,
        phenomd_eq14_phase_delta_c1,
    )
    x = torch.tensor([0.03, 0.08], dtype=torch.float64, requires_grad=True)
    args = (0.22, 0.1, 0.06, 0.07, 0.01, 0.012, 0.7, 1.1)
    y = phenomd_eq14_phase_delta(x, *args)
    analytic = phenomd_eq14_phase_delta_derivative(x, *args)
    assert torch.allclose(torch.autograd.grad(y.sum(), x)[0], analytic, atol=1e-12, rtol=1e-12)
    # The C1 attach point is in the carrier Mf coordinate (Khan Sec. V).
    join = 0.5 * args[2]
    eps = 1e-7
    left = phenomd_eq14_phase_delta_c1(torch.tensor([join-eps],dtype=torch.float64), *args)
    right = phenomd_eq14_phase_delta_c1(torch.tensor([join+eps],dtype=torch.float64), *args)
    assert torch.allclose(left, right, atol=2e-11)


def test_eq14_delta_is_exactly_zero_for_identical_22_inputs():
    from torchwave.approximants.imrphenom_hm_phase_clean import phenomd_eq14_phase_delta_c1
    x = torch.linspace(0.01, 0.2, 17, dtype=torch.float64)
    z = phenomd_eq14_phase_delta_c1(
        x, 0.22, 0.1, 0.06, 0.06, 0.01, 0.01, 0.7, 1.1
    )
    assert torch.equal(z, torch.zeros_like(z))

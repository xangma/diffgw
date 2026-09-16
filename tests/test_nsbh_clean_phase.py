"""Invariant checks for the published NRTidalv2 NSBH phase term."""

import torch

from torchwave.approximants.nsbh_clean_phase import (
    nrtidalv2_frequency_phase,
    nrtidalv2_frequency_phase_derivative,
)


def test_nrtidalv2_phase_is_zero_for_zero_deformability():
    f = torch.tensor([30.0, 100.0, 1000.0], dtype=torch.float64)
    z = torch.zeros_like(f)
    out = nrtidalv2_frequency_phase(f, torch.tensor(1.0), torch.tensor(0.2), z)
    torch.testing.assert_close(out, z, rtol=0.0, atol=0.0)


def test_nrtidalv2_phase_is_linear_in_lambda_tilde():
    f = torch.linspace(20.0, 1000.0, 31, dtype=torch.float64)
    m, eta = torch.tensor(6.4 * 4.92549095e-6), torch.tensor(0.22)
    one = nrtidalv2_frequency_phase(f, m, eta, torch.ones_like(f))
    four = nrtidalv2_frequency_phase(f, m, eta, torch.full_like(f, 4.0))
    torch.testing.assert_close(four, 4.0 * one, rtol=2e-15, atol=0.0)


def test_nrtidalv2_newtonian_limit_matches_eq_18():
    # As x -> 0, the Padé factor tends to one (Dietrich et al. Eq. 18).
    f = torch.tensor([1.0e-10], dtype=torch.float64)
    # Physical total mass: 6.4 Msun in seconds.
    m, eta, lam = (torch.tensor(v, dtype=torch.float64) for v in (6.4 * 4.92549095e-6, 0.21, 500.0))
    phase = nrtidalv2_frequency_phase(f, m, eta, lam)
    x52 = (torch.pi * m * f).pow(5.0 / 3.0)
    expected = -(117.0 / (256.0 * eta)) * x52 * lam
    torch.testing.assert_close(phase, expected, rtol=2e-6, atol=0.0)


def test_nrtidalv2_phase_derivative_matches_autograd_of_eq_18():
    f = torch.linspace(30.0, 1200.0, 17, dtype=torch.float64, requires_grad=True)
    m = torch.tensor(8.4 * 4.92549095e-6, dtype=torch.float64)
    eta = torch.tensor(0.19, dtype=torch.float64)
    lam = torch.tensor(700.0, dtype=torch.float64)
    phase = nrtidalv2_frequency_phase(f, m, eta, lam)
    autodiff = torch.autograd.grad(phase.sum(), f)[0]
    direct = nrtidalv2_frequency_phase_derivative(f, m, eta, lam)
    torch.testing.assert_close(direct, autodiff, rtol=2e-13, atol=2e-15)


def test_nrtidalv2_phase_derivative_is_linear_in_lambda_tilde():
    f = torch.linspace(30.0, 1000.0, 11, dtype=torch.float64)
    # Physical NSBH total mass (6.4 Msun) in seconds.
    m, eta = torch.tensor(6.4 * 4.92549095e-6), torch.tensor(0.22)
    one = nrtidalv2_frequency_phase_derivative(f, m, eta, torch.ones_like(f))
    four = nrtidalv2_frequency_phase_derivative(f, m, eta, torch.full_like(f, 4.0))
    torch.testing.assert_close(four, 4.0 * one, rtol=2e-15, atol=0.0)


def test_nrtidalv2_phase_and_derivative_support_physical_batches_and_gradients():
    f = torch.tensor([30.0, 100.0, 500.0], dtype=torch.float64)
    masses = torch.tensor([6.4, 8.4], dtype=torch.float64, requires_grad=True) * 4.92549095e-6
    # Frozen NSBH mass ratios 5/1.4 and 7/1.4.
    eta = torch.tensor([5.0 * 1.4 / 6.4**2, 7.0 * 1.4 / 8.4**2], dtype=torch.float64, requires_grad=True)
    lam = torch.tensor([400.0, 2600.0], dtype=torch.float64, requires_grad=True)
    phase = nrtidalv2_frequency_phase(f, masses[:, None], eta[:, None], lam[:, None])
    deriv = nrtidalv2_frequency_phase_derivative(f, masses[:, None], eta[:, None], lam[:, None])
    assert torch.isfinite(phase).all() and torch.isfinite(deriv).all()
    grads = torch.autograd.grad((phase + deriv).sum(), (masses, eta, lam))
    assert all(torch.isfinite(g).all() for g in grads)


def test_nrtidalv2_phase_rejects_nonpositive_or_nonfinite_frequency():
    m = torch.tensor(6.4 * 4.92549095e-6, dtype=torch.float64)
    eta = torch.tensor(0.2, dtype=torch.float64)
    lam = torch.tensor(400.0, dtype=torch.float64)
    for f in (torch.tensor([0.0]), torch.tensor([-1.0]), torch.tensor([float("nan")]), torch.tensor([float("inf")])):
        for fn in (nrtidalv2_frequency_phase, nrtidalv2_frequency_phase_derivative):
            try:
                fn(f, m, eta, lam)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid frequency was accepted")

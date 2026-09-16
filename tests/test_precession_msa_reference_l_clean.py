"""Independent checks for the published MSA reference-L candidates."""
from __future__ import annotations

import torch

from torchwave.approximants.precession_msa_reference_l_clean import (
    leading_spin_infinity_coefficients,
    orbital_angular_momentum_candidates,
    orbital_angular_momentum_l4pn_losinfty,
    orbital_angular_momentum_nonconserved_3pn,
)
from torchwave.approximants.precession_beta_clean import orbital_angular_momentum


def test_equal_mass_equal_spin_coefficients_reduce_to_simple_polynomials():
    eta = torch.tensor(0.25, dtype=torch.float64)
    delta = torch.tensor(0.0, dtype=torch.float64)
    chi = torch.tensor(0.37, dtype=torch.float64)

    l2, l35 = leading_spin_infinity_coefficients(eta, delta, chi, chi)

    # Substituting eta=1/4 and delta=0 into Appendix-G's two symmetric
    # polynomials gives L2=chi^2 and L3.5=3*chi^3/4.
    assert torch.allclose(l2, chi.square(), atol=2e-15, rtol=0.0)
    assert torch.allclose(l35, 0.75 * chi.pow(3), atol=2e-15, rtol=0.0)


def test_spin_infinity_terms_vanish_in_zero_spin_limit_and_are_quadratic():
    mass1 = torch.tensor(35.0, dtype=torch.float64)
    mass2 = torch.tensor(25.0, dtype=torch.float64)
    frequency = torch.tensor(20.0, dtype=torch.float64)
    zero1 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    zero2 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    candidates = orbital_angular_momentum_candidates(
        mass1, mass2, frequency, zero1, zero2,
    )
    assert torch.allclose(
        candidates["four_pn_spin_losinfty"], candidates["four_pn_spin"],
        atol=2e-15, rtol=0.0,
    )
    correction = candidates["four_pn_spin_losinfty"] - candidates["four_pn_spin"]
    d1, d2 = torch.autograd.grad(correction, (zero1, zero2), create_graph=True)
    assert torch.allclose(d1, torch.zeros_like(d1), atol=2e-14, rtol=0.0)
    assert torch.allclose(d2, torch.zeros_like(d2), atol=2e-14, rtol=0.0)

    # A finite second derivative is an independent check that the optional
    # contribution begins at spin-squared order rather than being a hidden
    # linear or constant adjustment.
    curvature = torch.autograd.grad(d1, zero1)[0]
    assert torch.isfinite(curvature) and curvature > 0.0


def test_candidate_broadcasting_and_gradients():
    mass1 = torch.tensor([[35.0], [45.0]], dtype=torch.float64, requires_grad=True)
    mass2 = torch.tensor([[25.0], [15.0]], dtype=torch.float64)
    frequency = torch.tensor([20.0, 80.0, 160.0], dtype=torch.float64, requires_grad=True)
    chi1_l = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    chi2_l = torch.tensor([-0.1, 0.0, 0.15], dtype=torch.float64, requires_grad=True)

    value = orbital_angular_momentum_l4pn_losinfty(
        mass1, mass2, frequency, chi1_l, chi2_l,
    )
    assert value.shape == (2, 3)
    assert torch.isfinite(value).all()
    loss = value.square().mean()
    grads = torch.autograd.grad(loss, (mass1, frequency, chi1_l, chi2_l))
    assert all(torch.isfinite(grad).all() for grad in grads)
    assert all(torch.any(grad.abs() > 0.0) for grad in grads)


def test_nonconserved_series_has_the_published_zero_spin_limit():
    args = (35.0, 25.0, 80.0, 0.0, 0.0)
    old = orbital_angular_momentum_nonconserved_3pn(*args)
    conserved = orbital_angular_momentum(*args, pn_version=102, compat_102=False)
    assert torch.allclose(old, conserved, atol=2e-15, rtol=0.0)


def test_nonconserved_series_is_broadcastable_and_spin_linear_at_first_order():
    mass1 = torch.tensor([[35.0], [45.0]], dtype=torch.float64)
    mass2 = torch.tensor([[25.0], [15.0]], dtype=torch.float64)
    frequency = torch.tensor([20.0, 80.0, 160.0], dtype=torch.float64)
    chi1 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    chi2 = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True)
    value = orbital_angular_momentum_nonconserved_3pn(
        mass1, mass2, frequency, chi1, chi2,
    )
    assert value.shape == (2, 3)
    assert torch.isfinite(value).all()
    d1, d2 = torch.autograd.grad(value.sum(), (chi1, chi2))
    assert torch.isfinite(d1) and torch.isfinite(d2)
    assert d1 != 0.0 and d2 != 0.0

    # Eq. (6.9) is affine in aligned spins through the displayed 2.5PN
    # spin-orbit terms, so the central finite difference removes the
    # nonspinning series to numerical precision.
    eps = 1.0e-6
    plus = orbital_angular_momentum_nonconserved_3pn(
        35.0, 25.0, 80.0, eps, -eps,
    )
    minus = orbital_angular_momentum_nonconserved_3pn(
        35.0, 25.0, 80.0, -eps, eps,
    )
    zero = orbital_angular_momentum_nonconserved_3pn(
        35.0, 25.0, 80.0, 0.0, 0.0,
    )
    assert torch.allclose((plus + minus) / 2.0, zero, atol=2e-14, rtol=0.0)


def test_label_exchange_preserves_the_physical_candidate():
    mass1, mass2, frequency = 35.0, 25.0, 80.0
    chi1_l, chi2_l = 0.2, -0.1
    original = orbital_angular_momentum_l4pn_losinfty(
        mass1, mass2, frequency, chi1_l, chi2_l,
    )
    exchanged = orbital_angular_momentum_l4pn_losinfty(
        mass2, mass1, frequency, chi2_l, chi1_l,
    )
    assert torch.allclose(original, exchanged, atol=2e-14, rtol=0.0)


def test_nonphysical_frequency_is_rejected_before_singular_series_evaluation():
    try:
        orbital_angular_momentum_l4pn_losinfty(35.0, 25.0, 0.0, 0.2, -0.1)
    except ValueError as exc:
        assert "frequency" in str(exc)
    else:
        raise AssertionError("zero frequency must be rejected")

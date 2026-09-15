"""Focused tests for the opt-in general circular PA initializer."""

import pytest
import torch
import mpmath

from torchwave.approximants.eob.teob_postadiabatic_clean import (
    TEOBPostAdiabaticError,
    circular_angular_momentum_squared,
)


def _equation_terms(
    A: torch.Tensor,
    Aprime: torch.Tensor,
    r_c: torch.Tensor,
    r_c_prime: torch.Tensor | float = 1.0,
    Gprime: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, ...]:
    rc_prime = torch.as_tensor(r_c_prime, dtype=A.dtype, device=A.device)
    gp = torch.as_tensor(Gprime, dtype=A.dtype, device=A.device)
    U = A / r_c.square()
    Uprime = (Aprime - 2.0 * A * rc_prime / r_c) / r_c.square()
    a = Uprime.square() - 4.0 * U * gp.square()
    b = 2.0 * Aprime * Uprime - 4.0 * A * gp.square()
    c = Aprime.square()
    return U, Uprime, a, b, c


def test_zero_spin_lower_root_value_diagnostic_satisfies_squared_equation():
    radius = torch.tensor([20.0, 18.7, 16.2, 14.1], dtype=torch.float64)
    A = 1.0 - 2.0 / radius + 0.01 / radius.square()
    Aprime = 2.0 / radius.square() - 0.02 / radius.pow(3)
    x = circular_angular_momentum_squared(A, Aprime, radius)
    _, Uprime, a, b, c = _equation_terms(A, Aprime, radius)
    assert torch.all(torch.isfinite(x))
    assert torch.all(x > 0)
    torch.testing.assert_close(
        a * x.square() + b * x + c,
        torch.zeros_like(x),
        rtol=5e-13,
        atol=5e-20,
    )
    torch.testing.assert_close(x, -Aprime / Uprime, rtol=1e-14, atol=1e-16)


def test_general_inputs_broadcast_and_lower_root_value_diagnostic_remains_finite():
    radius = torch.tensor([[20.0], [17.0]], dtype=torch.float64)
    A = 1.0 - 2.0 / radius
    Aprime = 2.0 / radius.square()
    # These are chart-level inputs for the helper contract; a small nonzero
    # G' keeps both published quadratic terms exercised without fitting.
    x = circular_angular_momentum_squared(
        A,
        Aprime.expand(2, 3),
        radius,
        r_c_prime=1.0,
        Gprime=torch.tensor([0.0, 1.0e-4, 2.0e-4], dtype=torch.float64),
    )
    A_b = A.expand(2, 3)
    r_b = radius.expand(2, 3)
    _, _, a, b, c = _equation_terms(
        A_b,
        Aprime.expand(2, 3),
        r_b,
        Gprime=torch.tensor([0.0, 1.0e-4, 2.0e-4], dtype=torch.float64),
    )
    assert x.shape == (2, 3)
    assert torch.isfinite(x).all()
    assert (x > 0).all()
    torch.testing.assert_close(
        a * x.square() + b * x + c,
        torch.zeros_like(x),
        rtol=2e-12,
        atol=2e-19,
    )


def mp_root_for_scalar(
    radius: float,
    A: float,
    Aprime: float,
    gprime: float,
) -> float:
    with mpmath.workdps(80):
        r_mp = mpmath.mpf(radius)
        a_mp = mpmath.mpf(A)
        p_mp = mpmath.mpf(Aprime)
        gp_mp = mpmath.mpf(gprime)
        U_mp = a_mp / (r_mp * r_mp)
        Uprime_mp = (p_mp - 2.0 * a_mp / r_mp) / (r_mp * r_mp)
        aa_mp = Uprime_mp * Uprime_mp - 4.0 * U_mp * gp_mp * gp_mp
        bb_mp = 2.0 * p_mp * Uprime_mp - 4.0 * a_mp * gp_mp * gp_mp
        cc_mp = p_mp * p_mp
        disc_mp = bb_mp * bb_mp - 4.0 * aa_mp * cc_mp
        if disc_mp < 0:
            disc_mp = mpmath.mpf("0")
        return float((-bb_mp - mpmath.sqrt(disc_mp)) / (2.0 * aa_mp))


@pytest.mark.parametrize(
    "gprime",
    [1.0e-12, -1.0e-12, 1.0e-8, -1.0e-8, 1.0e-4, -1.0e-4],
)
def test_near_zero_gprime_matches_mpmath_reference_for_tolerance(gprime: float) -> None:
    radius = torch.linspace(14.0, 20.0, 61, dtype=torch.float64)
    A = 1.0 - 2.0 / radius
    Aprime = 2.0 / radius.square()
    x = circular_angular_momentum_squared(
        A,
        Aprime,
        radius,
        Gprime=gprime,
    )
    _, Uprime, a, b, c = _equation_terms(A, Aprime, radius, Gprime=gprime)
    assert torch.all(torch.isfinite(x))
    discriminant = b.square() - 4.0 * a * c
    roots_disc = torch.where(discriminant < 0.0, torch.zeros_like(discriminant), discriminant)
    closed_form_root_minus = (-b - torch.sqrt(roots_disc)) / (2.0 * a)
    closed_form_root_plus = (-b + torch.sqrt(roots_disc)) / (2.0 * a)
    well_conditioned = discriminant > 10.0 * torch.finfo(radius.dtype).eps * (b.abs() + 1.0)
    if bool(torch.any(well_conditioned)):
        torch.testing.assert_close(
            x[well_conditioned],
            torch.minimum(
                closed_form_root_minus, closed_form_root_plus
            )[well_conditioned],
            rtol=2e-11,
            atol=1e-16,
        )
    for row in (0, 30, 60):
        expected = torch.as_tensor(
            mp_root_for_scalar(
                float(radius[row].item()),
                float(A[row].item()),
                float(Aprime[row].item()),
                gprime,
            ),
            dtype=x.dtype,
        )
        torch.testing.assert_close(x[row], expected, rtol=2e-14, atol=1e-14)


def test_zero_g_and_near_zero_lower_root_radius_gradient_is_finite():
    radius = torch.linspace(
        14.0, 20.0, 61, dtype=torch.float64, requires_grad=True
    )
    A = 1.0 - 2.0 / radius
    Aprime = 2.0 / radius.square()
    Gprime = torch.tensor([0.0, 1.0e-8, 2.0e-4], dtype=torch.float64)
    radius_batch = radius.unsqueeze(1).expand(61, 3)
    A_batch = A.unsqueeze(1).expand(61, 3)
    Aprime_batch = Aprime.unsqueeze(1).expand(61, 3)
    x = circular_angular_momentum_squared(
        A_batch,
        Aprime_batch,
        radius_batch,
        Gprime=Gprime,
    )

    radius_gradient = torch.autograd.grad(x[:, 0].sum(), radius, allow_unused=False)[0]
    assert torch.isfinite(radius_gradient).all()
    radius_zero = radius.clone().detach().requires_grad_(True)
    A_zero = 1.0 - 2.0 / radius_zero
    Aprime_zero = 2.0 / radius_zero.square()
    Uprime_zero = (
        Aprime_zero - 2.0 * A_zero / radius_zero
    ) / radius_zero.square()
    x_zero_exact = -Aprime_zero / Uprime_zero
    radius_gradient_zero = torch.autograd.grad(
        x_zero_exact.sum(),
        radius_zero,
        allow_unused=False,
    )[0]
    torch.testing.assert_close(
        radius_gradient,
        radius_gradient_zero,
        rtol=2e-12,
        atol=5e-12,
    )

    _, Uprime, a, b, c = _equation_terms(
        A_batch,
        Aprime_batch,
        radius_batch,
        Gprime=Gprime,
    )
    discriminant = b.square() - 4.0 * a * c
    torch.testing.assert_close(
        a * x.square() + b * x + c,
        torch.zeros_like(x),
        rtol=5e-13,
        atol=5e-20,
    )
    roots_disc = torch.where(discriminant < 0.0, torch.zeros_like(discriminant), discriminant)
    closed_form_root_minus = (-b - torch.sqrt(roots_disc)) / (2.0 * a)
    closed_form_root_plus = (-b + torch.sqrt(roots_disc)) / (2.0 * a)
    nonzero_g_mask = Gprime != 0.0
    well_conditioned = (
        discriminant > 10.0 * torch.finfo(radius.dtype).eps * (b.abs() + 1.0)
    ) & nonzero_g_mask.unsqueeze(0)
    if bool(torch.any(well_conditioned)):
        torch.testing.assert_close(
            x[well_conditioned],
            torch.minimum(
                closed_form_root_minus, closed_form_root_plus
            )[well_conditioned],
            rtol=2e-11,
            atol=1e-16,
        )


def test_initializer_has_finite_gradients_through_all_circular_inputs():
    A = torch.tensor([0.90, 0.88], dtype=torch.float64, requires_grad=True)
    Aprime = torch.tensor([0.005, 0.006], dtype=torch.float64, requires_grad=True)
    r_c = torch.tensor([20.0, 18.0], dtype=torch.float64, requires_grad=True)
    r_c_prime = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
    Gprime = torch.tensor([1.0e-4, 2.0e-4], dtype=torch.float64, requires_grad=True)
    x = circular_angular_momentum_squared(A, Aprime, r_c, r_c_prime, Gprime)
    objective = x.square().sum()
    gradients = torch.autograd.grad(
        objective, (A, Aprime, r_c, r_c_prime, Gprime), allow_unused=False
    )
    assert torch.isfinite(objective)
    assert torch.isfinite(x).all()
    for gradient in gradients:
        assert torch.isfinite(gradient).all()


def test_nonpositive_or_degenerate_circular_branch_is_explicitly_rejected():
    with pytest.raises(TEOBPostAdiabaticError, match="positive"):
        circular_angular_momentum_squared(
            torch.tensor([0.9], dtype=torch.float64),
            torch.tensor([-0.005], dtype=torch.float64),
            torch.tensor([20.0], dtype=torch.float64),
        )
    with pytest.raises(TEOBPostAdiabaticError, match="coefficient is zero"):
        circular_angular_momentum_squared(
            torch.tensor([0.9], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([20.0], dtype=torch.float64),
            r_c_prime=0.0,
        )

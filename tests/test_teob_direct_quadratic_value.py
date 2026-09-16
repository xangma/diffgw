"""Focused contract tests for the direct PA quadratic value policy."""

from fractions import Fraction

import pytest
import torch

from torchwave.approximants.eob import teob_conditioned_waveform_clean as adapter
from torchwave.approximants.eob.teob_postadiabatic_clean import (
    PA_INITIALIZER_POLICIES,
    TEOBPostAdiabaticError,
    alternating8pa,
    circular_angular_momentum_squared_direct,
)
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


def _terms(A, Aprime, r_c, r_c_prime, Gprime):
    """Independently evaluate the published quadratic coefficients."""
    u = torch.reciprocal(r_c)
    U = A * u.square()
    Uprime = u.square() * (Aprime - 2.0 * A * u * r_c_prime)
    a = Uprime.square() - 4.0 * U * Gprime.square()
    b = 2.0 * Aprime * Uprime - 4.0 * A * Gprime.square()
    c = Aprime.square()
    return U, Uprime, a, b, c


def test_direct_value_uses_independent_polynomial_and_signed_branch():
    dtype = torch.float64
    A = torch.tensor([0.9, 0.85], dtype=dtype)
    Aprime = torch.tensor([0.08, 0.06], dtype=dtype)
    radius = torch.tensor([20.0, 18.0], dtype=dtype)
    radius_prime = torch.ones_like(radius)
    Gprime = torch.tensor([-1.0e-3, 1.0e-3], dtype=dtype)
    x = circular_angular_momentum_squared_direct(
        A, Aprime, radius, radius_prime, Gprime
    )
    _, _, a, b, c = _terms(A, Aprime, radius, radius_prime, Gprime)
    discriminant = b.square() - 4.0 * a * c
    root_minus = (-b - torch.sqrt(torch.clamp_min(discriminant, 0.0))) / (2.0 * a)
    root_plus = (-b + torch.sqrt(torch.clamp_min(discriminant, 0.0))) / (2.0 * a)
    expected = torch.where(Gprime > 0.0, root_plus, root_minus)
    torch.testing.assert_close(x, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(a * x.square() + b * x + c, torch.zeros_like(x), rtol=2e-14, atol=1e-16)
    # Independently check the unsquared stationarity relation from
    # H_eff=sqrt(A+U*x)+G*sqrt(x), including the signed G' branch.
    u = torch.reciprocal(radius)
    U = A * u.square()
    Uprime = u.square() * (Aprime - 2.0 * A * u * radius_prime)
    stationarity = Aprime + Uprime * x + 2.0 * Gprime * torch.sqrt(x) * torch.sqrt(A + U * x)
    torch.testing.assert_close(stationarity, torch.zeros_like(x), rtol=2e-14, atol=1e-16)


def test_direct_value_preserves_positive_float32_roundoff_discriminant():
    # At G'=0 the exact polynomial discriminant is zero.  For these simple
    # rational inputs, Fraction establishes that exact result while float32
    # operations produce a small positive roundoff value.  The direct policy
    # must evaluate that value rather than replacing it analytically.
    dtype = torch.float32
    A, Aprime, radius, radius_prime = (
        torch.tensor(2.0, dtype=dtype),
        torch.tensor(-3.0, dtype=dtype),
        torch.tensor(3.0, dtype=dtype),
        torch.tensor(-3.0, dtype=dtype),
    )
    zero = torch.zeros((), dtype=dtype)
    exact_u = Fraction(1, 3)
    exact_Uprime = exact_u * exact_u * (Fraction(-3) - 2 * 2 * exact_u * Fraction(-3))
    exact_a = exact_Uprime * exact_Uprime
    exact_b = 2 * Fraction(-3) * exact_Uprime
    exact_c = Fraction(9)
    assert exact_b * exact_b - 4 * exact_a * exact_c == 0
    _, _, a, b, c = _terms(A, Aprime, radius, radius_prime, zero)
    discriminant = b.square() - 4.0 * a * c
    assert discriminant > 0
    x = circular_angular_momentum_squared_direct(
        A, Aprime, radius, radius_prime, zero
    )
    expected = (-b - torch.sqrt(discriminant)) / (2.0 * a)
    torch.testing.assert_close(x, expected, rtol=0.0, atol=0.0)


def test_direct_value_clamps_negative_discriminant_and_checks_guards():
    values = [
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(0.5, dtype=torch.float64),
        torch.tensor(1.5, dtype=torch.float64),
        torch.tensor(-4.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    ]
    A, Aprime, radius, radius_prime, Gprime = values
    _, _, a, b, c = _terms(A, Aprime, radius, radius_prime, Gprime)
    discriminant = b.square() - 4.0 * a * c
    assert discriminant < 0
    x = circular_angular_momentum_squared_direct(*values)
    expected = (-b + torch.sqrt(torch.zeros_like(discriminant))) / (2.0 * a)
    torch.testing.assert_close(x, expected, rtol=0.0, atol=0.0)

    with pytest.raises(TEOBPostAdiabaticError, match="A and r_c"):
        circular_angular_momentum_squared_direct(-1.0, 0.5, 2.0)
    with pytest.raises(TEOBPostAdiabaticError, match="coefficient is zero"):
        circular_angular_momentum_squared_direct(1.0, 0.0, 2.0, 0.0, 0.0)
    with pytest.raises(TEOBPostAdiabaticError, match="nonfinite"):
        circular_angular_momentum_squared_direct(float("nan"), 0.5, 2.0)
    with pytest.raises(TypeError, match="real tensors"):
        circular_angular_momentum_squared_direct(
            1.0, torch.tensor(0.5 + 0.0j), 2.0
        )


def test_direct_policy_is_opt_in_and_value_only_in_pa_and_adapter():
    assert "direct-quadratic-value" in PA_INITIALIZER_POLICIES
    assert adapter._policy(
        "direct-quadratic-value",
        name="pa_initializer_policy",
        choices=adapter.PA_INITIALIZER_POLICIES,
    ) == "direct-quadratic-value"

    zeros = [torch.tensor(0.0, dtype=torch.float64)] * 4
    ham = TEOBResumSHamiltonian(torch.tensor(0.24, dtype=torch.float64), *zeros)
    result = alternating8pa(
        ham, r0=20.0, r_min=18.0, dr=0.1, pa_order=4,
        initializer_policy="direct-quadratic-value",
    )
    assert result.complete
    assert result.metadata["initializer_value_only"] is True
    assert torch.isfinite(result.state).all()

    tracked = torch.tensor(0.24, dtype=torch.float64, requires_grad=True)
    tracked_ham = TEOBResumSHamiltonian(tracked, *zeros)
    with pytest.raises(TEOBPostAdiabaticError, match="value-only"):
        alternating8pa(
            tracked_ham, r0=20.0, r_min=18.0, dr=0.1,
            initializer_policy="direct-quadratic-value",
        )

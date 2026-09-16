"""Independent checks for the opt-in closed ``P^1_5`` orbital potential."""

import mpmath as mp
import pytest
import torch

from torchwave.approximants.eob.teob_orbital_potential_closed_clean import (
    orbital_potential_p15,
    orbital_potential_p15_radial_derivative,
    orbital_potential_p15_with_radial_derivative,
    teobresums_orbital_potential_closed,
)


def _mp_reference(u, nu):
    """Independent 5-by-5 coefficient solve in 80-digit arithmetic."""
    pi = mp.pi
    a4 = mp.mpf(94) / 3 - mp.mpf(41) * pi**2 / 32
    a50 = (-mp.mpf(4237) / 60 + mp.mpf(2275) * pi**2 / 512
           + mp.mpf(256) * mp.log(2) / 5 + mp.mpf(128) * mp.euler / 5)
    a51 = mp.mpf(41) * pi**2 / 32 - mp.mpf(221) / 6
    a5c = a50 + a51 * nu
    a6c = mp.mpf("3097.3") * nu**2 - mp.mpf("1330.6") * nu + mp.mpf("81.38")
    a6log = -mp.mpf(7004) / 105 - mp.mpf(144) * nu / 5
    c = [mp.mpf(1), -mp.mpf(2), mp.mpf(0), 2 * nu,
         nu * a4, nu * (a5c + mp.mpf(64) * mp.log(u) / 5),
         nu * (a6c + a6log * mp.log(u))]
    matrix = mp.matrix([[c[n - j] if n - j >= 0 else mp.mpf(0)
                         for j in range(1, 6)] for n in range(2, 7)])
    denominator_coefficients = mp.lu_solve(matrix, mp.matrix([-c[n] for n in range(2, 7)]))
    denominator = 1 + sum(denominator_coefficients[j - 1] * u**j for j in range(1, 6))
    numerator = 1 + (-2 + denominator_coefficients[0]) * u
    return numerator / denominator


@pytest.mark.parametrize("radius,nu", [("4.7", "0.2"), ("6.3", "0.23"),
                                        ("11.0", "0.12")])
def test_value_and_radial_derivative_match_independent_high_precision_solve(radius, nu):
    with mp.workdps(80):
        radius, nu = mp.mpf(radius), mp.mpf(nu)
        u = 1 / radius
        expected_value = _mp_reference(u, nu)
        expected_radial = mp.diff(lambda rr: _mp_reference(1 / rr, nu), radius)
    value, radial = orbital_potential_p15_with_radial_derivative(
        torch.tensor(float(u), dtype=torch.float64), float(nu)
    )
    assert float(value) == pytest.approx(float(expected_value), rel=2e-14, abs=2e-14)
    assert float(radial) == pytest.approx(float(expected_radial), rel=2e-13, abs=2e-14)


def test_schwarzschild_limit_and_entry_point_alias():
    u = torch.tensor([0.01, 0.1, 0.2], dtype=torch.float64)
    expected = 1 - 2 * u
    expected_radial = 2 * u.square()
    assert torch.equal(orbital_potential_p15(u, 0.0), expected)
    assert torch.equal(orbital_potential_p15_radial_derivative(u, 0.0), expected_radial)
    assert torch.equal(teobresums_orbital_potential_closed(u, 0.2), orbital_potential_p15(u, 0.2))


def test_batch_float32_float64_and_autograd_radial_identity():
    for dtype in (torch.float32, torch.float64):
        # B != T exercises the same trailing-axis convention as the existing
        # (u, nu) entry point, where nu[B] broadcasts across u[B, T].
        u = torch.tensor([[1 / 4.7, 1 / 6.3, 1 / 8.0], [1 / 9.0, 1 / 10.0, 1 / 11.0]], dtype=dtype, requires_grad=True)
        nu = torch.tensor([0.2, 0.23], dtype=dtype)
        value = orbital_potential_p15(u, nu)
        assert value.shape == u.shape
        assert torch.isfinite(value).all()
        grad_u = torch.autograd.grad(value.sum(), u)[0]
        assert torch.isfinite(grad_u).all()
        if dtype == torch.float64:
            radial = orbital_potential_p15_radial_derivative(u.detach(), nu)
            assert torch.allclose(radial, -u.detach().square() * grad_u, rtol=2e-13, atol=2e-14)


def test_scalar_mass_ratio_uses_u_dtype_and_zero_mass_ratio_gradient_is_retained():
    u = torch.tensor([0.16, 0.21], dtype=torch.float64)
    scalar = orbital_potential_p15(u, 0.24)
    explicit = orbital_potential_p15(u, torch.tensor(0.24, dtype=torch.float64))
    assert torch.equal(scalar, explicit)

    u0 = torch.tensor(0.16, dtype=torch.float64)
    nu0 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    value0 = orbital_potential_p15(u0, nu0)
    grad_nu0 = torch.autograd.grad(value0, nu0)[0]
    h = torch.tensor(1e-6, dtype=torch.float64)
    forward_difference = (orbital_potential_p15(u0, h) - orbital_potential_p15(u0, 0.0)) / h
    assert torch.isfinite(grad_nu0)
    assert torch.allclose(grad_nu0, forward_difference, rtol=2e-6, atol=2e-8)


def test_complex_mass_ratio_is_rejected_before_dtype_cast():
    u = torch.tensor(0.16, dtype=torch.float64)
    with pytest.raises(TypeError, match="nu must be real"):
        orbital_potential_p15(u, 0.24 + 0.01j)
    with pytest.raises(TypeError, match="nu must be real"):
        orbital_potential_p15(u, torch.tensor(0.24 + 0.01j, dtype=torch.complex128))

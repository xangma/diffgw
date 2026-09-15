"""Focused checks for the opt-in clean numerical EOB derivative experiment."""

import pytest
import torch

from torchwave.approximants.eob.hamiltonian import EOBHamiltonian
from torchwave.approximants.eob.numerical_derivative_clean import (
    NumericalDerivativeDomainError,
    adaptive_central_derivative,
    central_quadratic,
    central_quadratic_from_values,
    central_quartic,
    central_quartic_from_values,
    numerical_eob_conservative_derivatives,
)
import torchwave.approximants.eob.numerical_derivative_clean as numerical_derivative_module
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


DTYPE = torch.float64


def _tensor(value, *, requires_grad=False):
    return torch.as_tensor(value, dtype=DTYPE).requires_grad_(requires_grad)


def _nonspinning(nu):
    return EOBHamiltonian(nu, torch.zeros_like(nu), torch.zeros_like(nu))


def test_independent_quadratic_and_quartic_stencil_weights():
    x = _tensor(1.3)
    h = _tensor(0.07)
    quadratic_values = torch.stack(((x - h).square(), (x + h).square()))
    quartic_nodes = x + torch.stack((-h, -h / 2, h / 2, h))
    quartic_values = quartic_nodes.pow(4)

    torch.testing.assert_close(
        central_quadratic_from_values(quadratic_values, h), 2 * x,
        rtol=1e-14, atol=1e-14,
    )
    torch.testing.assert_close(
        central_quartic_from_values(quartic_values, h), 4 * x.pow(3),
        rtol=1e-14, atol=1e-14,
    )


def test_no_misleading_gsl_alias_is_exported():
    assert not hasattr(numerical_derivative_module, "gsl_deriv_central_clean")
    assert "gsl_deriv_central_clean" not in numerical_derivative_module.__all__


def test_callable_stencils_and_adaptation_on_smooth_functions():
    x = _tensor([-0.8, 0.2, 1.1])
    h = _tensor(1.0e-3)
    quartic = central_quartic(torch.sin, x, h)
    quadratic = central_quadratic(torch.sin, x, h)
    torch.testing.assert_close(quartic, torch.cos(x), rtol=2e-12, atol=2e-13)
    torch.testing.assert_close(quadratic, torch.cos(x), rtol=2e-6, atol=2e-9)

    result = adaptive_central_derivative(
        torch.exp, x, initial_step=1.0e-4, max_iterations=1, return_info=True,
    )
    torch.testing.assert_close(result.derivative, torch.exp(x), rtol=2e-9, atol=2e-11)
    assert torch.isfinite(result.error).all()
    assert torch.isfinite(result.truncation_error).all()
    assert torch.isfinite(result.roundoff_error).all()
    assert (result.step > 0).all()
    assert (result.evaluations >= 4).all()
    assert (result.evaluations <= 8).all()


def test_eob_chain_rule_matches_clean_analytic_path_nonspin_and_spin():
    cases = [
        (
            _nonspinning(_tensor(0.24)),
            _tensor([6.0, 9.0]),
            _tensor([-0.15, 0.10]),
            _tensor([3.0, 4.0]),
        ),
        (
            SEOBNRv4Hamiltonian(_tensor(0.16), _tensor(0.70), _tensor(-0.30)),
            _tensor([7.0, 11.0]),
            _tensor([-0.08, 0.12]),
            _tensor([3.2, 4.1]),
        ),
    ]
    for ham, radius, radial, angular in cases:
        numerical = numerical_eob_conservative_derivatives(
            ham, radius, radial, angular, initial_step=1.0e-4,
        )
        analytic = ham.equations_of_motion_conservative(radius, radial, angular)
        for actual, expected in zip(numerical, analytic):
            torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-9)


def test_eob_partial_derivatives_preserve_parameter_batches():
    nu = _tensor([0.16, 0.24])
    chi1 = _tensor([0.70, -0.30])
    chi2 = _tensor([-0.20, 0.60])
    ham = SEOBNRv4Hamiltonian(nu, chi1, chi2)
    radius, radial, angular = _tensor(8.5), _tensor(-0.06), _tensor(3.6)
    actual = numerical_eob_conservative_derivatives(
        ham, radius, radial, angular, initial_step=1.0e-4,
    )
    expected_rows = [
        numerical_eob_conservative_derivatives(
            SEOBNRv4Hamiltonian(nu[i], chi1[i], chi2[i]),
            radius, radial, angular, initial_step=1.0e-4,
        )
        for i in range(2)
    ]
    for actual_value, rows in zip(actual, zip(*expected_rows)):
        torch.testing.assert_close(actual_value, torch.stack(rows), rtol=2e-8, atol=2e-10)


def test_eob_field_keeps_state_and_parameter_gradients():
    radius = _tensor(8.0, requires_grad=True)
    radial = _tensor(-0.04, requires_grad=True)
    angular = _tensor(3.7, requires_grad=True)
    nu = _tensor(0.18, requires_grad=True)
    ham = EOBHamiltonian(nu, _tensor(0.2), _tensor(-0.1))

    field = numerical_eob_conservative_derivatives(
        ham, radius, radial, angular, initial_step=1.0e-4,
    )
    loss = sum(value.square().sum() for value in field)
    gradients = torch.autograd.grad(loss, (radius, radial, angular, nu))
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert any(bool(value.abs().max() > 0) for value in gradients)


def test_domain_errors_identify_invalid_nodes_and_states():
    with pytest.raises(NumericalDerivativeDomainError):
        central_quartic(torch.log, _tensor(1.0e-5), _tensor(1.0e-4))

    ham = _nonspinning(_tensor(0.2))
    with pytest.raises(NumericalDerivativeDomainError):
        numerical_eob_conservative_derivatives(ham, _tensor(0.0), _tensor(0.0), _tensor(3.0))
    with pytest.raises(NumericalDerivativeDomainError):
        numerical_eob_conservative_derivatives(
            ham, _tensor(float("nan")), _tensor(0.0), _tensor(3.0),
        )


def test_zero_mass_ratio_uses_finite_effective_hamiltonian_limit():
    ham = _nonspinning(_tensor(0.0))
    radius, radial, angular = _tensor(8.0), _tensor(0.07), _tensor(3.5)
    numerical = numerical_eob_conservative_derivatives(
        ham, radius, radial, angular, initial_step=1.0e-4,
    )
    analytic = ham.equations_of_motion_conservative(radius, radial, angular)
    for actual, expected in zip(numerical, analytic):
        torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-9)

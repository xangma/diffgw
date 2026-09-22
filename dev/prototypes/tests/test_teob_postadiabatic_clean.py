"""Contract tests for the bounded clean TEOB alternating PA profile."""

import pytest
import torch

from torchwave.approximants.eob.teob_postadiabatic_clean import (
    PA_INITIALIZER_POLICIES,
    TEOBPostAdiabaticError,
    alternating8pa,
    cubic_interval_integral,
    cumulative_cubic_integral,
    quartic_derivative,
)
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


def _ham(*, nu=0.24, dtype=torch.float64, chi1=0.0, lambda1=0.0):
    values = [nu, chi1, 0.0, lambda1, 0.0]
    tensors = [torch.tensor(value, dtype=dtype) for value in values]
    return TEOBResumSHamiltonian(*tensors)


def test_quartic_derivative_is_exact_on_descending_grid_and_endpoints():
    radius = torch.tensor([7.4, 6.8, 5.1, 4.0, 2.7, 1.3, -0.2], dtype=torch.float64)
    polynomial = radius.pow(4) - 2.0 * radius.pow(3) + 0.5 * radius.square() + 3.0 * radius - 1.0
    expected = 4.0 * radius.pow(3) - 6.0 * radius.square() + radius + 3.0

    derivative = quartic_derivative(polynomial, radius)
    batched = quartic_derivative(torch.stack((polynomial, 2.0 * polynomial)), radius)

    torch.testing.assert_close(derivative, expected, rtol=2e-13, atol=2e-12)
    torch.testing.assert_close(batched[0], expected, rtol=2e-13, atol=2e-12)
    torch.testing.assert_close(batched[1], 2.0 * expected, rtol=2e-13, atol=2e-12)


def test_cubic_integrals_are_exact_with_descending_endpoints():
    radius = torch.tensor([8.0, 7.4, 6.1, 5.5, 4.0, 2.8, 1.2], dtype=torch.float64)
    values = radius.pow(3) - 2.0 * radius.square() + 4.0 * radius - 1.0
    antiderivative = radius.pow(4) / 4.0 - 2.0 * radius.pow(3) / 3.0 + 2.0 * radius.square() - radius
    expected_intervals = antiderivative[1:] - antiderivative[:-1]
    expected_cumulative = torch.cat((torch.zeros(1, dtype=radius.dtype), torch.cumsum(expected_intervals, dim=0)))

    intervals = cubic_interval_integral(values, radius)
    cumulative = cumulative_cubic_integral(values, radius)
    constant_cumulative = cumulative_cubic_integral(torch.ones_like(radius), radius)
    stacked_cumulative = cumulative_cubic_integral(
        torch.stack((values, 2.0 * values)), radius, offset=2.0
    )

    torch.testing.assert_close(intervals, expected_intervals, rtol=2e-13, atol=2e-12)
    torch.testing.assert_close(cumulative, expected_cumulative, rtol=2e-13, atol=2e-12)
    torch.testing.assert_close(constant_cumulative, radius - radius[0], rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(
        stacked_cumulative,
        2.0 + torch.stack((expected_cumulative, 2.0 * expected_cumulative)),
        rtol=2e-13,
        atol=2e-12,
    )


def test_pa_balances_are_checked_independently():
    result = alternating8pa(_ham(), r0=20.0, r_min=16.0, dr=0.1, pa_order=8)

    assert result.complete
    assert torch.all(result.radial_velocity < 0)
    torch.testing.assert_close(
        result.pr_star_rate,
        result.p_prime * result.radial_velocity,
        rtol=3e-9,
        atol=3e-18,
    )
    # Angular balance is the finite-order diagnostic.  It is independently
    # evaluated from the returned derivative, velocity, and vacuum torque.
    scale = result.torque.abs().clamp_min(1e-30)
    assert torch.max(result.angular_balance_residual.abs() / scale) < 2e-7
    assert torch.isfinite(result.state).all()
    assert result.endpoint_state.shape == (4,)
    torch.testing.assert_close(result.endpoint_state, result.state[-1])


def test_pa_order_refinement_improves_weak_field_endpoint():
    profiles = {
        order: alternating8pa(_ham(), r0=24.0, r_min=20.0, dr=0.1, pa_order=order)
        for order in (2, 4, 8)
    }
    delta_24 = abs(profiles[2].phase[-1] - profiles[4].phase[-1])
    delta_48 = abs(profiles[4].phase[-1] - profiles[8].phase[-1])
    time_delta_24 = abs(profiles[2].time[-1] - profiles[4].time[-1])
    time_delta_48 = abs(profiles[4].time[-1] - profiles[8].time[-1])

    assert delta_48 < delta_24 * 1e-2
    assert time_delta_48 < time_delta_24 * 1e-2


def test_grid_refinement_is_finite_and_convergent():
    coarse = alternating8pa(_ham(), r0=20.0, r_min=16.0, dr=0.2, pa_order=8)
    fine = alternating8pa(_ham(), r0=20.0, r_min=16.0, dr=0.1, pa_order=8)
    assert coarse.radius[-1] == fine.radius[-1] == torch.tensor(16.0, dtype=torch.float64)
    assert torch.isfinite(fine.state).all()
    assert abs(fine.phase[-1] - coarse.phase[-1]) < 1e-5
    assert abs(fine.time[-1] - coarse.time[-1]) < 2e-3


def test_pa_has_finite_intrinsic_gradient_and_preserves_offsets():
    nu = torch.tensor(0.24, dtype=torch.float64, requires_grad=True)
    zero = torch.tensor(0.0, dtype=torch.float64)
    ham = TEOBResumSHamiltonian(nu, zero, zero, zero, zero)
    start_time = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    start_phase = torch.tensor(-0.5, dtype=torch.float64, requires_grad=True)

    result = alternating8pa(
        ham,
        r0=20.0,
        r_min=18.0,
        dr=0.1,
        pa_order=4,
        start_time=start_time,
        start_phase=start_phase,
    )
    objective = result.endpoint_time + result.endpoint_phase + result.pr_star[-1]
    grad_nu, grad_time, grad_phase = torch.autograd.grad(
        objective, (nu, start_time, start_phase)
    )
    assert torch.isfinite(objective)
    assert torch.isfinite(grad_nu)
    assert torch.equal(result.time[0], start_time)
    assert torch.equal(result.phase[0], start_phase)
    assert torch.equal(grad_time, torch.ones_like(grad_time))
    assert torch.equal(grad_phase, torch.ones_like(grad_phase))


def test_quadratic_value_initializer_is_labeled_and_value_only():
    result = alternating8pa(
        _ham(), r0=20.0, r_min=18.0, dr=0.1, pa_order=4,
        initializer_policy="quadratic-value",
    )
    assert result.complete
    assert result.initializer_policy == "quadratic-value"
    assert result.metadata["initializer_value_only"] is True
    assert PA_INITIALIZER_POLICIES == (
        "analytic", "quadratic-value", "direct-quadratic-value"
    )
    assert torch.isfinite(result.state).all()


def test_quadratic_value_initializer_rejects_autograd_tracked_inputs():
    nu = torch.tensor(0.24, dtype=torch.float64, requires_grad=True)
    zero = torch.tensor(0.0, dtype=torch.float64)
    ham = TEOBResumSHamiltonian(nu, zero, zero, zero, zero)
    with pytest.raises(TEOBPostAdiabaticError, match="value-only"):
        alternating8pa(
            ham, r0=20.0, r_min=18.0, dr=0.1,
            initializer_policy="quadratic-value",
        )


def test_f_start_uses_kepler_mapping_and_short_profiles_skip_explicitly():
    result = alternating8pa(
        _ham(),
        f_start=100.0,
        total_mass_msun=2.8,
        r_min=14.0,
        dr=100.0,
        start_time=7.0,
        start_phase=1.25,
    )
    expected_omega = torch.pi * torch.tensor(2.8 * 4.925490947641267e-6 * 100.0, dtype=torch.float64)
    expected_radius = expected_omega.pow(-2.0 / 3.0)
    assert result.skipped
    assert result.radius.numel() == 1
    torch.testing.assert_close(result.start_radius, expected_radius)
    assert result.start_radius > torch.tensor(14.0, dtype=torch.float64)
    torch.testing.assert_close(result.time, torch.full((1,), 7.0, dtype=torch.float64))
    torch.testing.assert_close(result.phase, torch.full((1,), 1.25, dtype=torch.float64))
    # The frequency is evaluated from the EOB state; the physical start
    # frequency is used only for the stated Kepler radius mapping.
    assert torch.isfinite(result.omega).all()
    assert expected_omega > 0


def test_f_start_requires_an_explicit_physical_mass():
    with pytest.raises(ValueError, match="requires total_mass_msun"):
        alternating8pa(_ham(), f_start=100.0, r_min=14.0, dr=100.0)


def test_skipped_pa_has_no_endpoint_and_complete_pa_inherits_endpoint_offsets():
    skipped = alternating8pa(
        _ham(), r0=14.5, r_min=14.0, dr=0.1,
        start_time=4.0, start_phase=2.0,
    )
    assert skipped.skipped
    assert skipped.skip_reason and "at least 10" in skipped.skip_reason
    assert skipped.endpoint is None
    assert torch.all(skipped.time == 4.0)
    assert torch.all(skipped.phase == 2.0)

    complete = alternating8pa(
        _ham(), r0=20.0, r_min=18.0, dr=0.1,
        start_time=4.0, start_phase=2.0,
    )
    assert complete.complete
    assert complete.endpoint is not None
    torch.testing.assert_close(complete.endpoint.state, complete.state[-1])
    torch.testing.assert_close(complete.endpoint.time, complete.time[-1])
    torch.testing.assert_close(complete.endpoint.phase, complete.phase[-1])
    assert complete.time[0] == torch.tensor(4.0, dtype=torch.float64)
    assert complete.phase[0] == torch.tensor(2.0, dtype=torch.float64)


def test_invalid_intrinsic_and_order_inputs_are_rejected():
    with pytest.raises(TEOBPostAdiabaticError, match="zero spins"):
        alternating8pa(_ham(chi1=0.1), r0=20.0)
    with pytest.raises(TEOBPostAdiabaticError, match="zero spins"):
        alternating8pa(_ham(lambda1=10.0), r0=20.0)
    batched = TEOBResumSHamiltonian(
        torch.tensor([0.22, 0.24], dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="scalar intrinsic"):
        alternating8pa(batched, r0=20.0)
    with pytest.raises(ValueError, match="one of r0"):
        alternating8pa(_ham())
    with pytest.raises(ValueError, match="one of 2"):
        alternating8pa(_ham(), r0=20.0, pa_order=3)
    with pytest.raises(ValueError, match="strictly descending"):
        quartic_derivative(torch.arange(5.0, dtype=torch.float64), torch.arange(5.0, dtype=torch.float64))


def test_nonzero_custom_radial_force_is_an_explicit_failure():
    class BadRadiation:
        def radiation_reaction_forces(self, omega, pr_star, p_phi, **kwargs):
            return torch.ones_like(omega), -torch.ones_like(omega)

    with pytest.raises(TEOBPostAdiabaticError, match="zero radial"):
        alternating8pa(_ham(), BadRadiation(), r0=20.0, r_min=19.0)

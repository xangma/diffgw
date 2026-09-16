"""Contract tests for the bounded clean TEOB PA-to-ODE continuation."""

import pytest
import torch

from torchwave.approximants.eob.adaptive_driver_clean import (
    AdaptiveIntegrationError,
    fixed_fehlberg45_step,
    integrate_adaptive,
)
from torchwave.approximants.eob.teob_adaptive_continuation_clean import (
    AcceptedFrequencyDecreaseEvent,
    TEOBAdaptiveContinuationError,
    integrate_adaptive_continuation,
    continue_teob_adaptive,
)
from torchwave.approximants.eob.teob_postadiabatic_clean import alternating8pa
from torchwave.approximants.eob.teob_vacuum_driver_clean import TEOBVacuumRadiationDriver
from torchwave.approximants.teobresums import TEOBResumSHamiltonian


DTYPE = torch.float64


def _turning_field(state: torch.Tensor) -> torch.Tensor:
    """A state-only field whose orbital frequency has one smooth maximum."""
    radius = state[..., 0]
    frequency = 1.0 - (radius - 4.7).square()
    return torch.stack((-torch.ones_like(radius), frequency), dim=-1)


def _time_turning_field(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """The same turning field expressed in an absolute time coordinate."""
    radius = state[..., 0]
    frequency = 1.0 - (time - 0.3).square()
    return torch.stack((-torch.ones_like(radius), frequency), dim=-1)


def _ham(*, nu: float = 0.24, dtype: torch.dtype = DTYPE,
         chi1: float = 0.0, lambda1: float = 0.0) -> TEOBResumSHamiltonian:
    values = [nu, chi1, 0.0, lambda1, 0.0]
    tensors = [torch.tensor(value, dtype=dtype) for value in values]
    return TEOBResumSHamiltonian(*tensors)


def _prefix() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time = torch.tensor([10.0, 10.05, 10.1], dtype=DTYPE)
    radius = torch.tensor([5.2, 5.1, 5.0], dtype=DTYPE)
    phase = torch.tensor([1.0, 1.13, 1.25], dtype=DTYPE)
    state = torch.stack((radius, phase), dim=-1)
    frequency = 1.0 - (radius - 4.7).square()
    return time, state, frequency


def _continuation_kwargs(**overrides):
    kwargs = dict(
        output_step=0.025,
        max_step=0.15,
        max_duration=10.0,
        fixed_step_cap=0.1,
        continuation_duration=0.2,
    )
    kwargs.update(overrides)
    return kwargs


def test_shared_driver_exposes_next_step_and_fixed_state_scaled_trial():
    initial = torch.tensor([5.0, 0.0], dtype=DTYPE)
    trajectory = integrate_adaptive(
        _turning_field,
        initial,
        0.1,
        t_end=0.2,
        controller="gsl_state_scaled",
    )

    assert trajectory.proposed_next_step is not None
    assert trajectory.proposed_next_step > 0
    derivative = _turning_field(initial)
    trial = fixed_fehlberg45_step(
        _turning_field,
        torch.zeros((), dtype=DTYPE),
        initial,
        0.1,
        atol=1e-13,
        rtol=1e-11,
        current_derivative=derivative,
    )
    assert trial.accepted
    assert trial.evaluations == 5
    assert torch.isfinite(trial.fifth).all()


def test_completed_pa_like_prefix_transfers_exactly_and_keeps_one_seam():
    prefix_time, prefix_state, prefix_frequency = _prefix()
    result = integrate_adaptive_continuation(
        _turning_field,
        prefix_state[-1],
        0.025,
        start_time=prefix_time[-1],
        prefix_time=prefix_time,
        prefix_state=prefix_state,
        prefix_frequency=prefix_frequency,
        **_continuation_kwargs(),
    )

    assert result.complete
    assert torch.equal(result.adaptive_state[0], prefix_state[-1])
    assert torch.equal(result.adaptive_time[0], prefix_time[-1])
    assert torch.equal(result.state[prefix_time.numel() - 1], prefix_state[-1])
    assert result.first_ode_step == torch.tensor(0.025, dtype=DTYPE)
    assert result.phase_continuity == torch.zeros((), dtype=DTYPE)
    assert result.phase_continuous
    assert torch.all(result.time[1:] > result.time[:-1])
    assert result.time.numel() == torch.unique(result.time).numel()
    assert result.peak_event_index is not None
    assert result.peak_event_time == result.time[result.peak_event_index]
    assert result.fixed_time[0] == result.peak_event_time
    assert result.fixed_step == torch.minimum(
        torch.tensor(0.1, dtype=DTYPE), result.adaptive_next_step,
    )
    assert result.fixed_time[-1] >= result.target_time
    assert result.fixed_time[-1] - result.peak_event_time >= torch.tensor(0.2, dtype=DTYPE)
    if result.fixed_time.numel() > 1:
        torch.testing.assert_close(
            result.fixed_time.diff(),
            result.fixed_step.expand(result.fixed_time.numel() - 1),
            rtol=0,
            atol=1e-14,
        )


def test_absolute_time_and_phase_offsets_are_covariant():
    state = torch.tensor([5.0, 0.0], dtype=DTYPE)
    shifted_state = torch.tensor([5.0, 2.0], dtype=DTYPE)
    base = integrate_adaptive_continuation(
        _turning_field, state, 0.1, start_time=0.0,
        **_continuation_kwargs(output_step=0.1),
    )
    shifted = integrate_adaptive_continuation(
        _turning_field, shifted_state, 0.1, start_time=7.0,
        **_continuation_kwargs(output_step=0.1),
    )

    torch.testing.assert_close(shifted.time, base.time + 7.0, rtol=0, atol=1e-13)
    torch.testing.assert_close(shifted.state[:, 0], base.state[:, 0], rtol=0, atol=1e-13)
    torch.testing.assert_close(shifted.state[:, 1], base.state[:, 1] + 2.0, rtol=0, atol=1e-13)
    torch.testing.assert_close(shifted.frequency, base.frequency, rtol=0, atol=1e-13)
    assert shifted.peak_event_time == base.peak_event_time + 7.0


def test_event_uses_immediately_adjacent_accepted_derivatives_and_floor_is_explicit():
    event = AcceptedFrequencyDecreaseEvent()
    event.initialize(1, dtype=DTYPE, device=torch.device("cpu"))
    indices = torch.zeros(1, dtype=torch.long)
    time = torch.zeros(1, dtype=DTYPE)
    state = torch.tensor([[4.8, 0.0]], dtype=DTYPE)
    previous = torch.tensor([[0.0, 0.96]], dtype=DTYPE)
    increasing = torch.tensor([[0.0, 0.97]], dtype=DTYPE)
    decreasing = torch.tensor([[0.0, 0.95]], dtype=DTYPE)

    stop, reason = event.check(time, state, increasing, previous, indices)
    assert not bool(stop.item())
    assert reason is None
    stop, reason = event.check(time, state, decreasing, increasing, indices)
    assert bool(stop.item())
    assert reason == "frequency_decrease"

    below = torch.tensor([[0.9, 0.0]], dtype=DTYPE)
    stop, reason = event.check(time, below, decreasing, increasing, indices)
    assert bool(stop.item())
    assert reason == "radius_below_one_before_frequency_decrease"


def test_fixed_continuation_uses_fifth_order_candidate_without_shortening():
    result = integrate_adaptive_continuation(
        _turning_field,
        torch.tensor([5.0, 0.0], dtype=DTYPE),
        0.1,
        **_continuation_kwargs(continuation_duration=0.23, fixed_step_cap=0.08),
    )
    assert result.complete
    assert result.fixed_step == torch.tensor(0.08, dtype=DTYPE)
    assert result.fixed_time[-1] - result.peak_event_time >= torch.tensor(0.23, dtype=DTYPE)
    assert result.fixed_time[-1] - result.peak_event_time < torch.tensor(0.31, dtype=DTYPE)
    assert result.statistics.fixed_accepted_steps == result.fixed_time.numel() - 1
    assert result.statistics.fixed_rejected_steps == 0


def test_skipped_pa_requires_explicit_ode_initial_state_and_step():
    skipped = alternating8pa(_ham(), r0=14.5, r_min=14.0, dr=0.1)
    assert skipped.skipped
    with pytest.raises(ValueError, match="explicit initial_state"):
        continue_teob_adaptive(_ham(), skipped, continuation_duration=0.2)
    with pytest.raises(ValueError, match="explicit initial_step"):
        continue_teob_adaptive(
            _ham(), skipped,
            initial_state=torch.tensor([14.5, 0.0, -0.001, 3.0], dtype=DTYPE),
            continuation_duration=0.2,
        )


def test_domain_and_budget_failures_return_accepted_diagnostics():
    def domain_field(time, state):
        derivative = _turning_field(state)
        return torch.where(
            (time <= 0.55), derivative, torch.full_like(derivative, float("nan")),
        )

    failed = integrate_adaptive_continuation(
        domain_field,
        torch.tensor([5.0, 0.0], dtype=DTYPE),
        0.1,
        raise_on_failure=False,
        **_continuation_kwargs(),
    )
    assert failed.status == "failed"
    assert failed.skip_reason == "domain_event"
    assert failed.statistics.fixed_accepted_steps >= 1
    assert failed.fixed_time[-1] < failed.target_time
    assert torch.isfinite(failed.endpoint_state).all()

    budget = integrate_adaptive_continuation(
        _turning_field,
        torch.tensor([5.0, 0.0], dtype=DTYPE),
        0.1,
        max_accepted_steps=4,
        raise_on_failure=False,
        **_continuation_kwargs(),
    )
    assert budget.status == "failed"
    assert budget.skip_reason == "accepted_step_budget"
    assert budget.statistics.accepted_steps <= 4
    assert torch.isfinite(budget.endpoint_state).all()


def test_adaptive_budget_failure_retains_two_accepted_knots_and_work_count():
    result = integrate_adaptive_continuation(
        _turning_field,
        torch.tensor([5.0, 0.0], dtype=DTYPE),
        0.1,
        start_time=7.0,
        max_accepted_steps=2,
        raise_on_failure=False,
        **_continuation_kwargs(),
    )
    assert result.status == "failed"
    assert result.skip_reason == "accepted_step_budget"
    torch.testing.assert_close(
        result.adaptive_time,
        torch.tensor([7.0, 7.1, 7.25], dtype=DTYPE),
        rtol=0,
        atol=1e-14,
    )
    assert result.statistics.adaptive_accepted_steps == 2
    # One initial RHS plus five stages and one endpoint probe per accepted
    # adaptive step.  The failure path must not evaluate the initial RHS a
    # second time while constructing diagnostics.
    assert result.statistics.adaptive_evaluations == 13
    assert result.fixed_time.numel() == 0


def test_adaptive_domain_failure_retains_prefix_after_two_accepted_knots():
    def domain_field(time, state):
        derivative = _turning_field(state)
        return torch.where(
            time[..., None] <= 0.251,
            derivative,
            torch.full_like(derivative, float("nan")),
        )

    result = integrate_adaptive_continuation(
        domain_field,
        torch.tensor([5.0, 0.0], dtype=DTYPE),
        0.1,
        raise_on_failure=False,
        **_continuation_kwargs(min_step=0.1, max_step=0.15, max_rejected_steps=100),
    )
    assert result.status == "failed"
    assert result.skip_reason == "domain_event"
    assert result.statistics.adaptive_accepted_steps >= 2
    assert result.adaptive_time.numel() >= 3
    torch.testing.assert_close(
        result.adaptive_time[-1], torch.tensor(0.25, dtype=DTYPE), rtol=0, atol=1e-14,
    )
    assert result.statistics.adaptive_rejected_steps == 1
    assert result.statistics.adaptive_evaluations == 18
    assert result.fixed_time.numel() == 0


def test_fixed_duration_budget_rejects_crossing_step_and_retains_prefix():
    result = integrate_adaptive_continuation(
        _turning_field,
        torch.tensor([5.0, 0.0], dtype=DTYPE),
        0.1,
        output_step=0.1,
        max_step=0.15,
        max_duration=0.55,
        fixed_step_cap=0.08,
        continuation_duration=0.1,
        raise_on_failure=False,
    )
    assert result.status == "failed"
    assert result.skip_reason == "duration_budget"
    torch.testing.assert_close(
        result.fixed_time,
        torch.tensor([0.4, 0.48], dtype=DTYPE),
        rtol=0,
        atol=1e-14,
    )
    assert result.fixed_time[-1] < result.target_time
    assert result.statistics.fixed_accepted_steps == 1
    assert result.statistics.fixed_evaluations == 6


def test_initial_radius_and_batch_inputs_are_rejected_explicitly():
    with pytest.raises(TEOBAdaptiveContinuationError, match="below the pre-event domain floor"):
        integrate_adaptive_continuation(
            _turning_field,
            torch.tensor([0.9, 0.0], dtype=DTYPE),
            0.1,
            **_continuation_kwargs(),
        )
    with pytest.raises(ValueError, match="one scalar state vector"):
        integrate_adaptive_continuation(
            _turning_field,
            torch.ones((2, 2), dtype=DTYPE),
            0.1,
            **_continuation_kwargs(),
        )


def test_continuation_preserves_gradients_through_accepted_states():
    phase_offset = torch.tensor(0.2, dtype=DTYPE, requires_grad=True)

    def field(state):
        derivative = _turning_field(state)
        return torch.stack((derivative[..., 0], derivative[..., 1] + phase_offset), dim=-1)

    result = integrate_adaptive_continuation(
        field,
        torch.tensor([5.0, 0.0], dtype=DTYPE),
        0.1,
        **_continuation_kwargs(),
    )
    objective = result.endpoint_state[1]
    gradient = torch.autograd.grad(objective, phase_offset)[0]
    assert torch.isfinite(objective)
    assert torch.isfinite(gradient)
    assert gradient > 0


def test_short_physical_fixed_step_refinement_is_finite():
    hamiltonian = _ham()
    pa = alternating8pa(hamiltonian, r0=20.0, r_min=18.5, dr=0.1, pa_order=8)
    assert pa.complete
    driver = TEOBVacuumRadiationDriver(hamiltonian)
    state = pa.endpoint_state
    time = pa.endpoint_time
    derivative = driver.vector_field(state)
    coarse = fixed_fehlberg45_step(
        driver.vector_field, time, state, 0.1,
        current_derivative=derivative,
    )
    half = fixed_fehlberg45_step(
        driver.vector_field, time, state, 0.05,
        current_derivative=derivative,
    )
    half_derivative = driver.vector_field(half.fifth)
    second_half = fixed_fehlberg45_step(
        driver.vector_field, time + 0.05, half.fifth, 0.05,
        current_derivative=half_derivative,
    )
    assert coarse.accepted and half.accepted and second_half.accepted
    assert torch.isfinite(coarse.fifth).all()
    refinement = (coarse.fifth - second_half.fifth).abs().amax()
    assert torch.isfinite(refinement)
    assert refinement < torch.tensor(1e-5, dtype=DTYPE)


def test_fixed_primitive_reports_domain_errors_without_retrying():
    def bad_field(time, state):
        if bool(time > 0.01):
            return torch.full_like(state, float("nan"))
        return torch.zeros_like(state)

    with pytest.raises(AdaptiveIntegrationError) as raised:
        fixed_fehlberg45_step(
            bad_field,
            torch.zeros((), dtype=DTYPE),
            torch.tensor([5.0, 0.0], dtype=DTYPE),
            0.1,
        )
    assert raised.value.reason == "domain_event"
    assert raised.value.evaluations == 6

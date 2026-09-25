
import torch
import pytest

import torchwave.approximants.eob.adaptive_driver_clean as adaptive_driver
from torchwave.approximants.eob.adaptive_driver_clean import AdaptiveIntegrationError, AlignedSpinV4Event, GSL_STATE_SCALED_CONTROLLER, _apply_calibrated_nqc, integrate_adaptive, resample_natural_cubic, state_scaled_controller_adjustment
from torchwave.approximants.eob.nqc import nqc_22_factor
from torchwave.approximants.eob.numerical_attachment import natural_cubic_spline


DTYPE = torch.float64


def test_rkf45_keeps_accepted_nodes_separate_from_uniform_output():
    trajectory = integrate_adaptive(
        lambda time, state: state,
        torch.ones(1, dtype=DTYPE),
        0.2,
        t_end=1.0,
    )

    assert trajectory.initial_step == torch.tensor(0.2, dtype=DTYPE)
    torch.testing.assert_close(
        trajectory.output_time,
        torch.arange(6, dtype=DTYPE) * 0.2,
    )
    assert trajectory.time.ndim == 1
    assert trajectory.time.numel() != trajectory.output_time.numel()
    assert trajectory.statistics.total_rejected >= 1
    torch.testing.assert_close(
        trajectory.output_state[-1, 0],
        torch.exp(torch.tensor(1.0, dtype=DTYPE)),
        rtol=3e-8,
        atol=3e-8,
    )


def test_accepted_state_event_stops_at_accepted_state():
    trajectory = integrate_adaptive(
        lambda time, state: torch.ones_like(state),
        torch.zeros(1, dtype=DTYPE),
        0.2,
        event=lambda time, state, derivative, previous: state[..., 0] >= 0.55,
    )

    assert trajectory.statistics.termination == ("event",)
    assert trajectory.statistics.event_time == trajectory.time[-1]
    assert trajectory.state[-1, 0] >= 0.55
    assert bool(trajectory.output_valid[-1])
    assert trajectory.output_time[-1] <= trajectory.time[-1]


def test_nonuniform_natural_cubic_resampling_uses_full_knot_series():
    knots = torch.tensor([0.0, 0.35, 1.1, 2.0], dtype=DTYPE)
    values = (knots**3 - 0.2 * knots**2 + knots).unsqueeze(-1)
    query, sampled = resample_natural_cubic(knots, values, 0.2)
    spline = natural_cubic_spline(knots, values, scalar_output=False)

    assert query.shape == (11,)
    assert sampled.shape == (11, 1)
    torch.testing.assert_close(sampled, spline.evaluate(query), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(spline.evaluate(knots), values, rtol=1e-12, atol=1e-12)
    # The last knot is natural: the second derivative vanishes at the endpoint.
    torch.testing.assert_close(
        spline.evaluate(knots[-1], derivative=2),
        torch.zeros(1, dtype=DTYPE),
        rtol=1e-12,
        atol=1e-12,
    )


def test_tolerance_refinement_reduces_ode_error():
    initial = torch.ones(1, dtype=DTYPE)

    loose = integrate_adaptive(
        lambda time, state: state,
        initial,
        0.1,
        t_end=1.0,
        atol=1e-7,
        rtol=1e-5,
    )
    tight = integrate_adaptive(
        lambda time, state: state,
        initial,
        0.1,
        t_end=1.0,
        atol=1e-11,
        rtol=1e-9,
    )
    exact = torch.exp(torch.tensor(1.0, dtype=DTYPE))
    loose_error = (loose.output_state[-1, 0] - exact).abs()
    tight_error = (tight.output_state[-1, 0] - exact).abs()
    assert tight_error < loose_error
    assert tight.statistics.total_accepted >= loose.statistics.total_accepted


def test_batch_and_parameter_gradient_survive_adaptive_sampling():
    parameter = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
    initial = torch.tensor([[1.0], [2.0]], dtype=DTYPE)
    trajectory = integrate_adaptive(
        lambda time, state: parameter * state,
        initial,
        0.1,
        t_end=0.5,
    )

    assert trajectory.state.shape[0] == 2
    assert trajectory.output_state.shape[0] == 2
    assert bool(trajectory.output_valid.all())
    loss = trajectory.output_state[:, -1, 0].sum()
    loss.backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad)
    assert parameter.grad > 0


def test_invalid_intermediate_stage_is_an_explicit_domain_event():
    def field(time, state):
        invalid = time > 0.01
        return torch.where(invalid, torch.full_like(state, float("nan")), state * 0.0)

    with pytest.raises(AdaptiveIntegrationError) as raised:
        integrate_adaptive(
            field,
            torch.ones(1, dtype=DTYPE),
            0.1,
            t_end=0.2,
            min_step=0.02,
            max_rejected_steps=100,
        )
    assert raised.value.reason == "domain_event"
    assert raised.value.row_indices is not None


def test_eob_event_compares_adjacent_accepted_frequency():
    event = AlignedSpinV4Event("coarse")
    event.initialize(1, dtype=DTYPE, device=torch.device("cpu"))
    indices = torch.zeros(1, dtype=torch.long)
    time = torch.zeros(1, dtype=DTYPE)
    state = torch.tensor([[5.0, 0.0, -1.0, 2.0]], dtype=DTYPE)
    derivative = torch.tensor([[0.0, 1.0, -1.0, 0.0]], dtype=DTYPE)
    previous = torch.tensor([[0.0, 0.5, -1.0, 0.0]], dtype=DTYPE)
    assert not bool(event.check(time, state, derivative, previous, indices).item())

    derivative = torch.tensor([[0.0, 2.0, -1.0, 0.0]], dtype=DTYPE)
    assert not bool(event.check(time, state, derivative, previous, indices).item())

    # The immediately preceding accepted frequency is 2.0.  A stale
    # two-step comparison against 1.0 would miss this decrease.
    previous = derivative.clone()
    derivative = torch.tensor([[0.0, 1.5, -1.0, 0.0]], dtype=DTYPE)
    assert bool(event.check(time, state, derivative, previous, indices).item())


def test_same_nonzero_nqc_coefficients_cross_the_waveform_seam():
    state = torch.tensor([
        [8.0, 0.1, -0.18, 3.0],
        [6.5, 0.3, -0.24, 3.1],
        [5.9, 0.5, -0.31, 3.2],
    ], dtype=DTYPE)
    orbital_frequency = torch.tensor([0.055, 0.072, 0.094], dtype=DTYPE)
    base_mode = torch.exp(1j * torch.tensor([0.2, 0.7, 1.3], dtype=DTYPE))
    coefficients = (
        torch.tensor([0.37, -0.22, 0.11], dtype=DTYPE),
        torch.tensor([-0.41, 0.29], dtype=DTYPE),
    )

    expected = base_mode * nqc_22_factor(
        state[:, 0], state[:, 2], orbital_frequency, coefficients, sample_dims=1)
    corrected = _apply_calibrated_nqc(
        base_mode, state, orbital_frequency, coefficients)
    torch.testing.assert_close(corrected, expected)
    assert not torch.allclose(corrected, base_mode)

    # The final coarse sample and first fine sample share the restart state;
    # applying the same calibration makes their seam value identical.
    coarse = _apply_calibrated_nqc(
        base_mode[:2], state[:2], orbital_frequency[:2], coefficients)
    fine = _apply_calibrated_nqc(
        base_mode[1:], state[1:], orbital_frequency[1:], coefficients)
    torch.testing.assert_close(coarse[-1], fine[0])


def test_state_scaled_controller_thresholds_are_inclusive_and_bounded():
    half = torch.tensor(0.5, dtype=DTYPE)
    one_point_one = torch.tensor(1.1, dtype=DTYPE)
    ratios = torch.stack((
        torch.zeros((), dtype=DTYPE),
        torch.nextafter(half, torch.zeros_like(half)),
        half,
        torch.nextafter(one_point_one, torch.zeros_like(one_point_one)),
        one_point_one,
        torch.nextafter(one_point_one, torch.full_like(one_point_one, 2.0)),
        torch.tensor(100.0, dtype=DTYPE),
    ))
    action, factor = state_scaled_controller_adjustment(ratios)
    assert action.tolist() == [1, 1, 0, 0, 0, -1, -1]
    torch.testing.assert_close(factor[0], torch.tensor(5.0, dtype=DTYPE))
    assert bool((factor >= 0.2).all())
    assert bool((factor <= 5.0).all())


def test_state_scaled_controller_uses_candidate_state_for_error_scale(monkeypatch):
    original_trial = adaptive_driver._rkf45_trial

    def controlled_trial(field, time, state, step, current_field, arity, **kwargs):
        # The candidate is much smaller than the old state.  With atol=.1 and
        # rtol=1, candidate scaling gives r=1.5 at the first trial, while the
        # old max(old,candidate) scale gives r=0.3/1.1 < 1.
        candidate = torch.full_like(state, 0.1)
        error = 0.3 * (step / 0.1).pow(6).unsqueeze(-1).expand_as(state)
        return state, candidate, error, 5

    monkeypatch.setattr(adaptive_driver, "_rkf45_trial", controlled_trial)
    generic = integrate_adaptive(
        lambda time, state: torch.zeros_like(state),
        torch.ones(1, dtype=DTYPE),
        0.1,
        t_end=0.1,
        atol=0.1,
        rtol=1.0,
    )
    state_scaled = integrate_adaptive(
        lambda time, state: torch.zeros_like(state),
        torch.ones(1, dtype=DTYPE),
        0.1,
        t_end=0.1,
        atol=0.1,
        rtol=1.0,
        controller=GSL_STATE_SCALED_CONTROLLER,
    )
    assert generic.statistics.total_rejected == 0
    assert state_scaled.statistics.total_rejected >= 1
    assert state_scaled.statistics.total_accepted >= 1
    monkeypatch.setattr(adaptive_driver, "_rkf45_trial", original_trial)


def test_state_scaled_controller_keeps_analytic_ode_accurate():
    trajectory = integrate_adaptive(
        lambda time, state: state,
        torch.ones(1, dtype=DTYPE),
        0.2,
        t_end=1.0,
        controller=GSL_STATE_SCALED_CONTROLLER,
    )
    torch.testing.assert_close(
        trajectory.output_state[-1, 0],
        torch.exp(torch.tensor(1.0, dtype=DTYPE)),
        rtol=3e-8,
        atol=3e-8,
    )


def test_rejected_domain_trial_restores_the_last_accepted_state():
    failed_once = False

    def field(time, state):
        nonlocal failed_once
        if not failed_once and bool((time > 0.05).any()):
            failed_once = True
            return torch.full_like(state, float("nan"))
        return state

    trajectory = integrate_adaptive(
        field,
        torch.ones(1, dtype=DTYPE),
        0.1,
        t_end=0.1,
        min_step=0.001,
        controller=GSL_STATE_SCALED_CONTROLLER,
    )
    torch.testing.assert_close(trajectory.time[1], torch.tensor(0.025, dtype=DTYPE))
    torch.testing.assert_close(
        trajectory.state[1, 0], torch.exp(torch.tensor(0.025, dtype=DTYPE)),
        rtol=1e-8,
        atol=1e-8,
    )
    assert trajectory.statistics.total_rejected >= 1


def test_full_shape_batch_rhs_handles_distinct_parameters_and_event_histories():
    rates = torch.tensor([[0.1], [10.0]], dtype=DTYPE)
    calls = []

    def field(time, state):
        calls.append((tuple(time.shape), tuple(state.shape)))
        return rates * state

    trajectory = integrate_adaptive(
        field,
        torch.ones(2, 1, dtype=DTYPE),
        0.01,
        t_end=0.5,
        event=lambda time, state, derivative, previous: state[..., 0] >= 1.5,
        controller=GSL_STATE_SCALED_CONTROLLER,
    )

    assert calls
    assert all(shape == ((2,), (2, 1)) for shape in calls)
    assert trajectory.statistics.termination == ("duration", "event")
    assert bool(trajectory.output_valid[1].sum() < trajectory.output_valid[0].sum())
    torch.testing.assert_close(
        trajectory.state[0, trajectory.valid[0].sum() - 1, 0],
        torch.exp(torch.tensor(0.05, dtype=DTYPE)),
        rtol=2e-8,
        atol=2e-8,
    )

"""Analytic event locations and independent batched stopping checks."""

import pytest
import torch

from torchwave.approximants.eob.events import orbital_frequency_peak
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quartic_peak_exact_time_curvature_and_gradient(dtype):
    time = torch.linspace(-2, 2, 41, dtype=dtype)
    center = torch.tensor([-.43, .27], dtype=dtype, requires_grad=True)
    x = time-center[:, None]
    frequency = 2-.12*x.square()-.006*x.pow(4)
    peak = orbital_frequency_peak(time, frequency)
    tolerance = 3e-5 if dtype == torch.float32 else 3e-13
    torch.testing.assert_close(peak.time, center, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(peak.frequency, center*0+2, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(peak.curvature, center*0-.24, rtol=10*tolerance, atol=10*tolerance)
    derivative = torch.autograd.grad(peak.time.sum(), center)[0]
    torch.testing.assert_close(derivative, torch.ones_like(center), rtol=10*tolerance, atol=10*tolerance)


def test_smooth_nonpolynomial_peak_refines_with_sample_spacing():
    errors = []
    for dt in (.2, .1, .05):
        t = torch.arange(-2, 2, dt, dtype=torch.float64)
        x = t-.273
        peak = orbital_frequency_peak(t, torch.exp(-x.square())*(1+.06*x.pow(3)))
        errors.append(abs(peak.time.item()-.273))
    assert errors[-1] < 3e-7
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] > 50*errors[2]


def test_degree_six_peak_uses_high_order_derivatives_and_gradient():
    time = torch.linspace(-1.5, 1.5, 31, dtype=torch.float64)
    center = torch.tensor([0.17, -0.23], dtype=torch.float64, requires_grad=True)
    x = time-center[:, None]
    frequency = 1.7-.24*x.square()+.035*x.pow(4)-.012*x.pow(6)
    peak = orbital_frequency_peak(time, frequency)
    torch.testing.assert_close(peak.time, center, atol=2e-12, rtol=0)
    torch.testing.assert_close(peak.curvature, center*0-.48, atol=2e-11, rtol=0)
    derivative = torch.autograd.grad(peak.time.sum(), center)[0]
    torch.testing.assert_close(derivative, torch.ones_like(center), atol=2e-11, rtol=0)


@pytest.mark.parametrize("case", ["endpoint", "short", "gap", "duplicate"])
def test_unresolved_or_invalid_samples_raise(case):
    t = torch.linspace(0, 1, 11, dtype=torch.float64)
    omega = 2-(t-.4).square()
    valid = None
    if case == "endpoint":
        omega = t+1
    elif case == "short":
        t, omega = t[:4], omega[:4]
    elif case == "gap":
        valid = torch.ones_like(t, dtype=torch.bool)
        valid[1] = False
    else:
        t[3] = t[2]
    with pytest.raises(ValueError):
        orbital_frequency_peak(t, omega, valid=valid)


def test_nonpositive_orbital_frequency_cannot_define_matching_event():
    t = torch.linspace(-2, 2, 41, dtype=torch.float64)
    omega = -2 - .12*t.square()
    with pytest.raises(ValueError, match="frequency samples must be positive"):
        orbital_frequency_peak(t, omega)


def test_nonpositive_invalid_padding_does_not_define_matching_event():
    t = torch.linspace(-2, 2, 41, dtype=torch.float64)
    omega = 2 - .12*(t + .4).square()
    valid = torch.arange(t.numel()) < 30
    expected = orbital_frequency_peak(t[:30], omega[:30])
    omega[30:] = 0
    actual = orbital_frequency_peak(t, omega, valid=valid)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


class AnalyticFlow:
    """Radius advances linearly; each row has a different Gaussian Omega peak."""
    def __init__(self, center):
        self.center = center

    def equations_of_motion_conservative(self, r, pr_star, pphi):
        # A domain boundary just beyond each peak exposes accidental RK
        # evolution of a completed row while another row is still active.
        if bool((r > self.center+.5).any()):
            raise ValueError("advanced a completed trajectory")
        return torch.ones_like(r), torch.exp(-.1*(r-self.center).square()), torch.zeros_like(r)


class NoRadiation:
    def radiation_reaction_forces(self, omega, **kwargs):
        return torch.zeros_like(omega), torch.zeros_like(omega)


def test_batched_events_stop_independently_and_match_scalar_inference():
    center = torch.tensor([2.37, 4.28], dtype=torch.float64)
    initial = torch.tensor([1., 0., 0., 1.], dtype=torch.float64).expand(2, 4)
    solver = EOBTrajectoryIntegrator(AnalyticFlow(center), NoRadiation())
    result = solver.integrate_to_frequency_peak(initial, .1, 50)
    torch.testing.assert_close(result.peak.time, center-1, atol=2e-6, rtol=0)
    counts = result.valid.sum(0)
    assert counts[0] < counts[1]
    for row in range(2):
        single = EOBTrajectoryIntegrator(AnalyticFlow(center[row]), NoRadiation())
        with torch.inference_mode():
            other = single.integrate_to_frequency_peak(initial[row], .1, 50)
        torch.testing.assert_close(result.state[:counts[row], row], other.state)
        torch.testing.assert_close(result.peak.time[row], other.peak.time)
    with pytest.raises(ValueError, match="not reached"):
        solver.integrate_to_frequency_peak(initial, .1, 5)


def test_event_integration_preserves_location_parameter_gradient():
    center = torch.tensor(2.37, dtype=torch.float64, requires_grad=True)
    initial = torch.tensor([1., 0., 0., 1.], dtype=torch.float64)
    result = EOBTrajectoryIntegrator(AnalyticFlow(center), NoRadiation()).integrate_to_frequency_peak(initial, .05, 50)
    derivative = torch.autograd.grad(result.peak.time, center)[0]
    torch.testing.assert_close(derivative, torch.ones_like(derivative), atol=2e-6, rtol=0)


def test_post_peak_duration_retains_requested_independent_history():
    center = torch.tensor([2.37, 2.37], dtype=torch.float64)
    initial = torch.tensor([1., 0., 0., 1.], dtype=torch.float64).expand(2, 4)
    result = EOBTrajectoryIntegrator(AnalyticFlow(center), NoRadiation()).integrate_to_frequency_peak(
        initial, .05, 50, post_peak_duration=torch.tensor([0., .2], dtype=torch.float64))
    torch.testing.assert_close(result.peak.time, center-1, atol=2e-6, rtol=0)
    assert result.valid.sum(0).tolist() == [30, 34]
    with pytest.raises(ValueError, match='nonnegative'):
        EOBTrajectoryIntegrator(AnalyticFlow(center), NoRadiation()).integrate_to_frequency_peak(
            initial, .05, 50, post_peak_duration=-1.)

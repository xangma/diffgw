"""Live continuous-event, target, grid-independence and gradient checks."""

import pytest
import torch

from torchwave.approximants.eob.continuous_matching import continuous_spline_peak
from torchwave.approximants.eob.numerical_attachment import natural_cubic_spline
from torchwave.approximants.eob.teob_merger_nqc_clean import assemble_teob_merger_nqc


def test_peak_matches_analytic_three_knot_natural_spline():
    time = torch.tensor([0., 1., 2.], dtype=torch.float64)
    values = torch.tensor([0., 2., 1.], dtype=time.dtype)
    peak = continuous_spline_peak(time, values)
    # On the second interval: y'=1/2 - 9u/2 + 9u^2/4.
    expected = 2 - torch.sqrt(time.new_tensor(7.)) / 3
    torch.testing.assert_close(peak, expected, rtol=0, atol=1e-14)
    spline = natural_cubic_spline(time, values)
    assert abs(spline.derivative(peak)) < 1e-14
    assert spline.derivative(peak, 2) < 0
    shifted = continuous_spline_peak(time + 12300, values)
    torch.testing.assert_close(shifted - 12300, peak, rtol=0, atol=2e-12)


def test_peak_gradients_through_values_and_nonuniform_knots():
    time = torch.tensor([0., .8, 2.], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([0., 2., 1.], dtype=time.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(continuous_spline_peak, (time, values))


@pytest.mark.parametrize("values", [[0., 1., 2.], [1., 1., 1.], [0., 2., 0., 2., 0.]])
def test_unresolved_endpoint_and_tied_peaks_rejected(values):
    values = torch.tensor(values, dtype=torch.float64)
    with pytest.raises(ValueError, match="continuous peak"):
        continuous_spline_peak(torch.arange(len(values), dtype=values.dtype), values)


def test_peak_is_smooth_across_a_sampled_maximum_switch():
    time = torch.linspace(-5, 5, 101, dtype=torch.float64)
    def locate(offset):
        return continuous_spline_peak(time, 10 - (time - offset).square())
    left, right = locate(.05 - 1e-6), locate(.05 + 1e-6)
    torch.testing.assert_close(right - left, time.new_tensor(2e-6), rtol=2e-7, atol=1e-12)


def _nqc(step=.1, tilt=None, targets=None):
    time = torch.linspace(0, 40, 81, dtype=torch.float64)
    state = torch.stack((20 - .08*time, .01*time.square(), -.04-.001*time, 3+.01*time), -1)
    frequency = .15 - .0001 * (time - 30.17).square()
    if tilt is not None:
        frequency = frequency + tilt * (time - 30)
    return assemble_teob_merger_nqc(
        time, state, .8 + .002*time + .00001*time**3, .1*time + .0002*time**3,
        frequency, .0008 + .00002*time, nu=.24, fixed_step=step,
        targets=targets if targets is not None else (.32, -.002, .38, .012),
        target_provenance="synthetic test", matching_method="continuous",
    )


def test_corrected_accepted_splines_satisfy_all_four_physical_targets():
    result = _nqc()
    amplitude = natural_cubic_spline(result.time, result.corrected_amplitude / (.24 * 24**.5))
    phase = natural_cubic_spline(result.time, result.corrected_phase)
    actual = torch.stack((amplitude(result.t_nqc), amplitude.derivative(result.t_nqc),
                          phase.derivative(result.t_nqc), phase.derivative(result.t_nqc, 2)))
    torch.testing.assert_close(actual, torch.stack(result.targets), rtol=2e-9, atol=2e-11)
    assert abs(float(result.t_peak) - 30.17) < 1e-8
    assert float((result.aux_time - result.t_nqc).abs().min()) > .01


def test_auxiliary_spacing_cannot_change_continuous_coefficients_or_events():
    reference = _nqc(.1)
    for step in (.05, .2, .5):
        result = _nqc(step)
        assert result.aux_time.shape != reference.aux_time.shape
        for actual, expected in ((result.t_peak, reference.t_peak), (result.t_nqc, reference.t_nqc),
                                 (result.coefficients.amplitude, reference.coefficients.amplitude),
                                 (result.coefficients.phase, reference.coefficients.phase),
                                 (result.corrected_amplitude, reference.corrected_amplitude),
                                 (result.corrected_phase, reference.corrected_phase)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nqc_gradient_includes_moving_event_and_explicit_targets():
    tilt = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([.32, -.002, .38, .012], dtype=tilt.dtype, requires_grad=True)
    def solve(tilt, targets):
        result = _nqc(tilt=tilt, targets=tuple(targets.unbind()))
        return torch.cat((result.t_peak.reshape(1), result.coefficients.amplitude, result.coefficients.phase))
    assert torch.autograd.gradcheck(solve, (tilt, targets), eps=1e-7, rtol=2e-4, atol=2e-4)


def test_unknown_matching_policy_rejected_before_assembly():
    with pytest.raises(ValueError, match="matching_method"):
        assemble_teob_merger_nqc(torch.arange(8.), torch.ones(8, 4), matching_method="unknown")

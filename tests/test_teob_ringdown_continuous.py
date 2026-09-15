"""Focused checks for the continuous TEOB ringdown attachment policy."""

import pytest
import torch

from torchwave.approximants.eob.teob_postmerger import postmerger_22
from torchwave.approximants.eob.teob_ringdown_assembly_clean import (
    assemble_teob_ringdown_22,
    natural_cubic_interpolate,
)


DTYPE = torch.float64


def _series(aux_shift=0.0, nonlinear_aux=False, aux_step=0.5):
    aux_time = torch.arange(aux_shift, aux_shift + 11.0, aux_step, dtype=DTYPE)
    delta = aux_time - 5.0
    aux_frequency = 0.30 - 0.002 * delta.square() + 0.00001 * delta.pow(3)
    aux_amplitude = 1.0 + 0.015 * delta
    aux_phase = 0.17 + 0.23 * delta + 0.003 * delta.square()
    if nonlinear_aux:
        # Deliberately make auxiliary knots disagree with the accepted full
        # series.  Continuous output before the event must still use the
        # accepted-series spline, independently of this resampling.
        aux_amplitude = aux_amplitude + 0.04 * torch.sin(1.7 * delta)
        aux_phase = aux_phase + 0.08 * torch.sin(0.9 * delta)

    full_time = torch.arange(-2.0, 11.0, 0.25, dtype=DTYPE)
    delta = full_time - 5.0
    full_amplitude = 1.0 + 0.015 * delta
    full_phase = 0.17 + 0.23 * delta + 0.003 * delta.square()
    return (full_time, full_amplitude, full_phase, aux_time, aux_amplitude,
            aux_phase, aux_frequency)


def _assemble(series, **kwargs):
    return assemble_teob_ringdown_22(
        *series,
        nu=torch.tensor(0.24, dtype=DTYPE),
        final_mass=torch.tensor(0.95, dtype=DTYPE),
        final_spin=torch.tensor(0.0, dtype=DTYPE),
        **kwargs,
    )


def test_grid_default_remains_the_grid_policy():
    series = _series()
    default = _assemble(series)
    explicit = _assemble(series, matching_method="grid", peak_time=torch.tensor(5.2, dtype=DTYPE))
    torch.testing.assert_close(default.mode, explicit.mode, rtol=0.0, atol=0.0)
    torch.testing.assert_close(default.events.attachment_time, explicit.events.attachment_time,
                               rtol=0.0, atol=0.0)
    assert int(default.events.attachment_index) == int(explicit.events.attachment_index)


def test_continuous_uses_exact_full_series_anchor_and_truthful_events():
    series = _series()
    peak = torch.tensor(5.2, dtype=DTYPE)
    result = _assemble(series, matching_method="continuous", peak_time=peak)
    full_time, full_amplitude, full_phase = series[:3]
    match = peak - 1.0
    amp = natural_cubic_interpolate(full_time, full_amplitude, match.reshape(1))[0]
    phase = natural_cubic_interpolate(full_time, full_phase, match.reshape(1))[0]
    baseline = postmerger_22(
        match.reshape(1), nu=0.24, final_mass=0.95, final_spin=0.0,
        merger_time=peak - 3.0,
    )[0]
    torch.testing.assert_close(result.events.peak_time, peak, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.events.merger_time, peak - 3.0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.events.match_time, match, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.events.attachment_time, match, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.events.match_time_final_mass, match / 0.95,
                               rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.events.attachment_time_final_mass, match / 0.95,
                               rtol=0.0, atol=0.0)
    assert result.events.attachment_time != result.joined_time[int(result.events.attachment_index)]
    expected_phase0 = torch.angle(baseline) - torch.angle(amp * torch.exp(-1j * phase))
    torch.testing.assert_close(result.ringdown_phase0, expected_phase0,
                               rtol=0.0, atol=2e-14)
    # The ansatz amplitude is retained; no multiplicative normalization to the
    # signed corrected full-series amplitude is applied.
    assert not torch.allclose(result.ringdown_mode[int(result.events.attachment_index)].abs(), amp.abs())


def test_continuous_uniform_output_splits_pre_spline_and_postmerger():
    series = _series()
    peak = torch.tensor(5.2, dtype=DTYPE)
    result = _assemble(series, matching_method="continuous", peak_time=peak,
                       output_step=torch.tensor(0.25, dtype=DTYPE))
    match = peak - 1.0
    before = result.time < match
    after = result.time > match
    assert bool(before.any()) and bool(after.any())
    pre_time = result.time[before][-1].reshape(1)
    expected_amp = natural_cubic_interpolate(series[0], series[1], pre_time)[0]
    expected_phase = natural_cubic_interpolate(series[0], series[2], pre_time)[0]
    torch.testing.assert_close(result.amplitude[before][-1], expected_amp, rtol=0.0, atol=2e-12)
    torch.testing.assert_close(result.phase[before][-1], expected_phase, rtol=0.0, atol=2e-12)
    post_time = result.time[after][0].reshape(1)
    expected_post = postmerger_22(
        post_time, nu=0.24, final_mass=0.95, final_spin=0.0,
        merger_time=peak - 3.0, phase0=result.ringdown_phase0,
    )[0]
    torch.testing.assert_close(result.mode[after][0], expected_post, rtol=0.0, atol=2e-13)


def test_continuous_anchor_is_independent_of_auxiliary_origin():
    first = _assemble(_series(0.0), matching_method="continuous",
                      peak_time=torch.tensor(5.2, dtype=DTYPE))
    shifted = _assemble(_series(0.25), matching_method="continuous",
                        peak_time=torch.tensor(5.2, dtype=DTYPE))
    torch.testing.assert_close(first.ringdown_phase0, shifted.ringdown_phase0,
                               rtol=0.0, atol=2e-13)
    torch.testing.assert_close(first.events.match_time, shifted.events.match_time,
                               rtol=0.0, atol=0.0)


def test_continuous_pre_event_output_uses_full_spline_for_nonlinear_aux_grid():
    first = _assemble(_series(0.0, nonlinear_aux=True), matching_method="continuous",
                      peak_time=torch.tensor(5.2, dtype=DTYPE),
                      output_step=torch.tensor(0.25, dtype=DTYPE))
    shifted = _assemble(_series(0.25, nonlinear_aux=True, aux_step=0.4), matching_method="continuous",
                        peak_time=torch.tensor(5.2, dtype=DTYPE),
                        output_step=torch.tensor(0.25, dtype=DTYPE))
    match = torch.tensor(4.2, dtype=DTYPE)
    first_common = first.time < match
    shifted_common = shifted.time < match
    assert bool(first_common.any()) and first_common.sum() == shifted_common.sum()
    torch.testing.assert_close(first.amplitude[first_common], shifted.amplitude[shifted_common],
                               rtol=0.0, atol=2e-12)
    torch.testing.assert_close(first.phase[first_common], shifted.phase[shifted_common],
                               rtol=0.0, atol=2e-12)
    expected = natural_cubic_interpolate(
        _series(0.0)[0], _series(0.0)[1], first.time[first_common]
    )
    torch.testing.assert_close(first.amplitude[first_common], expected, rtol=0.0, atol=2e-12)


def test_continuous_negative_anchor_preserves_signed_phase_and_mode_identity():
    series = list(_series())
    series[1] = -series[1].abs()
    series[4] = -series[4].abs()
    result = _assemble(tuple(series), matching_method="continuous",
                       peak_time=torch.tensor(5.2, dtype=DTYPE),
                       output_step=torch.tensor(0.25, dtype=DTYPE))
    torch.testing.assert_close(
        result.mode, result.amplitude * torch.exp(-1j * result.phase),
        rtol=0.0, atol=2e-13,
    )
    torch.testing.assert_close(
        result.extended_mode,
        result.extended_amplitude * torch.exp(-1j * result.extended_phase),
        rtol=0.0, atol=2e-13,
    )
    post = result.time >= result.events.match_time
    expected = postmerger_22(
        result.time[post], nu=0.24, final_mass=0.95, final_spin=0.0,
        merger_time=result.events.merger_time, phase0=result.ringdown_phase0,
    )
    torch.testing.assert_close(result.mode[post], expected, rtol=0.0, atol=3e-13)


def test_continuous_explicit_peak_does_not_fit_tied_auxiliary_frequency():
    series = list(_series())
    series[6] = torch.ones_like(series[6])
    result = _assemble(tuple(series), matching_method="continuous",
                       peak_time=torch.tensor(5.2, dtype=DTYPE))
    assert torch.isfinite(result.mode).all()
    assert torch.isnan(result.events.peak_first_derivative)
    assert torch.isnan(result.events.peak_second_derivative)
    assert torch.isnan(result.events.peak_polynomial_coefficients).all()


@pytest.mark.parametrize("output_step", [None, 0.25])
def test_continuous_gradients_match_centered_difference(output_step):
    base = list(_series())
    amplitude = base[1].clone().requires_grad_()
    phase = base[2].clone().requires_grad_()
    peak = torch.tensor(5.2, dtype=DTYPE, requires_grad=True)
    differentiable = list(base)
    differentiable[1] = amplitude
    differentiable[2] = phase

    def objective(series, event_peak):
        result = _assemble(
            tuple(series), matching_method="continuous", peak_time=event_peak,
            **({} if output_step is None else {
                "output_step": torch.tensor(output_step, dtype=DTYPE)
            }),
        )
        return result.mode.real.sum() + 0.37 * result.mode.imag.sum()

    loss = objective(differentiable, peak)
    gradients = torch.autograd.grad(loss, (peak, amplitude, phase))
    peak_direction = torch.tensor(0.7, dtype=DTYPE)
    amplitude_direction = torch.linspace(-0.4, 0.6, amplitude.numel(), dtype=DTYPE)
    phase_direction = torch.linspace(0.5, -0.3, phase.numel(), dtype=DTYPE)
    autodiff_directional = (
        gradients[0] * peak_direction
        + (gradients[1] * amplitude_direction).sum()
        + (gradients[2] * phase_direction).sum()
    )

    epsilon = 1e-5
    plus = list(base)
    minus = list(base)
    plus[1] = base[1] + epsilon * amplitude_direction
    minus[1] = base[1] - epsilon * amplitude_direction
    plus[2] = base[2] + epsilon * phase_direction
    minus[2] = base[2] - epsilon * phase_direction
    finite_difference = (
        objective(plus, torch.tensor(5.2, dtype=DTYPE) + epsilon * peak_direction)
        - objective(minus, torch.tensor(5.2, dtype=DTYPE) - epsilon * peak_direction)
    ) / (2.0 * epsilon)
    torch.testing.assert_close(autodiff_directional, finite_difference,
                               rtol=3e-5, atol=3e-6)


def test_continuous_preserves_existing_batch_support():
    series = _series()
    batched = tuple(torch.stack((value, value + (0.01 if i in (1, 2) else 0.0)))
                    for i, value in enumerate(series))
    peaks = torch.tensor([5.2, 5.25], dtype=DTYPE)
    result = assemble_teob_ringdown_22(
        *batched,
        nu=torch.tensor([0.24, 0.23], dtype=DTYPE),
        final_mass=torch.tensor([0.95, 0.96], dtype=DTYPE),
        final_spin=torch.tensor([0.0, 0.1], dtype=DTYPE),
        matching_method="continuous", peak_time=peaks,
    )
    assert result.mode.shape[0] == 2
    assert result.extension_samples.ndim == 0
    assert int(result.extension_samples) == 1500
    torch.testing.assert_close(result.events.peak_time, peaks, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.events.match_time, peaks - 1.0, rtol=0.0, atol=0.0)


def test_continuous_requires_peak_and_supported_match_domain():
    with pytest.raises(ValueError, match="requires explicit peak_time"):
        _assemble(_series(), matching_method="continuous")
    with pytest.raises(ValueError, match="outside full-series support"):
        _assemble(_series(), matching_method="continuous",
                  peak_time=torch.tensor(20.0, dtype=DTYPE))
    with pytest.raises(ValueError, match="matching_method"):
        _assemble(_series(), matching_method="nearest")


def test_support_end_none_keeps_legacy_auxiliary_knot_count():
    result = _assemble(_series(aux_step=0.5), matching_method="continuous",
                       peak_time=torch.tensor(5.2, dtype=DTYPE))
    assert int(result.extension_samples) == 1500
    torch.testing.assert_close(
        result.extended_time[-1] - result.joined_time[-1],
        torch.tensor(1500.0 * 0.5, dtype=DTYPE), rtol=0.0, atol=0.0,
    )


def test_support_end_clamps_nondivisible_final_knot():
    base = _assemble(_series(aux_step=0.5), matching_method="continuous",
                     peak_time=torch.tensor(5.2, dtype=DTYPE))
    requested = base.joined_time[-1] + torch.tensor(1.1, dtype=DTYPE)
    result = _assemble(_series(aux_step=0.5), matching_method="continuous",
                       peak_time=torch.tensor(5.2, dtype=DTYPE),
                       support_end=requested)
    assert int(result.extension_samples) == 3
    torch.testing.assert_close(result.extended_time[-1], requested,
                               rtol=0.0, atol=0.0)
    assert bool(torch.all(result.extended_time[1:] > result.extended_time[:-1]))


def test_support_end_makes_continuous_uniform_output_and_fd_spacing_independent():
    results = [
        _assemble(_series(aux_step=step), matching_method="continuous",
                  peak_time=torch.tensor(5.2, dtype=DTYPE),
                  output_step=torch.tensor(0.25, dtype=DTYPE),
                  support_end=torch.tensor(11.0, dtype=DTYPE))
        for step in (0.05, 0.1, 0.2, 0.5)
    ]
    reference = results[0]
    for result in results[1:]:
        torch.testing.assert_close(result.time, reference.time, rtol=0.0, atol=0.0)
        torch.testing.assert_close(result.mode, reference.mode, rtol=0.0, atol=0.0)
        torch.testing.assert_close(torch.fft.fft(result.mode), torch.fft.fft(reference.mode),
                                   rtol=0.0, atol=0.0)
        torch.testing.assert_close(result.extended_time[-1], reference.extended_time[-1],
                                   rtol=0.0, atol=0.0)


@pytest.mark.parametrize("bad_support", [0.0, -1.0, float("nan"), float("inf")])
def test_support_end_rejects_invalid_values(bad_support):
    with pytest.raises(ValueError, match="support_end"):
        _assemble(_series(), matching_method="continuous",
                  peak_time=torch.tensor(5.2, dtype=DTYPE),
                  support_end=torch.tensor(bad_support, dtype=DTYPE))


def test_support_end_must_follow_each_joined_endpoint():
    series = _series()
    joined_end = _assemble(series, matching_method="continuous",
                           peak_time=torch.tensor(5.2, dtype=DTYPE)).joined_time[-1]
    with pytest.raises(ValueError, match="later than the joined endpoint"):
        _assemble(series, matching_method="continuous",
                  peak_time=torch.tensor(5.2, dtype=DTYPE), support_end=joined_end)


def test_support_end_rejects_incompatible_batched_output_lengths():
    series = _series()
    batched = tuple(torch.stack((value, value)) for value in series)
    with pytest.raises(ValueError, match="same extended/output support length"):
        assemble_teob_ringdown_22(
            *batched,
            nu=torch.tensor([0.24, 0.24], dtype=DTYPE),
            final_mass=torch.tensor([0.95, 0.95], dtype=DTYPE),
            final_spin=torch.tensor([0.0, 0.0], dtype=DTYPE),
            matching_method="continuous",
            peak_time=torch.tensor([5.2, 5.2], dtype=DTYPE),
            output_step=torch.tensor(0.25, dtype=DTYPE),
            support_end=torch.tensor([11.0, 11.25], dtype=DTYPE),
        )

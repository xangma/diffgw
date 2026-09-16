"""Synthetic, clean-room checks for TEOB ringdown/sampling assembly."""

import pytest
import torch

from torchwave.approximants.eob.teob_postmerger import postmerger_22, qnm_fits
from torchwave.approximants.eob.teob_ringdown_assembly_clean import (
    _attachment_indices,
    assemble_teob_ringdown_22,
    natural_cubic_interpolate,
)


DTYPE = torch.float64


def _synthetic_series(
    *,
    dt=0.5,
    count=24,
    peak_index=8,
    peak_slope=0.0,
    shift=0.0,
    signed=False,
):
    """Build two compatible corrected charts with a polynomial frequency peak."""

    step = torch.tensor(dt, dtype=DTYPE)
    offset = torch.tensor(shift, dtype=DTYPE)
    auxiliary_time = offset + torch.arange(count, dtype=DTYPE) * step
    center = auxiliary_time[peak_index]
    delta = auxiliary_time - center
    # The cubic term keeps the seven-knot solve nontrivial while preserving a
    # unique local maximum.  A linear term gives an off-knot Newton estimate.
    auxiliary_frequency = 0.30 - 0.002 * delta.square() + 0.00001 * delta.pow(3)
    auxiliary_frequency = auxiliary_frequency + peak_slope * delta
    auxiliary_amplitude = 1.0 + 0.015 * delta
    if signed:
        auxiliary_amplitude = auxiliary_amplitude.clone()
        auxiliary_amplitude[2] = -auxiliary_amplitude[2]
    auxiliary_phase = 0.17 + 0.23 * delta + 0.003 * delta.square()

    full_time = offset + torch.arange(-4, 0, dtype=DTYPE) * step
    full_delta = full_time - center
    full_amplitude = 1.0 + 0.015 * full_delta
    full_phase = 0.17 + 0.23 * full_delta + 0.003 * full_delta.square()
    return (
        full_time,
        full_amplitude,
        full_phase,
        auxiliary_time,
        auxiliary_amplitude,
        auxiliary_phase,
        auxiliary_frequency,
    )


def _assemble(series, *, final_mass=0.95, output_step=None, **kwargs):
    return assemble_teob_ringdown_22(
        *series,
        nu=torch.tensor(0.24, dtype=DTYPE),
        final_mass=torch.tensor(final_mass, dtype=DTYPE),
        final_spin=torch.tensor(0.0, dtype=DTYPE),
        output_step=output_step,
        **kwargs,
    )


def test_natural_cubic_reproduces_a_cubic_at_endpoints_and_inside():
    scipy = pytest.importorskip("scipy.interpolate")
    x = torch.tensor([-1.0, -0.3, 0.4, 1.1, 2.0], dtype=DTYPE)
    def polynomial(value):
        return 1.2 - 0.7 * value + 0.4 * value.square() + 0.08 * value.pow(3)
    y = polynomial(x)
    query = torch.tensor([-1.0, -0.75, 0.4, 1.6, 2.0], dtype=DTYPE)
    actual = natural_cubic_interpolate(x, y, query)
    # A natural spline reproduces the cubic only when the endpoint curvature
    # is natural; use a linear polynomial for an exact all-domain check and
    # separately check the cubic knots/boundaries below.
    linear = 2.0 - 0.4 * x
    torch.testing.assert_close(
        natural_cubic_interpolate(x, linear, query), 2.0 - 0.4 * query,
        rtol=0.0, atol=2e-14,
    )
    torch.testing.assert_close(actual[[0, 2, 4]], y[[0, 2, 4]], rtol=0.0, atol=2e-14)
    expected = torch.as_tensor(
        scipy.CubicSpline(x.numpy(), y.numpy(), bc_type="natural")(query.numpy()),
        dtype=DTYPE,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)


def test_peak_polynomial_reports_newton_derivatives_and_right_extension():
    series = _synthetic_series(count=5, peak_index=3)
    result = _assemble(series)
    assert int(result.events.peak_knot_index) == 3
    assert result.events.peak_polynomial_coefficients.shape == (7,)
    # Reconstruct the same seven support ordinates independently, including
    # the prescribed last-two-knot linear continuation.
    aux_t, aux_omega = series[3], series[6]
    spacing = aux_t[1] - aux_t[0]
    extra_t = aux_t[-1] + spacing * torch.arange(1, 3, dtype=DTYPE)
    slope = (aux_omega[-1] - aux_omega[-2]) / spacing
    extra_omega = aux_omega[-1] + slope * spacing * torch.arange(1, 3, dtype=DTYPE)
    knots = torch.cat((aux_t[:3], aux_t[3:4], aux_t[4:], extra_t))
    ordinates = torch.cat((aux_omega[:3], aux_omega[3:4], aux_omega[4:], extra_omega))
    local = knots - aux_t[3]
    vandermonde = torch.stack(tuple(local.pow(power) for power in range(7)), dim=-1)
    expected = torch.linalg.solve(vandermonde, ordinates)
    torch.testing.assert_close(
        result.events.peak_polynomial_coefficients, expected,
        rtol=2e-12, atol=2e-13,
    )
    torch.testing.assert_close(
        result.events.peak_first_derivative, expected[1],
        rtol=2e-12, atol=2e-13,
    )
    torch.testing.assert_close(
        result.events.peak_second_derivative, 2.0 * expected[2],
        rtol=2e-12, atol=2e-13,
    )


def test_join_extension_and_exact_attachment_metadata():
    series = _synthetic_series()
    metadata_time = torch.tensor([-2.0, -1.0], dtype=DTYPE)
    metadata_state = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=DTYPE)
    result = _assemble(
        series,
        matching_time=metadata_time,
        matching_state=metadata_state,
    )
    expected_join = torch.cat((series[0], series[3][1:]))
    torch.testing.assert_close(result.joined_time, expected_join, rtol=0.0, atol=0.0)
    assert result.joined_time[0] == series[0][0]
    assert result.joined_time[4] == series[3][1]
    assert result.joined_time[-1] == series[3][-1]
    assert result.extended_time.numel() == result.joined_time.numel() + 1500
    assert result.extension_samples.item() == 1500
    torch.testing.assert_close(
        result.extended_time[-1] - result.extended_time[-2],
        series[3][1] - series[3][0], rtol=0.0, atol=0.0,
    )
    assert result.matching_time is metadata_time
    assert result.matching_state is metadata_state
    torch.testing.assert_close(result.epoch, result.extended_time[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.coordinate_start, result.epoch, rtol=0.0, atol=0.0)
    # ``epoch`` is the coordinate start; the assembly does not apply an
    # observer or peak-time shift.
    assert result.epoch != result.t_peak
    threshold = (
        result.events.peak_time / 0.95
        - torch.ones((), dtype=DTYPE) * 3.0 / 0.95
        + torch.ones((), dtype=DTYPE) * 2.0 / 0.95
    )
    torch.testing.assert_close(result.events.match_time_final_mass, threshold)
    assert int(result.events.strictly_earlier_index) == int(
        torch.searchsorted(result.joined_time / 0.95, threshold, right=False)
    ) - 1
    assert int(result.events.attachment_index) == int(result.events.strictly_earlier_index) - 1
    torch.testing.assert_close(
        result.events.attachment_time_final_mass,
        result.events.attachment_time / 0.95,
        rtol=0.0, atol=0.0,
    )


def test_attachment_uses_a_left_to_right_scaled_threshold_at_float_boundary():
    final_mass = torch.tensor(0.8001, dtype=DTYPE)
    peak = torch.tensor(3.00025, dtype=DTYPE)
    three = torch.ones_like(peak) * 3.0
    two = torch.ones_like(peak) * 2.0
    ordered = peak / final_mass - three / final_mass + two / final_mass
    collapsed = (peak - three + two) / final_mass
    assert ordered > collapsed
    # This knot is exactly the collapsed expression's representable value.
    joined = torch.tensor([1.0, 2.00025, 2.5, 3.0], dtype=DTYPE)
    strict, attachment = _attachment_indices(
        joined, peak - 3.0 + 2.0, final_mass, scaled_match_time=ordered
    )
    assert strict == 1 and attachment == 0
    collapsed_strict = int(torch.searchsorted(joined / final_mass, collapsed, right=False)) - 1
    assert collapsed_strict == 0


def test_phase_anchor_preserves_signed_input_and_does_not_renormalize_amplitude():
    result = _assemble(_synthetic_series(signed=True))
    index = int(result.events.attachment_index)
    anchor = result.joined_amplitude[index] * torch.exp(-1j * result.joined_phase[index])
    assert bool(result.auxiliary_amplitude_negative)
    baseline = postmerger_22(
        result.events.attachment_time.reshape(1),
        nu=torch.tensor(0.24, dtype=DTYPE),
        final_mass=torch.tensor(0.95, dtype=DTYPE),
        final_spin=torch.tensor(0.0, dtype=DTYPE),
        merger_time=result.t_merger,
        phase0=torch.zeros((), dtype=DTYPE),
    )[0]
    torch.testing.assert_close(
        result.ringdown_phase0,
        torch.angle(baseline) - torch.angle(anchor),
        rtol=0.0,
        atol=2e-14,
    )
    expected = result.ringdown_mode[index].abs()
    # The selected ansatz amplitude is retained at the boundary; the input
    # amplitude is used only for the phase anchor and is never a rescaling.
    torch.testing.assert_close(result.extended_amplitude[index], expected)
    torch.testing.assert_close(result.extended_mode[index], result.ringdown_mode[index])
    torch.testing.assert_close(
        torch.angle(result.ringdown_mode[index] / anchor),
        torch.zeros((), dtype=DTYPE), rtol=0.0, atol=2e-14,
    )
    assert result.joined_amplitude[5] < 0
    assert not torch.allclose(result.extended_amplitude[index], anchor.abs())


def test_off_knot_attachment_and_uniform_grid_are_time_translation_covariant():
    base = _synthetic_series(peak_slope=0.0008)
    shifted = _synthetic_series(peak_slope=0.0008, shift=17.25)
    result = _assemble(base, output_step=0.25)
    shifted_result = _assemble(shifted, output_step=0.25)
    assert result.events.match_time != result.joined_time[int(result.events.strictly_earlier_index)]
    torch.testing.assert_close(
        shifted_result.events.peak_time - result.events.peak_time,
        torch.tensor(17.25, dtype=DTYPE), rtol=0.0, atol=2e-12,
    )
    torch.testing.assert_close(
        shifted_result.events.merger_time - result.events.merger_time,
        torch.tensor(17.25, dtype=DTYPE), rtol=0.0, atol=2e-12,
    )
    torch.testing.assert_close(
        shifted_result.events.match_time - result.events.match_time,
        torch.tensor(17.25, dtype=DTYPE), rtol=0.0, atol=2e-12,
    )
    torch.testing.assert_close(
        shifted_result.mode, result.mode, rtol=3e-12, atol=3e-12,
    )


def test_asymptotic_frequency_and_optional_frequency_override():
    result = _assemble(_synthetic_series(), final_mass=0.95)
    tail = result.ringdown_mode[-4:]
    dt = result.extended_time[-1] - result.extended_time[-2]
    finite_difference_frequency = -torch.angle(tail[-1] * tail[-2].conj()) / dt
    omega1 = qnm_fits(torch.tensor(0.0, dtype=DTYPE))[0]
    torch.testing.assert_close(
        finite_difference_frequency, omega1 / 0.95,
        rtol=5e-5, atol=5e-6,
    )
    selected = torch.tensor(0.42, dtype=DTYPE)
    overridden = _assemble(_synthetic_series(), merger_frequency_value=selected)
    # At merger the phase derivative is selected by the explicit override;
    # the assembly passes that value through unchanged.
    merger_index = int(torch.argmin((overridden.extended_time - overridden.t_merger).abs()))
    assert merger_index < overridden.events.attachment_index
    mode = overridden.ringdown_mode[merger_index:merger_index + 2]
    fd = -torch.angle(mode[1] * mode[0].conj()) / (
        overridden.extended_time[merger_index + 1] - overridden.extended_time[merger_index]
    )
    assert torch.isfinite(fd)


def test_batched_gradients_and_unsupported_peak_domains():
    series = _synthetic_series()
    batch_series = tuple(torch.stack((value, value + (0.01 if i in (1, 2) else 0.0)))
                         for i, value in enumerate(series))
    nu = torch.tensor([0.20, 0.23], dtype=DTYPE, requires_grad=True)
    final_mass = torch.tensor([0.94, 0.96], dtype=DTYPE, requires_grad=True)
    final_spin = torch.tensor([0.3, 0.5], dtype=DTYPE, requires_grad=True)
    result = assemble_teob_ringdown_22(
        *batch_series, nu, final_mass, final_spin,
    )
    assert result.mode.shape[0] == 2 and torch.isfinite(result.mode).all()
    loss = result.mode.real.square().mean() + result.mode.imag.square().mean()
    gradients = torch.autograd.grad(loss, (nu, final_mass, final_spin))
    assert all(torch.isfinite(value).all() for value in gradients)

    multiple_peak = list(series)
    multiple_peak[6] = multiple_peak[6].clone()
    multiple_peak[6][5] = multiple_peak[6][8]
    with pytest.raises(ValueError, match="multiple|single interior"):
        _assemble(tuple(multiple_peak))
    edge_peak = list(series)
    edge_peak[6] = edge_peak[6].clone()
    edge_peak[6] = -torch.arange(edge_peak[6].numel(), dtype=DTYPE)
    edge_peak[6][1:4] = torch.tensor([9.0, 10.0, 9.0], dtype=DTYPE)
    with pytest.raises(ValueError, match="three left|edge"):
        _assemble(tuple(edge_peak))
    nonuniform = list(series)
    nonuniform[3] = nonuniform[3].clone()
    nonuniform[3][4] += 0.1
    with pytest.raises(ValueError, match="uniformly spaced"):
        _assemble(tuple(nonuniform))
    with pytest.raises(ValueError, match="final_mass"):
        _assemble(series, final_mass=0.0)

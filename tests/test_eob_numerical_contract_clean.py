"""Bounded Torch-only checks for the opt-in SEOBNRv4 numerical contract."""

import torch
import pytest

from torchwave.approximants.eob.nqc import nqc_22_basis, seobnrv4_nqc_targets
from torchwave.approximants import kerr_qnm_table_convention as qnm_table
from torchwave.approximants.eob.seobnrv4_amplitude22 import _rho22
from torchwave.approximants.eob.numerical_attachment import (
    _full_orbital_peak,
    attach_seobnrv4_22_numerical_compatibility,
    calibrate_nqc_22_full_series,
    natural_cubic_spline,
    seobnrv4_internal_step,
)


def _dense_natural_second_derivatives(knots, values):
    """Independent dense natural-cubic solve used only by these tests."""
    h = knots[1:] - knots[:-1]
    interior_count = knots.numel() - 2
    matrix = torch.zeros((interior_count, interior_count), dtype=knots.dtype)
    rhs = torch.empty(interior_count, dtype=knots.dtype)
    for row in range(interior_count):
        matrix[row, row] = 2 * (h[row] + h[row + 1])
        rhs[row] = 6 * ((values[row + 2] - values[row + 1]) / h[row + 1]
                        - (values[row + 1] - values[row]) / h[row])
        if row:
            matrix[row, row - 1] = h[row]
        if row + 1 < interior_count:
            matrix[row, row + 1] = h[row + 1]
    second = torch.zeros_like(values)
    second[1:-1] = torch.linalg.solve(matrix, rhs)
    return second


def test_diagnostic_local_nqc_phase_matrix_recovers_cubic_trajectory():
    from dev.native_rom.eob.torch.adaptive_driver_clean import (
        _diagnostic_local_nqc_phase_matrix,
    )

    time = torch.linspace(-0.5, 0.5, 65, dtype=torch.float64)
    state = torch.zeros((time.numel(), 4), dtype=time.dtype)
    state[:, 0] = 10.0
    state[:, 2] = 0.3 + 0.05*time
    omega = torch.full_like(time, 0.1)
    matrix = _diagnostic_local_nqc_phase_matrix(
        time, state, omega, torch.tensor(0.0, dtype=time.dtype))
    expected = torch.tensor([[0.05, 0.0135], [0.0, 0.0045]], dtype=time.dtype)
    torch.testing.assert_close(matrix, expected, rtol=1e-10, atol=1e-12)


def test_natural_spline_matches_dense_unequal_knot_system_and_endpoints():
    knots = torch.tensor([0.0, 0.13, 0.8, 1.9, 2.05, 3.7], dtype=torch.float64)
    values = torch.tensor([1.2, -0.4, 0.9, 2.1, -0.2, 1.7], dtype=torch.float64)
    spline = natural_cubic_spline(knots, values)
    expected_second = _dense_natural_second_derivatives(knots, values)

    # Evaluating at every knot also checks the interval polynomial assembly.
    torch.testing.assert_close(spline(knots), values, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(spline(knots, derivative=2), expected_second,
                               rtol=1e-12, atol=1e-12)
    assert spline(knots[[0, -1]], derivative=2).abs().max().item() == 0.0

    query = torch.tensor([0.07, 0.4, 1.2, 1.97, 3.2], dtype=torch.float64)
    interval = (torch.searchsorted(knots, query, right=True) - 1).clamp(0, knots.numel() - 2)
    h = knots[1:] - knots[:-1]
    dx = query - knots[interval]
    a = values[interval]
    b = ((values[interval + 1] - values[interval]) / h[interval]
         - h[interval] * (2 * expected_second[interval] + expected_second[interval + 1]) / 6)
    c = expected_second[interval] / 2
    d = (expected_second[interval + 1] - expected_second[interval]) / (6 * h[interval])
    expected = a + dx * (b + dx * (c + dx * d))
    torch.testing.assert_close(spline(query), expected, rtol=1e-12, atol=1e-12)


def test_natural_spline_query_gradient_matches_finite_difference():
    knots = torch.tensor([0.0, 0.2, 0.75, 1.8, 3.0], dtype=torch.float64)
    values = (torch.sin(knots) + 0.08 * knots.square()).requires_grad_()
    query = torch.tensor([0.31, 1.17, 2.4], dtype=torch.float64, requires_grad=True)
    spline = natural_cubic_spline(knots, values)
    loss = spline(query).sum()
    (gradient,) = torch.autograd.grad(loss, query)
    torch.testing.assert_close(gradient, spline(query, derivative=1),
                               rtol=1e-12, atol=1e-12)
    epsilon = 1e-6
    finite_difference = (spline(query.detach() + epsilon)
                         - spline(query.detach() - epsilon)) / (2 * epsilon)
    torch.testing.assert_close(finite_difference, gradient.detach(),
                               rtol=3e-7, atol=3e-9)


def test_natural_spline_extrapolation_uses_endpoint_cubic_interval():
    knots = torch.tensor([0.0, 0.3, 1.4, 2.0], dtype=torch.float64)
    values = torch.tensor([1.0, 0.2, 1.1, 0.8], dtype=torch.float64)
    spline = natural_cubic_spline(knots, values)
    query = torch.tensor([-0.4, 2.4], dtype=torch.float64)
    extrapolated = spline(query, extrapolate=True)
    left_expected = spline.a[0, 0] + (-0.4) * (
        spline.b[0, 0] + (-0.4) * (spline.c[0, 0] + (-0.4) * spline.d[0, 0]))
    right_dx = 2.4 - knots[-2]
    right_expected = spline.a[-1, 0] + right_dx * (
        spline.b[-1, 0] + right_dx * (spline.c[-1, 0] + right_dx * spline.d[-1, 0]))
    torch.testing.assert_close(extrapolated, torch.stack((left_expected, right_expected)),
                               rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(spline(query.clamp(knots[0], knots[-1])),
                              torch.tensor([values[0], values[-1]]),
                               rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError, match="outside"):
        spline(query)


def test_full_series_nqc_scalar_query_keeps_scalar_shape():
    series = _synthetic_attachment_series()
    nqc = calibrate_nqc_22_full_series(
        series['time'], series['state'][..., 0], series['state'][..., 2],
        series['orbital_frequency'], series['mode'].abs(), series['frequency'],
        series['match'], nu=series['nu'], chi=series['chi'], base_mode=series['mode'])
    scalar = nqc.evaluate(series['match'])
    vector = nqc.evaluate(series['match'].reshape(1))
    assert scalar.ndim == 0
    torch.testing.assert_close(scalar, vector[0], rtol=1e-13, atol=1e-13)


def test_internal_step_is_power_of_two_bounded():
    assert seobnrv4_internal_step(torch.tensor(0.5, dtype=torch.float64)).item() == 0.015625
    assert seobnrv4_internal_step(torch.tensor(0.02, dtype=torch.float64)).item() == 0.02
    output = torch.tensor([0.021, 0.5, 1.1], dtype=torch.float64)
    step = seobnrv4_internal_step(output)
    assert bool((step > 0).all())
    assert bool((step <= 0.02).all())
    assert bool((step / output * 2 ** torch.ceil(torch.log2(50 * output))).sub(1).abs().max() < 1e-14)


def test_compatibility_path_retains_pan_cubic_kerr_spin_term_explicitly():
    v = torch.tensor(0.31, dtype=torch.float64)
    nu = torch.tensor(0.21, dtype=torch.float64)
    delta = torch.tensor(0.18, dtype=torch.float64)
    chi_s = torch.tensor(0.24, dtype=torch.float64)
    chi_a = torch.tensor(-0.11, dtype=torch.float64)
    paper = _rho22(v, nu, delta, chi_s, chi_a)
    compatibility = _rho22(v, nu, delta, chi_s, chi_a, retain_pan_cubic_spin=True)
    kerr = (1 - 2 * nu) * chi_s + delta * chi_a
    expected = kerr.pow(3) / 3 * v.pow(7)
    torch.testing.assert_close(compatibility - paper, expected,
                               rtol=1e-13, atol=1e-15)


def test_qnm_table_inference_first_cache_retains_spin_gradient():
    qnm_table._coefficients.cache_clear()
    with torch.inference_mode():
        first = qnm_table.signed_qnm_220_table_convention(
            torch.tensor(0.23, dtype=torch.float64)
        )[0]
    assert torch.isfinite(first)
    spin = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    value = qnm_table.signed_qnm_220_table_convention(spin)[0]
    (gradient,) = torch.autograd.grad(value, spin)
    epsilon = 1.0e-6
    finite_difference = (
        qnm_table.signed_qnm_220_table_convention(spin.detach() + epsilon)[0]
        - qnm_table.signed_qnm_220_table_convention(spin.detach() - epsilon)[0]
    ) / (2 * epsilon)
    torch.testing.assert_close(gradient, finite_difference, rtol=2e-5, atol=2e-8)


def _synthetic_attachment_series():
    dtype = torch.float64
    dt = 0.02
    time = torch.arange(0.0, 30.0001, dt, dtype=dtype)
    match = torch.tensor(18.0, dtype=dtype)
    nu = torch.tensor(0.22, dtype=dtype)
    chi = torch.tensor(0.1, dtype=dtype)
    target = seobnrv4_nqc_targets(nu, chi)
    delta = time - match
    amplitude = target.amplitude + 1.0e-4 * delta.square() + 5.0e-6 * delta.pow(4)
    frequency = (target.frequency + target.frequency_derivative * delta
                 + 1.0e-6 * delta.square())
    phase = (2.7 + target.frequency * delta
             + 0.5 * target.frequency_derivative * delta.square()
             + (1.0e-6 / 3) * delta.pow(3))
    mode = amplitude * torch.exp(-1j * phase)
    radius = 8.0 - 0.025 * time + 0.00015 * time.square()
    radial_momentum = (-0.02 - 0.0005 * time + 1.0e-5 * time.square()
                       + 0.001 * torch.sin(0.37 * time))
    orbital_frequency = (0.08 + 0.002 * time + 0.0002 * time.square()
                         - 4.0e-6 * time.pow(3))
    state = torch.stack((radius, 0.1 * time, radial_momentum, 3.3 + 0.01 * time), dim=-1)
    output_time = torch.arange(0.0, 30.0001, 0.5, dtype=dtype)
    sigma = torch.tensor(0.5 - 0.08j, dtype=torch.complex128)
    return dict(time=time, state=state, orbital_frequency=orbital_frequency,
                mode=mode, frequency=frequency, nu=nu, chi=chi, sigma=sigma,
                output_time=output_time, match=match, phase=phase)


def _attach(series, match):
    return attach_seobnrv4_22_numerical_compatibility(
        series['time'], series['state'], series['orbital_frequency'], series['mode'],
        series['frequency'], series['nu'], series['chi'], series['sigma'],
        series['output_time'], match_time=match)


def test_attachment_uses_floor_sample_and_corrected_unwrapped_phase():
    series = _synthetic_attachment_series()
    exact = _attach(series, series['match'])
    between = _attach(series, series['match'] + 0.003)
    assert exact.boundary_index.item() == 900
    assert exact.boundary_time.item() == 18.0
    assert between.boundary_index.item() == exact.boundary_index.item()
    assert between.boundary_time.item() == exact.boundary_time.item()
    assert bool((between.boundary_time < series['match'] + 0.003).all())
    assert exact.internal_step.item() == 0.015625

    index = exact.boundary_index
    previous = index - 1
    corrected_phase = exact.nqc.sampled_corrected_phase
    expected_frequency = ((corrected_phase[index] - corrected_phase[previous])
                          / (series['time'][index] - series['time'][previous]))
    torch.testing.assert_close(exact.boundary_frequency, expected_frequency,
                               rtol=1e-13, atol=1e-13)
    expected_boundary_mode = exact.boundary_amplitude * torch.exp(-1j * exact.boundary_phase)
    torch.testing.assert_close(exact.corrected_inspiral[index], expected_boundary_mode,
                               rtol=1e-12, atol=1e-12)
    output_index = int((exact.boundary_time / 0.5).item())
    torch.testing.assert_close(exact.mode[output_index], expected_boundary_mode,
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(torch.diff(exact.nqc.sampled_waveform_phase),
                               torch.diff(series['phase']), rtol=2e-12, atol=2e-12)


def test_full_series_nqc_match_time_gradient_agrees_with_finite_difference():
    series = _synthetic_attachment_series()
    # Stay inside one spline interval: the natural spline is C2 at a knot,
    # while the derivative of the matching system can change with its third
    # derivative there.
    match = torch.tensor(17.31, dtype=torch.float64, requires_grad=True)

    def evaluate(coordinate):
        nqc = calibrate_nqc_22_full_series(
            series['time'], series['state'][..., 0], series['state'][..., 2],
            series['orbital_frequency'], series['mode'].abs(), series['frequency'],
            coordinate, nu=series['nu'], chi=series['chi'], base_mode=series['mode'])
        return nqc.evaluate(coordinate).real.squeeze()

    value = evaluate(match)
    (gradient,) = torch.autograd.grad(value, match)
    epsilon = 1e-5
    finite_difference = (evaluate(match.detach() + epsilon)
                         - evaluate(match.detach() - epsilon)) / (2 * epsilon)
    torch.testing.assert_close(gradient, finite_difference, rtol=3e-6, atol=3e-8)


def test_full_spline_orbital_peak_uses_first_nonincreasing_sample_interval():
    time = torch.tensor([0.0, 0.4, 1.1, 1.9, 2.6, 3.8, 5.1], dtype=torch.float64)
    frequency = 1.0 - (time - 2.2).square()
    peak = _full_orbital_peak(time, frequency)
    assert peak.sample_index.item() == 4
    assert 1.9 <= peak.time.item() <= 2.6
    assert peak.curvature.item() < 0


def test_shared_knots_batched_scalars_match_scipy_and_keep_channel_shape():
    scipy_interpolate = pytest.importorskip("scipy.interpolate")
    knots = torch.tensor([0.0, 0.13, 0.8, 1.9, 2.05, 3.7], dtype=torch.float64)
    values = torch.stack((torch.sin(knots) + 0.1 * knots.square(),
                          torch.cos(knots) - 0.07 * knots), dim=0)
    query = torch.tensor([0.07, 0.4, 1.2, 1.97, 3.2], dtype=torch.float64)

    scalar_spline = natural_cubic_spline(knots, values, scalar_output=True)
    assert scalar_spline(query).shape == (2, query.numel())
    for order in (0, 1, 2):
        expected = scipy_interpolate.CubicSpline(
            knots.numpy(), values.numpy(), axis=-1, bc_type="natural"
        )(query.numpy(), nu=order)
        torch.testing.assert_close(
            scalar_spline(query, derivative=order),
            torch.as_tensor(expected), rtol=2e-12, atol=2e-12,
        )

    channel_spline = natural_cubic_spline(
        knots, values.unsqueeze(-1), scalar_output=False,
    )
    assert channel_spline(query).shape == (2, query.numel(), 1)
    torch.testing.assert_close(channel_spline(query)[..., 0], scalar_spline(query),
                               rtol=1e-13, atol=1e-13)

    parameters = values.clone().requires_grad_()
    loss = natural_cubic_spline(knots, parameters, scalar_output=True)(query).square().sum()
    (gradient,) = torch.autograd.grad(loss, parameters)
    assert gradient.shape == parameters.shape
    assert bool(torch.isfinite(gradient).all())


def test_full_series_phase_rhs_uses_waveform_phase_spline_derivatives():
    series = _synthetic_attachment_series()
    phase = series["phase"] + 0.004 * torch.sin(0.37 * series["time"])
    targets = seobnrv4_nqc_targets(series["nu"], series["chi"])
    nqc = calibrate_nqc_22_full_series(
        series["time"], series["state"][..., 0], series["state"][..., 2],
        series["orbital_frequency"], series["mode"].abs(),
        # Deliberately perturb the analytic-frequency samples.  They must not
        # replace derivatives of the supplied unwrapped waveform phase.
        series["frequency"] + 0.01 * torch.cos(0.23 * series["time"]),
        series["match"], targets, waveform_phase=phase,
    )
    _, p = nqc_22_basis(series["state"][..., 0], series["state"][..., 2],
                         series["orbital_frequency"])
    phase_spline = natural_cubic_spline(series["time"], phase)
    phase_basis_spline = natural_cubic_spline(series["time"], p)
    first = phase_spline(series["match"], derivative=1)
    second = phase_spline(series["match"], derivative=2)
    matrix = torch.stack((
        phase_basis_spline(series["match"], derivative=1),
        phase_basis_spline(series["match"], derivative=2),
    ), dim=-2)
    rhs = torch.stack((first - targets.frequency,
                       second - targets.frequency_derivative), dim=-1)
    expected = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(nqc.coefficients.phase, expected,
                               rtol=2e-11, atol=2e-12)


def test_full_spline_orbital_peak_exact_zero_product_moves_left():
    # The natural spline has an exactly zero derivative at the first midpoint.
    # Equality must update the left endpoint, so repeated equality updates
    # converge to the right edge of this preceding interval.
    time = torch.arange(4.0, dtype=torch.float64)
    frequency = -(time - 1.5).square()
    peak = _full_orbital_peak(time, frequency)
    assert peak.sample_index.item() == 2
    assert 1.99 < peak.time.item() < 2.0


def test_attachment_rejects_floor_without_backward_phase_sample():
    series = _synthetic_attachment_series()
    with pytest.raises(ValueError, match="preceding phase sample"):
        _attach(series, torch.tensor(0.001, dtype=torch.float64))


def test_ringdown_support_count_is_applied_before_irregular_output_selection():
    series = _synthetic_attachment_series()
    series["sigma"] = torch.tensor(0.5 - 0.83j, dtype=torch.complex128)
    # The first output interval gives internal dt_high=.015625; later queries
    # intentionally are irregular so support cannot be inferred from their
    # spacing or from the coarse input trajectory's .02 spacing.
    series["output_time"] = torch.tensor([18.0, 18.5, 30.043, 30.05], dtype=torch.float64)
    attached = _attach(series, series["match"])
    assert attached.internal_step.item() == 0.015625
    expected_count = int(torch.floor(torch.tensor(10.0 / (0.83 * 0.015625))).item())
    assert expected_count == 771
    assert attached.mode[2].abs().item() > 0
    assert attached.mode[3].abs().item() == 0

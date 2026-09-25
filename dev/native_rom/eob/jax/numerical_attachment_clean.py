"""Development-only scalar numerical attachment conventions for native v4."""

from dataclasses import dataclass

from jax import lax
import jax.numpy as jnp

from jaxwave.approximants.eob.events import OrbitalFrequencyPeak
from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline
from jaxwave.approximants.eob.nqc import (
    NQCCalibration,
    NQCCoefficients,
    NQCMatchingTargets,
    nqc_22_basis,
    nqc_22_factor,
    seobnrv4_nqc_targets,
    seobnrv4_nqc_time_offset_numerical_compatibility,
)
from jaxwave.approximants.eob.merger_ringdown import (
    merger_ringdown_22,
    seobnrv4_merger_shape,
    seobnrv4_peak_preserving_shape,
)


def unwrap_mode_phase(mode):
    """Continuous positive phase of ``h=A*exp(-i*phi)`` on one time series."""
    mode = jnp.asarray(mode)
    if mode.ndim != 1 or len(mode) < 1 or not bool(jnp.all(jnp.isfinite(mode))):
        raise ValueError("mode must be a finite, nonempty scalar series")
    return _unwrap_mode_phase_unchecked(mode)


def _unwrap_mode_phase_unchecked(mode):
    first = -jnp.angle(mode[:1])
    increments = -jnp.angle(mode[1:] * jnp.conj(mode[:-1]))
    return jnp.concatenate((first, first + jnp.cumsum(increments)))


@dataclass(frozen=True)
class FullSeriesNQC:
    """Scalar full-series NQC calibration and its sampled intermediate values."""

    calibration: NQCCalibration
    amplitude_spline: object
    product_spline: object
    phase_basis_spline: object
    frequency_spline: object
    waveform_phase_spline: object
    sampled_amplitude: object
    sampled_products: object
    sampled_phase_basis: object
    sampled_waveform_phase: object
    sampled_corrected_amplitude: object
    sampled_corrected_phase: object
    amplitude_matching_matrix: object
    phase_matching_matrix: object

    @property
    def coefficients(self):
        return self.calibration.coefficients

    def evaluate(self, query):
        amplitude = self.amplitude_spline(query)
        products = self.product_spline(query)
        phase_basis = self.phase_basis_spline(query)
        phase = self.waveform_phase_spline(query)
        a, b = self.coefficients
        corrected_amplitude = amplitude + jnp.sum(products * a, axis=-1)
        corrected_phase = phase - jnp.sum(phase_basis * b, axis=-1)
        return corrected_amplitude * jnp.exp(-1j * corrected_phase)


@dataclass(frozen=True)
class NumericalAttachedMode22:
    """Development-only scalar numerical inspiral and ringdown result."""

    mode: object
    corrected_inspiral: object
    peak: OrbitalFrequencyPeak
    match_time: object
    match_phase: object
    calibration: NQCCalibration
    boundary_time: object
    boundary_index: object
    boundary_phase: object
    boundary_frequency: object
    boundary_amplitude: object
    internal_step: object
    nqc: FullSeriesNQC


def _solve_checked(matrix, rhs, max_condition, name):
    """Use the same row/column equilibration and failure gate as Torch."""
    if not bool(jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(rhs))):
        raise ValueError(f"{name} contains nonfinite values")
    row_scale = jnp.linalg.norm(matrix, axis=-1)
    if bool(jnp.any(row_scale == 0)):
        raise ValueError(f"{name} is rank deficient (zero matching row)")
    scaled = matrix / row_scale[:, None]
    column_scale = jnp.linalg.norm(scaled, axis=-2)
    if bool(jnp.any(column_scale == 0)):
        raise ValueError(f"{name} is rank deficient (zero basis column)")
    scaled = scaled / column_scale[None, :]
    singular = jnp.linalg.svd(scaled, compute_uv=False)
    condition = singular[0] / singular[-1]
    numerical_limit = 1 / (jnp.finfo(matrix.dtype).eps * max(matrix.shape))
    if not bool(jnp.isfinite(condition)) or float(condition) > min(max_condition, numerical_limit):
        raise ValueError(f"{name} is rank deficient or ill-conditioned")
    solution = jnp.linalg.solve(scaled, rhs / row_scale) / column_scale
    residual = matrix @ solution - rhs
    return solution, condition, residual


def _solve_traceable(matrix, rhs):
    """Equilibrated solve with no data-dependent Python branches."""
    row_scale = jnp.linalg.norm(matrix, axis=-1)
    scaled = matrix / row_scale[:, None]
    column_scale = jnp.linalg.norm(scaled, axis=-2)
    scaled = scaled / column_scale[None, :]
    singular = jnp.linalg.svd(scaled, compute_uv=False)
    solution = jnp.linalg.solve(scaled, rhs / row_scale) / column_scale
    return solution, singular[0] / singular[-1], matrix @ solution - rhs


def traceable_nqc_22_full_series_coefficients(
    time, r, pr_star, orbital_frequency, base_mode, match_time, targets,
    *, barrier_combined=False, amplitude_rows=None, phase_series=None,
    center_phase=False,
):
    """Differentiate a fixed-grid full-series NQC solve on validated inputs.

    The caller must validate the grid, physical samples, and matrix condition
    before treating these coefficients as a waveform. Peak selection and
    attachment boundary changes are outside this fixed-chart derivative.
    """
    time, r, pr_star, orbital_frequency, base_mode, match_time = (
        jnp.asarray(value) for value in
        (time, r, pr_star, orbital_frequency, base_mode, match_time))
    amplitude = jnp.abs(base_mode)
    phase = (_unwrap_mode_phase_unchecked(base_mode) if phase_series is None
             else jnp.asarray(phase_series))
    if center_phase:
        left = jnp.clip(jnp.searchsorted(time, match_time, side="right") - 1,
                        0, time.shape[0] - 2)
        phase = phase - phase[left]
    q, p = nqc_22_basis(r, pr_star, orbital_frequency)
    products = amplitude[:, None] * q
    combined = jnp.concatenate((amplitude[:, None], products, p), axis=-1)
    if barrier_combined:
        combined = lax.optimization_barrier(combined)
    spline = natural_cubic_spline(time, combined, scalar_output=False)
    value = spline(match_time)
    first = spline.derivative(match_time)
    second = spline.derivative(match_time, 2)
    amplitude_matrix = jnp.stack((value[1:4], first[1:4], second[1:4]))
    amplitude_rhs = jnp.stack(targets[:3]) - jnp.stack((value[0], first[0], second[0]))
    if amplitude_rows is not None:
        amplitude_rows = jnp.asarray(amplitude_rows)
        amplitude_matrix = amplitude_rows[:, 1:4]
        amplitude_rhs = jnp.stack(targets[:3]) - amplitude_rows[:, 0]
    phase_matrix = jnp.stack((first[4:6], second[4:6]))
    phase_spline = natural_cubic_spline(time, phase, scalar_output=True)
    phase_rhs = jnp.stack((phase_spline.derivative(match_time) - targets.frequency,
                           phase_spline.derivative(match_time, 2) - targets.frequency_derivative))
    a, amplitude_condition, amplitude_residual = _solve_traceable(
        amplitude_matrix, amplitude_rhs)
    b, phase_condition, phase_residual = _solve_traceable(
        phase_matrix, phase_rhs)
    return (a, b, amplitude_matrix, phase_matrix, amplitude_condition,
            phase_condition, amplitude_residual, phase_residual)


def traceable_fixed_chart_attached_mode22(
    time, r, pr_star, orbital_frequency, base_mode, match_time, targets,
    nu, chi, sigma, output_time, *, mass_scale=1.0,
    barrier_combined=False, amplitude_rows=None, phase_series=None,
    center_phase=False,
):
    """Replay NQC and continuous-boundary attachment on a validated fixed chart.

    The caller owns input and conditioning checks. Adaptive steps, the peak
    event, the sample chart, and the attachment policy are held fixed here.
    This is a development-only derivative diagnostic, not a release model.
    """
    a, b, *_ = traceable_nqc_22_full_series_coefficients(
        time, r, pr_star, orbital_frequency, base_mode, match_time, targets,
        barrier_combined=barrier_combined, amplitude_rows=amplitude_rows,
        phase_series=phase_series, center_phase=center_phase)
    amplitude = jnp.abs(base_mode)
    phase = (_unwrap_mode_phase_unchecked(base_mode) if phase_series is None
             else jnp.asarray(phase_series))
    phase_offset = jnp.asarray(0., dtype=phase.dtype)
    if center_phase:
        left = jnp.clip(jnp.searchsorted(time, match_time, side="right") - 1,
                        0, time.shape[0] - 2)
        phase_offset = phase[left]
        phase = phase - phase_offset
    q, p = nqc_22_basis(r, pr_star, orbital_frequency)
    products = amplitude[:, None] * q
    combined = jnp.concatenate((amplitude[:, None], products, p, phase[:, None]),
                               axis=-1)
    spline = natural_cubic_spline(time, combined, scalar_output=False)

    def corrected_at(query):
        values = spline(query)
        corrected_amplitude = values[..., 0] + jnp.sum(values[..., 1:4] * a, axis=-1)
        corrected_phase = (phase_offset + values[..., 6]
                           - jnp.sum(values[..., 4:6] * b, axis=-1))
        return corrected_amplitude * jnp.exp(-1j * corrected_phase), corrected_phase

    boundary_mode, boundary_phase = corrected_at(match_time)
    boundary_amplitude = jnp.abs(boundary_mode)
    ringdown_time = jnp.maximum(output_time - match_time, 0)
    shape = seobnrv4_peak_preserving_shape(sigma, seobnrv4_merger_shape(nu, chi))
    ringdown = merger_ringdown_22(
        ringdown_time, sigma, boundary_amplitude,
        jnp.zeros_like(boundary_amplitude), targets.frequency, boundary_phase, shape)
    dt_output = output_time[1] - output_time[0]
    exponent = jnp.maximum(0, jnp.ceil(jnp.log2(50 * dt_output / mass_scale)))
    internal_step = dt_output / (2**exponent)
    support_count = jnp.floor(10 / (-jnp.imag(sigma) * internal_step))
    ringdown = jnp.where(ringdown_time < support_count * internal_step,
                         ringdown, jnp.zeros_like(ringdown))
    inspiral, _ = corrected_at(jnp.clip(output_time, time[0], match_time))
    inspiral = jnp.where(output_time < time[0], jnp.zeros_like(inspiral), inspiral)
    mode = jnp.where(output_time < match_time, inspiral, ringdown)
    corrected = base_mode * nqc_22_factor(
        r, pr_star, orbital_frequency, NQCCoefficients(a, b))
    return mode, corrected, a, b


def calibrate_nqc_22_full_series(
    time, r, pr_star, orbital_frequency, base_amplitude, base_frequency,
    match_time, targets: NQCMatchingTargets, *, waveform_phase=None,
    base_mode=None, max_condition=1e10,
    diagnostic_amplitude_derivatives=None, diagnostic_phase_matrix=None,
    diagnostic_matching_rhs_derivatives=None,
) -> FullSeriesNQC:
    """Match NQC on scalar full-series natural splines; development only.

    ``base_frequency`` is retained as a diagnostic spline. Phase matching
    differentiates the continuously unwrapped waveform phase instead.
    """
    samples = tuple(jnp.asarray(x) for x in
                    (time, r, pr_star, orbital_frequency, base_amplitude, base_frequency))
    time, r, pr_star, orbital_frequency, base_amplitude, base_frequency = samples
    if time.ndim != 1 or len(time) < 3 or any(x.shape != time.shape for x in samples):
        raise ValueError("full-series NQC needs equal-length scalar sample arrays")
    if not all(bool(jnp.all(jnp.isfinite(x))) for x in samples):
        raise ValueError("full-series NQC sample arrays must be finite")
    if not bool(jnp.all(jnp.diff(time) > 0)):
        raise ValueError("full-series NQC times must be strictly increasing")
    if not bool(jnp.all((r > 0) & (orbital_frequency > 0) &
                        (base_amplitude > 0) & (base_frequency > 0))):
        raise ValueError("full-series NQC requires positive physical samples")
    if (waveform_phase is None) == (base_mode is None):
        raise ValueError("provide exactly one of waveform_phase and base_mode")
    if base_mode is not None:
        base_mode = jnp.asarray(base_mode)
        if base_mode.shape != time.shape or bool(jnp.any(jnp.abs(base_mode) <= 0)):
            raise ValueError("base_mode must have positive amplitude on the sample grid")
        waveform_phase = unwrap_mode_phase(base_mode)
    else:
        waveform_phase = jnp.asarray(waveform_phase)
    if waveform_phase.shape != time.shape or not bool(jnp.all(jnp.isfinite(waveform_phase))):
        raise ValueError("waveform_phase must be finite on the sample grid")
    match_time = jnp.asarray(match_time)
    if match_time.ndim != 0 or not bool(jnp.isfinite(match_time)) or not bool(time[0] <= match_time <= time[-1]):
        raise ValueError("match_time must be a finite scalar inside the sample grid")
    if not 1 <= max_condition < float("inf"):
        raise ValueError("max_condition must be finite and >= 1")
    if not isinstance(targets, NQCMatchingTargets) or any(
        jnp.asarray(x).ndim != 0 or not bool(jnp.isfinite(x)) for x in targets
    ):
        raise ValueError("targets must contain five finite scalars")

    q, p = nqc_22_basis(r, pr_star, orbital_frequency)
    products = base_amplitude[:, None] * q
    combined = jnp.concatenate((base_amplitude[:, None], products, p), axis=-1)
    combined_spline = natural_cubic_spline(time, combined, scalar_output=False)
    value = combined_spline(match_time)
    first = combined_spline.derivative(match_time)
    second = combined_spline.derivative(match_time, 2)
    if not bool(value[0] > 0):
        raise ValueError("full-series base amplitude is nonpositive at match_time")
    amplitude_matrix = jnp.stack((value[1:4], first[1:4], second[1:4]))
    amplitude_rhs = jnp.stack(targets[:3]) - jnp.stack((value[0], first[0], second[0]))
    phase_matrix = jnp.stack((first[4:6], second[4:6]))
    waveform_phase_spline = natural_cubic_spline(time, waveform_phase, scalar_output=True)
    phase_rhs = jnp.stack((waveform_phase_spline.derivative(match_time) - targets.frequency,
                           waveform_phase_spline.derivative(match_time, 2) - targets.frequency_derivative))
    for name, candidate, shape in (
        ("diagnostic_amplitude_derivatives", diagnostic_amplitude_derivatives, (2, 3)),
        ("diagnostic_phase_matrix", diagnostic_phase_matrix, (2, 2)),
        ("diagnostic_matching_rhs_derivatives", diagnostic_matching_rhs_derivatives, (2, 2)),
    ):
        if candidate is not None:
            candidate = jnp.asarray(candidate)
            if candidate.shape != shape or not bool(jnp.all(jnp.isfinite(candidate))):
                raise ValueError(f"{name} must have finite shape {shape}")
    if diagnostic_amplitude_derivatives is not None:
        amplitude_matrix = jnp.concatenate((
            amplitude_matrix[:1], jnp.asarray(diagnostic_amplitude_derivatives)))
    if diagnostic_phase_matrix is not None:
        phase_matrix = jnp.asarray(diagnostic_phase_matrix)
    if diagnostic_matching_rhs_derivatives is not None:
        rhs_derivatives = jnp.asarray(diagnostic_matching_rhs_derivatives)
        amplitude_rhs = jnp.stack(targets[:3]) - jnp.stack((
            value[0], rhs_derivatives[0, 0], rhs_derivatives[0, 1]))
        phase_rhs = jnp.stack((rhs_derivatives[1, 0] - targets.frequency,
                               rhs_derivatives[1, 1] - targets.frequency_derivative))
    a, amplitude_condition, amplitude_residual = _solve_checked(
        amplitude_matrix, amplitude_rhs, max_condition, "full-series amplitude matching system")
    b, phase_condition, phase_residual = _solve_checked(
        phase_matrix, phase_rhs, max_condition, "full-series phase matching system")
    spacing = jnp.diff(time)
    calibration = NQCCalibration(
        NQCCoefficients(a, b), jnp.max(spacing) / jnp.min(spacing),
        amplitude_condition, phase_condition, amplitude_residual, phase_residual,
    )
    corrected_amplitude = base_amplitude + jnp.sum(products * a, axis=-1)
    corrected_phase = waveform_phase - jnp.sum(p * b, axis=-1)
    corrected_mode = corrected_amplitude * jnp.exp(-1j * corrected_phase)
    return FullSeriesNQC(
        calibration, natural_cubic_spline(time, base_amplitude, scalar_output=True),
        natural_cubic_spline(time, products, scalar_output=False),
        natural_cubic_spline(time, p, scalar_output=False),
        natural_cubic_spline(time, base_frequency, scalar_output=True),
        waveform_phase_spline, base_amplitude, products, p, waveform_phase,
        corrected_amplitude, unwrap_mode_phase(corrected_mode),
        amplitude_matrix, phase_matrix,
    )


def full_orbital_peak(time, frequency) -> OrbitalFrequencyPeak:
    """Apply the full-spline/sign-product peak rule used by the Torch route.

    The first non-increasing sample selects an adjacent interval. Bisection
    follows a strictly negative slope product; zero belongs to the left move.
    This is a scalar, eager numerical policy, not a JIT event detector.
    """
    time = jnp.asarray(time)
    frequency = jnp.asarray(frequency)
    if time.ndim != 1 or frequency.shape != time.shape or len(time) < 4:
        raise ValueError("peak needs matching scalar grids of at least four samples")
    if not bool(jnp.all(jnp.isfinite(time)) & jnp.all(jnp.isfinite(frequency))):
        raise ValueError("peak grids must be finite")
    if not bool(jnp.all(jnp.diff(time) > 0)):
        raise ValueError("peak time grid must be strictly increasing")
    decreasing = frequency[1:] <= frequency[:-1]
    if not bool(jnp.any(decreasing)):
        raise ValueError("orbital-frequency peak is unresolved: no non-increasing sample")
    index = int(jnp.argmax(decreasing)) + 1
    spline = natural_cubic_spline(time, frequency, scalar_output=True)
    following = bool(spline.derivative(time[index]) > 0)
    if following and index + 1 >= len(time):
        raise ValueError("orbital-frequency peak has no following interval")
    left = time[index if following else index - 1]
    right = time[index + 1 if following else index]
    left_slope = spline.derivative(left)
    middle = (left + right) / 2
    middle_slope = spline.derivative(middle)
    last_middle = middle
    while float(right - left) > 1e-5:
        if bool(left_slope * middle_slope < 0):
            right = middle
        else:
            left = middle
            left_slope = middle_slope
        last_middle = middle
        if float(right - left) > 1e-5:
            middle = (left + right) / 2
            middle_slope = spline.derivative(middle)
    peak = last_middle
    return OrbitalFrequencyPeak(peak, spline(peak), spline.derivative(peak, 2),
                                jnp.asarray(index))


def attach_seobnrv4_22_numerical_compatibility(
    time, state, orbital_frequency, base_mode, base_frequency,
    nu, chi, sigma, output_time, *, match_time=None, mass_scale=1.0,
    max_condition=1e10, diagnostic_amplitude_derivatives=None,
    diagnostic_phase_matrix=None, diagnostic_matching_rhs_derivatives=None,
    diagnostic_continuous_boundary=False,
) -> NumericalAttachedMode22:
    """Apply the full-series NQC and discrete ringdown boundary to one row.

    This scalar, eager development route mirrors the Torch numerical contract.
    It is not a release-model entry point or a replacement for the paper's
    independent physical and whole-waveform validation gates.
    """
    time, state, orbital_frequency, base_mode, base_frequency, output_time = (
        jnp.asarray(value) for value in
        (time, state, orbital_frequency, base_mode, base_frequency, output_time))
    nu, chi, sigma, mass_scale = (jnp.asarray(value) for value in
                                  (nu, chi, sigma, mass_scale))
    if (time.ndim != 1 or len(time) < 4 or state.shape != (len(time), 4)
            or any(value.shape != time.shape for value in
                   (orbital_frequency, base_mode, base_frequency))
            or output_time.ndim != 1 or len(output_time) < 2):
        raise ValueError("attachment needs scalar input and output series")
    if not all(bool(jnp.all(jnp.isfinite(value))) for value in
               (time, state, orbital_frequency, base_mode, base_frequency,
                output_time, nu, chi, sigma, mass_scale)):
        raise ValueError("attachment inputs must be finite")
    if not bool(jnp.all(jnp.diff(time) > 0) & jnp.all(jnp.diff(output_time) > 0)):
        raise ValueError("attachment grids must be strictly increasing")
    if any(value.ndim != 0 for value in (nu, chi, sigma, mass_scale)):
        raise ValueError("attachment parameters must be scalar")
    if not bool(mass_scale > 0) or not bool(-jnp.imag(sigma) > 0):
        raise ValueError("mass_scale and QNM damping must be positive")
    if not isinstance(diagnostic_continuous_boundary, bool):
        raise TypeError("diagnostic_continuous_boundary must be a bool")

    offset = seobnrv4_nqc_time_offset_numerical_compatibility(nu, chi)
    if match_time is None:
        peak = full_orbital_peak(time, orbital_frequency)
        match = peak.time + offset
    else:
        match = jnp.asarray(match_time)
        peak_time = match - offset
        if not bool(time[0] <= peak_time <= time[-1]):
            raise ValueError("implied peak lies outside the input grid")
        peak_spline = natural_cubic_spline(time, orbital_frequency, scalar_output=True)
        peak_index = jnp.clip(jnp.searchsorted(time, peak_time, side="right") - 1,
                              0, len(time) - 1)
        peak = OrbitalFrequencyPeak(
            peak_time, peak_spline(peak_time), peak_spline.derivative(peak_time, 2),
            peak_index)
    if match.ndim != 0 or not bool(jnp.isfinite(match)) or not bool(time[0] < match < time[-1]):
        raise ValueError("continuous match_time must lie inside the input grid")
    boundary_index = jnp.searchsorted(time, match, side="right") - 1
    if int(boundary_index) < 1:
        raise ValueError("attachment floor sample has no preceding phase sample")
    boundary_time = time[boundary_index]
    step = boundary_time - time[boundary_index - 1]

    nqc = calibrate_nqc_22_full_series(
        time, state[:, 0], state[:, 2], orbital_frequency,
        jnp.abs(base_mode), base_frequency, match, seobnrv4_nqc_targets(nu, chi),
        base_mode=base_mode, max_condition=max_condition,
        diagnostic_amplitude_derivatives=diagnostic_amplitude_derivatives,
        diagnostic_phase_matrix=diagnostic_phase_matrix,
        diagnostic_matching_rhs_derivatives=diagnostic_matching_rhs_derivatives)
    coefficients = nqc.coefficients
    corrected = base_mode * nqc_22_factor(
        state[:, 0], state[:, 2], orbital_frequency, coefficients)
    match_phase = (nqc.waveform_phase_spline(match)
                   - jnp.sum(nqc.phase_basis_spline(match) * coefficients.phase))
    if diagnostic_continuous_boundary:
        boundary_time = match
        boundary_amplitude = jnp.abs(nqc.evaluate(match))
        boundary_phase = match_phase
        boundary_frequency = seobnrv4_nqc_targets(nu, chi).frequency
    else:
        boundary_amplitude = jnp.abs(corrected[boundary_index])
        boundary_phase = nqc.sampled_corrected_phase[boundary_index]
        boundary_frequency = (boundary_phase - nqc.sampled_corrected_phase[boundary_index - 1]) / step
    if not bool((boundary_amplitude > 0) & (boundary_frequency > 0)):
        raise ValueError("corrected boundary must have positive amplitude and frequency")

    dt_output = output_time[1] - output_time[0]
    exponent = jnp.maximum(0, jnp.ceil(jnp.log2(50 * dt_output / mass_scale)))
    internal_step = dt_output / (2**exponent)
    ringdown_time = jnp.maximum(output_time - boundary_time, 0)
    shape = seobnrv4_peak_preserving_shape(sigma, seobnrv4_merger_shape(nu, chi))
    ringdown = merger_ringdown_22(
        ringdown_time, sigma, boundary_amplitude,
        jnp.zeros_like(boundary_amplitude), boundary_frequency, boundary_phase, shape)
    support_count = jnp.floor(10 / (-jnp.imag(sigma) * internal_step))
    ringdown = jnp.where(ringdown_time < support_count * internal_step,
                         ringdown, jnp.zeros_like(ringdown))
    spline_query = jnp.clip(output_time, time[0], boundary_time)
    inspiral = nqc.evaluate(spline_query)
    inspiral = jnp.where(output_time < time[0], jnp.zeros_like(inspiral), inspiral)
    mode = jnp.where(output_time < boundary_time, inspiral, ringdown)
    return NumericalAttachedMode22(
        mode, corrected, peak, match, match_phase, nqc.calibration,
        boundary_time, boundary_index, boundary_phase, boundary_frequency,
        boundary_amplitude, internal_step, nqc)


__all__ = ["FullSeriesNQC", "NumericalAttachedMode22",
           "attach_seobnrv4_22_numerical_compatibility",
           "calibrate_nqc_22_full_series", "full_orbital_peak",
           "traceable_nqc_22_full_series_coefficients",
           "traceable_fixed_chart_attached_mode22", "unwrap_mode_phase"]

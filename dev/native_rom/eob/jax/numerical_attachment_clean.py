"""Development-only scalar numerical attachment conventions for native v4."""

from dataclasses import dataclass

import jax.numpy as jnp

from jaxwave.approximants.eob.events import OrbitalFrequencyPeak
from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline
from jaxwave.approximants.eob.nqc import (
    NQCCalibration,
    NQCCoefficients,
    NQCMatchingTargets,
    nqc_22_basis,
)


def unwrap_mode_phase(mode):
    """Continuous positive phase of ``h=A*exp(-i*phi)`` on one time series."""
    mode = jnp.asarray(mode)
    if mode.ndim != 1 or len(mode) < 1 or not bool(jnp.all(jnp.isfinite(mode))):
        raise ValueError("mode must be a finite, nonempty scalar series")
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


def calibrate_nqc_22_full_series(
    time, r, pr_star, orbital_frequency, base_amplitude, base_frequency,
    match_time, targets: NQCMatchingTargets, *, waveform_phase=None,
    base_mode=None, max_condition=1e10,
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


__all__ = ["FullSeriesNQC", "calibrate_nqc_22_full_series", "full_orbital_peak",
           "unwrap_mode_phase"]

"""Trajectory-to-merger wiring for the published SEOBNRv4 matching equations in JAX.

The printed free amplitude rate is projected to the ansatz's non-increasing
peak boundary when the supplied QNM damping would make the join rise.
"""

from typing import NamedTuple, Optional
import jax.numpy as jnp

from jaxwave.approximants.eob.events import OrbitalFrequencyPeak, orbital_frequency_peak
from jaxwave.approximants.eob.nqc import (
    NQCCalibration,
    calibrate_nqc_22,
    nqc_22_basis,
    nqc_22_factor,
    seobnrv4_nqc_targets,
    seobnrv4_nqc_time_offset,
)
from jaxwave.approximants.eob.merger_ringdown import (
    merger_ringdown_22, seobnrv4_merger_shape, seobnrv4_peak_preserving_shape,
)
from jaxwave.conversions import as_array


class AttachedMode22(NamedTuple):
    mode: jnp.ndarray  # (..., N_output), dimensionless (D/M)*h22
    corrected_inspiral: jnp.ndarray  # (..., N_input)
    peak: OrbitalFrequencyPeak
    match_time: jnp.ndarray
    match_phase: jnp.ndarray  # h=A*exp(-i phase)
    calibration: NQCCalibration


def _local_mode_interpolate(query: jnp.ndarray, time: jnp.ndarray, mode: jnp.ndarray) -> jnp.ndarray:
    """Seven-point degree-six Lagrange interpolation on the source grid."""
    count = time.shape[-1]
    index = jnp.clip(jnp.searchsorted(time, query, side="left"), 3, count - 4)
    offsets = jnp.arange(-3, 4)
    indices = jnp.clip(jnp.expand_dims(index, -1) + offsets, 0, count - 1)

    nodes = jnp.take_along_axis(jnp.expand_dims(time, -2), indices, axis=-1)
    ordinates = jnp.take_along_axis(jnp.expand_dims(mode, -2), indices, axis=-1)

    differences = jnp.expand_dims(query, -1) - nodes
    weights_list = []
    for j in range(7):
        mask = jnp.arange(7) != j
        numerator = jnp.prod(differences[..., mask], axis=-1)
        denominator = jnp.prod(nodes[..., j : j + 1] - nodes[..., mask], axis=-1)
        weights_list.append(numerator / denominator)
    weights = jnp.stack(weights_list, axis=-1)
    result = jnp.sum(ordinates * weights, axis=-1)
    return jnp.where((query < time[..., :1]) | (query > time[..., -1:]), 0.0, result)


def attach_seobnrv4_22(
    time: jnp.ndarray,
    state: jnp.ndarray,
    orbital_frequency: jnp.ndarray,
    base_mode: jnp.ndarray,
    base_frequency: jnp.ndarray,
    nu: jnp.ndarray,
    chi: jnp.ndarray,
    sigma: jnp.ndarray,
    output_time: jnp.ndarray,
    *,
    valid: Optional[jnp.ndarray] = None,
    matching_method: str = "continuous",
) -> AttachedMode22:
    """Locate Omega's peak, solve all five NQC conditions, and attach ringdown."""
    time = as_array(time, dtype=jnp.float64)
    state = as_array(state, dtype=jnp.float64)
    orbital_frequency = as_array(orbital_frequency, dtype=jnp.float64)
    base_mode = as_array(base_mode, dtype=jnp.complex128)
    base_frequency = as_array(base_frequency, dtype=jnp.float64)
    nu = as_array(nu, dtype=jnp.float64)
    chi = as_array(chi, dtype=jnp.float64)
    sigma = as_array(sigma, dtype=jnp.complex128)
    output_time = as_array(output_time, dtype=jnp.float64)

    count = time.shape[-1]
    batch = jnp.broadcast_shapes(
        time.shape[:-1],
        state.shape[:-2],
        orbital_frequency.shape[:-1],
        base_mode.shape[:-1],
        base_frequency.shape[:-1],
        nu.shape,
        chi.shape,
        sigma.shape,
        output_time.shape[:-1],
    )

    time = jnp.broadcast_to(time, batch + (count,))
    omega = jnp.broadcast_to(orbital_frequency, batch + (count,))
    base = jnp.broadcast_to(base_mode, batch + (count,))
    frequency = jnp.broadcast_to(base_frequency, batch + (count,))
    state = jnp.broadcast_to(state, batch + (count, 4))
    nu = jnp.broadcast_to(nu, batch)
    chi = jnp.broadcast_to(chi, batch)
    sigma = jnp.broadcast_to(sigma, batch)
    query = jnp.broadcast_to(output_time, batch + (output_time.shape[-1],))

    if valid is None:
        valid = jnp.ones_like(time, dtype=bool)
    else:
        valid = as_array(valid, dtype=bool)

    if matching_method == "continuous":
        from .continuous_matching import continuous_spline_peak
        from .numerical_attachment import natural_cubic_spline

        if time.ndim == 1:
            t_v = time[valid]
            om_v = omega[valid]
            peak_time = continuous_spline_peak(t_v, om_v)
            spline = natural_cubic_spline(t_v, om_v, scalar_output=True)
            peak_frequency = spline(peak_time)
            peak_curvature = spline.derivative(peak_time, 2)
            peak_idx = jnp.searchsorted(time, peak_time)
            peak = OrbitalFrequencyPeak(peak_time, peak_frequency, peak_curvature, peak_idx)
        else:
            flat_time = time.reshape(-1, count)
            flat_omega = omega.reshape(-1, count)
            flat_valid = valid.reshape(-1, count)
            pts, pfreqs, pcurvs, pidxs = [], [], [], []
            for t_i, om_i, v_i in zip(flat_time, flat_omega, flat_valid):
                t_sub = t_i[v_i]
                om_sub = om_i[v_i]
                pt = continuous_spline_peak(t_sub, om_sub)
                sp = natural_cubic_spline(t_sub, om_sub, scalar_output=True)
                pts.append(pt)
                pfreqs.append(sp(pt))
                pcurvs.append(sp.derivative(pt, 2))
                pidxs.append(jnp.searchsorted(t_i, pt))
            peak = OrbitalFrequencyPeak(
                jnp.array(pts).reshape(time.shape[:-1]),
                jnp.array(pfreqs).reshape(time.shape[:-1]),
                jnp.array(pcurvs).reshape(time.shape[:-1]),
                jnp.array(pidxs).reshape(time.shape[:-1]),
            )
    elif matching_method == "grid":
        peak = orbital_frequency_peak(time, omega, valid=valid)
    else:
        raise ValueError(f"matching_method must be 'continuous' or 'grid', got {matching_method!r}")

    valid = jnp.broadcast_to(valid, time.shape)
    match = peak.time + seobnrv4_nqc_time_offset(nu, chi)

    index = jnp.clip(jnp.searchsorted(time, jnp.expand_dims(match, -1)).squeeze(-1), 1, count - 1)
    a_time = jnp.take_along_axis(time, jnp.expand_dims(index - 1, -1), axis=-1).squeeze(-1)
    b_time = jnp.take_along_axis(time, jnp.expand_dims(index, -1), axis=-1).squeeze(-1)
    index = jnp.where(match - a_time < b_time - match, index - 1, index)

    offsets = jnp.arange(-3, 4)
    indices = jnp.expand_dims(index, -1) + offsets
    def stencil(value):
        return jnp.take_along_axis(value, indices, axis=-1)

    t = stencil(time)
    r = stencil(state[..., 0])
    pr = stencil(state[..., 2])
    om = stencil(omega)
    amplitude = stencil(jnp.abs(base))
    freq = stencil(frequency)

    targets = seobnrv4_nqc_targets(nu, chi)
    calibration = calibrate_nqc_22(t, r, pr, om, amplitude, freq, match, targets, degree=6)
    corrected = base * nqc_22_factor(
        state[..., 0], state[..., 2], omega, calibration.coefficients, sample_dims=1
    )

    q, p = nqc_22_basis(r, pr, om)
    span = t[..., -1] - t[..., 0]
    x = (t - jnp.expand_dims(match, -1)) / jnp.expand_dims(span, -1)
    matrix = jnp.stack([x**k for k in range(7)], axis=-1)
    ordinates = jnp.concatenate(
        [
            jnp.expand_dims(amplitude, -1),
            jnp.expand_dims(amplitude, -1) * q,
            p,
            jnp.expand_dims(freq, -1),
        ],
        axis=-1,
    )
    polynomial = jnp.linalg.solve(matrix, ordinates)
    amp_coeff, phase_coeff = calibration.coefficients

    anchor_x = x[..., 0]
    anchor_phase = -jnp.angle(stencil(base)[..., 0])

    def local_mode(z):
        values = sum(
            jnp.expand_dims(polynomial[..., k, :], -2) * jnp.expand_dims(z, -1)**k
            for k in range(7)
        )
        amp = values[..., 0] + jnp.sum(values[..., 1:4] * jnp.expand_dims(amp_coeff, -2), axis=-1)
        integral = sum(
            jnp.expand_dims(polynomial[..., k, 6], -1)
            * (z**(k + 1) - jnp.expand_dims(anchor_x, -1)**(k + 1))
            / (k + 1)
            for k in range(7)
        )
        phase = (
            jnp.expand_dims(anchor_phase, -1)
            + jnp.expand_dims(span, -1) * integral
            - jnp.sum(values[..., 4:6] * jnp.expand_dims(phase_coeff, -2), axis=-1)
        )
        return amp * jnp.exp(-1j * phase), phase

    _, phase_at_match = local_mode(jnp.expand_dims(jnp.zeros_like(match), -1))
    match_phase = phase_at_match.squeeze(-1)

    local_query = jnp.clip(query, t[..., :1], jnp.expand_dims(match, -1))
    near, _ = local_mode((local_query - jnp.expand_dims(match, -1)) / jnp.expand_dims(span, -1))

    inspiral = jnp.where(query >= t[..., :1], near, _local_mode_interpolate(query, time, corrected))
    ringdown = merger_ringdown_22(
        jnp.maximum(query - jnp.expand_dims(match, -1), 0.0),
        sigma,
        targets.amplitude,
        targets.amplitude_derivative,
        targets.frequency,
        match_phase,
        seobnrv4_peak_preserving_shape(sigma, seobnrv4_merger_shape(nu, chi)),
    )
    mode = jnp.where(query < jnp.expand_dims(match, -1), inspiral, ringdown)

    return AttachedMode22(mode, corrected, peak, match, match_phase, calibration)


__all__ = [
    "AttachedMode22",
    "attach_seobnrv4_22",
]

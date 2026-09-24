"""Published EOB mode components and finite-signal Fourier projection.

Complete unvalidated SEOBNR waveform assemblers are development-only."""

import math
from typing import NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp

from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC
from jaxwave.approximants.eob.attachment import AttachedMode22
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator, FrequencyPeakTrajectory
from jaxwave.approximants.eob.seobnrv4_mode22 import (
    seobnrv4_published_delta22,
    seobnrv4_published_mode22,
    seobnrv4_published_phase22,
)
from jaxwave.conversions import as_array

_DIRECT_FREQUENCY_CHUNK = 32


class PublishedEOB22(NamedTuple):
    time: jnp.ndarray  # (N,), t/M from the initial state
    mode: jnp.ndarray  # (..., N), (D/M)*h22
    trajectory: FrequencyPeakTrajectory
    attachment: AttachedMode22
    final_mass_fraction: jnp.ndarray
    final_spin: jnp.ndarray


def _direct_quadrupole_fd(f, h22, time_s, dt_sec, inclination):
    """Project and integrate a finite-support 22 mode on an arbitrary grid."""
    cos_i = jnp.cos(inclination)[..., None]
    hp_td = math.sqrt(5.0 / (16.0 * math.pi)) * (1.0 + cos_i**2) * h22.real
    hc_td = -math.sqrt(5.0 / (4.0 * math.pi)) * cos_i * h22.imag
    dt_sec = dt_sec[..., None]
    plus_chunks = []
    cross_chunks = []
    for start in range(0, f.shape[-1], _DIRECT_FREQUENCY_CHUNK):
        stop = min(start + _DIRECT_FREQUENCY_CHUNK, f.shape[-1])
        kernel = jnp.exp(
            (-2j * math.pi)
            * f[..., start:stop, None]
            * time_s[..., None, :]
        )
        plus_chunks.append(
            jnp.sum(kernel * hp_td[..., None, :], axis=-1) * dt_sec
        )
        cross_chunks.append(
            jnp.sum(kernel * hc_td[..., None, :], axis=-1) * dt_sec
        )
    return jnp.concatenate(plus_chunks, axis=-1), jnp.concatenate(cross_chunks, axis=-1)


def published_eob22_to_fd(
    result: PublishedEOB22,
    f: jnp.ndarray,
    mass1: Union[float, jnp.ndarray],
    mass2: Union[float, jnp.ndarray],
    distance: Union[float, jnp.ndarray] = 100.0,
    inclination: Union[float, jnp.ndarray] = 0.0,
    phic: Union[float, jnp.ndarray] = 0.0,
    tc: Union[float, jnp.ndarray] = 0.0,
    *,
    time_origin: str = "orbital_frequency_peak",
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Transform a reconstructed published 22 mode into FD polarizations."""
    if time_origin == "orbital_frequency_peak":
        origin = result.trajectory.peak.time
    elif time_origin == "nqc_match":
        origin = result.attachment.match_time
    else:
        raise ValueError("time_origin must be 'orbital_frequency_peak' or 'nqc_match'")

    f = as_array(f, dtype=jnp.float64)
    time = as_array(result.time, dtype=jnp.float64)
    dt_geom = time[1] - time[0]

    m1 = as_array(mass1, dtype=jnp.float64)
    m2 = as_array(mass2, dtype=jnp.float64)
    dist = as_array(distance, dtype=jnp.float64)
    inc = as_array(inclination, dtype=jnp.float64)
    phase = as_array(phic, dtype=jnp.float64)
    delay = as_array(tc, dtype=jnp.float64)
    origin = as_array(origin, dtype=jnp.float64)

    batch_shape = jnp.broadcast_shapes(
        f.shape[:-1], result.mode.shape[:-1], m1.shape, m2.shape,
        dist.shape, inc.shape, phase.shape, delay.shape, origin.shape
    )
    f = jnp.broadcast_to(f, batch_shape + (f.shape[-1],))
    mode = jnp.broadcast_to(result.mode, batch_shape + (time.size,))
    m1, m2, dist, inc, phase, delay, origin = (
        jnp.broadcast_to(v, batch_shape)
        for v in (m1, m2, dist, inc, phase, delay, origin)
    )

    m_sec = (m1 + m2) * MSUN_SEC
    mode = mode * jnp.exp(-2j * phase[..., None])
    mode = mode * (m_sec * C_SI / (dist * MPC_SI))[..., None]
    absolute_time_s = (
        delay[..., None]
        + (time - origin[..., None]) * m_sec[..., None]
    )
    return _direct_quadrupole_fd(
        f, mode, absolute_time_s, dt_geom * m_sec, inc
    )


def published_mode22_and_frequency(
    state, hamiltonian, radiation_reaction, *, spin_v6,
    include_pan_v8, include_positive_chi_540, retain_pan_cubic_spin=False,
):
    """Pre-NQC mode and continuous GW frequency along the EOB vector field.

    The directional derivative includes the orbital, tail, residual-phase,
    and radiation-reaction terms. This same-state component is not a
    qualified JAX trajectory or full waveform.
    """
    state = jnp.asarray(state)
    if state.ndim == 0 or state.shape[-1] != 4:
        raise ValueError("state must end in (r, phi, pr_star, pphi)")

    def quantities_and_residual(value):
        quantities = radiation_reaction.from_hamiltonian(value, hamiltonian)
        residual = seobnrv4_published_delta22(
            **{key: quantities[key] for key in
               ("omega", "hreal_over_M", "x1", "x2", "chi1", "chi2")},
            spin_v6=spin_v6, include_pan_v8=include_pan_v8,
            include_positive_chi_540=include_positive_chi_540)
        return quantities, residual

    def phase(value):
        quantities, residual = quantities_and_residual(value)
        return seobnrv4_published_phase22(
            quantities["omega"], value[..., 1],
            hreal_over_M=quantities["hreal_over_M"], residual_phase=residual)

    field = EOBTrajectoryIntegrator(hamiltonian, radiation_reaction).vector_field(state)
    _, frequency = jax.jvp(phase, (state,), (field,))
    quantities, residual = quantities_and_residual(state)
    mode = seobnrv4_published_mode22(
        phi=state[..., 1], residual_phase=residual, **quantities,
        retain_pan_cubic_spin=retain_pan_cubic_spin)
    return mode, frequency


__all__ = [
    'PublishedEOB22',
    'published_eob22_to_fd',
    'published_mode22_and_frequency',
]

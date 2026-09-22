"""Development-only prototype; scientific qualification is incomplete.

Published SEOBNRv4 waveform generator and utilities in JAX.
"""

import math
from typing import Optional, Tuple, Union
import jax.numpy as jnp
from diffgw_core.constants import MSUN_SEC, PI
from jaxwave.approximants.eob.attachment import attach_seobnrv4_22
from jaxwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from jaxwave.approximants.eob.nqc import (
    seobnrv4_nqc_parameters,
    seobnrv4_nqc_time_offset,
    seobnrv4_nqc_time_offset_numerical_compatibility,
)
from jaxwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
from jaxwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention
from jaxwave.conversions import as_array

from jaxwave.approximants.eob.published_waveform import PublishedEOB22, published_eob22_to_fd

def reconstruct_seobnrv4_22(
    omega_start: jnp.ndarray,
    mass1: Union[float, jnp.ndarray],
    mass2: Union[float, jnp.ndarray],
    chi1: Union[float, jnp.ndarray] = 0.0,
    chi2: Union[float, jnp.ndarray] = 0.0,
    *,
    spin_truncation: str,
    spin_v6: str,
    include_pan_v8: bool,
    include_positive_chi_540: bool,
    final_spin_fit_order: Tuple[int, int],
    dt_geom: float = 0.5,
    num_steps: int = 20000,
    ringdown_duration: float = 150.0,
    richardson_extrapolation: bool = True,
    numerical_compatibility: bool = False,
) -> PublishedEOB22:
    """Integrate, calibrate NQC, and attach the published (2,2) mode."""
    start = as_array(omega_start, dtype=jnp.float64)
    m1 = as_array(mass1, dtype=jnp.float64)
    m2 = as_array(mass2, dtype=jnp.float64)
    s1 = as_array(chi1, dtype=jnp.float64)
    s2 = as_array(chi2, dtype=jnp.float64)

    start, m1, m2, s1, s2 = jnp.broadcast_arrays(start, m1, m2, s1, s2)
    mass_fraction = eob_final_mass_fraction(m1, m2, s1, s2)
    final_spin = hofmann_final_spin(m1, m2, s1, s2, fit_order=final_spin_fit_order)

    if numerical_compatibility:
        qnm_w, qnm_damp = signed_qnm_220_table_convention(final_spin)
    else:
        qnm_w, qnm_damp = evaluate_kerr_qnm_220(final_spin)
    sigma = (qnm_w - 1j * qnm_damp) / mass_fraction

    scaled1 = m1 / jnp.maximum(m1, m2)
    scaled2 = m2 / jnp.maximum(m1, m2)
    x1 = scaled1 / (scaled1 + scaled2)
    x2 = scaled2 / (scaled1 + scaled2)
    nu, chi = seobnrv4_nqc_parameters(x1, x2, s1, s2)

    ham = SEOBNRv4Hamiltonian(nu, s1, s2, delta=x1 - x2)
    from .radiation_reaction import EOBRadiationReaction
    rr = EOBRadiationReaction(nu, s1, s2)

    r, pr, j = find_quasicircular_initial_conditions(ham, rr, start)
    initial = jnp.stack([r, jnp.zeros_like(r), pr, j], axis=-1)

    nqc_offset_fn = (
        seobnrv4_nqc_time_offset_numerical_compatibility
        if numerical_compatibility
        else seobnrv4_nqc_time_offset
    )
    after = jnp.maximum(nqc_offset_fn(nu, chi), 0.0) + dt_geom
    trajectory = EOBTrajectoryIntegrator(ham, rr).integrate_to_frequency_peak(
        initial, dt_geom, num_steps, post_peak_duration=after,
        richardson_extrapolation=richardson_extrapolation
    )
    state = jnp.moveaxis(trajectory.state, 0, -2)
    v = jnp.maximum(trajectory.frequency, 1e-8) ** (1.0 / 3.0)
    h22_amp = -8.0 * math.sqrt(PI / 5.0) * nu * (v * v)
    phi = state[..., 1]
    phase_gw = 2.0 * phi
    base_mode = jnp.moveaxis(h22_amp * jnp.cos(phase_gw) - 1j * (h22_amp * jnp.sin(phase_gw)), 0, -1)
    base_frequency = jnp.moveaxis(2.0 * trajectory.frequency, 0, -1)

    match = trajectory.peak.time + nqc_offset_fn(nu, chi)
    count = math.ceil(float(jnp.max(match) + ringdown_duration) / dt_geom) + 1
    time = jnp.arange(count, dtype=start.dtype) * dt_geom

    attachment = attach_seobnrv4_22(
        trajectory.time,
        state,
        jnp.moveaxis(trajectory.frequency, 0, -1),
        base_mode,
        base_frequency,
        nu,
        chi,
        sigma,
        time,
        valid=jnp.moveaxis(trajectory.valid, 0, -1),
    )
    output = jnp.where(
        time <= jnp.expand_dims(match, -1) + ringdown_duration,
        attachment.mode,
        jnp.zeros_like(attachment.mode),
    )
    return PublishedEOB22(time, output, trajectory, attachment, mass_fraction, final_spin)


class SEOBNRv4Published:
    """Experimental published-equation quadrupole in JAX."""

    @staticmethod
    def __call__(
        f: jnp.ndarray,
        mass1: Union[float, jnp.ndarray],
        mass2: Union[float, jnp.ndarray],
        chi1: Union[float, jnp.ndarray] = 0.0,
        chi2: Union[float, jnp.ndarray] = 0.0,
        distance: Union[float, jnp.ndarray] = 100.0,
        inclination: Union[float, jnp.ndarray] = 0.0,
        phic: Union[float, jnp.ndarray] = 0.0,
        tc: Union[float, jnp.ndarray] = 0.0,
        f_lower: Optional[float] = None,
        *,
        spin_truncation: str,
        spin_v6: str,
        include_pan_v8: bool,
        include_positive_chi_540: bool,
        final_spin_fit_order: Tuple[int, int],
        dt_geom: float = 0.5,
        num_steps: int = 20000,
        ringdown_duration: float = 150.0,
        richardson_extrapolation: bool = True,
        numerical_compatibility: bool = False,
        time_origin: str = "orbital_frequency_peak",
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        f = as_array(f, dtype=jnp.float64)
        m1 = as_array(mass1, dtype=jnp.float64)
        m2 = as_array(mass2, dtype=jnp.float64)
        s1 = as_array(chi1, dtype=jnp.float64)
        s2 = as_array(chi2, dtype=jnp.float64)

        if f_lower is None:
            positive = f[f > 0]
            if positive.size == 0:
                raise ValueError("f_lower is required when the frequency grid has no positive samples")
            lower = jnp.min(positive)
        else:
            lower = as_array(f_lower, dtype=jnp.float64)

        m_sec = (m1 + m2) * MSUN_SEC
        result = reconstruct_seobnrv4_22(
            PI * m_sec * lower,
            m1,
            m2,
            s1,
            s2,
            spin_truncation=spin_truncation,
            spin_v6=spin_v6,
            include_pan_v8=include_pan_v8,
            include_positive_chi_540=include_positive_chi_540,
            final_spin_fit_order=final_spin_fit_order,
            dt_geom=dt_geom,
            num_steps=num_steps,
            ringdown_duration=ringdown_duration,
            richardson_extrapolation=richardson_extrapolation,
            numerical_compatibility=numerical_compatibility,
        )
        return published_eob22_to_fd(
            result, f, m1, m2, distance=distance, inclination=inclination,
            phic=phic, tc=tc, time_origin=time_origin,
        )


class SEOBNRv4PublishedNonspinning(SEOBNRv4Published):
    """Published-equation driver with a fixed nonspinning policy."""

    @staticmethod
    def __call__(f, mass1, mass2, chi1=0.0, chi2=0.0, **kwargs):
        s1 = as_array(chi1, dtype=jnp.float64)
        s2 = as_array(chi2, dtype=jnp.float64)
        if bool(jnp.any(s1 != 0.0)) or bool(jnp.any(s2 != 0.0)):
            raise ValueError("SEOBNRv4PublishedNonspinning requires chi1=chi2=0")
        if "final_spin_fit_order" not in kwargs:
            raise TypeError("final_spin_fit_order must be explicit for the nonspinning preset")
        fit_order = kwargs.pop("final_spin_fit_order")
        choices = dict(
            spin_truncation="tabulated",
            spin_v6="bohe_total_energy",
            include_pan_v8=False,
            include_positive_chi_540=False,
            final_spin_fit_order=fit_order,
        )
        return SEOBNRv4Published.__call__(f, mass1, mass2, chi1=0.0, chi2=0.0, **kwargs, **choices)

"""Development-only prototype; scientific qualification is incomplete.

Effective-One-Body Waveform Generation and Polarizations in JAX.
"""

import math
from typing import Optional, Tuple, Union
import jax.numpy as jnp
from diffgw_core.constants import C_SI, MPC_SI, MSUN_SEC, PI
from jaxwave.approximants.eob.attachment import attach_seobnrv4_22
from .hamiltonian import EOBHamiltonian
from jaxwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from jaxwave.approximants.eob.nqc import seobnrv4_nqc_parameters, seobnrv4_nqc_time_offset
from .radiation_reaction import EOBRadiationReaction
from jaxwave.approximants.eob.radiation_reaction import (
    FactorizedRadiationReaction,
)
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from jaxwave.approximants.imrphenomd import IMRPhenomD
from jaxwave.conversions import as_array, component_masses_to_delta, component_masses_to_eta

from jaxwave.approximants.eob.waveform import _quadrupole_fd

class SEOBNR_ODE:
    """Native EOB prototype in JAX."""

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
        dt_geom: float = 0.5,
        num_steps: Optional[int] = None,
        f_lower: Optional[float] = None,
        *,
        conservative_model: str = "prototype",
        radiation_model: str = "prototype",
        assembly_model: str = "prototype",
        spin_truncation: Optional[str] = None,
        spin_v6: Optional[str] = None,
        include_pan_v8: Optional[bool] = None,
        include_positive_chi_540: Optional[bool] = None,
        final_spin_fit_order: Optional[Tuple[int, int]] = None,
        matching_method: str = "continuous",
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate frequency-domain polarizations via EOB dynamical integration."""
        if num_steps is None:
            num_steps = 20000 if assembly_model == "seobnrv4_published" else 1500

        published_choices = {
            "spin_truncation": spin_truncation,
            "spin_v6": spin_v6,
            "include_pan_v8": include_pan_v8,
            "include_positive_chi_540": include_positive_chi_540,
            "final_spin_fit_order": final_spin_fit_order,
        }
        if assembly_model not in {"prototype", "seobnrv4_published"}:
            raise ValueError("assembly_model must be 'prototype' or 'seobnrv4_published'")
        if assembly_model == "seobnrv4_published":
            missing = [name for name, value in published_choices.items() if value is None]
            if missing:
                raise ValueError("assembly_model='seobnrv4_published' requires explicit " + ", ".join(missing))
            if conservative_model != "prototype" or radiation_model != "prototype":
                raise ValueError(
                    "assembly_model='seobnrv4_published' selects its own conservative and radiation equations"
                )
            from dev.native_rom.eob.jax.published_waveform import SEOBNRv4Published

            return SEOBNRv4Published()(
                f=f,
                mass1=mass1,
                mass2=mass2,
                chi1=chi1,
                chi2=chi2,
                distance=distance,
                inclination=inclination,
                phic=phic,
                tc=tc,
                f_lower=f_lower,
                spin_truncation=spin_truncation,
                spin_v6=spin_v6,
                include_pan_v8=include_pan_v8,
                include_positive_chi_540=include_positive_chi_540,
                final_spin_fit_order=final_spin_fit_order,
                dt_geom=dt_geom,
                num_steps=num_steps,
            )
        elif any(value is not None for value in published_choices.values()):
            raise ValueError("published SEOBNRv4 choices require assembly_model='seobnrv4_published'")

        f = as_array(f, dtype=jnp.float64)
        m1 = as_array(mass1, dtype=jnp.float64)
        m2 = as_array(mass2, dtype=jnp.float64)
        s1 = as_array(chi1, dtype=jnp.float64)
        s2 = as_array(chi2, dtype=jnp.float64)
        dist = as_array(distance, dtype=jnp.float64)
        inc = as_array(inclination, dtype=jnp.float64)
        phi_c = as_array(phic, dtype=jnp.float64)
        t_c = as_array(tc, dtype=jnp.float64)

        m1, m2, s1, s2, dist, inc, phi_c, t_c = jnp.broadcast_arrays(
            m1, m2, s1, s2, dist, inc, phi_c, t_c
        )
        is_batched = m1.ndim > 0

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)

        conservative_models = {
            "prototype": EOBHamiltonian,
            "seobnrv4_published": SEOBNRv4Hamiltonian,
        }
        if conservative_model not in conservative_models:
            raise ValueError("conservative_model must be 'prototype' or 'seobnrv4_published'")
        hamiltonian = conservative_models[conservative_model](nu=eta, chi1=s1, chi2=s2, delta=delta)
        rr = (
            EOBRadiationReaction(nu=eta, chi1=s1, chi2=s2)
            if radiation_model == "prototype"
            else FactorizedRadiationReaction(radiation_model)
        )
        integrator = EOBTrajectoryIntegrator(hamiltonian, rr)

        m_sec = total_mass * MSUN_SEC
        positive = f[f > 0]
        if f_lower is None and positive.size == 0:
            raise ValueError("f_lower is required when the frequency grid has no positive samples")
        f_start_hz = as_array(f_lower, dtype=jnp.float64) if f_lower is not None else jnp.min(positive)
        if f_lower is not None and (not bool(jnp.all(jnp.isfinite(f_start_hz))) or bool(jnp.any(f_start_hz <= 0))):
            raise ValueError("f_lower must be finite and positive")
        omega_start = PI * m_sec * f_start_hz

        r0, pr0, pphi0 = find_quasicircular_initial_conditions(hamiltonian, rr, omega_start)
        init_state = jnp.stack([r0, jnp.zeros_like(r0), pr0, pphi0], axis=-1)

        attachment_trajectory = None
        if radiation_model == "prototype" and num_steps >= 4:
            nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, s1, s2)
            try:
                attachment_trajectory = integrator.integrate_to_frequency_peak(
                    init_state,
                    dt_geom,
                    num_steps,
                    post_peak_duration=jnp.maximum(seobnrv4_nqc_time_offset(nqc_nu, nqc_chi), 0.0) + dt_geom,
                    matching_method=matching_method,
                )
            except ValueError as error:
                if "peak not reached within num_steps" not in str(error):
                    raise

        if attachment_trajectory is None:
            t_geom, traj = integrator.integrate(init_state, dt=dt_geom, num_steps=num_steps)
        else:
            t_geom = attachment_trajectory.time
            traj = attachment_trajectory.state

        r_traj = traj[..., 0]
        phi_traj = traj[..., 1]
        pr_traj = traj[..., 2]
        pphi_traj = traj[..., 3]

        if attachment_trajectory is not None:
            omega_traj = attachment_trajectory.frequency
        else:
            _, omega_traj, _ = hamiltonian.equations_of_motion_conservative(r_traj, pr_traj, pphi_traj)
        v_traj = jnp.maximum(omega_traj, 1e-8) ** (1.0 / 3.0)

        h22_amp = -8.0 * math.sqrt(PI / 5.0) * eta * (v_traj * v_traj)
        phase_gw = 2.0 * (phi_traj + phi_c)

        h22_real = h22_amp * jnp.cos(phase_gw)
        h22_imag = -h22_amp * jnp.sin(phase_gw)

        if radiation_model != "prototype":
            h22_factorized = rr.mode(2, 2, omega_traj, pr_traj, pphi_traj, r=r_traj, phi=phi_traj + phi_c, hamiltonian=hamiltonian)
            h22_real, h22_imag = h22_factorized.real, h22_factorized.imag
            h22_amp = jnp.abs(h22_factorized)
            phase_gw = -jnp.angle(h22_factorized)

        _, _, f_rd, f_damp = IMRPhenomD.remnant_properties(m1, m2, s1, s2)

        if attachment_trajectory is not None:
            base_mode = jnp.moveaxis(h22_real + 1j * h22_imag, 0, -1)
            sigma = 2.0 * PI * f_rd - 1j * (2.0 * PI * f_damp)
            match_duration = 100.0
            match = attachment_trajectory.peak.time + seobnrv4_nqc_time_offset(nqc_nu, nqc_chi)
            sample_count = math.ceil(float(jnp.max(match) + match_duration) / dt_geom) + 1
            output_time = jnp.arange(sample_count, dtype=jnp.float64) * dt_geom
            attached = attach_seobnrv4_22(
                t_geom,
                jnp.moveaxis(traj, 0, -2),
                jnp.moveaxis(omega_traj, 0, -1),
                base_mode,
                jnp.moveaxis(2.0 * omega_traj, 0, -1),
                nqc_nu,
                nqc_chi,
                sigma,
                output_time,
                valid=jnp.moveaxis(attachment_trajectory.valid, 0, -1),
                matching_method=matching_method,
            )
            h22_full_real = jnp.moveaxis(attached.mode.real, -1, 0)
            h22_full_imag = jnp.moveaxis(attached.mode.imag, -1, 0)
        else:
            omega_qnm = 2.0 * PI * f_rd
            tau_qnm = 1.0 / (2.0 * PI * f_damp)

            amp_end = h22_amp[-1]
            phi_end = phase_gw[-1]

            t_rd = jnp.arange(dt_geom, 100.0, dt_geom, dtype=jnp.float64)
            if is_batched:
                t_rd_exp = t_rd.reshape((-1,) + (1,) * m1.ndim)
                amp_end_exp = jnp.expand_dims(amp_end, 0)
                tau_qnm_exp = jnp.expand_dims(tau_qnm, 0)
                omega_qnm_exp = jnp.expand_dims(omega_qnm, 0)
                phi_end_exp = jnp.expand_dims(phi_end, 0)
                rd_amp = amp_end_exp * jnp.exp(-t_rd_exp / tau_qnm_exp)
                rd_phase = phi_end_exp + omega_qnm_exp * t_rd_exp
            else:
                rd_amp = amp_end * jnp.exp(-t_rd / tau_qnm)
                rd_phase = phi_end + omega_qnm * t_rd

            h22_rd_real = rd_amp * jnp.cos(rd_phase)
            h22_rd_imag = -rd_amp * jnp.sin(rd_phase)

            h22_full_real = jnp.concatenate([h22_real, h22_rd_real], axis=0)
            h22_full_imag = jnp.concatenate([h22_imag, h22_rd_imag], axis=0)

        d_sec = (dist * MPC_SI) / C_SI
        phys_scale = m_sec / d_sec
        if is_batched:
            phys_scale = jnp.expand_dims(phys_scale, 0)

        h22_phys_real = h22_full_real * phys_scale
        h22_phys_imag = h22_full_imag * phys_scale

        h22_c = h22_phys_real + 1j * h22_phys_imag
        hp, hc = _quadrupole_fd(f, h22_c, dt_geom * m_sec, inc)

        phase_shift = -2.0 * PI * f * jnp.expand_dims(t_c, -1)
        shift_factor = jnp.cos(phase_shift) + 1j * jnp.sin(phase_shift)
        return hp * shift_factor, hc * shift_factor

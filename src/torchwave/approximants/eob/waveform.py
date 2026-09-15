"""Effective-One-Body Waveform Generation and Polarizations.

Native prototype with equation-derived repairs; complete calibrated EOB
waveforms and provenance of all inherited expressions remain unverified.
Relevant primary publications:
- Damour, Iyer, Nagar (2009), Phys. Rev. D 79, 064004 (arXiv:0811.2069)
- Bohé et al. (2017), Phys. Rev. D 95, 044028 (arXiv:1611.03703)
- Pompili et al. (2023), Phys. Rev. D 108, 124035 (arXiv:2303.18039)
- Berti et al. (2006), Phys. Rev. D 73, 064030 (arXiv:gr-qc/0512160)
"""

from typing import Optional, Tuple, Union
import torch
import math

from torchwave.constants import (
    C_SI,
    MSUN_SEC,
    MPC_SI,
    PI,
)
from torchwave.conversions import (
    as_tensor,
    component_masses_to_eta,
    component_masses_to_delta,
)
from torchwave.approximants.eob.hamiltonian import EOBHamiltonian
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.radiation_reaction import EOBRadiationReaction, FactorizedRadiationReaction
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.attachment import attach_seobnrv4_22
from torchwave.approximants.eob.nqc import seobnrv4_nqc_parameters, seobnrv4_nqc_time_offset
from torchwave.approximants.imrphenomd import IMRPhenomD


def _interp1d(x_new: torch.Tensor, x_old: torch.Tensor, y_old: torch.Tensor) -> torch.Tensor:
    """Interpolate along the final axis, with zero outside the sample interval.

    A scalar query returns one value per table batch; otherwise the query's
    final axis is its sample axis.
    """
    scalar_query = x_new.ndim == 0
    if scalar_query:
        x_new = x_new.unsqueeze(-1)
    batch_shape = torch.broadcast_shapes(x_new.shape[:-1], x_old.shape[:-1], y_old.shape[:-1])
    x_old_b = x_old.expand(*batch_shape, x_old.shape[-1])
    y_old_b = y_old.expand(*batch_shape, y_old.shape[-1])
    x_new_b = x_new.expand(*batch_shape, x_new.shape[-1])

    idx = torch.searchsorted(x_old_b.contiguous(), x_new_b.contiguous(), right=False)
    idx = torch.clamp(idx, 1, x_old_b.shape[-1] - 1)

    x0 = torch.gather(x_old_b, -1, idx - 1)
    x1 = torch.gather(x_old_b, -1, idx)
    y0 = torch.gather(y_old_b, -1, idx - 1)
    y1 = torch.gather(y_old_b, -1, idx)

    slope = (y1 - y0) / (x1 - x0 + 1e-30)
    y_new = y0 + slope * (x_new_b - x0)

    mask_low = x_new_b < x_old_b[..., :1]
    mask_high = x_new_b > x_old_b[..., -1:]
    y_new = torch.where(mask_low | mask_high, torch.zeros_like(y_new), y_new)

    return y_new.squeeze(-1) if scalar_query else y_new


class SEOBNR_ODE:
    """Native EOB prototype; calibrated SEOBNRv4 merger physics is incomplete."""

    @staticmethod
    def __call__(
        f: torch.Tensor,
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        chi1: Union[float, torch.Tensor] = 0.0,
        chi2: Union[float, torch.Tensor] = 0.0,
        distance: Union[float, torch.Tensor] = 100.0,
        inclination: Union[float, torch.Tensor] = 0.0,
        phic: Union[float, torch.Tensor] = 0.0,
        tc: Union[float, torch.Tensor] = 0.0,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate frequency-domain polarizations via EOB dynamical integration.

        Args:
            f: Frequency evaluation tensor in Hz
            mass1, mass2: Component masses in solar masses
            chi1, chi2: Aligned dimensionless spins
            distance: Luminosity distance in Mpc
            inclination: Inclination angle in radians
            phic: Coalescence phase in radians
            tc: Coalescence time in seconds
            dt_geom: Time step in geometric units (M)
            num_steps: Maximum trajectory integration steps
            conservative_model: ``prototype`` or experimental ``seobnrv4_published``.
                The latter changes conservative dynamics only.
            radiation_model: ``prototype``, ``din2009_nonspinning`` (35-mode
                flux), or ``pan2011_leading_spin`` (leading spin, ell<=4).
                Factorized options also supply the emitted (2,2) strain.
                Merger attachment remains an incomplete prototype.
            assembly_model: ``prototype`` (the backward-compatible native
                assembly) or ``seobnrv4_published``. The published assembly
                requires all five published-equation choices below.
            spin_truncation, spin_v6, include_pan_v8,
                include_positive_chi_540, final_spin_fit_order: Explicit
                published SEOBNRv4 flux, residual-phase, and remnant/QNM
                choices. These are valid only with
                ``assembly_model='seobnrv4_published'``.

        Returns:
            (h_plus, h_cross) frequency-domain complex tensors
        """
        # The native prototype historically used a short diagnostic horizon.
        # The published assembly owns a longer default (see
        # ``SEOBNRv4Published.__call__``); selecting it here avoids silently
        # replacing that event horizon when dispatching through this wrapper.
        if num_steps is None:
            num_steps = 20000 if assembly_model == "seobnrv4_published" else 1500

        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        published_choices = {
            "spin_truncation": spin_truncation,
            "spin_v6": spin_v6,
            "include_pan_v8": include_pan_v8,
            "include_positive_chi_540": include_positive_chi_540,
            "final_spin_fit_order": final_spin_fit_order,
        }
        if assembly_model not in {"prototype", "seobnrv4_published"}:
            raise ValueError(
                "assembly_model must be 'prototype' or 'seobnrv4_published'"
            )
        if assembly_model == "seobnrv4_published":
            missing = [name for name, value in published_choices.items() if value is None]
            if missing:
                raise ValueError(
                    "assembly_model='seobnrv4_published' requires explicit "
                    + ", ".join(missing)
                )
            if conservative_model != "prototype" or radiation_model != "prototype":
                raise ValueError(
                    "assembly_model='seobnrv4_published' selects its own "
                    "conservative and radiation equations"
                )
        elif any(value is not None for value in published_choices.values()):
            raise ValueError(
                "published SEOBNRv4 choices require "
                "assembly_model='seobnrv4_published'"
            )

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1 = as_tensor(chi1, device=device, dtype=dtype)
        s2 = as_tensor(chi2, device=device, dtype=dtype)
        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_c = as_tensor(phic, device=device, dtype=dtype)
        t_c = as_tensor(tc, device=device, dtype=dtype)

        m1, m2, s1, s2, dist, inc, phi_c, t_c = torch.broadcast_tensors(
            m1, m2, s1, s2, dist, inc, phi_c, t_c)

        if assembly_model == "seobnrv4_published":
            # Keep this import local: published_waveform uses this module's
            # quadrupole Fourier helper and would otherwise form a cycle.
            from torchwave.approximants.eob.published_waveform import SEOBNRv4Published

            return SEOBNRv4Published()(
                f=f.to(dtype=dtype),
                mass1=m1,
                mass2=m2,
                chi1=s1,
                chi2=s2,
                distance=dist,
                inclination=inc,
                phic=phi_c,
                tc=t_c,
                f_lower=f_lower,
                spin_truncation=spin_truncation,
                spin_v6=spin_v6,
                include_pan_v8=include_pan_v8,
                include_positive_chi_540=include_positive_chi_540,
                final_spin_fit_order=final_spin_fit_order,
                dt_geom=dt_geom,
                num_steps=num_steps,
            )
        is_batched = m1.dim() > 0

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)

        conservative_models = {"prototype": EOBHamiltonian, "seobnrv4_published": SEOBNRv4Hamiltonian}
        if conservative_model not in conservative_models:
            raise ValueError("conservative_model must be 'prototype' or 'seobnrv4_published'")
        hamiltonian = conservative_models[conservative_model](nu=eta, chi1=s1, chi2=s2, delta=delta)
        rr = (EOBRadiationReaction(nu=eta, chi1=s1, chi2=s2) if radiation_model == "prototype"
              else FactorizedRadiationReaction(radiation_model))
        integrator = EOBTrajectoryIntegrator(hamiltonian, rr)

        # The frequency grid may include DC. An explicit lower bound makes
        # dynamics independent of padding the requested output grid with zeros.
        m_sec = total_mass * MSUN_SEC
        positive = f[f > 0]
        if f_lower is None and not positive.numel():
            raise ValueError("f_lower is required when the frequency grid has no positive samples")
        f_start_hz = as_tensor(f_lower, device=device, dtype=dtype) if f_lower is not None else positive.min()
        if f_lower is not None and (not bool(torch.isfinite(f_start_hz).all())
                                    or bool((f_start_hz <= 0).any())):
            raise ValueError("f_lower must be finite and positive")
        omega_start = PI * m_sec * f_start_hz

        # Initial conditions
        r0, pr0, pphi0 = find_quasicircular_initial_conditions(hamiltonian, rr, omega_start)
        init_state = torch.stack([r0, torch.zeros_like(r0), pr0, pphi0], dim=-1)

        # The prototype has historically integrated a fixed number of steps
        # and attached at its final sample.  When a resolved orbital-frequency
        # maximum is available, use the equation-level NQC/C1 attachment.  A
        # short diagnostic run may not contain a maximum, so retain the old
        # splice for that explicitly under-resolved case.
        attachment_trajectory = None
        if radiation_model == "prototype" and num_steps >= 4:
            nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, s1, s2)
            try:
                attachment_trajectory = integrator.integrate_to_frequency_peak(
                    init_state, dt_geom, num_steps,
                    post_peak_duration=seobnrv4_nqc_time_offset(nqc_nu, nqc_chi).clamp_min(0)
                    + dt_geom,
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

        # Evaluate the same angular velocity used by the Hamilton equations.
        # Each row carries the original parameter batch layout.
        omega_traj = (attachment_trajectory.frequency if attachment_trajectory is not None else
                      torch.stack([
                          hamiltonian.equations_of_motion_conservative(r, pr, pp)[1]
                          for r, pr, pp in zip(r_traj, pr_traj, pphi_traj)
                      ]))
        v_traj = torch.clamp(omega_traj, min=1e-8) ** (1.0 / 3.0)

        # Multipolar strain: h22(t) = -8 * sqrt(pi/5) * nu * v^2 * exp(-2i * phi)
        # In geometric units
        h22_amp = -8.0 * math.sqrt(PI / 5.0) * eta * (v_traj * v_traj)
        # Rotate the time-domain modes before projection; finite signals
        # contain both Fourier poles, which acquire opposite orbital phases.
        phase_gw = 2.0 * (phi_traj + phi_c)

        # Complex time-domain multipole
        h22_real = h22_amp * torch.cos(phase_gw)
        h22_imag = -h22_amp * torch.sin(phase_gw)

        if radiation_model != "prototype":
            h22_factorized = torch.stack([
                rr.mode(2, 2, omega, pr, pp, r=r, phi=phi + phi_c, hamiltonian=hamiltonian)
                for r, phi, pr, pp, omega in zip(
                    r_traj, phi_traj, pr_traj, pphi_traj, omega_traj,
                )
            ])
            h22_real, h22_imag = h22_factorized.real, h22_factorized.imag
            h22_amp = h22_factorized.abs()
            phase_gw = -torch.angle(h22_factorized)

        # PhenomD returns M*f_RD and M*f_damp, so retain t/M throughout
        # ringdown and convert to seconds only in the Fourier measure below.
        _, _, f_rd, f_damp = IMRPhenomD.remnant_properties(m1, m2, s1, s2)

        if attachment_trajectory is not None:
            # The prototype mode has phase 2*phi (plus the caller's phase),
            # hence its continuous positive GW frequency is 2*Omega.  The
            # existing attachment applies the published peak targets and
            # solves the five NQC conditions before the single-QNM C1 join.
            base_mode = torch.complex(h22_real, h22_imag).movedim(0, -1)
            sigma = torch.complex(2.0 * PI * f_rd, -2.0 * PI * f_damp)
            match_duration = 100.0
            match = attachment_trajectory.peak.time + seobnrv4_nqc_time_offset(
                nqc_nu, nqc_chi)
            sample_count = math.ceil(float((match.max() + match_duration).detach()) / dt_geom) + 1
            output_time = torch.arange(sample_count, dtype=dtype, device=device) * dt_geom
            attached = attach_seobnrv4_22(
                t_geom,
                traj.movedim(0, -2),
                omega_traj.movedim(0, -1),
                base_mode,
                (2.0 * omega_traj).movedim(0, -1),
                nqc_nu,
                nqc_chi,
                sigma,
                output_time,
                valid=attachment_trajectory.valid.movedim(0, -1),
            )
            h22_full_real = attached.mode.real.movedim(-1, 0)
            h22_full_imag = attached.mode.imag.movedim(-1, 0)
        else:
            omega_qnm = 2.0 * PI * f_rd
            tau_qnm = 1.0 / (2.0 * PI * f_damp)

            # Match at the last sample in the same geometric time coordinate.
            amp_end = h22_amp[-1]
            phi_end = phase_gw[-1]

            # Synthesize ringdown extension
            t_rd = torch.arange(dt_geom, 100.0, dt_geom, dtype=dtype, device=device)
            if is_batched:
                t_rd_exp = t_rd.reshape(-1, *([1] * m1.ndim))
                amp_end_exp = amp_end.unsqueeze(0)
                tau_qnm_exp = tau_qnm.unsqueeze(0)
                omega_qnm_exp = omega_qnm.unsqueeze(0)
                phi_end_exp = phi_end.unsqueeze(0)
                rd_amp = amp_end_exp * torch.exp(-t_rd_exp / tau_qnm_exp)
                rd_phase = phi_end_exp + omega_qnm_exp * t_rd_exp
            else:
                rd_amp = amp_end * torch.exp(-t_rd / tau_qnm)
                rd_phase = phi_end + omega_qnm * t_rd

            h22_rd_real = rd_amp * torch.cos(rd_phase)
            h22_rd_imag = -rd_amp * torch.sin(rd_phase)

            h22_full_real = torch.cat([h22_real, h22_rd_real], dim=0)
            h22_full_imag = torch.cat([h22_imag, h22_rd_imag], dim=0)

        # Physical scaling: h_phys(t) = (M_sec / D_sec) * h_geom(t)
        d_sec = (dist * MPC_SI) / C_SI
        phys_scale = m_sec / d_sec
        if is_batched:
            phys_scale = phys_scale.unsqueeze(0)

        h22_phys_real = h22_full_real * phys_scale
        h22_phys_imag = h22_full_imag * phys_scale

        # h = h_plus - i h_cross = Y22 h22 + Y2,-2 conjugate(h22).
        # Transform the real polarizations: h22 itself rotates at negative
        # frequency, so taking its positive FFT bins discards the carrier.
        h22_c = torch.complex(h22_phys_real, h22_phys_imag)
        hp, hc = _quadrupole_fd(f, h22_c, dt_geom * m_sec, inc)

        # Delay covariance of the exp(-2 pi i f t) Fourier transform.
        phase_shift = -2.0 * PI * f * t_c.unsqueeze(-1)
        shift_factor = torch.complex(torch.cos(phase_shift), torch.sin(phase_shift))
        return hp * shift_factor, hc * shift_factor


def _quadrupole_fd(f, h22, dt_sec, inclination):
    """Project paired spherical quadrupoles and Fourier transform in seconds.

    Time is axis zero, followed by any parameter batch axes. The
    transform uses exp(-2 pi i f t), with frequencies on the last output axis.
    """
    cos_i = torch.cos(inclination)
    hp_td = math.sqrt(5.0 / (16.0 * PI)) * (1.0 + cos_i**2) * h22.real
    hc_td = -math.sqrt(5.0 / (4.0 * PI)) * cos_i * h22.imag
    fft_len = 2 ** math.ceil(math.log2(h22.shape[0] * 4))
    hp_fd = torch.fft.rfft(hp_td, n=fft_len, dim=0) * dt_sec
    hc_fd = torch.fft.rfft(hc_td, n=fft_len, dim=0) * dt_sec
    k = torch.arange(fft_len // 2 + 1, device=f.device, dtype=f.dtype)
    if h22.ndim > 1:
        fft_freqs = k / (fft_len * dt_sec.unsqueeze(-1))
        hp_fd, hc_fd = hp_fd.movedim(0, -1), hc_fd.movedim(0, -1)
    else:
        fft_freqs = k / (fft_len * dt_sec)
    return _interp1d(f, fft_freqs, hp_fd), _interp1d(f, fft_freqs, hc_fd)

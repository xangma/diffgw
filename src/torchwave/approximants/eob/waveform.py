"""Effective-One-Body Waveform Generation and Polarizations.

Strict Clean-Room Implementation based on published peer-reviewed literature:
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
from torchwave.approximants.eob.radiation_reaction import EOBRadiationReaction
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.imrphenomd import IMRPhenomD


def _interp1d(x_new: torch.Tensor, x_old: torch.Tensor, y_old: torch.Tensor) -> torch.Tensor:
    """1D piecewise linear interpolation on PyTorch tensors supporting batched evaluation."""
    is_batched = (y_old.dim() > 1) or (x_old.dim() > 1)
    if not is_batched:
        x_old_b = x_old.unsqueeze(0)
        y_old_b = y_old.unsqueeze(0)
    else:
        x_old_b = x_old if x_old.dim() > 1 else x_old.unsqueeze(0).expand(y_old.shape[0], -1)
        y_old_b = y_old

    x_new_b = x_new if x_new.dim() > 1 else x_new.unsqueeze(0).expand(x_old_b.shape[0], -1)

    idx = torch.searchsorted(x_old_b.contiguous(), x_new_b.contiguous(), right=False)
    idx = torch.clamp(idx, 1, x_old_b.shape[-1] - 1)

    x0 = torch.gather(x_old_b, -1, idx - 1)
    x1 = torch.gather(x_old_b, -1, idx)
    y0 = torch.gather(y_old_b, -1, idx - 1)
    y1 = torch.gather(y_old_b, -1, idx)

    slope = (y1 - y0) / (x1 - x0 + 1e-30)
    y_new = y0 + slope * (x_new_b - x0)

    mask_low = x_new_b < x_old_b[:, :1]
    mask_high = x_new_b > x_old_b[:, -1:]
    y_new = torch.where(mask_low | mask_high, torch.zeros_like(y_new), y_new)

    return y_new if is_batched else y_new.squeeze(0)


class SEOBNR_ODE:
    """Time-Domain Effective-One-Body Waveform Generator with Ringdown Attachment."""

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
        num_steps: int = 1500,
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

        Returns:
            (h_plus, h_cross) frequency-domain complex tensors
        """
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1 = as_tensor(chi1, device=device, dtype=dtype)
        s2 = as_tensor(chi2, device=device, dtype=dtype)
        dist = as_tensor(distance, device=device, dtype=dtype)
        inc = as_tensor(inclination, device=device, dtype=dtype)
        phi_c = as_tensor(phic, device=device, dtype=dtype)
        t_c = as_tensor(tc, device=device, dtype=dtype)

        is_batched = m1.dim() > 0

        total_mass = m1 + m2
        eta = component_masses_to_eta(m1, m2)
        delta = component_masses_to_delta(m1, m2)

        hamiltonian = EOBHamiltonian(nu=eta, chi1=s1, chi2=s2, delta=delta)
        rr = EOBRadiationReaction(nu=eta, chi1=s1, chi2=s2)
        integrator = EOBTrajectoryIntegrator(hamiltonian, rr)

        # Starting orbital frequency (e.g. corresponding to f_start ~ 20 Hz)
        m_sec = total_mass * MSUN_SEC
        f_start_hz = torch.clamp(f[0], min=15.0)
        omega_start = torch.clamp(PI * m_sec * f_start_hz, min=0.005, max=0.05)

        # Initial conditions
        r0, pr0, pphi0 = find_quasicircular_initial_conditions(hamiltonian, rr, omega_start)
        init_state = torch.stack([r0, torch.zeros_like(r0), pr0, pphi0], dim=-1)

        # Integrate trajectory
        t_geom, traj = integrator.integrate(init_state, dt=dt_geom, num_steps=num_steps)

        r_traj = traj[..., 0]
        phi_traj = traj[..., 1]
        pphi_traj = traj[..., 3]

        # Orbital angular velocity Omega = dphi/dt ~ pphi / r^2
        omega_traj = pphi_traj / (r_traj * r_traj)
        v_traj = torch.clamp(omega_traj, min=1e-8) ** (1.0 / 3.0)

        # Multipolar strain: h22(t) = -8 * sqrt(pi/5) * nu * v^2 * exp(-2i * phi)
        # In geometric units
        h22_amp = 8.0 * math.sqrt(PI / 5.0) * eta * (v_traj * v_traj)
        phase_gw = 2.0 * phi_traj

        # Complex time-domain multipole
        h22_real = h22_amp * torch.cos(phase_gw)
        h22_imag = -h22_amp * torch.sin(phase_gw)

        # Ringdown QNM parameters (Berti et al. 2006)
        _, _, f_rd, f_damp = IMRPhenomD.remnant_properties(m1, m2, s1, s2)
        omega_qnm = 2.0 * PI * f_rd
        tau_qnm = 1.0 / (2.0 * PI * f_damp)

        # Match ringdown at end of trajectory
        t_end = t_geom[-1]
        amp_end = h22_amp[-1]
        phi_end = phase_gw[-1]

        # Synthesize ringdown extension
        t_rd = torch.arange(1.0, 100.0, dt_geom, dtype=dtype, device=device)
        if is_batched:
            t_rd_exp = t_rd.unsqueeze(-1)
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

        # FFT to frequency domain along time dimension (dim 0)
        h22_c = torch.complex(h22_phys_real, h22_phys_imag)
        fft_len = 2 ** math.ceil(math.log2(h22_c.shape[0] * 4))
        h22_fd_full = torch.fft.fft(h22_c, n=fft_len, dim=0)
        num_pos = fft_len // 2 + 1
        h22_fd = h22_fd_full[:num_pos]

        dt_sec = dt_geom * m_sec
        if is_batched:
            h22_fd_scaled = (h22_fd * dt_sec.unsqueeze(0)).transpose(0, 1)  # (B, num_pos)
            k_indices = torch.arange(num_pos, device=device, dtype=dtype)
            fft_freqs = k_indices.unsqueeze(0) / (fft_len * dt_sec.unsqueeze(-1))  # (B, num_pos)
        else:
            dt_val = dt_sec.item()
            h22_fd_scaled = h22_fd * dt_val
            fft_freqs = torch.fft.rfftfreq(fft_len, d=dt_val).to(device=device, dtype=dtype)

        # Differentiable 1D interpolation onto user evaluation grid f
        h22_interp = _interp1d(f, fft_freqs, h22_fd_scaled)

        # Time and phase shift: exp(i * (2*pi*f*tc - 2*phic))
        if is_batched:
            phase_shift = 2.0 * PI * f.unsqueeze(0) * t_c.unsqueeze(-1) - 2.0 * phi_c.unsqueeze(-1)
        else:
            phase_shift = 2.0 * PI * f * t_c - 2.0 * phi_c
        shift_factor = torch.complex(torch.cos(phase_shift), torch.sin(phase_shift))
        h22_shifted = h22_interp * shift_factor

        # Inclination projections:
        # -2Y22 factor: sqrt(5 / (64*pi)) * (1 + cos(iota))^2
        cos_i = torch.cos(inc)
        plus_factor = 0.5 * (1.0 + cos_i * cos_i)
        cross_factor = cos_i
        if is_batched:
            plus_factor = plus_factor.unsqueeze(-1)
            cross_factor = cross_factor.unsqueeze(-1)

        hp = plus_factor * h22_shifted
        hc = cross_factor * h22_shifted

        return hp, hc

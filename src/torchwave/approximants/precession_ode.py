"""Differentiable Spin-Precession Dynamics and ODE Engine.

Strict Clean-Room Implementation derived strictly from cited literature:
- Chatziioannou et al. (2017), Phys. Rev. D 95, 104004 (arXiv:1703.03967)
  "Two-body problem in general relativity with spins: Precession on radiation-reaction timescales"
- Kidder (1995), Phys. Rev. D 52, 821 (arXiv:gr-qc/9506014)
  "Coalescing binary systems of compact objects to (post)^(5/2)-Newtonian order. V. Spin effects"
- Racine (2008), Phys. Rev. D 78, 044021 (arXiv:0803.1848)
  "Analysis of spin precession in binary black hole systems including two-spin effects"
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503)
"""

from typing import Dict, Optional, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.constants import (
    C_SI,
    G_SI,
    MSUN_SEC,
    PI,
)
from torchwave.conversions import as_tensor


class SpinPrecessionODE(nn.Module):
    """Orbit-averaged spin-precession numerical integrator with pure PyTorch autograd differentiability.

    Integrates:
    - 2PN Spin-Orbit (SO) couplings
    - 2PN Spin-Spin (SS) couplings
    - 2.5PN Radiation reaction orbital decay
    - Conserves total angular momentum magnitude |J| = |L + S1 + S2| in conservative limit to < 10^(-6)
    - Extracts precessing Euler angles (alpha, cos_beta, epsilon)
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def precession_vectors(
        s1: torch.Tensor,
        s2: torch.Tensor,
        l_vec: torch.Tensor,
        m1: torch.Tensor,
        m2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute precession frequency vectors Omega_1 and Omega_2.

        Omega_1 = (1/r^3) * [ (2 + 3*m2/(2*m1)) * L + (1/2)*S_2 - (3/2)*(S_2 . L_hat)*L_hat ]
        Omega_2 = (1/r^3) * [ (2 + 3*m1/(2*m2)) * L + (1/2)*S_1 - (3/2)*(S_1 . L_hat)*L_hat ]
        """
        l_mag = torch.norm(l_vec, dim=-1, keepdim=True)
        l_hat = l_vec / torch.clamp(l_mag, min=1e-24)

        total_m = m1 + m2
        eta = (m1 * m2) / (total_m * total_m)
        if total_m.dim() > 0 and l_mag.dim() > total_m.dim():
            eta_m = eta.unsqueeze(-1)
            m1_m = m1.unsqueeze(-1)
            m2_m = m2.unsqueeze(-1)
        else:
            eta_m = eta
            m1_m = m1
            m2_m = m2

        # In geometric units with M=1: L = eta * sqrt(r) => r = (L / eta)^2
        r = (l_mag / torch.clamp(eta_m, min=1e-12)) ** 2
        r3 = torch.clamp(r ** 3, min=1e-36)

        q1 = m2_m / m1_m
        q2 = m1_m / m2_m

        # Omega 1
        term1_so = (2.0 + 1.5 * q1) * l_vec
        s2_dot_lh = torch.sum(s2 * l_hat, dim=-1, keepdim=True)
        term1_ss = 0.5 * s2 - 1.5 * s2_dot_lh * l_hat
        omega1 = (term1_so + term1_ss) / r3

        # Omega 2
        term2_so = (2.0 + 1.5 * q2) * l_vec
        s1_dot_lh = torch.sum(s1 * l_hat, dim=-1, keepdim=True)
        term2_ss = 0.5 * s1 - 1.5 * s1_dot_lh * l_hat
        omega2 = (term2_so + term2_ss) / r3

        return omega1, omega2

    @classmethod
    def state_derivatives(
        cls,
        s1: torch.Tensor,
        s2: torch.Tensor,
        l_vec: torch.Tensor,
        m1: torch.Tensor,
        m2: torch.Tensor,
        radiation_reaction: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute time derivatives (dS1/dt, dS2/dt, dL/dt)."""
        omega1, omega2 = cls.precession_vectors(s1, s2, l_vec, m1, m2)

        ds1 = torch.linalg.cross(omega1, s1)
        ds2 = torch.linalg.cross(omega2, s2)
        dl = -(ds1 + ds2)

        if radiation_reaction:
            l_mag = torch.norm(l_vec, dim=-1, keepdim=True)
            l_hat = l_vec / torch.clamp(l_mag, min=1e-24)
            total_m = m1 + m2
            eta = (m1 * m2) / (total_m * total_m)
            if total_m.dim() > 0 and l_mag.dim() > total_m.dim():
                eta_m = eta.unsqueeze(-1)
            else:
                eta_m = eta
            r = (l_mag / torch.clamp(eta_m, min=1e-12)) ** 2
            dl_mag_dt = -(32.0 / 5.0) * (eta_m ** 2) / torch.clamp(r ** 3.5, min=1e-40)
            dl = dl + dl_mag_dt * l_hat

        return ds1, ds2, dl

    @classmethod
    def rk4_step(
        cls,
        s1: torch.Tensor,
        s2: torch.Tensor,
        l_vec: torch.Tensor,
        dt: Union[float, torch.Tensor],
        m1: torch.Tensor,
        m2: torch.Tensor,
        radiation_reaction: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one 4th-order Runge-Kutta step."""
        k1_s1, k1_s2, k1_l = cls.state_derivatives(s1, s2, l_vec, m1, m2, radiation_reaction)

        k2_s1, k2_s2, k2_l = cls.state_derivatives(
            s1 + 0.5 * dt * k1_s1,
            s2 + 0.5 * dt * k1_s2,
            l_vec + 0.5 * dt * k1_l,
            m1, m2,
            radiation_reaction,
        )

        k3_s1, k3_s2, k3_l = cls.state_derivatives(
            s1 + 0.5 * dt * k2_s1,
            s2 + 0.5 * dt * k2_s2,
            l_vec + 0.5 * dt * k2_l,
            m1, m2,
            radiation_reaction,
        )

        k4_s1, k4_s2, k4_l = cls.state_derivatives(
            s1 + dt * k3_s1,
            s2 + dt * k3_s2,
            l_vec + dt * k3_l,
            m1, m2,
            radiation_reaction,
        )

        s1_next = s1 + (dt / 6.0) * (k1_s1 + 2.0 * k2_s1 + 2.0 * k3_s1 + k4_s1)
        s2_next = s2 + (dt / 6.0) * (k1_s2 + 2.0 * k2_s2 + 2.0 * k3_s2 + k4_s2)
        l_next = l_vec + (dt / 6.0) * (k1_l + 2.0 * k2_l + 2.0 * k3_l + k4_l)

        return s1_next, s2_next, l_next

    @classmethod
    def integrate_time(
        cls,
        s1_0: torch.Tensor,
        s2_0: torch.Tensor,
        l_0: torch.Tensor,
        m1: torch.Tensor,
        m2: torch.Tensor,
        total_time: float = 1000.0,
        n_steps: int = 500,
        radiation_reaction: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Integrate spin precession over a fixed time duration in units of M."""
        dt = total_time / float(n_steps)
        s1_curr = s1_0.clone()
        s2_curr = s2_0.clone()
        l_curr = l_0.clone()

        times = [torch.tensor(0.0, dtype=s1_0.dtype, device=s1_0.device)]
        s1_traj = [s1_curr]
        s2_traj = [s2_curr]
        l_traj = [l_curr]
        j_traj = [s1_curr + s2_curr + l_curr]

        for step in range(n_steps):
            s1_curr, s2_curr, l_curr = cls.rk4_step(
                s1_curr, s2_curr, l_curr, dt, m1, m2, radiation_reaction
            )
            times.append(torch.tensor((step + 1) * dt, dtype=s1_0.dtype, device=s1_0.device))
            s1_traj.append(s1_curr)
            s2_traj.append(s2_curr)
            l_traj.append(l_curr)
            j_traj.append(s1_curr + s2_curr + l_curr)

        return {
            "times": torch.stack(times),
            "s1": torch.stack(s1_traj),
            "s2": torch.stack(s2_traj),
            "l": torch.stack(l_traj),
            "j": torch.stack(j_traj),
        }

    @classmethod
    def extract_euler_angles(
        cls,
        l_traj: torch.Tensor,
        j0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract precessing Euler angles (alpha, beta, epsilon) from L(t) relative to J_0.

        Args:
            l_traj: Orbital angular momentum trajectory, shape (..., N, 3) or (N, 3)
            j0: Initial total angular momentum vector, shape (..., 3) or (3,)

        Returns:
            Tuple of (alpha, beta, epsilon) tensors
        """
        j0_mag = torch.norm(j0, dim=-1, keepdim=True)
        z_j = j0 / torch.clamp(j0_mag, min=1e-24)

        # Reference coordinate system attached to J_0
        z_ref = torch.zeros_like(z_j)
        z_ref[..., 2] = 1.0

        cross_z = torch.linalg.cross(z_ref, z_j)
        cross_mag = torch.norm(cross_z, dim=-1, keepdim=True)

        x_j_default = torch.zeros_like(z_j)
        x_j_default[..., 0] = 1.0

        x_j = torch.where(cross_mag > 1e-4, cross_z / torch.clamp(cross_mag, min=1e-24), x_j_default)
        y_j = torch.linalg.cross(z_j, x_j)

        if z_j.dim() < l_traj.dim():
            z_j_exp = z_j.unsqueeze(-2)
            x_j_exp = x_j.unsqueeze(-2)
            y_j_exp = y_j.unsqueeze(-2)
        else:
            z_j_exp = z_j
            x_j_exp = x_j
            y_j_exp = y_j

        l_mag = torch.norm(l_traj, dim=-1, keepdim=True)
        l_hat = l_traj / torch.clamp(l_mag, min=1e-24)

        # 1. Opening angle beta: cos(beta) = L_hat . J_hat_0
        cos_beta = torch.clamp(torch.sum(l_hat * z_j_exp, dim=-1), -1.0 + 1e-12, 1.0 - 1e-12)
        beta = torch.arccos(cos_beta)

        # 2. Azimuthal precession angle alpha: angle in x_j, y_j plane
        lx = torch.sum(l_hat * x_j_exp, dim=-1)
        ly = torch.sum(l_hat * y_j_exp, dim=-1)
        alpha = torch.atan2(ly, lx)

        # 3. Minimal rotation angle epsilon: d_epsilon = -cos_beta * d_alpha
        # Cumulative trapezoidal integration
        d_alpha = alpha[..., 1:] - alpha[..., :-1]
        # Unwrap 2pi jumps in d_alpha
        d_alpha = torch.remainder(d_alpha + PI, 2.0 * PI) - PI
        cos_b_mid = 0.5 * (cos_beta[..., 1:] + cos_beta[..., :-1])
        d_eps = -cos_b_mid * d_alpha

        zero_lead = torch.zeros(alpha.shape[:-1] + (1,), dtype=alpha.dtype, device=alpha.device)
        eps_cum = torch.cumsum(d_eps, dim=-1)
        epsilon = torch.cat([zero_lead, eps_cum], dim=-1)

        return alpha, beta, epsilon

    @classmethod
    def compute_angles_frequency(
        cls,
        f: torch.Tensor,
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        spin1x: Union[float, torch.Tensor] = 0.0,
        spin1y: Union[float, torch.Tensor] = 0.0,
        spin1z: Union[float, torch.Tensor] = 0.0,
        spin2x: Union[float, torch.Tensor] = 0.0,
        spin2y: Union[float, torch.Tensor] = 0.0,
        spin2z: Union[float, torch.Tensor] = 0.0,
        f_ref: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute precession angles (alpha, beta, epsilon) on a frequency grid f using ODE integration.

        Integrates along velocity v = (pi * M * f)^(1/3) using 2.5PN radiation reaction.
        """
        device = f.device
        dtype = f.dtype if f.is_floating_point() else torch.float64

        m1 = as_tensor(mass1, device=device, dtype=dtype)
        m2 = as_tensor(mass2, device=device, dtype=dtype)
        s1x = as_tensor(spin1x, device=device, dtype=dtype)
        s1y = as_tensor(spin1y, device=device, dtype=dtype)
        s1z = as_tensor(spin1z, device=device, dtype=dtype)
        s2x = as_tensor(spin2x, device=device, dtype=dtype)
        s2y = as_tensor(spin2y, device=device, dtype=dtype)
        s2z = as_tensor(spin2z, device=device, dtype=dtype)

        total_m = m1 + m2
        m_sec = total_m * MSUN_SEC
        eta = (m1 * m2) / (total_m * total_m)
        q = m1 / m2

        m1_norm = m1 / total_m
        m2_norm = m2 / total_m

        s1_vec = (m1_norm ** 2) * torch.stack([s1x, s1y, s1z], dim=-1)
        s2_vec = (m2_norm ** 2) * torch.stack([s2x, s2y, s2z], dim=-1)

        f_ref_val = as_tensor(f_ref if f_ref is not None else 20.0, device=device, dtype=dtype)
        v_ref = (PI * m_sec * f_ref_val) ** (1.0 / 3.0)

        # Initial angular momentum at reference frequency
        l0_dir = torch.zeros_like(s1_vec)
        l0_dir[..., 2] = 1.0
        l0_vec = (eta / torch.clamp(v_ref, min=1e-12)).unsqueeze(-1) * l0_dir
        j0 = l0_vec + s1_vec + s2_vec

        # Velocity grid
        v_grid = (PI * m_sec * f) ** (1.0 / 3.0)
        n_pts = f.shape[-1]

        # Integrate along v_grid
        l_traj = []
        s1_curr = s1_vec
        s2_curr = s2_vec

        for i in range(n_pts):
            v_i = v_grid[..., i]
            l_mag_i = eta / torch.clamp(v_i, min=1e-12)

            if i > 0:
                dv = v_grid[..., i] - v_grid[..., i - 1]
                v_mid = 0.5 * (v_grid[..., i] + v_grid[..., i - 1])
                l_mag_mid = eta / torch.clamp(v_mid, min=1e-12)

                l_prev = l_traj[-1]
                l_hat_mid = l_prev / torch.clamp(torch.norm(l_prev, dim=-1, keepdim=True), min=1e-24)
                l_vec_mid = l_mag_mid.unsqueeze(-1) * l_hat_mid

                om1, om2 = cls.precession_vectors(s1_curr, s2_curr, l_vec_mid, m1, m2)
                dv_dt = (32.0 / 5.0) * eta * (v_mid ** 9)
                denom = torch.clamp(dv_dt.unsqueeze(-1), min=1e-36)

                ds1_dv = torch.linalg.cross(om1, s1_curr) / denom
                ds2_dv = torch.linalg.cross(om2, s2_curr) / denom

                s1_curr = s1_curr + dv.unsqueeze(-1) * ds1_dv
                s2_curr = s2_curr + dv.unsqueeze(-1) * ds2_dv

            # Update L direction: in the precessing frame, L aligns with J - S1 - S2
            l_dir = j0 - s1_curr - s2_curr
            l_dir_mag = torch.norm(l_dir, dim=-1, keepdim=True)
            l_hat_i = l_dir / torch.clamp(l_dir_mag, min=1e-24)
            l_vec_i = l_mag_i.unsqueeze(-1) * l_hat_i
            l_traj.append(l_vec_i)

        l_traj_t = torch.stack(l_traj, dim=-2)
        alpha, beta, epsilon = cls.extract_euler_angles(l_traj_t, j0)

        return alpha, beta, epsilon

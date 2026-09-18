"""Pure PyTorch SpinTaylorT4 spin-precession dynamics and Euler angle engine.

Provides a native, autograd-differentiable PyTorch implementation of the
numerical integration of 3.5PN orbit-averaged SpinTaylorT4 equations of motion,
computation of J-frame Euler precession angles (alpha, beta, gamma), and smooth
C^1 analytical continuation into the merger-ringdown regime.

References:
- Colleoni et al. (2025), Phys. Rev. D 111, 064025 (arXiv:2412.16721)
- Buonanno et al. (2009), Phys. Rev. D 80, 084043 (arXiv:0907.0700)
- Blanchet et al. (2006), Phys. Rev. D 74, 104034 (arXiv:gr-qc/0605140)
- Kidder (1995), Phys. Rev. D 52, 821 (arXiv:gr-qc/9506014)
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503)
"""

from typing import Dict, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn

from torchwave.constants import (
    MSUN_SEC,
    PI,
    EULER_GAMMA,
)
from torchwave.conversions import as_tensor
from torchwave.approximants.imrphenom_xas import XASParameterFits


def compute_chi_tot_perp(
    m1: torch.Tensor,
    m2: torch.Tensor,
    s1x: torch.Tensor,
    s1y: torch.Tensor,
    s2x: torch.Tensor,
    s2y: torch.Tensor,
) -> torch.Tensor:
    """Compute the vectorial in-plane spin parameter chi_tot_perp.

    Norm of the in-plane vector sum S_tot_perp = Norm[ S1_perp + S2_perp ]
    normalized by m1^2 / M^2. Ref: Colleoni et al. (2025), Eq. (26) &
    Pratten et al. (2021), Sec. IV D.
    """
    total_m = m1 + m2
    s1_perp_x = (m1 ** 2) * s1x
    s1_perp_y = (m1 ** 2) * s1y
    s2_perp_x = (m2 ** 2) * s2x
    s2_perp_y = (m2 ** 2) * s2y

    stot_x = s1_perp_x + s2_perp_x
    stot_y = s1_perp_y + s2_perp_y
    stot_norm = torch.sqrt(stot_x ** 2 + stot_y ** 2)

    chi_tot_perp = stot_norm * (total_m ** 2) / torch.clamp(m1 ** 4, min=1e-24)
    return torch.clamp(chi_tot_perp, 0.0, 1.0)


class SpinTaylorT4Engine:
    """Orbit-averaged SpinTaylorT4 numerical precession integrator in pure PyTorch."""

    def __init__(
        self,
        mass1: Union[float, torch.Tensor],
        mass2: Union[float, torch.Tensor],
        spin1: Union[Sequence[float], torch.Tensor],
        spin2: Union[Sequence[float], torch.Tensor],
        f_ref: Union[float, torch.Tensor] = 20.0,
    ):
        self.m1 = as_tensor(mass1)
        self.m2 = as_tensor(mass2)
        self.total_m = self.m1 + self.m2
        self.eta = (self.m1 * self.m2) / (self.total_m * self.total_m)
        self.m1_m = self.m1 / self.total_m
        self.m2_m = self.m2 / self.total_m
        self.m_sec = self.total_m * MSUN_SEC
        self.f_ref = as_tensor(f_ref)
        self.v_ref = (PI * self.m_sec * self.f_ref).pow(1.0 / 3.0)

        s1_t = as_tensor(spin1)
        s2_t = as_tensor(spin2)
        self.spin1 = s1_t
        self.spin2 = s2_t

    def integrate_nodes(
        self,
        v_min: float,
        v_max: float,
        num_steps: int = 150,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Integrate SpinTaylor trajectory on velocity nodes from v_min to v_max."""
        device = self.m1.device
        dtype = self.m1.dtype
        v_nodes = torch.linspace(v_min, v_max, num_steps, dtype=dtype, device=device)
        h = (v_max - v_min) / float(num_steps - 1)

        # Scale spins by (m_i/M)^2 internally
        s1_0 = self.spin1 * (self.m1_m ** 2)
        s2_0 = self.spin2 * (self.m2_m ** 2)
        s1_norm_0 = torch.linalg.vector_norm(s1_0, dim=-1, keepdim=True)
        s2_norm_0 = torch.linalg.vector_norm(s2_0, dim=-1, keepdim=True)

        s1_curr = s1_0.clone()
        s2_curr = s2_0.clone()
        lnh_curr = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)

        lnh_traj = [lnh_curr]
        s1_traj = [s1_curr / (self.m1_m ** 2)]
        s2_traj = [s2_curr / (self.m2_m ** 2)]

        m1_m = self.m1_m
        m2_m = self.m2_m
        eta = self.eta

        for i in range(num_steps - 1):
            v_cur = v_nodes[i]

            _, k1_l, k1_s1, k1_s2 = self.derivatives(v_cur, lnh_curr, s1_curr, s2_curr, m1_m, m2_m, eta)

            v_half = v_cur + 0.5 * h
            l2 = lnh_curr + 0.5 * h * k1_l
            l2 = l2 / torch.linalg.vector_norm(l2, dim=-1, keepdim=True).clamp_min(1e-24)
            s1_2 = s1_curr + 0.5 * h * k1_s1
            s2_2 = s2_curr + 0.5 * h * k1_s2
            _, k2_l, k2_s1, k2_s2 = self.derivatives(v_half, l2, s1_2, s2_2, m1_m, m2_m, eta)

            l3 = lnh_curr + 0.5 * h * k2_l
            l3 = l3 / torch.linalg.vector_norm(l3, dim=-1, keepdim=True).clamp_min(1e-24)
            s1_3 = s1_curr + 0.5 * h * k2_s1
            s2_3 = s2_curr + 0.5 * h * k2_s2
            _, k3_l, k3_s1, k3_s2 = self.derivatives(v_half, l3, s1_3, s2_3, m1_m, m2_m, eta)

            v_full = v_cur + h
            l4 = lnh_curr + h * k3_l
            l4 = l4 / torch.linalg.vector_norm(l4, dim=-1, keepdim=True).clamp_min(1e-24)
            s1_4 = s1_curr + h * k3_s1
            s2_4 = s2_curr + h * k3_s2
            _, k4_l, k4_s1, k4_s2 = self.derivatives(v_full, l4, s1_4, s2_4, m1_m, m2_m, eta)

            lnh_curr = lnh_curr + (h / 6.0) * (k1_l + 2.0 * k2_l + 2.0 * k3_l + k4_l)
            lnh_curr = lnh_curr / torch.linalg.vector_norm(lnh_curr, dim=-1, keepdim=True).clamp_min(1e-24)
            s1_curr = s1_curr + (h / 6.0) * (k1_s1 + 2.0 * k2_s1 + 2.0 * k3_s1 + k4_s1)
            s2_curr = s2_curr + (h / 6.0) * (k1_s2 + 2.0 * k2_s2 + 2.0 * k3_s2 + k4_s2)

            # Strict conservation of spin norms along trajectory
            s1_norm = torch.linalg.vector_norm(s1_curr, dim=-1, keepdim=True).clamp_min(1e-24)
            s2_norm = torch.linalg.vector_norm(s2_curr, dim=-1, keepdim=True).clamp_min(1e-24)
            s1_curr = s1_curr * (s1_norm_0 / s1_norm)
            s2_curr = s2_curr * (s2_norm_0 / s2_norm)

            lnh_traj.append(lnh_curr)
            s1_traj.append(s1_curr / (self.m1_m ** 2))
            s2_traj.append(s2_curr / (self.m2_m ** 2))

        return (
            v_nodes,
            torch.stack(lnh_traj, dim=0),
            torch.stack(s1_traj, dim=0),
            torch.stack(s2_traj, dim=0),
        )

    @staticmethod
    def derivatives(
        v: torch.Tensor,
        lnh: torch.Tensor,
        s1: torch.Tensor,
        s2: torch.Tensor,
        m1_m: torch.Tensor,
        m2_m: torch.Tensor,
        eta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute state derivatives with respect to velocity v: (dt_dv, dlnh_dv, ds1_dv, ds2_dv)."""
        v2 = v * v
        v3 = v2 * v
        v4 = v2 * v2
        v5 = v4 * v
        v6 = v3 * v3
        v7 = v4 * v3
        v11 = v6 * v5

        l_dot_s1 = torch.sum(lnh * s1, dim=-1, keepdim=True)
        l_dot_s2 = torch.sum(lnh * s2, dim=-1, keepdim=True)
        s1_dot_s2 = torch.sum(s1 * s2, dim=-1, keepdim=True)
        s1_sq = torch.sum(s1 * s1, dim=-1, keepdim=True)
        s2_sq = torch.sum(s2 * s2, dim=-1, keepdim=True)

        a0 = 1.0
        a2 = -(743.0 + 924.0 * eta) / 336.0
        a3 = 4.0 * PI
        w3 = (-19.0 / 6.0 - 25.0 / (4.0 * m1_m)) * l_dot_s1 + (-19.0 / 6.0 - 25.0 / (4.0 * m2_m)) * l_dot_s2

        a4 = (34103.0 + 122949.0 * eta + 59472.0 * (eta ** 2)) / 18144.0
        w4 = (
            -(247.0 / (48.0 * eta)) * s1_dot_s2
            + (721.0 / (48.0 * eta)) * l_dot_s1 * l_dot_s2
            + (7.0 / (96.0 * m1_m**2) - 2.5 / m1_m**2) * s1_sq
            + (-1.0 / (96.0 * m1_m**2) + 7.5 / m1_m**2) * (l_dot_s1 ** 2)
            + (7.0 / (96.0 * m2_m**2) - 2.5 / m2_m**2) * s2_sq
            + (-1.0 / (96.0 * m2_m**2) + 7.5 / m2_m**2) * (l_dot_s2 ** 2)
        )

        a5 = -PI * (4159.0 + 15876.0 * eta) / 672.0
        w5 = (
            (-809.0 / (84.0 * m1_m) + 13.795 / 1.008 - 527.0 * m1_m / 24.0 - 79.0 * (m1_m ** 2) / 6.0) * l_dot_s1
            + (-809.0 / (84.0 * m2_m) + 13.795 / 1.008 - 527.0 * m2_m / 24.0 - 79.0 * (m2_m ** 2) / 6.0) * l_dot_s2
        )

        a6 = (
            16447.322263 / 139.7088
            - (1712.0 / 105.0) * EULER_GAMMA
            - (561.98689 / 2.17728) * eta
            + (PI ** 2) * (16.0 / 3.0 + 451.0 * eta / 48.0)
            + (541.0 / 896.0) * (eta ** 2)
            - (5605.0 / 2592.0) * (eta ** 3)
            - (856.0 / 105.0) * math.log(16.0)
        )
        alog = -(1712.0 / 105.0)
        w6 = PI * ((-37.0 / 3.0 - 151.0 / (6.0 * m1_m)) * l_dot_s1 + (-37.0 / 3.0 - 151.0 / (6.0 * m2_m)) * l_dot_s2)

        a7 = (PI / 12096.0) * (-13245.0 + 717350.0 * eta + 731960.0 * (eta ** 2))

        p_v = (
            a0
            + v * (
                0.0
                + v * (
                    a2
                    + v * (
                        a3
                        + w3
                        + v * (
                            a4
                            + w4
                            + v * (
                                a5
                                + w5
                                + v * (a6 + w6 + alog * torch.log(torch.clamp(v, min=1e-12)) + v * a7)
                            )
                        )
                    )
                )
            )
        )
        domega_dt = (96.0 / 5.0) * eta * v11 * p_v
        dt_dv = 3.0 * v2 / torch.clamp(domega_dt, min=1e-30)

        # Spin precession equations
        lnh_cross_s1 = torch.linalg.cross(lnh, s1)
        lnh_cross_s2 = torch.linalg.cross(lnh, s2)
        s1_cross_s2 = torch.linalg.cross(s1, s2)

        # LO SO (v5)
        c_s1_lo = 1.5 - m1_m - 0.5 * (m1_m ** 2)
        c_s2_lo = 1.5 - m2_m - 0.5 * (m2_m ** 2)
        ds1_lo = c_s1_lo * v5 * lnh_cross_s1
        ds2_lo = c_s2_lo * v5 * lnh_cross_s2

        # NLO SS (v6)
        ds1_nl = v6 * (
            -0.5 * s1_cross_s2
            - 1.5 * l_dot_s2 * lnh_cross_s1
            + 1.5 * (1.0 - 1.0 / m1_m) * l_dot_s1 * lnh_cross_s1
        )
        ds2_nl = v6 * (
            0.5 * s1_cross_s2
            - 1.5 * l_dot_s1 * lnh_cross_s2
            + 1.5 * (1.0 - 1.0 / m2_m) * l_dot_s2 * lnh_cross_s2
        )

        # NNLO SO (v7)
        c1_nnlo = 9.0 / 8.0 - 0.5 * m1_m + (7.0 / 12.0) * (m1_m**2) - (7.0 / 6.0) * (m1_m**3) - (m1_m**4) / 24.0
        c2_nnlo = 9.0 / 8.0 - 0.5 * m2_m + (7.0 / 12.0) * (m2_m**2) - (7.0 / 6.0) * (m2_m**3) - (m2_m**4) / 24.0
        ds1_nnlo = c1_nnlo * v7 * lnh_cross_s1
        ds2_nnlo = c2_nnlo * v7 * lnh_cross_s2

        ds1_dt = ds1_lo + ds1_nl + ds1_nnlo
        ds2_dt = ds2_lo + ds2_nl + ds2_nnlo

        l_mag = eta / v
        dl_dt = -(ds1_dt + ds2_dt)
        omega_l = torch.linalg.cross(lnh, dl_dt / torch.clamp(l_mag, min=1e-24))
        dlnh_dt = torch.linalg.cross(omega_l, lnh)

        ds1_dv = ds1_dt * dt_dv
        ds2_dv = ds2_dt * dt_dv
        dlnh_dv = dlnh_dt * dt_dv

        return dt_dv, dlnh_dv, ds1_dv, ds2_dv


def evaluate_spintaylor_precession(
    f: torch.Tensor,
    f_ref: Union[float, torch.Tensor],
    mass1: Union[float, torch.Tensor],
    mass2: Union[float, torch.Tensor],
    spin1x: Union[float, torch.Tensor],
    spin1y: Union[float, torch.Tensor],
    spin1z: Union[float, torch.Tensor],
    spin2x: Union[float, torch.Tensor],
    spin2y: Union[float, torch.Tensor],
    spin2z: Union[float, torch.Tensor],
    inclination: Union[float, torch.Tensor],
    phi_ref: Union[float, torch.Tensor],
    convention: int = 1,
    observer_azimuth: Optional[Union[float, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate SpinTaylor precession Euler angles and frame geometry across frequency grid f.

    Returns:
        (alpha, beta, gamma, theta_jn, zeta, carrier_phase)
    """
    f_ref_t = as_tensor(f_ref).to(device=f.device, dtype=f.dtype)
    m1_t = as_tensor(mass1).to(device=f.device, dtype=f.dtype)
    m2_t = as_tensor(mass2).to(device=f.device, dtype=f.dtype)
    s1x_t = as_tensor(spin1x).to(device=f.device, dtype=f.dtype)
    s1y_t = as_tensor(spin1y).to(device=f.device, dtype=f.dtype)
    s1z_t = as_tensor(spin1z).to(device=f.device, dtype=f.dtype)
    s2x_t = as_tensor(spin2x).to(device=f.device, dtype=f.dtype)
    s2y_t = as_tensor(spin2y).to(device=f.device, dtype=f.dtype)
    s2z_t = as_tensor(spin2z).to(device=f.device, dtype=f.dtype)
    inc_t = as_tensor(inclination).to(device=f.device, dtype=f.dtype)
    phi_c_t = as_tensor(phi_ref).to(device=f.device, dtype=f.dtype)

    is_batched = m1_t.dim() > 0 and m1_t.shape[0] > 1

    total_m = m1_t + m2_t
    eta = (m1_t * m2_t) / (total_m * total_m)
    delta = (m1_t - m2_t) / total_m
    m1_m = m1_t / total_m
    m2_m = m2_t / total_m
    m_sec = total_m * MSUN_SEC

    # Scaled spins
    s1_0 = (m1_m ** 2).unsqueeze(-1) * torch.stack([s1x_t, s1y_t, s1z_t], dim=-1)
    s2_0 = (m2_m ** 2).unsqueeze(-1) * torch.stack([s2x_t, s2y_t, s2z_t], dim=-1)
    lnh_0 = torch.zeros_like(s1_0)
    lnh_0[..., 2] = 1.0

    v0 = (PI * m_sec * f_ref_t).pow(1.0 / 3.0)
    v2_0 = v0 * v0
    v3_0 = v2_0 * v0
    v4_0 = v2_0 * v2_0
    v5_0 = v4_0 * v0
    v6_0 = v3_0 * v3_0

    # 3PN reference orbital angular momentum (L0 through L6)
    L0 = 1.0
    L1 = 0.0
    L2 = 3.0 / 2.0 + eta / 6.0
    L3 = (5.0 * (s1z_t * (-2.0 - 2.0 * delta + eta) + s2z_t * (-2.0 + 2.0 * delta + eta))) / 6.0
    L4 = (81.0 + (-57.0 + eta) * eta) / 24.0
    L5 = (
        -7.0 * (
            s1z_t * (72.0 + delta * (72.0 - 31.0 * eta) + eta * (-121.0 + 2.0 * eta))
            + s2z_t * (72.0 + eta * (-121.0 + 2.0 * eta) + delta * (-72.0 + 31.0 * eta))
        )
    ) / 144.0
    L6 = (10935.0 + eta * (-62001.0 + eta * (1674.0 + 7.0 * eta) + 2214.0 * (PI ** 2))) / 1296.0

    l_mag0 = (eta / v0) * (L0 + L1 * v0 + L2 * v2_0 + L3 * v3_0 + L4 * v4_0 + L5 * v5_0 + L6 * v6_0)
    j0 = l_mag0.unsqueeze(-1) * lnh_0 + s1_0 + s2_0
    j0_mag = torch.linalg.vector_norm(j0, dim=-1, keepdim=True).clamp_min(1e-24)
    z_j = j0 / j0_mag

    # J-frame Euler rotation setup
    phiJ = torch.atan2(j0[..., 1], j0[..., 0])
    thetaJ = torch.acos(torch.clamp(j0[..., 2] / j0_mag.squeeze(-1), -1.0, 1.0))

    # Line of sight in source frame
    sin_i = torch.sin(inc_t)
    cos_i = torch.cos(inc_t)
    sin_p = torch.sin(phi_c_t)
    cos_p = torch.cos(phi_c_t)

    # In LAL, N_sf = (sin_i * sin(phi_ref), sin_i * cos(phi_ref), cos_i)
    n_src = torch.stack([sin_i * sin_p, sin_i * cos_p, cos_i], dim=-1)

    c_p = torch.cos(phiJ)
    s_p = torch.sin(phiJ)
    c_t = torch.cos(thetaJ)
    s_t = torch.sin(thetaJ)

    # Rotate N by Rz(-phiJ) then Ry(-thetaJ)
    n1_x = c_p * n_src[..., 0] + s_p * n_src[..., 1]
    n1_y = -s_p * n_src[..., 0] + c_p * n_src[..., 1]
    n1_z = n_src[..., 2]

    n2_x = c_t * n1_x - s_t * n1_z
    n2_y = n1_y
    n2_z = s_t * n1_x + c_t * n1_z

    kappa = torch.atan2(n2_y, n2_x)
    c_k = torch.cos(kappa)
    s_k = torch.sin(kappa)

    # Angle between J0 and N
    cos_theta_jn = torch.clamp(torch.sum(z_j * n_src, dim=-1), -1.0, 1.0)
    theta_jn = torch.acos(cos_theta_jn)

    # Polarization angle zeta
    # X_sf = (-cos_i * sin(phi_ref), -cos_i * cos(phi_ref), sin_i)
    x_src = torch.stack([-cos_i * sin_p, -cos_i * cos_p, sin_i], dim=-1)
    x1_x = c_p * x_src[..., 0] + s_p * x_src[..., 1]
    x1_y = -s_p * x_src[..., 0] + c_p * x_src[..., 1]
    x1_z = x_src[..., 2]

    x2_x = c_t * x1_x - s_t * x1_z
    x2_y = x1_y
    x2_z = s_t * x1_x + c_t * x1_z

    x3_x = c_k * x2_x + s_k * x2_y
    x3_y = -s_k * x2_x + c_k * x2_y
    x3_z = x2_z

    if convention == 1:
        # P = (Nz, 0, -Nx), Q = (0, 1, 0)
        x_dot_p = x3_x * cos_theta_jn - x3_z * torch.sin(theta_jn)
        x_dot_q = x3_y
        zeta = torch.atan2(x_dot_q, x_dot_p)
        alpha_0 = PI - kappa
        epsilon_0 = phiJ - PI
        carrier_phase = torch.zeros_like(phi_c_t)
    else:
        zeta = kappa
        alpha_0 = torch.zeros_like(phi_c_t)
        epsilon_0 = torch.zeros_like(phi_c_t)
        carrier_phase = -phiJ

    def rotate_to_j(vec: torch.Tensor) -> torch.Tensor:
        vx, vy, vz = vec[..., 0], vec[..., 1], vec[..., 2]
        c_pu = c_p.unsqueeze(-1) if c_p.dim() < vx.dim() else c_p
        s_pu = s_p.unsqueeze(-1) if s_p.dim() < vx.dim() else s_p
        c_tu = c_t.unsqueeze(-1) if c_t.dim() < vx.dim() else c_t
        s_tu = s_t.unsqueeze(-1) if s_t.dim() < vx.dim() else s_t
        c_ku = c_k.unsqueeze(-1) if c_k.dim() < vx.dim() else c_k
        s_ku = s_k.unsqueeze(-1) if s_k.dim() < vx.dim() else s_k

        v1x = c_pu * vx + s_pu * vy
        v1y = -s_pu * vx + c_pu * vy
        v1z = vz

        v2x = c_tu * v1x - s_tu * v1z
        v2y = v1y
        v2z = s_tu * v1x + c_tu * v1z

        v3x = c_ku * v2x + s_ku * v2y
        v3y = -s_ku * v2x + c_ku * v2y
        v3z = v2z
        return torch.stack([v3x, v3y, v3z], dim=-1)

    # Transition frequency f_meco and f_trans
    mf_meco = XASParameterFits.fit_f_meco(eta, s1z_t, s2z_t)
    f_meco = mf_meco / torch.clamp(m_sec, min=1e-24)
    f_meco = torch.clamp(f_meco, min=f_ref_t + 15.0, max=800.0)
    f_trans = 0.98 * f_meco
    v_trans = (PI * m_sec * f_trans).pow(1.0 / 3.0)

    # Target frequency bounds (physical integration down to seismic band)
    f_eval = f if f.dim() > 0 else f.unsqueeze(0)
    if is_batched and f_eval.dim() == 1:
        f_eval = f_eval.unsqueeze(0)

    pos_f_min = torch.where(f_eval > 0, f_eval, torch.full_like(f_eval, float("inf"))).amin()
    f_min_grid = torch.where(torch.isfinite(pos_f_min), pos_f_min, f_ref_t)
    # Clamp minimum physical frequency for ODE stability in PN band
    f_min_eval = torch.minimum(f_ref_t, torch.clamp(f_min_grid, min=15.0))
    v_min = (PI * m_sec * f_min_eval).pow(1.0 / 3.0)

    # Integrate ODE on velocity grid
    n_nodes = 200
    v_grid = torch.linspace(0.0, 1.0, n_nodes, dtype=f.dtype, device=f.device)
    v_nodes = v_min.unsqueeze(-1) + v_grid * (v_trans - v_min).unsqueeze(-1)

    s1_norm_0 = torch.linalg.vector_norm(s1_0, dim=-1, keepdim=True)
    s2_norm_0 = torch.linalg.vector_norm(s2_0, dim=-1, keepdim=True)

    lnh_curr = lnh_0
    s1_curr = s1_0.clone()
    s2_curr = s2_0.clone()

    lnh_traj = [lnh_curr]
    h_step = (v_trans - v_min) / float(n_nodes - 1)
    m1_exp = m1_m.unsqueeze(-1) if m1_m.dim() > 0 else m1_m
    m2_exp = m2_m.unsqueeze(-1) if m2_m.dim() > 0 else m2_m
    eta_exp = eta.unsqueeze(-1) if eta.dim() > 0 else eta

    for i in range(n_nodes - 1):
        v_cur = v_nodes[..., i].unsqueeze(-1)
        h = h_step.unsqueeze(-1)

        def step_derivs(v_in, l_in, s1_in, s2_in):
            return SpinTaylorT4Engine.derivatives(v_in, l_in, s1_in, s2_in, m1_exp, m2_exp, eta_exp)

        _, k1_l, k1_s1, k1_s2 = step_derivs(v_cur, lnh_curr, s1_curr, s2_curr)

        v_half = v_cur + 0.5 * h
        l2 = lnh_curr + 0.5 * h * k1_l
        l2 = l2 / torch.linalg.vector_norm(l2, dim=-1, keepdim=True).clamp_min(1e-24)
        s1_2 = s1_curr + 0.5 * h * k1_s1
        s2_2 = s2_curr + 0.5 * h * k1_s2
        _, k2_l, k2_s1, k2_s2 = step_derivs(v_half, l2, s1_2, s2_2)

        l3 = lnh_curr + 0.5 * h * k2_l
        l3 = l3 / torch.linalg.vector_norm(l3, dim=-1, keepdim=True).clamp_min(1e-24)
        s1_3 = s1_curr + 0.5 * h * k2_s1
        s2_3 = s2_curr + 0.5 * h * k2_s2
        _, k3_l, k3_s1, k3_s2 = step_derivs(v_half, l3, s1_3, s2_3)

        v_full = v_cur + h
        l4 = lnh_curr + h * k3_l
        l4 = l4 / torch.linalg.vector_norm(l4, dim=-1, keepdim=True).clamp_min(1e-24)
        s1_4 = s1_curr + h * k3_s1
        s2_4 = s2_curr + h * k3_s2
        _, k4_l, k4_s1, k4_s2 = step_derivs(v_full, l4, s1_4, s2_4)

        lnh_curr = lnh_curr + (h / 6.0) * (k1_l + 2.0 * k2_l + 2.0 * k3_l + k4_l)
        lnh_curr = lnh_curr / torch.linalg.vector_norm(lnh_curr, dim=-1, keepdim=True).clamp_min(1e-24)
        s1_curr = s1_curr + (h / 6.0) * (k1_s1 + 2.0 * k2_s1 + 2.0 * k3_s1 + k4_s1)
        s2_curr = s2_curr + (h / 6.0) * (k1_s2 + 2.0 * k2_s2 + 2.0 * k3_s2 + k4_s2)

        s1_norm = torch.linalg.vector_norm(s1_curr, dim=-1, keepdim=True).clamp_min(1e-24)
        s2_norm = torch.linalg.vector_norm(s2_curr, dim=-1, keepdim=True).clamp_min(1e-24)
        s1_curr = s1_curr * (s1_norm_0 / s1_norm)
        s2_curr = s2_curr * (s2_norm_0 / s2_norm)

        lnh_traj.append(lnh_curr)

    lnh_all = torch.stack(lnh_traj, dim=-2)
    f_nodes = (v_nodes ** 3) / (PI * m_sec.unsqueeze(-1) if m_sec.dim() > 0 else PI * m_sec)

    # Rotate to J-frame
    lnh_j_nodes = rotate_to_j(lnh_all)
    alpha_raw = torch.atan2(lnh_j_nodes[..., 1], lnh_j_nodes[..., 0])

    diff_alpha = alpha_raw[..., 1:] - alpha_raw[..., :-1]
    diff_alpha_unwrapped = torch.remainder(diff_alpha + PI, 2.0 * PI) - PI
    alpha_nodes = torch.cat([
        alpha_raw[..., :1],
        alpha_raw[..., :1] + torch.cumsum(diff_alpha_unwrapped, dim=-1)
    ], dim=-1)
    cosbeta_nodes = torch.clamp(lnh_j_nodes[..., 2], -1.0, 1.0)
    beta_nodes = torch.acos(cosbeta_nodes)

    # Gamma evaluation from minimal rotation condition: dgamma = -cos(beta) * dalpha
    dalpha_step = alpha_nodes[..., 1:] - alpha_nodes[..., :-1]
    cosbeta_mid = 0.5 * (cosbeta_nodes[..., 1:] + cosbeta_nodes[..., :-1])
    dgamma_step = -cosbeta_mid * dalpha_step
    gamma_nodes = torch.cat([
        torch.zeros_like(alpha_nodes[..., :1]),
        torch.cumsum(dgamma_step, dim=-1)
    ], dim=-1)

    # Shape preparation for broadcasting
    f_trans_exp = f_trans.unsqueeze(-1) if is_batched or f_trans.dim() < f_eval.dim() else f_trans
    m_sec_exp = m_sec.unsqueeze(-1) if is_batched or m_sec.dim() < f_eval.dim() else m_sec
    v_min_exp = v_min.unsqueeze(-1) if is_batched or v_min.dim() < f_eval.dim() else v_min
    v_trans_exp = v_trans.unsqueeze(-1) if is_batched or v_trans.dim() < f_eval.dim() else v_trans

    # Interpolation indices in v-space (v varies linearly on nodes)
    v_eval = (PI * m_sec_exp * torch.clamp(f_eval, min=1e-12)).pow(1.0 / 3.0)
    u_v = (v_eval - v_min_exp) / torch.clamp(v_trans_exp - v_min_exp, min=1e-12)
    u_v = torch.clamp(u_v, 0.0, 1.0)
    idx_float = u_v * float(n_nodes - 1)
    idx_low = torch.clamp(idx_float.to(torch.int64), 0, n_nodes - 2)
    t_frac = (idx_float - idx_low.to(f.dtype)).clamp(0.0, 1.0)

    def gather_val(nodes_t):
        if nodes_t.dim() > 1 and nodes_t.shape[:-1] == idx_low.shape[:-1]:
            val_low = torch.gather(nodes_t, -1, idx_low)
            val_high = torch.gather(nodes_t, -1, idx_low + 1)
        elif nodes_t.dim() > 1 and is_batched:
            val_low = torch.gather(nodes_t, -1, idx_low)
            val_high = torch.gather(nodes_t, -1, idx_low + 1)
        else:
            val_low = nodes_t[..., idx_low]
            val_high = nodes_t[..., idx_low + 1]
        return val_low * (1.0 - t_frac) + val_high * t_frac

    alpha_insp = gather_val(alpha_nodes)
    cosb_insp = gather_val(cosbeta_nodes)
    gamma_insp = gather_val(gamma_nodes)

    # MRD analytical continuation for f >= f_trans
    mf_nodes = (v_nodes ** 3) / PI
    idx1 = int(0.97 * (n_nodes - 1))
    idx2 = int(0.99 * (n_nodes - 1))
    f1 = mf_nodes[..., idx1]
    f2 = mf_nodes[..., idx2]
    a1 = -alpha_nodes[..., idx1]
    a2 = -alpha_nodes[..., idx2]

    df_step = f2 - f1
    da1 = -(alpha_nodes[..., idx2] - alpha_nodes[..., idx1]) / torch.clamp(df_step, min=1e-12)

    f1sq, f2sq = f1 * f1, f2 * f2
    f1cube = f1sq * f1
    den_a = 2.0 * ((f1sq - f2sq) ** 2).clamp_min(1e-24)
    aC = (f1cube * (f1 - f2) * (f1 + f2) * da1 + 2.0 * (f1**4 - 2.0 * f1sq * f2sq) * a1 + 2.0 * (f2**4) * a2) / den_a
    bC = ((f1**4) * f2sq * (f1 * (f1 - f2) * (f1 + f2) * da1 + 2.0 * f2sq * (-a1 + a2))) / den_a
    cC = (f1sq * (f1 * (-(f1**4) + f2**4) * da1 + 4.0 * (f2**4) * (a1 - a2))) / den_a

    mf_eval = f_eval * m_sec_exp
    mf2 = mf_eval * mf_eval
    mf4 = mf2 * mf2
    aC_u = aC.unsqueeze(-1) if is_batched or aC.dim() < f_eval.dim() else aC
    bC_u = bC.unsqueeze(-1) if is_batched or bC.dim() < f_eval.dim() else bC
    cC_u = cC.unsqueeze(-1) if is_batched or cC.dim() < f_eval.dim() else cC

    alpha_mrd = -(aC_u + bC_u / torch.clamp(mf4, min=1e-16) + cC_u / torch.clamp(mf2, min=1e-8))
    cosb_mrd = cosb_insp.clone()
    gamma_mrd = gamma_insp - cosb_insp * (alpha_mrd - alpha_insp)

    is_mrd = f_eval >= f_trans_exp
    alpha_out = torch.where(is_mrd, alpha_mrd, alpha_insp)
    cosb_out = torch.where(is_mrd, cosb_mrd, cosb_insp)
    gamma_out = torch.where(is_mrd, gamma_mrd, gamma_insp)
    beta_out = torch.acos(torch.clamp(cosb_out, -1.0, 1.0))

    # Reference value offset alignment at f_ref
    u_ref = (v0 - v_min) / torch.clamp(v_trans - v_min, min=1e-12)
    idx_ref = torch.clamp((u_ref * float(n_nodes - 1)).to(torch.int64), 0, n_nodes - 2)
    idx_g = idx_ref.unsqueeze(-1)
    alpha_ref_val = torch.gather(alpha_nodes, -1, idx_g).squeeze(-1)
    gamma_ref_val = torch.gather(gamma_nodes, -1, idx_g).squeeze(-1)

    # In LAL, alpha(f_ref) = 0 for SpinTaylor
    alpha_offset = -alpha_ref_val
    alpha_offset_u = alpha_offset.unsqueeze(-1) if is_batched or alpha_offset.dim() < alpha_out.dim() else alpha_offset
    alpha_final = alpha_out + alpha_offset_u

    # Return gamma = -epsilon to match MSA convention in twisting-up:
    # gamma(f) = -epsilon(f) = (gamma(f) - gamma_ref) - epsilon_0
    gamma_offset = -gamma_ref_val - epsilon_0
    gamma_offset_u = gamma_offset.unsqueeze(-1) if is_batched or gamma_offset.dim() < gamma_out.dim() else gamma_offset
    gamma_final = gamma_out + gamma_offset_u

    if not is_batched and alpha_final.dim() > f.dim():
        alpha_final = alpha_final.squeeze(0)
        beta_out = beta_out.squeeze(0)
        gamma_final = gamma_final.squeeze(0)

    return alpha_final, beta_out, gamma_final, theta_jn, zeta, carrier_phase

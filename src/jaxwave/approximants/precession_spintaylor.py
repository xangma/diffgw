"""Pure JAX SpinTaylorT4 spin-precession dynamics and Euler angle engine.

Provides a native JAX implementation of the numerical integration of 3.5PN
orbit-averaged SpinTaylorT4 equations of motion, computation of J-frame Euler
precession angles (alpha, beta, gamma), and smooth C^1 analytical continuation
into the merger-ringdown regime.

References:
- Colleoni et al. (2025), Phys. Rev. D 111, 104019 (arXiv:2412.16721)
- Buonanno et al. (2009), Phys. Rev. D 80, 084043 (arXiv:0907.0700)
- Blanchet et al. (2006), Phys. Rev. D 74, 104034 (arXiv:gr-qc/0605140)
- Kidder (1995), Phys. Rev. D 52, 821 (arXiv:gr-qc/9506014)
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503)
"""

from typing import Optional, Sequence, Tuple, Union
import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    MSUN_SEC,
    PI,
    EULER_GAMMA,
)
from jaxwave.conversions import as_array
from jaxwave.approximants.imrphenom_xas import XASParameterFits


def compute_chi_tot_perp(
    m1: Union[float, jnp.ndarray],
    m2: Union[float, jnp.ndarray],
    s1x: Union[float, jnp.ndarray],
    s1y: Union[float, jnp.ndarray],
    s2x: Union[float, jnp.ndarray],
    s2y: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """Compute the vectorial in-plane spin parameter chi_tot_perp in JAX.

    Norm of the in-plane vector sum S_tot_perp = Norm[ S1_perp + S2_perp ]
    normalized by m1^2 / M^2. Ref: Colleoni et al. (2025), Eq. (26) &
    Pratten et al. (2021), Sec. IV D.
    """
    m1_arr = as_array(m1, dtype=jnp.float64)
    m2_arr = as_array(m2, dtype=jnp.float64)
    s1x_arr = as_array(s1x, dtype=jnp.float64)
    s1y_arr = as_array(s1y, dtype=jnp.float64)
    s2x_arr = as_array(s2x, dtype=jnp.float64)
    s2y_arr = as_array(s2y, dtype=jnp.float64)

    total_m = m1_arr + m2_arr
    s1_perp_x = (m1_arr ** 2) * s1x_arr
    s1_perp_y = (m1_arr ** 2) * s1y_arr
    s2_perp_x = (m2_arr ** 2) * s2x_arr
    s2_perp_y = (m2_arr ** 2) * s2y_arr

    stot_x = s1_perp_x + s2_perp_x
    stot_y = s1_perp_y + s2_perp_y
    stot_norm = jnp.sqrt(stot_x ** 2 + stot_y ** 2)

    chi_tot_perp = stot_norm * (total_m ** 2) / jnp.maximum(m1_arr ** 4, 1e-24)
    return jnp.clip(chi_tot_perp, 0.0, 1.0)


class SpinTaylorT4Engine:
    """Orbit-averaged SpinTaylorT4 numerical precession integrator in pure JAX."""

    def __init__(
        self,
        mass1: Union[float, jnp.ndarray],
        mass2: Union[float, jnp.ndarray],
        spin1: Union[Sequence[float], jnp.ndarray],
        spin2: Union[Sequence[float], jnp.ndarray],
        f_ref: Union[float, jnp.ndarray] = 20.0,
    ):
        self.m1 = as_array(mass1, dtype=jnp.float64)
        self.m2 = as_array(mass2, dtype=jnp.float64)
        self.total_m = self.m1 + self.m2
        self.eta = (self.m1 * self.m2) / (self.total_m * self.total_m)
        self.m1_m = self.m1 / self.total_m
        self.m2_m = self.m2 / self.total_m
        self.m_sec = self.total_m * MSUN_SEC
        self.f_ref = as_array(f_ref, dtype=jnp.float64)
        self.v_ref = (PI * self.m_sec * self.f_ref) ** (1.0 / 3.0)

        self.spin1 = as_array(spin1, dtype=jnp.float64)
        self.spin2 = as_array(spin2, dtype=jnp.float64)

    def integrate_nodes(
        self,
        v_min: float,
        v_max: float,
        num_steps: int = 150,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Integrate SpinTaylor trajectory on velocity nodes from v_min to v_max."""
        v_nodes = jnp.linspace(v_min, v_max, num_steps, dtype=jnp.float64)
        h = (v_max - v_min) / float(num_steps - 1)

        # Scale spins by (m_i/M)^2 internally
        s1_0 = self.spin1 * (self.m1_m ** 2)
        s2_0 = self.spin2 * (self.m2_m ** 2)
        s1_norm_0 = jnp.linalg.norm(s1_0, axis=-1, keepdims=True)
        s2_norm_0 = jnp.linalg.norm(s2_0, axis=-1, keepdims=True)

        s1_curr = s1_0
        s2_curr = s2_0
        lnh_curr = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)

        m1_m = self.m1_m
        m2_m = self.m2_m
        eta = self.eta

        def step_fn(state, v_cur):
            lnh_c, s1_c, s2_c = state

            _, k1_l, k1_s1, k1_s2 = self.derivatives(v_cur, lnh_c, s1_c, s2_c, m1_m, m2_m, eta)

            v_half = v_cur + 0.5 * h
            l2 = lnh_c + 0.5 * h * k1_l
            l2 = l2 / jnp.maximum(jnp.linalg.norm(l2, axis=-1, keepdims=True), 1e-24)
            s1_2 = s1_c + 0.5 * h * k1_s1
            s2_2 = s2_c + 0.5 * h * k1_s2
            _, k2_l, k2_s1, k2_s2 = self.derivatives(v_half, l2, s1_2, s2_2, m1_m, m2_m, eta)

            l3 = lnh_c + 0.5 * h * k2_l
            l3 = l3 / jnp.maximum(jnp.linalg.norm(l3, axis=-1, keepdims=True), 1e-24)
            s1_3 = s1_c + 0.5 * h * k2_s1
            s2_3 = s2_c + 0.5 * h * k2_s2
            _, k3_l, k3_s1, k3_s2 = self.derivatives(v_half, l3, s1_3, s2_3, m1_m, m2_m, eta)

            v_full = v_cur + h
            l4 = lnh_c + h * k3_l
            l4 = l4 / jnp.maximum(jnp.linalg.norm(l4, axis=-1, keepdims=True), 1e-24)
            s1_4 = s1_c + h * k3_s1
            s2_4 = s2_c + h * k3_s2
            _, k4_l, k4_s1, k4_s2 = self.derivatives(v_full, l4, s1_4, s2_4, m1_m, m2_m, eta)

            lnh_next = lnh_c + (h / 6.0) * (k1_l + 2.0 * k2_l + 2.0 * k3_l + k4_l)
            lnh_next = lnh_next / jnp.maximum(jnp.linalg.norm(lnh_next, axis=-1, keepdims=True), 1e-24)
            s1_next = s1_c + (h / 6.0) * (k1_s1 + 2.0 * k2_s1 + 2.0 * k3_s1 + k4_s1)
            s2_next = s2_c + (h / 6.0) * (k1_s2 + 2.0 * k2_s2 + 2.0 * k3_s2 + k4_s2)

            # Strict conservation of spin norms along trajectory
            s1_norm = jnp.maximum(jnp.linalg.norm(s1_next, axis=-1, keepdims=True), 1e-24)
            s2_norm = jnp.maximum(jnp.linalg.norm(s2_next, axis=-1, keepdims=True), 1e-24)
            s1_next = s1_next * (s1_norm_0 / s1_norm)
            s2_next = s2_next * (s2_norm_0 / s2_norm)

            next_state = (lnh_next, s1_next, s2_next)
            out = (lnh_next, s1_next / (m1_m ** 2), s2_next / (m2_m ** 2))
            return next_state, out

        init_state = (lnh_curr, s1_curr, s2_curr)
        _, (lnh_steps, s1_steps, s2_steps) = jax.lax.scan(step_fn, init_state, v_nodes[:-1])

        lnh_traj = jnp.concatenate([lnh_curr[None, ...], lnh_steps], axis=0)
        s1_traj = jnp.concatenate([(s1_curr / (m1_m ** 2))[None, ...], s1_steps], axis=0)
        s2_traj = jnp.concatenate([(s2_curr / (m2_m ** 2))[None, ...], s2_steps], axis=0)

        return (
            v_nodes,
            lnh_traj,
            s1_traj,
            s2_traj,
        )

    @staticmethod
    def derivatives(
        v: jnp.ndarray,
        lnh: jnp.ndarray,
        s1: jnp.ndarray,
        s2: jnp.ndarray,
        m1_m: jnp.ndarray,
        m2_m: jnp.ndarray,
        eta: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute state derivatives with respect to velocity v: (dt_dv, dlnh_dv, ds1_dv, ds2_dv)."""
        v2 = v * v
        v3 = v2 * v
        v4 = v2 * v2
        v5 = v4 * v
        v6 = v3 * v3
        v7 = v4 * v3
        v11 = v6 * v5

        l_dot_s1 = jnp.sum(lnh * s1, axis=-1, keepdims=True)
        l_dot_s2 = jnp.sum(lnh * s2, axis=-1, keepdims=True)
        s1_dot_s2 = jnp.sum(s1 * s2, axis=-1, keepdims=True)
        s1_sq = jnp.sum(s1 * s1, axis=-1, keepdims=True)
        s2_sq = jnp.sum(s2 * s2, axis=-1, keepdims=True)

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
                                + v * (a6 + w6 + alog * jnp.log(jnp.maximum(v, 1e-12)) + v * a7)
                            )
                        )
                    )
                )
            )
        )
        domega_dt = (96.0 / 5.0) * eta * v11 * p_v
        dt_dv = 3.0 * v2 / jnp.maximum(domega_dt, 1e-30)

        # Spin precession equations
        lnh_cross_s1 = jnp.cross(lnh, s1, axis=-1)
        lnh_cross_s2 = jnp.cross(lnh, s2, axis=-1)
        s1_cross_s2 = jnp.cross(s1, s2, axis=-1)

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
        omega_l = jnp.cross(lnh, dl_dt / jnp.maximum(l_mag, 1e-24), axis=-1)
        dlnh_dt = jnp.cross(omega_l, lnh, axis=-1)

        ds1_dv = ds1_dt * dt_dv
        ds2_dv = ds2_dt * dt_dv
        dlnh_dv = dlnh_dt * dt_dv

        return dt_dv, dlnh_dv, ds1_dv, ds2_dv


def evaluate_spintaylor_precession(
    f: jnp.ndarray,
    f_ref: Union[float, jnp.ndarray],
    mass1: Union[float, jnp.ndarray],
    mass2: Union[float, jnp.ndarray],
    spin1x: Union[float, jnp.ndarray],
    spin1y: Union[float, jnp.ndarray],
    spin1z: Union[float, jnp.ndarray],
    spin2x: Union[float, jnp.ndarray],
    spin2y: Union[float, jnp.ndarray],
    spin2z: Union[float, jnp.ndarray],
    inclination: Union[float, jnp.ndarray],
    phi_ref: Union[float, jnp.ndarray],
    convention: int = 1,
    observer_azimuth: Optional[Union[float, jnp.ndarray]] = None,
    damp_beta: bool = False,
    damping_rate: float = 2.0,
    **kwargs,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate SpinTaylor precession Euler angles and frame geometry across frequency grid f in JAX.

    Returns:
        (alpha, beta, gamma, theta_jn, zeta, carrier_phase)
    """
    dtype = f.dtype if f.dtype == jnp.float64 else jnp.float64
    f_arr = as_array(f, dtype=dtype)
    f_ref_t = as_array(f_ref, dtype=dtype)
    m1_t = as_array(mass1, dtype=dtype)
    m2_t = as_array(mass2, dtype=dtype)
    s1x_t = as_array(spin1x, dtype=dtype)
    s1y_t = as_array(spin1y, dtype=dtype)
    s1z_t = as_array(spin1z, dtype=dtype)
    s2x_t = as_array(spin2x, dtype=dtype)
    s2y_t = as_array(spin2y, dtype=dtype)
    s2z_t = as_array(spin2z, dtype=dtype)
    inc_t = as_array(inclination, dtype=dtype)
    phi_c_t = as_array(phi_ref, dtype=dtype)

    total_m = m1_t + m2_t
    is_batched = total_m.ndim >= 1
    f_eval = f_arr if (not is_batched or f_arr.ndim > 1) else f_arr[None, :]

    def unsqueeze_param(t: jnp.ndarray) -> jnp.ndarray:
        if is_batched and f_eval.ndim > t.ndim:
            return jnp.expand_dims(t, axis=-1)
        return t

    eta = (m1_t * m2_t) / (total_m * total_m)
    delta = (m1_t - m2_t) / total_m
    m1_m = m1_t / total_m
    m2_m = m2_t / total_m
    m_sec = total_m * MSUN_SEC

    # Scaled spins
    s1_0 = (m1_m ** 2)[..., None] * jnp.stack([s1x_t, s1y_t, s1z_t], axis=-1)
    s2_0 = (m2_m ** 2)[..., None] * jnp.stack([s2x_t, s2y_t, s2z_t], axis=-1)
    lnh_0 = jnp.zeros_like(s1_0)
    lnh_0 = lnh_0.at[..., 2].set(1.0)

    v0 = (PI * m_sec * f_ref_t) ** (1.0 / 3.0)
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
    j0 = l_mag0[..., None] * lnh_0 + s1_0 + s2_0
    j0_mag = jnp.maximum(jnp.linalg.norm(j0, axis=-1, keepdims=True), 1e-24)
    z_j = j0 / j0_mag

    # J-frame Euler rotation setup
    phiJ = jnp.arctan2(j0[..., 1], j0[..., 0])
    thetaJ = jnp.arccos(jnp.clip(j0[..., 2] / jnp.squeeze(j0_mag, -1), -1.0, 1.0))

    # Line of sight in source frame
    sin_i = jnp.sin(inc_t)
    cos_i = jnp.cos(inc_t)
    sin_p = jnp.sin(phi_c_t)
    cos_p = jnp.cos(phi_c_t)

    # In LAL, N_sf = (sin_i * sin(phi_ref), sin_i * cos(phi_ref), cos_i)
    n_src = jnp.stack([sin_i * sin_p, sin_i * cos_p, cos_i], axis=-1)

    c_p = jnp.cos(phiJ)
    s_p = jnp.sin(phiJ)
    c_t = jnp.cos(thetaJ)
    s_t = jnp.sin(thetaJ)

    # Rotate N by Rz(-phiJ) then Ry(-thetaJ)
    n1_x = c_p * n_src[..., 0] + s_p * n_src[..., 1]
    n1_y = -s_p * n_src[..., 0] + c_p * n_src[..., 1]
    n1_z = n_src[..., 2]

    n2_x = c_t * n1_x - s_t * n1_z
    n2_y = n1_y
    n2_z = s_t * n1_x + c_t * n1_z

    kappa = jnp.arctan2(n2_y, n2_x)
    c_k = jnp.cos(kappa)
    s_k = jnp.sin(kappa)

    # Angle between J0 and N
    cos_theta_jn = jnp.clip(jnp.sum(z_j * n_src, axis=-1), -1.0, 1.0)
    theta_jn = jnp.arccos(cos_theta_jn)

    # Polarization angle zeta
    # X_sf = (-cos_i * sin(phi_ref), -cos_i * cos(phi_ref), sin_i)
    x_src = jnp.stack([-cos_i * sin_p, -cos_i * cos_p, sin_i], axis=-1)
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
        x_dot_p = x3_x * cos_theta_jn - x3_z * jnp.sin(theta_jn)
        x_dot_q = x3_y
        zeta = jnp.arctan2(x_dot_q, x_dot_p)
        alpha_0 = PI - kappa
        epsilon_0 = phiJ - PI
        carrier_phase = jnp.zeros_like(phi_c_t)
    else:
        zeta = kappa
        alpha_0 = jnp.zeros_like(phi_c_t)
        epsilon_0 = jnp.zeros_like(phi_c_t)
        carrier_phase = -phiJ

    def rotate_to_j(vec: jnp.ndarray) -> jnp.ndarray:
        vx, vy, vz = vec[..., 0], vec[..., 1], vec[..., 2]
        c_pu, s_pu = c_p, s_p
        c_tu, s_tu = c_t, s_t
        c_ku, s_ku = c_k, s_k
        while c_pu.ndim < vx.ndim:
            c_pu = jnp.expand_dims(c_pu, -1)
            s_pu = jnp.expand_dims(s_pu, -1)
            c_tu = jnp.expand_dims(c_tu, -1)
            s_tu = jnp.expand_dims(s_tu, -1)
            c_ku = jnp.expand_dims(c_ku, -1)
            s_ku = jnp.expand_dims(s_ku, -1)

        v1x = c_pu * vx + s_pu * vy
        v1y = -s_pu * vx + c_pu * vy
        v1z = vz

        v2x = c_tu * v1x - s_tu * v1z
        v2y = v1y
        v2z = s_tu * v1x + c_tu * v1z

        v3x = c_ku * v2x + s_ku * v2y
        v3y = -s_ku * v2x + c_ku * v2y
        v3z = v2z
        return jnp.stack([v3x, v3y, v3z], axis=-1)

    # Transition frequency f_meco and f_trans
    mf_meco = XASParameterFits.fit_f_meco(eta, s1z_t, s2z_t)
    f_meco = mf_meco / jnp.maximum(m_sec, 1e-24)
    f_meco = jnp.clip(f_meco, f_ref_t + 15.0, 800.0)
    f_trans = 0.98 * f_meco
    v_trans = (PI * m_sec * f_trans) ** (1.0 / 3.0)

    # Fixed endpoint and reference anchored IVP make the trajectory independent
    # of the frequencies requested by the caller.
    f_min_eval = jnp.minimum(f_ref_t, 15.0)
    v_min = (PI * m_sec * f_min_eval) ** (1.0 / 3.0)

    # Integrate in both directions from the reference state.
    n_low, n_high = 100, 200

    s1_norm_0 = jnp.linalg.norm(s1_0, axis=-1, keepdims=True)
    s2_norm_0 = jnp.linalg.norm(s2_0, axis=-1, keepdims=True)

    m1_exp = m1_m[..., None]
    m2_exp = m2_m[..., None]
    eta_exp = eta[..., None]

    def integrate_segment(v_start, v_end, num_nodes):
        v_grid = jnp.linspace(0.0, 1.0, num_nodes, dtype=dtype)
        segment_nodes = v_start[..., None] + v_grid * (v_end - v_start)[..., None]
        v_nodes_t = jnp.moveaxis(segment_nodes, -1, 0)
        h_step = (v_end - v_start) / float(num_nodes - 1)
        h_exp = h_step[..., None]
        lnh_curr, s1_curr, s2_curr = lnh_0, s1_0, s2_0
        def step_derivs(v_in, l_in, s1_in, s2_in):
            return SpinTaylorT4Engine.derivatives(v_in, l_in, s1_in, s2_in, m1_exp, m2_exp, eta_exp)

        def scan_step(state, v_cur_val):
            lnh_c, s1_c, s2_c = state
            v_cur = v_cur_val[..., None]
            h = h_exp

            _, k1_l, k1_s1, k1_s2 = step_derivs(v_cur, lnh_c, s1_c, s2_c)

            v_half = v_cur + 0.5 * h
            l2 = lnh_c + 0.5 * h * k1_l
            l2 = l2 / jnp.maximum(jnp.linalg.norm(l2, axis=-1, keepdims=True), 1e-24)
            s1_2 = s1_c + 0.5 * h * k1_s1
            s2_2 = s2_c + 0.5 * h * k1_s2
            _, k2_l, k2_s1, k2_s2 = step_derivs(v_half, l2, s1_2, s2_2)

            l3 = lnh_c + 0.5 * h * k2_l
            l3 = l3 / jnp.maximum(jnp.linalg.norm(l3, axis=-1, keepdims=True), 1e-24)
            s1_3 = s1_c + 0.5 * h * k2_s1
            s2_3 = s2_c + 0.5 * h * k2_s2
            _, k3_l, k3_s1, k3_s2 = step_derivs(v_half, l3, s1_3, s2_3)

            v_full = v_cur + h
            l4 = lnh_c + h * k3_l
            l4 = l4 / jnp.maximum(jnp.linalg.norm(l4, axis=-1, keepdims=True), 1e-24)
            s1_4 = s1_c + h * k3_s1
            s2_4 = s2_c + h * k3_s2
            _, k4_l, k4_s1, k4_s2 = step_derivs(v_full, l4, s1_4, s2_4)

            lnh_next = lnh_c + (h / 6.0) * (k1_l + 2.0 * k2_l + 2.0 * k3_l + k4_l)
            lnh_next = lnh_next / jnp.maximum(jnp.linalg.norm(lnh_next, axis=-1, keepdims=True), 1e-24)
            s1_next = s1_c + (h / 6.0) * (k1_s1 + 2.0 * k2_s1 + 2.0 * k3_s1 + k4_s1)
            s2_next = s2_c + (h / 6.0) * (k1_s2 + 2.0 * k2_s2 + 2.0 * k3_s2 + k4_s2)

            s1_norm = jnp.maximum(jnp.linalg.norm(s1_next, axis=-1, keepdims=True), 1e-24)
            s2_norm = jnp.maximum(jnp.linalg.norm(s2_next, axis=-1, keepdims=True), 1e-24)
            s1_next = s1_next * (s1_norm_0 / s1_norm)
            s2_next = s2_next * (s2_norm_0 / s2_norm)

            return (lnh_next, s1_next, s2_next), lnh_next

        init_state = (lnh_curr, s1_curr, s2_curr)
        _, lnh_steps = jax.lax.scan(scan_step, init_state, v_nodes_t[:-1])

        lnh_all_t = jnp.concatenate([lnh_curr[None, ...], lnh_steps], axis=0)
        lnh_all = jnp.moveaxis(lnh_all_t, 0, -2)
        return segment_nodes, lnh_all

    v_low_desc, lnh_low_desc = integrate_segment(v0, v_min, n_low)
    v_high, lnh_high = integrate_segment(v0, v_trans, n_high)
    v_nodes = jnp.concatenate([jnp.flip(v_low_desc, axis=-1)[..., :-1], v_high], axis=-1)
    lnh_all = jnp.concatenate([jnp.flip(lnh_low_desc, axis=-2)[..., :-1, :], lnh_high], axis=-2)
    n_nodes = n_low + n_high - 1

    # Rotate to J-frame
    lnh_j_nodes = rotate_to_j(lnh_all)
    alpha_raw = jnp.arctan2(lnh_j_nodes[..., 1], lnh_j_nodes[..., 0])

    diff_alpha = alpha_raw[..., 1:] - alpha_raw[..., :-1]
    diff_alpha_unwrapped = jnp.remainder(diff_alpha + PI, 2.0 * PI) - PI
    alpha_nodes = jnp.concatenate([
        alpha_raw[..., :1],
        alpha_raw[..., :1] + jnp.cumsum(diff_alpha_unwrapped, axis=-1)
    ], axis=-1)
    cosbeta_nodes = jnp.clip(lnh_j_nodes[..., 2], -1.0, 1.0)
    beta_nodes = jnp.arccos(cosbeta_nodes)

    # Gamma evaluation from minimal rotation condition: dgamma = -cos(beta) * dalpha
    dalpha_step = alpha_nodes[..., 1:] - alpha_nodes[..., :-1]
    cosbeta_mid = 0.5 * (cosbeta_nodes[..., 1:] + cosbeta_nodes[..., :-1])
    dgamma_step = -cosbeta_mid * dalpha_step
    gamma_nodes = jnp.concatenate([
        jnp.zeros_like(alpha_nodes[..., :1]),
        jnp.cumsum(dgamma_step, axis=-1)
    ], axis=-1)

    # Shape preparation for broadcasting
    f_trans_exp = unsqueeze_param(f_trans)
    m_sec_exp = unsqueeze_param(m_sec)
    v_min_exp = unsqueeze_param(v_min)
    v_ref_exp = unsqueeze_param(v0)
    v_trans_exp = unsqueeze_param(v_trans)

    # Each side of the reference frequency has its own uniform v grid.
    v_eval = (PI * m_sec_exp * jnp.maximum(f_eval, 1e-12)) ** (1.0 / 3.0)
    u_low = jnp.clip((v_eval - v_min_exp) / jnp.maximum(v_ref_exp - v_min_exp, 1e-12), 0.0, 1.0)
    u_high = jnp.clip((v_eval - v_ref_exp) / jnp.maximum(v_trans_exp - v_ref_exp, 1e-12), 0.0, 1.0)
    idx_float = jnp.where(v_eval < v_ref_exp, u_low * (n_low - 1), (n_low - 1) + u_high * (n_high - 1))
    idx_low = jnp.clip(idx_float.astype(jnp.int64), 0, n_nodes - 2)
    t_frac = jnp.clip(idx_float - idx_low.astype(dtype), 0.0, 1.0)

    def gather_val(nodes_t: jnp.ndarray) -> jnp.ndarray:
        if nodes_t.ndim < idx_low.ndim:
            nodes_expanded = jnp.broadcast_to(nodes_t, idx_low.shape[:-1] + (nodes_t.shape[-1],))
        else:
            nodes_expanded = nodes_t
        val_low = jnp.take_along_axis(nodes_expanded, idx_low, axis=-1)
        val_high = jnp.take_along_axis(nodes_expanded, idx_low + 1, axis=-1)
        return val_low * (1.0 - t_frac) + val_high * t_frac

    alpha_insp = gather_val(alpha_nodes)
    cosb_insp = gather_val(cosbeta_nodes)
    gamma_insp = gather_val(gamma_nodes)

    # MRD analytical continuation for f >= f_trans
    mf_nodes = (v_nodes ** 3) / PI
    idx1 = (n_low - 1) + int(0.97 * (n_high - 1))
    idx2 = (n_low - 1) + int(0.99 * (n_high - 1))
    f1 = mf_nodes[..., idx1]
    f2 = mf_nodes[..., idx2]
    a1 = -alpha_nodes[..., idx1]
    a2 = -alpha_nodes[..., idx2]

    df_step = f2 - f1
    da1 = -(alpha_nodes[..., idx2] - alpha_nodes[..., idx1]) / jnp.maximum(df_step, 1e-12)

    f1sq, f2sq = f1 * f1, f2 * f2
    f1cube = f1sq * f1
    den_a = 2.0 * jnp.maximum((f1sq - f2sq) ** 2, 1e-24)
    aC = (f1cube * (f1 - f2) * (f1 + f2) * da1 + 2.0 * (f1**4 - 2.0 * f1sq * f2sq) * a1 + 2.0 * (f2**4) * a2) / den_a
    bC = ((f1**4) * f2sq * (f1 * (f1 - f2) * (f1 + f2) * da1 + 2.0 * f2sq * (-a1 + a2))) / den_a
    cC = (f1sq * (f1 * (-(f1**4) + f2**4) * da1 + 4.0 * (f2**4) * (a1 - a2))) / den_a

    mf_eval = f_eval * m_sec_exp
    mf2 = mf_eval * mf_eval
    mf4 = mf2 * mf2
    aC_u = unsqueeze_param(aC)
    bC_u = unsqueeze_param(bC)
    cC_u = unsqueeze_param(cC)

    alpha_mrd = -(aC_u + bC_u / jnp.maximum(mf4, 1e-16) + cC_u / jnp.maximum(mf2, 1e-8))
    if damp_beta:
        beta_insp = jnp.arccos(jnp.clip(cosb_insp, -1.0, 1.0))
        delta_f = jnp.maximum((f_eval - f_trans_exp) / f_trans_exp, 0.0)
        beta_damped = beta_insp * jnp.exp(-damping_rate * delta_f)
        cosb_mrd = jnp.cos(beta_damped)
    else:
        cosb_mrd = cosb_insp
    gamma_mrd = gamma_insp - cosb_mrd * (alpha_mrd - alpha_insp)

    is_mrd = f_eval >= f_trans_exp
    alpha_out = jnp.where(is_mrd, alpha_mrd, alpha_insp)
    cosb_out = jnp.where(is_mrd, cosb_mrd, cosb_insp)
    gamma_out = jnp.where(is_mrd, gamma_mrd, gamma_insp)
    beta_out = jnp.arccos(jnp.clip(cosb_out, -1.0, 1.0))

    # Pratten et al. (2021), Appendix C, fixes the reference gauge:
    # alpha_0 = pi - kappa and gamma_0 = -epsilon_0 (convention 1).
    alpha_ref_val = alpha_nodes[..., n_low - 1]
    gamma_ref_val = gamma_nodes[..., n_low - 1]
    alpha_offset = alpha_0 - alpha_ref_val
    alpha_final = alpha_out + unsqueeze_param(alpha_offset)

    # Return gamma = -epsilon to match MSA convention in twisting-up:
    # gamma(f) = -epsilon(f) = (gamma(f) - gamma_ref) - epsilon_0
    gamma_offset = -gamma_ref_val - epsilon_0
    gamma_final = gamma_out + unsqueeze_param(gamma_offset)

    if not is_batched and alpha_final.ndim > f_arr.ndim:
        alpha_final = jnp.squeeze(alpha_final, 0)
        beta_out = jnp.squeeze(beta_out, 0)
        gamma_final = jnp.squeeze(gamma_final, 0)
        theta_jn = jnp.squeeze(theta_jn, 0) if theta_jn.ndim > 0 else theta_jn
        zeta = jnp.squeeze(zeta, 0) if zeta.ndim > 0 else zeta
        carrier_phase = jnp.squeeze(carrier_phase, 0) if carrier_phase.ndim > 0 else carrier_phase

    return alpha_final, beta_out, gamma_final, theta_jn, zeta, carrier_phase


__all__ = [
    "compute_chi_tot_perp",
    "SpinTaylorT4Engine",
    "evaluate_spintaylor_precession",
]

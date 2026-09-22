"""Differentiable Spin-Precession Dynamics and ODE Engine in JAX.

Mathematical references for the clean-room integrator:
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
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    C_SI,
    G_SI,
    MSUN_SEC,
    PI,
)
from jaxwave.conversions import as_array


@jax.tree_util.register_pytree_node_class
class SpinPrecessionODE:
    """Orbit-averaged spin-precession numerical integrator in JAX.

    Integrates:
    - 2PN Spin-Orbit (SO) couplings
    - 2PN Spin-Spin (SS) couplings
    - 2.5PN Radiation reaction orbital decay
    - Extracts precessing Euler angles (alpha, beta, epsilon)
    """

    def __init__(self):
        pass

    def tree_flatten(self):
        return (), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()

    @staticmethod
    def precession_vectors(
        s1: jnp.ndarray,
        s2: jnp.ndarray,
        l_vec: jnp.ndarray,
        m1: Union[float, jnp.ndarray],
        m2: Union[float, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute precession frequency vectors Omega_1 and Omega_2 in JAX.

        Omega_1 = (1/r^3) * [ (2 + 3*m2/(2*m1)) * L + (1/2)*S_2 - (3/2)*(S_2 . L_hat)*L_hat ]
        Omega_2 = (1/r^3) * [ (2 + 3*m1/(2*m2)) * L + (1/2)*S_1 - (3/2)*(S_1 . L_hat)*L_hat ]
        """
        s1_t = as_array(s1, dtype=jnp.float64)
        s2_t = as_array(s2, dtype=jnp.float64)
        l_t = as_array(l_vec, dtype=jnp.float64)
        m1_t = as_array(m1, dtype=jnp.float64)
        m2_t = as_array(m2, dtype=jnp.float64)

        l_mag = jnp.linalg.norm(l_t, axis=-1, keepdims=True)
        l_hat = l_t / jnp.maximum(l_mag, 1e-24)

        total_m = m1_t + m2_t
        eta = (m1_t * m2_t) / (total_m * total_m)
        if total_m.ndim > 0 and l_mag.ndim > total_m.ndim:
            eta_m = jnp.expand_dims(eta, -1)
            m1_m = jnp.expand_dims(m1_t, -1)
            m2_m = jnp.expand_dims(m2_t, -1)
        else:
            eta_m = eta
            m1_m = m1_t
            m2_m = m2_t

        # In geometric units with M=1: L = eta * sqrt(r) => r = (L / eta)^2
        r = (l_mag / jnp.maximum(eta_m, 1e-12)) ** 2
        r3 = jnp.maximum(r ** 3, 1e-36)

        q1 = m2_m / m1_m
        q2 = m1_m / m2_m

        # Omega 1
        term1_so = (2.0 + 1.5 * q1) * l_t
        s2_dot_lh = jnp.sum(s2_t * l_hat, axis=-1, keepdims=True)
        term1_ss = 0.5 * s2_t - 1.5 * s2_dot_lh * l_hat
        omega1 = (term1_so + term1_ss) / r3

        # Omega 2
        term2_so = (2.0 + 1.5 * q2) * l_t
        s1_dot_lh = jnp.sum(s1_t * l_hat, axis=-1, keepdims=True)
        term2_ss = 0.5 * s1_t - 1.5 * s1_dot_lh * l_hat
        omega2 = (term2_so + term2_ss) / r3

        return omega1, omega2

    @classmethod
    def state_derivatives(
        cls,
        s1: jnp.ndarray,
        s2: jnp.ndarray,
        l_vec: jnp.ndarray,
        m1: Union[float, jnp.ndarray],
        m2: Union[float, jnp.ndarray],
        radiation_reaction: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute time derivatives (dS1/dt, dS2/dt, dL/dt) in JAX."""
        s1_t = as_array(s1, dtype=jnp.float64)
        s2_t = as_array(s2, dtype=jnp.float64)
        l_t = as_array(l_vec, dtype=jnp.float64)
        m1_t = as_array(m1, dtype=jnp.float64)
        m2_t = as_array(m2, dtype=jnp.float64)

        omega1, omega2 = cls.precession_vectors(s1_t, s2_t, l_t, m1_t, m2_t)

        ds1 = jnp.cross(omega1, s1_t, axis=-1)
        ds2 = jnp.cross(omega2, s2_t, axis=-1)
        dl = -(ds1 + ds2)

        if radiation_reaction:
            l_mag = jnp.linalg.norm(l_t, axis=-1, keepdims=True)
            l_hat = l_t / jnp.maximum(l_mag, 1e-24)
            total_m = m1_t + m2_t
            eta = (m1_t * m2_t) / (total_m * total_m)
            if total_m.ndim > 0 and l_mag.ndim > total_m.ndim:
                eta_m = jnp.expand_dims(eta, -1)
            else:
                eta_m = eta
            r = (l_mag / jnp.maximum(eta_m, 1e-12)) ** 2
            dl_mag_dt = -(32.0 / 5.0) * (eta_m ** 2) / jnp.maximum(r ** 3.5, 1e-40)
            dl = dl + dl_mag_dt * l_hat

        return ds1, ds2, dl

    @classmethod
    def rk4_step(
        cls,
        s1: jnp.ndarray,
        s2: jnp.ndarray,
        l_vec: jnp.ndarray,
        dt: Union[float, jnp.ndarray],
        m1: Union[float, jnp.ndarray],
        m2: Union[float, jnp.ndarray],
        radiation_reaction: bool = True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Perform one 4th-order Runge-Kutta step in JAX."""
        k1_s1, k1_s2, k1_l = cls.state_derivatives(s1, s2, l_vec, m1, m2, radiation_reaction)

        k2_s1, k2_s2, k2_l = cls.state_derivatives(
            s1 + 0.5 * dt * k1_s1,
            s2 + 0.5 * dt * k1_s2,
            l_vec + 0.5 * dt * k1_l,
            m1,
            m2,
            radiation_reaction,
        )

        k3_s1, k3_s2, k3_l = cls.state_derivatives(
            s1 + 0.5 * dt * k2_s1,
            s2 + 0.5 * dt * k2_s2,
            l_vec + 0.5 * dt * k2_l,
            m1,
            m2,
            radiation_reaction,
        )

        k4_s1, k4_s2, k4_l = cls.state_derivatives(
            s1 + dt * k3_s1,
            s2 + dt * k3_s2,
            l_vec + dt * k3_l,
            m1,
            m2,
            radiation_reaction,
        )

        s1_next = s1 + (dt / 6.0) * (k1_s1 + 2.0 * k2_s1 + 2.0 * k3_s1 + k4_s1)
        s2_next = s2 + (dt / 6.0) * (k1_s2 + 2.0 * k2_s2 + 2.0 * k3_s2 + k4_s2)
        l_next = l_vec + (dt / 6.0) * (k1_l + 2.0 * k2_l + 2.0 * k3_l + k4_l)

        return s1_next, s2_next, l_next

    @classmethod
    def integrate_time(
        cls,
        s1_0: jnp.ndarray,
        s2_0: jnp.ndarray,
        l_0: jnp.ndarray,
        m1: Union[float, jnp.ndarray],
        m2: Union[float, jnp.ndarray],
        total_time: float = 1000.0,
        n_steps: int = 500,
        radiation_reaction: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """Integrate spin precession over a fixed time duration in units of M in JAX."""
        s1_0 = as_array(s1_0, dtype=jnp.float64)
        s2_0 = as_array(s2_0, dtype=jnp.float64)
        l_0 = as_array(l_0, dtype=jnp.float64)
        m1 = as_array(m1, dtype=jnp.float64)
        m2 = as_array(m2, dtype=jnp.float64)
        dt = total_time / float(n_steps)

        def scan_body(state, step_idx):
            s1, s2, l = state
            s1_next, s2_next, l_next = cls.rk4_step(s1, s2, l, dt, m1, m2, radiation_reaction)
            t_next = (step_idx + 1) * dt
            j_next = s1_next + s2_next + l_next
            return (s1_next, s2_next, l_next), (t_next, s1_next, s2_next, l_next, j_next)

        init_state = (s1_0, s2_0, l_0)
        _, (times_rest, s1_rest, s2_rest, l_rest, j_rest) = jax.lax.scan(
            scan_body, init_state, jnp.arange(n_steps)
        )

        times = jnp.concatenate([jnp.array([0.0], dtype=s1_0.dtype), times_rest])
        s1_traj = jnp.concatenate([jnp.expand_dims(s1_0, 0), s1_rest], axis=0)
        s2_traj = jnp.concatenate([jnp.expand_dims(s2_0, 0), s2_rest], axis=0)
        l_traj = jnp.concatenate([jnp.expand_dims(l_0, 0), l_rest], axis=0)
        j_traj = jnp.concatenate([jnp.expand_dims(s1_0 + s2_0 + l_0, 0), j_rest], axis=0)

        return {
            "times": times,
            "s1": s1_traj,
            "s2": s2_traj,
            "l": l_traj,
            "j": j_traj,
        }

    @classmethod
    def extract_euler_angles(
        cls,
        l_traj: jnp.ndarray,
        j0: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Extract precessing Euler angles (alpha, beta, epsilon) from L(t) relative to J_0 in JAX.

        Args:
            l_traj: Orbital angular momentum trajectory, shape (..., N, 3) or (N, 3)
            j0: Initial total angular momentum vector, shape (..., 3) or (3,)

        Returns:
            Tuple of (alpha, beta, epsilon) arrays
        """
        l_traj = as_array(l_traj, dtype=jnp.float64)
        j0 = as_array(j0, dtype=jnp.float64)

        j0_mag = jnp.linalg.norm(j0, axis=-1, keepdims=True)
        z_j = j0 / jnp.maximum(j0_mag, 1e-24)

        # Reference coordinate system attached to J_0
        z_ref = jnp.zeros_like(z_j).at[..., 2].set(1.0)
        cross_z = jnp.cross(z_ref, z_j, axis=-1)
        cross_mag = jnp.linalg.norm(cross_z, axis=-1, keepdims=True)

        x_j_default = jnp.zeros_like(z_j).at[..., 0].set(1.0)
        x_j = jnp.where(cross_mag > 1e-4, cross_z / jnp.maximum(cross_mag, 1e-24), x_j_default)
        y_j = jnp.cross(z_j, x_j, axis=-1)

        if z_j.ndim < l_traj.ndim:
            z_j_exp = jnp.expand_dims(z_j, -2)
            x_j_exp = jnp.expand_dims(x_j, -2)
            y_j_exp = jnp.expand_dims(y_j, -2)
        else:
            z_j_exp = z_j
            x_j_exp = x_j
            y_j_exp = y_j

        l_mag = jnp.linalg.norm(l_traj, axis=-1, keepdims=True)
        l_hat = l_traj / jnp.maximum(l_mag, 1e-24)

        # 1. Opening angle beta: cos(beta) = L_hat . J_hat_0
        cos_beta = jnp.minimum(jnp.maximum(jnp.sum(l_hat * z_j_exp, axis=-1), -1.0), 1.0)
        perpendicular = jnp.linalg.norm(
            jnp.cross(l_hat, jnp.broadcast_to(z_j_exp, l_hat.shape), axis=-1), axis=-1
        )
        beta = jnp.arctan2(perpendicular, cos_beta)

        # 2. Azimuthal precession angle alpha: angle in x_j, y_j plane
        lx = jnp.sum(l_hat * x_j_exp, axis=-1)
        ly = jnp.sum(l_hat * y_j_exp, axis=-1)
        alpha = jnp.arctan2(ly, lx)

        # 3. Minimal rotation condition: epsilon = -gamma, d_epsilon = cos_beta * d_alpha
        d_alpha = alpha[..., 1:] - alpha[..., :-1]
        d_alpha = jnp.remainder(d_alpha + PI, 2.0 * PI) - PI
        cos_b_mid = 0.5 * (cos_beta[..., 1:] + cos_beta[..., :-1])
        d_eps = cos_b_mid * d_alpha

        zero_lead = jnp.zeros(alpha.shape[:-1] + (1,), dtype=alpha.dtype)
        eps_cum = jnp.cumsum(d_eps, axis=-1)
        epsilon = jnp.concatenate([zero_lead, eps_cum], axis=-1)

        return alpha, beta, epsilon

    @classmethod
    def compute_angles_frequency(
        cls,
        f: jnp.ndarray,
        mass1: Union[float, jnp.ndarray],
        mass2: Union[float, jnp.ndarray],
        spin1x: Union[float, jnp.ndarray] = 0.0,
        spin1y: Union[float, jnp.ndarray] = 0.0,
        spin1z: Union[float, jnp.ndarray] = 0.0,
        spin2x: Union[float, jnp.ndarray] = 0.0,
        spin2y: Union[float, jnp.ndarray] = 0.0,
        spin2z: Union[float, jnp.ndarray] = 0.0,
        f_ref: Optional[Union[float, jnp.ndarray]] = None,
        max_steps: Optional[int] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute precession angles (alpha, beta, epsilon) on a frequency grid f using ODE integration in JAX.

        Integrates along velocity v = (pi * M * f)^(1/3) using 2.5PN radiation reaction.
        """
        f_val = as_array(f, dtype=jnp.float64)
        m1 = as_array(mass1, dtype=jnp.float64)
        m2 = as_array(mass2, dtype=jnp.float64)
        s1x = as_array(spin1x, dtype=jnp.float64)
        s1y = as_array(spin1y, dtype=jnp.float64)
        s1z = as_array(spin1z, dtype=jnp.float64)
        s2x = as_array(spin2x, dtype=jnp.float64)
        s2y = as_array(spin2y, dtype=jnp.float64)
        s2z = as_array(spin2z, dtype=jnp.float64)
        f_ref_val = as_array(20.0 if f_ref is None else f_ref, dtype=jnp.float64)

        m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, f_ref_val = jnp.broadcast_arrays(
            m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, f_ref_val
        )

        total_m = m1 + m2
        eta = m1 * m2 / jnp.square(total_m)
        s1_ref = jnp.expand_dims(jnp.square(m1 / total_m), -1) * jnp.stack((s1x, s1y, s1z), axis=-1)
        s2_ref = jnp.expand_dims(jnp.square(m2 / total_m), -1) * jnp.stack((s2x, s2y, s2z), axis=-1)
        v_ref = jnp.power(PI * total_m * MSUN_SEC * f_ref_val, 1.0 / 3.0)
        l_ref = jnp.zeros_like(s1_ref).at[..., 2].set(eta / v_ref)
        j_ref = l_ref + s1_ref + s2_ref
        z_j = j_ref / jnp.linalg.norm(j_ref, axis=-1, keepdims=True)
        z_axis = jnp.zeros_like(z_j).at[..., 2].set(1.0)
        cross = jnp.cross(z_axis, z_j, axis=-1)
        cross_norm = jnp.linalg.norm(cross, axis=-1, keepdims=True)
        y_axis = jnp.zeros_like(z_j).at[..., 1].set(1.0)
        fallback = jnp.cross(y_axis, z_j, axis=-1)
        fallback = fallback / jnp.linalg.norm(fallback, axis=-1, keepdims=True)
        x_j = jnp.where(cross_norm > 1e-12, cross / jnp.maximum(cross_norm, 1e-24), fallback)
        y_j = jnp.cross(z_j, x_j, axis=-1)

        v_target = jnp.power(PI * jnp.expand_dims(total_m, -1) * MSUN_SEC * f_val, 1.0 / 3.0)
        v_start = jnp.broadcast_to(jnp.expand_dims(v_ref, -1), v_target.shape)
        span = v_target - v_start
        n_steps = jnp.maximum(
            jnp.ceil(jax.lax.stop_gradient(jnp.abs(span)) / 0.001).astype(jnp.int64), 1
        )
        step = span / n_steps
        state_shape = v_target.shape + (3,)
        s1 = jnp.broadcast_to(jnp.expand_dims(s1_ref, -2), state_shape)
        s2 = jnp.broadcast_to(jnp.expand_dims(s2_ref, -2), state_shape)
        l_vec = jnp.broadcast_to(jnp.expand_dims(l_ref, -2), state_shape)
        epsilon = jnp.zeros_like(v_target)
        x_j_exp = jnp.expand_dims(x_j, -2)
        y_j_exp = jnp.expand_dims(y_j, -2)
        z_j_exp = jnp.expand_dims(z_j, -2)
        m1_f = jnp.expand_dims(m1, -1)
        m2_f = jnp.expand_dims(m2, -1)
        eta_f = jnp.expand_dims(eta, -1)

        def derivatives(velocity, state):
            spin1, spin2, orbital, _ = state
            l_hat = orbital / jnp.maximum(
                jnp.linalg.norm(orbital, axis=-1, keepdims=True), 1e-24
            )
            orbital_pn = jnp.expand_dims(eta_f / velocity, -1) * l_hat
            omega1, omega2 = cls.precession_vectors(spin1, spin2, orbital_pn, m1_f, m2_f)
            dt_dv = jnp.expand_dims(5.0 / (32.0 * eta_f * jnp.power(velocity, 9)), -1)
            ds1 = jnp.cross(omega1, spin1, axis=-1) * dt_dv
            ds2 = jnp.cross(omega2, spin2, axis=-1) * dt_dv
            dl = -(ds1 + ds2) - jnp.expand_dims(eta_f / jnp.square(velocity), -1) * l_hat
            lx = jnp.sum(orbital * x_j_exp, axis=-1)
            ly = jnp.sum(orbital * y_j_exp, axis=-1)
            dx = jnp.sum(dl * x_j_exp, axis=-1)
            dy = jnp.sum(dl * y_j_exp, axis=-1)
            dalpha = (lx * dy - ly * dx) / jnp.maximum(jnp.square(lx) + jnp.square(ly), 1e-24)
            cos_beta = jnp.sum(l_hat * z_j_exp, axis=-1)
            return ds1, ds2, dl, cos_beta * dalpha

        if max_steps is not None:
            total_steps = int(max_steps)
        else:
            try:
                total_steps = int(jnp.max(n_steps))
            except (TypeError, jax.errors.ConcretizationTypeError):
                total_steps = 300

        def scan_body(state, index):
            h = jnp.where(index < n_steps, step, jnp.zeros_like(step))
            velocity = v_start + jnp.minimum(jnp.full_like(n_steps, index), n_steps) * step

            def advance(base, delta, factor):
                return tuple(
                    a + factor * (jnp.expand_dims(h, -1) if a.ndim > h.ndim else h) * b
                    for a, b in zip(base, delta)
                )

            k1 = derivatives(velocity, state)
            k2 = derivatives(velocity + h / 2, advance(state, k1, 0.5))
            k3 = derivatives(velocity + h / 2, advance(state, k2, 0.5))
            k4 = derivatives(velocity + h, advance(state, k3, 1.0))
            delta = tuple((a + 2 * b + 2 * c + d) / 6.0 for a, b, c, d in zip(k1, k2, k3, k4))
            state_next = advance(state, delta, 1.0)
            return state_next, None

        init_state = (s1, s2, l_vec, epsilon)
        final_state, _ = jax.lax.scan(scan_body, init_state, jnp.arange(total_steps))

        _, _, orbital, epsilon_final = final_state
        lx = jnp.sum(orbital * x_j_exp, axis=-1)
        ly = jnp.sum(orbital * y_j_exp, axis=-1)
        lz = jnp.sum(orbital * z_j_exp, axis=-1)
        transverse = jnp.linalg.norm(jnp.stack((lx, ly), axis=-1), axis=-1)
        alpha = jnp.arctan2(ly, jnp.where(transverse > 0, lx, jnp.ones_like(lx)))
        beta = jnp.arctan2(transverse, lz)
        return alpha, beta, epsilon_final

    def __call__(self, *args, **kwargs):
        return self.compute_angles_frequency(*args, **kwargs)

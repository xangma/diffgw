"""Pure JAX Batched ODE Integrator for EOB Dynamical Trajectories."""

import math
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from jaxwave.approximants.eob.events import OrbitalFrequencyPeak, orbital_frequency_peak
from jaxwave.approximants.eob.hamiltonian import _HamiltonianBase
from jaxwave.conversions import as_array


class FrequencyPeakTrajectory(NamedTuple):
    time: jnp.ndarray  # (N,)
    state: jnp.ndarray  # (N, ..., 4)
    frequency: jnp.ndarray  # (N, ...)
    valid: jnp.ndarray  # (N, ...), false for frozen completed batch rows
    peak: OrbitalFrequencyPeak


class EOBTrajectoryIntegrator:
    """Integrates the full conservative and dissipative EOB equations of motion in JAX."""

    def __init__(
        self,
        hamiltonian: _HamiltonianBase,
        radiation_reaction,
    ):
        self.hamiltonian = hamiltonian
        self.rr = radiation_reaction

    def vector_field(self, state: jnp.ndarray) -> jnp.ndarray:
        """Compute dydt = [dr/dt, dphi/dt, dpr_star/dt, dpphi/dt]."""
        r = state[..., 0]
        phi = state[..., 1]
        pr_star = state[..., 2]
        pphi = state[..., 3]

        dr_dt, omega, dpr_star_cons = self.hamiltonian.equations_of_motion_conservative(
            r=r,
            pr_star=pr_star,
            pphi=pphi,
        )

        f_r_star, f_phi = self.rr.radiation_reaction_forces(
            omega=omega,
            pr_star=pr_star,
            pphi=pphi,
            r=r,
            hamiltonian=self.hamiltonian,
        )

        dpr_star_dt = dpr_star_cons + f_r_star
        dpphi_dt = f_phi

        return jnp.stack([dr_dt, omega, dpr_star_dt, dpphi_dt], axis=-1)

    def integrate(
        self,
        initial_state: jnp.ndarray,
        dt: float,
        num_steps: int,
        r_min: float = 2.5,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Integrate trajectory using batched 4th-order Runge-Kutta (RK4)."""
        initial_state = as_array(initial_state, dtype=jnp.float64)
        half_dt = 0.5 * dt
        sixth_dt = dt / 6.0

        # Always use jax.lax.scan for fast compilation and execution
        def scan_body(state, _):
            k1 = self.vector_field(state)
            k2 = self.vector_field(state + half_dt * k1)
            k3 = self.vector_field(state + half_dt * k2)
            k4 = self.vector_field(state + dt * k3)
            next_state = state + sixth_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            evolving = state[..., 0] >= r_min
            next_state = jnp.where(jnp.expand_dims(evolving, -1), next_state, state)
            return next_state, state

        final_state, traj_all = jax.lax.scan(scan_body, initial_state, None, length=num_steps)
        traj_full = jnp.concatenate([traj_all, final_state[None, ...]], axis=0)
        times = jnp.arange(num_steps + 1, dtype=jnp.float64) * dt
        return times, traj_full

    def integrate_to_frequency_peak(
        self,
        initial_state: jnp.ndarray,
        dt: float,
        num_steps: int,
        *,
        post_peak_duration: float = 0.0,
        richardson_extrapolation: bool = False,
        matching_method: str = "continuous",
    ) -> FrequencyPeakTrajectory:
        """Integrate through a resolved Omega maximum in each batch row."""
        initial_state = as_array(initial_state, dtype=jnp.float64)
        if not math.isfinite(dt) or dt <= 0 or num_steps < 4:
            raise ValueError("dt must be finite and positive; num_steps must be at least four")

        duration = jnp.broadcast_to(
            as_array(post_peak_duration, dtype=initial_state.dtype),
            initial_state.shape[:-1],
        )
        required_after = 2 + jnp.ceil(duration / dt).astype(int)

        def field(state_val, evolving=None):
            result = self.vector_field(state_val)
            return (
                result
                if evolving is None
                else jnp.where(jnp.expand_dims(evolving, -1), result, jnp.zeros_like(result))
            )

        state = initial_state
        current_field = field(state)
        omega = current_field[..., 1]
        active = jnp.ones_like(omega, dtype=bool)
        maximum = omega
        maximum_index = jnp.zeros_like(omega, dtype=int)
        states, frequencies, masks = [state], [omega], [active]

        def rk4_step(value, step_dt, evolving):
            k1 = field(value, evolving)
            k2 = field(value + (step_dt / 2.0) * k1, evolving)
            k3 = field(value + (step_dt / 2.0) * k2, evolving)
            k4 = field(value + step_dt * k3, evolving)
            return value + (step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        rk4_step_jitted = jax.jit(rk4_step)
        field_jitted = jax.jit(field)

        for step in range(1, num_steps + 1):
            if richardson_extrapolation:
                full = rk4_step_jitted(state, dt, active)
                half = rk4_step_jitted(state, dt / 2.0, active)
                half = rk4_step_jitted(half, dt / 2.0, active)
                candidate = half + (half - full) / 15.0
            else:
                candidate = rk4_step_jitted(state, dt, active)

            state = jnp.where(jnp.expand_dims(active, -1), candidate, state)
            current_field = field_jitted(state)
            omega = current_field[..., 1]
            states.append(state)
            frequencies.append(omega)
            masks.append(active)

            higher = active & (omega > maximum)
            maximum = jnp.where(higher, omega, maximum)
            maximum_index = jnp.where(higher, jnp.full_like(maximum_index, step), maximum_index)
            done = active & (step - maximum_index >= required_after) & (omega < frequencies[-2])
            active = active & ~done

            current_field = jnp.where(
                jnp.expand_dims(active, -1), current_field, jnp.zeros_like(current_field)
            )
            if not bool(jnp.any(active)):
                break

        if bool(jnp.any(active)):
            raise ValueError("orbital-frequency peak not reached within num_steps")

        time = jnp.arange(len(states), dtype=state.dtype) * dt
        trajectory = jnp.stack(states, axis=0)
        frequency = jnp.stack(frequencies, axis=0)
        valid = jnp.stack(masks, axis=0)

        if matching_method == "continuous":
            from .continuous_matching import continuous_spline_peak
            from .numerical_attachment import natural_cubic_spline

            if frequency.ndim == 1:
                t_sub = time[valid]
                om_sub = frequency[valid]
                pt = continuous_spline_peak(t_sub, om_sub)
                sp = natural_cubic_spline(t_sub, om_sub, scalar_output=True)
                peak = OrbitalFrequencyPeak(
                    pt, sp(pt), sp.derivative(pt, 2), jnp.searchsorted(time, pt)
                )
            else:
                flat_time = jnp.broadcast_to(time, frequency.shape[1:] + (time.shape[-1],)).reshape(
                    -1, time.shape[-1]
                )
                flat_omega = jnp.moveaxis(frequency, 0, -1).reshape(-1, time.shape[-1])
                flat_valid = jnp.moveaxis(valid, 0, -1).reshape(-1, time.shape[-1])
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
                    jnp.array(pts).reshape(frequency.shape[1:]),
                    jnp.array(pfreqs).reshape(frequency.shape[1:]),
                    jnp.array(pcurvs).reshape(frequency.shape[1:]),
                    jnp.array(pidxs).reshape(frequency.shape[1:]),
                )
        elif matching_method == "grid":
            peak = orbital_frequency_peak(
                time, jnp.moveaxis(frequency, 0, -1), valid=jnp.moveaxis(valid, 0, -1)
            )
        else:
            raise ValueError(f"matching_method must be 'continuous' or 'grid', got {matching_method!r}")

        return FrequencyPeakTrajectory(time, trajectory, frequency, valid, peak)


__all__ = [
    "FrequencyPeakTrajectory",
    "EOBTrajectoryIntegrator",
]

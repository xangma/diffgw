"""Pure PyTorch Batched ODE Integrator for EOB Dynamical Trajectories.

Strict Clean-Room Implementation.
"""

from typing import NamedTuple, Tuple
import math
import torch

from torchwave.approximants.eob.hamiltonian import _HamiltonianBase
from .events import OrbitalFrequencyPeak, orbital_frequency_peak


class FrequencyPeakTrajectory(NamedTuple):
    time: torch.Tensor  # (N,)
    state: torch.Tensor  # (N, ..., 4)
    frequency: torch.Tensor  # (N, ...)
    valid: torch.Tensor  # (N, ...), false for frozen completed batch rows
    peak: OrbitalFrequencyPeak


class EOBTrajectoryIntegrator:
    """Integrates the full conservative and dissipative EOB equations of motion."""

    def __init__(
        self,
        hamiltonian: _HamiltonianBase,
        radiation_reaction,
    ):
        self.hamiltonian = hamiltonian
        self.rr = radiation_reaction

    def vector_field(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dydt = [dr/dt, dphi/dt, dpr_star/dt, dpphi/dt].

        Args:
            state: Tensor of shape (..., 4) containing [r, phi, pr_star, pphi]

        Returns:
            dstate_dt: Tensor of shape (..., 4)
        """
        r = state[..., 0]
        phi = state[..., 1]
        pr_star = state[..., 2]
        pphi = state[..., 3]

        # Conservative derivatives
        dr_dt, omega, dpr_star_cons = self.hamiltonian.equations_of_motion_conservative(
            r=r,
            pr_star=pr_star,
            pphi=pphi,
        )

        # Dissipative radiation reaction forces
        f_r_star, f_phi = self.rr.radiation_reaction_forces(
            omega=omega,
            pr_star=pr_star,
            pphi=pphi,
            r=r,
            hamiltonian=self.hamiltonian,
        )

        # Full equations of motion
        dpr_star_dt = dpr_star_cons + f_r_star
        dpphi_dt = f_phi

        return torch.stack([dr_dt, omega, dpr_star_dt, dpphi_dt], dim=-1)

    def integrate(
        self,
        initial_state: torch.Tensor,
        dt: float,
        num_steps: int,
        r_min: float = 2.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate trajectory using batched 4th-order Runge-Kutta (RK4).

        Args:
            initial_state: Tensor of shape (..., 4) [r_0, phi_0, pr_star_0, pphi_0]
            dt: Time step in geometric units (M)
            num_steps: Maximum number of integration steps
            r_min: Minimum radius threshold to terminate plunge

        Returns:
            t_array: Tensor of integration times, shape (N_steps,)
            traj_array: Tensor of states, shape (N_steps, ..., 4)
        """
        device = initial_state.device
        dtype = initial_state.dtype

        state = initial_state.clone()
        states = [state]
        times = [0.0]

        half_dt = 0.5 * dt
        sixth_dt = dt / 6.0

        for step in range(1, num_steps + 1):
            k1 = self.vector_field(state)
            k2 = self.vector_field(state + half_dt * k1)
            k3 = self.vector_field(state + half_dt * k2)
            k4 = self.vector_field(state + dt * k3)

            next_state = state + sixth_dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # Plunge termination check
            r_current = next_state[..., 0]
            if (r_current < r_min).all():
                break

            state = next_state
            states.append(state)
            times.append(step * dt)

        traj_tensor = torch.stack(states, dim=0)  # (N_steps, ..., 4)
        t_tensor = torch.tensor(times, dtype=dtype, device=device)

        return t_tensor, traj_tensor

    def integrate_to_frequency_peak(self, initial_state, dt, num_steps, *,
                                    post_peak_duration=0.,
                                    richardson_extrapolation=False,
                                    matching_method="continuous"):
        """Integrate through a resolved Omega maximum in each batch row.

        Fixed-step RK4 retains at least two samples beyond each observed maximum and
        then freezes that row. ``valid`` distinguishes frozen padding from
        evolved samples. A quartic local root refines the peak; it never uses
        a fixed radius as a substitute. This event defines the first maximum
        followed by two decreasing samples, appropriate for a smooth inspiral
        and plunge. Oscillatory/eccentric trajectories need another event rule.

        Exhausted steps and invalid Hamiltonian/RK stages raise. Choose a
        smaller dt if a stage crosses the Hamiltonian's domain and establish
        time-step convergence before using the result for merger matching.
        ``post_peak_duration`` adds nonnegative time beyond those two samples,
        independently per parameter row, for matching after the maximum.
        """
        if (not isinstance(initial_state, torch.Tensor) or initial_state.ndim < 1
                or initial_state.shape[-1] != 4
                or initial_state.dtype not in (torch.float32, torch.float64)):
            raise TypeError("initial_state must be float32/float64 with final axis 4")
        if not math.isfinite(dt) or dt <= 0 or not isinstance(num_steps, int) or num_steps < 4:
            raise ValueError("dt must be finite and positive; num_steps must be at least four")
        duration = torch.as_tensor(post_peak_duration, dtype=initial_state.dtype,
                                   device=initial_state.device).expand(initial_state.shape[:-1])
        if not bool(torch.isfinite(duration).all()) or bool((duration < 0).any()):
            raise ValueError("post_peak_duration must be finite and nonnegative")
        if not isinstance(richardson_extrapolation, bool):
            raise TypeError("richardson_extrapolation must be a bool")
        required_after = 2+torch.ceil(duration/dt).to(torch.long)

        def field(state, evolving=None):
            if not bool(torch.isfinite(state).all()) or bool((state[..., 0] <= 0).any()):
                raise ValueError("invalid RK stage; reduce dt or check the Hamiltonian domain")
            result = self.vector_field(state)
            if not bool(torch.isfinite(result).all()) or bool((result[..., 1] <= 0).any()):
                raise ValueError("invalid orbital dynamics; reduce dt or check the Hamiltonian domain")
            return result if evolving is None else torch.where(
                evolving.unsqueeze(-1), result, torch.zeros_like(result))

        state = initial_state.clone()
        current_field = field(state)
        omega = current_field[..., 1]
        active = torch.ones_like(omega, dtype=torch.bool)
        maximum = omega
        maximum_index = torch.zeros_like(omega, dtype=torch.long)
        states, frequencies, masks = [state], [omega], [active]

        def rk4_step(value, step_dt, evolving):
            """One RK4 step, masking rows frozen by the event detector."""
            k1 = field(value, evolving)
            k2 = field(value + step_dt/2*k1, evolving)
            k3 = field(value + step_dt/2*k2, evolving)
            k4 = field(value + step_dt*k3, evolving)
            return value + step_dt/6*(k1+2*k2+2*k3+k4)

        for step in range(1, num_steps+1):
            if richardson_extrapolation:
                # Two half steps cancel the leading RK4 truncation term when
                # combined with one full step.  The event grid remains dt.
                full = rk4_step(state, dt, active)
                half = rk4_step(state, dt/2, active)
                half = rk4_step(half, dt/2, active)
                candidate = half + (half-full)/15
            else:
                candidate = rk4_step(state, dt, active)
            state = torch.where(active.unsqueeze(-1), candidate, state)
            current_field = field(state)
            omega = current_field[..., 1]
            states.append(state)
            frequencies.append(omega)
            masks.append(active)
            higher = active & (omega > maximum)
            maximum = torch.where(higher, omega, maximum)
            maximum_index = torch.where(higher, torch.full_like(maximum_index, step), maximum_index)
            done = active & (step-maximum_index >= required_after) & (omega < frequencies[-2])
            active = active & ~done
            # Zero frozen rows before forming subsequent RK stages; evaluating
            # extra trial stages beyond their valid endpoint can leave domain.
            current_field = torch.where(active.unsqueeze(-1), current_field, torch.zeros_like(current_field))
            if not bool(active.any()):
                break
        if bool(active.any()):
            raise ValueError("orbital-frequency peak not reached within num_steps")
        time = torch.arange(len(states), dtype=state.dtype, device=state.device)*dt
        trajectory, frequency, valid = torch.stack(states), torch.stack(frequencies), torch.stack(masks)
        if matching_method == "continuous":
            from .continuous_matching import continuous_spline_peak
            from .numerical_attachment import natural_cubic_spline
            if frequency.ndim == 1:
                t_sub = time[valid]
                om_sub = frequency[valid]
                pt = continuous_spline_peak(t_sub, om_sub)
                sp = natural_cubic_spline(t_sub, om_sub, scalar_output=True)
                peak = OrbitalFrequencyPeak(pt, sp(pt), sp.derivative(pt, 2), torch.searchsorted(time, pt))
            else:
                flat_time = time.expand(frequency.shape[1:] + (time.shape[-1],)).reshape(-1, time.shape[-1])
                flat_omega = frequency.movedim(0, -1).reshape(-1, time.shape[-1])
                flat_valid = valid.movedim(0, -1).reshape(-1, time.shape[-1])
                pts, pfreqs, pcurvs, pidxs = [], [], [], []
                for t_i, om_i, v_i in zip(flat_time, flat_omega, flat_valid):
                    t_sub = t_i[v_i]
                    om_sub = om_i[v_i]
                    pt = continuous_spline_peak(t_sub, om_sub)
                    sp = natural_cubic_spline(t_sub, om_sub, scalar_output=True)
                    pts.append(pt)
                    pfreqs.append(sp(pt))
                    pcurvs.append(sp.derivative(pt, 2))
                    pidxs.append(torch.searchsorted(t_i, pt))
                peak = OrbitalFrequencyPeak(
                    torch.stack(pts).reshape(frequency.shape[1:]),
                    torch.stack(pfreqs).reshape(frequency.shape[1:]),
                    torch.stack(pcurvs).reshape(frequency.shape[1:]),
                    torch.stack(pidxs).reshape(frequency.shape[1:]),
                )
        elif matching_method == "grid":
            peak = orbital_frequency_peak(time, frequency.movedim(0, -1), valid=valid.movedim(0, -1))
        else:
            raise ValueError(f"matching_method must be 'continuous' or 'grid', got {matching_method!r}")
        return FrequencyPeakTrajectory(time, trajectory, frequency, valid, peak)

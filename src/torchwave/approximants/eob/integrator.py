"""Pure PyTorch Batched ODE Integrator for EOB Dynamical Trajectories.

Strict Clean-Room Implementation.
"""

from typing import Callable, List, Tuple
import torch

from torchwave.approximants.eob.hamiltonian import EOBHamiltonian
from torchwave.approximants.eob.radiation_reaction import EOBRadiationReaction


class EOBTrajectoryIntegrator:
    """Integrates the full conservative and dissipative EOB equations of motion."""

    def __init__(
        self,
        hamiltonian: EOBHamiltonian,
        radiation_reaction: EOBRadiationReaction,
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

"""Bounded clean-room TEOB nonspinning vacuum trajectory driver.

This module wires the reviewed clean multipolar radiation component to the
native TEOBResumS conservative equations.  It intentionally exposes a small
trajectory API: the fixed-step integrator is useful for short diagnostics and
the existing frequency-peak event integrator can be used when a bounded
plunge is required.  It does not choose NQC or merger matching conventions.

The clean route is selected only for zero aligned spins and zero
quadrupolar/octupolar tides.  Other intrinsic sectors belong to the existing
TEOBResumS fallback.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import torch

from torchwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions,
)
from torchwave.approximants.eob.integrator import (
    EOBTrajectoryIntegrator,
    FrequencyPeakTrajectory,
)
from torchwave.approximants.teobresums import (
    TEOBResumSHamiltonian,
    TEOBResumSRadiationReaction,
)


class TEOBVacuumTrajectory(NamedTuple):
    """A short clean vacuum trajectory and its Hamiltonian frequency samples."""

    time: torch.Tensor
    state: torch.Tensor
    frequency: torch.Tensor


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


class TEOBVacuumRadiationDriver:
    """Drive the clean 35-mode vacuum torque for a nonspinning TEOB system.

    ``include_horizon_absorption`` defaults to ``True`` for the compatibility
    vacuum route.  The flag is stored on the supplied radiation reaction so
    every RK stage uses the same explicit horizon policy.  A supplied
    radiation reaction is accepted for testing or specialization, but it must
    expose the clean TEOB force interface.
    """

    def __init__(
        self,
        hamiltonian: TEOBResumSHamiltonian,
        radiation_reaction: Optional[TEOBResumSRadiationReaction] = None,
        *,
        include_horizon_absorption: bool = True,
    ):
        _require_bool(include_horizon_absorption, "include_horizon_absorption")
        if not isinstance(hamiltonian, TEOBResumSHamiltonian):
            raise TypeError("hamiltonian must be a TEOBResumSHamiltonian")
        if not self._is_nonspinning_vacuum(hamiltonian):
            raise ValueError(
                "TEOBVacuumRadiationDriver requires zero spins and zero tides"
            )

        if radiation_reaction is None:
            radiation_reaction = TEOBResumSRadiationReaction(
                nu=hamiltonian.nu,
                chi1=hamiltonian.chi1,
                chi2=hamiltonian.chi2,
                include_horizon_absorption=include_horizon_absorption,
            )
        elif not hasattr(radiation_reaction, "radiation_reaction_forces"):
            raise TypeError("radiation_reaction must provide radiation_reaction_forces")

        # The integrator invokes the RR interface once per RK stage.  Keep the
        # selected policy on that object rather than relying on a one-off force
        # call, so all stages share the same horizon treatment.
        if hasattr(radiation_reaction, "include_horizon_absorption"):
            radiation_reaction.include_horizon_absorption = include_horizon_absorption

        self.hamiltonian = hamiltonian
        self.radiation_reaction = radiation_reaction
        self.include_horizon_absorption = include_horizon_absorption
        self.integrator = EOBTrajectoryIntegrator(hamiltonian, radiation_reaction)

    @staticmethod
    def _is_zero_tensor(value: object) -> bool:
        return isinstance(value, torch.Tensor) and bool(torch.all(value == 0))

    @classmethod
    def _is_nonspinning_vacuum(cls, hamiltonian: object) -> bool:
        return (
            cls._is_zero_tensor(getattr(hamiltonian, "chi1", None))
            and cls._is_zero_tensor(getattr(hamiltonian, "chi2", None))
            and cls._is_zero_tensor(getattr(hamiltonian, "lambda1", None))
            and cls._is_zero_tensor(getattr(hamiltonian, "lambda2", None))
            and cls._is_zero_tensor(getattr(hamiltonian, "kappa3_1", None))
            and cls._is_zero_tensor(getattr(hamiltonian, "kappa3_2", None))
        )

    def initial_state(
        self,
        omega_start: torch.Tensor,
        *,
        phase: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        """Solve the supplied positive orbital frequency and return ``(r,phi,p_r*,p_phi)``."""
        # Inspect caller values before casting to the Hamiltonian dtype.  A
        # complex-to-real ``as_tensor`` cast otherwise discards the imaginary
        # part and can silently solve a different requested frequency.
        omega_input = torch.as_tensor(omega_start)
        if omega_input.dtype.is_complex:
            raise TypeError("omega_start must be real-valued")
        omega_start = torch.as_tensor(
            omega_start,
            dtype=self.hamiltonian.nu.dtype,
            device=self.hamiltonian.nu.device,
        )
        if omega_start.dtype not in (torch.float32, torch.float64):
            raise TypeError("omega_start must use float32 or float64")
        if (not bool(torch.isfinite(omega_start).all())
                or bool((omega_start <= 0).any())):
            raise ValueError("omega_start must be finite and positive")
        phase_input = torch.as_tensor(phase)
        if phase_input.dtype.is_complex:
            raise TypeError("phase must be real-valued")
        phase = torch.as_tensor(
            phase,
            dtype=self.hamiltonian.nu.dtype,
            device=self.hamiltonian.nu.device,
        )
        if not bool(torch.isfinite(phase).all()):
            raise ValueError("phase must be finite")
        r0, pr0, pphi0 = find_quasicircular_initial_conditions(
            self.hamiltonian,
            self.radiation_reaction,
            omega_start,
        )
        phase = torch.broadcast_to(phase, r0.shape)
        if not bool(torch.isfinite(phase).all()):
            raise ValueError("phase must be finite")
        return torch.stack((r0, phase, pr0, pphi0), dim=-1)

    @staticmethod
    def _validate_state(state: torch.Tensor) -> None:
        if not isinstance(state, torch.Tensor):
            raise TypeError("state must be a torch.Tensor")
        if state.ndim < 1 or state.shape[-1] != 4:
            raise ValueError("state must have final axis (r, phi, p_r*, p_phi)")
        if state.dtype not in (torch.float32, torch.float64):
            raise TypeError("state must use float32 or float64")
        if not bool(torch.isfinite(state).all()):
            raise ValueError("state must be finite")
        if bool((state[..., 0] <= 0).any()):
            raise ValueError("state radius must be positive")

    def vector_field(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate the clean dissipative EOB vector field at ``state``."""
        self._validate_state(state)
        return self.integrator.vector_field(state)

    def integrate(
        self,
        initial_state: torch.Tensor,
        dt: float,
        num_steps: int,
        *,
        r_min: float = 2.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a bounded fixed-step RK4 trajectory in geometric time units."""
        self._validate_state(initial_state)
        self._validate_grid(dt, num_steps)
        if not math.isfinite(r_min) or r_min <= 0:
            raise ValueError("r_min must be finite and positive")
        return self.integrator.integrate(initial_state, dt, num_steps, r_min=r_min)

    def integrate_to_frequency_peak(
        self,
        initial_state: torch.Tensor,
        dt: float,
        num_steps: int,
        *,
        post_peak_duration: float = 0.0,
        richardson_extrapolation: bool = False,
    ) -> FrequencyPeakTrajectory:
        """Integrate through the first resolved orbital-frequency maximum."""
        self._validate_state(initial_state)
        self._validate_grid(dt, num_steps, minimum_steps=4)
        _require_bool(richardson_extrapolation, "richardson_extrapolation")
        return self.integrator.integrate_to_frequency_peak(
            initial_state,
            dt,
            num_steps,
            post_peak_duration=post_peak_duration,
            richardson_extrapolation=richardson_extrapolation,
        )

    @staticmethod
    def _validate_grid(dt: float, num_steps: int, *, minimum_steps: int = 0) -> None:
        if isinstance(dt, bool) or not isinstance(dt, (int, float)):
            raise TypeError("dt must be a real scalar")
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if isinstance(num_steps, bool) or not isinstance(num_steps, int):
            raise TypeError("num_steps must be an integer")
        if num_steps < minimum_steps:
            raise ValueError(f"num_steps must be at least {minimum_steps}")

    def trajectory(
        self,
        omega_start: torch.Tensor,
        dt: float,
        num_steps: int,
        *,
        phase: torch.Tensor | float = 0.0,
        r_min: float = 2.5,
    ) -> TEOBVacuumTrajectory:
        """Solve initial data, integrate briefly, and sample ``Omega`` consistently."""
        initial = self.initial_state(omega_start, phase=phase)
        time, state = self.integrate(initial, dt, num_steps, r_min=r_min)
        frequency = torch.stack(
            [self.vector_field(sample)[..., 1] for sample in state],
            dim=0,
        )
        return TEOBVacuumTrajectory(time=time, state=state, frequency=frequency)


# A short name is convenient for diagnostics while retaining the descriptive
# class name for callers that want the radiation policy to be explicit.
TEOBVacuumDriver = TEOBVacuumRadiationDriver


def integrate_teob_vacuum(
    hamiltonian: TEOBResumSHamiltonian,
    initial_state: torch.Tensor,
    dt: float,
    num_steps: int,
    *,
    include_horizon_absorption: bool = True,
    radiation_reaction: Optional[TEOBResumSRadiationReaction] = None,
    r_min: float = 2.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper around :class:`TEOBVacuumRadiationDriver`."""
    driver = TEOBVacuumRadiationDriver(
        hamiltonian,
        radiation_reaction,
        include_horizon_absorption=include_horizon_absorption,
    )
    return driver.integrate(initial_state, dt, num_steps, r_min=r_min)


__all__ = [
    "TEOBVacuumTrajectory",
    "TEOBVacuumRadiationDriver",
    "TEOBVacuumDriver",
    "integrate_teob_vacuum",
]

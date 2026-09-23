"""Development-only native SEOBNRv4 pre-evolution to an orbital frequency.

The quasicircular solve, orbital ODE, and crossing refinement use the supplied
native Hamiltonian and flux.  No sampled reference state is an input.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from torchwave.approximants.eob.adaptive_driver_clean import (
    GSL_STATE_SCALED_CONTROLLER,
    _rkf45_trial,
    integrate_adaptive,
)
from torchwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions,
)
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator


@dataclass(frozen=True)
class WarmStartResult:
    state: Tensor
    crossing_time: Tensor
    initial_state: Tensor
    accepted_steps: int
    evaluations: int
    frequency_residual: Tensor


class _FrequencyCrossing:
    """Stop at the first accepted state at or above the requested frequency."""

    def __init__(self, target: Tensor):
        self.target = target

    def check(self, time, state, derivative, previous, indices):
        if bool((state[..., 0] <= 0).any()):
            raise ValueError("native pre-evolution left the positive-radius domain")
        return derivative[..., 1] >= self.target


def native_frequency_warm_start(
    hamiltonian,
    flux,
    start_frequency: Tensor,
    target_frequency: Tensor,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-11,
    first_step_M: float = 0.125,
    frequency_tolerance: float = 1e-12,
    max_duration_M: float = 1e6,
) -> WarmStartResult:
    """Return a phase-zero native state at the first increasing-frequency crossing.

    The accepted trajectory uses the bounded Fehlberg 4/5 integrator.  Its
    final step brackets the target, and fixed Fehlberg substeps from the left
    accepted knot refine the crossing without interpolating model parameters.
    """
    start = torch.as_tensor(start_frequency)
    target = torch.as_tensor(target_frequency, dtype=start.dtype, device=start.device)
    if (start.ndim or target.ndim or start.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(start) & torch.isfinite(target))
            or not bool((start > 0) & (target > start))):
        raise ValueError("scalar finite frequencies must satisfy 0 < start < target")
    r, pr_star, pphi = find_quasicircular_initial_conditions(hamiltonian, flux, start)
    initial = torch.stack((r, torch.zeros_like(r), pr_star, pphi))
    field = EOBTrajectoryIntegrator(hamiltonian, flux).vector_field
    initial_frequency = field(initial)[1]
    if not bool(initial_frequency < target):
        raise ValueError("native initial orbital frequency must be below target")
    run = integrate_adaptive(
        field, initial, first_step_M, event=_FrequencyCrossing(target),
        atol=atol, rtol=rtol, initial_step=first_step_M,
        controller=GSL_STATE_SCALED_CONTROLLER,
        max_duration=max_duration_M,
    )
    if run.statistics.termination != ("event",) or run.time.numel() < 2:
        raise RuntimeError("native pre-evolution did not bracket the target frequency")
    left_time, right_time = run.time[-2], run.time[-1]
    left_state, right_state = run.state[-2], run.state[-1]
    left_derivative = run.derivative[-2]
    if not bool((left_derivative[1] < target) & (run.derivative[-1, 1] >= target)):
        raise RuntimeError("native frequency crossing is not bracketed")
    low = torch.zeros_like(left_time)
    high = right_time - left_time
    candidate = right_state
    residual = run.derivative[-1, 1] - target
    for _ in range(40):
        offset = (low + high) / 2
        _, candidate, _, _ = _rkf45_trial(
            field, left_time, left_state, offset, left_derivative, 1)
        residual = field(candidate)[1] - target
        if bool(residual.abs() <= frequency_tolerance):
            break
        if bool(residual < 0):
            low = offset
        else:
            high = offset
    else:
        raise RuntimeError("native frequency crossing refinement did not converge")
    # The waveform clock and phase begin at the requested-frequency crossing.
    state = torch.stack((candidate[0], torch.zeros_like(candidate[1]),
                         candidate[2], candidate[3]))
    return WarmStartResult(
        state=state, crossing_time=left_time+offset, initial_state=initial,
        accepted_steps=run.statistics.total_accepted,
        evaluations=run.statistics.total_evaluations,
        frequency_residual=residual,
    )

"""Development-only coarse/fine adaptive trajectory for a scalar autonomous EOB field.

The fine stage has a local clock and retains the coarse restart state's
continuous orbital phase. This builds only the pre-attachment trajectory;
NQC calibration and ringdown remain separate, unqualified work.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax.numpy as jnp

from .adaptive_driver_clean import (
    AcceptedTrajectory,
    AlignedSpinV4Event,
    integrate_adaptive_to_time,
    resample_accepted_trajectory,
)


class TwoStageTrajectory(NamedTuple):
    coarse: AcceptedTrajectory
    fine: AcceptedTrajectory
    coarse_time: jnp.ndarray
    coarse_state: jnp.ndarray
    fine_time: jnp.ndarray
    fine_state: jnp.ndarray
    restart_index: int
    refinement_factor: int
    time: jnp.ndarray
    state: jnp.ndarray


def integrate_two_stage_trajectory(
    vector_field: Callable,
    initial_state,
    *,
    output_step: float = 0.5,
    max_duration: float = 1000.0,
    coarse_initial_step: float | None = None,
    coarse_max_step: float | None = None,
    fine_max_step: float | None = None,
    fine_refinement_factor: int | None = None,
    atol: float = 1e-10,
    rtol: float = 1e-9,
    max_evaluations: int = 200000,
) -> TwoStageTrajectory:
    """Build the Torch-compatible pre-attachment two-stage output grid.

    ``vector_field(state)`` must be autonomous. Both stages must reach their
    accepted-state plunge event before ``max_duration``. The returned splice
    stops at the fine event; it does not invent ringdown samples.
    ``fine_refinement_factor`` fixes the fine input grid relative to the
    output grid for controlled numerical studies; by default the historical
    adaptive refinement rule is retained.
    """
    if not math.isfinite(float(output_step)) or output_step <= 0:
        raise ValueError("output_step must be positive and finite")
    if not math.isfinite(float(max_duration)) or max_duration <= 0:
        raise ValueError("max_duration must be positive and finite")
    dt = float(output_step)
    if fine_refinement_factor is not None and (
            type(fine_refinement_factor) is not int
            or fine_refinement_factor < 1
            or fine_refinement_factor & (fine_refinement_factor - 1)):
        raise ValueError("fine_refinement_factor must be a positive power of two")

    def field(_time, state):
        return vector_field(state)
    coarse = integrate_adaptive_to_time(
        field, initial_state, max_duration,
        event=AlignedSpinV4Event("coarse"),
        initial_step=dt if coarse_initial_step is None else coarse_initial_step,
        max_step=coarse_max_step, atol=atol, rtol=rtol,
        max_evaluations=max_evaluations,
    )
    if coarse.termination != "event":
        raise RuntimeError("coarse plunge event did not occur before max_duration")
    coarse_time, coarse_state = resample_accepted_trajectory(coarse, dt)
    n_coarse = len(coarse_time)
    if n_coarse < 4:
        raise RuntimeError("coarse trajectory has fewer than four output samples")
    duration = (n_coarse - 1) * dt
    if duration >= 150.0:
        restart_index = n_coarse - math.ceil(150.0 / dt)
    else:
        restart_index = n_coarse - math.ceil(n_coarse / 2.0)
    if not 1 <= restart_index < n_coarse:
        raise RuntimeError("restart index is outside the coarse trajectory")
    restart_state = coarse_state[restart_index]
    refinement = (fine_refinement_factor if fine_refinement_factor is not None
                  else 2 ** max(0, math.ceil(math.log2(50.0 * dt))))
    fine_step = dt / refinement
    fine = integrate_adaptive_to_time(
        field, restart_state, max_duration,
        event=AlignedSpinV4Event("fine"), initial_step=fine_step,
        max_step=fine_max_step, atol=atol, rtol=rtol,
        max_evaluations=max_evaluations,
    )
    if fine.termination != "event":
        raise RuntimeError("fine plunge event did not occur before max_duration")
    fine_time, fine_state = resample_accepted_trajectory(fine, fine_step)
    fine_indices = jnp.arange(0, len(fine_time), refinement)
    time = jnp.concatenate((coarse_time[:restart_index],
                            coarse_time[restart_index] + fine_time[fine_indices]))
    state = jnp.concatenate((coarse_state[:restart_index], fine_state[fine_indices]))
    return TwoStageTrajectory(coarse, fine, coarse_time, coarse_state,
                              fine_time, fine_state, restart_index,
                              refinement, time, state)


__all__ = ["TwoStageTrajectory", "integrate_two_stage_trajectory"]

"""Development-only JAX Fehlberg 4/5 core for the native v4 trajectory.

The embedded error is formed from derivative weights before adding the state.
This avoids cancellation when an accumulated orbital phase is large.  The
adaptive wrapper keeps accepted knots; event handling, two-stage restarts and
waveform-grid resampling are separate, still-unqualified work.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


_C = (0.0, 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2)
_B = (
    (),
    (1 / 4,),
    (3 / 32, 9 / 32),
    (1932 / 2197, -7200 / 2197, 7296 / 2197),
    (439 / 216, -8.0, 3680 / 513, -845 / 4104),
    (-8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40),
)
_B4 = (25 / 216, 0.0, 1408 / 2565, 2197 / 4104, -1 / 5, 0.0)
_B5 = (16 / 135, 0.0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55)
_B5_MINUS_B4 = (1 / 360, 0.0, -128 / 4275, -2197 / 75240, 1 / 50, 2 / 55)


class FixedTrial(NamedTuple):
    fourth: jnp.ndarray
    fifth: jnp.ndarray
    error: jnp.ndarray
    error_ratio: jnp.ndarray
    accepted: jnp.ndarray
    current_derivative: jnp.ndarray


class AcceptedTrajectory(NamedTuple):
    time: jnp.ndarray
    state: jnp.ndarray
    derivative: jnp.ndarray
    accepted_steps: int
    rejected_steps: int
    evaluations: int
    termination: str


class AlignedSpinV4Event:
    """Scalar accepted-knot plunge event matching the Torch v4 policy."""

    def __init__(self, stage: str = "coarse", *, radius: float = 6.0,
                 frequency_decreases: int = 5):
        if stage not in {"coarse", "fine"}:
            raise ValueError("stage must be 'coarse' or 'fine'")
        if not math.isfinite(radius) or radius <= 0 or frequency_decreases < 1:
            raise ValueError("event thresholds must be positive")
        self.stage = stage
        self.radius = float(radius)
        self.frequency_decreases = int(frequency_decreases)
        self.count = 0

    def check(self, _time, state, derivative, previous):
        inside = bool(state[0] < self.radius)
        decrease = inside and previous is not None and bool(derivative[1] < previous[1])
        if decrease:
            self.count += 1
        radial_nonnegative = bool(derivative[2] >= 0)
        if self.stage == "coarse":
            return inside and (decrease or radial_nonnegative)
        return inside and (self.count >= self.frequency_decreases or radial_nonnegative)


def _weighted(stages, coefficients):
    return jnp.sum(jnp.stack(stages, axis=0) * jnp.asarray(
        coefficients, dtype=stages[0].dtype
    ).reshape((-1,) + (1,) * stages[0].ndim), axis=0)


def fixed_fehlberg45_step(
    vector_field: Callable,
    time,
    state,
    step,
    *,
    atol: float = 1e-13,
    rtol: float = 1e-11,
    current_derivative=None,
) -> FixedTrial:
    """Return one unretried RKF45 trial for ``field(time, state)``.

    Works with scalar or batched states and under ``jax.jit``.  The caller
    supplies k1 when it has already evaluated the field at the current knot.
    """
    state = jnp.asarray(state)
    time = jnp.asarray(time, dtype=state.dtype)
    step = jnp.asarray(step, dtype=state.dtype)
    k1 = vector_field(time, state) if current_derivative is None else current_derivative
    stages = [k1]
    for index in range(1, 6):
        argument = state + jnp.expand_dims(step, -1) * _weighted(stages, _B[index])
        stages.append(vector_field(time + _C[index] * step, argument))
    scaled_step = jnp.expand_dims(step, -1)
    fourth = state + scaled_step * _weighted(stages, _B4)
    fifth = state + scaled_step * _weighted(stages, _B5)
    error = scaled_step * _weighted(stages, _B5_MINUS_B4)
    ratio = jnp.max(jnp.abs(error) / (atol + rtol * jnp.abs(fifth)), axis=-1)
    return FixedTrial(fourth, fifth, error, ratio, ratio <= 1, k1)


def state_scaled_controller_adjustment(error_ratio, *, safety: float = 0.9):
    """Return the public state-scaled RKF45 action and next-step factor."""
    ratio = jnp.asarray(error_ratio)
    safe = jnp.maximum(ratio, jnp.finfo(ratio.dtype).tiny)
    shrink = ratio > 1.1
    grow = ratio < 0.5
    shrink_factor = jnp.maximum(0.2, safety * safe ** (-1 / 5))
    growth_factor = jnp.minimum(5.0, jnp.maximum(1.0, safety * safe ** (-1 / 6)))
    return (jnp.where(shrink, -1, jnp.where(grow, 1, 0)),
            jnp.where(shrink, shrink_factor,
                      jnp.where(grow, growth_factor, jnp.ones_like(ratio))))


def integrate_adaptive_to_time(
    vector_field: Callable,
    initial_state,
    t_end: float,
    *,
    initial_step: float,
    event: AlignedSpinV4Event | None = None,
    atol: float = 1e-10,
    rtol: float = 1e-9,
    max_step: float | None = None,
    max_rejected_steps: int = 10000,
    max_evaluations: int = 200000,
    max_accepted_steps: int = 100000,
) -> AcceptedTrajectory:
    """Integrate one state vector to a fixed time, retaining accepted knots.

    Controller decisions are discrete Python branches.  The accepted-step
    derivative is available through :func:`replay_accepted_steps`; this
    wrapper itself is not a JIT-able or differentiable whole-trajectory driver.
    """
    if not all(math.isfinite(float(x)) and float(x) > 0 for x in (t_end, initial_step, atol)):
        raise ValueError("t_end, initial_step and atol must be positive and finite")
    if not math.isfinite(float(rtol)) or rtol < 0:
        raise ValueError("rtol must be finite and nonnegative")
    if max_step is not None and (not math.isfinite(float(max_step)) or max_step <= 0):
        raise ValueError("max_step must be positive and finite")
    state = jnp.asarray(initial_state)
    if state.ndim != 1 or not jnp.issubdtype(state.dtype, jnp.floating):
        raise ValueError("initial_state must be a floating-point vector")
    time = jnp.asarray(0.0, dtype=state.dtype)
    derivative = vector_field(time, state)
    times, states, derivatives = [time], [state], [derivative]
    evaluations, rejected = 1, 0
    step = min(initial_step, max_step) if max_step is not None else initial_step
    if event is not None and bool(event.check(time, state, derivative, None)):
        raise RuntimeError("event fired at initial state")
    termination = "duration"
    while float(time) < t_end:
        if len(times) - 1 >= max_accepted_steps or evaluations + 6 > max_evaluations:
            raise RuntimeError("adaptive integration work limit reached")
        step = min(step, t_end - float(time))
        if step <= 0 or float(time + step) == float(time):
            raise RuntimeError("adaptive step underflow")
        trial = fixed_fehlberg45_step(vector_field, time, state, step,
                                      atol=atol, rtol=rtol, current_derivative=derivative)
        evaluations += 5
        ratio = float(trial.error_ratio)
        if not math.isfinite(ratio):
            raise RuntimeError("nonfinite embedded error")
        action, factor = state_scaled_controller_adjustment(trial.error_ratio)
        if int(action) >= 0:
            previous_derivative = derivative
            next_time = time + step
            next_state = trial.fifth
            next_derivative = vector_field(next_time, next_state)
            evaluations += 1
            if not bool(jnp.all(jnp.isfinite(next_state)) & jnp.all(jnp.isfinite(next_derivative))):
                raise RuntimeError("nonfinite accepted state or derivative")
            time, state, derivative = next_time, next_state, next_derivative
            times.append(time)
            states.append(state)
            derivatives.append(derivative)
            if event is not None and bool(event.check(time, state, derivative, previous_derivative)):
                termination = "event"
                break
        else:
            rejected += 1
            if rejected > max_rejected_steps:
                raise RuntimeError("adaptive rejection limit reached")
        step *= float(factor)
        if max_step is not None:
            step = min(step, max_step)
    return AcceptedTrajectory(jnp.stack(times), jnp.stack(states),
                              jnp.stack(derivatives), len(times) - 1, rejected,
                              evaluations, termination)


def replay_accepted_steps(vector_field: Callable, initial_state, accepted_time):
    """Replay a frozen accepted mesh with differentiable RKF45 arithmetic.

    This is a pathwise derivative holding the discrete controller decisions
    and accepted times fixed, not a derivative of the adaptive event time.
    """
    time = jnp.asarray(accepted_time)
    state = jnp.asarray(initial_state)

    def advance(current, interval):
        start, end = interval
        trial = fixed_fehlberg45_step(vector_field, start, current, end - start)
        return trial.fifth, trial.fifth

    _, subsequent = jax.lax.scan(advance, state, (time[:-1], time[1:]))
    return jnp.concatenate((state[None, :], subsequent), axis=0)


__all__ = ["FixedTrial", "AcceptedTrajectory", "AlignedSpinV4Event", "fixed_fehlberg45_step",
           "state_scaled_controller_adjustment", "integrate_adaptive_to_time",
           "replay_accepted_steps"]

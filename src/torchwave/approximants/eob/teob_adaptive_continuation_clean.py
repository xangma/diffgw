"""Bounded clean-room TEOB PA-to-ODE continuation.

The module owns the small trajectory hand-off needed after the scalar
nonspinning post-adiabatic profile.  It reuses the reviewed adaptive Fehlberg
driver and its fixed-step primitive; no merger, NQC, calibration, or waveform
assembly is performed here.

The generic entry point is useful for synthetic controller and event tests.
``continue_teob_adaptive`` wires the same continuation to the existing clean
TEOB vacuum vector field and requires a completed PA result (or an explicit
initial state and step when the PA result was skipped).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from .adaptive_driver_clean import (
    AdaptiveIntegrationError,
    _TrialDomainFailure,
    _call_field,
    _field_arity,
    fixed_fehlberg45_step,
    integrate_adaptive,
)
from .teob_postadiabatic_clean import TEOBPostAdiabaticResult
from .teob_vacuum_driver_clean import TEOBVacuumRadiationDriver


DEFAULT_ATOL = 1.0e-13
DEFAULT_RTOL = 1.0e-11
DEFAULT_FIXED_STEP_CAP = 0.1
DEFAULT_CONTINUATION_DURATION = 2.0
DEFAULT_RADIUS_FLOOR = 1.0
DEFAULT_MAX_REJECTED_STEPS = 10_000
DEFAULT_MAX_EVALUATIONS = 30_000
DEFAULT_MAX_ACCEPTED_STEPS = 2_000
DEFAULT_MAX_DURATION = 6_000.0


class TEOBAdaptiveContinuationError(RuntimeError):
    """A bounded continuation failure with its accepted-prefix diagnostics."""

    def __init__(
        self,
        reason: str,
        message: str,
        *,
        result: Optional["TEOBAdaptiveContinuationResult"] = None,
        time: Optional[Tensor] = None,
        state: Optional[Tensor] = None,
        evaluations: Optional[int] = None,
        accepted_steps: Optional[int] = None,
        rejected_steps: Optional[int] = None,
    ):
        super().__init__(message)
        self.reason = str(reason)
        self.result = result
        self.time = time
        self.state = state
        self.evaluations = evaluations
        self.accepted_steps = accepted_steps
        self.rejected_steps = rejected_steps


@dataclass(frozen=True)
class TEOBContinuationStatistics:
    """Scalar work counters for the adaptive and fixed continuation stages."""

    adaptive_accepted_steps: int
    adaptive_rejected_steps: int
    adaptive_evaluations: int
    fixed_accepted_steps: int
    fixed_rejected_steps: int
    fixed_evaluations: int
    termination: str

    @property
    def accepted_steps(self) -> int:
        return self.adaptive_accepted_steps + self.fixed_accepted_steps

    @property
    def rejected_steps(self) -> int:
        return self.adaptive_rejected_steps + self.fixed_rejected_steps

    @property
    def evaluations(self) -> int:
        return self.adaptive_evaluations + self.fixed_evaluations

    @property
    def total_accepted(self) -> int:
        return self.accepted_steps

    @property
    def total_rejected(self) -> int:
        return self.rejected_steps

    @property
    def total_evaluations(self) -> int:
        return self.evaluations


@dataclass(frozen=True)
class TEOBAdaptiveContinuationResult:
    """PA knots plus accepted adaptive and fixed ODE continuation knots."""

    time: Tensor
    state: Tensor
    frequency: Tensor
    valid: Tensor
    pa_time: Optional[Tensor]
    pa_state: Optional[Tensor]
    pa_frequency: Optional[Tensor]
    adaptive_time: Tensor
    adaptive_state: Tensor
    adaptive_frequency: Tensor
    fixed_time: Tensor
    fixed_state: Tensor
    fixed_frequency: Tensor
    peak_event_index: Optional[int]
    peak_event_time: Optional[Tensor]
    peak_event_frequency: Optional[Tensor]
    phase_continuity: Tensor
    first_ode_step: Tensor
    adaptive_next_step: Tensor
    fixed_step: Optional[Tensor]
    target_time: Optional[Tensor]
    status: str
    skip_reason: Optional[str]
    statistics: TEOBContinuationStatistics

    @property
    def endpoint_state(self) -> Tensor:
        """Final state in the accepted trajectory prefix."""
        return self.state[-1]

    @property
    def endpoint_time(self) -> Tensor:
        """Final absolute geometric time in the accepted trajectory prefix."""
        return self.time[-1]

    @property
    def event_index(self) -> Optional[int]:
        return self.peak_event_index

    @property
    def event_time(self) -> Optional[Tensor]:
        return self.peak_event_time

    @property
    def adaptive_initial_step(self) -> Tensor:
        """The mandated first adaptive trial size."""
        return self.first_ode_step

    @property
    def peak_time(self) -> Optional[Tensor]:
        return self.peak_event_time

    @property
    def phase_seam_error(self) -> Tensor:
        """Difference between the transferred PA phase and ODE phase."""
        return self.phase_continuity

    @property
    def phase_continuous(self) -> bool:
        return bool(self.phase_continuity == 0)

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    @property
    def metadata(self) -> dict[str, object]:
        """Machine-readable scheduling and event metadata."""
        return {
            "status": self.status,
            "skip_reason": self.skip_reason,
            "peak_event_index": self.peak_event_index,
            "peak_event_time": self.peak_event_time,
            "first_ode_step": self.first_ode_step,
            "adaptive_next_step": self.adaptive_next_step,
            "fixed_step": self.fixed_step,
            "target_time": self.target_time,
            "phase_continuity": self.phase_continuity,
            "adaptive_accepted_steps": self.statistics.adaptive_accepted_steps,
            "fixed_accepted_steps": self.statistics.fixed_accepted_steps,
        }


class AcceptedFrequencyDecreaseEvent:
    """Stop on the first strict accepted-state frequency decrease.

    The frequency is the second component of the vector-field derivative,
    namely ``d phi / dt`` in the scalar zero-spin sector.  A radius below the
    supplied floor is reported separately so an unresolved peak cannot be
    mistaken for an event.
    """

    def __init__(self, *, radius_floor: float = DEFAULT_RADIUS_FLOOR):
        if isinstance(radius_floor, bool) or not math.isfinite(float(radius_floor)):
            raise ValueError("radius_floor must be finite")
        if float(radius_floor) <= 0:
            raise ValueError("radius_floor must be positive")
        self.radius_floor = float(radius_floor)

    def initialize(self, rows: int, *, dtype: torch.dtype, device: torch.device):
        if rows != 1:
            raise ValueError("AcceptedFrequencyDecreaseEvent is scalar-only")

    def check(
        self,
        time: Tensor,
        state: Tensor,
        derivative: Tensor,
        previous: Optional[Tensor],
        indices: Tensor,
    ):
        if time.numel() != 1 or state.shape[0] != 1:
            raise ValueError("AcceptedFrequencyDecreaseEvent is scalar-only")
        below_floor = state[..., 0] < self.radius_floor
        if previous is None:
            decrease = torch.zeros_like(below_floor)
        else:
            decrease = derivative[..., 1] < previous[..., 1]
        stop = decrease | below_floor
        # Reaching the floor at the same accepted sample as a decrease still
        # leaves the peak unresolved.  Keep the domain failure ahead of the
        # event label so callers cannot mistake an under-resolved plunge for a
        # completed continuation.
        if bool(below_floor.any()):
            return stop, "radius_below_one_before_frequency_decrease"
        if bool(decrease.any()):
            return stop, "frequency_decrease"
        return stop, None


def _scalar(value: object, reference: Tensor, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.ndim != 0 or not bool(torch.isfinite(result)):
        raise ValueError(f"{name} must be a finite scalar")
    if positive and bool(result <= 0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and bool(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validate_limits(
    *,
    atol: float,
    rtol: float,
    max_rejected_steps: int,
    max_evaluations: int,
    max_accepted_steps: int,
    max_duration: object,
    fixed_step_cap: object,
    continuation_duration: object,
    reference: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if isinstance(atol, bool) or not math.isfinite(float(atol)) or float(atol) <= 0:
        raise ValueError("atol must be finite and positive")
    if isinstance(rtol, bool) or not math.isfinite(float(rtol)) or float(rtol) < 0:
        raise ValueError("rtol must be finite and nonnegative")
    for name, value in (
        ("max_rejected_steps", max_rejected_steps),
        ("max_evaluations", max_evaluations),
        ("max_accepted_steps", max_accepted_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    duration = _scalar(max_duration, reference, "max_duration", positive=True)
    cap = _scalar(fixed_step_cap, reference, "fixed_step_cap", positive=True)
    continuation = _scalar(
        continuation_duration, reference, "continuation_duration", positive=True,
    )
    return duration, cap, continuation


def _validate_prefix(
    prefix_time: Optional[Tensor],
    prefix_state: Optional[Tensor],
    prefix_frequency: Optional[Tensor],
    *,
    state: Tensor,
    start_time: Tensor,
) -> bool:
    values = (prefix_time, prefix_state, prefix_frequency)
    if all(value is None for value in values):
        return False
    if any(value is None for value in values):
        raise ValueError("prefix_time, prefix_state, and prefix_frequency must be supplied together")
    assert prefix_time is not None and prefix_state is not None
    assert prefix_frequency is not None
    if (prefix_time.ndim != 1 or prefix_state.ndim != 2
            or prefix_frequency.ndim != 1):
        raise ValueError("prefix arrays must be one-dimensional or (samples, components)")
    if (prefix_state.shape[0] != prefix_time.numel()
            or prefix_frequency.numel() != prefix_time.numel()
            or prefix_state.shape[-1] != state.numel()):
        raise ValueError("prefix arrays have inconsistent sample shapes")
    if prefix_time.numel() < 1:
        raise ValueError("prefix must contain at least one accepted knot")
    for name, value in (
        ("prefix_time", prefix_time),
        ("prefix_state", prefix_state),
        ("prefix_frequency", prefix_frequency),
    ):
        if value.dtype != state.dtype or value.device != state.device:
            raise TypeError(f"{name} must match initial_state dtype and device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
    if bool(torch.any(prefix_time[1:] <= prefix_time[:-1])):
        raise ValueError("prefix_time must be strictly increasing")
    if not torch.equal(prefix_time[-1], start_time):
        raise ValueError("prefix endpoint time must equal start_time")
    if not torch.equal(prefix_state[-1], state):
        raise ValueError("initial_state must equal the transferred prefix endpoint")
    return True


def _empty_segment(state: Tensor, *, frequency: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    times = torch.empty((0,), dtype=state.dtype, device=state.device)
    states = torch.empty((0, state.numel()), dtype=state.dtype, device=state.device)
    freqs = torch.empty((0,), dtype=state.dtype, device=state.device)
    return times, states, freqs


def _make_result(
    *,
    prefix: bool,
    prefix_time: Optional[Tensor],
    prefix_state: Optional[Tensor],
    prefix_frequency: Optional[Tensor],
    adaptive_time: Tensor,
    adaptive_state: Tensor,
    adaptive_frequency: Tensor,
    fixed_time: Tensor,
    fixed_state: Tensor,
    fixed_frequency: Tensor,
    peak_event_index: Optional[int],
    peak_event_time: Optional[Tensor],
    peak_event_frequency: Optional[Tensor],
    phase_continuity: Tensor,
    first_ode_step: Tensor,
    adaptive_next_step: Tensor,
    fixed_step: Optional[Tensor],
    target_time: Optional[Tensor],
    status: str,
    skip_reason: Optional[str],
    statistics: TEOBContinuationStatistics,
    initial_state: Tensor,
    start_time: Tensor,
) -> TEOBAdaptiveContinuationResult:
    if prefix:
        assert prefix_time is not None and prefix_state is not None
        assert prefix_frequency is not None
        combined_time_parts = [prefix_time, adaptive_time[1:]]
        combined_state_parts = [prefix_state, adaptive_state[1:]]
        combined_frequency_parts = [prefix_frequency, adaptive_frequency[1:]]
    else:
        # With no PA profile, the initial accepted state is the first adaptive
        # knot and is retained in the returned ODE trajectory.
        combined_time_parts = [adaptive_time]
        combined_state_parts = [adaptive_state]
        combined_frequency_parts = [adaptive_frequency]
    if fixed_time.numel() > 1:
        combined_time_parts.append(fixed_time[1:])
        combined_state_parts.append(fixed_state[1:])
        combined_frequency_parts.append(fixed_frequency[1:])
    time = torch.cat(combined_time_parts, dim=0)
    state = torch.cat(combined_state_parts, dim=0)
    frequency = torch.cat(combined_frequency_parts, dim=0)
    valid = torch.ones(time.shape, dtype=torch.bool, device=time.device)
    return TEOBAdaptiveContinuationResult(
        time=time,
        state=state,
        frequency=frequency,
        valid=valid,
        pa_time=prefix_time if prefix else None,
        pa_state=prefix_state if prefix else None,
        pa_frequency=prefix_frequency if prefix else None,
        adaptive_time=adaptive_time,
        adaptive_state=adaptive_state,
        adaptive_frequency=adaptive_frequency,
        fixed_time=fixed_time,
        fixed_state=fixed_state,
        fixed_frequency=fixed_frequency,
        peak_event_index=peak_event_index,
        peak_event_time=peak_event_time,
        peak_event_frequency=peak_event_frequency,
        phase_continuity=phase_continuity,
        first_ode_step=first_ode_step,
        adaptive_next_step=adaptive_next_step,
        fixed_step=fixed_step,
        target_time=target_time,
        status=status,
        skip_reason=skip_reason,
        statistics=statistics,
    )


def integrate_adaptive_continuation(
    vector_field: Callable,
    initial_state: Tensor,
    initial_step: Tensor | float,
    *,
    start_time: Tensor | float = 0.0,
    prefix_time: Optional[Tensor] = None,
    prefix_state: Optional[Tensor] = None,
    prefix_frequency: Optional[Tensor] = None,
    output_step: Tensor | float | None = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    min_step: Optional[float] = None,
    max_step: Optional[float] = None,
    max_rejected_steps: int = DEFAULT_MAX_REJECTED_STEPS,
    max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
    max_accepted_steps: int = DEFAULT_MAX_ACCEPTED_STEPS,
    max_duration: float = DEFAULT_MAX_DURATION,
    fixed_step_cap: float = DEFAULT_FIXED_STEP_CAP,
    continuation_duration: float = DEFAULT_CONTINUATION_DURATION,
    radius_floor: float = DEFAULT_RADIUS_FLOOR,
    raise_on_failure: bool = True,
) -> TEOBAdaptiveContinuationResult:
    """Continue a scalar trajectory through an accepted frequency peak.

    ``prefix_*`` contains the original PA knots.  Its last sample must equal
    ``initial_state`` and ``start_time`` exactly; that sample is emitted only
    once in the combined result.  If no prefix is supplied, the caller owns
    the explicit ODE initial state and step.
    """
    if not isinstance(initial_state, Tensor):
        raise TypeError("initial_state must be a torch.Tensor")
    if initial_state.ndim != 1 or initial_state.numel() < 2:
        raise ValueError("initial_state must be one scalar state vector with at least two components")
    if initial_state.dtype not in (torch.float32, torch.float64):
        raise TypeError("initial_state must use float32 or float64")
    if not bool(torch.isfinite(initial_state).all()):
        raise ValueError("initial_state must be finite")
    dtype, device = initial_state.dtype, initial_state.device
    start_time_t = _scalar(start_time, initial_state, "start_time")
    initial_step_t = _scalar(initial_step, initial_state, "initial_step", positive=True)
    duration_limit, fixed_cap_t, continuation_t = _validate_limits(
        atol=atol,
        rtol=rtol,
        max_rejected_steps=max_rejected_steps,
        max_evaluations=max_evaluations,
        max_accepted_steps=max_accepted_steps,
        max_duration=max_duration,
        fixed_step_cap=fixed_step_cap,
        continuation_duration=continuation_duration,
        reference=initial_state,
    )
    if not isinstance(raise_on_failure, bool):
        raise TypeError("raise_on_failure must be a bool")
    if bool(initial_state[0] < float(radius_floor)):
        raise TEOBAdaptiveContinuationError(
            "radius_below_one_before_frequency_decrease",
            "initial radius is below the pre-event domain floor",
            time=start_time_t,
            state=initial_state,
        )
    prefix = _validate_prefix(
        prefix_time,
        prefix_state,
        prefix_frequency,
        state=initial_state,
        start_time=start_time_t,
    )

    # A completed PA profile mandates its first ODE trial size.  Reject a
    # caller bound that would silently clamp that exact half-interval.
    if min_step is not None and float(min_step) > float(initial_step_t):
        raise ValueError("min_step would change the mandated first ODE trial")
    if max_step is not None and float(max_step) < float(initial_step_t):
        raise ValueError("max_step would change the mandated first ODE trial")
    out_step = initial_step_t if output_step is None else _scalar(
        output_step, initial_state, "output_step", positive=True,
    )

    field_arity = _field_arity(vector_field)
    if field_arity == 2:
        def local_field(local_time: Tensor, state: Tensor) -> Tensor:
            return vector_field(start_time_t + local_time, state)

        local_field.time_dependent = True
    else:
        def local_field(state: Tensor) -> Tensor:
            return vector_field(state)

    event = AcceptedFrequencyDecreaseEvent(radius_floor=radius_floor)
    adaptive: Optional[object] = None
    try:
        adaptive = integrate_adaptive(
            local_field,
            initial_state,
            out_step,
            event=event,
            atol=atol,
            rtol=rtol,
            controller="gsl_state_scaled",
            batch_mode="active",
            initial_step=initial_step_t,
            min_step=min_step,
            max_step=max_step,
            max_rejected_steps=max_rejected_steps,
            max_evaluations=max_evaluations,
            max_accepted_steps=max_accepted_steps,
            max_duration=duration_limit,
        )
    except AdaptiveIntegrationError as exc:
        # The shared driver carries every accepted knot accumulated before the
        # failing trial.  Consume that prefix directly, preserving absolute
        # time and the already evaluated frequency derivative.  In particular,
        # do not probe the initial RHS again: a retry would inflate the work
        # counter and could fabricate a frequency after a domain failure.
        partial_time = exc.accepted_time
        partial_state = exc.accepted_state
        partial_derivative = exc.accepted_derivative
        if isinstance(partial_time, tuple):
            if len(partial_time) != 1:
                raise RuntimeError(
                    "continuation received a non-scalar adaptive failure prefix"
                ) from exc
            partial_time = partial_time[0]
            partial_state = partial_state[0]
            partial_derivative = partial_derivative[0]
        if (isinstance(partial_time, Tensor)
                and isinstance(partial_state, Tensor)
                and isinstance(partial_derivative, Tensor)):
            adaptive_time_local = partial_time
            adaptive_state = partial_state
            adaptive_frequency = partial_derivative[..., 1]
        else:
            # If the very first RHS evaluation failed, there is no accepted
            # derivative to report.  A supplied PA prefix still provides its
            # endpoint frequency; otherwise NaN marks the unavailable value
            # explicitly instead of silently substituting zero.
            adaptive_time_local = torch.zeros((1,), dtype=dtype, device=device)
            adaptive_state = initial_state.unsqueeze(0)
            adaptive_frequency = torch.full(
                (1,), float("nan"), dtype=dtype, device=device,
            )
            if prefix:
                assert prefix_frequency is not None
                adaptive_frequency[0] = prefix_frequency[-1]
        adaptive_time = start_time_t + adaptive_time_local
        fixed_time, fixed_state, fixed_frequency = _empty_segment(initial_state)

        def _count(value: object, default: int = 0) -> int:
            if value is None:
                return default
            if isinstance(value, Tensor):
                return int(value.detach().sum().item())
            return int(value)

        adaptive_accepted = _count(
            exc.accepted_steps,
            max(int(adaptive_time_local.numel()) - 1, 0)
            if isinstance(partial_time, Tensor) else 0,
        )
        adaptive_evals = _count(exc.evaluations)
        adaptive_rej = _count(exc.rejections)
        stats = TEOBContinuationStatistics(
            adaptive_accepted_steps=adaptive_accepted,
            adaptive_rejected_steps=adaptive_rej,
            adaptive_evaluations=adaptive_evals,
            fixed_accepted_steps=0,
            fixed_rejected_steps=0,
            fixed_evaluations=0,
            termination=exc.reason,
        )
        result = _make_result(
            prefix=prefix,
            prefix_time=prefix_time,
            prefix_state=prefix_state,
            prefix_frequency=prefix_frequency,
            adaptive_time=adaptive_time,
            adaptive_state=adaptive_state,
            adaptive_frequency=adaptive_frequency,
            fixed_time=fixed_time,
            fixed_state=fixed_state,
            fixed_frequency=fixed_frequency,
            peak_event_index=None,
            peak_event_time=None,
            peak_event_frequency=None,
            phase_continuity=(adaptive_state[0, 1] - prefix_state[-1, 1]
                              if prefix else torch.zeros((), dtype=dtype, device=device)),
            first_ode_step=initial_step_t,
            adaptive_next_step=initial_step_t,
            fixed_step=None,
            target_time=None,
            status="failed",
            skip_reason=exc.reason,
            statistics=stats,
            initial_state=initial_state,
            start_time=start_time_t,
        )
        failure_time = exc.time
        if isinstance(failure_time, Tensor):
            failure_time = start_time_t + failure_time
        failure_state = exc.state
        if not isinstance(failure_state, Tensor):
            failure_state = adaptive_state[-1]
        error = TEOBAdaptiveContinuationError(
            exc.reason,
            str(exc),
            result=result,
            time=failure_time,
            state=failure_state,
            evaluations=adaptive_evals,
            accepted_steps=adaptive_accepted,
            rejected_steps=adaptive_rej,
        )
        if raise_on_failure:
            raise error from exc
        return result

    assert adaptive is not None
    adaptive_time_local = adaptive.time
    adaptive_time = start_time_t + adaptive_time_local
    adaptive_state = adaptive.state
    adaptive_frequency = adaptive.derivative[..., 1]
    adaptive_next_step = adaptive.next_step
    if adaptive_next_step is None:
        # This fallback is only for an older in-memory driver object; current
        # adaptive_driver_clean always returns the additive metadata.
        adaptive_next_step = adaptive.initial_step
    adaptive_next_step = adaptive_next_step.reshape(())
    adaptive_accepted = adaptive.statistics.total_accepted
    adaptive_rejected = adaptive.statistics.total_rejected
    adaptive_evaluations = adaptive.statistics.total_evaluations
    termination = adaptive.statistics.termination[0]

    if termination != "frequency_decrease":
        # Radius and unresolved-budget terminations are failures to resolve
        # the requested accepted-state peak, never complete trajectories.
        reason = termination
        message = {
            "radius_below_one_before_frequency_decrease":
                "radius fell below one before an accepted frequency decrease",
            "duration": "duration budget ended before an accepted frequency decrease",
        }.get(reason, "adaptive continuation ended before an accepted frequency decrease")
        fixed_time, fixed_state, fixed_frequency = _empty_segment(initial_state)
        stats = TEOBContinuationStatistics(
            adaptive_accepted_steps=adaptive_accepted,
            adaptive_rejected_steps=adaptive_rejected,
            adaptive_evaluations=adaptive_evaluations,
            fixed_accepted_steps=0,
            fixed_rejected_steps=0,
            fixed_evaluations=0,
            termination=reason,
        )
        result = _make_result(
            prefix=prefix,
            prefix_time=prefix_time,
            prefix_state=prefix_state,
            prefix_frequency=prefix_frequency,
            adaptive_time=adaptive_time,
            adaptive_state=adaptive_state,
            adaptive_frequency=adaptive_frequency,
            fixed_time=fixed_time,
            fixed_state=fixed_state,
            fixed_frequency=fixed_frequency,
            peak_event_index=None,
            peak_event_time=None,
            peak_event_frequency=None,
            phase_continuity=(adaptive_state[0, 1] - prefix_state[-1, 1]
                              if prefix else torch.zeros((), dtype=dtype, device=device)),
            first_ode_step=initial_step_t,
            adaptive_next_step=adaptive_next_step,
            fixed_step=None,
            target_time=None,
            status="failed",
            skip_reason=reason,
            statistics=stats,
            initial_state=initial_state,
            start_time=start_time_t,
        )
        error = TEOBAdaptiveContinuationError(
            reason,
            message,
            result=result,
            time=adaptive_time[-1],
            state=adaptive_state[-1],
            evaluations=adaptive_evaluations,
            accepted_steps=adaptive_accepted,
            rejected_steps=adaptive_rejected,
        )
        if raise_on_failure:
            raise error
        return result

    event_local_time = adaptive_time_local[-1]
    event_time = adaptive_time[-1]
    event_frequency = adaptive_frequency[-1]
    target_local_time = event_local_time + continuation_t
    target_time = start_time_t + target_local_time
    if bool(target_local_time > duration_limit):
        reason = "duration_budget"
        fixed_time, fixed_state, fixed_frequency = _empty_segment(initial_state)
        stats = TEOBContinuationStatistics(
            adaptive_accepted_steps=adaptive_accepted,
            adaptive_rejected_steps=adaptive_rejected,
            adaptive_evaluations=adaptive_evaluations,
            fixed_accepted_steps=0,
            fixed_rejected_steps=0,
            fixed_evaluations=0,
            termination=reason,
        )
        result = _make_result(
            prefix=prefix,
            prefix_time=prefix_time,
            prefix_state=prefix_state,
            prefix_frequency=prefix_frequency,
            adaptive_time=adaptive_time,
            adaptive_state=adaptive_state,
            adaptive_frequency=adaptive_frequency,
            fixed_time=fixed_time,
            fixed_state=fixed_state,
            fixed_frequency=fixed_frequency,
            peak_event_index=(prefix_time.numel() - 1 + adaptive_time.numel() - 1
                              if prefix else adaptive_time.numel() - 1),
            peak_event_time=event_time,
            peak_event_frequency=event_frequency,
            phase_continuity=(adaptive_state[0, 1] - prefix_state[-1, 1]
                              if prefix else torch.zeros((), dtype=dtype, device=device)),
            first_ode_step=initial_step_t,
            adaptive_next_step=adaptive_next_step,
            fixed_step=torch.minimum(fixed_cap_t, adaptive_next_step),
            target_time=target_time,
            status="failed",
            skip_reason=reason,
            statistics=stats,
            initial_state=initial_state,
            start_time=start_time_t,
        )
        error = TEOBAdaptiveContinuationError(
            reason,
            "duration budget cannot accommodate the required fixed continuation",
            result=result,
            time=event_time,
            state=adaptive_state[-1],
            evaluations=adaptive_evaluations,
            accepted_steps=adaptive_accepted,
            rejected_steps=adaptive_rejected,
        )
        if raise_on_failure:
            raise error
        return result

    fixed_step = torch.minimum(fixed_cap_t, adaptive_next_step)
    fixed_time_values = [event_local_time]
    fixed_state_values = [adaptive_state[-1]]
    fixed_frequency_values = [event_frequency]
    current_local_time = event_local_time
    current_state = adaptive_state[-1]
    current_derivative = adaptive.derivative[-1]
    fixed_accepted = 0
    fixed_rejected = 0
    fixed_evaluations = 0

    def partial_result(status: str, reason: str) -> TEOBAdaptiveContinuationResult:
        fixed_times = start_time_t + torch.stack(fixed_time_values)
        fixed_states = torch.stack(fixed_state_values)
        fixed_freqs = torch.stack(fixed_frequency_values)
        event_index = (prefix_time.numel() - 1 + adaptive_time.numel() - 1
                       if prefix else adaptive_time.numel() - 1)
        stats = TEOBContinuationStatistics(
            adaptive_accepted_steps=adaptive_accepted,
            adaptive_rejected_steps=adaptive_rejected,
            adaptive_evaluations=adaptive_evaluations,
            fixed_accepted_steps=fixed_accepted,
            fixed_rejected_steps=fixed_rejected,
            fixed_evaluations=fixed_evaluations,
            termination=reason,
        )
        return _make_result(
            prefix=prefix,
            prefix_time=prefix_time,
            prefix_state=prefix_state,
            prefix_frequency=prefix_frequency,
            adaptive_time=adaptive_time,
            adaptive_state=adaptive_state,
            adaptive_frequency=adaptive_frequency,
            fixed_time=fixed_times,
            fixed_state=fixed_states,
            fixed_frequency=fixed_freqs,
            peak_event_index=event_index,
            peak_event_time=event_time,
            peak_event_frequency=event_frequency,
            phase_continuity=(adaptive_state[0, 1] - prefix_state[-1, 1]
                              if prefix else torch.zeros((), dtype=dtype, device=device)),
            first_ode_step=initial_step_t,
            adaptive_next_step=adaptive_next_step,
            fixed_step=fixed_step,
            target_time=target_time,
            status=status,
            skip_reason=reason if status != "complete" else None,
            statistics=stats,
            initial_state=initial_state,
            start_time=start_time_t,
        )

    def fail_fixed(reason: str, message: str, *, time: Tensor, state: Tensor,
                   cause: Optional[BaseException] = None):
        result = partial_result("failed", reason)
        error = TEOBAdaptiveContinuationError(
            reason,
            message,
            result=result,
            time=time,
            state=state,
            evaluations=adaptive_evaluations + fixed_evaluations,
            accepted_steps=adaptive_accepted + fixed_accepted,
            rejected_steps=adaptive_rejected + fixed_rejected,
        )
        if cause is not None:
            raise error from cause
        raise error

    while bool(current_local_time < target_local_time):
        total_accepted = adaptive_accepted + fixed_accepted
        total_rejected = adaptive_rejected + fixed_rejected
        if total_accepted >= max_accepted_steps:
            if raise_on_failure:
                fail_fixed(
                    "accepted_step_budget", "accepted-step budget exhausted during fixed continuation",
                    time=start_time_t + current_local_time, state=current_state,
                )
            return partial_result("failed", "accepted_step_budget")
        if total_rejected >= max_rejected_steps:
            if raise_on_failure:
                fail_fixed(
                    "rejection_budget", "rejected-step budget exhausted during fixed continuation",
                    time=start_time_t + current_local_time, state=current_state,
                )
            return partial_result("failed", "rejection_budget")
        if fixed_evaluations + adaptive_evaluations + 5 > max_evaluations:
            if raise_on_failure:
                fail_fixed(
                    "evaluation_budget", "vector-field evaluation budget exhausted during fixed continuation",
                    time=start_time_t + current_local_time, state=current_state,
                )
            return partial_result("failed", "evaluation_budget")
        next_local_time = current_local_time + fixed_step
        if bool(next_local_time > duration_limit):
            # Fixed continuation deliberately never shortens its step.  If
            # the next mandated knot would cross the hard duration budget,
            # retain the accepted prefix and report the budget failure.
            if raise_on_failure:
                fail_fixed(
                    "duration_budget",
                    "fixed continuation step would cross the duration budget",
                    time=start_time_t + current_local_time,
                    state=current_state,
                )
            return partial_result("failed", "duration_budget")
        try:
            trial = fixed_fehlberg45_step(
                local_field,
                current_local_time,
                current_state,
                fixed_step,
                atol=atol,
                rtol=rtol,
                current_derivative=current_derivative,
            )
        except AdaptiveIntegrationError as exc:
            fixed_evaluations += int(exc.evaluations or 0)
            fixed_rejected += 1
            if raise_on_failure:
                fail_fixed(
                    exc.reason,
                    f"fixed Fehlberg trial failed: {exc}",
                    time=start_time_t + current_local_time,
                    state=current_state,
                    cause=exc,
                )
            return partial_result("failed", exc.reason)
        fixed_evaluations += trial.evaluations
        if not trial.accepted:
            fixed_rejected += 1
            if raise_on_failure:
                fail_fixed(
                    "fixed_step_error",
                    "fixed Fehlberg trial exceeded the state-scaled error tolerance",
                    time=start_time_t + current_local_time,
                    state=current_state,
                )
            return partial_result("failed", "fixed_step_error")
        next_state = trial.fifth
        if fixed_evaluations + adaptive_evaluations + 1 > max_evaluations:
            if raise_on_failure:
                fail_fixed(
                    "evaluation_budget",
                    "endpoint evaluation budget exhausted during fixed continuation",
                    time=start_time_t + current_local_time,
                    state=current_state,
                )
            return partial_result("failed", "evaluation_budget")
        try:
            next_derivative = _call_field(
                local_field, next_local_time, next_state, field_arity,
            )
        except _TrialDomainFailure as exc:
            # The endpoint probe is part of the bounded work accounting even
            # when it rejects the candidate and leaves the last accepted state
            # untouched.
            fixed_evaluations += 1
            fixed_rejected += 1
            if raise_on_failure:
                fail_fixed(
                    "domain_event",
                    f"fixed endpoint left the vector-field domain: {exc}",
                    time=start_time_t + current_local_time,
                    state=current_state,
                    cause=exc,
                )
            return partial_result("failed", "domain_event")
        fixed_evaluations += 1
        current_local_time = next_local_time
        current_state = next_state
        current_derivative = next_derivative
        fixed_time_values.append(current_local_time)
        fixed_state_values.append(current_state)
        fixed_frequency_values.append(current_derivative[1])
        fixed_accepted += 1

    result = partial_result("complete", "complete")
    return result


def continue_teob_adaptive(
    hamiltonian,
    pa_result: Optional[TEOBPostAdiabaticResult] = None,
    radiation_reaction: Optional[object] = None,
    *,
    initial_state: Optional[Tensor] = None,
    initial_step: Optional[Tensor | float] = None,
    start_time: Tensor | float = 0.0,
    include_horizon_absorption: bool = True,
    **kwargs,
) -> TEOBAdaptiveContinuationResult:
    """Run the bounded continuation on the clean TEOB vacuum vector field.

    A complete PA result supplies the transferred endpoint and forces the
    first ODE trial to half its final PA time interval.  A skipped or absent
    PA result requires both ``initial_state`` and ``initial_step`` explicitly.
    """
    if pa_result is not None and not isinstance(pa_result, TEOBPostAdiabaticResult):
        # Permit the natural ``(hamiltonian, radiation_reaction)`` positional
        # form while keeping a PA result unambiguous.
        if radiation_reaction is None and hasattr(pa_result, "radiation_reaction_forces"):
            radiation_reaction = pa_result
            pa_result = None
        else:
            raise TypeError("pa_result must be a TEOBPostAdiabaticResult")
    driver = TEOBVacuumRadiationDriver(
        hamiltonian,
        radiation_reaction,
        include_horizon_absorption=include_horizon_absorption,
    )
    if pa_result is not None and pa_result.complete:
        if pa_result.endpoint_state is None or pa_result.endpoint_time is None:
            raise ValueError("completed PA result has no endpoint transfer")
        if pa_result.time.numel() < 2:
            raise ValueError("completed PA result needs at least two knots for its final interval")
        state = pa_result.endpoint_state
        start = pa_result.endpoint_time
        first_step = 0.5 * (pa_result.time[-1] - pa_result.time[-2])
        return integrate_adaptive_continuation(
            driver.vector_field,
            state,
            first_step,
            start_time=start,
            prefix_time=pa_result.time,
            prefix_state=pa_result.state,
            prefix_frequency=pa_result.frequency,
            **kwargs,
        )
    if initial_state is None:
        raise ValueError("a skipped or absent PA result requires explicit initial_state")
    if initial_step is None:
        raise ValueError("a skipped or absent PA result requires explicit initial_step")
    # A skipped PA grid is diagnostic initialization only.  The caller's
    # explicit state is the ODE origin, so no circular state is guessed.
    return integrate_adaptive_continuation(
        driver.vector_field,
        initial_state,
        initial_step,
        start_time=start_time,
        **kwargs,
    )


# Descriptive aliases keep the isolated component discoverable to diagnostics
# and downstream reviewed assembly work without creating alternate code paths.
integrate_teob_adaptive_continuation = continue_teob_adaptive
teob_adaptive_continuation = continue_teob_adaptive
continue_teob_postadiabatic = continue_teob_adaptive


__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "DEFAULT_FIXED_STEP_CAP",
    "DEFAULT_CONTINUATION_DURATION",
    "DEFAULT_RADIUS_FLOOR",
    "DEFAULT_MAX_REJECTED_STEPS",
    "DEFAULT_MAX_EVALUATIONS",
    "DEFAULT_MAX_ACCEPTED_STEPS",
    "DEFAULT_MAX_DURATION",
    "TEOBAdaptiveContinuationError",
    "TEOBContinuationStatistics",
    "TEOBAdaptiveContinuationResult",
    "AcceptedFrequencyDecreaseEvent",
    "integrate_adaptive_continuation",
    "continue_teob_adaptive",
    "integrate_teob_adaptive_continuation",
    "teob_adaptive_continuation",
    "continue_teob_postadiabatic",
]

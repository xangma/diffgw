"""Independent bounded Fehlberg 4/5 drivers for the clean EOB route.

The adaptive state history in this module is deliberately separate from the
uniform waveform grid.  Accepted states are advanced with the embedded
Fehlberg pair from NASA TR R-315; a natural cubic interpolant is then used to
sample the requested output spacing.  The event tests operate on accepted
states, never on the interpolated samples.

The generic :func:`integrate_adaptive` routine accepts a state-only callable
(``field(state)``) or a time-dependent callable (``field(time, state)``).  It
supports leading batch axes, preserves the Torch graph through all accepted
arithmetic, and makes the discrete controller and event decisions explicit.
The EOB-specific two-stage helper is opt-in and is intentionally kept out of
the default fixed-step published route.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor

from .integrator import EOBTrajectoryIntegrator
from .numerical_attachment import (
    NumericalAttachedMode22,
    attach_seobnrv4_22_numerical_compatibility,
    natural_cubic_spline,
)
from .nqc import nqc_22_factor


# Fehlberg's published tableau.  Keeping the coefficients in ordinary
# Python floats makes the arithmetic follow the state's dtype/device through
# multiplication by the stage tensors.
_RKF45_C = (0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0)
_RKF45_B = (
    (),
    (1.0 / 4.0,),
    (3.0 / 32.0, 9.0 / 32.0),
    (1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0),
    (439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0),
    (-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0),
)
_RKF45_B4 = (25.0 / 216.0, 0.0, 1408.0 / 2565.0,
             2197.0 / 4104.0, -1.0 / 5.0, 0.0)
_RKF45_B5 = (16.0 / 135.0, 0.0, 6656.0 / 12825.0,
             28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0)
# Subtract the published rational weights before converting to binary
# floating point.  Computing ``fifth - fourth`` after both states have been
# formed can erase the embedded error when a component of ``state`` is large
# (the orbital phase is the relevant clean EOB example).
_RKF45_B5_MINUS_B4 = (
    1.0 / 360.0,
    0.0,
    -128.0 / 4275.0,
    -2197.0 / 75240.0,
    1.0 / 50.0,
    2.0 / 55.0,
)


# Public names for the two controller policies.  ``generic`` retains the
# original driver behavior for existing callers.  ``gsl_state_scaled`` is an
# explicitly named compatibility policy whose equations are taken from the
# public adaptive-step manual and the public RKF45 controller probes recorded
# in the review contract.  The implementation does not depend on GSL.
GENERIC_CONTROLLER = "generic"
GSL_STATE_SCALED_CONTROLLER = "gsl_state_scaled"
STATE_SCALED_CONTROLLER = GSL_STATE_SCALED_CONTROLLER


def _normalise_controller(controller: str) -> str:
    """Return a canonical adaptive controller name."""
    if not isinstance(controller, str):
        raise TypeError("controller must be a string")
    key = controller.strip().lower().replace("-", "_")
    aliases = {
        "generic": GENERIC_CONTROLLER,
        "legacy": GENERIC_CONTROLLER,
        "default": GENERIC_CONTROLLER,
        "state_scaled": GSL_STATE_SCALED_CONTROLLER,
        "gsl_state_scaled": GSL_STATE_SCALED_CONTROLLER,
        "public_state_scaled": GSL_STATE_SCALED_CONTROLLER,
        "gsl": GSL_STATE_SCALED_CONTROLLER,
    }
    try:
        return aliases[key]
    except KeyError as exc:
        choices = ", ".join(sorted({GENERIC_CONTROLLER,
                                      GSL_STATE_SCALED_CONTROLLER}))
        raise ValueError(f"unknown controller {controller!r}; choose {choices}") from exc


def _normalise_batch_mode(batch_mode: str) -> str:
    """Normalize the generic RHS batching contract.

    ``full`` evaluates a batched RHS on every row and selects active rows
    afterwards.  This permits closures that capture row-shaped parameters,
    even after event histories make the active set ragged.  ``active`` keeps
    the historical subset call and is available for a RHS that explicitly
    indexes or otherwise understands the active-row subset.
    """
    if not isinstance(batch_mode, str):
        raise TypeError("batch_mode must be a string")
    key = batch_mode.strip().lower().replace("-", "_")
    aliases = {
        "full": "full",
        "full_batch": "full",
        "full_shape": "full",
        "all_rows": "full",
        "active": "active",
        "active_rows": "active",
        "subset": "active",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown batch_mode {batch_mode!r}; choose 'full' or 'active'"
        ) from exc


def state_scaled_controller_adjustment(
    error_ratio: Tensor | float,
    *,
    order: int = 5,
    safety: float = 0.9,
    shrink_threshold: float = 1.1,
    growth_threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Return public-controller action and step factor for ``error_ratio``.

    The returned action is ``-1`` for rejection, ``0`` for an accepted trial
    whose size is unchanged, and ``1`` for an accepted growth trial.  The
    candidate state supplies the denominator used by the integrator; this
    helper only applies the controller to the resulting maximum ratio.  The
    strict thresholds deliberately keep ``0.5`` and ``1.1`` in the unchanged
    branch.  A zero error ratio receives the bounded factor five.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order <= 0:
        raise ValueError("order must be a positive integer")
    if (not math.isfinite(float(safety)) or float(safety) <= 0
            or not math.isfinite(float(shrink_threshold))
            or not math.isfinite(float(growth_threshold))
            or not (0 < float(growth_threshold) < float(shrink_threshold))):
        raise ValueError("controller thresholds and safety must be finite and ordered")
    ratio = torch.as_tensor(error_ratio)
    if not (ratio.dtype.is_floating_point or ratio.dtype.is_complex):
        ratio = ratio.to(torch.get_default_dtype())
    if ratio.is_complex():
        ratio = ratio.abs()
    if not bool(torch.isfinite(ratio).all()) or bool((ratio < 0).any()):
        raise ValueError("error_ratio must be finite and nonnegative")
    real_dtype = ratio.dtype
    # The public rule is q=5 for RKF45.  Keep the helper general for focused
    # controller tests while retaining the documented default.
    safe = ratio.clamp_min(torch.finfo(real_dtype).tiny)
    shrink = ratio > float(shrink_threshold)
    grow = ratio < float(growth_threshold)
    shrink_factor = torch.maximum(
        torch.full_like(ratio, 1.0 / 5.0),
        float(safety) * safe.pow(-1.0 / float(order)),
    )
    growth_factor = torch.minimum(
        torch.full_like(ratio, 5.0),
        torch.maximum(
            torch.ones_like(ratio),
            float(safety) * safe.pow(-1.0 / float(order + 1)),
        ),
    )
    factor = torch.where(shrink, shrink_factor,
                        torch.where(grow, growth_factor, torch.ones_like(ratio)))
    action = torch.where(shrink, torch.full_like(ratio, -1, dtype=torch.int8),
                         torch.where(grow, torch.ones_like(ratio, dtype=torch.int8),
                                     torch.zeros_like(ratio, dtype=torch.int8)))
    return action, factor


# Short aliases make the explicitly named mathematical policy easy to find
# from focused tests and downstream diagnostics.
gsl_state_scaled_controller = state_scaled_controller_adjustment
state_scaled_controller = state_scaled_controller_adjustment


class AdaptiveIntegrationError(RuntimeError):
    """A bounded integration failure with a machine-readable reason."""

    def __init__(self, reason: str, message: str, *, time=None, state=None,
                 row_indices=None, evaluations=None, rejections=None,
                 accepted_steps=None, accepted_time=None,
                 accepted_state=None, accepted_derivative=None):
        super().__init__(message)
        self.reason = str(reason)
        self.time = time
        self.state = state
        self.row_indices = row_indices
        self.evaluations = evaluations
        self.rejections = rejections
        # These additive fields are populated by ``integrate_adaptive`` when
        # a failure occurs after accepted knots have been recorded.  Keeping
        # them on the shared error lets bounded callers retain the exact
        # prefix without retrying a derivative evaluation or inventing a
        # state/frequency sample.
        self.accepted_steps = accepted_steps
        self.accepted_time = accepted_time
        self.accepted_state = accepted_state
        self.accepted_derivative = accepted_derivative

    @property
    def partial_time(self):
        """Alias for the accepted-time prefix carried by a failed run."""
        return self.accepted_time

    @property
    def partial_state(self):
        """Alias for the accepted-state prefix carried by a failed run."""
        return self.accepted_state

    @property
    def partial_derivative(self):
        """Alias for the accepted-derivative prefix carried by a failed run."""
        return self.accepted_derivative


@dataclass(frozen=True)
class AdaptiveStatistics:
    """Discrete work counters and termination labels for each batch row."""

    accepted_steps: Tensor
    rejected_steps: Tensor
    evaluations: Tensor
    termination: tuple[str, ...]
    event_time: Tensor

    @property
    def total_accepted(self) -> int:
        return int(self.accepted_steps.sum().detach())

    @property
    def total_rejected(self) -> int:
        return int(self.rejected_steps.sum().detach())

    @property
    def total_evaluations(self) -> int:
        return int(self.evaluations.sum().detach())


@dataclass(frozen=True)
class AdaptiveTrajectory:
    """Accepted and uniformly resampled states from one adaptive run.

    For an unbatched initial state, ``time``/``state`` have shapes ``(N,)``
    and ``(N,D)``.  For leading batch shape ``B``, they have shapes
    ``(B,N)`` and ``(B,N,D)``.  Rows with different event times are padded by
    their final accepted state and identified by ``valid``.  ``output_valid``
    gives the corresponding prefix mask on the natural-cubic output grid.
    """

    time: Tensor
    state: Tensor
    derivative: Tensor
    valid: Tensor
    output_time: Tensor
    output_state: Tensor
    output_valid: Tensor
    statistics: AdaptiveStatistics
    output_step: Tensor
    initial_step: Tensor
    # The controller's proposal for the next trial after the final accepted
    # state.  Existing callers do not need this field; it is additive
    # metadata used by the bounded TEOB PA-to-ODE continuation route.
    next_step: Optional[Tensor] = None

    @property
    def accepted_time(self) -> Tensor:
        return self.time

    @property
    def accepted_state(self) -> Tensor:
        return self.state

    @property
    def accepted_derivative(self) -> Tensor:
        return self.derivative

    @property
    def uniform_time(self) -> Tensor:
        return self.output_time

    @property
    def uniform_state(self) -> Tensor:
        return self.output_state

    @property
    def termination(self) -> tuple[str, ...]:
        return self.statistics.termination

    @property
    def proposed_next_step(self) -> Optional[Tensor]:
        """Return the controller proposal following the last accepted knot."""
        return self.next_step


@dataclass(frozen=True)
class _RunConfig:
    atol: float
    rtol: float
    controller: str
    batch_mode: str
    min_step: float
    max_step: float
    safety: float
    min_factor: float
    max_factor: float
    max_rejected_steps: int
    max_evaluations: int
    max_accepted_steps: int
    max_duration: float


class _TrialDomainFailure(Exception):
    """Internal marker: a trial stage left the vector-field domain."""


def _as_step(value, *, dtype: torch.dtype, device: torch.device, name: str,
             allow_zero: bool = False) -> Tensor:
    result = torch.as_tensor(value, dtype=dtype, device=device)
    lower_ok = result >= 0 if allow_zero else result > 0
    qualifier = "nonnegative" if allow_zero else "positive"
    if result.ndim != 0 or not bool(torch.isfinite(result)) or not bool(lower_ok):
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _check_initial_state(initial_state: Tensor) -> tuple[Tensor, tuple[int, ...], int]:
    if not isinstance(initial_state, Tensor):
        raise TypeError("initial_state must be a torch.Tensor")
    if initial_state.ndim < 1 or initial_state.shape[-1] < 1:
        raise ValueError("initial_state must have a nonempty component axis")
    if initial_state.dtype not in (torch.float32, torch.float64):
        raise TypeError("initial_state must use float32 or float64")
    if not bool(torch.isfinite(initial_state).all()):
        raise ValueError("initial_state must be finite")
    batch_shape = tuple(initial_state.shape[:-1])
    components = initial_state.shape[-1]
    rows = math.prod(batch_shape) if batch_shape else 1
    return initial_state.reshape(rows, components), batch_shape, rows


def _field_arity(field: Callable) -> int:
    """Infer whether a field wants ``state`` or ``(time,state)``."""
    explicit = getattr(field, "time_dependent", None)
    if explicit is not None:
        return 2 if bool(explicit) else 1
    try:
        signature = inspect.signature(field)
        positional = [parameter for parameter in signature.parameters.values()
                      if parameter.kind in (parameter.POSITIONAL_ONLY,
                                            parameter.POSITIONAL_OR_KEYWORD)]
        if any(parameter.kind == parameter.VAR_POSITIONAL
               for parameter in signature.parameters.values()):
            return 2
        return 2 if len(positional) >= 2 else 1
    except (TypeError, ValueError):
        return 1


def _call_field(field: Callable, time: Tensor, state: Tensor, arity: int) -> Tensor:
    try:
        result = field(time, state) if arity == 2 else field(state)
    except (FloatingPointError, ArithmeticError, RuntimeError, ValueError) as exc:
        # Domain failures raised by clean EOB helpers are trial-local.  Shape
        # and type failures are re-raised below after the common validation.
        raise _TrialDomainFailure(str(exc)) from exc
    if not isinstance(result, Tensor) or result.shape != state.shape:
        raise ValueError("vector_field must return a tensor with state's shape")
    if result.dtype != state.dtype or result.device != state.device:
        raise TypeError("vector_field must preserve state dtype and device")
    if not bool(torch.isfinite(result).all()):
        raise _TrialDomainFailure("vector_field returned a nonfinite derivative")
    return result


def _call_field_for_rows(
    field: Callable,
    time: Tensor,
    state: Tensor,
    arity: int,
    *,
    batch_mode: str,
    full_time: Optional[Tensor] = None,
    full_state: Optional[Tensor] = None,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Evaluate a batched RHS under the selected row-shape contract.

    The adaptive loop advances only active rows, but a full-shape RHS can
    capture parameters with the original batch shape.  In that mode construct
    a temporary full state/time array, evaluate the RHS once, and select the
    active derivatives.  ``active`` is an explicit escape hatch for callables
    that already understand indexed subsets.  It is intentionally never used
    by the EOB compatibility route, whose scalar row closures avoid this
    ambiguity altogether.
    """
    if batch_mode == "active":
        return _call_field(field, time, state, arity)
    if batch_mode != "full":  # pragma: no cover - normalized at the API edge
        raise ValueError(f"unknown batch_mode {batch_mode!r}")
    if full_time is None or full_state is None or indices is None:
        raise ValueError("full batch evaluation needs full_time, full_state, and indices")
    trial_time = full_time.index_copy(0, indices, time)
    trial_state = full_state.index_copy(0, indices, state)
    result = _call_field(field, trial_time, trial_state, arity)
    return result.index_select(0, indices)


def _rkf45_weighted_sum(
    stages: Sequence[Tensor], coefficients: Sequence[float]
) -> Tensor:
    """Accumulate a Fehlberg tableau row without a large-state subtraction.

    The weights are converted to the stage dtype only after the rational
    tableau row has been formed in the module constants.  A leading stack
    axis keeps the operation valid for scalar and batched state vectors while
    preserving gradients through every stage tensor.
    """
    if not stages or len(stages) != len(coefficients):
        raise ValueError("RKF45 weighted sum needs one coefficient per stage")
    first = stages[0]
    weights = torch.as_tensor(coefficients, dtype=first.dtype, device=first.device)
    weight_shape = (len(coefficients),) + (1,) * first.ndim
    return (torch.stack(tuple(stages), dim=0)
            * weights.reshape(weight_shape)).sum(dim=0)


def _rkf45_trial(field: Callable, time: Tensor, state: Tensor, step: Tensor,
                 current_field: Tensor, arity: int, *, batch_mode: str = "active",
                 full_time: Optional[Tensor] = None,
                 full_state: Optional[Tensor] = None,
                 indices: Optional[Tensor] = None) -> tuple[Tensor, Tensor, Tensor, int]:
    """One Fehlberg trial; ``current_field`` is the already known k1."""
    stages = [current_field]
    for stage in range(1, 6):
        combination = _rkf45_weighted_sum(stages, _RKF45_B[stage])
        stage_state = state + step.unsqueeze(-1) * combination
        stage_time = time + _RKF45_C[stage] * step
        try:
            stages.append(_call_field_for_rows(
                field, stage_time, stage_state, arity, batch_mode=batch_mode,
                full_time=full_time, full_state=full_state, indices=indices,
            ))
        except _TrialDomainFailure:
            raise
    # Form both the candidate and the local error from the stage derivatives
    # directly.  In particular, do not subtract two state vectors that share
    # a large accumulated phase component.
    fourth_increment = _rkf45_weighted_sum(stages, _RKF45_B4)
    fifth_increment = _rkf45_weighted_sum(stages, _RKF45_B5)
    error_increment = _rkf45_weighted_sum(stages, _RKF45_B5_MINUS_B4)
    scaled_step = step.unsqueeze(-1)
    fourth = state + scaled_step * fourth_increment
    fifth = state + scaled_step * fifth_increment
    error = scaled_step * error_increment
    if not bool(torch.isfinite(fifth).all() & torch.isfinite(error).all()):
        raise _TrialDomainFailure("Fehlberg trial produced a nonfinite state")
    return fourth, fifth, error, 5


@dataclass(frozen=True)
class Fehlberg45FixedTrial:
    """One scalar fixed-step Fehlberg 4/5 trial.

    The candidate and embedded error remain connected to the Torch graph.
    ``accepted`` is a Python bool because it records the piecewise numerical
    decision; callers must keep the supplied step unchanged when it is true.
    ``evaluations`` counts only vector-field calls made by this primitive.  A
    caller that needs the derivative at the accepted endpoint evaluates it
    separately, just as :func:`integrate_adaptive` does.
    """

    fourth: Tensor
    fifth: Tensor
    error: Tensor
    error_ratio: Tensor
    accepted: bool
    current_derivative: Tensor
    evaluations: int


def fixed_fehlberg45_step(
    vector_field: Callable,
    time: Tensor | float,
    state: Tensor,
    step: Tensor | float,
    *,
    atol: float = 1.0e-13,
    rtol: float = 1.0e-11,
    current_derivative: Optional[Tensor] = None,
) -> Fehlberg45FixedTrial:
    """Evaluate one fixed RKF45 step with the state-scaled local test.

    This is the reusable fixed continuation primitive.  It intentionally
    accepts one state vector and never retries or changes ``step``.  Domain
    failures are raised as :class:`AdaptiveIntegrationError` so the caller
    can report the accepted prefix and its work counters explicitly.
    """
    flat_state, batch_shape, _ = _check_initial_state(state)
    if batch_shape:
        raise ValueError("fixed_fehlberg45_step accepts one scalar state vector")
    # Keep the caller's final component axis and its graph edges while using a
    # scalar shape internally for the shared Fehlberg implementation.
    state_vector = flat_state[0]
    dtype, device = state_vector.dtype, state_vector.device
    time_t = torch.as_tensor(time, dtype=dtype, device=device)
    if time_t.ndim != 0 or not bool(torch.isfinite(time_t)):
        raise ValueError("time must be a finite scalar")
    step_t = _as_step(step, dtype=dtype, device=device, name="step")
    if isinstance(atol, bool) or not math.isfinite(float(atol)) or float(atol) <= 0:
        raise ValueError("atol must be finite and positive")
    if isinstance(rtol, bool) or not math.isfinite(float(rtol)) or float(rtol) < 0:
        raise ValueError("rtol must be finite and nonnegative")
    arity = _field_arity(vector_field)
    evaluations = 0
    if current_derivative is None:
        try:
            current = _call_field(vector_field, time_t, state_vector, arity)
        except _TrialDomainFailure as exc:
            raise AdaptiveIntegrationError(
                "domain_event", str(exc), time=time_t, state=state_vector,
                evaluations=1, rejections=0,
            ) from exc
        evaluations = 1
    else:
        if (not isinstance(current_derivative, Tensor)
                or current_derivative.shape != state_vector.shape
                or current_derivative.dtype != dtype
                or current_derivative.device != device):
            raise ValueError("current_derivative must match state shape, dtype, and device")
        if not bool(torch.isfinite(current_derivative).all()):
            raise ValueError("current_derivative must be finite")
        current = current_derivative
    try:
        fourth, fifth, error, used = _rkf45_trial(
            vector_field, time_t, state_vector, step_t, current, arity,
            batch_mode="active",
        )
    except _TrialDomainFailure as exc:
        raise AdaptiveIntegrationError(
            "domain_event", str(exc), time=time_t, state=state_vector,
            evaluations=evaluations + 5, rejections=0,
        ) from exc
    evaluations += used
    scale = float(atol) + float(rtol) * fifth.abs()
    error_ratio = (error.abs() / scale).amax()
    if not bool(torch.isfinite(error_ratio)):
        raise AdaptiveIntegrationError(
            "nonfinite_error", "Fehlberg error estimate became nonfinite",
            time=time_t, state=state_vector, evaluations=evaluations,
        )
    return Fehlberg45FixedTrial(
        fourth=fourth,
        fifth=fifth,
        error=error,
        error_ratio=error_ratio,
        accepted=bool(error_ratio <= 1.0),
        current_derivative=current,
        evaluations=evaluations,
    )


# Descriptive aliases keep the primitive easy to discover without adding a
# second implementation or a second tableau.
fehlberg45_fixed_step = fixed_fehlberg45_step
rkf45_fixed_step = fixed_fehlberg45_step


def _controller_decision(
    normalized: Tensor,
    old_state: Tensor,
    candidate_state: Tensor,
    config: _RunConfig,
    *,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Return the acceptance mask and bounded next-step factor."""
    if config.controller == GSL_STATE_SCALED_CONTROLLER:
        # The compatibility policy's ratio has already been computed with the
        # candidate-state denominator.  Its strict thresholds and q=5
        # exponents are centralized in the named public helper above.
        action, factor = state_scaled_controller_adjustment(
            normalized, order=5, safety=config.safety,
        )
        return action >= 0, factor

    # Preserve the original generic controller exactly for existing callers.
    accepted = normalized <= 1.0
    safe_error = normalized.clamp_min(torch.finfo(dtype).tiny)
    factor = config.safety * safe_error.pow(-0.2)
    factor = torch.where(
        accepted,
        factor.clamp(config.min_factor, config.max_factor),
        factor.clamp(0.1, 1.0),
    )
    return accepted, factor


def _event_call(event, time, state, derivative, previous, indices):
    if event is None:
        return torch.zeros(time.shape, dtype=torch.bool, device=time.device), None
    checker = getattr(event, "check", None)
    if checker is None:
        try:
            answer = event(time, state, derivative, previous)
        except TypeError:
            answer = event(time, state, derivative)
    else:
        answer = checker(time, state, derivative, previous, indices)
    reason = None
    if isinstance(answer, tuple):
        answer, reason = answer
    if not isinstance(answer, Tensor):
        answer = torch.as_tensor(answer, dtype=torch.bool, device=time.device)
    if answer.dtype != torch.bool:
        answer = answer.to(torch.bool)
    if answer.shape != time.shape:
        answer = torch.broadcast_to(answer, time.shape)
    return answer, reason


def _build_resampled_grid(times: Sequence[Tensor], values: Sequence[Tensor],
                          output_step: Tensor, batch_shape: tuple[int, ...],
                          *, derivatives: Optional[Sequence[Tensor]] = None):
    """Natural-cubic resample independently for each flattened batch row."""
    rows = len(times)
    dtype, device = output_step.dtype, output_step.device
    counts = [int(torch.floor(t[-1].detach() / output_step).item()) + 1 for t in times]
    max_count = max(counts)
    components = values[0].shape[-1]
    sampled = []
    sampled_derivative = [] if derivatives is not None else None
    for row in range(rows):
        count = counts[row]
        query = torch.arange(count, dtype=dtype, device=device) * output_step
        if times[row].numel() == 1:
            # A zero-duration diagnostic has one accepted knot and therefore
            # no cubic interval.  Its only valid output is the initial state.
            sampled.append(values[row][0:1])
            if derivatives is not None:
                sampled_derivative.append(derivatives[row][0:1])
        else:
            # The state always has an explicit component axis, including the
            # one-component analytic-driver case; disambiguate that shape
            # from a scalar series accepted by the shared spline utility.
            spline = natural_cubic_spline(times[row], values[row], scalar_output=False)
            sampled.append(spline.evaluate(query))
            if derivatives is not None:
                derivative_spline = natural_cubic_spline(
                    times[row], derivatives[row], scalar_output=False)
                sampled_derivative.append(derivative_spline.evaluate(query))
    output_state = torch.zeros((rows, max_count, components), dtype=dtype, device=device)
    output_time = torch.zeros((rows, max_count), dtype=dtype, device=device)
    output_valid = torch.zeros((rows, max_count), dtype=torch.bool, device=device)
    output_derivative = (torch.zeros((rows, max_count, components), dtype=dtype, device=device)
                         if sampled_derivative is not None else None)
    for row, count in enumerate(counts):
        output_state[row, :count] = sampled[row]
        output_time[row, :count] = torch.arange(count, dtype=dtype, device=device) * output_step
        output_valid[row, :count] = True
        if output_derivative is not None:
            output_derivative[row, :count] = sampled_derivative[row]
        if count < max_count:
            output_state[row, count:] = sampled[row][-1]
            output_time[row, count:] = output_time[row, count - 1]
            if output_derivative is not None:
                output_derivative[row, count:] = sampled_derivative[row][-1]
    shape = batch_shape + (max_count,)
    output_state = output_state.reshape(batch_shape + (max_count, components))
    output_time = output_time.reshape(shape)
    output_valid = output_valid.reshape(shape)
    if output_derivative is not None:
        output_derivative = output_derivative.reshape(batch_shape + (max_count, components))
    return output_time, output_state, output_valid, output_derivative


def resample_natural_cubic(times: Tensor, values: Tensor, output_step: float | Tensor,
                           *, output_end: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    """Resample a trajectory on ``0, output_step, ...`` with natural splines.

    ``times`` is ``(N,)`` or ``(...,N)`` and ``values`` is ``(...,N,D)``.
    The result is ``(query, samples)``; batched rows are padded to the longest
    row and the returned query is accompanied by a finite trailing value.  A
    caller that needs row lengths can use ``floor(times[...,-1]/dt)+1``.
    """
    if not isinstance(times, Tensor) or not isinstance(values, Tensor):
        raise TypeError("times and values must be tensors")
    if times.dtype not in (torch.float32, torch.float64) or values.dtype != times.dtype:
        raise TypeError("times and values must share float32/float64 dtype")
    if times.ndim < 1 or values.ndim != times.ndim + 1 or values.shape[-2] != times.shape[-1]:
        raise ValueError("values must have shape (..., samples, components)")
    if not bool(torch.isfinite(times).all() & torch.isfinite(values).all()):
        raise ValueError("times and values must be finite")
    if bool((times.diff(dim=-1) <= 0).any()):
        raise ValueError("times must be strictly increasing")
    step = _as_step(output_step, dtype=times.dtype, device=times.device, name="output_step")
    if output_end is not None:
        end = torch.as_tensor(output_end, dtype=times.dtype, device=times.device)
        if end.ndim != 0 or not bool(torch.isfinite(end)) or not bool(end >= 0):
            raise ValueError("output_end must be finite and nonnegative")
    else:
        end = None
    batch_shape = tuple(times.shape[:-1])
    rows = math.prod(batch_shape) if batch_shape else 1
    flat_times = times.reshape(rows, times.shape[-1])
    flat_values = values.reshape(rows, values.shape[-2], values.shape[-1])
    if end is not None:
        flat_end = int(torch.floor(end.detach() / step).item())
        flat_counts = [min(flat_end + 1,
                           int(torch.floor(row[-1].detach() / step).item()) + 1)
                       for row in flat_times]
        if any(count < 1 for count in flat_counts):
            raise ValueError("output_end precedes the output origin")
        # Interpolate on the complete accepted trajectory, then retain only
        # requested knots no later than the explicit endpoint.  A row whose
        # accepted trajectory ends before output_end is naturally truncated
        # to its available prefix.
        query, samples, valid, _ = _build_resampled_grid(
            tuple(flat_times), tuple(flat_values), step, (rows,))
        max_count = max(flat_counts)
        query, samples, valid = query[:, :max_count], samples[:, :max_count], valid[:, :max_count]
        query = query.masked_fill(~valid, 0)
        return query.reshape(batch_shape + (max_count,)), samples.reshape(
            batch_shape + (max_count, values.shape[-1]))
    query, samples, _, _ = _build_resampled_grid(
        tuple(flat_times), tuple(flat_values), step, (rows,))
    return query.reshape(batch_shape + (query.shape[-1],)), samples.reshape(
        batch_shape + (samples.shape[-2], values.shape[-1]))


def _integrate_adaptive_impl(
    vector_field: Callable,
    initial_state: Tensor,
    output_step: float | Tensor,
    *,
    t_end: Optional[float | Tensor] = None,
    event: Optional[Callable] = None,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-9,
    controller: str = GENERIC_CONTROLLER,
    batch_mode: str = "full",
    initial_step: Optional[float | Tensor] = None,
    min_step: Optional[float] = None,
    max_step: Optional[float] = None,
    max_rejected_steps: int = 10000,
    max_evaluations: int = 200000,
    max_accepted_steps: int = 100000,
    max_duration: Optional[float] = None,
    _partial: Optional[dict] = None,
) -> AdaptiveTrajectory:
    """Integrate a bounded ODE with Fehlberg's embedded fourth/fifth pair.

    The first trial is exactly ``initial_step`` when supplied and otherwise
    ``output_step``.  The default ``generic`` controller retains the original
    maximum-old/candidate-state scaling and step policy.  The explicitly named
    ``gsl_state_scaled`` controller uses the candidate state in
    ``atol + rtol*abs(y5)`` and the documented q=5 threshold policy.  These
    policy choices affect work and interpolation nodes, not the ODE equations.

    The RHS batching contract defaults to ``batch_mode="full"``: every trial
    is evaluated on the original flattened ``(rows, components)`` shape, with
    active-row results selected afterwards.  This supports closures capturing
    distinct row parameters after event histories diverge.  ``batch_mode="active"``
    is an explicit escape hatch for a callable that indexes its own parameters
    to match the active subset.  The EOB compatibility route uses scalar row
    closures instead of either subset contract.

    ``t_end`` and ``event`` are alternative stopping mechanisms and either
    one must be supplied.  Domain failures, exhausted budgets, and unresolved
    events raise :class:`AdaptiveIntegrationError` with an explicit reason.
    """
    flat_initial, batch_shape, rows = _check_initial_state(initial_state)
    dtype, device = initial_state.dtype, initial_state.device
    controller_name = _normalise_controller(controller)
    batch_mode_name = _normalise_batch_mode(batch_mode)
    out_step = _as_step(output_step, dtype=dtype, device=device, name="output_step")
    first_step = (_as_step(initial_step, dtype=dtype, device=device, name="initial_step")
                  if initial_step is not None else out_step)
    if isinstance(atol, bool) or not math.isfinite(float(atol)) or float(atol) <= 0:
        raise ValueError("atol must be finite and positive")
    if isinstance(rtol, bool) or not math.isfinite(float(rtol)) or float(rtol) < 0:
        raise ValueError("rtol must be finite and nonnegative")
    if event is None and t_end is None:
        raise ValueError("supply t_end or an accepted-state event")
    end = (None if t_end is None else
           _as_step(t_end, dtype=dtype, device=device, name="t_end", allow_zero=True))
    field_arity = _field_arity(vector_field)
    duration_limit = (float(max_duration) if max_duration is not None else
                      (max(float(end.detach()), 1.0) if end is not None else 1.0e6))
    if not math.isfinite(duration_limit) or duration_limit <= 0:
        raise ValueError("max_duration must be finite and positive")
    minimum = (float(min_step) if min_step is not None else float(torch.finfo(dtype).eps) ** 0.5 * float(out_step))
    maximum = (float(max_step) if max_step is not None else float("inf"))
    if (not math.isfinite(minimum) or minimum <= 0 or maximum <= 0
            or math.isnan(maximum) or maximum < minimum):
        raise ValueError("min_step/max_step must satisfy 0 < min_step <= max_step")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (max_rejected_steps, max_evaluations, max_accepted_steps)):
        raise ValueError("step, evaluation, and rejection budgets must be positive integers")
    config = _RunConfig(float(atol), float(rtol), controller_name,
                        batch_mode_name, minimum, maximum,
                        0.9, 0.2, 5.0, max_rejected_steps,
                        max_evaluations, max_accepted_steps, duration_limit)

    state = flat_initial
    time = torch.zeros((rows,), dtype=dtype, device=device)
    try:
        current = _call_field(vector_field, time, state, field_arity)
    except _TrialDomainFailure as exc:
        raise AdaptiveIntegrationError(
            "domain_event", str(exc), time=time, state=state,
            row_indices=torch.arange(rows, dtype=torch.long, device=device),
            evaluations=torch.ones((rows,), dtype=torch.long, device=device),
            rejections=torch.zeros((rows,), dtype=torch.long, device=device),
        ) from exc
    if hasattr(event, "initialize"):
        event.initialize(rows, dtype=dtype, device=device)
    active = torch.ones((rows,), dtype=torch.bool, device=device)
    accepted_steps = torch.zeros((rows,), dtype=torch.long, device=device)
    rejected_steps = torch.zeros((rows,), dtype=torch.long, device=device)
    evaluations = torch.ones((rows,), dtype=torch.long, device=device)
    termination = ["running"] * rows
    event_time = torch.full((rows,), torch.nan, dtype=dtype, device=device)
    step = (torch.ones((rows,), dtype=dtype, device=device) * first_step).clamp_max(maximum)

    accepted_times: list[list[Tensor]] = [[time[row]] for row in range(rows)]
    accepted_states: list[list[Tensor]] = [[state[row]] for row in range(rows)]
    accepted_derivatives: list[list[Tensor]] = [[current[row]] for row in range(rows)]
    if _partial is not None:
        # Store the live lists rather than a copy.  Every accepted knot added
        # below is therefore visible to the public wrapper if a later trial
        # exhausts a budget or leaves the field domain.
        _partial.update(
            accepted_times=accepted_times,
            accepted_states=accepted_states,
            accepted_derivatives=accepted_derivatives,
            accepted_steps=accepted_steps,
        )

    # An event may resolve a row at the initial accepted state.  This is kept
    # separate from trial logic so event timing always refers to an accepted
    # state and its already evaluated vector field.
    if event is not None:
        indices = torch.arange(rows, dtype=torch.long, device=device)
        initial_stop, initial_reason = _event_call(event, time, state, current, None, indices)
        if bool(initial_stop.any()):
            for row in torch.nonzero(initial_stop, as_tuple=False).flatten().tolist():
                active[row] = False
                termination[row] = (str(initial_reason) if initial_reason is not None
                                    else "event")
                event_time[row] = time[row]
        if not bool(active.any()):
            # A single initial sample cannot support natural interpolation.
            raise AdaptiveIntegrationError(
                "event_at_initial_state", "accepted-state event fired at the initial state")
    if end is not None and bool(end == 0):
        for row in torch.nonzero(active, as_tuple=False).flatten().tolist():
            active[row] = False
            termination[row] = "duration"
            event_time[row] = time[row]

    while bool(active.any()):
        if bool((active & (accepted_steps >= config.max_accepted_steps)).any()):
            rows_hit = torch.nonzero(
                active & (accepted_steps >= config.max_accepted_steps), as_tuple=False
            ).flatten()
            raise AdaptiveIntegrationError("accepted_step_budget", "accepted-step budget exhausted",
                                           time=time[rows_hit], state=state[rows_hit], row_indices=rows_hit,
                                           evaluations=evaluations[rows_hit], rejections=rejected_steps[rows_hit])
        if bool(((time >= config.max_duration) & active).any()):
            rows_hit = torch.nonzero(active & (time >= config.max_duration), as_tuple=False).flatten()
            raise AdaptiveIntegrationError("duration_budget", "integration duration budget exhausted",
                                           time=time[rows_hit], state=state[rows_hit], row_indices=rows_hit,
                                           evaluations=evaluations[rows_hit], rejections=rejected_steps[rows_hit])
        indices = torch.nonzero(active, as_tuple=False).flatten()
        row_time = time[indices]
        row_state = state[indices]
        row_current = current[indices]
        row_step = step[indices]
        if end is not None:
            remaining = end - row_time
            if bool((remaining <= 0).any()):
                for row in indices[remaining <= 0].tolist():
                    active[row] = False
                    termination[row] = "duration"
                    event_time[row] = time[row]
                continue
            row_step = torch.minimum(row_step, remaining)
        row_step = row_step.clamp_min(config.min_step)
        if end is not None:
            row_step = torch.minimum(row_step, end - row_time)
        if bool((row_step <= 0).any()):
            raise AdaptiveIntegrationError("step_underflow", "adaptive step became nonpositive",
                                           time=row_time, state=row_state, row_indices=indices)
        if bool((evaluations[indices] + 5 > config.max_evaluations).any()):
            hit = indices[evaluations[indices] + 5 > config.max_evaluations]
            raise AdaptiveIntegrationError("evaluation_budget", "vector-field evaluation budget exhausted",
                                           time=time[hit], state=state[hit], row_indices=hit,
                                           evaluations=evaluations[hit], rejections=rejected_steps[hit])
        try:
            _, fifth, error, used = _rkf45_trial(
                vector_field, row_time, row_state, row_step, row_current, field_arity,
                batch_mode=config.batch_mode,
                full_time=time,
                full_state=state,
                indices=indices,
            )
        except _TrialDomainFailure as exc:
            # Invalid intermediate stages are recoverable only while the
            # bounded controller can reduce the trial.  Repeated failure is a
            # domain event, never a silently frozen row.
            rejected_steps[indices] += 1
            evaluations[indices] += 5
            if bool((rejected_steps[indices] > config.max_rejected_steps).any()):
                bad = indices[rejected_steps[indices] > config.max_rejected_steps]
                raise AdaptiveIntegrationError("domain_event", str(exc),
                                               time=time[bad], state=state[bad], row_indices=bad,
                                               evaluations=evaluations[bad], rejections=rejected_steps[bad])
            step_factor = torch.full_like(row_step, 0.25).clamp_min(0.1)
            step_new = (row_step * step_factor).clamp_min(config.min_step)
            step = step.index_copy(0, indices, step_new)
            if bool((step_new <= config.min_step).any()):
                bad = indices[step_new <= config.min_step]
                raise AdaptiveIntegrationError("domain_event", str(exc),
                                               time=time[bad], state=state[bad], row_indices=bad,
                                               evaluations=evaluations[bad], rejections=rejected_steps[bad])
            continue
        evaluations[indices] += used
        if config.controller == GSL_STATE_SCALED_CONTROLLER:
            scale = config.atol + config.rtol * fifth.abs()
        else:
            scale = config.atol + config.rtol * torch.maximum(row_state.abs(), fifth.abs())
        normalized = (error.abs() / scale).amax(dim=-1)
        if not bool(torch.isfinite(normalized).all()):
            raise AdaptiveIntegrationError("nonfinite_error", "Fehlberg error estimate became nonfinite",
                                           time=row_time, state=row_state, row_indices=indices)
        accepted, factor = _controller_decision(
            normalized, row_state, fifth, config, dtype=dtype,
        )
        if bool((~accepted).any()):
            rejected_indices = indices[~accepted]
            rejected_steps[rejected_indices] += 1
            if bool((rejected_steps[rejected_indices] > config.max_rejected_steps).any()):
                bad = rejected_indices[rejected_steps[rejected_indices] > config.max_rejected_steps]
                raise AdaptiveIntegrationError("rejection_budget", "rejected-step budget exhausted",
                                               time=time[bad], state=state[bad], row_indices=bad,
                                               evaluations=evaluations[bad], rejections=rejected_steps[bad])
        next_step = (row_step * factor).clamp(config.min_step, config.max_step)
        if bool((~accepted).all()):
            step = step.index_copy(0, indices, next_step)
            if bool((next_step <= config.min_step).all()):
                raise AdaptiveIntegrationError("step_underflow", "all trial steps reached min_step before acceptance",
                                               time=row_time, state=row_state, row_indices=indices,
                                               evaluations=evaluations[indices], rejections=rejected_steps[indices])
            continue

        accepted_indices = indices[accepted]
        accepted_next = fifth[accepted]
        accepted_time = row_time[accepted] + row_step[accepted]
        # ``current`` still denotes the derivative at the prior accepted
        # state here.  Capture it before index_copy replaces the active rows;
        # accepted-state event policies must compare adjacent accepted knots.
        previous_field = row_current[accepted]
        if bool((evaluations[accepted_indices] + 1 > config.max_evaluations).any()):
            hit = accepted_indices[evaluations[accepted_indices] + 1 > config.max_evaluations]
            raise AdaptiveIntegrationError("evaluation_budget", "endpoint evaluation budget exhausted",
                                           time=time[hit], state=state[hit], row_indices=hit,
                                           evaluations=evaluations[hit], rejections=rejected_steps[hit])
        try:
            accepted_field = _call_field_for_rows(
                vector_field, accepted_time, accepted_next, field_arity,
                batch_mode=config.batch_mode,
                full_time=time,
                full_state=state,
                indices=accepted_indices,
            )
        except _TrialDomainFailure as exc:
            bad = accepted_indices
            raise AdaptiveIntegrationError("domain_event", str(exc),
                                           time=accepted_time, state=accepted_next,
                                           row_indices=bad,
                                           evaluations=evaluations[bad], rejections=rejected_steps[bad]) from exc
        evaluations[accepted_indices] += 1
        accepted_steps[accepted_indices] += 1
        state = state.index_copy(0, accepted_indices, accepted_next)
        time = time.index_copy(0, accepted_indices, accepted_time)
        current = current.index_copy(0, accepted_indices, accepted_field)
        step = step.index_copy(0, indices, next_step)

        for position, row in enumerate(accepted_indices.tolist()):
            accepted_times[row].append(time[row])
            accepted_states[row].append(state[row])
            accepted_derivatives[row].append(current[row])
        accepted_stop, stop_reason = _event_call(
            event, accepted_time, accepted_next, accepted_field,
            previous_field, accepted_indices)
        for position, row in enumerate(accepted_indices.tolist()):
            if bool(accepted_stop[position]):
                active[row] = False
                termination[row] = (str(stop_reason) if stop_reason is not None
                                    else "event")
                event_time[row] = time[row]
        if end is not None:
            at_end = accepted_time >= end
            for position, row in enumerate(accepted_indices.tolist()):
                if bool(at_end[position]) and active[row]:
                    active[row] = False
                    termination[row] = "duration"
                    event_time[row] = time[row]

    counts = [len(row) for row in accepted_times]
    max_count = max(counts)
    components = flat_initial.shape[-1]
    acc_time = torch.zeros((rows, max_count), dtype=dtype, device=device)
    acc_state = torch.zeros((rows, max_count, components), dtype=dtype, device=device)
    acc_derivative = torch.zeros_like(acc_state)
    valid = torch.zeros((rows, max_count), dtype=torch.bool, device=device)
    for row, count in enumerate(counts):
        row_time = torch.stack(accepted_times[row])
        row_state = torch.stack(accepted_states[row])
        row_derivative = torch.stack(accepted_derivatives[row])
        acc_time[row, :count] = row_time
        acc_state[row, :count] = row_state
        acc_derivative[row, :count] = row_derivative
        valid[row, :count] = True
        if count < max_count:
            acc_time[row, count:] = row_time[-1]
            acc_state[row, count:] = row_state[-1]
            acc_derivative[row, count:] = row_derivative[-1]
    output_time, output_state, output_valid, _ = _build_resampled_grid(
        tuple(torch.stack(accepted_times[row]) for row in range(rows)),
        tuple(torch.stack(accepted_states[row]) for row in range(rows)),
        out_step, (rows,))
    if not batch_shape:
        acc_time, acc_state, acc_derivative, valid = (
            value.squeeze(0) for value in (acc_time, acc_state, acc_derivative, valid))
        output_time, output_state, output_valid = (
            value.squeeze(0) for value in (output_time, output_state, output_valid))
    else:
        acc_time = acc_time.reshape(batch_shape + (max_count,))
        acc_state = acc_state.reshape(batch_shape + (max_count, components))
        acc_derivative = acc_derivative.reshape(batch_shape + (max_count, components))
        valid = valid.reshape(batch_shape + (max_count,))
        output_time = output_time.reshape(batch_shape + (output_time.shape[-1],))
        output_state = output_state.reshape(batch_shape + (output_state.shape[-2], components))
        output_valid = output_valid.reshape(batch_shape + (output_valid.shape[-1],))
    stats = AdaptiveStatistics(
        accepted_steps=accepted_steps.reshape(batch_shape),
        rejected_steps=rejected_steps.reshape(batch_shape),
        evaluations=evaluations.reshape(batch_shape),
        termination=tuple(termination),
        event_time=event_time.reshape(batch_shape),
    )
    return AdaptiveTrajectory(
        acc_time, acc_state, acc_derivative, valid,
        output_time, output_state, output_valid, stats, out_step, first_step,
        step.squeeze(0) if not batch_shape else step.reshape(batch_shape),
    )


def integrate_adaptive(
    vector_field: Callable,
    initial_state: Tensor,
    output_step: float | Tensor,
    *,
    t_end: Optional[float | Tensor] = None,
    event: Optional[Callable] = None,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-9,
    controller: str = GENERIC_CONTROLLER,
    batch_mode: str = "full",
    initial_step: Optional[float | Tensor] = None,
    min_step: Optional[float] = None,
    max_step: Optional[float] = None,
    max_rejected_steps: int = 10000,
    max_evaluations: int = 200000,
    max_accepted_steps: int = 100000,
    max_duration: Optional[float] = None,
) -> AdaptiveTrajectory:
    """Run the bounded adaptive driver and retain any accepted error prefix.

    The implementation remains unchanged for successful runs.  On an
    :class:`AdaptiveIntegrationError`, the error receives per-row accepted
    times, states, derivatives, and accepted-step counts accumulated before
    the failing trial.  A scalar run stores tensors directly; a batched run
    stores one tensor per flattened row so ragged prefixes remain exact.
    """
    partial: dict[str, object] = {}
    try:
        return _integrate_adaptive_impl(
            vector_field,
            initial_state,
            output_step,
            t_end=t_end,
            event=event,
            atol=atol,
            rtol=rtol,
            controller=controller,
            batch_mode=batch_mode,
            initial_step=initial_step,
            min_step=min_step,
            max_step=max_step,
            max_rejected_steps=max_rejected_steps,
            max_evaluations=max_evaluations,
            max_accepted_steps=max_accepted_steps,
            max_duration=max_duration,
            _partial=partial,
        )
    except AdaptiveIntegrationError as exc:
        accepted_times = partial.get("accepted_times")
        accepted_states = partial.get("accepted_states")
        accepted_derivatives = partial.get("accepted_derivatives")
        accepted_steps = partial.get("accepted_steps")
        if accepted_times is not None and accepted_states is not None and accepted_derivatives is not None:
            # The lists are created together and have one row for every
            # flattened input row.  Stack only the accepted prefix; unlike
            # the successful batch trajectory this path never pads rows.
            times = tuple(torch.stack(row) for row in accepted_times)
            states = tuple(torch.stack(row) for row in accepted_states)
            derivatives = tuple(torch.stack(row) for row in accepted_derivatives)
            if len(times) == 1:
                exc.accepted_time = times[0]
                exc.accepted_state = states[0]
                exc.accepted_derivative = derivatives[0]
                if accepted_steps is not None:
                    exc.accepted_steps = int(accepted_steps[0].detach().item())
            else:
                exc.accepted_time = times
                exc.accepted_state = states
                exc.accepted_derivative = derivatives
                if accepted_steps is not None:
                    exc.accepted_steps = accepted_steps.detach().clone()
        raise


class AlignedSpinV4Event:
    """Accepted-state plunge event used by the non-optimized v4 route."""

    def __init__(self, stage: str = "coarse", *, radius: float = 6.0,
                 frequency_decreases: int = 5):
        if stage not in {"coarse", "fine"}:
            raise ValueError("stage must be 'coarse' or 'fine'")
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive")
        if stage == "fine" and frequency_decreases < 1:
            raise ValueError("frequency_decreases must be positive")
        self.stage = stage
        self.radius = float(radius)
        self.frequency_decreases = int(frequency_decreases)
        self.count: Optional[Tensor] = None

    def initialize(self, rows: int, *, dtype: torch.dtype, device: torch.device):
        self.count = torch.zeros((rows,), dtype=torch.long, device=device)

    def check(self, time: Tensor, state: Tensor, derivative: Tensor,
              previous: Optional[Tensor], indices: Tensor):
        if self.count is None:
            self.initialize(int(indices.max().detach()) + 1, dtype=state.dtype, device=state.device)
        inside = state[..., 0] < self.radius
        if previous is None:
            decrease = torch.zeros_like(inside)
        else:
            decrease = inside & (derivative[..., 1] < previous[..., 1])
        self.count = self.count.index_add(
            0, indices, decrease.to(self.count.dtype))
        radial_nonnegative = derivative[..., 2] >= 0
        if self.stage == "coarse":
            stop = inside & (decrease | radial_nonnegative)
        else:
            stop = inside & ((self.count[indices] >= self.frequency_decreases)
                             | radial_nonnegative)
        return stop


@dataclass(frozen=True)
class SplicedNumericalAttachment:
    """Fine attachment plus its coarse-grid splice metadata."""

    mode: Tensor
    corrected_inspiral: Tensor
    peak: object
    match_time: Tensor
    boundary_time: Tensor
    boundary_index: Tensor
    boundary_phase: Tensor
    boundary_frequency: Tensor
    boundary_amplitude: Tensor
    internal_step: Tensor
    fine: NumericalAttachedMode22
    restart_index: int
    refinement_factor: int


@dataclass(frozen=True)
class TwoStageEOBTrajectory:
    """Coarse/fine trajectories and the global peak used by the FD adapter."""

    coarse: AdaptiveTrajectory
    fine: AdaptiveTrajectory
    restart_index: int
    restart_time: Tensor
    peak: object
    time: Tensor
    state: Tensor
    frequency: Tensor
    valid: Tensor


def _stack_adaptive_trajectories(runs: Sequence[AdaptiveTrajectory],
                                 batch_shape: tuple[int, ...]) -> AdaptiveTrajectory:
    """Pad and stack scalar adaptive runs while retaining graph edges."""
    if not runs:
        raise ValueError("at least one trajectory is required")

    def padded(values, length, *, mask=False):
        result = []
        for value in values:
            if value.shape[0] < length:
                tail_shape = (length - value.shape[0],) + value.shape[1:]
                if mask:
                    tail = torch.zeros(tail_shape, dtype=torch.bool, device=value.device)
                else:
                    tail = value[-1:].expand(tail_shape)
                value = torch.cat((value, tail), dim=0)
            result.append(value)
        return torch.stack(result).reshape(batch_shape + (length,) + result[0].shape[1:])

    max_accepted = max(run.time.shape[0] for run in runs)
    max_output = max(run.output_time.shape[0] for run in runs)
    statistics = AdaptiveStatistics(
        accepted_steps=torch.stack([run.statistics.accepted_steps for run in runs]).reshape(batch_shape),
        rejected_steps=torch.stack([run.statistics.rejected_steps for run in runs]).reshape(batch_shape),
        evaluations=torch.stack([run.statistics.evaluations for run in runs]).reshape(batch_shape),
        termination=tuple(label for run in runs for label in run.statistics.termination),
        event_time=torch.stack([run.statistics.event_time for run in runs]).reshape(batch_shape),
    )
    return AdaptiveTrajectory(
        padded([run.time for run in runs], max_accepted),
        padded([run.state for run in runs], max_accepted),
        padded([run.derivative for run in runs], max_accepted),
        padded([run.valid for run in runs], max_accepted, mask=True),
        padded([run.output_time for run in runs], max_output),
        padded([run.output_state for run in runs], max_output),
        padded([run.output_valid for run in runs], max_output, mask=True),
        statistics,
        runs[0].output_step,
        runs[0].initial_step,
        (torch.stack([
            run.next_step if run.next_step.ndim == 0 else run.next_step.reshape(())
            for run in runs
        ]).reshape(batch_shape) if runs[0].next_step is not None else None),
    )


def _preserve_output_phase(state: Tensor) -> Tensor:
    # Kept as a named seam for reviewers: the state itself carries the
    # continuous orbital phase, so no post-restart phase re-zeroing occurs.
    return state


def _apply_calibrated_nqc(mode: Tensor, state: Tensor, orbital_frequency: Tensor,
                          coefficients) -> Tensor:
    """Apply one fine-series NQC calibration to a trajectory mode.

    The fine attachment solves one pair of amplitude/phase coefficient
    vectors.  Applying that same pair to the coarse prefix keeps the splice
    in one calibrated waveform chart; evaluating the two segments separately
    would otherwise introduce a correction jump at the restart state.
    """
    if state.shape[-1] != 4 or mode.shape[-1] != state.shape[-2]:
        raise ValueError("mode/state sample axes do not agree for NQC correction")
    return mode * nqc_22_factor(
        state[..., 0], state[..., 2], orbital_frequency, coefficients,
        sample_dims=1,
    )


def reconstruct_seobnrv4_22_adaptive(
    omega_start: Tensor,
    mass1,
    mass2,
    chi1=0.0,
    chi2=0.0,
    *,
    spin_truncation: str = "generic-leading-ell8",
    spin_v6: str = "bohe_total_energy",
    include_pan_v8: bool = False,
    include_positive_chi_540: bool = False,
    final_spin_fit_order=(3, 4),
    dt_geom: float = 0.5,
    num_steps: int = 20000,
    ringdown_duration: float = 150.0,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-9,
    max_rejected_steps: int = 10000,
    max_evaluations: int = 200000,
    max_accepted_steps: Optional[int] = None,
    max_duration: Optional[float] = None,
):
    """Run the opt-in two-stage numerical-compatibility EOB reconstruction.

    The coarse stage starts at ``dt_geom`` and is naturally resampled on that
    grid.  A fine restart is selected from a real coarse sample using the
    150M/half-trajectory rule; its local clock starts at zero while the
    restart state's orbital phase is retained.  The returned mode is the
    coarse prefix followed by every power-of-two-th fine sample.  This helper
    is intentionally independent of the fixed-step default reconstruction.
    """
    if not isinstance(omega_start, Tensor) or omega_start.dtype not in (torch.float32, torch.float64):
        raise TypeError("omega_start must be a float32/float64 tensor")
    if not math.isfinite(float(dt_geom)) or dt_geom <= 0:
        raise ValueError("dt_geom must be finite and positive")
    if not isinstance(num_steps, int) or num_steps < 4:
        raise ValueError("num_steps must be an integer of at least four")
    if not math.isfinite(float(ringdown_duration)) or ringdown_duration <= 0:
        raise ValueError("ringdown_duration must be finite and positive")
    if spin_truncation != "generic-leading-ell8":
        raise ValueError(
            "adaptive numerical compatibility requires spin_truncation='generic-leading-ell8'"
        )
    if tuple(final_spin_fit_order) != (3, 4):
        raise ValueError(
            "adaptive numerical compatibility requires final_spin_fit_order=(3, 4)"
        )
    if spin_v6 != "bohe_total_energy" or include_pan_v8 is not False or include_positive_chi_540 is not False:
        raise ValueError("adaptive numerical compatibility requires the Bohé total-energy phase choices")
    from .initial_conditions import find_quasicircular_initial_conditions
    from .published_waveform import published_mode22_and_frequency
    from .remnant import eob_final_mass_fraction, hofmann_final_spin
    from .seobnrv4_flux import SEOBNRv4PublishedFlux
    from .seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
    from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention
    from .nqc import seobnrv4_nqc_parameters, seobnrv4_nqc_time_offset

    values = [torch.as_tensor(value, dtype=omega_start.dtype, device=omega_start.device)
              for value in (mass1, mass2, chi1, chi2)]
    start, m1, m2, s1, s2 = torch.broadcast_tensors(omega_start, *values)
    if not bool((torch.isfinite(start) & (start > 0)).all()):
        raise ValueError("omega_start must be finite and positive")
    dtype, device = start.dtype, start.device
    batch_shape = tuple(start.shape)
    rows = math.prod(batch_shape) if batch_shape else 1
    maximum_steps = int(max_accepted_steps if max_accepted_steps is not None else num_steps)
    mass_fraction = eob_final_mass_fraction(m1, m2, s1, s2)
    final_spin = hofmann_final_spin(m1, m2, s1, s2, fit_order=final_spin_fit_order)
    qnm_w, qnm_damp = signed_qnm_220_table_convention(final_spin)
    sigma = torch.complex(qnm_w, -qnm_damp) / mass_fraction
    scaled1, scaled2 = m1 / torch.maximum(m1, m2), m2 / torch.maximum(m1, m2)
    x1, x2 = scaled1 / (scaled1 + scaled2), scaled2 / (scaled1 + scaled2)
    nu, chi = seobnrv4_nqc_parameters(x1, x2, s1, s2)
    ham = SEOBNRv4Hamiltonian(nu, s1, s2, delta=x1 - x2)
    rr = SEOBNRv4PublishedFlux(spin_truncation=spin_truncation,
                               retain_pan_cubic_spin=True)
    # A scalar Hamiltonian is needed for each fine row.  Slicing its tensor
    # parameters preserves the graph while avoiding accidental broadcasting
    # of a row state back across the whole batch.
    flat_nu, flat_s1, flat_s2 = nu.reshape(-1), s1.reshape(-1), s2.reshape(-1)
    flat_delta = (x1 - x2).reshape(-1)
    row_hamiltonians = [SEOBNRv4Hamiltonian(
        flat_nu[row], flat_s1[row], flat_s2[row], delta=flat_delta[row])
        for row in range(rows)]
    r0, pr0, pphi0 = find_quasicircular_initial_conditions(ham, rr, start)
    initial = torch.stack((r0, torch.zeros_like(r0), pr0, pphi0), dim=-1)
    initial_flat = initial.reshape(rows, 4)
    # Run every coarse row through a scalar closure.  A full-batch EOB field
    # would bind all Hamiltonian parameters while the adaptive loop selected a
    # shrinking subset after row-specific event histories diverged.
    coarse_runs = []
    for row in range(rows):
        row_field = EOBTrajectoryIntegrator(row_hamiltonians[row], rr).vector_field
        coarse_runs.append(integrate_adaptive(
            row_field, initial_flat[row], dt_geom,
            event=AlignedSpinV4Event("coarse"), atol=atol, rtol=rtol,
            controller=GSL_STATE_SCALED_CONTROLLER,
            max_rejected_steps=max_rejected_steps, max_evaluations=max_evaluations,
            max_accepted_steps=maximum_steps, max_duration=max_duration,
        ))
    coarse_counts = [int(run.output_valid.sum().detach()) for run in coarse_runs]
    if len(set(coarse_counts)) != 1:
        raise ValueError(
            "adaptive EOB compatibility requires equal coarse output lengths "
            f"across a batch; got {coarse_counts}"
        )
    coarse = (coarse_runs[0] if not batch_shape else
              _stack_adaptive_trajectories(coarse_runs, batch_shape))
    n_coarse = coarse_counts[0]
    if n_coarse < 4:
        raise AdaptiveIntegrationError("short_coarse_trajectory", "coarse trajectory needs at least four output samples")
    duration = (n_coarse - 1) * float(dt_geom)
    if duration >= 150.0:
        restart_index = n_coarse - math.ceil(150.0 / float(dt_geom))
    else:
        restart_index = n_coarse - math.ceil(n_coarse / 2.0)
    if restart_index < 1 or restart_index >= n_coarse:
        raise AdaptiveIntegrationError("invalid_restart_index", "two-stage restart index is outside the coarse trajectory")
    coarse_state = coarse.output_state
    coarse_time = coarse.output_time
    coarse_state_flat = coarse_state.reshape(rows, n_coarse, 4)
    coarse_time_flat = coarse_time.reshape(rows, n_coarse)
    restart_state = _preserve_output_phase(coarse_state_flat[:, restart_index])
    restart_time = coarse_time_flat[:, restart_index]
    fine_step = float(dt_geom) / 2.0 ** max(0, math.ceil(math.log2(50.0 * float(dt_geom))))
    fine_events = [AlignedSpinV4Event("fine") for _ in range(rows)]
    fine_runs = []
    for row in range(rows):
        # Calling the same vector field with a single row keeps all parameter
        # tensors on their native graph while making the event count local.
        row_initial = restart_state[row]
        row_field = EOBTrajectoryIntegrator(row_hamiltonians[row], rr).vector_field
        fine_runs.append(integrate_adaptive(
            row_field, row_initial, fine_step, event=fine_events[row],
            atol=atol, rtol=rtol, controller=GSL_STATE_SCALED_CONTROLLER,
            max_rejected_steps=max_rejected_steps, max_evaluations=max_evaluations,
            max_accepted_steps=maximum_steps,
            max_duration=max_duration,
        ))
    if any(tuple(run.output_state.shape[:-2]) for run in fine_runs):
        raise RuntimeError("unexpected fine trajectory shape")
    fine_counts = [int(run.output_valid.sum().detach()) for run in fine_runs]
    if len(set(fine_counts)) != 1:
        raise ValueError("adaptive EOB compatibility requires equal fine output lengths across a batch")
    n_fine = fine_counts[0]
    refinement = int(round(float(dt_geom) / fine_step))
    if refinement < 1 or refinement & (refinement - 1):
        raise RuntimeError("fine step is not a power-of-two refinement of the coarse step")

    # Build each row's base mode and attachment on the actual fine trajectory.
    attachments = []
    trajectories = []
    from .events import OrbitalFrequencyPeak
    for row, run in enumerate(fine_runs):
        row_coarse_state = coarse_state_flat[row, :n_coarse]
        row_fine_state = run.output_state[:n_fine]
        row_ham = row_hamiltonians[row]
        row_field = EOBTrajectoryIntegrator(row_ham, rr).vector_field
        row_coarse_mode, _ = published_mode22_and_frequency(
            row_coarse_state, row_ham, rr, spin_v6=spin_v6,
            include_pan_v8=False, include_positive_chi_540=False,
            retain_pan_cubic_spin=True)
        row_fine_mode, row_fine_frequency = published_mode22_and_frequency(
            row_fine_state, row_ham, rr, spin_v6=spin_v6,
            include_pan_v8=False, include_positive_chi_540=False,
            retain_pan_cubic_spin=True)
        coarse_field = row_field(row_coarse_state)
        fine_field = row_field(row_fine_state)
        # Calibrate on the complete fine series, then carry the same five
        # coefficients back over the coarse prefix.  The raw coarse mode is
        # only an input to this correction; splicing it directly to the
        # corrected fine mode would create an artificial seam.
        row_fine_time = run.output_time[:n_fine]
        # First obtain the continuous peak/match from the actual fine
        # trajectory.  A second attachment evaluation then includes the full
        # finite-support ringdown on its local fine grid before downsampling.
        provisional = attach_seobnrv4_22_numerical_compatibility(
            row_fine_time, row_fine_state, fine_field[..., 1], row_fine_mode,
            row_fine_frequency, nu.reshape(-1)[row], chi.reshape(-1)[row],
            sigma.reshape(-1)[row], row_fine_time,
        )
        match_local = provisional.match_time
        output_count = max(
            2,
            int(torch.ceil((match_local + ringdown_duration).detach() / fine_step).item()) + 1,
        )
        row_output_time = torch.arange(output_count, dtype=dtype, device=device) * fine_step
        attachment = attach_seobnrv4_22_numerical_compatibility(
            row_fine_time, row_fine_state, fine_field[..., 1], row_fine_mode,
            row_fine_frequency, nu.reshape(-1)[row], chi.reshape(-1)[row],
            sigma.reshape(-1)[row], row_output_time,
        )
        coarse_coefficients = attachment.calibration.coefficients
        row_coarse_mode = _apply_calibrated_nqc(
            row_coarse_mode, row_coarse_state, coarse_field[..., 1],
            coarse_coefficients,
        )
        fine_mode = torch.where(
            row_output_time <= attachment.match_time + ringdown_duration,
            attachment.mode, torch.zeros_like(attachment.mode),
        )
        fine_indices = torch.arange(0, output_count, refinement, device=device)
        actual_indices = fine_indices.clamp_max(n_fine - 1)
        fine_state_samples = row_fine_state[actual_indices]
        fine_frequency_samples = fine_field[actual_indices, 1]
        fine_valid_samples = fine_indices < n_fine
        prefix_mode = row_coarse_mode[:restart_index]
        spliced_mode = torch.cat((prefix_mode, fine_mode[fine_indices]), dim=-1)
        spliced_state = torch.cat((row_coarse_state[:restart_index], fine_state_samples), dim=0)
        spliced_time = torch.cat((coarse_time_flat[row, :restart_index],
                                  restart_time[row] + row_output_time[fine_indices]))
        spliced_frequency = torch.cat((coarse_field[:restart_index, 1], fine_frequency_samples))
        spliced_valid = torch.cat((
            torch.ones((restart_index,), dtype=torch.bool, device=device),
            fine_valid_samples,
        ))
        peak = OrbitalFrequencyPeak(
            attachment.peak.time + restart_time[row], attachment.peak.frequency,
            attachment.peak.curvature, attachment.peak.sample_index)
        attachments.append(SplicedNumericalAttachment(
            spliced_mode, attachment.corrected_inspiral, peak,
            attachment.match_time + restart_time[row],
            attachment.boundary_time + restart_time[row], attachment.boundary_index,
            attachment.boundary_phase, attachment.boundary_frequency,
            attachment.boundary_amplitude, attachment.internal_step,
            attachment, restart_index, refinement))
        trajectories.append((spliced_time, spliced_state, spliced_frequency, peak, spliced_valid))

    if len({item[0].shape[0] for item in trajectories}) != 1:
        raise ValueError("adaptive EOB compatibility requires equal splice lengths across a batch")

    # All rows were checked to have equal lengths; stack while retaining each
    # row's parameter graph.  ``PublishedEOB22`` is imported lazily to keep
    # this module usable by analytic-driver tests without a waveform import.
    from .published_waveform import PublishedEOB22
    mode = torch.stack([item.mode for item in attachments]).reshape(batch_shape + (-1,)) if batch_shape else attachments[0].mode
    if batch_shape:
        trajectory_time = torch.stack([item[0] for item in trajectories]).reshape(batch_shape + (-1,))
        trajectory_state = torch.stack([item[1] for item in trajectories]).reshape(batch_shape + (-1, 4))
        trajectory_frequency = torch.stack([item[2] for item in trajectories]).reshape(batch_shape + (-1,))
        trajectory_valid = torch.stack([item[4] for item in trajectories]).reshape(batch_shape + (-1,))
        peak = attachments[0].peak
        # The public FD path accepts a batch peak tensor.  Build it from all
        # rows rather than broadcasting row zero's event metadata.
        peak = OrbitalFrequencyPeak(
            torch.stack([item.peak.time for item in attachments]).reshape(batch_shape),
            torch.stack([item.peak.frequency for item in attachments]).reshape(batch_shape),
            torch.stack([item.peak.curvature for item in attachments]).reshape(batch_shape),
            torch.stack([item.peak.sample_index for item in attachments]).reshape(batch_shape),
        )
        # The public FD adapter needs row-wise match times and mode metadata.
        # Keep the first row's opaque calibration object as an implementation
        # detail while stacking all numerical fields that participate in the
        # public result.
        attachment = SplicedNumericalAttachment(
            mode=mode,
            corrected_inspiral=torch.stack([item.corrected_inspiral for item in attachments]),
            peak=peak,
            match_time=torch.stack([item.match_time for item in attachments]).reshape(batch_shape),
            boundary_time=torch.stack([item.boundary_time for item in attachments]).reshape(batch_shape),
            boundary_index=torch.stack([item.boundary_index for item in attachments]).reshape(batch_shape),
            boundary_phase=torch.stack([item.boundary_phase for item in attachments]).reshape(batch_shape),
            boundary_frequency=torch.stack([item.boundary_frequency for item in attachments]).reshape(batch_shape),
            boundary_amplitude=torch.stack([item.boundary_amplitude for item in attachments]).reshape(batch_shape),
            internal_step=torch.stack([item.internal_step for item in attachments]).reshape(batch_shape),
            fine=attachments[0].fine,
            restart_index=restart_index,
            refinement_factor=refinement,
        )
    else:
        trajectory_time, trajectory_state, trajectory_frequency, peak, trajectory_valid = trajectories[0]
        attachment = attachments[0]
    fine_trajectory = (fine_runs[0] if not batch_shape else
                       _stack_adaptive_trajectories(fine_runs, batch_shape))
    trajectory = TwoStageEOBTrajectory(
        coarse=coarse, fine=fine_trajectory,
        restart_index=restart_index, restart_time=restart_time,
        peak=peak, time=trajectory_time, state=trajectory_state,
        frequency=trajectory_frequency,
        valid=trajectory_valid,
    )
    public_time = trajectory_time if not batch_shape else trajectory_time.reshape(rows, -1)[0]
    return PublishedEOB22(
        public_time, mode,
        trajectory, attachment, mass_fraction, final_spin,
    )


reconstruct_seobnrv4_22_numerical_compatibility = reconstruct_seobnrv4_22_adaptive


__all__ = [
    "AdaptiveIntegrationError",
    "AdaptiveStatistics",
    "AdaptiveTrajectory",
    "AlignedSpinV4Event",
    "GENERIC_CONTROLLER",
    "GSL_STATE_SCALED_CONTROLLER",
    "STATE_SCALED_CONTROLLER",
    "SplicedNumericalAttachment",
    "TwoStageEOBTrajectory",
    "integrate_adaptive",
    "resample_natural_cubic",
    "reconstruct_seobnrv4_22_adaptive",
    "reconstruct_seobnrv4_22_numerical_compatibility",
    "state_scaled_controller_adjustment",
    "gsl_state_scaled_controller",
    "state_scaled_controller",
]

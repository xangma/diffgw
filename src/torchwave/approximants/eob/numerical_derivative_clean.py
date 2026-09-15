"""Independent central finite differences for clean EOB diagnostics.

This module is deliberately additive.  It does not replace the analytic
Hamiltonian derivative path or select a trajectory/integrator default.  The
Hamiltonian is treated as an opaque object exposing ``real_hamiltonian``,
``effective_hamiltonian`` and ``metric_potentials``.

The numerical convention follows the public description of GSL's central
derivative routine, while the spacing estimate below is derived locally.  For
an initial trial spacing ``h`` we evaluate

    D2 = (f(x+h) - f(x-h))/(2*h)
    D4 = (f(x-h) - 8*f(x-h/2) + 8*f(x+h/2) - f(x+h))/(6*h).

The difference ``abs(D4-D2)`` is used as an ``O(h**2)`` truncation estimate.
For the four-point D4 numerator, independent relative evaluation errors give
the roundoff bound

    eps*(abs(f(-h)) + 8*abs(f(-h/2)) +
         8*abs(f(h/2)) + abs(f(h)))/(6*abs(h)).

Writing these estimates as ``b*h**2`` and ``a/h`` gives the trial update
``h=(a/(2*b))**(1/3)``.  The update is limited to a factor of ten in either
direction and one update is made by default.  These bounds and limits are
diagnostic choices, rather than an assertion about undocumented native GSL
implementation details.

For an aligned state ``(r, phi, p_r*, j)``, the Cartesian orientation is
``position=(r,0,0)`` and ``momentum=(p_r*,j/r,0)``.  The three partials are
then obtained by varying ``x``, ``p_x`` and ``p_y`` independently, as in the
review contract.  The full real Hamiltonian divided by ``nu`` is used for
these scalar evaluations; at exactly ``nu=0`` its finite derivative limit is
the effective Hamiltonian because the additive rest-energy constant has no
coordinate derivative.
"""

from __future__ import annotations

import copy
from typing import Callable, NamedTuple, Optional, Tuple, Union

import torch


Tensor = torch.Tensor
Scalar = Union[float, Tensor]
ScalarFunction = Callable[[Tensor], Tensor]


class NumericalDerivativeDomainError(ValueError):
    """Raised when a finite-difference node or value is outside the domain.

    ``row_indices`` contains the indices of invalid rows when they can be
    identified.  A scalar failure uses an empty tensor because there is no
    row axis; callers can still distinguish it from an ordinary type error by
    catching this exception class.
    """

    def __init__(self, message: str, row_indices: Optional[Tensor] = None):
        super().__init__(message)
        self.row_indices = row_indices


class NumericalDerivativeResult(NamedTuple):
    """Derivative and diagnostics for one adaptive scalar-function call."""

    derivative: Tensor
    error: Tensor
    truncation_error: Tensor
    roundoff_error: Tensor
    step: Tensor
    initial_step: Tensor
    evaluations: Tensor


class CartesianHamiltonianPartials(NamedTuple):
    """Cartesian partials and tortoise factor at an aligned EOB state."""

    h_x: Tensor
    h_px: Tensor
    h_py: Tensor
    xi: Tensor

    @property
    def H_x(self) -> Tensor:
        """Compatibility spelling for the contract's ``H_x`` notation."""
        return self.h_x

    @property
    def H_px(self) -> Tensor:
        return self.h_px

    @property
    def H_py(self) -> Tensor:
        return self.h_py


class CartesianPartialReport(NamedTuple):
    """Cartesian partials together with each one-dimensional estimate."""

    partials: CartesianHamiltonianPartials
    x: NumericalDerivativeResult
    p_x: NumericalDerivativeResult
    p_y: NumericalDerivativeResult


class EOBConservativeField(NamedTuple):
    """Numerically differentiated conservative EOB vector field."""

    dr_dt: Tensor
    omega: Tensor
    dpr_star_dt: Tensor


class EOBNumericalDerivativeReport(NamedTuple):
    """Conservative field, Cartesian partials, and spacing diagnostics."""

    field: EOBConservativeField
    partials: CartesianHamiltonianPartials
    x: NumericalDerivativeResult
    p_x: NumericalDerivativeResult
    p_y: NumericalDerivativeResult

    @property
    def dr_dt(self) -> Tensor:
        return self.field.dr_dt

    @property
    def omega(self) -> Tensor:
        return self.field.omega

    @property
    def dpr_star_dt(self) -> Tensor:
        return self.field.dpr_star_dt


def _ordinary_tensor(value: Tensor) -> Tensor:
    """Clone inference tensors while preserving ordinary autograd inputs."""
    if value.is_inference():
        return value.clone()
    return value


def _ordinary_hamiltonian(hamiltonian):
    """Make an inference-safe shallow copy without changing the caller."""
    try:
        values = vars(hamiltonian)
    except TypeError:
        return hamiltonian
    if not any(isinstance(value, Tensor) and value.is_inference()
               for value in values.values()):
        return hamiltonian
    local = copy.copy(hamiltonian)
    for name, value in values.items():
        if isinstance(value, Tensor) and value.is_inference():
            setattr(local, name, value.clone())
    return local


def _as_float_tensor(value, *, like: Optional[Tensor], name: str) -> Tensor:
    if isinstance(value, Tensor):
        result = value
    else:
        kwargs = {}
        if like is not None:
            kwargs.update(dtype=like.dtype, device=like.device)
        result = torch.as_tensor(value, **kwargs)
    if result.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    return _ordinary_tensor(result)


def _row_indices(mask: Tensor) -> Tensor:
    """Return useful row/index information for a boolean invalid mask."""
    if mask.ndim == 0:
        return torch.empty(0, dtype=torch.long, device=mask.device)
    return torch.nonzero(mask, as_tuple=False)


def _require_finite(value: Tensor, name: str, *, domain: bool = True) -> None:
    if not bool(torch.isfinite(value).all()):
        bad = ~torch.isfinite(value)
        error = NumericalDerivativeDomainError if domain else ValueError
        raise error(f"{name} must be finite", _row_indices(bad) if domain else None)


def _require_positive(value: Tensor, name: str) -> None:
    _require_finite(value, name)
    if bool((value <= 0).any()):
        raise NumericalDerivativeDomainError(
            f"{name} must be positive", _row_indices(value <= 0))


def _prepare_step(step: Scalar, *, like: Tensor, name: str = "step") -> Tensor:
    result = _as_float_tensor(step, like=like, name=name)
    _require_positive(result, name)
    return result


def _prepare_values(values, *, like: Tensor, name: str = "function value") -> Tensor:
    if not isinstance(values, Tensor):
        values = torch.as_tensor(values, dtype=like.dtype, device=like.device)
    if values.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be a real floating-point tensor")
    _require_finite(values, name)
    return values


def central_quadratic_from_values(values: Tensor, step: Scalar) -> Tensor:
    """Apply the central two-point (quadratic-error) stencil.

    ``values[..., 0]`` and ``values[..., 1]`` must be evaluations at ``x-h``
    and ``x+h`` respectively.  This low-level function is useful for checking
    the independently derived weights without invoking a Hamiltonian.
    """
    if not isinstance(values, Tensor) or values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("values must have final axis (x-h, x+h)")
    values = _prepare_values(values, like=values)
    step = _prepare_step(step, like=values)
    return (values[..., 1] - values[..., 0]) / (2.0 * step)


def central_quartic_from_values(values: Tensor, step: Scalar) -> Tensor:
    """Apply the central four-point quartic-accuracy stencil.

    ``values[..., :]`` are ordered ``(x-h, x-h/2, x+h/2, x+h)``.  The center
    value is intentionally absent, matching the public central-difference
    convention.
    """
    if not isinstance(values, Tensor) or values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError("values must have final axis (x-h, x-h/2, x+h/2, x+h)")
    values = _prepare_values(values, like=values)
    step = _prepare_step(step, like=values)
    numerator = (values[..., 0] - 8.0 * values[..., 1]
                 + 8.0 * values[..., 2] - values[..., 3])
    return numerator / (6.0 * step)


def _evaluate(function: ScalarFunction, point: Tensor, *, name: str) -> Tensor:
    if not isinstance(function, Callable):
        raise TypeError("function must be callable")
    _require_finite(point, name + " node")
    value = function(point)
    return _prepare_values(value, like=point, name=name + " value")


def _stencil_values(function: ScalarFunction, x: Tensor, step: Tensor):
    """Evaluate the four quartic nodes once and return both stencil values."""
    h = step
    values = (
        _evaluate(function, x - h, name="x-h"),
        _evaluate(function, x - 0.5 * h, name="x-h/2"),
        _evaluate(function, x + 0.5 * h, name="x+h/2"),
        _evaluate(function, x + h, name="x+h"),
    )
    try:
        values = torch.broadcast_tensors(*values)
    except RuntimeError as exc:
        raise ValueError("function values at stencil nodes do not broadcast") from exc
    stacked = torch.stack(values, dim=-1)
    return stacked


def central_quadratic(function: ScalarFunction, x: Tensor, step: Scalar,
                      *, return_values: bool = False):
    """Evaluate the central two-point derivative at a supplied spacing."""
    if not isinstance(x, Tensor):
        raise TypeError("x must be a torch.Tensor")
    x = _as_float_tensor(x, like=x, name="x")
    _require_finite(x, "x")
    step = _prepare_step(step, like=x)
    values = (
        _evaluate(function, x - step, name="x-h"),
        _evaluate(function, x + step, name="x+h"),
    )
    try:
        values = torch.broadcast_tensors(*values)
    except RuntimeError as exc:
        raise ValueError("function values at stencil nodes do not broadcast") from exc
    values = torch.stack(values, dim=-1)
    result = central_quadratic_from_values(values, step)
    return (result, values) if return_values else result


def central_quartic(function: ScalarFunction, x: Tensor, step: Scalar,
                    *, return_values: bool = False):
    """Evaluate the central four-point derivative at a supplied spacing."""
    if not isinstance(x, Tensor):
        raise TypeError("x must be a torch.Tensor")
    x = _as_float_tensor(x, like=x, name="x")
    _require_finite(x, "x")
    step = _prepare_step(step, like=x)
    values = _stencil_values(function, x, step)
    result = central_quartic_from_values(values, step)
    return (result, values) if return_values else result


def _sample(function: ScalarFunction, x: Tensor, step: Tensor):
    values = _stencil_values(function, x, step)
    d4 = central_quartic_from_values(values, step)
    d2 = central_quadratic_from_values(values[..., (0, 3)], step)
    step_for_result = torch.broadcast_to(step, d4.shape)
    # The error model is an absolute bound for the D4 numerator.  Its
    # coefficient is deliberately simple and uses dtype machine epsilon.
    epsilon = torch.finfo(values.dtype).eps
    roundoff = epsilon * (
        values[..., 0].abs() + 8.0 * values[..., 1].abs()
        + 8.0 * values[..., 2].abs() + values[..., 3].abs()
    ) / (6.0 * step_for_result.abs())
    truncation = (d4 - d2).abs()
    error = truncation + roundoff
    return d4, truncation, roundoff, error, step_for_result, values


def _adapted_step(x: Tensor, step: Tensor, truncation: Tensor,
                  roundoff: Tensor) -> Tensor:
    """Derive one spacing update from the ``a/h+b*h**2`` model."""
    step = torch.broadcast_to(step, truncation.shape)
    # abs(x) is a coordinate scale, while one avoids a vanishing floor near
    # the origin.  The floor is only a roundoff safeguard, not a domain clamp.
    x_scale = torch.broadcast_to(torch.maximum(x.abs(), torch.ones_like(x)), truncation.shape)
    a = roundoff * step
    b = truncation / step.square()
    valid = torch.isfinite(a) & torch.isfinite(b) & (a > 0) & (b > 0)
    raw = torch.where(valid, (a / (2.0 * b)).pow(1.0 / 3.0), step)
    eps = torch.finfo(truncation.dtype).eps
    floor = 32.0 * eps * x_scale
    # The factor bound prevents an order estimate from jumping far away from
    # the point when a smooth function has an accidentally tiny D4-D2 value.
    updated = torch.minimum(torch.maximum(raw, 0.1 * step), 10.0 * step)
    updated = torch.maximum(updated, floor)
    updated = torch.where(torch.isfinite(updated) & (updated > 0), updated, step)
    # Step selection is discrete in this experiment.  Detached spacing keeps
    # gradients through the selected function values while avoiding a hidden
    # derivative through the error-model argmin.
    return updated.detach()


def adaptive_central_derivative(
    function: ScalarFunction,
    x: Tensor,
    initial_step: Scalar = 1.0e-4,
    *,
    max_iterations: int = 1,
    return_info: bool = False,
):
    """Compute an adaptive central quartic derivative in Torch.

    The supplied spacing is the initial trial spacing, not a fixed output
    spacing.  One model-based update is used by default.  If the updated trial
    has a larger estimated error, the initial trial is retained independently
    for each broadcast row.  ``max_iterations`` can request additional
    updates, each still bounded by a factor of ten from the preceding trial.
    The returned derivative remains differentiable through the chosen stencil
    evaluations; ``step`` and error diagnostics are detached selection data.
    """
    if not isinstance(x, Tensor):
        raise TypeError("x must be a torch.Tensor")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    x = _as_float_tensor(x, like=x, name="x")
    _require_finite(x, "x")
    # Spacing is a hyperparameter and a discrete selection variable in this
    # experiment; do not expose a derivative through it even when supplied as
    # a tensor requiring grad.
    trial_step = _prepare_step(initial_step, like=x, name="initial_step").detach()

    initial = _sample(function, x, trial_step)
    best_derivative, best_t, best_r, best_error, best_step, _ = initial
    best_step = best_step.detach()
    best_t = best_t.detach()
    best_r = best_r.detach()
    best_error = best_error.detach()
    # Every trial evaluates all four nodes for every row, including a trial
    # that is later rejected.  Count actual function calls rather than only
    # accepted candidates so the diagnostic cost remains honest.
    evaluations = torch.full_like(best_error, 4, dtype=torch.long)

    # Each iteration evaluates a fresh quartic stencil at a detached spacing.
    # The best candidate is selected row by row; this is the only intentionally
    # nondifferentiable part of the method.
    for _ in range(max_iterations):
        trial_step = _adapted_step(x, trial_step, best_t, best_r)
        candidate = _sample(function, x, trial_step)
        cand_d, cand_t, cand_r, cand_error, cand_step, _ = candidate
        cand_step = cand_step.detach()
        cand_t = cand_t.detach()
        cand_r = cand_r.detach()
        cand_error = cand_error.detach()
        choose = cand_error < best_error
        best_derivative = torch.where(choose, cand_d, best_derivative)
        best_t = torch.where(choose, cand_t, best_t)
        best_r = torch.where(choose, cand_r, best_r)
        best_error = torch.where(choose, cand_error, best_error)
        best_step = torch.where(choose, cand_step, best_step)
        evaluations = evaluations + 4
        # The next model update starts from the selected step/estimates.  For
        # rejected rows this is the initial trial, so a noisy candidate cannot
        # silently move the reported derivative.
        trial_step = best_step

    result = NumericalDerivativeResult(
        derivative=best_derivative,
        error=best_error,
        truncation_error=best_t,
        roundoff_error=best_r,
        step=best_step,
        initial_step=torch.broadcast_to(
            _prepare_step(initial_step, like=x, name="initial_step"), best_step.shape
        ).detach(),
        evaluations=evaluations,
    )
    return result if return_info else result.derivative


def _state_inputs(hamiltonian, r, pr_star, pphi):
    if not hasattr(hamiltonian, "real_hamiltonian"):
        raise TypeError("hamiltonian must provide real_hamiltonian")
    if not hasattr(hamiltonian, "effective_hamiltonian"):
        raise TypeError("hamiltonian must provide effective_hamiltonian")
    if not hasattr(hamiltonian, "metric_potentials"):
        raise TypeError("hamiltonian must provide metric_potentials")
    reference = getattr(hamiltonian, "nu", None)
    if not isinstance(reference, Tensor):
        raise TypeError("hamiltonian.nu must be a torch.Tensor")
    r = _as_float_tensor(r, like=reference, name="r")
    pr_star = _as_float_tensor(pr_star, like=r, name="pr_star")
    pphi = _as_float_tensor(pphi, like=r, name="pphi")
    try:
        r, pr_star, pphi = torch.broadcast_tensors(r, pr_star, pphi)
    except RuntimeError as exc:
        raise ValueError("r, pr_star and pphi must broadcast") from exc
    _require_positive(r, "r")
    _require_finite(pr_star, "pr_star")
    _require_finite(pphi, "pphi")
    nu = _as_float_tensor(reference, like=r, name="hamiltonian.nu")
    _require_finite(nu, "hamiltonian.nu")
    if bool((nu < 0).any()):
        raise NumericalDerivativeDomainError(
            "hamiltonian.nu must be nonnegative", _row_indices(nu < 0))
    return r, pr_star, pphi, nu


def _normalized_real_hamiltonian(hamiltonian, r: Tensor, pr_star: Tensor,
                                 pphi: Tensor, nu: Tensor) -> Tensor:
    """Evaluate ``H_real/nu`` with a finite test-mass derivative limit."""
    _require_positive(r, "Hamiltonian radius")
    h_real = hamiltonian.real_hamiltonian(r, pr_star, pphi)
    if not isinstance(h_real, Tensor):
        raise TypeError("real_hamiltonian must return a torch.Tensor")
    h_real = _prepare_values(h_real, like=r, name="real Hamiltonian")
    h_real, nu_b = torch.broadcast_tensors(h_real, nu)
    if bool((nu_b == 0).all()):
        # H_real/nu is singular as a value at nu=0, but its coordinate
        # derivative has the effective-Hamiltonian limit.  This branch keeps
        # the useful Schwarzschild/Kerr diagnostic finite without fitting or
        # changing the positive-nu convention.
        value = hamiltonian.effective_hamiltonian(r, pr_star, pphi)
        return _prepare_values(value, like=r, name="effective Hamiltonian")

    safe_nu = torch.where(nu_b == 0, torch.ones_like(nu_b), nu_b)
    value = h_real / safe_nu
    if bool((nu_b == 0).any()):
        effective = hamiltonian.effective_hamiltonian(r, pr_star, pphi)
        effective = _prepare_values(effective, like=r, name="effective Hamiltonian")
        value = torch.where(nu_b == 0, effective, value)
    _require_finite(value, "normalized Hamiltonian")
    return value


def eob_cartesian_partials(
    hamiltonian,
    r: Tensor,
    pr_star: Tensor,
    pphi: Tensor,
    *,
    initial_step: Scalar = 1.0e-4,
    max_iterations: int = 1,
    return_info: bool = False,
):
    """Evaluate ``(H_x,H_px,H_py,xi)`` using the contract's orientation.

    ``H`` in the three scalar functions is the real Hamiltonian divided by
    ``nu``.  ``H_x`` varies the Cartesian position while holding Cartesian
    momenta fixed; ``H_px`` and ``H_py`` independently vary the two Cartesian
    momenta.  The output supports ordinary Torch gradients through each
    selected stencil.
    """
    with torch.inference_mode(False):
        hamiltonian = _ordinary_hamiltonian(hamiltonian)
        r, pr_star, pphi, nu = _state_inputs(hamiltonian, r, pr_star, pphi)
        p_y = pphi / r

        def h_x(x):
            return _normalized_real_hamiltonian(
                hamiltonian, x, pr_star, x * p_y, nu)

        def h_px(p_x):
            return _normalized_real_hamiltonian(
                hamiltonian, r, p_x, pphi, nu)

        def h_py(p_y_var):
            return _normalized_real_hamiltonian(
                hamiltonian, r, pr_star, r * p_y_var, nu)

        x_result = adaptive_central_derivative(
            h_x, r, initial_step, max_iterations=max_iterations, return_info=True)
        px_result = adaptive_central_derivative(
            h_px, pr_star, initial_step, max_iterations=max_iterations, return_info=True)
        py_result = adaptive_central_derivative(
            h_py, p_y, initial_step, max_iterations=max_iterations, return_info=True)

        xi = hamiltonian.metric_potentials(r)[2]
        if not isinstance(xi, Tensor):
            raise TypeError("metric_potentials must return a tensor xi")
        xi = _prepare_values(xi, like=r, name="tortoise factor")
        h_x_value = x_result.derivative
        h_px_value = px_result.derivative
        h_py_value = py_result.derivative
        h_x_value, h_px_value, h_py_value, xi = torch.broadcast_tensors(
            h_x_value, h_px_value, h_py_value, xi)
        partials = CartesianHamiltonianPartials(
            h_x=h_x_value, h_px=h_px_value, h_py=h_py_value, xi=xi)
        if return_info:
            return CartesianPartialReport(partials, x_result, px_result, py_result)
        return partials


def numerical_eob_conservative_derivatives(
    hamiltonian,
    r: Tensor,
    pr_star: Tensor,
    pphi: Tensor,
    *,
    initial_step: Scalar = 1.0e-4,
    max_iterations: int = 1,
    return_info: bool = False,
):
    """Return the finite-difference conservative EOB field.

    The chain rule is

    ``dr/dt = xi*H_px``, ``Omega = H_py/r``, and
    ``dp_r*/dt = xi*(-H_x + H_py*pphi/r**2)``.

    Radiation reaction is intentionally absent.  Set ``return_info=True`` to
    retain pointwise spacing and error diagnostics for all three partials.
    """
    with torch.inference_mode(False):
        hamiltonian = _ordinary_hamiltonian(hamiltonian)
        r, pr_star, pphi, _ = _state_inputs(hamiltonian, r, pr_star, pphi)
        report = eob_cartesian_partials(
            hamiltonian, r, pr_star, pphi,
            initial_step=initial_step,
            max_iterations=max_iterations,
            return_info=True,
        )
        partials = report.partials
        field = EOBConservativeField(
            dr_dt=partials.xi * partials.h_px,
            omega=partials.h_py / r,
            dpr_star_dt=partials.xi * (
                -partials.h_x + partials.h_py * pphi / r.square()
            ),
        )
        if not torch.is_grad_enabled():
            field = EOBConservativeField(*(value.detach() for value in field))
        if return_info:
            return EOBNumericalDerivativeReport(
                field, partials, report.x, report.p_x, report.p_y)
        return field


class NumericalEOBConservative:
    """Small opt-in wrapper exposing the numerical field as Hamiltonian APIs."""

    def __init__(self, hamiltonian, *, initial_step: Scalar = 1.0e-4,
                 max_iterations: int = 1):
        self.hamiltonian = hamiltonian
        self.initial_step = initial_step
        self.max_iterations = max_iterations

    def cartesian_partials(self, r, pr_star, pphi, *, return_info=False):
        return eob_cartesian_partials(
            self.hamiltonian, r, pr_star, pphi,
            initial_step=self.initial_step,
            max_iterations=self.max_iterations,
            return_info=return_info,
        )

    def equations_of_motion_conservative(self, r, pr_star, pphi,
                                         *, return_info=False):
        return numerical_eob_conservative_derivatives(
            self.hamiltonian, r, pr_star, pphi,
            initial_step=self.initial_step,
            max_iterations=self.max_iterations,
            return_info=return_info,
        )


# Descriptive aliases make the standalone experiment easy to discover while
# keeping the primary function names explicit.
central_quadratic_derivative = central_quadratic
central_quartic_derivative = central_quartic
eob_conservative_field = numerical_eob_conservative_derivatives
numerical_equations_of_motion_conservative = numerical_eob_conservative_derivatives


__all__ = [
    "NumericalDerivativeDomainError",
    "NumericalDerivativeResult",
    "CartesianHamiltonianPartials",
    "CartesianPartialReport",
    "EOBConservativeField",
    "EOBNumericalDerivativeReport",
    "NumericalEOBConservative",
    "central_quadratic_from_values",
    "central_quartic_from_values",
    "central_quadratic",
    "central_quartic",
    "central_quadratic_derivative",
    "central_quartic_derivative",
    "adaptive_central_derivative",
    "eob_cartesian_partials",
    "numerical_eob_conservative_derivatives",
    "eob_conservative_field",
    "numerical_equations_of_motion_conservative",
]

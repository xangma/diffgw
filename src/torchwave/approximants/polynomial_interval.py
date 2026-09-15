"""Small, device-native interval checks for low-order polynomials.

The XHM reconstruction uses a degree-five polynomial as a denominator.  A
fixed grid is not sufficient for that check: a polynomial can have a narrow
negative interval between two grid points.  This module isolates the real
roots of the derivative (recursively, using the derivative's extrema) and
checks the polynomial at every resulting candidate extremum and endpoint.

The answer is a discrete, detached decision.  The root locations are found
to floating-point accuracy; this is a numerical guard, rather than an exact
algebraic certificate.
"""

from __future__ import annotations

import torch


_MAX_DEGREE = 5
_BISECTION_STEPS = 48


def _evaluate(coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluate ascending-power coefficients with Horner's rule."""

    value = torch.zeros_like(x)
    for power in range(_MAX_DEGREE, -1, -1):
        term = coefficients[..., power]
        while term.ndim < x.ndim:
            term = term.unsqueeze(-1)
        value = value * x + term
    return value


def _absolute_scale(coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A Horner evaluation of ``sum(abs(c_k) abs(x)**k)``."""

    value = torch.zeros_like(x)
    absolute_x = x.abs()
    for power in range(_MAX_DEGREE, -1, -1):
        term = coefficients[..., power].abs()
        while term.ndim < x.ndim:
            term = term.unsqueeze(-1)
        value = value * absolute_x + term
    return value


def _derivative(coefficients: torch.Tensor) -> torch.Tensor:
    """Return derivative coefficients, padded to six ascending powers."""

    terms = torch.stack(
        [coefficients[..., power] * float(power) for power in range(1, _MAX_DEGREE + 1)],
        dim=-1,
    )
    return torch.cat((terms, torch.zeros_like(coefficients[..., :1])), dim=-1)


def _affine_transform(coefficients: torch.Tensor, center: torch.Tensor, half_width: torch.Tensor) -> torch.Tensor:
    """Express ``P(center + half_width*t)`` in ascending powers of ``t``."""

    transformed = torch.zeros_like(coefficients)
    for power in range(_MAX_DEGREE, -1, -1):
        next_coefficients = torch.zeros_like(transformed)
        next_coefficients[..., 0] = transformed[..., 0] * center + coefficients[..., power]
        for output_power in range(1, _MAX_DEGREE + 1):
            next_coefficients[..., output_power] = (
                transformed[..., output_power] * center
                + transformed[..., output_power - 1] * half_width
            )
        transformed = next_coefficients
    return transformed


def _roots_in_interval(
    coefficients: torch.Tensor,
    degree: int,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find all real roots in an interval, with a small padded root list.

    The derivative roots partition the interval into monotone pieces.  Each
    piece has at most one root, which is refined by bisection.  Roots at a
    derivative extremum are retained as well, covering repeated/tangent
    roots.  The returned list deliberately permits duplicate roots at segment
    boundaries; callers only evaluate them, so deduplication would add
    numerical complexity without changing the decision.
    """

    if degree <= 0:
        shape = coefficients.shape[:-1] + (0,)
        return coefficients.new_empty(shape), torch.zeros(shape, dtype=torch.bool, device=coefficients.device)

    derivative = _derivative(coefficients)
    if degree == 1:
        denominator = coefficients[..., 1]
        safe_denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        root = -coefficients[..., 0] / safe_denominator
        finite = torch.isfinite(root) & torch.isfinite(denominator)
        mask = finite & (denominator != 0) & (root >= lower) & (root <= upper)
        return root.unsqueeze(-1), mask.unsqueeze(-1)

    critical, critical_mask = _roots_in_interval(derivative, degree - 1, lower, upper)
    # Invalid roots are placed at the upper endpoint.  Their mask controls
    # whether the corresponding segment is active, and sorting keeps valid
    # roots at the front of the list without any host-side indexing.
    critical_safe = torch.where(critical_mask, critical, upper.unsqueeze(-1))
    critical_safe, order = torch.sort(critical_safe, dim=-1)
    critical_mask = torch.gather(critical_mask, -1, order)

    boundaries = torch.cat((lower.unsqueeze(-1), critical_safe, upper.unsqueeze(-1)), dim=-1)
    eps = torch.finfo(coefficients.dtype).eps
    left = boundaries[..., :-1]
    right = boundaries[..., 1:]
    active = torch.cat((torch.ones_like(left[..., :1], dtype=torch.bool), critical_mask), dim=-1)
    f_left = _evaluate(coefficients, left)
    f_right = _evaluate(coefficients, right)
    scale = torch.maximum(_absolute_scale(coefficients, left), _absolute_scale(coefficients, right))
    tolerance = (128.0 * eps) * torch.clamp(scale, min=torch.finfo(coefficients.dtype).tiny)
    left_zero = f_left.abs() <= tolerance
    right_zero = f_right.abs() <= tolerance
    sign_change = (f_left < 0) != (f_right < 0)
    segment_masks = active & (left_zero | right_zero | sign_change)

    # Bisection is batched over all monotone segments.  Inactive segments
    # still follow the same fixed loop, which keeps the operation entirely
    # device-native and bounds work independently of input data.
    bisect_left = left
    bisect_right = right
    bisect_f_left = f_left
    for _ in range(_BISECTION_STEPS):
        midpoint = bisect_left + (bisect_right - bisect_left) * 0.5
        f_midpoint = _evaluate(coefficients, midpoint)
        same_side = (bisect_f_left < 0) == (f_midpoint < 0)
        bisect_left = torch.where(same_side, midpoint, bisect_left)
        bisect_f_left = torch.where(same_side, f_midpoint, bisect_f_left)
        bisect_right = torch.where(same_side, bisect_right, midpoint)
    segment_roots = torch.where(
        left_zero, left, torch.where(right_zero, right, (bisect_left + bisect_right) * 0.5)
    )

    # A root which merely touches zero has no sign change.  Such a root is a
    # stationary point of this polynomial, so test the residual at the
    # recursively isolated critical points and carry qualifying points up.
    critical_residual = _evaluate(coefficients, critical_safe)
    critical_scale = _absolute_scale(coefficients, critical_safe)
    critical_tolerance = (256.0 * eps) * torch.clamp(
        critical_scale, min=torch.finfo(coefficients.dtype).tiny
    )
    repeated_mask = critical_mask & (critical_residual.abs() <= critical_tolerance)

    roots = torch.cat((segment_roots, critical_safe), dim=-1)
    masks = torch.cat((segment_masks, repeated_mask), dim=-1)
    roots = torch.where(masks, roots, upper.unsqueeze(-1))
    roots, order = torch.sort(roots, dim=-1)
    masks = torch.gather(masks, -1, order)
    return roots, masks


def polynomial_has_nonpositive(
    coefficients: torch.Tensor,
    lower: torch.Tensor | float,
    upper: torch.Tensor | float,
) -> torch.Tensor:
    """Return whether a polynomial is nonpositive anywhere on a closed interval.

    ``coefficients[..., k]`` is the coefficient of ``x**k`` and at most six
    coefficients (degree five) are accepted.  ``lower`` and ``upper`` and
    all leading batch dimensions broadcast.  The returned boolean tensor has
    the broadcast batch shape and is detached from coefficient gradients.

    Nonfinite inputs conservatively return ``True``.  Finite decisions use a
    machine-precision-scaled residual tolerance when evaluating numerically
    isolated stationary points; this handles repeated roots while avoiding a
    fixed relative floor such as the old five-percent heuristic.
    """

    coefficients = torch.as_tensor(coefficients)
    if coefficients.ndim == 0 or coefficients.shape[-1] == 0 or coefficients.shape[-1] > _MAX_DEGREE + 1:
        raise ValueError("coefficients must have a final dimension containing 1 through 6 powers")
    if coefficients.is_complex():
        raise TypeError("polynomial coefficients must be real")
    if not coefficients.is_floating_point():
        coefficients = coefficients.to(torch.get_default_dtype())
    if coefficients.dtype in (torch.float16, torch.bfloat16):
        coefficients = coefficients.float()
    degree = coefficients.shape[-1] - 1

    lower = torch.as_tensor(lower, device=coefficients.device, dtype=coefficients.dtype)
    upper = torch.as_tensor(upper, device=coefficients.device, dtype=coefficients.dtype)
    batch_shape = torch.broadcast_shapes(coefficients.shape[:-1], lower.shape, upper.shape)
    coefficients = coefficients.expand(batch_shape + (coefficients.shape[-1],))
    if coefficients.shape[-1] < _MAX_DEGREE + 1:
        coefficients = torch.nn.functional.pad(coefficients, (0, _MAX_DEGREE + 1 - coefficients.shape[-1]))
    lower = lower.expand(batch_shape)
    upper = upper.expand(batch_shape)

    with torch.no_grad():
        nonfinite = (~torch.isfinite(coefficients).all(dim=-1)) | ~torch.isfinite(lower) | ~torch.isfinite(upper)
        left = torch.minimum(lower, upper)
        right = torch.maximum(lower, upper)
        center = left * 0.5 + right * 0.5
        half_width = right * 0.5 - left * 0.5
        coefficients = _affine_transform(coefficients, center, half_width)
        nonfinite = nonfinite | ~torch.isfinite(coefficients).all(dim=-1)
        left = -torch.ones_like(center)
        right = torch.ones_like(center)

        values_left = _evaluate(coefficients, left)
        values_right = _evaluate(coefficients, right)
        result = (values_left <= 0) | (values_right <= 0)
        nonfinite = nonfinite | ~torch.isfinite(values_left) | ~torch.isfinite(values_right)

        derivative = _derivative(coefficients)
        stationary, stationary_mask = _roots_in_interval(derivative, degree - 1, left, right)
        stationary_values = _evaluate(coefficients, stationary)
        nonfinite = nonfinite | ~torch.isfinite(stationary_values).all(dim=-1)
        # Root bisection can leave a tangent extremum a few ulps from its
        # exact location (for x**2 this produces a tiny positive value).  The
        # interval was normalized to [-1, 1], so this scale tracks polynomial
        # amplitude without introducing a unit-sized absolute floor.
        coefficient_scale = coefficients.abs().amax(dim=-1, keepdim=True)
        value_scale = torch.maximum(
            coefficient_scale.expand_as(stationary_values),
            torch.full_like(stationary_values, torch.finfo(coefficients.dtype).tiny),
        )
        tolerance = (512.0 * torch.finfo(coefficients.dtype).eps) * value_scale
        result = result | torch.any(stationary_mask & (stationary_values <= tolerance), dim=-1)
        return (result | nonfinite).detach()


__all__ = ["polynomial_has_nonpositive"]

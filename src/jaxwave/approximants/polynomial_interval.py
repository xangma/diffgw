"""Small, device-native interval checks for low-order polynomials in JAX.

The XHM reconstruction uses a degree-five polynomial as a denominator.
This module isolates the real roots of the derivative (recursively, using
the derivative's extrema) and checks the polynomial at every resulting
candidate extremum and endpoint.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_MAX_DEGREE = 5


def _evaluate_fast(coeffs_u: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate ascending-power coefficients on points x."""
    v = coeffs_u[..., 5] * x + coeffs_u[..., 4]
    v = v * x + coeffs_u[..., 3]
    v = v * x + coeffs_u[..., 2]
    v = v * x + coeffs_u[..., 1]
    return v * x + coeffs_u[..., 0]


def _absolute_scale_fast(abs_coeffs_u: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Horner evaluation of sum(abs(c_k) * abs(x)**k)."""
    ax = jnp.abs(x)
    v = abs_coeffs_u[..., 5] * ax + abs_coeffs_u[..., 4]
    v = v * ax + abs_coeffs_u[..., 3]
    v = v * ax + abs_coeffs_u[..., 2]
    v = v * ax + abs_coeffs_u[..., 1]
    return v * ax + abs_coeffs_u[..., 0]


def _derivative(coefficients: jnp.ndarray) -> jnp.ndarray:
    """Return derivative coefficients, padded to six ascending powers."""
    terms = jnp.stack(
        [coefficients[..., power] * float(power) for power in range(1, _MAX_DEGREE + 1)],
        axis=-1,
    )
    return jnp.concatenate((terms, jnp.zeros_like(coefficients[..., :1])), axis=-1)


def _affine_transform(coefficients: jnp.ndarray, center: jnp.ndarray, half_width: jnp.ndarray) -> jnp.ndarray:
    """Express ``P(center + half_width*t)`` in ascending powers of ``t``."""
    transformed = jnp.zeros_like(coefficients)
    for power in range(_MAX_DEGREE, -1, -1):
        cols = [transformed[..., 0] * center + coefficients[..., power]]
        for output_power in range(1, _MAX_DEGREE + 1):
            cols.append(
                transformed[..., output_power] * center
                + transformed[..., output_power - 1] * half_width
            )
        transformed = jnp.stack(cols, axis=-1)
    return transformed


def _roots_in_interval(
    coefficients: jnp.ndarray,
    degree: int,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find all real roots in an interval, with a small padded root list."""
    if degree <= 0:
        shape = coefficients.shape[:-1] + (0,)
        return jnp.empty(shape, dtype=coefficients.dtype), jnp.zeros(shape, dtype=bool)

    if degree == 1:
        denominator = coefficients[..., 1]
        safe_denominator = jnp.where(denominator == 0, 1.0, denominator)
        root = -coefficients[..., 0] / safe_denominator
        finite = jnp.isfinite(root) & jnp.isfinite(denominator)
        mask = finite & (denominator != 0) & (root >= lower) & (root <= upper)
        return root[..., None], mask[..., None]

    if degree == 2:
        c0 = coefficients[..., 0]
        c1 = coefficients[..., 1]
        c2 = coefficients[..., 2]
        eps = jnp.finfo(coefficients.dtype).eps
        scale = jnp.maximum(jnp.abs(c0), jnp.maximum(jnp.abs(c1), jnp.abs(c2)))
        scale = jnp.maximum(scale, jnp.finfo(coefficients.dtype).tiny)
        tol = 256.0 * eps * scale

        is_linear = c2 == 0
        safe_c1 = jnp.where(c1 == 0, 1.0, c1)
        r_lin = -c0 / safe_c1
        m_lin = (c1 != 0) & jnp.isfinite(r_lin) & (r_lin >= lower) & (r_lin <= upper)

        safe_c2 = jnp.where(is_linear, 1.0, c2)
        disc = c1 * c1 - 4.0 * safe_c2 * c0
        disc_effective = jnp.maximum(disc, 0.0)
        has_roots = disc >= -tol * jnp.abs(safe_c2)

        sqrt_disc = jnp.sqrt(disc_effective)
        r1 = (-c1 - sqrt_disc) / (2.0 * safe_c2)
        r2 = (-c1 + sqrt_disc) / (2.0 * safe_c2)
        r_min = jnp.minimum(r1, r2)
        r_max = jnp.maximum(r1, r2)

        m1 = (~is_linear) & has_roots & jnp.isfinite(r_min) & (r_min >= lower) & (r_min <= upper)
        m2 = (~is_linear) & has_roots & jnp.isfinite(r_max) & (r_max >= lower) & (r_max <= upper)

        r_crit = -c1 / (2.0 * safe_c2)
        crit_val = c0 + c1 * r_crit + safe_c2 * r_crit * r_crit
        m_crit = (~is_linear) & jnp.isfinite(r_crit) & (r_crit >= lower) & (r_crit <= upper) & (jnp.abs(crit_val) <= tol)

        root1 = jnp.where(is_linear, r_lin, r_min)
        mask1 = jnp.where(is_linear, m_lin, m1 | m_crit)
        root2 = jnp.where(is_linear, upper, r_max)
        mask2 = jnp.where(is_linear, False, m2)

        roots = jnp.stack([root1, root2], axis=-1)
        masks = jnp.stack([mask1, mask2], axis=-1)
        roots = jnp.where(masks, roots, upper[..., None])
        order = jnp.argsort(roots, axis=-1)
        roots = jnp.take_along_axis(roots, order, axis=-1)
        masks = jnp.take_along_axis(masks, order, axis=-1)
        return roots, masks

    derivative = _derivative(coefficients)
    critical, critical_mask = _roots_in_interval(derivative, degree - 1, lower, upper)
    critical_safe = jnp.where(critical_mask, critical, upper[..., None])
    order = jnp.argsort(critical_safe, axis=-1)
    critical_safe = jnp.take_along_axis(critical_safe, order, axis=-1)
    critical_mask = jnp.take_along_axis(critical_mask, order, axis=-1)

    boundaries = jnp.concatenate((lower[..., None], critical_safe, upper[..., None]), axis=-1)
    eps = jnp.finfo(coefficients.dtype).eps
    left = boundaries[..., :-1]
    right = boundaries[..., 1:]
    active = jnp.concatenate((jnp.ones_like(left[..., :1], dtype=bool), critical_mask), axis=-1)

    coeffs_u = coefficients[..., None, :]
    abs_coeffs_u = jnp.abs(coefficients)[..., None, :]

    f_left = _evaluate_fast(coeffs_u, left)
    f_right = _evaluate_fast(coeffs_u, right)
    scale = jnp.maximum(_absolute_scale_fast(abs_coeffs_u, left), _absolute_scale_fast(abs_coeffs_u, right))
    tolerance = (128.0 * eps) * jnp.maximum(scale, jnp.finfo(coefficients.dtype).tiny)
    left_zero = jnp.abs(f_left) <= tolerance
    right_zero = jnp.abs(f_right) <= tolerance
    sign_change = (f_left < 0) != (f_right < 0)
    segment_masks = active & (left_zero | right_zero | sign_change)

    bisect_left = left
    bisect_right = right
    bisect_f_left = f_left
    bisection_steps = 28 if coefficients.dtype == jnp.float32 else 48
    for _ in range(bisection_steps):
        midpoint = bisect_left + (bisect_right - bisect_left) * 0.5
        f_midpoint = _evaluate_fast(coeffs_u, midpoint)
        same_side = (bisect_f_left < 0) == (f_midpoint < 0)
        bisect_left = jnp.where(same_side, midpoint, bisect_left)
        bisect_f_left = jnp.where(same_side, f_midpoint, bisect_f_left)
        bisect_right = jnp.where(same_side, bisect_right, midpoint)

    segment_roots = jnp.where(
        left_zero, left, jnp.where(right_zero, right, (bisect_left + bisect_right) * 0.5)
    )

    critical_residual = _evaluate_fast(coeffs_u, critical_safe)
    critical_scale = _absolute_scale_fast(abs_coeffs_u, critical_safe)
    critical_tolerance = (256.0 * eps) * jnp.maximum(
        critical_scale, jnp.finfo(coefficients.dtype).tiny
    )
    repeated_mask = critical_mask & (jnp.abs(critical_residual) <= critical_tolerance)

    roots = jnp.concatenate((segment_roots, critical_safe), axis=-1)
    masks = jnp.concatenate((segment_masks, repeated_mask), axis=-1)
    roots = jnp.where(masks, roots, upper[..., None])
    order = jnp.argsort(roots, axis=-1)
    roots = jnp.take_along_axis(roots, order, axis=-1)
    masks = jnp.take_along_axis(masks, order, axis=-1)
    return roots, masks


def polynomial_has_nonpositive(
    coefficients: jnp.ndarray,
    lower: jnp.ndarray | float,
    upper: jnp.ndarray | float,
) -> jnp.ndarray:
    """Return whether a polynomial is nonpositive anywhere on a closed interval.

    ``coefficients[..., k]`` is the coefficient of ``x**k`` and at most six
    coefficients (degree five) are accepted. ``lower`` and ``upper`` and
    all leading batch dimensions broadcast. The returned boolean array has
    the broadcast batch shape and is detached from coefficient gradients.
    """
    coefficients = jnp.asarray(coefficients)
    if coefficients.ndim == 0 or coefficients.shape[-1] == 0 or coefficients.shape[-1] > _MAX_DEGREE + 1:
        raise ValueError("coefficients must have a final dimension containing 1 through 6 powers")
    if not jnp.issubdtype(coefficients.dtype, jnp.floating):
        coefficients = coefficients.astype(jnp.float64)
    degree = coefficients.shape[-1] - 1

    lower = jnp.asarray(lower, dtype=coefficients.dtype)
    upper = jnp.asarray(upper, dtype=coefficients.dtype)
    batch_shape = jnp.broadcast_shapes(coefficients.shape[:-1], lower.shape, upper.shape)
    coefficients = jnp.broadcast_to(coefficients, batch_shape + (coefficients.shape[-1],))
    if coefficients.shape[-1] < _MAX_DEGREE + 1:
        pad_width = [(0, 0)] * len(batch_shape) + [(0, _MAX_DEGREE + 1 - coefficients.shape[-1])]
        coefficients = jnp.pad(coefficients, pad_width)
    lower = jnp.broadcast_to(lower, batch_shape)
    upper = jnp.broadcast_to(upper, batch_shape)

    nonfinite = (~jnp.all(jnp.isfinite(coefficients), axis=-1)) | ~jnp.isfinite(lower) | ~jnp.isfinite(upper)
    left = jnp.minimum(lower, upper)
    right = jnp.maximum(lower, upper)
    center = left * 0.5 + right * 0.5
    half_width = right * 0.5 - left * 0.5
    coefficients = _affine_transform(coefficients, center, half_width)
    nonfinite = nonfinite | ~jnp.all(jnp.isfinite(coefficients), axis=-1)

    coeffs_u = coefficients[..., None, :]
    minus_one = -jnp.ones_like(center)[..., None]
    plus_one = jnp.ones_like(center)[..., None]

    values_left = _evaluate_fast(coeffs_u, minus_one)[..., 0]
    values_right = _evaluate_fast(coeffs_u, plus_one)[..., 0]
    result = (values_left <= 0) | (values_right <= 0)
    nonfinite = nonfinite | ~jnp.isfinite(values_left) | ~jnp.isfinite(values_right)

    # Fast-path bound: on [-1, 1], P(t) >= c0 - sum(|ck|).
    c0 = coefficients[..., 0]
    tail_sum = jnp.sum(jnp.abs(coefficients[..., 1:]), axis=-1)
    eps = jnp.finfo(coefficients.dtype).eps
    tolerance_bound = 512.0 * eps * jnp.maximum(jnp.abs(c0), jnp.finfo(coefficients.dtype).tiny)
    strictly_positive = (c0 - tail_sum) > tolerance_bound

    resolved = result | strictly_positive | nonfinite
    derivative = _derivative(coefficients)
    stationary, stationary_mask = _roots_in_interval(
        derivative, degree - 1, -jnp.ones_like(center), jnp.ones_like(center)
    )
    stationary_values = _evaluate_fast(coeffs_u, stationary)
    nonfinite = nonfinite | ~jnp.all(jnp.isfinite(stationary_values), axis=-1)

    coefficient_scale = jnp.max(jnp.abs(coefficients), axis=-1, keepdims=True)
    value_scale = jnp.maximum(
        jnp.broadcast_to(coefficient_scale, stationary_values.shape),
        jnp.finfo(coefficients.dtype).tiny,
    )
    tolerance = (512.0 * jnp.finfo(coefficients.dtype).eps) * value_scale
    stationary_has_nonpos = jnp.any(stationary_mask & (stationary_values <= tolerance), axis=-1)
    final_result = jnp.where(resolved, result | nonfinite, result | stationary_has_nonpos | nonfinite)
    return jax.lax.stop_gradient(final_result)


__all__ = ["polynomial_has_nonpositive"]

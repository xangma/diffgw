"""Numerical SEOBNRv4 attachment conventions in JAX.

Provides NaturalCubicSpline, natural_cubic_spline, and Thomas-algorithm tridiagonal solve.
"""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp

from jaxwave.conversions import as_array


def _natural_second_derivatives(x: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """Return natural-cubic knot second derivatives using Thomas algorithm."""
    n = x.shape[-1]
    result = jnp.zeros_like(values)
    if n <= 2:
        return result

    h = x[..., 1:] - x[..., :-1]
    slopes = (values[..., 1:, :] - values[..., :-1, :]) / jnp.expand_dims(h, -1)
    m = n - 2
    lower = h[..., 1:-1]
    diagonal = 2.0 * (h[..., :-1] + h[..., 1:])
    upper = h[..., 1:-1]
    rhs = 6.0 * (slopes[..., 1:, :] - slopes[..., :-1, :])

    cprime = []
    dprime = []
    denominator = diagonal[..., 0]
    if m > 1:
        cprime.append(upper[..., 0] / denominator)
    dprime.append(rhs[..., 0, :] / jnp.expand_dims(denominator, -1))
    for j in range(1, m):
        denominator = diagonal[..., j] - lower[..., j - 1] * cprime[j - 1]
        if m > j + 1:
            cprime.append(upper[..., j] / denominator)
        dprime.append(
            (rhs[..., j, :] - jnp.expand_dims(lower[..., j - 1], -1) * dprime[-1])
            / jnp.expand_dims(denominator, -1)
        )

    interior = [None] * m
    interior[-1] = dprime[-1]
    for j in range(m - 2, -1, -1):
        interior[j] = dprime[j] - jnp.expand_dims(cprime[j], -1) * interior[j + 1]

    interior_stacked = jnp.stack(interior, axis=-2)
    result = result.at[..., 1:-1, :].set(interior_stacked)
    return result


@dataclass(frozen=True)
class NaturalCubicSpline:
    """A differentiable natural cubic interpolant on the last sample axis in JAX."""

    knots: jnp.ndarray
    values: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray
    d: jnp.ndarray
    scalar_output: bool

    @property
    def x(self) -> jnp.ndarray:
        return self.knots

    @property
    def y(self) -> jnp.ndarray:
        return self.values[..., 0] if self.scalar_output else self.values

    @property
    def channels(self) -> int:
        return self.values.shape[-1]

    def evaluate(self, query: jnp.ndarray, derivative: int = 0, *, extrapolate: bool = False) -> jnp.ndarray:
        """Evaluate value, first derivative, or second derivative."""
        q = as_array(query, dtype=self.knots.dtype)
        scalar_query = q.ndim == 0
        if scalar_query:
            q = q[None]

        batch = jnp.broadcast_shapes(self.knots.shape[:-1], q.shape[:-1])
        n = self.knots.shape[-1]
        channels = self.channels
        knots = jnp.broadcast_to(self.knots, batch + (n,))
        q = jnp.broadcast_to(q, batch + (q.shape[-1],))

        q_eval = q if extrapolate else jnp.clip(q, knots[..., :1], knots[..., -1:])
        interval = jnp.clip(jnp.searchsorted(knots, q_eval, side="right") - 1, 0, n - 2)

        dx = q_eval - jnp.take_along_axis(knots, interval, axis=-1)
        gather_index = jnp.expand_dims(interval, -1)

        a = jnp.take_along_axis(jnp.broadcast_to(self.a, batch + (n - 1, channels)), gather_index, axis=-2)
        b = jnp.take_along_axis(jnp.broadcast_to(self.b, batch + (n - 1, channels)), gather_index, axis=-2)
        c = jnp.take_along_axis(jnp.broadcast_to(self.c, batch + (n - 1, channels)), gather_index, axis=-2)
        d = jnp.take_along_axis(jnp.broadcast_to(self.d, batch + (n - 1, channels)), gather_index, axis=-2)
        dx = jnp.expand_dims(dx, -1)

        if derivative == 0:
            result = a + dx * (b + dx * (c + dx * d))
        elif derivative == 1:
            result = b + dx * (2.0 * c + dx * 3.0 * d)
        else:
            result = 2.0 * c + dx * 6.0 * d

        if self.scalar_output:
            result = result[..., 0]
        if scalar_query:
            result = result[..., 0] if self.scalar_output else result[..., 0, :]
        return result

    def derivative(self, query: jnp.ndarray, order: int = 1, *, extrapolate: bool = False) -> jnp.ndarray:
        return self.evaluate(query, derivative=order, extrapolate=extrapolate)

    __call__ = evaluate


def natural_cubic_spline(
    x: jnp.ndarray,
    values: jnp.ndarray,
    *,
    scalar_output: Optional[bool] = None,
) -> NaturalCubicSpline:
    """Construct a full-series natural cubic spline."""
    x = as_array(x, dtype=jnp.float64)
    values = as_array(values, dtype=jnp.float64)
    n = x.shape[-1]
    if x.ndim < 1 or n < 2:
        raise ValueError("natural cubic spline needs at least two knots")

    has_sample_axis = values.ndim >= 1 and values.shape[-1] == n
    has_channel_axis = values.ndim >= 2 and values.shape[-2] == n

    if scalar_output is True:
        if has_sample_axis:
            ordinate = jnp.expand_dims(values, -1)
        elif has_channel_axis and values.shape[-1] == 1:
            ordinate = values
        else:
            raise ValueError("scalar spline values must end in the knot axis")
    elif scalar_output is False:
        if not has_channel_axis:
            raise ValueError("channel spline values must end in (..., knots, channels)")
        ordinate = values
    elif values.ndim == x.ndim and has_sample_axis:
        ordinate = jnp.expand_dims(values, -1)
        scalar_output = True
    elif has_channel_axis:
        ordinate = values
        scalar_output = values.shape[-1] == 1
    elif has_sample_axis:
        ordinate = jnp.expand_dims(values, -1)
        scalar_output = True
    else:
        raise ValueError("spline values must end in knot axis")

    batch = jnp.broadcast_shapes(x.shape[:-1], ordinate.shape[:-2])
    channels = ordinate.shape[-1]
    knots = jnp.broadcast_to(x, batch + (n,))
    ordinate = jnp.broadcast_to(ordinate, batch + (n, channels))

    second = _natural_second_derivatives(knots, ordinate)
    h = knots[..., 1:] - knots[..., :-1]
    h_exp = jnp.expand_dims(h, -1)
    a = ordinate[..., :-1, :]
    b = (
        (ordinate[..., 1:, :] - ordinate[..., :-1, :]) / h_exp
        - h_exp * (2.0 * second[..., :-1, :] + second[..., 1:, :]) / 6.0
    )
    c = second[..., :-1, :] / 2.0
    d = (second[..., 1:, :] - second[..., :-1, :]) / (6.0 * h_exp)

    return NaturalCubicSpline(knots, ordinate, a, b, c, d, scalar_output)


__all__ = [
    "NaturalCubicSpline",
    "natural_cubic_spline",
]

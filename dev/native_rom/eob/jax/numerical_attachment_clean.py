"""Development-only scalar numerical attachment conventions for native v4."""

import jax.numpy as jnp

from jaxwave.approximants.eob.events import OrbitalFrequencyPeak
from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline


def full_orbital_peak(time, frequency) -> OrbitalFrequencyPeak:
    """Apply the full-spline/sign-product peak rule used by the Torch route.

    The first non-increasing sample selects an adjacent interval. Bisection
    follows a strictly negative slope product; zero belongs to the left move.
    This is a scalar, eager numerical policy, not a JIT event detector.
    """
    time = jnp.asarray(time)
    frequency = jnp.asarray(frequency)
    if time.ndim != 1 or frequency.shape != time.shape or len(time) < 4:
        raise ValueError("peak needs matching scalar grids of at least four samples")
    if not bool(jnp.all(jnp.isfinite(time)) & jnp.all(jnp.isfinite(frequency))):
        raise ValueError("peak grids must be finite")
    if not bool(jnp.all(jnp.diff(time) > 0)):
        raise ValueError("peak time grid must be strictly increasing")
    decreasing = frequency[1:] <= frequency[:-1]
    if not bool(jnp.any(decreasing)):
        raise ValueError("orbital-frequency peak is unresolved: no non-increasing sample")
    index = int(jnp.argmax(decreasing)) + 1
    spline = natural_cubic_spline(time, frequency, scalar_output=True)
    following = bool(spline.derivative(time[index]) > 0)
    if following and index + 1 >= len(time):
        raise ValueError("orbital-frequency peak has no following interval")
    left = time[index if following else index - 1]
    right = time[index + 1 if following else index]
    left_slope = spline.derivative(left)
    middle = (left + right) / 2
    middle_slope = spline.derivative(middle)
    last_middle = middle
    while float(right - left) > 1e-5:
        if bool(left_slope * middle_slope < 0):
            right = middle
        else:
            left = middle
            left_slope = middle_slope
        last_middle = middle
        if float(right - left) > 1e-5:
            middle = (left + right) / 2
            middle_slope = spline.derivative(middle)
    peak = last_middle
    return OrbitalFrequencyPeak(peak, spline(peak), spline.derivative(peak, 2),
                                jnp.asarray(index))


__all__ = ["full_orbital_peak"]

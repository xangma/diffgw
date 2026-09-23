"""Published pre-NQC positive-m flux sum, independent of mode generation.

Bohé et al. 1611.03703v3 Eq. (9). This normalization does not supply the
historical SEOBNRv4 mode amplitudes or a complete radiation reaction.
"""

from collections.abc import Mapping
import math

import jax
import jax.numpy as jnp


def emitted_flux_from_modes(omega, positive_m_modes: Mapping[tuple[int, int], object]):
    """Return ``sum((m*omega)**2 * |R*h_lm/M|**2)/(8*pi)``.

    Inputs broadcast, and ``omega`` is the dimensionless orbital frequency.
    Concrete invalid inputs raise; compiled invalid rows return NaN. Supply
    only pre-NQC, positive-m modes with ``2<=ell<=8``. A partial mapping gives
    a partial flux, not the full SEOBNRv4 torque.
    """
    if not isinstance(positive_m_modes, Mapping) or not positive_m_modes:
        raise ValueError("at least one positive-m mode is required")
    for mode in positive_m_modes:
        if (not isinstance(mode, tuple) or len(mode) != 2
                or any(type(index) is not int for index in mode)
                or not 2 <= mode[0] <= 8 or not 1 <= mode[1] <= mode[0]):
            raise ValueError("mode keys must satisfy 2<=ell<=8 and 1<=m<=ell")

    frequency = jnp.asarray(omega)
    if frequency.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("omega must have float32 or float64 dtype")
    amplitudes = {
        mode: jnp.asarray(value, dtype=jnp.result_type(frequency.dtype, value))
        for mode, value in positive_m_modes.items()
    }
    shapes = jnp.broadcast_arrays(frequency, *amplitudes.values())
    frequency = shapes[0]
    amplitudes = dict(zip(amplitudes, shapes[1:]))
    valid = jnp.isfinite(frequency) & (frequency > 0)
    total = jnp.zeros_like(frequency)
    for (_, m), amplitude in amplitudes.items():
        valid &= jnp.isfinite(amplitude)
        total = total + (m*frequency)**2 * (jnp.real(amplitude)**2 + jnp.imag(amplitude)**2)
    if not isinstance(valid, jax.core.Tracer) and not bool(jnp.all(valid)):
        raise ValueError("frequency must be positive and all inputs finite")
    return jnp.where(valid, total/(8*math.pi), jnp.nan)


__all__ = ["emitted_flux_from_modes"]

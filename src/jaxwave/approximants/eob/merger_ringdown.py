"""Published SEOBNRv4 merger-ringdown ansatz in JAX, Bohé et al. 1611.03703v3.

Eqs. (2.10), (2.12)-(2.17), with constrained coefficients derived from C1 matching.
"""

from typing import NamedTuple
import jax.numpy as jnp

from jaxwave.conversions import as_array


class MergerRingdownShape(NamedTuple):
    amplitude_rate: jnp.ndarray
    amplitude_offset: jnp.ndarray
    phase_rate: jnp.ndarray
    phase_offset: jnp.ndarray


def seobnrv4_merger_shape(nu: jnp.ndarray, chi: jnp.ndarray) -> MergerRingdownShape:
    """Free coefficients c1f, c2f, d1f, d2f, Eqs. (2.13), (2.14), (2.16), (2.17)."""
    nu = as_array(nu, dtype=jnp.float64)
    chi = as_array(chi, dtype=jnp.float64)
    nu, chi = jnp.broadcast_arrays(nu, chi)

    c1 = (
        -0.0893454 * (nu**2)
        + 0.0612892 * nu
        + 0.00146142 * nu * chi
        - 0.0136459 * (chi**2)
        - 0.0196758 * chi
        + 0.0830664
    )
    c2 = (
        -1.82173 * (nu**2)
        - 5.25339 * (nu**2) * chi
        + 2.40203 * nu * chi
        + 1.39777 * nu
        - 0.371365 * chi
        - 0.623953
    )
    d1 = (
        -0.808987 * (nu**2)
        + 0.263456 * nu
        - 0.120853 * nu * chi
        - 0.0244358 * (chi**2)
        + 0.00779176 * chi
        + 0.147584
    )
    d2 = (
        17.5646 * (nu**2)
        - 6.99396 * nu
        - 9.61861 * nu * chi
        + 0.581626 * (chi**2)
        + 3.13067 * chi
        + 2.46654
    )
    return MergerRingdownShape(c1, c2, d1, d2)


def seobnrv4_peak_preserving_shape(sigma: jnp.ndarray, shape: MergerRingdownShape) -> MergerRingdownShape:
    """Project c1f to the Eq. (2.12) peak boundary when its fit rises there.

    From Bohé et al. (1611.03703v3), Eqs. (2.10), (2.12)-(2.14),
    A''(0)/A(0)=-gamma*[gamma+2*c1f*tanh(c2f)] for A'(0)=0.
    This is the ansatz-derived guard in historical SEOBNRv4 ringdown source
    (LALSimIMREOBHybridRingdown.c, Eqs. (57)-(58) of LIGO-T1600383).
    """
    c1, c2, d1, d2 = shape
    damping = -jnp.imag(sigma)
    offset = jnp.tanh(c2)
    safe_offset = jnp.where(offset < 0, offset, jnp.ones_like(offset))
    boundary_rate = -damping / (2 * safe_offset)
    rate = jnp.where(damping + 2 * c1 * offset < 0, boundary_rate, c1)
    return MergerRingdownShape(rate, c2, d1, d2)


def _log1p_over_x(x: jnp.ndarray) -> jnp.ndarray:
    eps = jnp.finfo(x.dtype).eps
    small = jnp.abs(x) < (eps**0.2)
    safe = jnp.where(small, 1.0, x)
    polynomial = 1.0 + x * (-0.5 + x * (1.0 / 3.0 + x * (-0.25 + x * 0.2)))
    return jnp.where(small, polynomial, jnp.log1p(x) / safe)


def merger_ringdown_22(
    time_since_match: jnp.ndarray,
    sigma: jnp.ndarray,
    amplitude: jnp.ndarray,
    amplitude_derivative: jnp.ndarray,
    frequency: jnp.ndarray,
    phase: jnp.ndarray,
    shape: MergerRingdownShape,
) -> jnp.ndarray:
    """C1 matched complex h22 with convention h=A*exp(-i*phase)."""
    t_in = as_array(time_since_match, dtype=jnp.float64)
    sigma_in = as_array(sigma, dtype=jnp.complex128)
    a0_in = as_array(amplitude, dtype=jnp.float64)
    adot0_in = as_array(amplitude_derivative, dtype=jnp.float64)
    w0_in = as_array(frequency, dtype=jnp.float64)
    phi0_in = as_array(phase, dtype=jnp.float64)

    w_val = sigma_in.real
    damping_val = -sigma_in.imag

    params = jnp.broadcast_arrays(
        w_val, damping_val, a0_in, adot0_in, w0_in, phi0_in,
        shape.amplitude_rate, shape.amplitude_offset, shape.phase_rate, shape.phase_offset,
    )
    batch = jnp.broadcast_shapes(t_in.shape[:-1], params[0].shape)
    t = jnp.broadcast_to(t_in, batch + (t_in.shape[-1],))

    w, damping, a0, adot0, w0, phi0, c1, c2, d1, d2 = (
        jnp.expand_dims(jnp.broadcast_to(p, batch), -1) for p in params
    )

    transition = jnp.tanh(c1 * t)
    rescaled_amplitude = a0 + (adot0 + damping * a0) / c1 * transition / (1.0 + transition * jnp.tanh(c2))

    exponential_change = jnp.expm1(-d1 * t)
    log_argument_change = d2 / (1.0 + d2) * exponential_change
    residual_phase = (w - w0) / d1 * exponential_change * _log1p_over_x(log_argument_change)
    total_phase = phi0 + w * t + residual_phase

    return rescaled_amplitude * jnp.exp(-damping * t - 1j * total_phase)


__all__ = [
    "MergerRingdownShape",
    "seobnrv4_merger_shape",
    "seobnrv4_peak_preserving_shape",
    "merger_ringdown_22",
]

"""Direct PN/Kerr hybrid MECO in JAX; no waveform fits or calibration data.

Cabero et al., https://arxiv.org/html/1602.03134v3, Eqs. (2), (5)--(7),
(23)--(26); Pratten et al., https://arxiv.org/html/2001.11412v1,
Eqs. (48)--(49). This implements Cabero's black-hole hybrid convention:
mass-weighted effective spin, spin-orbit and spin-spin terms, no spin-cubic
term. It is not certified equivalent to the inherited XAS MECO fit.
"""

import math
from typing import Tuple, Union
import jax
import jax.numpy as jnp

from jaxwave.conversions import as_array

Real = Union[float, jnp.ndarray]


def _photon_velocity(chi: jnp.ndarray) -> jnp.ndarray:
    # Physical exterior branch of D=0: r^3-6r^2+9r-4chi^2=0,
    # with its sign fixed by 1-3/r+2chi/r^(3/2)=0.
    radius = 2.0 * (1.0 + jnp.cos((2.0 / 3.0) * jnp.arccos(-chi)))
    return (radius ** 1.5 + chi) ** (-1.0 / 3.0)


def _parameters(mass1, mass2, chi1, chi2, pn_order=3.5):
    scale = jnp.maximum(mass1, mass2)
    u1, u2 = mass1 / scale, mass2 / scale
    x1, x2 = u1 / (u1 + u2), u2 / (u1 + u2)
    eta = x1 * x2
    delta = (u1 - u2) / (u1 + u2)
    chi = (chi1 + chi2) / 2.0 + delta * ((chi1 - chi2) / 2.0)
    spin_sum = chi1 + chi2
    spin = x1**2 * chi1 + x2**2 * chi2
    delta_sigma = (x1 - x2) * (x2 * chi2 - x1 * chi1)

    c2 = -eta / 12.0
    c3 = -2.0 * eta * spin_sum / 3.0
    c4 = eta * (19.0 / 8.0 - eta / 24.0)
    c5 = -eta * (5.0 * spin_sum + 61.0 * spin / 9.0 + 10.0 * delta_sigma / 3.0)
    ss3_residual = 5.0 * eta / 18.0 * (
        x1 * (7.0 * x1 + 22.0) * chi1**2
        + x2 * (7.0 * x2 + 22.0) * chi2**2
        + 2.0 * (10.0 - eta) * chi1 * chi2
    )
    c6 = eta * (34445.0 / 576.0 - 205.0 * math.pi**2 / 96.0
                - 155.0 * eta / 96.0 - 35.0 * eta**2 / 5184.0) + ss3_residual
    c7 = eta * (-81.0 * spin_sum / 4.0 - 367.0 * spin / 4.0 - 39.0 * delta_sigma
                + eta * (29.0 * spin / 12.0 + 5.0 * delta_sigma / 4.0))
    c7 = jnp.where(pn_order == 3, jnp.zeros_like(c7), c7)
    return chi, (c2, c3, c4, c5, c6, c7)


def _energy(v, chi, coefficients, derivative=0):
    u = 1.0 - chi * v**3
    w = v**2 * u**(-2.0 / 3.0)
    sqrt_w = jnp.sqrt(w)
    denominator = 1.0 - 3.0 * w + 2.0 * chi * w * sqrt_w
    if derivative == 0:
        numerator = 1.0 - 2.0 * w + chi * w * sqrt_w
        rad = jnp.sqrt(denominator)
        kerr = (-w + 4.0 * w**2 - 4.0 * chi * w**2 * sqrt_w
                + chi**2 * w**3) / (rad * (numerator + rad))
    elif derivative == 1:
        stability = 1.0 - 6.0 * w + 8.0 * chi * w * sqrt_w - 3.0 * chi**2 * w**2
        dw = 2.0 * v * u**(-5.0 / 3.0)
        kerr = -0.5 * dw * stability * denominator**(-1.5)
    else:
        stability = 1.0 - 6.0 * w + 8.0 * chi * w * sqrt_w - 3.0 * chi**2 * w**2
        dw = 2.0 * v * u**(-5.0 / 3.0)
        ddw = 2.0 * u**(-5.0 / 3.0) + 10.0 * chi * v**3 * u**(-8.0 / 3.0)
        d_stability = -6.0 + 12.0 * chi * sqrt_w - 6.0 * chi**2 * w
        d_denominator = -3.0 + 3.0 * chi * sqrt_w
        kerr = -0.5 * denominator**(-1.5) * (
            ddw * stability + dw**2 * d_stability
            - 1.5 * dw**2 * stability * d_denominator / denominator
        )
    residual = jnp.zeros_like(v)
    for power, coefficient in enumerate(coefficients, start=4):
        factor = 1.0 if derivative == 0 else float(power)
        if derivative == 2:
            factor *= float(power - 1)
        residual = residual + factor * coefficient * v**(power - derivative)
    return kerr - 0.5 * residual


def hybrid_binding_energy(
    velocity: Real,
    mass1: Real,
    mass2: Real,
    chi1: Real = 0.0,
    chi2: Real = 0.0,
    *,
    pn_order: float = 3.5,
    derivative: int = 0,
) -> jnp.ndarray:
    """Energy per reduced mass, or its first/second velocity derivative in JAX."""
    v = as_array(velocity, dtype=jnp.float64)
    m1 = as_array(mass1, dtype=jnp.float64)
    m2 = as_array(mass2, dtype=jnp.float64)
    s1 = as_array(chi1, dtype=jnp.float64)
    s2 = as_array(chi2, dtype=jnp.float64)

    chi, coefficients = _parameters(m1, m2, s1, s2, pn_order)
    return _energy(v, chi, coefficients, derivative)


def hybrid_meco_velocity(
    mass1: Real,
    mass2: Real,
    chi1: Real = 0.0,
    chi2: Real = 0.0,
    *,
    pn_order: float = 3.5,
    scan_points: int = 256,
    bisection_steps: int = 64,
) -> jnp.ndarray:
    """First stable minimum of the hybrid energy, in PN velocity units."""
    m1 = as_array(mass1, dtype=jnp.float64)
    m2 = as_array(mass2, dtype=jnp.float64)
    s1 = as_array(chi1, dtype=jnp.float64)
    s2 = as_array(chi2, dtype=jnp.float64)

    chi, coefficients = _parameters(m1, m2, s1, s2, pn_order)
    upper = (1.0 - 1e-5) * _photon_velocity(chi)
    fractions = jnp.linspace(1e-4, 1.0, scan_points, dtype=jnp.float64)
    grid = jnp.expand_dims(upper, -1) * fractions
    slope = _energy(grid, jnp.expand_dims(chi, -1), tuple(jnp.expand_dims(c, -1) for c in coefficients), 1)
    crossings = (slope[..., :-1] < 0) & (slope[..., 1:] >= 0)
    index = jnp.argmax(crossings.astype(jnp.int32), axis=-1, keepdims=True)
    low = jnp.take_along_axis(grid, index, axis=-1).squeeze(-1)
    high = jnp.take_along_axis(grid, index + 1, axis=-1).squeeze(-1)

    def bisect_step(i, state):
        l, h = state
        mid = 0.5 * (l + h)
        neg = _energy(mid, chi, coefficients, 1) < 0
        l_new = jnp.where(neg, mid, l)
        h_new = jnp.where(neg, h, mid)
        return l_new, h_new

    low, high = jax.lax.fori_loop(0, bisection_steps, bisect_step, (low, high))
    velocity = 0.5 * (low + high)

    def newton_step(i, v):
        e1 = _energy(v, chi, coefficients, 1)
        e2 = _energy(v, chi, coefficients, 2)
        return v - e1 / e2

    velocity = jax.lax.fori_loop(0, 3, newton_step, velocity)
    return velocity


def hybrid_meco_frequency(
    mass1: Real,
    mass2: Real,
    chi1: Real = 0.0,
    chi2: Real = 0.0,
    **kwargs,
) -> jnp.ndarray:
    """Dimensionless dominant-mode frequency M*f_22 = v_MECO**3/pi."""
    return hybrid_meco_velocity(mass1, mass2, chi1, chi2, **kwargs) ** 3 / math.pi

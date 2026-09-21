"""JAX orbit-averaged N4LO spin evolution; see the Torch twin for details."""
from __future__ import annotations

import math

import jax.numpy as jnp


def spin_derivatives(v, lhat, s1, s2, m1_fraction, m2_fraction):
    """Return ``(dLhat/d(t/M), dS1/d(t/M), dS2/d(t/M))`` through N4LO."""
    lhat, s1, s2 = jnp.broadcast_arrays(
        jnp.asarray(lhat), jnp.asarray(s1), jnp.asarray(s2)
    )
    lead = lhat.shape[:-1]
    dtype = lhat.dtype
    v = jnp.broadcast_to(jnp.asarray(v, dtype=dtype), lead)[..., None]
    m1 = jnp.broadcast_to(jnp.asarray(m1_fraction, dtype=dtype), lead)[..., None]
    m2 = jnp.broadcast_to(jnp.asarray(m2_fraction, dtype=dtype), lead)[..., None]

    eta = m1 * m2
    l1 = jnp.sum(lhat * s1, axis=-1, keepdims=True)
    l2 = jnp.sum(lhat * s2, axis=-1, keepdims=True)
    c1 = 1.5 - m1 - 0.5 * m1 * m1
    c2 = 1.5 - m2 - 0.5 * m2 * m2
    x1 = jnp.cross(lhat, s1)
    x2 = jnp.cross(lhat, s2)
    s12 = jnp.cross(s1, s2)

    ds1 = c1 * v**5 * x1
    ds2 = c2 * v**5 * x2
    ds1 = ds1 + v**6 * (-0.5 * s12 - 1.5 * l2 * x1 + 1.5 * (1.0 - 1.0 / m1) * l1 * x1)
    ds2 = ds2 + v**6 * (0.5 * s12 - 1.5 * l1 * x2 + 1.5 * (1.0 - 1.0 / m2) * l2 * x2)

    # NNLO spin-orbit, Sturani Eq. (22).
    q1 = 9 / 8 - m1 / 2 + 7 * m1**2 / 12 - 7 * m1**3 / 6 - m1**4 / 24
    q2 = 9 / 8 - m2 / 2 + 7 * m2**2 / 12 - 7 * m2**3 / 6 - m2**4 / 24
    ds1 = ds1 + q1 * v**7 * x1
    ds2 = ds2 + q2 * v**7 * x2

    # N3LO orbit-averaged quadratic spin, Sturani Eq. (24), kappa=1.
    a1 = -13 / 4 + 9 * m1 / 4 + m1**2
    a2 = -13 / 4 + 9 * m2 / 4 + m2**2
    b1 = m1 / 2 + m1**2 / 2
    b2 = m2 / 2 + m2**2 / 2
    d1 = -3 / 2 - 3 * m1 / 2 - m1**2 / 2
    d2 = -3 / 2 - 3 * m2 / 2 - m2**2 / 2
    ds1 = ds1 + v**8 * jnp.cross(a1 * l1 * lhat + b1 * s2 + d1 * l2 * lhat, s1)
    ds2 = ds2 + v**8 * jnp.cross(a2 * l2 * lhat + b2 * s1 + d2 * l1 * lhat, s2)

    # N4LO spin-orbit, Sturani Eq. (25).
    r1 = 27 / 16 - 51 * m1 / 8 + 181 * m1**2 / 16 - 23 * m1**3 / 6 - 39 * m1**4 / 16 - 3 * m1**5 / 8 - m1**6 / 48
    r2 = 27 / 16 - 51 * m2 / 8 + 181 * m2**2 / 16 - 23 * m2**3 / 6 - 39 * m2**4 / 16 - 3 * m2**5 / 8 - m2**6 / 48
    ds1 = ds1 + r1 * v**9 * x1
    ds2 = ds2 + r2 * v**9 * x2

    # TP intentionally disables higher-order linear-in-spin corrections to L;
    # use nonspin 3PN L with the projected Eq. 13 constraint below.
    L = eta / jnp.maximum(v, jnp.finfo(dtype).tiny) * (
        1
        + (1.5 + eta / 6) * v**2
        + (27 / 8 - 19 * eta / 8 + eta**2 / 24) * v**4
        + (135 / 16 + (-6889 / 144 + 41 * math.pi**2 / 24) * eta + 31 * eta**2 / 24 + 7 * eta**3 / 1296) * v**6
    )
    total = ds1 + ds2
    l2norm = jnp.maximum(jnp.sum(lhat * lhat, axis=-1, keepdims=True), jnp.finfo(dtype).tiny)
    dl = -(total - lhat * jnp.sum(lhat * total, axis=-1, keepdims=True) / l2norm) / jnp.maximum(L, jnp.finfo(dtype).tiny)
    return dl, ds1, ds2


__all__ = ["spin_derivatives"]

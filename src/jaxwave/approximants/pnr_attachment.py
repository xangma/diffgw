"""JAX XPNR angle joins from Hamilton et al. (2025), Eqs. (17)--(20).

These component kernels do not choose SpinTaylor stop events or change public
XPNR dispatch. Their boundary data must come from a separately qualified PN
trajectory and merger-ringdown ansatz.
"""

from __future__ import annotations

import jax.numpy as jnp


def _hermite(t):
    return (2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2)


def alpha_attachment(mf, start, end, value_start, slope_start, value_end, slope_end):
    """Eq. (19): ``a0*f² + a1*f + a2 + a3/f`` with C1 joins."""
    width = end - start
    h00, h10, h01, h11 = _hermite((mf - start) / width)
    q0, q1 = start * value_start, end * value_end
    dq0, dq1 = value_start + start * slope_start, value_end + end * slope_end
    return (h00 * q0 + h10 * width * dq0 + h01 * q1 + h11 * width * dq1) / mf


def beta_connection(mf, start, end, value_start, slope_start, value_end, slope_end):
    """Eq. (20): cubic joining PN and merger-ringdown beta with C1 joins."""
    width = end - start
    h00, h10, h01, h11 = _hermite((mf - start) / width)
    return h00 * value_start + h10 * width * slope_start + h01 * value_end + h11 * width * slope_end


def alpha_exterior_continuation(mf, start, end, value_start, value_end, slope_end):
    """Eq. (17): quadratic in ``1/f²`` from two PN values and one slope."""
    u, u0, u1 = mf**-2, start**-2, end**-2
    derivative_u1 = -0.5 * end**3 * slope_end
    delta = u0 - u1
    curvature = (value_start - value_end - derivative_u1 * delta) / delta**2
    return value_end + derivative_u1 * (u - u1) + curvature * (u - u1) ** 2


def beta_exterior_continuation(
    mf, start, end, value_start, value_end, slope_end, beta_ringdown_value, kappa,
):
    """Eq. (18): exponential/rational continuation from two PN values and slope."""
    u, u0, u1 = 1 / mf, 1 / start, 1 / end
    r0 = (value_start - beta_ringdown_value) * jnp.exp(kappa * (start - end)) * start**2
    r1 = (value_end - beta_ringdown_value) * end**2
    derivative_u1 = -end**4 * (
        slope_end + (value_end - beta_ringdown_value) * (kappa + 2 / end)
    )
    delta = u0 - u1
    curvature = (r0 - r1 - derivative_u1 * delta) / delta**2
    r = r1 + derivative_u1 * (u - u1) + curvature * (u - u1) ** 2
    return beta_ringdown_value + jnp.exp(-kappa * (mf - end)) * r / mf**2

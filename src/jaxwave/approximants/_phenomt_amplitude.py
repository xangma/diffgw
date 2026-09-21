"""Differentiable IMRPhenomT amplitude ansatz (JAX port).

The algebra is shared with ``torchwave.approximants._phenomt_amplitude`` and
comes from the MIT licensed author reference listed in THIRD_PARTY_NOTICES.
"""

import math

import jax.numpy as jnp

from diffgw_core import phenomt_fits as fits
from diffgw_core.constants import EULER_GAMMA


def _pn_coefficients(eta, delta, chi1, chi2):
    chis, chia = (chi1 + chi2) / 2, (chi1 - chi2) / 2
    s0 = ((1 + delta) * chi1 + (1 - delta) * chi2) / 2
    real = (
        -107 / 42 + 55 * eta / 42,
        -4 * chis / 3 - 4 * chia * delta / 3 + 4 * chis * eta / 3 + 2 * math.pi,
        -2173 / 1512 - 1069 * eta / 216 + 2047 * eta**2 / 1512 + s0**2,
        -107 * math.pi / 21 + 34 * eta * math.pi / 21,
        27027409 / 646800
        - 278185 * eta / 33264
        - 20261 * eta**2 / 2772
        + 114635 * eta**3 / 99792
        - 856 * EULER_GAMMA / 105
        + 2 * math.pi**2 / 3
        + 41 * eta * math.pi**2 / 96,
        -2173 * math.pi / 756 - 2495 * eta * math.pi / 378 + 40 * eta**2 * math.pi / 27,
    )
    imag = (-24 * eta, 428 * math.pi / 105 + eta * 0, 14333 * eta / 162 - 4066 * eta**2 / 945)
    return real, imag, 2 * eta * math.sqrt(16 * math.pi / 5)


def _pn(x, pn, pseudo, derivative=False):
    real, imag, fac = pn
    r, i = jnp.ones_like(x), jnp.zeros_like(x)
    dr, di = jnp.zeros_like(x), jnp.zeros_like(x)
    for power, coeff in zip((1, 1.5, 2, 2.5, 3, 3.5), real):
        r = r + coeff[..., None] * x**power
        dr = dr + power * coeff[..., None] * x ** (power - 1)
    for power, coeff in zip((2.5, 3, 3.5), imag):
        i = i + coeff[..., None] * x**power
        di = di + power * coeff[..., None] * x ** (power - 1)
    logarithm = jnp.log(16 * x)
    r = r - 428 / 105 * logarithm * x**3
    dr = dr - 428 / 105 * (3 * logarithm + 1) * x**2
    for idx, power in enumerate((4, 4.5, 5)):
        r = r + pseudo[..., idx, None] * x**power
        dr = dr + power * pseudo[..., idx, None] * x ** (power - 1)
    value = fac[..., None] * x * (r + 1j * i)
    if derivative:
        return value, fac[..., None] * (r + x * dr + 1j * (i + x * di))
    return value


def _merger_basis(t, alpha, derivative=False):
    # Broadcast the time grid before stacking so scalar, batched, and vmap
    # inputs all produce a common (..., ntime, 4) basis shape.
    phi, tb = jnp.broadcast_arrays(alpha[..., None] * t, t)
    sech1, sech2 = 1 / jnp.cosh(phi), 1 / jnp.cosh(2 * phi)
    if derivative:
        return jnp.stack(
            (
                jnp.zeros_like(phi),
                -alpha[..., None] * sech1 * jnp.tanh(phi),
                -2 / 7 * alpha[..., None] * jnp.tanh(2 * phi) * sech2 ** (1 / 7),
                2 * tb,
            ),
            axis=-1,
        )
    return jnp.stack((jnp.ones_like(phi), sech1, sech2 ** (1 / 7), tb**2), axis=-1)


def amplitude_coefficients(eta, delta, chi1, chi2, fd, fd2, omega, domega, rd_fd=None, rd_fd2=None):
    pn = _pn_coefficients(eta, delta, chi1, chi2)
    spin = fits._s_tot_r(eta, chi1, chi2, delta)
    dchi = chi1 - chi2
    cp_times = eta[..., None] * 0 + jnp.asarray((-2000.0, -250.0, -150.0), dtype=eta.dtype)
    omega_cp = omega(cp_times)
    x = (omega_cp / 2) ** (2 / 3)
    zero = jnp.zeros_like(x)
    base = _pn(x, pn, zero).real
    cp_values = jnp.stack(
        [fits._IMRPhenomT_Inspiral_Amp_CP(eta, spin, dchi, delta, 22, idx) for idx in (1, 2, 3)], axis=-1
    )
    matrix = jnp.stack((x**4, x**4.5, x**5), axis=-1)
    rhs = (cp_values - base) / (pn[2][..., None] * x)
    pseudo = jnp.linalg.solve(matrix, rhs[..., None])[..., 0]
    xcut = x[..., -1:]
    acut, da_dx = _pn(xcut, pn, pseudo, derivative=True)
    sign = jnp.sign(acut.real)
    abs_cut = jnp.abs(acut)
    dx_dt = 2 / 3 * xcut / omega_cp[..., -1:] * domega(cp_times[..., -1:])
    derivative = sign * (acut.conj() * da_dx).real / abs_cut * dx_dt
    alpha = 2 * math.pi * fd
    rd_alpha = alpha if rd_fd is None else 2 * math.pi * rd_fd
    alpha2 = 2 * math.pi * (fd2 if rd_fd2 is None else rd_fd2)
    peak = fits._IMRPhenomT_PeakAmp(eta, spin, dchi, delta, 22)
    midpoint = fits._IMRPhenomT_Intermediate_Amp_CP1(eta, spin, dchi, delta, 22)
    c3 = fits._IMRPhenomT_Ringdown_Amp_C3(eta, spin, dchi, 22)
    c2 = (alpha2 - rd_alpha) / 2
    bound = -0.5 * rd_alpha / jnp.tanh(c3)
    c2 = jnp.where(c2 > jnp.abs(bound), bound, c2)
    c1 = peak * rd_alpha * jnp.cosh(c3) ** 2 / c2
    c4 = peak - c1 * jnp.tanh(c3)
    mt = eta[..., None] * 0 + jnp.asarray((-150.0, -25.0, 0.0), dtype=eta.dtype)
    matrix = jnp.concatenate((_merger_basis(mt, alpha), _merger_basis(mt[..., :1], alpha, derivative=True)), axis=-2)
    rhs = jnp.stack(((sign * abs_cut)[..., 0], midpoint, peak, derivative[..., 0]), axis=-1)
    merger = jnp.linalg.solve(matrix, rhs[..., None])[..., 0]
    return pn, pseudo, merger, alpha, c1, c2, c3, c4, rd_alpha


def amplitude(times, state, omega):
    pn, pseudo, merger, alpha, c1, c2, c3, c4, rd_alpha = state
    inspiral_times = jnp.minimum(times, -150.0)
    x = (omega(inspiral_times) / 2) ** (2 / 3)
    inspiral = jnp.abs(_pn(x, pn, pseudo))
    merger_times = jnp.clip(times, -150.0, 0.0)
    middle = jnp.sum(_merger_basis(merger_times, alpha) * merger[..., None, :], axis=-1)
    rd_times = jnp.maximum(times, 0.0)
    ringdown = jnp.exp(-rd_alpha[..., None] * rd_times) * (
        c1[..., None] * jnp.tanh(c2[..., None] * rd_times + c3[..., None]) + c4[..., None]
    )
    return jnp.where(times < -150.0, inspiral, jnp.abs(jnp.where(times < 0.0, middle, ringdown)))

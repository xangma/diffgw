"""Dominant-mode IMRPhenomT amplitude, in total-mass geometric units.

Adapted from the MIT author formulas in phenomxpy/phenomt/internals.py and
numba_ansatze.py at 4121fafa0484f3db2a7352f4eea697faaaf29f65.
Copyright (C) 2023, 2024, 2025 Cecilio García Quirós.
See THIRD_PARTY_NOTICES.md and Estellés et al., PRD 103, 124060 (2021).
"""

import math

import torch

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
    r, i = torch.ones_like(x), torch.zeros_like(x)
    dr, di = torch.zeros_like(x), torch.zeros_like(x)
    for power, coeff in zip((1, 1.5, 2, 2.5, 3, 3.5), real):
        r = r + coeff[..., None] * x**power
        dr = dr + power * coeff[..., None] * x ** (power - 1)
    for power, coeff in zip((2.5, 3, 3.5), imag):
        i = i + coeff[..., None] * x**power
        di = di + power * coeff[..., None] * x ** (power - 1)
    logarithm = torch.log(16 * x)
    r = r - 428 / 105 * logarithm * x**3
    dr = dr - 428 / 105 * (3 * logarithm + 1) * x**2
    for idx, power in enumerate((4, 4.5, 5)):
        r = r + pseudo[..., idx, None] * x**power
        dr = dr + power * pseudo[..., idx, None] * x ** (power - 1)
    value = fac[..., None] * x * torch.complex(r, i)
    if derivative:
        return value, fac[..., None] * torch.complex(r + x * dr, i + x * di)
    return value


def _merger_basis(t, alpha, derivative=False):
    phi = alpha[..., None] * t
    t = torch.broadcast_to(t, phi.shape)
    sech1, sech2 = 1 / torch.cosh(phi), 1 / torch.cosh(2 * phi)
    if derivative:
        return torch.stack(
            (
                torch.zeros_like(t),
                -alpha[..., None] * sech1 * torch.tanh(phi),
                -2 / 7 * alpha[..., None] * torch.tanh(2 * phi) * sech2 ** (1 / 7),
                2 * t,
            ),
            dim=-1,
        )
    return torch.stack((torch.ones_like(t), sech1, sech2 ** (1 / 7), t**2), dim=-1)


def amplitude_coefficients(eta, delta, chi1, chi2, fd, fd2, omega, domega):
    """Solve amplitude matching; callbacks evaluate angular frequency in M units."""
    pn = _pn_coefficients(eta, delta, chi1, chi2)
    spin = fits._s_tot_r(eta, chi1, chi2, delta)
    dchi = chi1 - chi2
    cp_times = eta[..., None] * 0 + torch.tensor((-2000.0, -250.0, -150.0), dtype=eta.dtype, device=eta.device)
    omega_cp = omega(cp_times)
    x = (omega_cp / 2) ** (2 / 3)
    zero = torch.zeros_like(x)
    base = _pn(x, pn, zero).real
    cp_values = torch.stack(
        [fits._IMRPhenomT_Inspiral_Amp_CP(eta, spin, dchi, delta, 22, idx) for idx in (1, 2, 3)], dim=-1
    )
    matrix = torch.stack((x**4, x**4.5, x**5), dim=-1)
    rhs = (cp_values - base) / (pn[2][..., None] * x)
    pseudo = torch.linalg.solve(matrix, rhs[..., None])[..., 0]

    # Analytic derivative of |A_PN(x(t))| at the inspiral/merger boundary.
    xcut = x[..., -1:]
    acut, da_dx = _pn(xcut, pn, pseudo, derivative=True)
    sign = torch.sign(acut.real)
    abs_cut = torch.abs(acut)
    dx_dt = 2 / 3 * xcut / omega_cp[..., -1:] * domega(cp_times[..., -1:])
    derivative = sign * (acut.conj() * da_dx).real / abs_cut * dx_dt

    alpha = 2 * math.pi * fd
    alpha2 = 2 * math.pi * fd2
    peak = fits._IMRPhenomT_PeakAmp(eta, spin, dchi, delta, 22)
    midpoint = fits._IMRPhenomT_Intermediate_Amp_CP1(eta, spin, dchi, delta, 22)
    c3 = fits._IMRPhenomT_Ringdown_Amp_C3(eta, spin, dchi, 22)
    c2 = (alpha2 - alpha) / 2
    bound = -0.5 * alpha / torch.tanh(c3)
    c2 = torch.where(c2 > torch.abs(bound), bound, c2)
    c1 = peak * alpha * torch.cosh(c3) ** 2 / c2
    c4 = peak - c1 * torch.tanh(c3)

    mt = eta[..., None] * 0 + torch.tensor((-150.0, -25.0, 0.0), dtype=eta.dtype, device=eta.device)
    matrix = torch.cat((_merger_basis(mt, alpha), _merger_basis(mt[..., :1], alpha, derivative=True)), dim=-2)
    rhs = torch.stack(((sign * abs_cut)[..., 0], midpoint, peak, derivative[..., 0]), dim=-1)
    merger = torch.linalg.solve(matrix, rhs[..., None])[..., 0]
    return pn, pseudo, merger, alpha, c1, c2, c3, c4


def amplitude(times, state, omega):
    """Return |h22| with inactive branches evaluated only on finite safe domains."""
    pn, pseudo, merger, alpha, c1, c2, c3, c4 = state
    inspiral_times = torch.clamp(times, max=-150.0)
    x = (omega(inspiral_times) / 2) ** (2 / 3)
    inspiral = torch.abs(_pn(x, pn, pseudo))
    merger_times = torch.clamp(times, min=-150.0, max=0.0)
    middle = torch.sum(_merger_basis(merger_times, alpha) * merger[..., None, :], dim=-1)
    rd_times = torch.clamp(times, min=0.0)
    ringdown = torch.exp(-alpha[..., None] * rd_times) * (
        c1[..., None] * torch.tanh(c2[..., None] * rd_times + c3[..., None]) + c4[..., None]
    )
    return torch.where(times < -150.0, inspiral, torch.abs(torch.where(times < 0.0, middle, ringdown)))

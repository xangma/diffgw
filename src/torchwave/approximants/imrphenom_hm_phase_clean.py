"""Independent London PhenomHM phase-map equations.

This module is diagnostic-only.  It transcribes London et al.
arXiv:1708.00404 Eqs. (3), (5), and (8)-(14), without routing production
waveforms through these helpers.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import torch


def low_frequency_offset(ell: int, m: int) -> float:
    """London Eq. (9) low-frequency phase offset (radians)."""
    return math.pi / 2 * (3 * ell + ((ell + m) % 2)) - math.pi


def mapped_phase_c1(
    f: torch.Tensor,
    f0: float | torch.Tensor,
    f_rd_lm: float | torch.Tensor,
    f_rd_22: float | torch.Tensor,
    m: int,
    carrier_phase: Callable[[torch.Tensor], torch.Tensor],
    ell: int | None = None,
    low_offset: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Map a carrier phase with London's derivative-consistent ``kappa``.

    The frequency map is London Eq. (5), with the PhenomD multiplicative
    ringdown branch of Eq. (13).  On each branch the phase is
    ``kappa*phi22(g(f)) + Delta`` (London Eq. (8)); ``Delta`` is constructed
    from continuity at both joins.  Since ``kappa*g' == 1`` on every branch,
    the returned phase has matching one-sided derivatives when the carrier
    phase is differentiable.  ``low_offset`` defaults to London Eq. (9).
    """
    if m <= 0:
        raise ValueError("m must be positive")
    if ell is None:
        ell = m
    x = torch.as_tensor(f)
    f0t = torch.as_tensor(f0, dtype=x.dtype, device=x.device)
    fr = torch.as_tensor(f_rd_lm, dtype=x.dtype, device=x.device)
    f22 = torch.as_tensor(f_rd_22, dtype=x.dtype, device=x.device)
    if torch.any(fr <= f0t) or torch.any(f22 <= 0):
        raise ValueError("require f_rd_lm > f0 and f_rd_22 > 0")
    mid_slope = (f22 - 2 * f0t / m) / (fr - f0t)
    g0 = 2 * f0t / m
    g_mid = mid_slope * (x - f0t) + g0
    g_high = (f22 / fr) * x
    g = torch.where(x <= f0t, 2 * x / m, torch.where(x <= fr, g_mid, g_high))
    k_mid = 1 / mid_slope
    k_high = fr / f22
    low_delta = torch.as_tensor(
        low_frequency_offset(ell, m) if low_offset is None else low_offset,
        dtype=x.dtype, device=x.device,
    )
    # Delta_mid and Delta_high enforce C0.  kappa*g' makes C1 at joins.
    k_low = m / 2
    phi_at_g0 = carrier_phase(torch.reshape(g0, (1,))).reshape(())
    delta_mid = low_delta + k_low * phi_at_g0 - k_mid * phi_at_g0
    phifr = carrier_phase(torch.reshape(f22, (1,))).reshape(())
    phim_fr = carrier_phase(torch.reshape(f22, (1,))).reshape(())
    delta_high = k_mid * phim_fr + delta_mid - k_high * phifr
    low = k_low * carrier_phase(g) + low_delta
    middle = k_mid * carrier_phase(g) + delta_mid
    high = k_high * carrier_phase(g) + delta_high
    return torch.where(x <= f0t, low, torch.where(x <= fr, middle, high))


def phenomd_mr_phase(
    f: torch.Tensor,
    eta: float | torch.Tensor,
    alpha: Sequence[float | torch.Tensor],
    f_rd_22: float | torch.Tensor,
    f_rd_lm: float | torch.Tensor,
    f_damp_lm: float | torch.Tensor,
) -> torch.Tensor:
    """London Eq. (14) modified PhenomD merger-ringdown phase."""
    if len(alpha) != 6:
        raise ValueError("alpha must contain alpha_0 through alpha_5")
    x = torch.as_tensor(f)
    vals = [torch.as_tensor(a, dtype=x.dtype, device=x.device) for a in alpha]
    et = torch.as_tensor(eta, dtype=x.dtype, device=x.device)
    f22 = torch.as_tensor(f_rd_22, dtype=x.dtype, device=x.device)
    flm = torch.as_tensor(f_rd_lm, dtype=x.dtype, device=x.device)
    damp = torch.as_tensor(f_damp_lm, dtype=x.dtype, device=x.device)
    r = f22 / flm
    arg = (x - vals[5] * f22) / (r * damp)
    return (vals[0] + vals[1] * x - vals[2] / x + (4 / 3) * vals[3] * x ** (3 / 4) + vals[4] * r * torch.atan(arg)) / et


def phenomd_mr_phase_derivative(
    f: torch.Tensor, eta: float | torch.Tensor, alpha: Sequence[float | torch.Tensor],
    f_rd_22: float | torch.Tensor, f_rd_lm: float | torch.Tensor,
    f_damp_lm: float | torch.Tensor,
) -> torch.Tensor:
    """Analytic derivative of :func:`phenomd_mr_phase` (London Eq. (14))."""
    if len(alpha) != 6:
        raise ValueError("alpha must contain alpha_0 through alpha_5")
    x = torch.as_tensor(f)
    vals = [torch.as_tensor(a, dtype=x.dtype, device=x.device) for a in alpha]
    et = torch.as_tensor(eta, dtype=x.dtype, device=x.device)
    f22 = torch.as_tensor(f_rd_22, dtype=x.dtype, device=x.device)
    flm = torch.as_tensor(f_rd_lm, dtype=x.dtype, device=x.device)
    damp = torch.as_tensor(f_damp_lm, dtype=x.dtype, device=x.device)
    r = f22 / flm
    arg = (x - vals[5] * f22) / (r * damp)
    return (vals[1] + vals[2] / x ** 2 + vals[3] * x ** (-1 / 4) + vals[4] / damp / (1 + arg ** 2)) / et


def phenomd_eq14_phase_delta(
    mf: torch.Tensor,
    eta: float | torch.Tensor,
    chi_pn: float | torch.Tensor,
    f_rd_22: float | torch.Tensor,
    f_rd_lm: float | torch.Tensor,
    f_damp_22: float | torch.Tensor,
    f_damp_lm: float | torch.Tensor,
    alpha4: float | torch.Tensor | None = None,
    alpha5: float | torch.Tensor | None = None,
    qnm_ratio: float | torch.Tensor | None = None,
    base_fr22: float | torch.Tensor | None = None,
    base_fd22: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Eq. (14) arctangent replacement relative to the D carrier term.

    Frequencies are dimensionless ``Mf``.  ``alpha4`` and ``alpha5`` may be
    supplied from the public PhenomD ``eval_table_v`` API; omission is an
    error to avoid silently inventing calibration coefficients.

    ``f_rd_22``/``f_rd_lm`` and ``f_damp_22``/``f_damp_lm`` are the HM QNM
    values used by the outer HM frequency map.  The optional ``base_fr22``
    and ``base_fd22`` values are the absolute PhenomD carrier QNM values.
    When they are supplied, Eq. (14) keeps those carrier values in both
    arctangents and uses the HM QNM values only through
    ``R = f_rd_22/f_rd_lm`` and ``T = f_damp_lm/f_damp_22``:

    ``R*atan((f - alpha5*fRD_D)/(R*fdamp_D*T)) -
    atan((f - alpha5*fRD_D)/fdamp_D)``.

    Omitting the base values makes them the HM 22 values, preserving the
    original helper's default behavior.
    """
    if alpha4 is None or alpha5 is None:
        raise ValueError("alpha4 and alpha5 are required")
    x = torch.as_tensor(mf)
    dtype, device = x.dtype, x.device
    vals = [torch.as_tensor(z, dtype=dtype, device=device) for z in
            (eta, chi_pn, f_rd_22, f_rd_lm, f_damp_22, f_damp_lm, alpha4, alpha5)]
    et, _, fr22, flm, fd22, fdlm, a4, a5 = vals
    ratio = fr22 / flm if qnm_ratio is None else torch.as_tensor(qnm_ratio, dtype=dtype, device=device)
    # The mapped HM frequencies remain the HM QNM inputs.  Eq. (14), however,
    # is a correction to the supplied D carrier and therefore must use one
    # absolute carrier pair consistently in both arctangents.  Keep the
    # no-base branch in its original algebraic form so omission of the
    # explicit carrier pair remains numerically backward-compatible.
    if base_fr22 is None and base_fd22 is None:
        arg_lm = (x - a5 * fr22) / (fdlm * ratio)
        arg_22 = (x - a5 * fr22) / fd22
    else:
        carrier_fr22 = fr22 if base_fr22 is None else torch.as_tensor(base_fr22, dtype=dtype, device=device)
        carrier_fd22 = fd22 if base_fd22 is None else torch.as_tensor(base_fd22, dtype=dtype, device=device)
        damping_ratio = fdlm / fd22
        arg_lm = (x - a5 * carrier_fr22) / (carrier_fd22 * ratio * damping_ratio)
        arg_22 = (x - a5 * carrier_fr22) / carrier_fd22
    return a4 / et * (ratio * torch.atan(arg_lm) - torch.atan(arg_22))


def phenomd_eq14_phase_delta_derivative(
    mf: torch.Tensor, eta: float | torch.Tensor, chi_pn: float | torch.Tensor,
    f_rd_22: float | torch.Tensor, f_rd_lm: float | torch.Tensor,
    f_damp_22: float | torch.Tensor, f_damp_lm: float | torch.Tensor,
    alpha4: float | torch.Tensor, alpha5: float | torch.Tensor,
    qnm_ratio: float | torch.Tensor | None = None,
    base_fr22: float | torch.Tensor | None = None,
    base_fd22: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Analytic derivative of :func:`phenomd_eq14_phase_delta`."""
    x = torch.as_tensor(mf)
    dtype, device = x.dtype, x.device
    et, fr22, flm, fd22, fdlm, a4, a5 = [torch.as_tensor(z, dtype=dtype, device=device)
                                            for z in (eta, f_rd_22, f_rd_lm, f_damp_22, f_damp_lm, alpha4, alpha5)]
    ratio = fr22 / flm if qnm_ratio is None else torch.as_tensor(qnm_ratio, dtype=dtype, device=device)
    if base_fr22 is None and base_fd22 is None:
        arg_lm = (x - a5 * fr22) / (fdlm * ratio)
        arg_22 = (x - a5 * fr22) / fd22
        width_lm, width_22 = fdlm * ratio, fd22
    else:
        carrier_fr22 = fr22 if base_fr22 is None else torch.as_tensor(base_fr22, dtype=dtype, device=device)
        carrier_fd22 = fd22 if base_fd22 is None else torch.as_tensor(base_fd22, dtype=dtype, device=device)
        damping_ratio = fdlm / fd22
        arg_lm = (x - a5 * carrier_fr22) / (carrier_fd22 * ratio * damping_ratio)
        arg_22 = (x - a5 * carrier_fr22) / carrier_fd22
        width_lm, width_22 = carrier_fd22 * ratio * damping_ratio, carrier_fd22
    return a4 / et * (ratio / width_lm / (1 + arg_lm**2) - 1 / width_22 / (1 + arg_22**2))


def phenomd_eq14_phase_delta_c1(
    mf: torch.Tensor, eta: float | torch.Tensor, chi_pn: float | torch.Tensor,
    f_rd_22: float | torch.Tensor, f_rd_lm: float | torch.Tensor,
    f_damp_22: float | torch.Tensor, f_damp_lm: float | torch.Tensor,
    alpha4: float | torch.Tensor, alpha5: float | torch.Tensor,
    join_factor: float = 0.5,
    qnm_ratio: float | torch.Tensor | None = None,
    base_fr22: float | torch.Tensor | None = None,
    base_fd22: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Attach Eq. (14)'s delta above the Khan Sec. V MR join with C1 matching.

    The join is expressed in the supplied PhenomD carrier ``Mf`` coordinate.
    London first maps the higher-mode frequency into that coordinate, while
    this Eq. (14) correction keeps the carrier QNM pair absolute.  Thus the
    ``0.5 f_RD`` transition uses ``base_fr22`` when supplied and otherwise
    defaults to the HM ``f_rd_22`` value.
    """
    x = torch.as_tensor(mf)
    # Preserve the algebraic 22 control exactly when both QNM pairs are the
    # carrier pair; this is also useful for detecting accidental remapping.
    ratio_is_one = qnm_ratio is None or float(torch.as_tensor(qnm_ratio)) == 1.0
    if (ratio_is_one and
            torch.equal(torch.as_tensor(f_rd_22), torch.as_tensor(f_rd_lm)) and
            torch.equal(torch.as_tensor(f_damp_22), torch.as_tensor(f_damp_lm))):
        return torch.zeros_like(x)
    carrier_fr22 = torch.as_tensor(
        f_rd_22 if base_fr22 is None else base_fr22,
        dtype=x.dtype,
        device=x.device,
    )
    join = torch.as_tensor(join_factor, dtype=x.dtype, device=x.device) * carrier_fr22
    d0 = phenomd_eq14_phase_delta(join, eta, chi_pn, f_rd_22, f_rd_lm, f_damp_22, f_damp_lm, alpha4, alpha5, qnm_ratio, base_fr22, base_fd22)
    slope = phenomd_eq14_phase_delta_derivative(join, eta, chi_pn, f_rd_22, f_rd_lm, f_damp_22, f_damp_lm, alpha4, alpha5, qnm_ratio, base_fr22, base_fd22)
    raw = phenomd_eq14_phase_delta(x, eta, chi_pn, f_rd_22, f_rd_lm, f_damp_22, f_damp_lm, alpha4, alpha5, qnm_ratio, base_fr22, base_fd22)
    return torch.where(x > join, raw - d0 - slope * (x - join), torch.zeros_like(x))

"""Native, differentiable published-NNLO twisting for IMRPhenomTP(HM).

The implementation uses the published NNLO angle prescription and the
repository's clean-room coefficient helpers.  It contains only tensor
algebra: no SciPy, interpolation, or LAL calls are used in the time-domain
path.
"""

from __future__ import annotations

import math
import torch

from diffgw_core.constants import MSUN_SEC
from .imrphenom_xas import XASParameterFits
from .precession_beta_clean import orbital_angular_momentum
from .precession_nnlo_clean import coefficients as _angle_coefficients
from .precession_source_frame_clean import source_frame_geometry


def _safe_norm(x, y):
    eps = torch.finfo(x.dtype).eps
    return torch.sqrt(x.square() + y.square() + eps * eps) - eps


def _effective_chip(m1, m2, s1x, s1y, s2x, s2y):
    q = m1 / m2
    a1, a2 = 2.0 + 1.5 / q, 2.0 + 1.5 * q
    p1 = m1.square() * _safe_norm(s1x, s1y)
    p2 = m2.square() * _safe_norm(s2x, s2y)
    return torch.maximum(a1 * p1, a2 * p2) / (a1 * m1.square())


def _nnlo_beta(m1, m2, f, s1x, s1y, s1z, s2x, s2y, s2z):
    total = m1 + m2
    n1, n2 = m1 / total, m2 / total
    L = orbital_angular_momentum(m1, m2, f, s1z, s2z, pn_version=102, compat_102=False)
    spar = n1.square() * s1z + n2.square() * s2z
    a1, a2 = 2.0 + 1.5 * m2 / m1, 2.0 + 1.5 * m1 / m2
    chip = torch.maximum(a1 * m1.square() * _safe_norm(s1x, s1y), a2 * m2.square() * _safe_norm(s2x, s2y)) / (
        a1 * m1.square()
    )
    return torch.atan2(n1.square() * chip, L + spar)


def _canonicalize_components(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z):
    """Return the published convention ``m1 >= m2`` without Python branching."""
    swap = m1 < m2
    return (
        torch.where(swap, m2, m1),
        torch.where(swap, m1, m2),
        torch.where(swap, s2x, s1x),
        torch.where(swap, s2y, s1y),
        torch.where(swap, s2z, s1z),
        torch.where(swap, s1x, s2x),
        torch.where(swap, s1y, s2y),
        torch.where(swap, s1z, s2z),
    )


def nnlo_final_spin(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z):
    """Published NNLO precessing remnant spin, in the canonical mass order."""
    m1, m2, s1x, s1y, s1z, s2x, s2y, s2z = _canonicalize_components(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z)
    total = m1 + m2
    eta = m1 * m2 / total.square()
    chip = _effective_chip(m1, m2, s1x, s1y, s2x, s2y)
    return XASParameterFits.fit_final_spin_nnlo(eta, s1z, s2z, chip)


def _raw_angles(v, eta, delta, m1, chi_l, chi_p):
    """Broadcast the published series over parameter batches and time."""
    aa, ee = _angle_coefficients(eta, delta, m1, chi_l, chi_p)
    basis = torch.stack((v**-3, v**-2, v**-1, v, torch.log(v**3)), dim=-1)
    if v.ndim > eta.ndim:
        aa, ee = aa[..., None, :], ee[..., None, :]
    return (aa * basis).sum(dim=-1), (ee * basis).sum(dim=-1)


def nnlo_geometry(
    times,
    frequency,
    mass1,
    mass2,
    spin1x,
    spin1y,
    spin1z,
    spin2x,
    spin2y,
    spin2z,
    inclination,
    phic,
    f_ref,
    frequency_peak=None,
    final_spin=None,
):
    """Return ``alpha, beta, gamma, theta_jn, zeta`` on a time grid."""
    mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = _canonicalize_components(
        mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z
    )
    m = mass1 + mass2
    ms = m * MSUN_SEC
    fr = torch.where(f_ref > 0, f_ref, torch.full_like(f_ref, 20.0))
    f = torch.clamp(frequency, min=torch.finfo(frequency.dtype).tiny)
    if frequency_peak is None:
        frequency_peak = frequency[..., 0]
    fp = torch.clamp(frequency_peak, min=torch.finfo(frequency.dtype).tiny)
    v_peak = (math.pi * ms * fp) ** (1.0 / 3.0)
    v = (math.pi * ms[..., None] * f) ** (1.0 / 3.0)
    v = torch.where(times >= 0, v_peak[..., None], v)
    v0 = (math.pi * ms * fr) ** (1.0 / 3.0)
    chip = _effective_chip(mass1, mass2, spin1x, spin1y, spin2x, spin2y)
    eta = mass1 * mass2 / m.square()
    delta = (mass1 - mass2) / m
    x1 = mass1 / m
    chi_l = (mass1 * spin1z + mass2 * spin2z) / m
    # The published NNLO series uses the single-spin variables of
    # Pratten et al. (delta is defined with the opposite sign here).
    alpha_raw, epsilon_raw = _raw_angles(v, eta, -delta, x1, chi_l / x1, chip)
    alpha_ref_raw, epsilon_ref_raw = _raw_angles(v0, eta, -delta, x1, chi_l / x1, chip)
    alpha_peak_raw, epsilon_peak_raw = _raw_angles(v_peak, eta, -delta, x1, chi_l / x1, chip)
    alpha0, theta_jn, zeta0, phi_j = source_frame_geometry(
        mass1,
        mass2,
        spin1x,
        spin1y,
        spin1z,
        spin2x,
        spin2y,
        spin2z,
        inclination,
        phic,
        fr,
    )
    # The published construction anchors both angles at f_ref.  The peak
    # values below are only used to start the ringdown continuation.
    if alpha_raw.ndim > alpha0.ndim:
        alpha_ref_t, epsilon_ref_t = alpha_ref_raw[..., None], epsilon_ref_raw[..., None]
        alpha0_t, phi_j_t = alpha0[..., None], phi_j[..., None]
    else:
        alpha_ref_t, epsilon_ref_t, alpha0_t, phi_j_t = alpha_ref_raw, epsilon_ref_raw, alpha0, phi_j
    alpha = alpha_raw - alpha_ref_t + alpha0_t
    gamma = -epsilon_raw + epsilon_ref_t + math.pi - phi_j_t
    alpha_peak = alpha_peak_raw - alpha_ref_raw + alpha0
    gamma_peak = -epsilon_peak_raw + epsilon_ref_raw + math.pi - phi_j
    beta_dyn = _nnlo_beta(
        mass1[..., None] if f.ndim > mass1.ndim else mass1,
        mass2[..., None] if f.ndim > mass2.ndim else mass2,
        f,
        spin1x[..., None] if f.ndim > spin1x.ndim else spin1x,
        spin1y[..., None] if f.ndim > spin1y.ndim else spin1y,
        spin1z[..., None] if f.ndim > spin1z.ndim else spin1z,
        spin2x[..., None] if f.ndim > spin2x.ndim else spin2x,
        spin2y[..., None] if f.ndim > spin2y.ndim else spin2y,
        spin2z[..., None] if f.ndim > spin2z.ndim else spin2z,
    )
    beta_peak = _nnlo_beta(
        mass1,
        mass2,
        fp,
        spin1x,
        spin1y,
        spin1z,
        spin2x,
        spin2y,
        spin2z,
    )
    # ``project_modes`` uses the standard ``d^l_{m,2}`` spin-weighted
    # harmonic convention, for which the source-frame polarization angle is
    # already in the required basis.  The clean positive-mode helper uses a
    # transposed harmonic convention and therefore subtracts pi/2; applying
    # that shift here would introduce a spurious global minus in the aligned
    # limit.
    zeta = zeta0
    # Appendix A, Sec. II.5: beta is held fixed after the peak and alpha is
    # continued with omega_RD,22 - omega_RD,21.  Use the precessing remnant
    # spin in the QNM fits rather than the aligned-spin carrier fit.
    from diffgw_core import phenomt_fits as fits

    af = (
        nnlo_final_spin(mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z)
        if final_spin is None
        else final_spin
    )
    delta = (mass1 - mass2) / m
    mf = fits.IMRPhenomX_FinalMass2017(eta, spin1z, spin2z, delta)
    f22 = fits.IMRPhenomT_fringfit(af, 22) / mf
    f21 = fits.IMRPhenomT_fringfit(af, 21) / mf
    slope = (2.0 * math.pi / ms) * (f22 - f21)
    alpha_rd = alpha_peak[..., None] + slope[..., None] * times
    gamma_rd = gamma_peak[..., None] - torch.cos(beta_peak)[..., None] * (alpha_rd - alpha_peak[..., None])
    post = times >= 0
    return (
        torch.where(post, alpha_rd, alpha),
        torch.where(post, beta_peak[..., None], beta_dyn),
        torch.where(post, gamma_rd, gamma),
        theta_jn,
        zeta,
    )


def wigner_d(l: int, m: int, mp: int, beta: torch.Tensor) -> torch.Tensor:
    """Finite factorial sum for the real Wigner small-d element."""
    if not (-l <= m <= l and -l <= mp <= l):
        raise ValueError("magnetic indices must lie in [-l,l]")
    c, s = torch.cos(beta / 2), torch.sin(beta / 2)
    pref = math.sqrt(math.factorial(l + m) * math.factorial(l - m) * math.factorial(l + mp) * math.factorial(l - mp))
    out = torch.zeros_like(beta)
    for k in range(2 * l + 1):
        a, q, r, z = l + mp - k, k, m - mp + k, l - m - k
        if min(a, q, r, z) < 0:
            continue
        out = out + ((-1.0) ** (k + m - mp)) * pref / (
            math.factorial(a) * math.factorial(q) * math.factorial(r) * math.factorial(z)
        ) * c ** (2 * l + mp - m - 2 * k) * s ** (m - mp + 2 * k)
    return out


def project_modes(coprecessing_modes, alpha, beta, gamma, theta_jn, zeta):
    """Rotate signed coprecessing modes and return real ``(hplus,hcross)``.

    ``h_l,-m = (-1)**l conjugate(h_l,m)`` is supplied when a provider gives
    only positive modes.  This is the time-domain mode sum; the intermediate
    complex strain is retained until the final real/imaginary projection.
    """
    if not coprecessing_modes:
        raise ValueError("at least one coprecessing mode is required")
    full = dict(coprecessing_modes)
    for (ell, mp), value in tuple(coprecessing_modes.items()):
        if mp > 0 and (ell, -mp) not in full:
            full[(ell, -mp)] = (-1) ** ell * torch.conj(value)
    first = next(iter(full.values()))
    strain = torch.zeros_like(torch.as_tensor(first, dtype=torch.complex128))
    for (ell, mp), mode in full.items():
        mode = torch.as_tensor(mode, dtype=torch.complex128, device=strain.device)
        yscale = math.sqrt((2 * ell + 1) / (4 * math.pi))
        for m in range(-ell, ell + 1):
            # _{-2}Y_lm(theta,0), with the author's azimuth convention.
            y = yscale * wigner_d(ell, m, 2, theta_jn)
            rotation = torch.exp(-1j * m * alpha) * wigner_d(ell, m, mp, beta)
            rotation = rotation * torch.exp(-1j * mp * gamma)
            strain = strain + mode * rotation * y
    # zeta is a polarization-basis rotation: h_+ - i h_x -> exp(2 i zeta) h.
    strain = torch.exp(2j * zeta) * strain
    return strain.real, -strain.imag


__all__ = ["nnlo_geometry", "nnlo_final_spin", "wigner_d", "project_modes"]

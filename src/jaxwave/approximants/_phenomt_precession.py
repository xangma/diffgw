"""Pure-JAX published-NNLO twisting algebra for IMRPhenomTP(HM).

The angle prescription follows the published NNLO construction and uses the
repository's clean-room coefficient helpers.
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC
from .imrphenom_xas import XASParameterFits


def _safe_norm(x, y):
    eps = jnp.finfo(jnp.result_type(x, y, jnp.float64)).eps
    return jnp.sqrt(x * x + y * y + eps * eps) - eps


def _effective_chip(m1, m2, s1x, s1y, s2x, s2y):
    q = m1 / m2
    a1, a2 = 2.0 + 1.5 / q, 2.0 + 1.5 * q
    p1 = m1 * m1 * _safe_norm(s1x, s1y)
    p2 = m2 * m2 * _safe_norm(s2x, s2y)
    return jnp.maximum(a1 * p1, a2 * p2) / (a1 * m1 * m1)


def _canonicalize_components(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z):
    """Return the published convention ``m1 >= m2`` without Python branching."""
    swap = m1 < m2
    return (
        jnp.where(swap, m2, m1),
        jnp.where(swap, m1, m2),
        jnp.where(swap, s2x, s1x),
        jnp.where(swap, s2y, s1y),
        jnp.where(swap, s2z, s1z),
        jnp.where(swap, s1x, s2x),
        jnp.where(swap, s1y, s2y),
        jnp.where(swap, s1z, s2z),
    )


def nnlo_final_spin(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z):
    """Published NNLO precessing remnant spin, in the canonical mass order."""
    m1, m2, s1x, s1y, s1z, s2x, s2y, s2z = _canonicalize_components(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z)
    total = m1 + m2
    eta = m1 * m2 / total**2
    chip = _effective_chip(m1, m2, s1x, s1y, s2x, s2y)
    return XASParameterFits.fit_final_spin_nnlo(eta, s1z, s2z, chip)


def _raw_angle_coefficients(eta, delta, m1, chi_l, chi_p):
    """Published NNLO single-spin coefficients (Pratten et al. Eqs. 120-122)."""
    pi = math.pi
    a3 = -5 * delta / (64 * m1) - 35 / 192
    a2 = -5 * m1 * chi_l * (3 * delta + 7 * m1) / (128 * eta)
    a1 = (
        -5 * (824 * eta + 1103) / 3072
        - 15 * delta**2 * eta / (256 * m1**2)
        - 15 * delta * m1**3 * chi_p**2 / (128 * eta**2)
        - 5 * delta * (980 * eta + 911) / (7168 * m1)
        - 35 * m1**4 * chi_p**2 / (128 * eta**2)
    )
    a_1 = (
        5 * (36 * eta * (85568 * eta + 23817) + 8024297) / 9289728
        + 5 * m1**2 * (3 * delta**2 * chi_p**2 + 75 * delta**2 * chi_l**2 - 112 * pi * chi_l) / (256 * eta)
        - 15 * delta * m1**7 * (chi_p**4 - 4 * chi_p**2 * chi_l**2) / (512 * eta**4)
        + 5 * delta * m1**3 * ((812 * eta - 97) * chi_p**2 + 20328 * eta * chi_l**2) / (14336 * eta**2)
        - 15 * pi * delta * m1 * chi_l / (16 * eta)
        - 35 * m1**8 * (chi_p**4 - 4 * chi_p**2 * chi_l**2) / (512 * eta**4)
        + m1**4 * (25 * (92 * eta + 19) * chi_p**2 + 52640 * eta * chi_l**2) / (6144 * eta**2)
        + 15 * delta**3 * eta**2 / (1024 * m1**3)
        + 5 * delta**2 * eta * (784 * eta + 323) / (28672 * m1**2)
        + 5 * delta * (504 * eta * (7630 * eta - 159) + 5579177) / (21676032 * m1)
    )
    a_log = (
        -5 / 48 * (7 * pi - 3 * delta**2 * chi_l)
        - 5 * delta * m1**5 * chi_p**2 * chi_l / (128 * eta**3)
        + 5 * delta * (7168 * eta + 407) * m1 * chi_l / (21504 * eta)
        - 35 * m1**6 * chi_p**2 * chi_l / (384 * eta**3)
        + 5 * (4072 * eta + 599) * m1**2 * chi_l / (9216 * eta)
        - 5 * pi * delta / (16 * m1)
    )
    e3 = a3
    e2 = a2
    e1 = (
        -5 * (824 * eta + 1103) / 3072
        - 15 * delta**2 * eta / (256 * m1**2)
        - 5 * delta * (980 * eta + 911) / (7168 * m1)
    )
    e_1 = (
        5 * (36 * eta * (85568 * eta + 23817) + 8024297) / 9289728
        + 5 * m1**2 * chi_l * (75 * delta**2 * chi_l - 112 * pi) / (256 * eta)
        + 1815 * delta * m1**3 * chi_l**2 / (256 * eta)
        - 15 * pi * delta * m1 * chi_l / (16 * eta)
        + 1645 * m1**4 * chi_l**2 / (192 * eta)
        + 15 * delta**3 * eta**2 / (1024 * m1**3)
        + 5 * delta**2 * eta * (784 * eta + 323) / (28672 * m1**2)
        + 5 * delta * (504 * eta * (7630 * eta - 159) + 5579177) / (21676032 * m1)
    )
    e_log = (
        -5 / 48 * (7 * pi - 3 * delta**2 * chi_l)
        + 5 * delta * (7168 * eta + 407) * m1 * chi_l / (21504 * eta)
        + 5 * (4072 * eta + 599) * m1**2 * chi_l / (9216 * eta)
        - 5 * pi * delta / (16 * m1)
    )
    return jnp.stack((a3, a2, a1, a_1, a_log), axis=-1), jnp.stack((e3, e2, e1, e_1, e_log), axis=-1)


def _raw_angles(v, eta, delta, m1, chi_l, chi_p):
    aa, ee = _raw_angle_coefficients(eta, delta, m1, chi_l, chi_p)
    basis = jnp.stack((v**-3, v**-2, v**-1, v, jnp.log(v**3)), axis=-1)
    if v.ndim == eta.ndim:
        return jnp.sum(aa * basis, axis=-1), jnp.sum(ee * basis, axis=-1)
    return jnp.sum(aa[..., None, :] * basis, axis=-1), jnp.sum(ee[..., None, :] * basis, axis=-1)


def _orbital_angular_momentum(m1, m2, f, s1z, s2z):
    total = m1 + m2
    eta = m1 * m2 / total**2
    delta = (m1 - m2) / total
    n1, n2 = m1 / total, m2 / total
    x = (math.pi * total * f * MSUN_SEC) ** (2 / 3)
    sl = n1**2 * s1z + n2**2 * s2z
    sigma = n2 * s2z - n1 * s1z
    l0 = 1.0
    l1 = eta / 6 + 1.5
    l2 = eta**2 / 24 - 19 * eta / 8 + 27 / 8
    l3 = 7 * eta**3 / 1296 + 31 * eta**2 / 24 + (41 * math.pi**2 / 24 - 6889 / 144) * eta + 135 / 16
    ls15 = -35 * sl / 6 - 5 * delta * sigma / 2
    ls25 = (-77 / 8 + 427 * eta / 72) * sl + delta * (-21 / 8 + 35 * eta / 12) * sigma
    return eta / jnp.sqrt(x) * (l0 + l1 * x + l2 * x**2 + l3 * x**3 + ls15 * x**1.5 + ls25 * x**2.5)


def _clean_beta(m1, m2, f, s1x, s1y, s1z, s2x, s2y, s2z):
    total = m1 + m2
    n1, n2 = m1 / total, m2 / total
    L = _orbital_angular_momentum(m1, m2, f, s1z, s2z)
    a1 = 2 + 1.5 * m2 / m1
    a2 = 2 + 1.5 * m1 / m2
    chip = jnp.maximum(a1 * m1**2 * jnp.hypot(s1x, s1y), a2 * m2**2 * jnp.hypot(s2x, s2y)) / (a1 * m1**2)
    spar = n1**2 * s1z + n2**2 * s2z
    return jnp.arctan2(n1**2 * chip, L + spar)


def _source_geometry(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, iota, phi_ref, f_ref):
    total = m1 + m2
    n1, n2 = m1 / total, m2 / total
    L = _orbital_angular_momentum(m1, m2, f_ref, s1z, s2z)
    spin = jnp.stack((n1**2 * s1x + n2**2 * s2x, n1**2 * s1y + n2**2 * s2y, n1**2 * s1z + n2**2 * s2z), axis=0)
    J = spin + jnp.stack((jnp.zeros_like(L), jnp.zeros_like(L), L), axis=0)
    jhat = J / jnp.linalg.norm(J, axis=0, keepdims=True)
    n = jnp.stack((jnp.sin(iota) * jnp.sin(phi_ref), jnp.sin(iota) * jnp.cos(phi_ref), jnp.cos(iota)), axis=0)
    observer_x = jnp.stack(
        (-jnp.cos(iota) * jnp.sin(phi_ref), -jnp.cos(iota) * jnp.cos(phi_ref), jnp.sin(iota)), axis=0
    )
    yj = jnp.cross(jhat, n, axis=0)
    yn = jnp.linalg.norm(yj, axis=0)
    tangent = jnp.cross(jhat, observer_x, axis=0)
    tn = jnp.linalg.norm(tangent, axis=0)
    eps = jnp.finfo(jnp.result_type(m1, jnp.float64)).eps
    yjg = yj / jnp.where(yn > eps, yn, 1.0)[None, ...]
    yjf = tangent / jnp.where(tn > eps, tn, 1.0)[None, ...]
    yj = jnp.where((yn > eps)[None, ...], yjg, yjf)
    xj = jnp.cross(yj, jhat, axis=0)
    ar = _safe_norm(yj[2], xj[2])
    alpha0 = jnp.arctan2(jnp.where(ar > eps, yj[2], 0.0), jnp.where(ar > eps, xj[2], 1.0))
    jp = _safe_norm(J[0], J[1])
    phi_j = jnp.where(
        jp > eps, jnp.arctan2(jnp.where(jp > eps, J[1], 0.0), jnp.where(jp > eps, J[0], 1.0)), math.pi / 2 - phi_ref
    )
    jn = jnp.sum(jhat * n, axis=0)
    cn = jnp.linalg.norm(jnp.cross(jhat, n, axis=0), axis=0)
    raw_theta = jnp.arctan2(cn, jn)
    endpoint = cn <= eps
    endpoint_raw = jnp.where(endpoint, jax.lax.stop_gradient(raw_theta), raw_theta)
    theta = endpoint_raw + endpoint * (jnp.where(jn < 0, math.pi, 0.0) - endpoint_raw)
    x_sphere = jnp.cos(theta) * xj - jnp.sin(theta) * jhat
    zeta = jnp.arctan2(jnp.sum(observer_x * yj, axis=0), jnp.sum(observer_x * x_sphere, axis=0))
    return alpha0, theta, zeta, phi_j


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
    mass = mass1 + mass2
    ms = mass * MSUN_SEC
    fr = jnp.where(f_ref > 0, f_ref, jnp.full_like(f_ref, 20.0))
    f = jnp.maximum(frequency, jnp.finfo(frequency.dtype).tiny)
    if frequency_peak is None:
        frequency_peak = frequency[..., 0]
    fp = jnp.maximum(frequency_peak, jnp.finfo(frequency.dtype).tiny)
    v_peak = (math.pi * ms * fp) ** (1.0 / 3.0)
    v = (math.pi * ms[..., None] * f) ** (1.0 / 3.0)
    v = jnp.where(times >= 0, v_peak[..., None], v)
    v0 = (math.pi * ms * fr) ** (1.0 / 3.0)
    chip = _effective_chip(mass1, mass2, spin1x, spin1y, spin2x, spin2y)
    eta = mass1 * mass2 / mass**2
    delta = (mass1 - mass2) / mass
    x1 = mass1 / mass
    chi_l = (mass1 * spin1z + mass2 * spin2z) / mass
    alpha_raw, epsilon_raw = _raw_angles(v, eta, -delta, x1, chi_l / x1, chip)
    alpha_ref_raw, epsilon_ref_raw = _raw_angles(v0, eta, -delta, x1, chi_l / x1, chip)
    alpha_peak_raw, epsilon_peak_raw = _raw_angles(v_peak, eta, -delta, x1, chi_l / x1, chip)
    alpha0, theta_jn, zeta0, phi_j = _source_geometry(
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
    if f.ndim > mass1.ndim:
        beta_dyn = _clean_beta(
            mass1[..., None],
            mass2[..., None],
            f,
            spin1x[..., None],
            spin1y[..., None],
            spin1z[..., None],
            spin2x[..., None],
            spin2y[..., None],
            spin2z[..., None],
        )
    else:
        beta_dyn = _clean_beta(mass1, mass2, f, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z)
    beta_peak = _clean_beta(mass1, mass2, fp, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z)
    # ``project_modes`` uses the standard ``d^l_{m,2}`` spin-weighted
    # harmonic convention, for which the source-frame polarization angle is
    # already in the required basis.  The clean positive-mode helper uses a
    # transposed harmonic convention and therefore subtracts pi/2; applying
    # that shift here would introduce a spurious global minus in the aligned
    # limit.
    zeta = zeta0
    from diffgw_core import phenomt_fits as fits

    af = (
        nnlo_final_spin(mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z)
        if final_spin is None
        else final_spin
    )
    delta = (mass1 - mass2) / mass
    mf = fits.IMRPhenomX_FinalMass2017(eta, spin1z, spin2z, delta)
    f22 = fits.IMRPhenomT_fringfit(af, 22) / mf
    f21 = fits.IMRPhenomT_fringfit(af, 21) / mf
    slope = (2.0 * math.pi / ms) * (f22 - f21)
    alpha_rd = alpha_peak[..., None] + slope[..., None] * times
    gamma_rd = gamma_peak[..., None] - jnp.cos(beta_peak)[..., None] * (alpha_rd - alpha_peak[..., None])
    post = times >= 0
    return (
        jnp.where(post, alpha_rd, alpha),
        jnp.where(post, beta_peak[..., None], beta_dyn),
        jnp.where(post, gamma_rd, gamma),
        theta_jn,
        zeta,
    )


def wigner_d(l: int, m: int, mp: int, beta):
    """Finite factorial sum for the real Wigner small-d element."""
    if not (-l <= m <= l and -l <= mp <= l):
        raise ValueError("magnetic indices must lie in [-l,l]")
    c, s = jnp.cos(beta / 2), jnp.sin(beta / 2)
    pref = math.sqrt(math.factorial(l + m) * math.factorial(l - m) * math.factorial(l + mp) * math.factorial(l - mp))
    out = jnp.zeros_like(beta)
    for k in range(2 * l + 1):
        a, q, r, z = l + mp - k, k, m - mp + k, l - m - k
        if min(a, q, r, z) < 0:
            continue
        out = out + ((-1.0) ** (k + m - mp)) * pref / (
            math.factorial(a) * math.factorial(q) * math.factorial(r) * math.factorial(z)
        ) * c ** (2 * l + mp - m - 2 * k) * s ** (m - mp + 2 * k)
    return out


def project_modes(coprecessing_modes, alpha, beta, gamma, theta_jn, zeta):
    """Rotate signed coprecessing modes and return real ``(hplus,hcross)``."""
    if not coprecessing_modes:
        raise ValueError("at least one coprecessing mode is required")
    full = dict(coprecessing_modes)
    for (ell, mp), value in tuple(coprecessing_modes.items()):
        if mp > 0 and (ell, -mp) not in full:
            full[(ell, -mp)] = (-1) ** ell * jnp.conj(value)
    first = next(iter(full.values()))
    strain = jnp.zeros_like(jnp.asarray(first, dtype=jnp.complex128))
    for (ell, mp), mode in full.items():
        mode = jnp.asarray(mode, dtype=jnp.complex128)
        yscale = math.sqrt((2 * ell + 1) / (4 * math.pi))
        for m in range(-ell, ell + 1):
            y = yscale * wigner_d(ell, m, 2, theta_jn)
            rotation = jnp.exp(-1j * m * alpha) * wigner_d(ell, m, mp, beta)
            rotation = rotation * jnp.exp(-1j * mp * gamma)
            strain = strain + mode * rotation * y
    strain = jnp.exp(2j * zeta) * strain
    return strain.real, -strain.imag


__all__ = ["nnlo_geometry", "nnlo_final_spin", "wigner_d", "project_modes"]

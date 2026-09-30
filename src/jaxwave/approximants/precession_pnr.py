"""Pure JAX PNR angle engine used by the native XPNR-labelled route.

Provides a native JAX implementation of the NR-tuned precession angles (alpha,
beta, gamma), two-spin to single-spin mapping, multi-harmonic linear frequency mapping,
and minimal-rotation coprecessing phase quadrature for IMRPhenomXPNR (prec_version=330).

References:
- Hamilton et al. (2025), arXiv:2507.02604v1
- Thompson et al. (2024), Phys. Rev. D 109, 063012 (arXiv:2312.10025v1)
"""

from typing import Tuple, Union
import jax
import jax.numpy as jnp

from diffgw_core.constants import (
    MSUN_SEC,
    PI,
)
from jaxwave.conversions import as_array
from jaxwave.approximants.pnr_coefficients import COEFF_ARRAYS
from jaxwave.approximants.imrphenom_xas import XASParameterFits
from jaxwave.approximants.precession_spintaylor import (
    evaluate_spintaylor_precession,
)


def _eval_coeff_poly(
    coeff_array: jnp.ndarray,
    eta: jnp.ndarray,
    chi: jnp.ndarray,
    costheta: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate 3D polynomial sum_{i=0..3, j=0..3, k=0..4} C_{ijk} eta^i chi^j costheta^k."""
    dtype = coeff_array.dtype
    eta_p = jnp.stack([(eta.astype(dtype)) ** i for i in range(4)], axis=-1)
    chi_p = jnp.stack([(chi.astype(dtype)) ** j for j in range(4)], axis=-1)
    costheta_p = jnp.stack([(costheta.astype(dtype)) ** k for k in range(5)], axis=-1)
    return jnp.einsum("ijk,...i,...j,...k->...", coeff_array, eta_p, chi_p, costheta_p)


def compute_pnr_single_spin(
    m1: jnp.ndarray,
    m2: jnp.ndarray,
    s1x: jnp.ndarray,
    s1y: jnp.ndarray,
    s1z: jnp.ndarray,
    s2x: jnp.ndarray,
    s2y: jnp.ndarray,
    s2z: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the algebra of Hamilton et al. (2025), Eqs. (3)--(6).

    Full paper fidelity requires spins at the SpinTaylor stopping frequency;
    the current caller still supplies reference-frequency spins.
    """
    total_m = m1 + m2
    q = m1 / m2
    eta = (m1 * m2) / (total_m * total_m)

    chi_eff = (m1 * s1z + m2 * s2z) / total_m
    chi_parr = total_m * chi_eff / m1

    s_tot_x = (m1 ** 2) * s1x + (m2 ** 2) * s2x
    s_tot_y = (m1 ** 2) * s1y + (m2 ** 2) * s2y
    chis = jnp.sqrt(s_tot_x ** 2 + s_tot_y ** 2) / (m1 ** 2)

    chi_perp = chis
    chi_mag = jnp.sqrt(chi_parr ** 2 + chi_perp ** 2)

    costheta = jnp.where(chi_mag >= 1e-6, chi_parr / jnp.maximum(chi_mag, 1e-12), jnp.zeros_like(chi_mag))
    costheta = jnp.clip(costheta, -1.0, 1.0)

    # Final spin estimate (Hamilton et al. Sec. VII B)
    chi1L = chi_mag * costheta
    xf_parr = XASParameterFits.fit_final_spin(eta, chi1L, jnp.zeros_like(chi1L))
    qfactor = q / (1.0 + q)
    sintheta = jnp.sqrt(jnp.maximum(1.0 - costheta ** 2, 0.0))
    xf_perp = (qfactor ** 2) * chi_mag * sintheta
    xf = jnp.sqrt(xf_parr ** 2 + xf_perp ** 2)
    costheta_final = jnp.where(xf > 1e-6, xf_parr / jnp.maximum(xf, 1e-12), jnp.zeros_like(xf))
    costheta_final = jnp.clip(costheta_final, -1.0, 1.0)

    return chi_mag, costheta, costheta_final, chi_eff


def compute_pnr_alpha_coefficients(
    eta: jnp.ndarray,
    chi: jnp.ndarray,
    costheta: jnp.ndarray,
    q: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute merger-ringdown alpha Ansatz parameters (A1, A2, A3, A4)."""
    eta_eff = jnp.where(eta >= 0.09, eta, jnp.full_like(eta, 0.09))
    chiboundary = 0.80 - 0.20 * jnp.exp(-(((q - 6.0) / 1.5) ** 8))
    chi_eff = jnp.where(chi <= chiboundary, chi, chiboundary)

    c_a1 = COEFF_ARRAYS["A1"]
    c_a2 = COEFF_ARRAYS["A2"]
    c_a3 = COEFF_ARRAYS["A3"]
    c_a4 = COEFF_ARRAYS["A4"]

    sintheta = jnp.sqrt(jnp.maximum(1.0 - costheta ** 2, 0.0))

    poly_a1 = _eval_coeff_poly(c_a1, eta_eff, chi_eff, costheta)
    a1 = chi_eff * sintheta * (poly_a1 ** 2)
    a1 = jnp.abs(a1)

    poly_a2 = _eval_coeff_poly(c_a2, eta_eff, chi_eff, costheta)
    a2 = -chi_eff * sintheta * (poly_a2 ** 2)
    a2 = jnp.minimum(a2, 0.0)

    poly_a3 = _eval_coeff_poly(c_a3, eta_eff, chi_eff, costheta)
    a3 = poly_a3 ** 2
    a3 = jnp.maximum(a3, 1e-5)

    max_a2 = PI * PI * jnp.sqrt(a3)
    a2 = jnp.where(a2 < -max_a2, -max_a2, a2)

    poly_a4 = _eval_coeff_poly(c_a4, eta_eff, chi_eff, costheta)
    a4 = poly_a4 ** 2

    return a1, a2, a3, a4


def compute_pnr_beta_coefficients(
    eta: jnp.ndarray,
    chi: jnp.ndarray,
    costheta: jnp.ndarray,
    costheta_final: jnp.ndarray,
    q: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute merger-ringdown beta Ansatz parameters (B0, B1, B2, B3, B4, B5)."""
    eta_eff = jnp.where(eta >= 0.09, eta, jnp.full_like(eta, 0.09))
    chiboundary = 0.80 - 0.20 * jnp.exp(-(((q - 6.0) / 1.5) ** 8))
    chi_eff = jnp.where(chi <= chiboundary, chi, chiboundary)

    c_b0 = COEFF_ARRAYS["B0"]
    c_b1 = COEFF_ARRAYS["B1"]
    c_b2 = COEFF_ARRAYS["B2"]
    c_b3 = COEFF_ARRAYS["B3"]
    c_b4 = COEFF_ARRAYS["B4"]
    c_b5 = COEFF_ARRAYS["B5"]

    sintheta = jnp.sqrt(jnp.maximum(1.0 - costheta ** 2, 0.0))

    poly_b4 = _eval_coeff_poly(c_b4, eta_eff, chi_eff, costheta)
    b4 = jnp.maximum(poly_b4 ** 2, 175.0)

    poly_b0 = _eval_coeff_poly(c_b0, eta_eff, chi_eff, costheta)
    b0 = jnp.arccos(costheta_final) - chi_eff * sintheta * poly_b0

    poly_b1 = _eval_coeff_poly(c_b1, eta_eff, chi_eff, costheta)
    b1 = chi_eff * sintheta * jnp.exp(poly_b1)

    poly_b2 = _eval_coeff_poly(c_b2, eta_eff, chi_eff, costheta)
    b2 = -chi_eff * sintheta * jnp.exp(poly_b2)

    poly_b3 = _eval_coeff_poly(c_b3, eta_eff, chi_eff, costheta)
    b3 = chi_eff * sintheta * (jnp.exp(poly_b3) - 200.0)

    poly_b5 = _eval_coeff_poly(c_b5, eta_eff, chi_eff, costheta)
    b5 = -(poly_b5 ** 2)

    return b0, b1, b2, b3, b4, b5


def eval_pnr_alpha_mr(
    mf: jnp.ndarray,
    a1: jnp.ndarray,
    a2: jnp.ndarray,
    a3: jnp.ndarray,
    a4: jnp.ndarray,
) -> jnp.ndarray:
    """Merger-ringdown alpha Ansatz, Hamilton et al. (2025), Eq. (15)."""
    return -(a1 / jnp.maximum(mf, 1e-8) + (a2 * jnp.sqrt(a3)) / (a3 + (mf - a4) ** 2))


def eval_pnr_beta_mr(
    mf: jnp.ndarray,
    b0: jnp.ndarray,
    b1: jnp.ndarray,
    b2: jnp.ndarray,
    b3: jnp.ndarray,
    b4: jnp.ndarray,
    b5: jnp.ndarray,
) -> jnp.ndarray:
    """Merger-ringdown beta Ansatz, Hamilton et al. (2025), Eq. (16)."""
    return b0 + (b1 + b2 * mf + b3 * (mf ** 2)) / (1.0 + b4 * ((mf + b5) ** 2))


def pnr_angles_window(q: jnp.ndarray, chi: jnp.ndarray) -> jnp.ndarray:
    """Smooth blending window outside the PNR calibration boundaries."""
    q_low, q_high = 8.5, 12.0
    chi_low, chi_high = 0.85, 1.2

    wq = jnp.where(
        q <= q_low,
        jnp.ones_like(q),
        jnp.where(
            q >= q_high,
            jnp.zeros_like(q),
            jnp.cos(0.5 * PI * (q - q_low) / (q_high - q_low)) ** 2,
        ),
    )
    wchi = jnp.where(
        chi <= chi_low,
        jnp.ones_like(chi),
        jnp.where(
            chi >= chi_high,
            jnp.zeros_like(chi),
            jnp.cos(0.5 * PI * (chi - chi_low) / (chi_high - chi_low)) ** 2,
        ),
    )
    return wq * wchi


def eval_pnr_linear_frequency_map(
    mf: jnp.ndarray,
    ell: int,
    emm: int,
    mf_low: jnp.ndarray,
    mf_high: jnp.ndarray,
    mf_rd_22: jnp.ndarray,
    mf_rd_lm: jnp.ndarray,
) -> jnp.ndarray:
    """Higher-mode frequency map, Thompson et al. (2024), Eqs. (50)--(52)."""
    if ell == 2 and emm == 2:
        return mf

    mf_spa = 2.0 * mf / float(emm)
    delta_rd = mf_rd_lm - mf_rd_22
    mf_post = mf - delta_rd

    slope = (mf_high - delta_rd - 2.0 * mf_low / float(emm)) / jnp.maximum(mf_high - mf_low, 1e-8)
    mf_inter = slope * (mf - mf_low) + (2.0 * mf_low / float(emm))

    return jnp.where(
        mf <= mf_low,
        mf_spa,
        jnp.where(
            mf >= mf_high,
            mf_post,
            mf_inter,
        ),
    )


def evaluate_pnr_precession(
    f: jnp.ndarray,
    f_ref: Union[float, jnp.ndarray],
    mass1: Union[float, jnp.ndarray],
    mass2: Union[float, jnp.ndarray],
    spin1x: Union[float, jnp.ndarray] = 0.0,
    spin1y: Union[float, jnp.ndarray] = 0.0,
    spin1z: Union[float, jnp.ndarray] = 0.0,
    spin2x: Union[float, jnp.ndarray] = 0.0,
    spin2y: Union[float, jnp.ndarray] = 0.0,
    spin2z: Union[float, jnp.ndarray] = 0.0,
    inclination: Union[float, jnp.ndarray] = 0.0,
    phi_ref: Union[float, jnp.ndarray] = 0.0,
    convention: int = 1,
    observer_azimuth: Union[float, jnp.ndarray] = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate the provisional native PNR angle route in pure JAX.

    Returns:
        alpha: J-frame precession azimuth angle
        beta: J-frame precession opening angle (between J and L_N)
        gamma: coprecessing polarization rotation from minimal rotation condition
        theta_jn: inclination of J with respect to line-of-sight N
        zeta: carrier phase offset
        carrier_phase: reference phase
    """
    dtype = f.dtype if f.dtype == jnp.float64 else jnp.float64
    f_arr = as_array(f, dtype=dtype)
    m1 = as_array(mass1, dtype=dtype)
    m2 = as_array(mass2, dtype=dtype)
    s1x = as_array(spin1x, dtype=dtype)
    s1y = as_array(spin1y, dtype=dtype)
    s1z = as_array(spin1z, dtype=dtype)
    s2x = as_array(spin2x, dtype=dtype)
    s2y = as_array(spin2y, dtype=dtype)
    s2z = as_array(spin2z, dtype=dtype)
    inc = as_array(inclination, dtype=dtype)
    pref = as_array(phi_ref, dtype=dtype)
    fref = as_array(f_ref, dtype=dtype)
    phi0 = as_array(observer_azimuth, dtype=dtype)

    total_m = m1 + m2
    is_batched = total_m.ndim >= 1
    f_eval = f_arr if (not is_batched or f_arr.ndim > 1) else f_arr[None, :]

    def u_param(t: jnp.ndarray) -> jnp.ndarray:
        if is_batched and f_eval.ndim > t.ndim:
            return jnp.expand_dims(t, axis=-1)
        return t

    q = m1 / m2
    eta = (m1 * m2) / (total_m * total_m)
    m_sec = total_m * MSUN_SEC
    mf_eval = f_eval * u_param(m_sec)

    # 1. Base SpinTaylor inspiral
    a_st, b_st, g_st, th_jn, zeta, cp = evaluate_spintaylor_precession(
        f=f_arr,
        f_ref=fref,
        mass1=m1,
        mass2=m2,
        spin1x=s1x,
        spin1y=s1y,
        spin1z=s1z,
        spin2x=s2x,
        spin2y=s2y,
        spin2z=s2z,
        inclination=inc,
        phi_ref=pref,
        convention=convention,
        observer_azimuth=phi0,
    )

    # In-plane spin check
    in_plane_sq = s1x * s1x + s1y * s1y + s2x * s2x + s2y * s2y
    is_aligned = in_plane_sq == 0.0

    if not isinstance(is_aligned, jax.core.Tracer) and bool(jnp.all(is_aligned)):
        return a_st, b_st, g_st, th_jn, zeta, cp

    # 2. PNR single spin & fits
    chi_mag, costheta, costheta_f, chi_eff = compute_pnr_single_spin(
        m1, m2, s1x, s1y, s1z, s2x, s2y, s2z
    )
    a1, a2, a3, a4 = compute_pnr_alpha_coefficients(eta, chi_mag, costheta, q)
    b0, b1, b2, b3, b4, b5 = compute_pnr_beta_coefficients(eta, chi_mag, costheta, costheta_f, q)

    a1_u, a2_u, a3_u, a4_u = u_param(a1), u_param(a2), u_param(a3), u_param(a4)
    b0_u, b1_u, b2_u, b3_u, b4_u, b5_u = u_param(b0), u_param(b1), u_param(b2), u_param(b3), u_param(b4), u_param(b5)

    # 3. Connection frequencies
    mf_alpha_upper = a4 / 3.0
    mf_alpha_lower = (3.0 / 3.5) * mf_alpha_upper

    root_term = b4 * (b2 - 2.0 * b3 * b5) * (b2 - 2.0 * b1 * b4 * b5 + b2 * b4 * b5 * b5) + (b3 - b1 * b4 + b3 * b4 * b5 * b5) ** 2
    denom_extr = b4 * (b2 - 2.0 * b3 * b5)
    mf_plus = ((b3 - b1 * b4 + b3 * b4 * b5 * b5) + jnp.sqrt(jnp.maximum(root_term, 0.0))) / jnp.maximum(denom_extr, 1e-12)
    mf_minus = ((b3 - b1 * b4 + b3 * b4 * b5 * b5) - jnp.sqrt(jnp.maximum(root_term, 0.0))) / jnp.maximum(denom_extr, 1e-12)

    mf_beta_lower = jnp.where(mf_minus > 0.009, mf_minus, jnp.full_like(mf_minus, 0.05))
    mf_beta_upper = jnp.where(mf_plus > mf_beta_lower, mf_plus, jnp.full_like(mf_plus, 100.0))

    mf_b_target = u_param(mf_beta_lower)
    mf_b_upper_u = u_param(mf_beta_upper)

    def searchsorted_last_axis(a, v):
        if a.ndim == 1:
            return jnp.searchsorted(a, v)
        return jax.vmap(lambda a_row, v_row: jnp.searchsorted(a_row, v_row))(a, v)

    def gather_last_axis(arr, idx):
        if arr.ndim == 1 and (idx.ndim == 0 or idx.ndim == 1):
            return arr[idx]
        if idx.ndim < arr.ndim:
            idx = jnp.expand_dims(idx, axis=-1)
        return jnp.take_along_axis(arr, idx, axis=-1)

    # Search attachment point in frequency grid
    idx_b_raw = searchsorted_last_axis(jax.lax.stop_gradient(mf_eval), jax.lax.stop_gradient(mf_b_target))
    idx_b_low = jnp.clip(idx_b_raw, 1, mf_eval.shape[-1] - 2)

    beta1 = gather_last_axis(b_st, idx_b_low)
    b_high = gather_last_axis(b_st, idx_b_low + 1)
    b_low = gather_last_axis(b_st, idx_b_low - 1)
    mf_high = gather_last_axis(mf_eval, idx_b_low + 1)
    mf_low = gather_last_axis(mf_eval, idx_b_low - 1)
    dbeta1 = (b_high - b_low) / jnp.maximum(mf_high - mf_low, 1e-12)

    beta2 = eval_pnr_beta_mr(mf_b_target, b0_u, b1_u, b2_u, b3_u, b4_u, b5_u)
    dmf = 0.0005
    dbeta2 = (eval_pnr_beta_mr(mf_b_target + dmf, b0_u, b1_u, b2_u, b3_u, b4_u, b5_u) - eval_pnr_beta_mr(mf_b_target - dmf, b0_u, b1_u, b2_u, b3_u, b4_u, b5_u)) / (2.0 * dmf)

    D1 = beta1 * beta1 * mf_b_target
    N1 = -2.0 * beta1 * (beta2 - beta1) + (beta1 * dbeta2 - beta2 * dbeta1) * mf_b_target
    b_rescale_1 = -N1 / jnp.maximum(D1, 1e-12)

    D2 = (beta1 * mf_b_target) ** 2
    N2 = beta1 * (beta2 - beta1) - (beta1 * dbeta2 - beta2 * dbeta1) * mf_b_target
    b_rescale_2 = -N2 / jnp.maximum(D2, 1e-12)

    rescale_beta = 1.0 + b_rescale_1 * mf_eval + b_rescale_2 * (mf_eval ** 2)

    beta_mr_eval = eval_pnr_beta_mr(mf_eval, b0_u, b1_u, b2_u, b3_u, b4_u, b5_u)
    beta_mr_upper = eval_pnr_beta_mr(mf_b_upper_u, b0_u, b1_u, b2_u, b3_u, b4_u, b5_u)

    beta_pnr = jnp.where(
        mf_eval <= mf_b_target,
        b_st * rescale_beta,
        jnp.where(
            mf_eval >= mf_b_upper_u,
            beta_mr_upper,
            beta_mr_eval,
        ),
    )
    beta_pnr = jnp.clip(beta_pnr, 0.0, PI)

    # 4. Alpha connection
    mf_a_target = u_param(mf_alpha_lower)
    mf_a_upper_u = u_param(mf_alpha_upper)

    idx_a_raw = searchsorted_last_axis(jax.lax.stop_gradient(mf_eval), jax.lax.stop_gradient(mf_a_target))
    idx_a_low = jnp.clip(idx_a_raw, 0, mf_eval.shape[-1] - 1)
    a_st_low = gather_last_axis(a_st, idx_a_low)
    alpha_mr_offset = a_st_low - eval_pnr_alpha_mr(mf_a_target, a1_u, a2_u, a3_u, a4_u)

    t_al = jnp.clip((mf_eval - mf_a_target) / jnp.maximum(mf_a_upper_u - mf_a_target, 1e-8), 0.0, 1.0)
    h00 = (1.0 + 2.0 * t_al) * ((1.0 - t_al) ** 2)
    h01 = (t_al ** 2) * (3.0 - 2.0 * t_al)
    alpha_mr_upper = eval_pnr_alpha_mr(mf_a_upper_u, a1_u, a2_u, a3_u, a4_u) + alpha_mr_offset
    alpha_interp = h00 * a_st_low + h01 * alpha_mr_upper

    alpha_mr_eval = eval_pnr_alpha_mr(mf_eval, a1_u, a2_u, a3_u, a4_u) + alpha_mr_offset

    alpha_pnr = jnp.where(
        mf_eval <= mf_a_target,
        a_st,
        jnp.where(
            mf_eval >= mf_a_upper_u,
            alpha_mr_eval,
            alpha_interp,
        ),
    )

    # 5. Gamma from minimal rotation condition
    dalpha = jnp.diff(alpha_pnr, axis=-1)
    cosbeta_mid = 0.5 * (jnp.cos(beta_pnr[..., :-1]) + jnp.cos(beta_pnr[..., 1:]))
    dgamma = -cosbeta_mid * dalpha
    gamma_pnr = jnp.concatenate([jnp.zeros_like(alpha_pnr[..., :1]), jnp.cumsum(dgamma, axis=-1)], axis=-1)

    # 6. Outside calibration window smooth blending
    w = pnr_angles_window(q, chi_mag)
    w_u = u_param(w)
    alpha_final = w_u * alpha_pnr + (1.0 - w_u) * a_st
    beta_final = w_u * beta_pnr + (1.0 - w_u) * b_st
    gamma_final = w_u * gamma_pnr + (1.0 - w_u) * g_st

    # Guard against zero in-plane spins in traced / JIT evaluation
    is_al_u = u_param(is_aligned)
    alpha_final = jnp.where(is_al_u, a_st, alpha_final)
    beta_final = jnp.where(is_al_u, b_st, beta_final)
    gamma_final = jnp.where(is_al_u, g_st, gamma_final)

    if not is_batched and alpha_final.ndim > f_arr.ndim:
        alpha_final = jnp.squeeze(alpha_final, 0)
        beta_final = jnp.squeeze(beta_final, 0)
        gamma_final = jnp.squeeze(gamma_final, 0)
        th_jn = jnp.squeeze(th_jn, 0) if th_jn.ndim > 0 else th_jn
        zeta = jnp.squeeze(zeta, 0) if zeta.ndim > 0 else zeta
        cp = jnp.squeeze(cp, 0) if cp.ndim > 0 else cp

    return alpha_final, beta_final, gamma_final, th_jn, zeta, cp


__all__ = [
    "_eval_coeff_poly",
    "compute_pnr_single_spin",
    "compute_pnr_alpha_coefficients",
    "compute_pnr_beta_coefficients",
    "eval_pnr_alpha_mr",
    "eval_pnr_beta_mr",
    "pnr_angles_window",
    "eval_pnr_linear_frequency_map",
    "evaluate_pnr_precession",
]

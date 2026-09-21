"""Native JAX MSA twisting geometry for PhenomT.

The low-level evaluator is the JAX translation of the published
Chatziioannou--Pratten PV223 equations.  This wrapper supplies the PhenomT
carrier frequency and the Sec. II.5 ringdown continuation.
"""
from __future__ import annotations

import math
import jax
import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC
from diffgw_core import phenomt_fits as fits
from .precession_msa import (
    compute_msa_averaged_couplings,
    compute_msa_final_spin_squared,
    evaluate_msa_precession,
)
from .imrphenom_xas import XASParameterFits
from ._phenomt_precession import _source_geometry
from ._phenomt_angle_options import apply_angle_option


def _canonicalize(m1, m2, x1, y1, z1, x2, y2, z2):
    swap = m2 > m1
    return (jnp.where(swap, m2, m1), jnp.where(swap, m1, m2),
            jnp.where(swap, x2, x1), jnp.where(swap, y2, y1),
            jnp.where(swap, z2, z1), jnp.where(swap, x1, x2),
            jnp.where(swap, y1, y2), jnp.where(swap, z1, z2))


def msa_final_spin(mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
                   f_ref=20.0, prepared=None):
    """Return the signed PV223/FS3 remnant Kerr parameter.

    The published precessing MSA averages have an unresolved equal-mass
    representation singularity.  Aligned equal-mass rows use the calibrated
    aligned limit; nonaligned equal-mass rows are outside this API's domain
    and raise in eager execution.  Numerical precession is the supported
    equal-mass alternative.  A traced unsupported row is represented by NaN
    because Python exceptions cannot be raised from a JAX compiled graph.
    Rows outside the published real-root, positive-remnant, or Kerr domain
    follow the same eager/traced error policy.
    """
    mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = _canonicalize(
        jnp.asarray(mass1), jnp.asarray(mass2), jnp.asarray(spin1x),
        jnp.asarray(spin1y), jnp.asarray(spin1z), jnp.asarray(spin2x),
        jnp.asarray(spin2y), jnp.asarray(spin2z),
    )
    f0 = jnp.asarray(f_ref, dtype=jnp.result_type(mass1, jnp.float64))
    f0 = jnp.where(f0 > 0, f0, jnp.full_like(f0, 20.0))
    aligned = (spin1x == 0) & (spin1y == 0) & (spin2x == 0) & (spin2y == 0)
    unsupported_equal_mass = (mass1 == mass2) & ~aligned
    try:
        if bool(jnp.any(unsupported_equal_mass)):
            raise ValueError(
                "MSA precession requires unequal masses for nonaligned rows; "
                "use precession_method='numerical' for equal-mass precession."
            )
    except jax.errors.TracerBoolConversionError:
        pass
    v0 = (jnp.pi * (mass1 + mass2) * MSUN_SEC * f0) ** (1.0 / 3.0)
    v0, mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = jnp.broadcast_arrays(
        v0, mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z
    )
    inplane_sq = spin1x * spin1x + spin1y * spin1y + spin2x * spin2x + spin2y * spin2y
    numerical_degeneracy = inplane_sq <= 10.0 * jnp.finfo(v0.dtype).eps * (
        1.0 + spin1z * spin1z + spin2z * spin2z
    )
    finite_inputs = jnp.stack((v0, mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z), axis=0)
    invalid_inputs = ~jnp.all(jnp.isfinite(finite_inputs), axis=0)
    invalid_inputs |= (mass1 <= 0) | (mass2 <= 0) | (v0 <= 0)
    invalid_inputs |= (
        (spin1x * spin1x + spin1y * spin1y + spin1z * spin1z > 1.0 + 1e-12)
        | (spin2x * spin2x + spin2y * spin2y + spin2z * spin2z > 1.0 + 1e-12)
    )
    def eager_reject(mask, message):
        try:
            if bool(jnp.any(mask)):
                raise ValueError(message)
        except jax.errors.TracerBoolConversionError:
            pass
    eager_reject(invalid_inputs, "MSA final spin requires finite, positive masses, spins, and v0.")

    total_mass = mass1 + mass2
    eta = mass1 * mass2 / (total_mass * total_mass)
    a_aligned = XASParameterFits.fit_final_spin(eta, spin1z, spin2z)
    m_final = 1.0 - XASParameterFits.fit_erad(eta, spin1z, spin2z)
    invalid_fit = (~jnp.isfinite(a_aligned)) | (jnp.abs(a_aligned) > 1.0 + 1e-10)
    invalid_fit |= (~jnp.isfinite(m_final)) | (m_final <= 0)
    eager_reject(invalid_fit, "MSA final spin fit is outside the physical remnant domain.")

    sav_sq, s1_l_avg, s2_l_avg, valid = compute_msa_averaged_couplings(
        v0, mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z
    )
    s_parallel = s1_l_avg + s2_l_avg
    l_fin = (m_final * m_final) * a_aligned - s_parallel
    s_fin_sq = compute_msa_final_spin_squared(sav_sq, s1_l_avg, s2_l_avg, l_fin)
    invalid_spin = (~jnp.isfinite(s_fin_sq)) | (s_fin_sq < -1e-12)
    magnitude = jnp.sqrt(jnp.maximum(s_fin_sq, 0.0)) / (m_final * m_final)
    invalid_spin |= (~jnp.isfinite(magnitude)) | (magnitude > 1.0 + 1e-10)
    unsupported_domain = (~valid) & ~numerical_degeneracy
    eager_reject(
        unsupported_domain,
        "MSA final spin has a nonaligned row outside the supported root domain.",
    )
    eager_reject(invalid_spin, "MSA final spin produced an invalid or non-Kerr remnant spin.")
    signed = jnp.where(a_aligned < 0.0, -magnitude, magnitude)
    result = jnp.where(numerical_degeneracy, a_aligned, signed)
    invalid = invalid_inputs | invalid_fit | invalid_spin | unsupported_domain | unsupported_equal_mass
    return jnp.where(invalid, jnp.nan, result)


def msa_geometry(times, frequency, mass1, mass2, spin1x, spin1y, spin1z,
                 spin2x, spin2y, spin2z, inclination, phic, f_ref,
                 frequency_peak=None, final_spin=None, precession_version=22310,
                 frequency_at_time=None, reference_time=None, meco_time=None, prepared=None):
    """Return ``alpha, beta, gamma, theta_jn, zeta`` on a physical time grid."""
    t = jnp.asarray(times, dtype=jnp.float64)
    f = jnp.asarray(frequency, dtype=jnp.float64)
    vals = [jnp.asarray(v, dtype=jnp.float64) for v in
            (mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
             inclination, phic, f_ref)]
    m1, m2, x1, y1, z1, x2, y2, z2, inc, pc, fr = jnp.broadcast_arrays(*vals)
    m1, m2, x1, y1, z1, x2, y2, z2 = _canonicalize(m1, m2, x1, y1, z1, x2, y2, z2)
    fr = jnp.where(fr > 0, fr, jnp.full_like(fr, 20.0))
    supplied_unsupported = None
    if final_spin is not None:
        aligned = (x1 == 0) & (y1 == 0) & (x2 == 0) & (y2 == 0)
        supplied_unsupported = (m1 == m2) & ~aligned
        try:
            if bool(jnp.any(supplied_unsupported)):
                raise ValueError(
                    "MSA precession requires unequal masses for nonaligned rows; "
                    "use precession_method='numerical' for equal-mass precession."
                )
        except jax.errors.TracerBoolConversionError:
            pass
    fp = f[..., 0] if frequency_peak is None and f.ndim else (f if frequency_peak is None else jnp.asarray(frequency_peak, dtype=f.dtype))
    t_grid = (t if t.ndim > m1.ndim else
              t.reshape((1,) * m1.ndim + t.shape[-1:])) if t.ndim else t
    post = t_grid >= 0
    f_grid = f if frequency_at_time is None else jnp.asarray(
        frequency_at_time(t_grid), dtype=f.dtype
    )
    suffix = int(precession_version) % 100
    f_eval = jnp.maximum(
        f_grid if suffix == 0 else jnp.where(post, fp[..., None], f_grid),
        jnp.finfo(f.dtype).tiny,
    )
    total = m1 + m2
    v = (jnp.pi * total[..., None] * MSUN_SEC * f_eval) ** (1.0 / 3.0)
    v0 = (jnp.pi * total * MSUN_SEC * fr) ** (1.0 / 3.0)
    alpha, beta, gamma, _theta_unused, _zeta_unused = evaluate_msa_precession(
        v, v0[..., None], m1[..., None], m2[..., None], x1[..., None], y1[..., None], z1[..., None],
        x2[..., None], y2[..., None], z2[..., None], inc[..., None], pc[..., None],
        prec_version=223, convention=1,
    )
    _alpha0, theta_jn, zeta, _phi_j = _source_geometry(
        m1, m2, x1, y1, z1, x2, y2, z2, inc, pc, fr,
    )
    if final_spin is None:
        af = msa_final_spin(m1, m2, x1, y1, z1, x2, y2, z2, f_ref=fr)
    else:
        af = jnp.asarray(final_spin, dtype=f.dtype)
    mf = fits.IMRPhenomX_FinalMass2017(m1 * m2 / total**2, z1, z2, (m1 - m2) / total)
    slope = (2 * math.pi / (total * MSUN_SEC)) * (
        fits.IMRPhenomT_fringfit(af, 22) / mf - fits.IMRPhenomT_fringfit(af, 21) / mf
    )
    fp_v = (jnp.pi * total * MSUN_SEC * fp) ** (1.0 / 3.0)
    apeak, bpeak, gpeak, _, _ = evaluate_msa_precession(
        fp_v, v0, m1, m2, x1, y1, z1, x2, y2, z2, inc, pc,
        prec_version=223, convention=1,
    )
    def angle_evaluator(query):
        if frequency_at_time is None:
            raise ValueError("frequency_at_time is required for MSA numerical angle options")
        lead = query.ndim - fp.ndim
        qfp = fp.reshape(fp.shape + (1,) * lead)
        fq = jnp.asarray(frequency_at_time(query), dtype=f.dtype)
        fq = jnp.maximum(fq, jnp.finfo(f.dtype).tiny)
        fq = jnp.where(query >= 0, qfp, fq)
        def qexpand(value):
            return value.reshape(value.shape + (1,) * (query.ndim - value.ndim))
        qv = (jnp.pi * qexpand(total) * MSUN_SEC * fq) ** (1.0 / 3.0)
        qa, qb, _, _, _ = evaluate_msa_precession(
            qv, qexpand(v0), qexpand(m1), qexpand(m2), qexpand(x1), qexpand(y1), qexpand(z1),
            qexpand(x2), qexpand(y2), qexpand(z2), qexpand(inc), qexpand(pc),
            prec_version=223, convention=1,
        )
        return qa, qb

    if suffix in (11, 21):
        if frequency_at_time is None or reference_time is None:
            raise ValueError("MSA numerical angle options require frequency_at_time and reference_time")
        if suffix == 21 and meco_time is None:
            raise ValueError("MSA option 22321 requires meco_time from the carrier model")
        _, _, gamma_ref, _, _ = evaluate_msa_precession(
            v0, v0, m1, m2, x1, y1, z1, x2, y2, z2, inc, pc, prec_version=223, convention=1,
        )
    else:
        gamma_ref = None
    alpha, beta, gamma = apply_angle_option(
        t_grid, alpha, beta, gamma, apeak, bpeak, gpeak, slope, int(precession_version),
        angle_evaluator=angle_evaluator if gamma_ref is not None else None,
        meco_time=meco_time, reference_time=reference_time, gamma_reference=gamma_ref,
    )
    result = (alpha, beta, gamma, theta_jn, zeta)
    unsupported = ~jnp.isfinite(af)
    if supplied_unsupported is not None:
        unsupported = unsupported | supplied_unsupported
    def mask_value(value):
        mask = unsupported
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        return jnp.where(mask, jnp.nan, value)
    return tuple(mask_value(value) for value in result)


__all__ = ["msa_geometry", "msa_final_spin"]

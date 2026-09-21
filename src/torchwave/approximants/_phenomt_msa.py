"""Native MSA twisting geometry for the PhenomT family.

The inspiral angles use the repository's audited PV223 MSA evaluator and its
clean source-frame geometry helper.  The aligned PhenomT carrier supplies the
orbital frequency; only the post-peak QNM continuation depends on
``final_spin``.
"""
from __future__ import annotations

import math
import torch

from diffgw_core.constants import MSUN_SEC
from diffgw_core import phenomt_fits as fits
from .precession_source_frame_clean import source_frame_geometry
from .precession_msa import evaluate_msa_precession as _evaluate_msa_precession
from .precession_msa import compute_msa_final_spin as _published_msa_final_spin
from ._phenomt_angle_options import apply_angle_option


def _canonicalize(m1, m2, x1, y1, z1, x2, y2, z2):
    swap = m2 > m1
    return (
        torch.where(swap, m2, m1), torch.where(swap, m1, m2),
        torch.where(swap, x2, x1), torch.where(swap, y2, y1),
        torch.where(swap, z2, z1), torch.where(swap, x1, x2),
        torch.where(swap, y1, y2), torch.where(swap, z1, z2),
    )


def msa_final_spin(mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
                   f_ref=20.0, prepared=None):
    """Return the signed PV223/FS3 remnant Kerr parameter.

    The published precessing MSA averages have an unresolved equal-mass
    representation singularity.  Aligned equal-mass rows use the calibrated
    aligned limit; nonaligned equal-mass rows are outside this API's domain
    and raise.  Numerical precession is the supported equal-mass alternative.
    Rows outside the published real-root, positive-remnant, or Kerr domain
    also raise.
    """
    inputs = (mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z, f_ref)
    device = next((v.device for v in inputs if isinstance(v, torch.Tensor)), torch.device("cpu"))
    m1, m2, x1, y1, z1, x2, y2, z2, f_ref = [
        torch.as_tensor(v, dtype=torch.float64, device=device) for v in inputs
    ]
    m1, m2, x1, y1, z1, x2, y2, z2 = _canonicalize(m1, m2, x1, y1, z1, x2, y2, z2)
    total = m1 + m2
    f0 = f_ref.to(device=total.device)
    # Direct calls use non-positive ``f_ref`` to request the canonical
    # 20-Hz reference. Normalize before taking the PN velocity so the
    # source-frame and MSA equations never see v_ref=0.
    f0 = torch.where(f0 > 0, f0, torch.full_like(f0, 20.0))
    v0 = (math.pi * total * MSUN_SEC * f0).pow(1.0 / 3.0)
    aligned = (x1 == 0) & (y1 == 0) & (x2 == 0) & (y2 == 0)
    unsupported_equal_mass = (m1 == m2) & ~aligned
    is_comp = hasattr(torch.compiler, "is_compiling") and torch.compiler.is_compiling()
    if not is_comp and bool(torch.any(unsupported_equal_mass)):
        raise ValueError(
            "MSA precession requires unequal masses for nonaligned rows; "
            "use precession_method='numerical' for equal-mass precession."
        )
    # The production entry point is the native Torch implementation of the
    # same audited PV223 equations and supplies the FS3 averaged couplings.
    result = _published_msa_final_spin(v0, m1, m2, x1, y1, z1, x2, y2, z2)
    # Torch compilation cannot raise during tracing.  Keep the unsupported
    # rows explicit in compiled graphs instead of exposing the low-level
    # evaluator's regularized surrogate as an MSA remnant spin.
    return torch.where(unsupported_equal_mass, torch.full_like(result, torch.nan), result)


def msa_geometry(times, frequency, mass1, mass2, spin1x, spin1y, spin1z,
                 spin2x, spin2y, spin2z, inclination, phic, f_ref,
                 frequency_peak=None, final_spin=None, precession_version=22310,
                 frequency_at_time=None, reference_time=None, meco_time=None, prepared=None):
    """Return ``alpha, beta, gamma, theta_jn, zeta`` for native PhenomT MSA."""
    t = torch.as_tensor(times, dtype=torch.float64)
    f = torch.as_tensor(frequency, dtype=torch.float64, device=t.device)
    vals = [torch.as_tensor(v, dtype=torch.float64, device=t.device)
            for v in (mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z,
                      inclination, phic, f_ref)]
    m1, m2, x1, y1, z1, x2, y2, z2, inc, pc, fr = torch.broadcast_tensors(*vals)
    m1, m2, x1, y1, z1, x2, y2, z2 = _canonicalize(m1, m2, x1, y1, z1, x2, y2, z2)
    fr = torch.where(fr > 0, fr, torch.full_like(fr, 20.0))
    supplied_unsupported = None
    if final_spin is not None:
        aligned = (x1 == 0) & (y1 == 0) & (x2 == 0) & (y2 == 0)
        supplied_unsupported = (m1 == m2) & ~aligned
        is_comp = hasattr(torch.compiler, "is_compiling") and torch.compiler.is_compiling()
        if not is_comp and bool(torch.any(supplied_unsupported)):
            raise ValueError(
                "MSA precession requires unequal masses for nonaligned rows; "
                "use precession_method='numerical' for equal-mass precession."
            )
    if frequency_peak is None:
        fp = f[..., 0] if f.ndim else f
    else:
        fp = torch.as_tensor(frequency_peak, dtype=f.dtype, device=f.device)
    # The waveform's physical time grid is the final axis.  Shared grids are
    # broadcast over parameter batches by the assembler.
    t_grid = (t if t.ndim > m1.ndim else
              t.reshape((1,) * m1.ndim + t.shape[-1:])) if t.ndim else t
    f_grid = f if frequency_at_time is None else torch.as_tensor(
        frequency_at_time(t_grid), dtype=f.dtype, device=f.device
    )
    fp_grid = fp.reshape(fp.shape + (1,))
    post = t_grid >= 0
    suffix = int(precession_version) % 100
    f_eval = f_grid if suffix == 0 else torch.where(post, fp_grid, f_grid)
    f_eval = torch.clamp(f_eval, min=torch.finfo(f_eval.dtype).tiny)
    total = m1 + m2
    v_eval = (math.pi * total[..., None] * MSUN_SEC * f_eval).pow(1.0 / 3.0)
    v0 = (math.pi * total * MSUN_SEC * fr).pow(1.0 / 3.0)
    # Expand the reference velocity across the physical grid; this avoids a
    # scalar special case in the low-level evaluator and keeps batch grids
    # differentiable.
    v0_eval = v0[..., None].expand_as(v_eval)
    alpha, beta, gamma, _theta_unused, _zeta_unused = _evaluate_msa_precession(
        v_eval, v0_eval, m1[..., None].expand_as(v_eval), m2[..., None].expand_as(v_eval),
        x1[..., None].expand_as(v_eval), y1[..., None].expand_as(v_eval), z1[..., None].expand_as(v_eval),
        x2[..., None].expand_as(v_eval), y2[..., None].expand_as(v_eval), z2[..., None].expand_as(v_eval),
        inc[..., None].expand_as(v_eval), pc[..., None].expand_as(v_eval), prec_version=223,
    )
    # Use the audited source-frame construction at f_ref.  It is static over
    # the carrier grid and keeps Torch/JAX endpoint charts identical.
    _alpha0, theta_jn, zeta, _phi_j = source_frame_geometry(
        m1, m2, x1, y1, z1, x2, y2, z2, inc, pc, fr,
    )
    if final_spin is None:
        af = msa_final_spin(m1, m2, x1, y1, z1, x2, y2, z2, f_ref=fr)
    else:
        af = torch.as_tensor(final_spin, dtype=t.dtype, device=t.device)
    eta = m1 * m2 / total.square()
    delta = (m1 - m2) / total
    mf = fits.IMRPhenomX_FinalMass2017(eta, z1, z2, delta)
    f22 = fits.IMRPhenomT_fringfit(af, 22) / mf
    f21 = fits.IMRPhenomT_fringfit(af, 21) / mf
    slope = (2 * math.pi / (total * MSUN_SEC)) * (f22 - f21)
    # ``f_eval`` is held at fp on post rows, so the last pre-peak sample is
    # obtained independently at the peak frequency for arbitrary grids.
    vp = (math.pi * total * MSUN_SEC * fp).pow(1.0 / 3.0)
    alpha_peak, beta_peak, gamma_peak, _, _ = _evaluate_msa_precession(
        vp, v0, m1, m2, x1, y1, z1, x2, y2, z2, inc, pc, prec_version=223,
    )

    def angle_evaluator(query):
        if frequency_at_time is None:
            raise ValueError("frequency_at_time is required for MSA numerical angle options")
        fq = torch.as_tensor(frequency_at_time(query), dtype=f.dtype, device=f.device)
        fq = torch.clamp(fq, min=torch.finfo(f.dtype).tiny)
        qfp = fp.reshape(fp.shape + (1,) * (query.ndim - fp.ndim))
        fq = torch.where(query >= 0, qfp, fq)
        qv = (math.pi * total.reshape(total.shape + (1,) * (query.ndim - total.ndim)) * MSUN_SEC * fq).pow(1.0 / 3.0)
        qv0 = v0.reshape(v0.shape + (1,) * (query.ndim - v0.ndim)).expand_as(qv)
        def qparams(value):
            return value.reshape(value.shape + (1,) * (query.ndim - value.ndim)).expand_as(qv)
        qa, qb, _, _, _ = _evaluate_msa_precession(
            qv, qv0, qparams(m1), qparams(m2), qparams(x1), qparams(y1), qparams(z1),
            qparams(x2), qparams(y2), qparams(z2), qparams(inc), qparams(pc), prec_version=223,
        )
        return qa, qb

    if suffix in (11, 21):
        if frequency_at_time is None or reference_time is None:
            raise ValueError("MSA numerical angle options require frequency_at_time and reference_time")
        if int(precession_version) % 100 == 21 and meco_time is None:
            raise ValueError("MSA option 22321 requires meco_time from the carrier model")
        vref = v0
        _, _, gamma_ref, _, _ = _evaluate_msa_precession(
            vref, vref, m1, m2, x1, y1, z1, x2, y2, z2, inc, pc, prec_version=223,
        )
        gamma_ref = gamma_ref[..., 0] if gamma_ref.ndim > gamma_peak.ndim else gamma_ref
    else:
        gamma_ref = None
    alpha, beta, gamma = apply_angle_option(
        t_grid, alpha, beta, gamma, alpha_peak, beta_peak, gamma_peak, slope,
        int(precession_version), angle_evaluator=angle_evaluator if gamma_ref is not None else None,
        meco_time=meco_time, reference_time=reference_time, gamma_reference=gamma_ref,
    )
    result = (alpha, beta, gamma, theta_jn, zeta)
    unsupported = ~torch.isfinite(af)
    if supplied_unsupported is not None:
        unsupported = unsupported | supplied_unsupported
    def mask_value(value):
        mask = unsupported
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, torch.full_like(value, torch.nan), value)
    return tuple(mask_value(value) for value in result)


__all__ = ["msa_geometry", "msa_final_spin"]

"""Native JAX time-domain IMRPhenomTP using the published NNLO angles."""

from __future__ import annotations

import math
import jax.numpy as jnp

from .imrphenomt import IMRPhenomT
from ._phenomt_precession import _canonicalize_components, nnlo_final_spin, nnlo_geometry, project_modes
from diffgw_core.constants import MSUN_SEC


class IMRPhenomTP(IMRPhenomT):
    """Precessing dominant-harmonic PhenomT (NNLO prescription)."""

    def time_domain(
        self,
        times,
        mass1,
        mass2,
        chi1=0.0,
        chi2=0.0,
        distance=100.0,
        inclination=0.0,
        phic=0.0,
        tc=0.0,
        f_ref=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin2x=0.0,
        spin2y=0.0,
        precession_method="nnlo",
        modes=None,
        final_spin=None,
    ):
        if str(precession_method).lower() != "nnlo":
            raise ValueError("IMRPhenomTP supports only precession_method='nnlo'")
        if modes is not None:
            requested = tuple((int(mode[0]), int(mode[1])) for mode in modes)
            if not requested or any((l, abs(m)) != (2, 2) for l, m in requested):
                raise ValueError("IMRPhenomTP supports only mode (2, 2)")
        t = jnp.asarray(times, dtype=jnp.float64)
        vals = [
            jnp.asarray(x, dtype=t.dtype)
            for x in (mass1, mass2, chi1, chi2, distance, inclination, phic, tc, f_ref, spin1x, spin1y, spin2x, spin2y)
        ]
        if final_spin is None:
            m1, m2, z1, z2, dist, inc, pc, tcenter, fr, x1, y1, x2, y2 = jnp.broadcast_arrays(*vals)
            fs = None
        else:
            vals.append(jnp.asarray(final_spin, dtype=t.dtype))
            m1, m2, z1, z2, dist, inc, pc, tcenter, fr, x1, y1, x2, y2, fs = jnp.broadcast_arrays(*vals)
        m1, m2, x1, y1, z1, x2, y2, z2 = _canonicalize_components(m1, m2, x1, y1, z1, x2, y2, z2)
        if fs is None:
            fs = nnlo_final_spin(m1, m2, x1, y1, z1, x2, y2, z2)
        hp0, hc0 = IMRPhenomT.time_domain(
            self,
            t,
            m1,
            m2,
            z1,
            z2,
            dist,
            0.0,
            0.0,
            tcenter,
            fr,
            _final_spin_override=fs,
        )
        h22 = -(hp0 - 1j * hc0) / math.sqrt(5.0 / (4.0 * math.pi))
        shifted = t.reshape((1,) * m1.ndim + (-1,)) - tcenter[..., None]
        # Eq. 14 uses the aligned carrier for the angle series; the augmented
        # remnant enters only the post-peak QNM continuation.
        state_angle = self._coeff(m1, m2, z1, z2)
        freq = self._evaluate(shifted / ((m1 + m2)[..., None] * MSUN_SEC), state_angle)[0]
        freq = freq / (2.0 * math.pi * (m1 + m2)[..., None] * MSUN_SEC)
        fpeak = state_angle[13] / (2.0 * math.pi * (m1 + m2) * MSUN_SEC)
        a, b, g, thjn, zeta = nnlo_geometry(
            shifted,
            freq,
            m1,
            m2,
            x1,
            y1,
            z1,
            x2,
            y2,
            z2,
            inc,
            pc,
            fr,
            frequency_peak=fpeak,
            final_spin=fs,
        )
        hp_rot, hc_rot = project_modes({(2, 2): h22}, a, b, g, thjn[..., None], zeta[..., None])
        hp_al, hc_al = IMRPhenomT.time_domain(
            self,
            t,
            m1,
            m2,
            z1,
            z2,
            dist,
            inc,
            pc,
            tcenter,
            fr,
            _final_spin_override=fs,
        )
        aligned = (x1**2 + y1**2 + x2**2 + y2**2) == 0
        return jnp.where(aligned[..., None], hp_al, hp_rot), jnp.where(aligned[..., None], hc_al, hc_rot)

    __call__ = time_domain
    forward = time_domain


__all__ = ["IMRPhenomTP"]

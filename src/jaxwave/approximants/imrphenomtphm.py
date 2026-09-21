"""Native JAX time-domain IMRPhenomTPHM with NNLO twisting."""

from __future__ import annotations

import math
import jax.numpy as jnp

from diffgw_core.constants import MSUN_SEC
from .imrphenomt import IMRPhenomT
from ._phenomt_precession import _canonicalize_components, nnlo_final_spin, nnlo_geometry, project_modes


DEFAULT_MODES = ((2, 2), (2, 1), (3, 3), (4, 4), (5, 5))
_SUPPORTED_MODES = frozenset(DEFAULT_MODES)


def _mode_key(key):
    if isinstance(key, str):
        p = key.find("-", 1)
        return (int(key[:p]), int(key[p:])) if p > 0 else (int(key[0]), int(key[1:]))
    return tuple(int(x) for x in key)


def _as_complex(value, dtype):
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return jnp.asarray(value[0], dtype=dtype) - 1j * jnp.asarray(value[1], dtype=dtype)
    return jnp.asarray(value, dtype=jnp.complex128)


class IMRPhenomTPHM(IMRPhenomT):
    """Precessing multipolar PhenomT; only the published NNLO path is enabled."""

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
            raise ValueError("IMRPhenomTPHM supports only precession_method='nnlo'")
        requested = tuple(_mode_key(mode) for mode in modes) if modes is not None else DEFAULT_MODES
        positive = tuple(dict.fromkeys((l, abs(m)) for l, m in requested))
        if not positive:
            raise ValueError("modes must contain at least one mode")
        if any(key not in _SUPPORTED_MODES for key in positive):
            raise ValueError(f"unsupported THM mode; supported positive modes are {sorted(_SUPPORTED_MODES)}")
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
        from .imrphenomthm import IMRPhenomTHM

        thm = IMRPhenomTHM()
        selected = tuple(positive) + tuple((l, -m) for l, m in positive)
        raw = thm.modes(
            t,
            m1,
            m2,
            z1,
            z2,
            dist,
            phic=0.0,
            tc=tcenter,
            mode_array=selected,
            final_spin=fs,
            f_ref=fr,
        )
        cp = {}
        for key, value in raw.items():
            lm = _mode_key(key)
            if lm in selected:
                cp[lm] = _as_complex(value, t.dtype)
        if not cp:
            raise ValueError("THM mode provider returned no requested positive modes")
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
        hp_rot, hc_rot = project_modes(cp, a, b, g, thjn[..., None], zeta[..., None])
        hp_al, hc_al = thm.time_domain(
            t,
            m1,
            m2,
            z1,
            z2,
            dist,
            inc,
            phic=pc,
            tc=tcenter,
            f_ref=fr,
            mode_array=selected,
            final_spin=fs,
        )
        aligned = (x1**2 + y1**2 + x2**2 + y2**2) == 0
        return jnp.where(aligned[..., None], hp_al, hp_rot), jnp.where(aligned[..., None], hc_al, hc_rot)

    __call__ = time_domain
    forward = time_domain


__all__ = ["IMRPhenomTPHM", "DEFAULT_MODES"]

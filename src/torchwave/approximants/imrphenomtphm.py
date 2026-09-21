"""Native time-domain IMRPhenomTPHM."""

from __future__ import annotations

import math
import torch

from diffgw_core.constants import MSUN_SEC

from .imrphenomt import IMRPhenomT
from .imrphenomtp import _mode_key
from ._phenomt_precession import _canonicalize_components, project_modes
from ._phenomt_dispatch import prepare_precession, precession_geometry, carrier_frequency
from diffgw_core.phenomt_options import resolve_precession


DEFAULT_MODES = ((2, 2), (2, 1), (3, 3), (4, 4), (5, 5))
_SUPPORTED_MODES = frozenset(DEFAULT_MODES)


def _as_complex_mode(value, dtype, device):
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return torch.as_tensor(value[0], dtype=dtype, device=device) - 1j * torch.as_tensor(
            value[1], dtype=dtype, device=device
        )
    return torch.as_tensor(value, dtype=torch.complex128, device=device)


class IMRPhenomTPHM(IMRPhenomT):
    """Precessing multipolar PhenomT model.

    Select the Euler-angle prescription with ``precession_method`` and its
    continuation with ``precession_version``. NNLO remains the default.
    Rotations and mode projection use native differentiable tensors.
    """

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
        precession_version=None,
        _precession_state=None,
    ):
        method, version = resolve_precession(precession_method, precession_version)
        requested = tuple(_mode_key(mode) for mode in modes) if modes is not None else DEFAULT_MODES
        positive = tuple(dict.fromkeys((l, abs(m)) for l, m in requested))
        if not positive:
            raise ValueError("modes must contain at least one mode")
        if any(key not in _SUPPORTED_MODES for key in positive):
            raise ValueError(f"unsupported THM mode; supported positive modes are {sorted(_SUPPORTED_MODES)}")
        t = torch.as_tensor(times, dtype=torch.float64)
        vals = [
            torch.as_tensor(x, dtype=t.dtype, device=t.device)
            for x in (mass1, mass2, chi1, chi2, distance, inclination, phic, tc, f_ref, spin1x, spin1y, spin2x, spin2y)
        ]
        if final_spin is None:
            m1, m2, z1, z2, dist, inc, pc, tcenter, fr, x1, y1, x2, y2 = torch.broadcast_tensors(*vals)
            fs = None
        else:
            vals.append(torch.as_tensor(final_spin, dtype=t.dtype, device=t.device))
            m1, m2, z1, z2, dist, inc, pc, tcenter, fr, x1, y1, x2, y2, fs = torch.broadcast_tensors(*vals)
        m1, m2, x1, y1, z1, x2, y2, z2 = _canonicalize_components(m1, m2, x1, y1, z1, x2, y2, z2)
        fs, prepared = prepare_precession(
            method, m1, m2, x1, y1, z1, x2, y2, z2, fr,
            final_spin=fs, prepared=_precession_state,
        )
        try:
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
        except ImportError as exc:
            raise RuntimeError("IMRPhenomTPHM requires the native IMRPhenomTHM implementation") from exc
        cp = {}
        for key, value in raw.items():
            lm = _mode_key(key)
            if lm in selected:
                cp[lm] = _as_complex_mode(value, t.dtype, t.device)
        if not cp:
            raise ValueError("THM mode provider returned no requested positive modes")
        # Eq. 14 uses the aligned carrier for the angle series; the augmented
        # remnant enters only the post-peak QNM continuation.
        state_angle = self._coeff(m1, m2, z1, z2)
        shifted = t.reshape((1,) * m1.ndim + (-1,)) - tcenter[..., None]
        freq = self._evaluate(shifted / ((m1 + m2)[..., None] * MSUN_SEC), state_angle)[0]
        freq = freq / (2.0 * math.pi * (m1 + m2)[..., None] * MSUN_SEC)
        fpeak = state_angle[13] / (2.0 * math.pi * (m1 + m2) * MSUN_SEC)
        a, b, g, thjn, zeta = precession_geometry(
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
            precession_method=method,
            precession_version=version,
            prepared=prepared,
            frequency_at_time=lambda query: carrier_frequency(self, state_angle, m1, m2, query),
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
        aligned = (x1.square() + y1.square() + x2.square() + y2.square()) == 0
        return torch.where(aligned[..., None], hp_al, hp_rot), torch.where(aligned[..., None], hc_al, hc_rot)

    forward = time_domain


__all__ = ["IMRPhenomTPHM", "DEFAULT_MODES"]

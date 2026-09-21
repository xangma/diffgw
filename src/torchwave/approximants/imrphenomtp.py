"""Native time-domain IMRPhenomTP (NNLO precession) in Torch."""

from __future__ import annotations

import math
import torch

from torchwave.constants import MSUN_SEC
from .imrphenomt import IMRPhenomT
from ._phenomt_precession import _canonicalize_components, nnlo_final_spin, nnlo_geometry, project_modes


def _mode_key(key):
    if isinstance(key, str):
        # Author mode labels are strings such as ``"22"`` and ``"2-2"``.
        if "-" in key[1:]:
            p = key.find("-", 1)
            return int(key[:p]), int(key[p:])
        return int(key[0]), int(key[1:])
    return tuple(int(x) for x in key)


class IMRPhenomTP(IMRPhenomT):
    """Precessing dominant-harmonic PhenomT.

    ``precession_method='nnlo'`` is the supported native prescription.  The
    implementation uses the published NNLO Euler-angle series and the
    differentiable finite Wigner-d rotation; numerical and MSA ODE variants
    are intentionally rejected until independently audited.
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
    ):
        if str(precession_method).lower() != "nnlo":
            raise ValueError("IMRPhenomTP supports only precession_method='nnlo'")
        if modes is not None:
            requested = tuple(_mode_key(mode) for mode in modes)
            if not requested or any((l, abs(m)) != (2, 2) for l, m in requested):
                raise ValueError("IMRPhenomTP supports only mode (2, 2)")
        return _project_t(
            self,
            times,
            mass1,
            mass2,
            chi1,
            chi2,
            distance,
            inclination,
            phic,
            tc,
            f_ref,
            spin1x,
            spin1y,
            spin2x,
            spin2y,
            final_spin=final_spin,
        )

    # IMRPhenomT assigns ``forward`` at class definition time, so bind the
    # override explicitly for nn.Module callers.
    forward = time_domain


def _project_t(
    model,
    times,
    mass1,
    mass2,
    chi1,
    chi2,
    distance,
    inclination,
    phic,
    tc,
    f_ref,
    spin1x,
    spin1y,
    spin2x,
    spin2y,
    modes=((2, 2),),
    final_spin=None,
):
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
    if fs is None:
        fs = nnlo_final_spin(m1, m2, x1, y1, z1, x2, y2, z2)
    # Construct the coprecessing 22 value from the face-on aligned carrier.
    hp0, hc0 = IMRPhenomT.time_domain(
        model,
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
    # Face-on T is h_22 * _{-2}Y_22(0), with Y_22=sqrt(5/(4 pi)).
    # The coprecessing provider uses the canonical positive carrier; the
    # face-on public polarization carries the conventional overall minus.
    h22 = -(hp0 - 1j * hc0) / math.sqrt(5.0 / (4.0 * math.pi))
    ts = t.reshape((1,) * m1.ndim + (-1,))
    shifted = ts - tcenter[..., None]
    # Eq. 14 uses the aligned carrier for the angle series; the augmented
    # remnant enters only the post-peak QNM continuation.
    state_angle = model._coeff(m1, m2, z1, z2)
    freq = model._evaluate(shifted / ((m1 + m2)[..., None] * MSUN_SEC), state_angle)[0]
    freq = freq / (2.0 * math.pi * (m1 + m2)[..., None] * MSUN_SEC)
    # The NNLO angle prescription uses the carrier peak for its post-merger
    # continuation.  Taking it from the coefficient state keeps that value
    # independent of the requested time grid.
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
    # In the aligned limit the NNLO Euler gauge is singular, while the
    # aligned T polarization is exact and continuous.  Select it analytically.
    hp_al, hc_al = IMRPhenomT.time_domain(
        model,
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
    aligned = (x1.square() + y1.square() + x2.square() + y2.square()) == 0
    return torch.where(aligned[..., None], hp_al, hp_rot), torch.where(aligned[..., None], hc_al, hc_rot)


__all__ = ["IMRPhenomTP"]

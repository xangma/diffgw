"""Published EOB mode components and finite-signal Fourier projection.

Complete unvalidated SEOBNR waveform assemblers are development-only."""

import math
from typing import NamedTuple

import torch

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC
from .attachment import AttachedMode22
from .hamiltonian import _ordinary_tensor_attributes
from .integrator import EOBTrajectoryIntegrator, FrequencyPeakTrajectory
from .seobnrv4_mode22 import (seobnrv4_published_delta22,
                              seobnrv4_published_mode22, seobnrv4_published_phase22)


_DIRECT_FREQUENCY_CHUNK = 32


class PublishedEOB22(NamedTuple):
    time: torch.Tensor  # (N,), t/M from the initial state
    mode: torch.Tensor  # (..., N), (D/M)*h22
    trajectory: FrequencyPeakTrajectory
    attachment: AttachedMode22
    final_mass_fraction: torch.Tensor
    final_spin: torch.Tensor


def _direct_quadrupole_fd(f, h22, time_s, dt_sec, inclination):
    """Project and integrate a finite-support 22 mode on an arbitrary grid.

    ``f`` has shape ``(..., F)`` and the other inputs have matching leading
    batch axes with a final time axis. The rectangular rule is evaluated
    directly at every recorded sample; frequency chunks bound each forward
    ``(..., chunk, N)`` kernel allocation while retaining autograd semantics.
    """
    cos_i = torch.cos(inclination).unsqueeze(-1)
    hp_td = math.sqrt(5.0 / (16.0 * math.pi)) * (1.0 + cos_i**2) * h22.real
    hc_td = -math.sqrt(5.0 / (4.0 * math.pi)) * cos_i * h22.imag
    dt_sec = dt_sec.unsqueeze(-1)
    plus_chunks = []
    cross_chunks = []
    for start in range(0, f.shape[-1], _DIRECT_FREQUENCY_CHUNK):
        stop = min(start + _DIRECT_FREQUENCY_CHUNK, f.shape[-1])
        kernel = torch.exp(
            (-2j * math.pi)
            * f[..., start:stop].unsqueeze(-1)
            * time_s.unsqueeze(-2)
        )
        plus_chunks.append(
            (kernel * hp_td.unsqueeze(-2)).sum(dim=-1) * dt_sec
        )
        cross_chunks.append(
            (kernel * hc_td.unsqueeze(-2)).sum(dim=-1) * dt_sec
        )
    return torch.cat(plus_chunks, dim=-1), torch.cat(cross_chunks, dim=-1)


def published_eob22_to_fd(result, f, mass1, mass2, distance=100., inclination=0.,
                          phic=0., tc=0., *, time_origin='orbital_frequency_peak'):
    """Transform a reconstructed published 22 mode into FD polarizations.

    ``result.time`` is the geometric-time chart whose zero is the initial
    state.  The rectangular Fourier sum is evaluated directly on the physical
    chart ``tc + (result.time - origin) * M``.  The default origin is the
    equation-level maximum of the continuous orbital frequency.  The
    ``nqc_match`` option retains the historical attachment-time origin for
    controlled comparisons.
    """
    if time_origin == 'orbital_frequency_peak':
        origin = result.trajectory.peak.time
    elif time_origin == 'nqc_match':
        origin = result.attachment.match_time
    else:
        raise ValueError("time_origin must be 'orbital_frequency_peak' or 'nqc_match'")
    if (not isinstance(f, torch.Tensor) or f.ndim < 1
            or f.dtype not in (torch.float32, torch.float64)
            or f.shape[-1] == 0):
        raise ValueError("f must be a nonempty finite nonnegative float32/float64 frequency tensor")
    if not bool(torch.isfinite(f).all()) or bool((f < 0).any()):
        raise ValueError("f must be a finite nonnegative float32/float64 frequency tensor")
    time = result.time
    if (not isinstance(time, torch.Tensor) or time.ndim != 1
            or time.numel() < 2 or time.dtype not in (torch.float32, torch.float64)):
        raise ValueError("result.time must be a one-dimensional time tensor with at least two samples")
    time = time.to(device=f.device, dtype=f.dtype)
    if not bool(torch.isfinite(time).all()):
        raise ValueError("result.time must be finite")
    dt_geom = time[1] - time[0]
    if not bool(torch.isfinite(dt_geom)) or not bool(dt_geom > 0):
        raise ValueError("result.time must have a positive uniform sample spacing")
    if not bool(torch.allclose(time[1:] - time[:-1], dt_geom,
                               rtol=1e-5 if f.dtype == torch.float32 else 1e-10,
                               atol=1e-6 if f.dtype == torch.float32 else 1e-12)):
        raise ValueError("result.time must have a uniform sample spacing")
    values = [torch.as_tensor(value, dtype=f.dtype, device=f.device)
              for value in (mass1, mass2, distance, inclination, phic, tc)]
    m1, m2, dist, inc, phase, delay = torch.broadcast_tensors(*values)
    if (not all(bool(torch.isfinite(value).all()) for value in values)
            or bool((m1 <= 0).any()) or bool((m2 <= 0).any())
            or bool((dist <= 0).any())):
        raise ValueError("masses and physical parameters must be finite, masses positive, and distance positive")
    origin = torch.as_tensor(origin, dtype=f.dtype, device=f.device)
    if not bool(torch.isfinite(origin).all()):
        raise ValueError("selected time origin must be finite")
    mode = result.mode
    if (not isinstance(mode, torch.Tensor) or mode.ndim < 1
            or mode.shape[-1] != time.numel()):
        raise ValueError("result.mode must end in the result.time sample dimension")
    if not mode.is_complex() or not bool(torch.isfinite(mode).all()):
        raise ValueError("result.mode must be a finite complex tensor")
    mode = mode.to(device=f.device)
    # Treat the final frequency axis like the final sample axis of mode.  Any
    # frequency-grid batch axes therefore participate in ordinary leading
    # broadcasting, and malformed combinations fail here rather than inside
    # the interpolation kernel with a less useful shape error.
    batch_shape = torch.broadcast_shapes(f.shape[:-1], mode.shape[:-1], m1.shape, m2.shape,
                                         dist.shape, inc.shape, phase.shape,
                                         delay.shape, origin.shape)
    f = f.expand(batch_shape + (f.shape[-1],))
    mode = mode.expand(batch_shape + (time.numel(),))
    m1, m2, dist, inc, phase, delay, origin = (
        value.expand(batch_shape)
        for value in (m1, m2, dist, inc, phase, delay, origin)
    )
    m_sec = (m1 + m2) * MSUN_SEC
    mode = mode * torch.exp(-2j * phase.unsqueeze(-1))
    mode = mode * (m_sec * C_SI / (dist * MPC_SI)).unsqueeze(-1)
    absolute_time_s = (
        delay.unsqueeze(-1)
        + (time - origin.unsqueeze(-1)) * m_sec.unsqueeze(-1)
    )
    return _direct_quadrupole_fd(
        f, mode, absolute_time_s, dt_geom * m_sec, inc
    )


def published_mode22_and_frequency(state, hamiltonian, radiation_reaction, *,
                                   spin_v6, include_pan_v8, include_positive_chi_540,
                                   retain_pan_cubic_spin=False):
    """Mode and continuous positive GW frequency along the full vector field.

    State ends in (..., samples, 4), with Hamiltonian parameters on preceding
    batch axes. The directional derivative includes tail/residual evolution
    and radiation reaction. A local autograd graph works in no_grad/inference
    mode, while a differentiating caller retains state and parameter gradients.
    """
    track = torch.is_grad_enabled() and (state.requires_grad or any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for value in vars(hamiltonian).values()))
    with torch.inference_mode(False), torch.enable_grad():
        local_ham = _ordinary_tensor_attributes(hamiltonian)
        local_state = state.clone()
        field = EOBTrajectoryIntegrator(local_ham, radiation_reaction).vector_field(local_state)
        shift = torch.zeros_like(local_state[..., 0], requires_grad=True)
        shifted = local_state+shift.unsqueeze(-1)*field
        quantities = radiation_reaction.from_hamiltonian(shifted, local_ham)
        residual = seobnrv4_published_delta22(
            **{key: quantities[key] for key in ('omega', 'hreal_over_M', 'x1', 'x2', 'chi1', 'chi2')},
            spin_v6=spin_v6, include_pan_v8=include_pan_v8,
            include_positive_chi_540=include_positive_chi_540)
        phase = seobnrv4_published_phase22(quantities['omega'], shifted[..., 1],
            hreal_over_M=quantities['hreal_over_M'], residual_phase=residual)
        mode = seobnrv4_published_mode22(
            phi=shifted[..., 1], residual_phase=residual, **quantities,
            retain_pan_cubic_spin=retain_pan_cubic_spin,
        )
        frequency = torch.autograd.grad(phase.sum(), shift, create_graph=track)[0]
    return (mode, frequency) if track else (mode.detach(), frequency.detach())

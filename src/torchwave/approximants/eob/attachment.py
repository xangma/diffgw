"""Trajectory-to-merger wiring for the published SEOBNRv4 matching equations.

The caller supplies the complex inspiral mode, its continuous positive GW
frequency, and the fundamental remnant QNM. No unresolved residual-phase or
remnant-fit selection is inferred here. This is an experimental reconstruction
with an explicit local interpolation, not a claim of LAL implementation parity.
"""

from typing import NamedTuple

import torch

from .events import OrbitalFrequencyPeak, orbital_frequency_peak
from .nqc import (NQCCalibration, calibrate_nqc_22, nqc_22_basis, nqc_22_factor,
                  seobnrv4_nqc_targets, seobnrv4_nqc_time_offset)
from .merger_ringdown import merger_ringdown_22, seobnrv4_merger_shape


class AttachedMode22(NamedTuple):
    mode: torch.Tensor  # (..., N_output), dimensionless (D/M)*h22
    corrected_inspiral: torch.Tensor  # (..., N_input)
    peak: OrbitalFrequencyPeak
    match_time: torch.Tensor
    match_phase: torch.Tensor  # h=A*exp(-i phase)
    calibration: NQCCalibration


def _local_mode_interpolate(query, time, mode):
    """Seven-point degree-six Lagrange interpolation on the source grid."""
    count = time.shape[-1]
    index = torch.searchsorted(time.contiguous(), query.contiguous()).clamp(3, count-4)
    offsets = torch.arange(-3, 4, device=time.device)
    indices = (index.unsqueeze(-1) + offsets).clamp(0, count-1)
    nodes = torch.gather(time.unsqueeze(-2).expand(*time.shape[:-1], query.shape[-1], count),
                         -1, indices)
    ordinates = torch.gather(mode.unsqueeze(-2).expand(*mode.shape[:-1], query.shape[-1], count),
                             -1, indices)
    weights = torch.empty_like(nodes)
    differences = query.unsqueeze(-1)-nodes
    for j in range(7):
        mask = torch.arange(7, device=time.device) != j
        numerator = differences[..., mask].prod(-1)
        denominator = (nodes[..., j:j+1]-nodes[..., mask]).prod(-1)
        weights[..., j] = numerator/denominator
    result = (ordinates*weights).sum(-1)
    return torch.where((query < time[..., :1]) | (query > time[..., -1:]), 0, result)


def attach_seobnrv4_22(time, state, orbital_frequency, base_mode, base_frequency,
                       nu, chi, sigma, output_time, *, valid=None):
    """Locate Omega's peak, solve all five NQC conditions, and attach ringdown.

    Input samples are on the final axis, except ``state`` is (..., N, 4) in
    [r, phi, p_r*, p_phi] order. All quantities use initial total mass M=1;
    ``base_mode`` is (D/M)*h22 and ``base_frequency`` is -d(arg h22)/d(t/M),
    including the chosen tail and residual phases. Do not substitute 2*Omega.
    ``chi`` is Bohé Eq. (2.8), and ``sigma`` is M_initial*(omega220-i/tau220).
    The caller must choose the final-spin fit/QNM convention explicitly.

    Eqs. (4.9),(4.15) set t_match=t_Omega_peak+Delta_t22_peak. Seven local
    samples then determine the Eq. (2.7) NQC coefficients and Appendix A's
    five target conditions. Between that stencil's first time and t_match,
    the same degree-six amplitude/basis/frequency polynomials define a continuous
    mode: integrating the positive frequency avoids phase unwrapping. Its
    phase is anchored to the first stencil sample. This gives an exact C1
    join to Eqs. (2.10),(2.12)-(2.17) at the imposed peak targets. Earlier
    corrected samples are locally interpolated on the source grid; times before input start
    are zero. Output samples occupy the last axis and may be batched.

    Peak/stencil selection is discrete. Within a selected stencil gradients
    are retained. This interpolation is an explicit numerical choice; reduce
    trajectory spacing to check convergence. Unresolved peaks, insufficient
    pre-peak history, or ill-conditioned matching fail without a fallback.
    """
    reals = (time, state, orbital_frequency, base_frequency, nu, chi, output_time)
    if not all(isinstance(value, torch.Tensor) for value in (*reals, base_mode, sigma)):
        raise TypeError("attachment inputs must be torch tensors")
    if time.dtype not in (torch.float32, torch.float64):
        raise TypeError("attachment requires float32/float64 inputs")
    if any(value.dtype != time.dtype or value.device != time.device for value in reals):
        raise TypeError("real attachment inputs must share time's dtype and device")
    complex_dtype = torch.complex128 if time.dtype == torch.float64 else torch.complex64
    if any(value.dtype != complex_dtype or value.device != time.device for value in (base_mode, sigma)):
        raise TypeError("base_mode and sigma must use the corresponding complex dtype/device")
    if any(not bool(torch.isfinite(value).all()) for value in (*reals, base_mode, sigma)):
        raise ValueError("attachment inputs must be finite")
    if (time.ndim < 1 or state.ndim < 2 or state.shape[-1] != 4 or output_time.ndim < 1
            or any(value.ndim < 1 or value.shape[-1] != time.shape[-1]
                   for value in (orbital_frequency, base_mode, base_frequency))
            or state.shape[-2] != time.shape[-1]):
        raise ValueError("input arrays must share the specified sample axis and state must end in (N,4)")
    count = time.shape[-1]
    batch = torch.broadcast_shapes(time.shape[:-1], state.shape[:-2], orbital_frequency.shape[:-1],
                                   base_mode.shape[:-1], base_frequency.shape[:-1], nu.shape, chi.shape,
                                   sigma.shape, output_time.shape[:-1])
    time, omega, base, frequency = (value.expand(batch+(count,)) for value in
                                   (time, orbital_frequency, base_mode, base_frequency))
    state = state.expand(batch+(count, 4))
    nu, chi, sigma = (value.expand(batch) for value in (nu, chi, sigma))
    query = output_time.expand(batch+(output_time.shape[-1],))
    if valid is None:
        valid = torch.ones_like(time, dtype=torch.bool)
    peak = orbital_frequency_peak(time, omega, valid=valid)
    valid = valid.expand(time.shape)
    match = peak.time+seobnrv4_nqc_time_offset(nu, chi)
    index = torch.searchsorted(time.contiguous(), match.unsqueeze(-1).contiguous()).squeeze(-1).clamp(1, count-1)
    a, b = torch.gather(time, -1, (index-1).unsqueeze(-1)).squeeze(-1), torch.gather(time, -1, index.unsqueeze(-1)).squeeze(-1)
    index = torch.where(match-a < b-match, index-1, index)
    if bool(((index < 3) | (index+3 >= valid.sum(-1))).any()):
        raise ValueError("NQC matching needs three valid samples on both sides; extend trajectory history")
    indices = index.unsqueeze(-1)+torch.arange(-3, 4, device=time.device)
    stencil = lambda value: torch.gather(value, -1, indices)
    t, r, pr, om, amplitude, freq = (stencil(value) for value in
        (time, state[..., 0], state[..., 2], omega, base.abs(), frequency))
    targets = seobnrv4_nqc_targets(nu, chi)
    calibration = calibrate_nqc_22(t, r, pr, om, amplitude, freq, match, targets,
                                    degree=6)
    corrected = base*nqc_22_factor(state[..., 0], state[..., 2], omega,
                                   calibration.coefficients, sample_dims=1)

    q, p = nqc_22_basis(r, pr, om)
    span = t[..., -1]-t[..., 0]
    x = (t-match.unsqueeze(-1))/span.unsqueeze(-1)
    matrix = torch.stack(tuple(x.pow(k) for k in range(7)), dim=-1)
    ordinates = torch.cat((amplitude.unsqueeze(-1), amplitude.unsqueeze(-1)*q, p, freq.unsqueeze(-1)), -1)
    polynomial = torch.linalg.solve(matrix, ordinates)
    amp_coeff, phase_coeff = calibration.coefficients
    # Integral from the first sample anchors a continuous phase without
    # selecting any arbitrary 2*pi branch outside this local interval.
    anchor_x = x[..., 0]
    anchor_phase = -torch.angle(stencil(base)[..., 0])

    def local_mode(z):
        values = sum(polynomial[..., k, :].unsqueeze(-2)*z.unsqueeze(-1).pow(k) for k in range(7))
        amp = values[..., 0]+(values[..., 1:4]*amp_coeff.unsqueeze(-2)).sum(-1)
        integral = sum(polynomial[..., k, 6].unsqueeze(-1)
                       *(z.pow(k+1)-anchor_x.unsqueeze(-1).pow(k+1))/(k+1) for k in range(7))
        phase = (anchor_phase.unsqueeze(-1)+span.unsqueeze(-1)*integral
                 -(values[..., 4:6]*phase_coeff.unsqueeze(-2)).sum(-1))
        return amp*torch.exp(-1j*phase), phase

    _, phase_at_match = local_mode(torch.zeros_like(match).unsqueeze(-1))
    match_phase = phase_at_match.squeeze(-1)
    # Avoid evaluating the local polynomial far outside its stencil, even
    # though that branch would be discarded by the final selection.
    local_query = query.clamp(min=t[..., :1], max=match.unsqueeze(-1))
    near, _ = local_mode((local_query-match.unsqueeze(-1))/span.unsqueeze(-1))
    inspiral = torch.where(query >= t[..., :1], near, _local_mode_interpolate(query, time, corrected))
    ringdown = merger_ringdown_22((query-match.unsqueeze(-1)).clamp_min(0), sigma,
        targets.amplitude, targets.amplitude_derivative, targets.frequency,
        match_phase, seobnrv4_merger_shape(nu, chi))
    mode = torch.where(query < match.unsqueeze(-1), inspiral, ringdown)
    return AttachedMode22(mode, corrected, peak, match, match_phase, calibration)

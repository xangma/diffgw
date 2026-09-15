"""Experimental end-to-end assembly of the published SEOBNRv4 equations.

Unresolved phase inheritance, spin truncation and final-spin fit order are
required choices. This is not an implementation-compatible SEOBNRv4 alias.
See docs/reviews/2026-09-12-clean-room/pass-6-eob-driver.md.
"""

import math
from typing import NamedTuple

import torch

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm
from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention
from .attachment import AttachedMode22, attach_seobnrv4_22
from .numerical_attachment import attach_seobnrv4_22_numerical_compatibility
from .hamiltonian import _ordinary_tensor_attributes
from .initial_conditions import find_quasicircular_initial_conditions
from .integrator import EOBTrajectoryIntegrator, FrequencyPeakTrajectory
from .nqc import (seobnrv4_nqc_parameters, seobnrv4_nqc_time_offset,
                  seobnrv4_nqc_time_offset_numerical_compatibility)
from .remnant import eob_final_mass_fraction, hofmann_final_spin
from .seobnrv4_flux import SEOBNRv4PublishedFlux
from .seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
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


def reconstruct_seobnrv4_22(omega_start, mass1, mass2, chi1=0., chi2=0., *,
                            spin_truncation, spin_v6, include_pan_v8,
                            include_positive_chi_540, final_spin_fit_order,
                            dt_geom=.5, num_steps=20000, ringdown_duration=150.,
                            richardson_extrapolation=True,
                            numerical_compatibility=False):
    """Integrate, calibrate all five NQC conditions, and attach the (2,2) mode.

    omega_start is positive M_initial*Omega; masses share any positive unit.
    Times and QNM frequencies use the initial total mass. The selected signed
    Hofmann fit must yield 0<=a_f<=0.99999, the independently solved QNM domain.
    No absolute value, clipping, remnant-model fallback or fit choice is used.

    The returned time starts at initial orbital phase zero. Output spacing is
    dt_geom; ringdown is truncated ringdown_duration M after each row's match.
    Its final zero padding can vary by row. No start/end taper is applied.
    Event/stencil/sample-count selections are discrete; gradients within each
    selected branch are retained. Float64 and a time-step convergence check
    are recommended for the NQC linear systems.
    """
    if not isinstance(omega_start, torch.Tensor) or omega_start.dtype not in (torch.float32, torch.float64):
        raise TypeError("omega_start must be a float32/float64 tensor")
    if not math.isfinite(ringdown_duration) or ringdown_duration <= 0:
        raise ValueError("ringdown_duration must be finite and positive")
    if not isinstance(richardson_extrapolation, bool):
        raise TypeError("richardson_extrapolation must be a bool")
    if not isinstance(numerical_compatibility, bool):
        raise TypeError("numerical_compatibility must be a bool")
    if numerical_compatibility and (
            spin_v6 != 'bohe_total_energy' or include_pan_v8 is not False
            or include_positive_chi_540 is not False):
        raise ValueError(
            "numerical compatibility requires spin_v6='bohe_total_energy', "
            "include_pan_v8=False, and include_positive_chi_540=False")
    if numerical_compatibility:
        # The compatibility branch is deliberately opt-in.  Keep the
        # published default below on its established fixed-step path while
        # routing this flag to the independent bounded RKF45/two-stage
        # implementation.
        from .adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
        return reconstruct_seobnrv4_22_adaptive(
            omega_start, mass1, mass2, chi1, chi2,
            spin_truncation=spin_truncation, spin_v6=spin_v6,
            include_pan_v8=include_pan_v8,
            include_positive_chi_540=include_positive_chi_540,
            final_spin_fit_order=final_spin_fit_order,
            dt_geom=dt_geom, num_steps=num_steps,
            ringdown_duration=ringdown_duration,
        )
    values = [torch.as_tensor(value, dtype=omega_start.dtype, device=omega_start.device)
              for value in (mass1, mass2, chi1, chi2)]
    start, m1, m2, s1, s2 = torch.broadcast_tensors(omega_start, *values)
    if not bool((torch.isfinite(start) & (start > 0)).all()):
        raise ValueError("omega_start must be finite and positive")
    # Validate remnant/phase choices before starting the expensive trajectory.
    mass_fraction = eob_final_mass_fraction(m1, m2, s1, s2)
    final_spin = hofmann_final_spin(m1, m2, s1, s2, fit_order=final_spin_fit_order)
    if numerical_compatibility:
        # The compatibility table is regenerated from the independent
        # angular QNM helper and is deliberately opt-in; the paper path keeps
        # its direct solver convention.
        qnm_w, qnm_damp = signed_qnm_220_table_convention(final_spin)
    else:
        qnm_w, qnm_damp = dimensionless_qnm(2, 2, 0, final_spin)
    sigma = torch.complex(qnm_w, -qnm_damp)/mass_fraction
    scaled1, scaled2 = m1/torch.maximum(m1, m2), m2/torch.maximum(m1, m2)
    x1, x2 = scaled1/(scaled1+scaled2), scaled2/(scaled1+scaled2)
    nu, chi = seobnrv4_nqc_parameters(x1, x2, s1, s2)
    phase_choices = dict(spin_v6=spin_v6, include_pan_v8=include_pan_v8,
                         include_positive_chi_540=include_positive_chi_540)
    seobnrv4_published_delta22(start, hreal_over_M=torch.ones_like(start),
        x1=x1, x2=x2, chi1=s1, chi2=s2, **phase_choices)
    ham = SEOBNRv4Hamiltonian(nu, s1, s2, delta=x1-x2)
    rr = SEOBNRv4PublishedFlux(
        spin_truncation=spin_truncation,
        retain_pan_cubic_spin=numerical_compatibility,
    )
    r, pr, j = find_quasicircular_initial_conditions(ham, rr, start)
    initial = torch.stack((r, torch.zeros_like(r), pr, j), -1)
    # One extra interval covers the sub-sample refinement of the maximum.
    nqc_time_offset = (seobnrv4_nqc_time_offset_numerical_compatibility
                       if numerical_compatibility else seobnrv4_nqc_time_offset)
    after = nqc_time_offset(nu, chi).clamp_min(0)+dt_geom
    trajectory = EOBTrajectoryIntegrator(ham, rr).integrate_to_frequency_peak(
        initial, dt_geom, num_steps, post_peak_duration=after,
        richardson_extrapolation=richardson_extrapolation)
    state = trajectory.state.movedim(0, -2)
    mode, frequency = published_mode22_and_frequency(
        state, ham, rr, **phase_choices,
        retain_pan_cubic_spin=numerical_compatibility,
    )
    match = trajectory.peak.time+nqc_time_offset(nu, chi)
    count = math.ceil(float((match.max()+ringdown_duration).detach())/dt_geom)+1
    time = torch.arange(count, dtype=start.dtype, device=start.device)*dt_geom
    attachment_args = (trajectory.time, state,
        trajectory.frequency.movedim(0, -1), mode, frequency, nu, chi, sigma, time)
    attachment_kwargs = dict(valid=trajectory.valid.movedim(0, -1))
    if numerical_compatibility:
        attachment = attach_seobnrv4_22_numerical_compatibility(
            *attachment_args, match_time=match, **attachment_kwargs)
    else:
        attachment = attach_seobnrv4_22(*attachment_args, **attachment_kwargs)
    output = torch.where(time <= match.unsqueeze(-1)+ringdown_duration,
                         attachment.mode, torch.zeros_like(attachment.mode))
    return PublishedEOB22(time, output, trajectory, attachment, mass_fraction, final_spin)


class SEOBNRv4Published:
    """Experimental published-equation quadrupole, with explicit model choices."""

    @staticmethod
    def __call__(f, mass1, mass2, chi1=0., chi2=0., distance=100., inclination=0.,
                 phic=0., tc=0., f_lower=None, *, spin_truncation, spin_v6,
                 include_pan_v8, include_positive_chi_540, final_spin_fit_order,
                 dt_geom=.5, num_steps=20000, ringdown_duration=150.,
                 richardson_extrapolation=True,
                 numerical_compatibility=False,
                 time_origin='orbital_frequency_peak'):
        """FD paired spherical quadrupoles with an explicit time origin.

        f and f_lower are Hz, masses solar, distance Mpc, tc seconds. phic
        rotates the orbital mode by exp(-2i*phic). The default
        ``orbital_frequency_peak`` origin places the continuous orbital
        frequency maximum at tc; ``nqc_match`` retains the historical NQC
        attachment-time convention. Both use an exp(-2*pi*i*f*t) Fourier
        transform of the real polarizations. f_lower determines the initial
        orbital frequency pi*M*f_lower, including the usual distinction
        between twice orbital and tail-corrected instantaneous GW frequency.
        """
        if time_origin not in ('orbital_frequency_peak', 'nqc_match'):
            raise ValueError("time_origin must be 'orbital_frequency_peak' or 'nqc_match'")
        if (not isinstance(f, torch.Tensor) or f.ndim < 1
                or f.dtype not in (torch.float32, torch.float64)
                or not bool(torch.isfinite(f).all()) or bool((f < 0).any())):
            raise ValueError("f must be a finite nonnegative float32/float64 frequency tensor")
        values = [torch.as_tensor(value, dtype=f.dtype, device=f.device)
                  for value in (mass1, mass2, chi1, chi2, distance, inclination, phic, tc)]
        m1, m2, s1, s2, dist, inc, phase, delay = torch.broadcast_tensors(*values)
        if (not all(bool(torch.isfinite(value).all()) for value in values)
                or bool((m1 <= 0).any()) or bool((m2 <= 0).any())
                or bool((dist <= 0).any())):
            raise ValueError("masses and physical parameters must be finite, masses positive, and distance positive")
        if f_lower is None:
            positive = f[f > 0]
            if not positive.numel():
                raise ValueError("f_lower is required when the frequency grid has no positive samples")
            lower = positive.min()
        else:
            lower = torch.as_tensor(f_lower, dtype=f.dtype, device=f.device)
            if (not bool(torch.isfinite(lower).all())
                    or bool((lower <= 0).any())):
                raise ValueError("f_lower must be finite and positive")
        m_sec = (m1+m2)*MSUN_SEC
        reconstruction_kwargs = dict(
            spin_truncation=spin_truncation,
            final_spin_fit_order=final_spin_fit_order,
            spin_v6=spin_v6, include_pan_v8=include_pan_v8,
            include_positive_chi_540=include_positive_chi_540,
            dt_geom=dt_geom, num_steps=num_steps,
            ringdown_duration=ringdown_duration,
            richardson_extrapolation=richardson_extrapolation,
        )
        if numerical_compatibility:
            reconstruction_kwargs['numerical_compatibility'] = True
        result = reconstruct_seobnrv4_22(
            math.pi*m_sec*lower, m1, m2, s1, s2, **reconstruction_kwargs)
        return published_eob22_to_fd(
            result, f, m1, m2, distance=dist, inclination=inc,
            phic=phase, tc=delay, time_origin=time_origin)


class SEOBNRv4PublishedNonspinning(SEOBNRv4Published):
    """Published-equation driver with a fixed, explicit nonspinning policy.

    This preset is useful when spin parameters are physically irrelevant.  It
    supplies the published phase, flux, and Hofmann fit choices used by the
    clean-room nonspinning reconstruction and rejects nonzero aligned spins so
    they cannot be silently ignored.
    """

    @staticmethod
    def __call__(f, mass1, mass2, chi1=0., chi2=0., **kwargs):
        s1 = torch.as_tensor(chi1, dtype=f.dtype, device=f.device)
        s2 = torch.as_tensor(chi2, dtype=f.dtype, device=f.device)
        if bool((s1 != 0).any()) or bool((s2 != 0).any()):
            raise ValueError("SEOBNRv4PublishedNonspinning requires chi1=chi2=0")
        if 'final_spin_fit_order' not in kwargs:
            raise TypeError("final_spin_fit_order must be explicit for the nonspinning preset")
        fit_order = kwargs.pop('final_spin_fit_order')
        choices = dict(spin_truncation='tabulated',
                       spin_v6='bohe_total_energy',
                       include_pan_v8=False,
                       include_positive_chi_540=False,
                       final_spin_fit_order=fit_order)
        overlap = set(choices).intersection(kwargs)
        if overlap:
            raise ValueError("nonspinning preset fixes " + ", ".join(sorted(overlap)))
        return SEOBNRv4Published.__call__(f, mass1, mass2, chi1=0., chi2=0.,
                                           **kwargs, **choices)

"""Development-only prototype; scientific qualification is incomplete.

Experimental end-to-end assembly of the published SEOBNRv4 equations.

Unresolved phase inheritance, spin truncation and final-spin fit order are
required choices. This is not an implementation-compatible SEOBNRv4 alias.
See docs/reviews/2026-09-12-clean-room/pass-6-eob-driver.md.
"""

import math
import torch
from torchwave.constants import MSUN_SEC
from torchwave.approximants.kerr_qnm_generated import dimensionless_qnm
from torchwave.approximants.kerr_qnm_table_convention import signed_qnm_220_table_convention
from torchwave.approximants.eob.attachment import attach_seobnrv4_22
from torchwave.approximants.eob.numerical_attachment import (
    attach_seobnrv4_22_numerical_compatibility,
)
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.nqc import (seobnrv4_nqc_parameters, seobnrv4_nqc_time_offset,
                  seobnrv4_nqc_time_offset_numerical_compatibility)
from torchwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from torchwave.approximants.eob.seobnrv4_mode22 import seobnrv4_published_delta22

from torchwave.approximants.eob.published_waveform import (
    PublishedEOB22,
    published_eob22_to_fd,
    published_mode22_and_frequency,
)

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
        from dev.native_rom.eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
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

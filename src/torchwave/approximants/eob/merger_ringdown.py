"""Published SEOBNRv4 merger-ringdown ansatz, Bohé et al. 1611.03703v3.

Eqs. (2.10), (2.12)-(2.17), with constrained coefficients derived from C1
matching. Times are (t-t_match)/M_initial; sigma=M_initial*(omega-i/tau).
The caller supplies the QNM and matching data. No remnant fit, trajectory
peak selection, or inspiral attachment is implicit in this component.
"""

from typing import NamedTuple

import torch


class MergerRingdownShape(NamedTuple):
    amplitude_rate: torch.Tensor
    amplitude_offset: torch.Tensor
    phase_rate: torch.Tensor
    phase_offset: torch.Tensor


def seobnrv4_merger_shape(nu, chi):
    """Free coefficients c1f,c2f,d1f,d2f, Eqs. (2.13),(2.14),(2.16),(2.17).

    chi is Eq. (2.8)'s effective spin, also returned by
    ``nqc.seobnrv4_nqc_parameters``. These are the printed NR fits.
    """
    if not isinstance(nu, torch.Tensor) or not isinstance(chi, torch.Tensor):
        raise TypeError("nu and chi must be floating torch tensors")
    if nu.dtype not in (torch.float32, torch.float64) or chi.dtype != nu.dtype or chi.device != nu.device:
        raise TypeError("nu and chi must share float32/float64 dtype and device")
    nu, chi = torch.broadcast_tensors(nu, chi)
    if bool((~torch.isfinite(nu) | ~torch.isfinite(chi) | (nu < 0) | (nu > .25) | (chi.abs() > 1)).any()):
        raise ValueError("published merger fits require 0 <= nu <= 1/4 and |chi| <= 1")
    c1 = (-.0893454*nu.square() + .0612892*nu + .00146142*nu*chi
          - .0136459*chi.square() - .0196758*chi + .0830664)
    c2 = (-1.82173*nu.square() - 5.25339*nu.square()*chi + 2.40203*nu*chi
          + 1.39777*nu - .371365*chi - .623953)
    d1 = (-.808987*nu.square() + .263456*nu - .120853*nu*chi
          - .0244358*chi.square() + .00779176*chi + .147584)
    d2 = (17.5646*nu.square() - 6.99396*nu - 9.61861*nu*chi
          + .581626*chi.square() + 3.13067*chi + 2.46654)
    return MergerRingdownShape(c1, c2, d1, d2)


def seobnrv4_peak_preserving_shape(sigma, shape):
    """Continue the printed amplitude fit to a non-increasing peak attachment.

    Bohé et al. (1611.03703v3), Eqs. (2.10), (2.12)-(2.14), imply
    A''(0)/A(0) = -gamma * [gamma + 2*c1f*tanh(c2f)] at A'(0)=0,
    where gamma=-Im(sigma)>0.  If the printed free rate c1f makes the
    bracket negative, project only c1f to its nearest admissible boundary.
    This is the ansatz-derived guard in historical SEOBNRv4 ringdown source
    (LALSimIMREOBHybridRingdown.c, Eqs. (57)-(58) of LIGO-T1600383).
    """
    c1, c2, d1, d2 = shape
    damping = -sigma.imag
    offset = torch.tanh(c2)
    safe_offset = torch.where(offset < 0, offset, torch.ones_like(offset))
    boundary_rate = -damping / (2 * safe_offset)
    rate = torch.where(damping + 2*c1*offset < 0, boundary_rate, c1)
    return MergerRingdownShape(rate, c2, d1, d2)


def _log1p_over_x(x):
    # Analytic continuation at zero. This also keeps gradients finite when
    # the published d2f fit crosses zero inside the physical parameter domain.
    small = x.abs() < torch.finfo(x.dtype).eps**.2
    safe = torch.where(small, torch.ones_like(x), x)
    polynomial = 1 + x*(-.5 + x*(1/3 + x*(-.25 + x*.2)))
    return torch.where(small, polynomial, torch.log1p(x)/safe)


def merger_ringdown_22(time_since_match, sigma, amplitude, amplitude_derivative,
                       frequency, phase, shape):
    """C1 matched complex h22 with convention h=A*exp(-i*phase).

    ``time_since_match`` has samples on its last axis, with optional leading
    batch axes. All other arguments describe broadcastable parameter batches.
    ``sigma`` is a damped positive-frequency complex QNM in initial-mass
    units; the published fit is calibrated to the fundamental (2,2,0) QNM.
    Amplitude normalization is inherited from the supplied matching
    value (for example D_L/M times strain); no extra nu factor is inserted.

    The paper attaches at an amplitude peak, where amplitude_derivative=0.
    A nonzero derivative is accepted to make the general C1 conditions
    explicit, but does not implement the calibrated peak prescription.
    """
    reals = (time_since_match, amplitude, amplitude_derivative, frequency, phase, *shape)
    if not all(isinstance(value, torch.Tensor) for value in (*reals, sigma)):
        raise TypeError("merger-ringdown inputs must be torch tensors")
    reference = time_since_match
    if reference.dtype not in (torch.float32, torch.float64) or reference.ndim < 1:
        raise TypeError("time must be a float32/float64 tensor with a sample axis")
    if any(value.dtype != reference.dtype or value.device != reference.device for value in reals):
        raise TypeError("real inputs must share time's floating dtype and device")
    complex_dtype = torch.complex128 if reference.dtype == torch.float64 else torch.complex64
    if sigma.dtype != complex_dtype or sigma.device != reference.device:
        raise TypeError("sigma must have time's corresponding complex dtype and device")
    if any(not bool(torch.isfinite(value).all()) for value in (*reals, sigma)):
        raise ValueError("merger-ringdown inputs must be finite")
    if bool((reference < 0).any()) or bool(((sigma.real <= 0) | (sigma.imag >= 0)).any()):
        raise ValueError("time must be nonnegative and sigma must have real > 0, imaginary < 0")
    if bool((amplitude <= 0).any()) or bool((frequency <= 0).any()):
        raise ValueError("matching amplitude and frequency must be positive")
    c1, c2, d1, d2 = shape
    if bool((c1 <= 0).any()) or bool((d1 <= 0).any()) or bool((d2 <= -1).any()):
        raise ValueError("shape requires c1f,d1f > 0 and d2f > -1")

    parameters = torch.broadcast_tensors(sigma.real, -sigma.imag, amplitude,
                                         amplitude_derivative, frequency, phase, *shape)
    batch = torch.broadcast_shapes(reference.shape[:-1], parameters[0].shape)
    t = reference.expand(batch + (reference.shape[-1],))
    w, damping, a0, adot0, w0, phi0, c1, c2, d1, d2 = (
        value.expand(batch).unsqueeze(-1) for value in parameters)

    # Enforcing A(0)=a0 and A'(0)=adot0 gives C1=(adot0+gamma*a0)
    # cosh(c2)^2/c1. Use the tanh addition identity to avoid cancellation.
    transition = torch.tanh(c1*t)
    rescaled_amplitude = a0 + (adot0 + damping*a0)/c1 * transition/(1 + transition*torch.tanh(c2))

    # d1c=(w-w0)(1+d2)/(d1*d2). Express d1c*log(...) without a 0/0
    # at d2=0. Then total phase derivative at zero is exactly w0.
    exponential_change = torch.expm1(-d1*t)
    log_argument_change = d2/(1 + d2)*exponential_change
    residual_phase = (w - w0)/d1 * exponential_change*_log1p_over_x(log_argument_change)
    total_phase = phi0 + w*t + residual_phase
    return rescaled_amplitude*torch.exp(torch.complex(-damping*t, -total_phase))

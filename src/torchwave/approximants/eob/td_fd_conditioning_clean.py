"""Torch-only time-domain to frequency-domain conditioning for EOB modes.

This module deliberately contains only the sampling/conditioning layer.  It
does not generate an EOB trajectory and it has no dependency on LALSuite or
SciPy.  The support estimate follows the 2PN TaylorT2 timing expression and
the filter is an explicit cascade of bilinear-transformed Butterworth
biquads.  The implementation is written with ordinary Torch operations so
that waveform values, filter parameters, and support quantities can remain on
the caller's device and autograd graph.

The public entry points are:

``plan_td_support``
    Compute the conservative generation support and its PN/QNM metadata.
``design_butterworth_highpass`` / ``apply_butterworth_highpass``
    Design and apply the prescribed eighth-order high-pass filter.
``condition_td``
    Apply the two tapers, zero-phase high-pass, and paired trailing-zero trim.
``fft_length_and_sampling`` / ``pair_leading_pad_final_crop``
    Implement the explicit and automatic Fourier sizing rules.
``condition_td_to_fd`` / ``td_to_fd``
    Compose the conditioning and scaled real rFFT, with an optional separate
    user ``tc`` phase rotation.

The formulas and numerical choices are recorded in
``docs/reviews/2026-09-13-parity-review/td-fd-conditioning-contract.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import torch

from torchwave.constants import C_SI, G_SI, MSUN_SEC, MSUN_SI, PI


# The following are the nonspinning terms obtained by integrating the 2PN
# TaylorT2 dt/dv expression.  Writing the coefficients this way keeps the
# integration factors visible: the v^2 term is 4/3 times the dt/dv term and
# the v^4 term is twice the dt/dv term.  This gives the timing coefficients
# used by the public validation samples in the conditioning contract.
# The eta-independent part is retained as a named constant for callers that
# want to inspect the timing series.  The full 1PN coefficient is
# ``743/252 + 11 eta/3`` below.
T2_TIME_1PN = 743.0 / 252.0
T2_TIME_2PN_ETA0 = 2.0 * (3058673.0 / 1016064.0)
T2_TIME_2PN_ETA1 = 2.0 * (5429.0 / 1008.0)
T2_TIME_2PN_ETA2 = 2.0 * (617.0 / 144.0)
SPIN_TIME_COEFFICIENT = 226.0 / 15.0

_BUTTERWORTH_ORDER = 8
_BUTTERWORTH_SECTIONS = 4
_FILTER_POWER_TRANSMISSION = 0.99
_QNM_F1 = 1.5251
_QNM_F2 = -1.1568
_QNM_F3 = 0.1292
_QNM_Q1 = 0.7000
_QNM_Q2 = 1.4187
_QNM_Q3 = -0.4990
_TICHY_EQUAL_MASS_SPIN0 = 0.686
_TICHY_EQUAL_MASS_SPIN_LINEAR = 0.15


def _infer_dtype_device(
    *values: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str] = None,
) -> Tuple[torch.dtype, torch.device]:
    """Infer one real dtype/device without silently moving tensor inputs."""

    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if any(value.is_complex() for value in tensors):
        raise TypeError("conditioning parameters must be real")
    if any(value.dtype == torch.bool for value in tensors):
        raise TypeError("conditioning parameters must be real numbers")

    if device is None:
        devices = [value.device for value in tensors]
        if devices and any(item != devices[0] for item in devices):
            raise ValueError("conditioning tensors must share a device")
        target_device = devices[0] if devices else torch.device("cpu")
    else:
        target_device = torch.device(device)

    if dtype is None:
        floating = [value.dtype for value in tensors if value.dtype.is_floating_point]
        if floating:
            target_dtype = floating[0]
            for item in floating[1:]:
                target_dtype = torch.promote_types(target_dtype, item)
        else:
            target_dtype = torch.float64
    else:
        target_dtype = dtype
    if not target_dtype.is_floating_point:
        raise TypeError("conditioning dtype must be floating point")
    return target_dtype, target_device


def _real_tensor(
    value: Any,
    *others: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str] = None,
    name: str = "value",
) -> torch.Tensor:
    target_dtype, target_device = _infer_dtype_device(
        value, *others, dtype=dtype, device=device
    )
    tensors = [item for item in (value, *others) if isinstance(item, torch.Tensor)]
    if any(item.is_complex() for item in tensors):
        raise TypeError(f"{name} must be real")
    if any(isinstance(item, bool) for item in (value, *others)):
        raise TypeError(f"{name} must be real numbers")
    return torch.as_tensor(value, dtype=target_dtype, device=target_device)


def _finite_positive(value: torch.Tensor, name: str) -> None:
    if torch.any(~torch.isfinite(value)) or torch.any(value <= 0):
        raise ValueError(f"{name} must be finite and positive")


def _finite(value: torch.Tensor, name: str) -> None:
    if torch.any(~torch.isfinite(value)):
        raise ValueError(f"{name} must be finite")


def _broadcast_pair_parameters(
    mass1: Any,
    mass2: Any,
    f_low: Any,
    chi1z: Any,
    chi2z: Any,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_dtype, target_device = _infer_dtype_device(
        mass1, mass2, f_low, chi1z, chi2z, dtype=dtype, device=device
    )
    values = tuple(
        _real_tensor(value, dtype=target_dtype, device=target_device)
        for value in (mass1, mass2, f_low, chi1z, chi2z)
    )
    return torch.broadcast_tensors(*values)


def _coerce_like(value: Any, like: torch.Tensor, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor) and value.is_complex():
        raise TypeError(f"{name} must be real")
    if isinstance(value, bool) or (
        isinstance(value, torch.Tensor) and value.dtype == torch.bool
    ):
        raise TypeError(f"{name} must be real")
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


@dataclass(frozen=True)
class TDSupportPlan:
    """Differentiable metadata for the conservative TD support estimate."""

    mass1: torch.Tensor
    mass2: torch.Tensor
    eta: torch.Tensor
    total_mass_seconds: torch.Tensor
    f_low: torch.Tensor
    f_safe: torch.Tensor
    v_safe: torch.Tensor
    spin_bound: torch.Tensor
    chirp_time: torch.Tensor
    plunge_time: torch.Tensor
    ringdown_time: torch.Tensor
    plunge_ringdown_time: torch.Tensor
    support_time: torch.Tensor
    start_frequency: torch.Tensor
    first_taper_duration: torch.Tensor
    qnm_dimensionless_frequency: torch.Tensor
    qnm_quality: torch.Tensor
    qnm_damping_time: torch.Tensor

    @property
    def M(self) -> torch.Tensor:
        """Detector-frame total mass in seconds."""

        return self.total_mass_seconds

    @property
    def t_bound(self) -> torch.Tensor:
        return self.chirp_time

    @property
    def duration(self) -> torch.Tensor:
        return self.support_time

    @property
    def f_start(self) -> torch.Tensor:
        return self.start_frequency

    @property
    def f_isco(self) -> torch.Tensor:
        return 1.0 / (6.0 ** 1.5 * PI * self.total_mass_seconds)

    @property
    def first_taper_time(self) -> torch.Tensor:
        return self.first_taper_duration

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return the support metadata without converting tensors to scalars."""

        return {
            "mass1": self.mass1,
            "mass2": self.mass2,
            "eta": self.eta,
            "total_mass_seconds": self.total_mass_seconds,
            "f_low": self.f_low,
            "f_safe": self.f_safe,
            "v_safe": self.v_safe,
            "spin_bound": self.spin_bound,
            "chirp_time": self.chirp_time,
            "plunge_time": self.plunge_time,
            "ringdown_time": self.ringdown_time,
            "plunge_ringdown_time": self.plunge_ringdown_time,
            "support_time": self.support_time,
            "start_frequency": self.start_frequency,
            "first_taper_duration": self.first_taper_duration,
            "qnm_dimensionless_frequency": self.qnm_dimensionless_frequency,
            "qnm_quality": self.qnm_quality,
            "qnm_damping_time": self.qnm_damping_time,
        }


def plan_td_support(
    mass1: Any,
    mass2: Any,
    f_low: Any,
    chi1z: Any = 0.0,
    chi2z: Any = 0.0,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str] = None,
) -> TDSupportPlan:
    """Plan sufficient time support from masses, requested low frequency and spins.

    ``mass1`` and ``mass2`` are detector-frame solar masses.  The timing
    bound retains the positive Newtonian, 1PN and 2PN TaylorT2 terms and the
    positive aligned-spin overestimate specified by the conditioning
    contract.  The negative 1.5PN tail term is intentionally omitted.
    """

    m1, m2, flow, c1, c2 = _broadcast_pair_parameters(
        mass1, mass2, f_low, chi1z, chi2z, dtype=dtype, device=device
    )
    _finite_positive(m1, "mass1")
    _finite_positive(m2, "mass2")
    _finite_positive(flow, "f_low")
    _finite(c1, "chi1z")
    _finite(c2, "chi2z")

    # The public timing APIs receive SI masses.  Reproduce that unit
    # conversion explicitly in float64 for the precision path.  Computing
    # the reduced mass as ``m1SI*m2SI/totalSI`` is the published definition
    # and also avoids silently changing the arithmetic through a mass-ratio
    # shortcut.  A float32 SI product would overflow (m1SI*m2SI is O(1e60)),
    # so lower precision callers use the algebraically equivalent scaled
    # ratio while retaining their requested output dtype.
    if m1.dtype == torch.float64:
        msun_si = m1.new_tensor(MSUN_SI)
        m1_si = m1 * msun_si
        m2_si = m2 * msun_si
        total_si = m1_si + m2_si
        reduced_mass_si = m1_si * m2_si / total_si
        eta = reduced_mass_si / total_si
        # Keep the SI conversion used by the dimensionful Newtonian
        # frequency prefactor separate from the solar-mass unit ratio used
        # in TaylorT3's dimensionless time variable.  These constants are
        # physically equal but need not round to the same binary64 value.
        total_mass_seconds = total_si * m1.new_tensor(G_SI / (C_SI**3))
        # Keep the multiplication by five on the SI mass before applying the
        # independent solar-mass-to-seconds conversion.  This is the
        # dimensionless inverse-time numerator in the TaylorT3 expression.
        inverse_time_numerator = (5.0 * total_si) * m1.new_tensor(
            MSUN_SEC / MSUN_SI
        )
    else:
        mass_scale = torch.maximum(m1, m2)
        x1 = m1 / mass_scale
        x2 = m2 / mass_scale
        eta = (x1 * x2) / (x1 + x2).square()
        total_mass_seconds = (m1 + m2) * m1.new_tensor(MSUN_SEC)
        inverse_time_numerator = (5.0 * (m1 + m2)) * m1.new_tensor(MSUN_SEC)

    f_safe_cap = 1.0 / (27.0 * m1.new_tensor(PI) * total_mass_seconds)
    f_safe = torch.minimum(flow, f_safe_cap)
    v_safe = (m1.new_tensor(PI) * total_mass_seconds * f_safe).pow(1.0 / 3.0)

    a_1pn = m1.new_tensor(743.0 / 252.0) + (11.0 / 3.0) * eta
    a_2pn = (
        m1.new_tensor(T2_TIME_2PN_ETA0)
        + m1.new_tensor(T2_TIME_2PN_ETA1) * eta
        + m1.new_tensor(T2_TIME_2PN_ETA2) * eta.square()
    )
    spin_abs = torch.maximum(torch.abs(c1), torch.abs(c2))
    bracket = (
        1.0
        + a_1pn * v_safe.square()
        + a_2pn * v_safe.pow(4.0)
        + m1.new_tensor(SPIN_TIME_COEFFICIENT) * spin_abs * v_safe.pow(3.0)
    )
    chirp_time = (
        5.0
        * total_mass_seconds
        / (256.0 * eta * v_safe.pow(8.0))
        * bracket
    )

    spin_estimate = (
        m1.new_tensor(_TICHY_EQUAL_MASS_SPIN0)
        + m1.new_tensor(_TICHY_EQUAL_MASS_SPIN_LINEAR) * (c1 + c2)
    )
    spin_bound = torch.minimum(torch.maximum(spin_abs, spin_estimate), m1.new_tensor(0.998))
    qnm_dimensionless_frequency = m1.new_tensor(_QNM_F1) + m1.new_tensor(
        _QNM_F2
    ) * (1.0 - spin_bound).pow(_QNM_F3)
    qnm_quality = m1.new_tensor(_QNM_Q1) + m1.new_tensor(_QNM_Q2) * (
        1.0 - spin_bound
    ).pow(_QNM_Q3)
    qnm_damping_time = (
        2.0 * qnm_quality * total_mass_seconds / qnm_dimensionless_frequency
    )
    plunge_time = 54.0 * m1.new_tensor(PI) * total_mass_seconds
    ringdown_time = 11.0 * qnm_damping_time
    plunge_ringdown_time = plunge_time + ringdown_time

    support_time = 1.1 * chirp_time + plunge_ringdown_time + 3.0 / f_safe
    # Invert the Newtonian chirp-time relation directly.  Factoring this as
    # an eighth root followed by a cube changes the last bit on some inputs.
    inverse_dimensionless_time = inverse_time_numerator / (eta * support_time)
    newtonian_frequency_prefactor = 1.0 / (
        8.0 * m1.new_tensor(PI) * total_mass_seconds
    )
    start_frequency = newtonian_frequency_prefactor * inverse_dimensionless_time.pow(
        3.0 / 8.0
    )
    first_taper_duration = 0.1 * chirp_time + 3.0 / f_safe

    return TDSupportPlan(
        mass1=m1,
        mass2=m2,
        eta=eta,
        total_mass_seconds=total_mass_seconds,
        f_low=flow,
        f_safe=f_safe,
        v_safe=v_safe,
        spin_bound=spin_bound,
        chirp_time=chirp_time,
        plunge_time=plunge_time,
        ringdown_time=ringdown_time,
        plunge_ringdown_time=plunge_ringdown_time,
        support_time=support_time,
        start_frequency=start_frequency,
        first_taper_duration=first_taper_duration,
        qnm_dimensionless_frequency=qnm_dimensionless_frequency,
        qnm_quality=qnm_quality,
        qnm_damping_time=qnm_damping_time,
    )


# Friendly aliases used by callers that describe the same operation as a
# generic support calculation.
compute_td_support = plan_td_support
support_plan = plan_td_support
plan_support = plan_td_support


def qnm_22_properties(
    total_mass_seconds: Any,
    spin: Any,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(M omega_R, Q, damping_time_seconds)`` for the Berti 22 fit.

    The fit is the fundamental 22 expression from Berti, Cardoso and Will,
    Appendix E, Table VIII.  ``total_mass_seconds`` is already in geometric
    seconds; no remnant-mass prescription is applied.
    """

    mass = _real_tensor(total_mass_seconds, spin, dtype=dtype, device=device, name="mass")
    a = _real_tensor(spin, total_mass_seconds, dtype=mass.dtype, device=mass.device, name="spin")
    mass, a = torch.broadcast_tensors(mass, a)
    _finite_positive(mass, "total_mass_seconds")
    _finite(a, "spin")
    if torch.any(a >= 1.0):
        raise ValueError("QNM spin must be below extremality")
    w = mass.new_tensor(_QNM_F1) + mass.new_tensor(_QNM_F2) * (1.0 - a).pow(_QNM_F3)
    q = mass.new_tensor(_QNM_Q1) + mass.new_tensor(_QNM_Q2) * (1.0 - a).pow(_QNM_Q3)
    tau = 2.0 * q * mass / w
    return w, q, tau


def chirp_time_bound(
    mass1: Any,
    mass2: Any,
    f_low: Any,
    chi1z: Any = 0.0,
    chi2z: Any = 0.0,
    **kwargs: Any,
) -> torch.Tensor:
    """Return only the positive-term 2PN TaylorT2 chirp bound."""

    return plan_td_support(mass1, mass2, f_low, chi1z, chi2z, **kwargs).chirp_time


def round_half_up(value: Any, *, dtype: torch.dtype = torch.int64) -> torch.Tensor:
    """Round nonnegative values to nearest integer, resolving halves upward."""

    x = _real_tensor(value, name="value")
    _finite(x, "value")
    if torch.any(x < 0):
        raise ValueError("round_half_up expects nonnegative values")
    return torch.floor(x + 0.5).to(dtype=dtype)


def count_trailing_zero_padding(
    h_plus: torch.Tensor, h_cross: torch.Tensor
) -> torch.Tensor:
    """Count samples that are exactly zero in both real polarizations."""

    if not isinstance(h_plus, torch.Tensor) or not isinstance(h_cross, torch.Tensor):
        raise TypeError("polarizations must be Torch tensors")
    if h_plus.shape != h_cross.shape or h_plus.ndim < 1:
        raise ValueError("polarizations must have equal nonempty shapes")
    if h_plus.device != h_cross.device:
        raise ValueError("polarizations must share a device")
    if h_plus.shape[-1] == 0:
        return torch.zeros(h_plus.shape[:-1], dtype=torch.int64, device=h_plus.device)
    both_zero = (h_plus == 0) & (h_cross == 0)
    # The cumulative product of the reversed zero mask remains one exactly
    # while the suffix is zero, so summing it gives the trailing count.
    return torch.cumprod(both_zero.flip(-1).to(torch.int64), dim=-1).sum(-1)


@dataclass(frozen=True)
class ButterworthSections:
    """Normalized SOS coefficients for the prescribed high-pass cascade.

    ``numerator`` and ``denominator`` have shape ``(..., 4, 3)``.  The last
    axis contains ``b0,b1,b2`` and ``a0,a1,a2`` respectively, with ``a0=1``.
    ``sos`` presents the same data as ``(..., 4, 6)`` in SciPy's SOS layout.
    """

    numerator: torch.Tensor
    denominator: torch.Tensor
    corner: torch.Tensor
    theta: torch.Tensor

    @property
    def sos(self) -> torch.Tensor:
        return torch.cat((self.numerator, self.denominator), dim=-1)

    def __iter__(self) -> Iterable[torch.Tensor]:
        yield self.numerator
        yield self.denominator


def design_butterworth_highpass(
    dt: Any,
    f_low: Any,
    *,
    power_transmission: float = _FILTER_POWER_TRANSMISSION,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str] = None,
) -> ButterworthSections:
    """Design the fourth-biquad eighth-order bilinear high-pass cascade.

    The normalized analog corner is
    ``tan(pi*f_low*dt) * (1/sqrt(power_transmission)-1)**(1/16)``.
    Each real-conjugate Butterworth section has denominator
    ``s^2 + 2 sin(theta) corner s + corner^2`` and numerator ``s^2``
    before bilinear transformation, with
    ``theta=(j+1/2) pi/8`` for ``j=0,...,3``.
    """

    target_dtype, target_device = _infer_dtype_device(dt, f_low, dtype=dtype, device=device)
    dtt = _real_tensor(dt, dtype=target_dtype, device=target_device, name="dt")
    flow = _real_tensor(f_low, dtype=target_dtype, device=target_device, name="f_low")
    dtt, flow = torch.broadcast_tensors(dtt, flow)
    _finite_positive(dtt, "dt")
    _finite_positive(flow, "f_low")
    if torch.any(flow * dtt >= 0.5):
        raise ValueError("f_low must be below the sampling Nyquist frequency")
    power = float(power_transmission)
    if not math.isfinite(power) or not 0.0 < power < 1.0:
        raise ValueError("power_transmission must lie strictly between zero and one")

    factor = dtt.new_tensor(1.0 / math.sqrt(power) - 1.0).pow(1.0 / 16.0)
    corner = torch.tan(dtt.new_tensor(PI) * flow * dtt) * factor
    theta = dtt.new_tensor(
        [(j + 0.5) * math.pi / _BUTTERWORTH_ORDER for j in range(_BUTTERWORTH_SECTIONS)]
    )
    damping = torch.sin(theta).reshape((1,) * corner.ndim + (_BUTTERWORTH_SECTIONS,))
    c = corner.unsqueeze(-1)
    d0 = 1.0 + 2.0 * damping * c + c.square()
    d1 = -2.0 + 2.0 * c.square()
    d2 = 1.0 - 2.0 * damping * c + c.square()
    numerator = torch.stack((torch.ones_like(d0), -2.0 * torch.ones_like(d0), torch.ones_like(d0)), dim=-1) / d0.unsqueeze(-1)
    denominator = torch.stack((torch.ones_like(d0), d1 / d0, d2 / d0), dim=-1)
    return ButterworthSections(
        numerator=numerator,
        denominator=denominator,
        corner=corner,
        theta=theta,
    )


def butterworth_highpass_coefficients(*args: Any, **kwargs: Any) -> ButterworthSections:
    """Alias for :func:`design_butterworth_highpass`."""

    return design_butterworth_highpass(*args, **kwargs)


def _broadcast_sections(
    sections: ButterworthSections,
    batch_shape: torch.Size,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target = tuple(batch_shape) + (_BUTTERWORTH_SECTIONS, 3)
    numerator = torch.broadcast_to(sections.numerator, target)
    denominator = torch.broadcast_to(sections.denominator, target)
    return numerator, denominator


def _sos_direction(
    x: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    reverse: bool = False,
) -> torch.Tensor:
    """Run one zero-state directional pass through one second-order section."""

    original_shape = x.shape
    n = original_shape[-1]
    flat_x = x.reshape(-1, n)
    flat_b = numerator.reshape(-1, 3)
    flat_a = denominator.reshape(-1, 3)
    if reverse:
        flat_x = flat_x.flip(-1)

    output = torch.empty_like(flat_x)
    x1 = torch.zeros_like(flat_x[:, 0])
    x2 = torch.zeros_like(flat_x[:, 0])
    y1 = torch.zeros_like(flat_x[:, 0])
    y2 = torch.zeros_like(flat_x[:, 0])
    b0, b1, b2 = flat_b.unbind(dim=-1)
    _, a1, a2 = flat_a.unbind(dim=-1)
    for index in range(n):
        current = flat_x[:, index]
        value = b0 * current + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        output[:, index] = value
        x2, x1 = x1, current
        y2, y1 = y1, value
    if reverse:
        output = output.flip(-1)
    return output.reshape(original_shape)


def apply_butterworth_highpass(
    signal: torch.Tensor,
    dt: Any,
    f_low: Any,
    *,
    sections: Optional[ButterworthSections] = None,
    power_transmission: float = _FILTER_POWER_TRANSMISSION,
) -> torch.Tensor:
    """Apply the four-section forward/reverse zero-state high-pass cascade.

    Each section is run forward and then backward before the next section is
    visited.  This is intentionally a direct Torch recurrence rather than a
    call to a host-library filter.  Its cost is linear in the number of time
    samples and in the four sections; all leading batch dimensions run in
    parallel.
    """

    if not isinstance(signal, torch.Tensor) or signal.ndim < 1:
        raise TypeError("signal must be a nonempty Torch tensor")
    if signal.shape[-1] == 0:
        raise ValueError("signal must contain at least one sample")
    if not (signal.dtype.is_floating_point or signal.dtype.is_complex):
        raise TypeError("signal must have a floating or complex dtype")
    if sections is None:
        sections = design_butterworth_highpass(
            dt, f_low, dtype=signal.real.dtype, device=signal.device,
            power_transmission=power_transmission,
        )
    elif not isinstance(sections, ButterworthSections):
        raise TypeError("sections must be ButterworthSections")
    # A designed bank can be reused across waveforms with a different
    # floating dtype or device.  Converting coefficients here preserves their
    # graph while ensuring the recurrence never mixes host and accelerator
    # tensors.
    sections = ButterworthSections(
        numerator=sections.numerator.to(
            device=signal.device, dtype=signal.real.dtype
        ),
        denominator=sections.denominator.to(
            device=signal.device, dtype=signal.real.dtype
        ),
        corner=sections.corner.to(device=signal.device, dtype=signal.real.dtype),
        theta=sections.theta.to(device=signal.device, dtype=signal.real.dtype),
    )
    numerator, denominator = _broadcast_sections(sections, signal.shape[:-1])
    output = signal
    for section in range(_BUTTERWORTH_SECTIONS):
        b = numerator[..., section, :]
        a = denominator[..., section, :]
        output = _sos_direction(output, b, a)
        output = _sos_direction(output, b, a, reverse=True)
    return output


zero_phase_butterworth_highpass = apply_butterworth_highpass
butterworth_highpass = apply_butterworth_highpass


def _taper_counts(
    duration: torch.Tensor,
    dt: torch.Tensor,
    n: int,
    *,
    minimum: int = 1,
    label: str,
) -> torch.Tensor:
    duration, dt = torch.broadcast_tensors(duration, dt)
    _finite_positive(duration, label)
    _finite_positive(dt, "dt")
    counts = round_half_up(duration / dt)
    if torch.any(counts < minimum):
        raise ValueError(f"{label} rounds to fewer than {minimum} samples")
    if torch.any(counts > n):
        raise ValueError(f"{label} exceeds the available time series")
    return counts


def _half_cosine_rise(
    signal: torch.Tensor,
    count: torch.Tensor,
) -> torch.Tensor:
    n = signal.shape[-1]
    count = torch.broadcast_to(count, signal.shape[:-1]).to(device=signal.device)
    count_real = count.to(dtype=signal.real.dtype if signal.is_complex() else signal.dtype)
    offset = torch.arange(n, dtype=count_real.dtype, device=signal.device)
    offset = offset.reshape((1,) * (signal.ndim - 1) + (n,))
    denominator = count_real.unsqueeze(-1)
    angle = signal.real.new_tensor(PI) * offset / denominator
    taper = 0.5 * (1.0 - torch.cos(angle))
    taper = torch.where(offset < denominator, taper, torch.ones_like(taper))
    return signal * taper


def _second_stage_taper(
    signal: torch.Tensor,
    f_safe: torch.Tensor,
    f_isco: torch.Tensor,
    dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    n = signal.shape[-1]
    if n < 8:
        empty = torch.zeros(signal.shape[:-1], dtype=torch.int64, device=signal.device)
        return signal, empty, empty, False
    begin_duration = 1.0 / f_safe
    end_duration = 1.0 / f_isco
    # Round the physical cycle lengths first, then apply the contract's
    # four-sample floor.  A valid cycle can be shorter than four samples at a
    # deliberately coarse rate; that case is represented by the floor rather
    # than rejected as an invalid duration.
    begin_duration, dt_begin = torch.broadcast_tensors(begin_duration, dt)
    end_duration, dt_end = torch.broadcast_tensors(end_duration, dt)
    _finite_positive(begin_duration, "second-stage beginning taper")
    _finite_positive(end_duration, "second-stage ending taper")
    _finite_positive(dt_begin, "dt")
    _finite_positive(dt_end, "dt")
    begin_count = torch.clamp(round_half_up(begin_duration / dt_begin), min=4)
    end_count = torch.clamp(round_half_up(end_duration / dt_end), min=4)
    # The public contract requires a minimum four samples per second-stage
    # taper.  The duration rounding remains nearest-half-up before this
    # safety floor is applied.
    if torch.any(begin_count <= 0) or torch.any(end_count <= 0):  # pragma: no cover
        raise ValueError("second-stage taper counts must be positive")
    if torch.any(begin_count > n) or torch.any(end_count > n):
        raise ValueError("second-stage taper requires at least four in-range samples")

    signal = _half_cosine_rise(signal, begin_count)
    batch_shape = signal.shape[:-1]
    count_real = end_count.to(dtype=signal.real.dtype if signal.is_complex() else signal.dtype)
    backward = torch.arange(n, 0, -1, dtype=count_real.dtype, device=signal.device)
    backward = backward.reshape((1,) * len(batch_shape) + (n,))
    denom = count_real.unsqueeze(-1)
    end_angle = signal.real.new_tensor(PI) * backward / denom
    end_weight = 0.5 * (1.0 - torch.cos(end_angle))
    end_weight = torch.where(backward <= denom, end_weight, torch.ones_like(end_weight))
    return signal * end_weight, begin_count, end_count, True


@dataclass(frozen=True)
class ConditionedTD:
    """Conditioned real polarizations and their discrete taper metadata."""

    h_plus: torch.Tensor
    h_cross: torch.Tensor
    n_trailing_zeros: torch.Tensor
    first_taper_samples: torch.Tensor
    second_begin_samples: torch.Tensor
    second_end_samples: torch.Tensor
    second_stage_applied: bool
    support: TDSupportPlan

    def __iter__(self) -> Iterable[torch.Tensor]:
        yield self.h_plus
        yield self.h_cross


def _resolve_support(
    support: Optional[TDSupportPlan],
    h_plus: torch.Tensor,
    mass1: Any,
    mass2: Any,
    f_low: Any,
    chi1z: Any,
    chi2z: Any,
) -> TDSupportPlan:
    if support is not None:
        if not isinstance(support, TDSupportPlan):
            raise TypeError("support must be a TDSupportPlan")
        return support
    if mass1 is None or mass2 is None or f_low is None:
        raise ValueError("mass1, mass2 and f_low are required when support is omitted")
    return plan_td_support(
        mass1,
        mass2,
        f_low,
        chi1z,
        chi2z,
        dtype=h_plus.real.dtype if h_plus.is_complex() else h_plus.dtype,
        device=h_plus.device,
    )


def _support_for_waveform(
    support: TDSupportPlan,
    *,
    batch_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
) -> TDSupportPlan:
    """Move and broadcast a support plan to a waveform's tensor layout.

    A plan is often computed once in float64 on CPU and then consumed by a
    float32 or accelerator waveform.  ``Tensor.to`` keeps the autograd edge
    while making every field used by conditioning compatible with the signal.
    Broadcasting all fields also means returned metadata has the same batch
    layout as the conditioned waveform.
    """

    values = []
    for field_name in TDSupportPlan.__dataclass_fields__:
        field_value = torch.as_tensor(
            getattr(support, field_name), dtype=dtype, device=device
        )
        try:
            field_value = torch.broadcast_to(field_value, batch_shape)
        except RuntimeError as exc:
            raise ValueError(
                "support parameters are not broadcastable to the waveform batch"
            ) from exc
        values.append(field_value)
    return TDSupportPlan(*values)


def condition_td(
    h_plus: torch.Tensor,
    h_cross: torch.Tensor,
    *,
    dt: Any,
    support: Optional[TDSupportPlan] = None,
    mass1: Any = None,
    mass2: Any = None,
    f_low: Any = None,
    chi1z: Any = 0.0,
    chi2z: Any = 0.0,
    return_metadata: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | ConditionedTD:
    """Apply first taper, Torch high-pass, trailing trim, and second taper.

    The input's last axis is time.  All leading axes are treated as a batch;
    a supplied support plan may carry matching broadcastable parameter axes.
    A batch must share one trailing-zero count because a dense tensor cannot
    represent different trimmed lengths without adding a new padding policy.
    """

    if not isinstance(h_plus, torch.Tensor) or not isinstance(h_cross, torch.Tensor):
        raise TypeError("polarizations must be Torch tensors")
    if h_plus.shape != h_cross.shape or h_plus.ndim < 1:
        raise ValueError("polarizations must have equal nonempty shapes")
    if h_plus.device != h_cross.device:
        raise ValueError("polarizations must share a device")
    if h_plus.shape[-1] == 0:
        raise ValueError("polarizations must contain at least one sample")
    if not (h_plus.dtype.is_floating_point or h_plus.dtype.is_complex):
        raise TypeError("polarizations must have floating or complex dtype")
    if not (h_cross.dtype.is_floating_point or h_cross.dtype.is_complex):
        raise TypeError("polarizations must have floating or complex dtype")
    if torch.any(~torch.isfinite(h_plus)) or torch.any(~torch.isfinite(h_cross)):
        raise ValueError("polarizations must be finite")

    support = _resolve_support(
        support, h_plus, mass1, mass2, f_low, chi1z, chi2z
    )
    dt_t = _coerce_like(dt, h_plus.real if h_plus.is_complex() else h_plus, name="dt")
    _finite_positive(dt_t, "dt")
    if h_plus.is_complex() or h_cross.is_complex():
        raise TypeError("TD conditioning expects real polarizations")

    trailing = count_trailing_zero_padding(h_plus, h_cross)
    if trailing.numel() and not bool(torch.all(trailing == trailing.reshape(-1)[0])):
        raise ValueError("batched polarizations must share a trailing-zero count")
    trim = int(trailing.reshape(-1)[0].item()) if trailing.numel() else 0

    batch_shape = h_plus.shape[:-1]
    # A support plan may have originated on another device or in another
    # floating dtype.  Normalize it before arithmetic; ``to`` remains
    # differentiable for tensor-valued support inputs.
    support = _support_for_waveform(
        support,
        batch_shape=batch_shape,
        dtype=h_plus.dtype,
        device=h_plus.device,
    )
    flow = support.f_low
    fsafe = support.f_safe
    msec = support.total_mass_seconds
    first_duration = support.first_taper_duration

    first_count = _taper_counts(
        first_duration, dt_t, h_plus.shape[-1], minimum=1, label="first taper"
    )
    hp = _half_cosine_rise(h_plus, first_count)
    hc = _half_cosine_rise(h_cross, first_count)
    sections = design_butterworth_highpass(
        dt_t,
        flow,
        dtype=h_plus.dtype,
        device=h_plus.device,
    )
    hp = apply_butterworth_highpass(hp, dt_t, flow, sections=sections)
    hc = apply_butterworth_highpass(hc, dt_t, flow, sections=sections)

    if trim >= hp.shape[-1]:
        raise ValueError("TD conditioning removed every sample as trailing zero padding")
    if trim:
        hp = hp[..., :-trim]
        hc = hc[..., :-trim]

    # ISCO uses the Schwarzschild GW frequency specified by the contract.
    f_isco = 1.0 / (6.0 ** 1.5 * msec.new_tensor(PI) * msec)
    hp, begin_count, end_count, applied = _second_stage_taper(
        hp, fsafe, f_isco, dt_t
    )
    hc, _, _, _ = _second_stage_taper(hc, fsafe, f_isco, dt_t)
    if return_metadata:
        return ConditionedTD(
            h_plus=hp,
            h_cross=hc,
            n_trailing_zeros=trailing,
            first_taper_samples=first_count,
            second_begin_samples=begin_count,
            second_end_samples=end_count,
            second_stage_applied=applied,
            support=support,
        )
    return hp, hc


condition_td_polarizations = condition_td
two_stage_condition = condition_td


@dataclass(frozen=True)
class FourierSampling:
    """Discrete Fourier sizing and sample-rate metadata."""

    n_time: int
    n_frequency: int
    n_nyquist_intervals: int
    f_max_requested: Optional[torch.Tensor]
    f_nyquist: torch.Tensor
    delta_f: torch.Tensor
    dt: torch.Tensor
    automatic: bool

    @property
    def N(self) -> int:
        return self.n_time

    @property
    def f_max(self) -> torch.Tensor:
        return self.f_nyquist

    def frequencies(self, *, device: Optional[torch.device | str] = None) -> torch.Tensor:
        target_device = self.dt.device if device is None else torch.device(device)
        index = torch.arange(self.n_frequency, dtype=self.dt.dtype, device=target_device)
        return index * self.delta_f.to(device=target_device)


def _next_power_of_two_at_least(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _next_power_of_two_strict(value: int) -> int:
    result = 1
    while result <= value:
        result <<= 1
    return result


def fft_length_and_sampling(
    f_max: Any = None,
    delta_f: Any = None,
    *,
    dt: Any = None,
    n_nonzero: Optional[int] = None,
    n_time: Optional[int] = None,
    n_signal: Optional[int] = None,
) -> FourierSampling:
    """Choose the Fourier length and sampling prescribed by the contract.

    With an explicit ``delta_f``, ``f_max/delta_f`` is nearest-half-up
    rounded, the result is raised to the next power of two, and the final TD
    length is twice that interval count.  Without ``delta_f``, ``dt`` and the
    nonzero TD length are required; a power-of-two length strictly greater
    than the nonzero length is selected.
    """

    if delta_f is not None:
        if f_max is None:
            raise ValueError("f_max is required when delta_f is explicit")
        fmax_t = _real_tensor(f_max, delta_f, name="f_max")
        df_t = _real_tensor(delta_f, f_max, dtype=fmax_t.dtype, device=fmax_t.device, name="delta_f")
        if fmax_t.numel() != 1 or df_t.numel() != 1:
            raise ValueError("Fourier sizing currently requires scalar f_max and delta_f")
        _finite_positive(fmax_t, "f_max")
        _finite_positive(df_t, "delta_f")
        raw = int(math.floor(float((fmax_t / df_t).detach().cpu().item()) + 0.5))
        raw = max(raw, 1)
        n_intervals = _next_power_of_two_at_least(raw)
        n_samples = 2 * n_intervals
        f_nyquist = n_intervals * df_t
        chosen_dt = 1.0 / (2.0 * f_nyquist)
        if dt is not None:
            supplied_dt = _coerce_like(dt, chosen_dt, name="dt")
            _finite_positive(supplied_dt, "dt")
            if not torch.allclose(supplied_dt, chosen_dt, rtol=1e-10, atol=0.0):
                raise ValueError("explicit dt disagrees with the requested Fourier spacing")
            chosen_dt = supplied_dt
        return FourierSampling(
            n_time=n_samples,
            n_frequency=n_intervals + 1,
            n_nyquist_intervals=n_intervals,
            f_max_requested=fmax_t,
            f_nyquist=f_nyquist,
            delta_f=df_t,
            dt=chosen_dt,
            automatic=False,
        )

    if n_nonzero is None:
        n_nonzero = n_signal
    if n_nonzero is None:
        n_nonzero = n_time
    if n_nonzero is None or isinstance(n_nonzero, bool) or int(n_nonzero) <= 0:
        raise ValueError("positive n_nonzero is required for automatic Fourier sizing")
    n_nonzero = int(n_nonzero)
    chosen_dt = None if dt is None else _real_tensor(dt, name="dt")
    if chosen_dt is None or chosen_dt.numel() != 1:
        raise ValueError("scalar positive dt is required for automatic Fourier sizing")
    _finite_positive(chosen_dt, "dt")
    n_samples = _next_power_of_two_strict(n_nonzero)
    n_intervals = n_samples // 2
    f_nyquist = 1.0 / (2.0 * chosen_dt)
    df_t = 1.0 / (n_samples * chosen_dt)
    return FourierSampling(
        n_time=n_samples,
        n_frequency=n_intervals + 1,
        n_nyquist_intervals=n_intervals,
        f_max_requested=None,
        f_nyquist=f_nyquist,
        delta_f=df_t,
        dt=chosen_dt,
        automatic=True,
    )


compute_fft_sampling = fft_length_and_sampling
plan_fft = fft_length_and_sampling


def pair_leading_pad_final_crop(
    h_plus: torch.Tensor,
    h_cross: torch.Tensor,
    n_final: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad both polarizations at the front or crop both from the front.

    The retained data always end at the same final sample, which is the
    pairing required by the real rFFT interface.
    """

    if not isinstance(h_plus, torch.Tensor) or not isinstance(h_cross, torch.Tensor):
        raise TypeError("polarizations must be Torch tensors")
    if h_plus.shape != h_cross.shape or h_plus.ndim < 1:
        raise ValueError("polarizations must have equal shapes")
    if h_plus.device != h_cross.device:
        raise ValueError("polarizations must share a device")
    if isinstance(n_final, bool) or int(n_final) <= 0:
        raise ValueError("n_final must be a positive integer")
    n_final = int(n_final)
    current = h_plus.shape[-1]
    if n_final == current:
        return h_plus, h_cross
    if n_final < current:
        return h_plus[..., -n_final:], h_cross[..., -n_final:]
    zeros_plus = torch.zeros(
        h_plus.shape[:-1] + (n_final - current,), dtype=h_plus.dtype, device=h_plus.device
    )
    zeros_cross = torch.zeros(
        h_cross.shape[:-1] + (n_final - current,), dtype=h_cross.dtype, device=h_cross.device
    )
    return torch.cat((zeros_plus, h_plus), dim=-1), torch.cat((zeros_cross, h_cross), dim=-1)


paired_leading_pad_final_crop = pair_leading_pad_final_crop
pad_crop_pair = pair_leading_pad_final_crop


def apply_tc_phase(
    h_plus_fd: torch.Tensor,
    h_cross_fd: torch.Tensor,
    frequencies: torch.Tensor,
    tc: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the explicit user coalescence-time phase rotation separately."""

    if h_plus_fd.shape != h_cross_fd.shape or h_plus_fd.ndim < 1:
        raise ValueError("frequency-domain polarizations must have equal shapes")
    freq = _coerce_like(frequencies, h_plus_fd.real, name="frequencies")
    shift = _coerce_like(tc, h_plus_fd.real, name="tc")
    _finite(shift, "tc")
    _finite(freq, "frequencies")
    if freq.shape[-1] != h_plus_fd.shape[-1]:
        raise ValueError("frequencies and FD polarizations must have equal lengths")
    while shift.ndim < h_plus_fd.ndim:
        shift = shift.unsqueeze(-1)
    phase = torch.exp(-2j * h_plus_fd.real.new_tensor(PI) * freq * shift)
    return h_plus_fd * phase, h_cross_fd * phase


@dataclass(frozen=True)
class ConditionedFD:
    """Scaled rFFT output, frequencies, and conditioning metadata."""

    frequencies: torch.Tensor
    h_plus: torch.Tensor
    h_cross: torch.Tensor
    sampling: FourierSampling
    conditioning: ConditionedTD
    tc: torch.Tensor

    def __iter__(self) -> Iterable[torch.Tensor]:
        yield self.h_plus
        yield self.h_cross


def _frequency_grid(sampling: FourierSampling, device: torch.device) -> torch.Tensor:
    index = torch.arange(sampling.n_frequency, dtype=sampling.dt.dtype, device=device)
    return index * sampling.delta_f.to(device=device)


def condition_td_to_fd(
    h_plus: torch.Tensor,
    h_cross: torch.Tensor,
    *,
    dt: Any = None,
    delta_f: Any = None,
    f_max: Any = None,
    support: Optional[TDSupportPlan] = None,
    mass1: Any = None,
    mass2: Any = None,
    f_low: Any = None,
    chi1z: Any = 0.0,
    chi2z: Any = 0.0,
    tc: Any = 0.0,
    return_metadata: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | ConditionedFD:
    """Condition TD polarizations and return their scaled one-sided rFFT.

    ``tc`` is applied after the FFT as ``exp(-2 pi i f tc)``.  Leaving it at
    zero keeps the physical TD epoch as metadata only; no epoch phase is
    inserted into the transform.
    """

    if not isinstance(h_plus, torch.Tensor) or not isinstance(h_cross, torch.Tensor):
        raise TypeError("polarizations must be Torch tensors")
    if dt is None and delta_f is None:
        raise ValueError("dt is required unless explicit delta_f and f_max determine it")

    if delta_f is not None:
        # Derive the mandated dt before conditioning, so timing/taper sample
        # counts use the same rate as the final Fourier grid.
        preliminary = fft_length_and_sampling(f_max, delta_f, dt=dt)
        dt_work = preliminary.dt
    else:
        dt_work = _coerce_like(dt, h_plus, name="dt")
        _finite_positive(dt_work, "dt")
        preliminary = None

    conditioned = condition_td(
        h_plus,
        h_cross,
        dt=dt_work,
        support=support,
        mass1=mass1,
        mass2=mass2,
        f_low=f_low,
        chi1z=chi1z,
        chi2z=chi2z,
        return_metadata=True,
    )
    if not isinstance(conditioned, ConditionedTD):  # pragma: no cover - type guard
        raise RuntimeError("conditioning metadata was not returned")

    if preliminary is None:
        sampling = fft_length_and_sampling(
            None,
            None,
            dt=dt_work,
            n_nonzero=conditioned.h_plus.shape[-1],
        )
    else:
        sampling = preliminary
    hp, hc = pair_leading_pad_final_crop(
        conditioned.h_plus, conditioned.h_cross, sampling.n_time
    )
    # ``sampling.dt`` is differentiable when delta_f is a differentiable
    # tensor.  The generated TD waveform is expected to use this exact rate.
    hp_fd = sampling.dt * torch.fft.rfft(hp, n=sampling.n_time, dim=-1)
    hc_fd = sampling.dt * torch.fft.rfft(hc, n=sampling.n_time, dim=-1)
    frequencies = _frequency_grid(sampling, hp.device)
    tc_tensor = _coerce_like(tc, hp, name="tc")
    # Keep this operation in the graph even when tc is numerically zero: the
    # derivative of the requested phase with respect to tc is meaningful at
    # zero and callers may optimize tc jointly with waveform parameters.
    hp_fd, hc_fd = apply_tc_phase(hp_fd, hc_fd, frequencies, tc_tensor)
    if return_metadata:
        return ConditionedFD(
            frequencies=frequencies,
            h_plus=hp_fd,
            h_cross=hc_fd,
            sampling=sampling,
            conditioning=conditioned,
            tc=tc_tensor,
        )
    return hp_fd, hc_fd


def td_to_fd(*args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(frequencies, h_plus_fd, h_cross_fd)`` for convenience."""

    kwargs = dict(kwargs)
    kwargs["return_metadata"] = True
    result = condition_td_to_fd(*args, **kwargs)
    if not isinstance(result, ConditionedFD):  # pragma: no cover - type guard
        raise RuntimeError("Fourier metadata was not returned")
    return result.frequencies, result.h_plus, result.h_cross


__all__ = [
    "TDSupportPlan",
    "plan_td_support",
    "compute_td_support",
    "support_plan",
    "plan_support",
    "qnm_22_properties",
    "chirp_time_bound",
    "round_half_up",
    "count_trailing_zero_padding",
    "ButterworthSections",
    "design_butterworth_highpass",
    "butterworth_highpass_coefficients",
    "apply_butterworth_highpass",
    "zero_phase_butterworth_highpass",
    "butterworth_highpass",
    "ConditionedTD",
    "condition_td",
    "condition_td_polarizations",
    "two_stage_condition",
    "FourierSampling",
    "fft_length_and_sampling",
    "compute_fft_sampling",
    "plan_fft",
    "pair_leading_pad_final_crop",
    "paired_leading_pad_final_crop",
    "pad_crop_pair",
    "apply_tc_phase",
    "ConditionedFD",
    "condition_td_to_fd",
    "td_to_fd",
]

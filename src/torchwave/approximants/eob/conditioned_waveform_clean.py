"""Bounded clean room SEOBNRv4 time domain to frequency domain adapter.

This module owns the small orchestration layer between the independent
adaptive SEOBNRv4 reconstruction and the Torch only conditioning contract.
It deliberately does not replace :mod:`published_waveform`: the latter's
finite integral remains available for its existing diagnostics and defaults.

The public adapter accepts physical component masses (solar masses), aligned
spins, and scalar ``f_low``, ``f_max`` and ``delta_f`` values.  The explicit
Fourier plan is made first.  Its sample spacing is passed to the adaptive
generator in geometric units, while the support planner supplies the
generation start frequency.  The resulting 22 mode is paired with its
nonprecessing ``(2,-2)`` conjugate, projected onto real polarizations, and
then sent through :func:`condition_td_to_fd`.

``phic`` has a concrete mode meaning: it multiplies the 22 and (2,-2) modes
by ``exp(-2 i phic)`` and ``exp(+2 i phic)`` respectively.  The generated
clock starts with orbital phase zero; there is no reference-frequency phase
reanchoring.  ``fref`` is therefore rejected explicitly rather than being
silently ignored.  ``tc`` is the separate physical coalescence-time shift
and is applied by the conditioning layer as ``exp(-2 pi i f tc)``.  The
physical location of the first generated sample relative to the continuous
frequency peak is returned as metadata only and never supplies another phase
factor.

The equations and numerical choices used here are MIT clean-room work based
on the local conditioning contract and public literature: Berti, Cardoso and
Will, arXiv:gr-qc/0512160 (fundamental 22 damping fit); Tichy and Marronetti,
arXiv:0807.2985 (support spin estimate); and the spin-weighted quadrupole
projection convention.  The implementation imports no LALSuite code or
source-only calibration table.
"""

# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC, PI

from .adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from .td_fd_conditioning_clean import (
    ConditionedFD,
    FourierSampling,
    TDSupportPlan,
    condition_td_to_fd,
    fft_length_and_sampling,
    plan_td_support,
)


# These are intentionally ordinary values so a caller or a test can inspect
# the exact reviewed route sent to the adaptive reconstruction.
SEOBNRV4_ADAPTIVE_CHOICES = {
    "spin_truncation": "generic-leading-ell8",
    "spin_v6": "bohe_total_energy",
    "include_pan_v8": False,
    "include_positive_chi_540": False,
    "final_spin_fit_order": (3, 4),
}
DEFAULT_RINGDOWN_DURATION = 150.0


def _infer_dtype_device(values: Sequence[Any]) -> Tuple[torch.dtype, torch.device]:
    """Infer one floating dtype/device without silently moving tensors."""

    tensors = [value for value in values if isinstance(value, Tensor)]
    if any(value.is_complex() for value in tensors):
        raise TypeError("SEOBNRv4 adapter parameters must be real")
    if any(value.dtype == torch.bool for value in tensors):
        raise TypeError("SEOBNRv4 adapter parameters must be real numbers")
    devices = [value.device for value in tensors]
    if devices and any(device != devices[0] for device in devices):
        raise ValueError("SEOBNRv4 adapter tensors must share a device")
    floating = [value.dtype for value in tensors if value.dtype.is_floating_point]
    dtype = floating[0] if floating else torch.float64
    for item in floating[1:]:
        dtype = torch.promote_types(dtype, item)
    return dtype, (devices[0] if devices else torch.device("cpu"))


def _real_tensor(value: Any, *, dtype: torch.dtype, device: torch.device,
                 name: str) -> Tensor:
    if isinstance(value, bool) or (
        isinstance(value, Tensor) and value.dtype == torch.bool
    ):
        raise TypeError(f"{name} must be a real number")
    if isinstance(value, Tensor) and value.is_complex():
        raise TypeError(f"{name} must be real")
    try:
        result = torch.as_tensor(value, dtype=dtype, device=device)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not result.dtype.is_floating_point:
        result = result.to(dtype=dtype)
    return result


def _finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _positive_scalar(value: Any, *, dtype: torch.dtype, device: torch.device,
                     name: str) -> Tensor:
    result = _real_tensor(value, dtype=dtype, device=device, name=name)
    if result.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if not bool(torch.isfinite(result)) or not bool(result > 0):
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_scalar(value: Any, *, dtype: torch.dtype, device: torch.device,
                   name: str) -> Tensor:
    result = _real_tensor(value, dtype=dtype, device=device, name=name)
    if result.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    _finite(result, name)
    return result


def _parameter_broadcast(
    values: Sequence[Tuple[str, Any]], *, dtype: torch.dtype,
    device: torch.device,
) -> Mapping[str, Tensor]:
    """Convert and broadcast intrinsic/extrinsic scalar or batch values."""

    names = [name for name, _ in values]
    tensors = [
        _real_tensor(value, dtype=dtype, device=device, name=name)
        for name, value in values
    ]
    try:
        broadcasted = torch.broadcast_tensors(*tensors)
    except RuntimeError as exc:
        raise ValueError(
            "masses, spins, distance, inclination, phic and tc must be "
            "broadcastable"
        ) from exc
    return dict(zip(names, broadcasted))


def _attribute_or_key(value: Any, names: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _extract_generated_modes(result: Any, *, dtype: torch.dtype,
                             device: torch.device) -> Tuple[Tensor, Tensor, Tensor, Any]:
    """Read the intentionally small result contract of an injected generator.

    The native adaptive route returns ``PublishedEOB22`` with ``time`` and
    ``mode`` fields.  Test or downstream generators may additionally expose
    ``mode_minus``/``h2m2``.  A two mode nonprecessing generator may instead
    return ``(time, mode22, mode2m2)``.  Omitting the negative mode is
    well-defined here and uses ``conj(mode22)``.
    """

    time = _attribute_or_key(result, ("time", "td_time", "times"))
    mode22 = _attribute_or_key(result, ("mode", "mode_22", "h22", "h_22"))
    mode2m2 = _attribute_or_key(
        result,
        ("mode_minus", "mode_2m2", "h2m2", "h_2m2", "mode_m22", "h22_minus"),
    )

    if isinstance(result, (tuple, list)) and not hasattr(result, "_fields"):
        # Plain tuple injection is useful in small tests.  A NamedTuple result
        # (including PublishedEOB22) already exposes its semantic fields;
        # treating its third positional field as ``mode2m2`` would mistake
        # the trajectory metadata for a negative mode.
        if time is None and len(result) >= 2:
            time = result[0]
        if mode22 is None and len(result) >= 2:
            mode22 = result[1]
        if mode2m2 is None and len(result) >= 3:
            mode2m2 = result[2]

    if time is None or mode22 is None:
        raise TypeError(
            "the TD generator must return time and mode (or h22) fields"
        )
    time_t = _real_tensor(time, dtype=dtype, device=device, name="TD time")
    if time_t.ndim != 1:
        if time_t.numel() == 0 or math.prod(time_t.shape[:-1]) != 1:
            raise ValueError("the injected scalar TD generator must return a one-dimensional time")
        time_t = time_t.reshape(-1)
    if time_t.numel() < 2:
        raise ValueError("the generated TD waveform must contain at least two samples")
    if not bool(torch.isfinite(time_t).all()):
        raise ValueError("generated TD time must be finite")

    def mode_tensor(value: Any, name: str) -> Tensor:
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, device=device)
        if value.is_complex():
            target = torch.complex128 if dtype == torch.float64 else torch.complex64
            mode = value.to(device=device, dtype=target)
        else:
            real = _real_tensor(value, dtype=dtype, device=device, name=name)
            mode = torch.complex(real, torch.zeros_like(real))
        if mode.ndim != 1:
            if mode.numel() == 0 or math.prod(mode.shape[:-1]) != 1:
                raise ValueError(
                    "the injected scalar TD generator must return a one-dimensional mode"
                )
            mode = mode.reshape(-1)
        if mode.shape[-1] != time_t.numel():
            raise ValueError("generated mode and time axes must have equal lengths")
        if not bool(torch.isfinite(mode).all()):
            raise ValueError(f"generated {name} must be finite")
        return mode

    mode22_t = mode_tensor(mode22, "22 mode")
    mode2m2_t = torch.conj(mode22_t) if mode2m2 is None else mode_tensor(mode2m2, "(2,-2) mode")
    return time_t, mode22_t, mode2m2_t, result


def _validate_generated_clock(time: Tensor, dt_geom: Tensor) -> None:
    spacing = time[1:] - time[:-1]
    if not bool(torch.isfinite(spacing).all()) or not bool((spacing > 0).all()):
        raise ValueError("generated TD time must be uniformly increasing")
    if not bool(torch.allclose(
        spacing,
        dt_geom,
        # Multiplication by a geometric step accumulates a few ulps before
        # differencing a float32 arange.  Keep the check tight enough to catch
        # a genuinely different requested rate while accepting that native
        # representation of an otherwise uniform fixed grid.
        rtol=2.0e-5 if time.dtype == torch.float32 else 1.0e-10,
        atol=2.0e-5 if time.dtype == torch.float32 else 1.0e-12,
    )):
        raise ValueError("generated TD spacing does not match the Fourier plan")


def project_22_pair_to_polarizations(
    h22: Tensor,
    h2m2: Optional[Tensor] = None,
    *,
    mass1: Any,
    mass2: Any,
    distance: Any = 100.0,
    inclination: Any = 0.0,
    phic: Any = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Project a paired nonprecessing 22 mode onto real ``(h_plus,h_cross)``.

    ``h22`` and ``h2m2`` are dimensionless geometric modes with a trailing
    time axis.  The physical strain scale is ``M_seconds*c/(D*Mpc)``.  The
    mode phase convention is explicit: ``m=2`` receives ``exp(-2 i phic)``
    and ``m=-2`` receives ``exp(+2 i phic)``.  When ``h2m2`` is omitted, the
    standard aligned-spin relation ``h2m2=conj(h22)`` is used.
    """

    if not isinstance(h22, Tensor) or h22.ndim < 1 or h22.shape[-1] == 0:
        raise TypeError("h22 must be a nonempty tensor with a trailing time axis")
    # ``torch.complex`` accepts floating inputs only.  Integer modes are not
    # physically meaningful, but converting them here keeps the projection's
    # dtype/device contract explicit instead of leaking a backend error.
    if not h22.dtype.is_floating_point and not h22.is_complex():
        h22 = h22.to(dtype=torch.float64)
    if not h22.is_complex():
        h22 = torch.complex(h22, torch.zeros_like(h22))
    if h2m2 is None:
        h2m2 = torch.conj(h22)
    if not isinstance(h2m2, Tensor) or h2m2.shape != h22.shape:
        raise ValueError("h22 and h2m2 must have equal shapes")
    if h2m2.device != h22.device:
        raise ValueError("h22 and h2m2 must share a device")
    if not bool(torch.isfinite(h22).all()) or not bool(torch.isfinite(h2m2).all()):
        raise ValueError("h22 and h2m2 must be finite")

    real_dtype = h22.real.dtype
    device = h22.device
    values = _parameter_broadcast(
        (("mass1", mass1), ("mass2", mass2), ("distance", distance),
         ("inclination", inclination), ("phic", phic)),
        dtype=real_dtype,
        device=device,
    )
    m1, m2 = values["mass1"], values["mass2"]
    dist, inc, phase = values["distance"], values["inclination"], values["phic"]
    if not bool(torch.isfinite(m1).all()) or not bool((m1 > 0).all()):
        raise ValueError("mass1 must be finite and positive")
    if not bool(torch.isfinite(m2).all()) or not bool((m2 > 0).all()):
        raise ValueError("mass2 must be finite and positive")
    if not bool(torch.isfinite(dist).all()) or not bool((dist > 0).all()):
        raise ValueError("distance must be finite and positive")
    if not bool(torch.isfinite(inc).all()):
        raise ValueError("inclination must be finite")
    if not bool(torch.isfinite(phase).all()):
        raise ValueError("phic must be finite")

    batch_shape = torch.broadcast_shapes(
        h22.shape[:-1], m1.shape, m2.shape, dist.shape, inc.shape, phase.shape
    )
    n_time = h22.shape[-1]
    h22 = h22.expand(batch_shape + (n_time,))
    h2m2 = h2m2.to(device=device, dtype=h22.dtype).expand(batch_shape + (n_time,))
    m1, m2, dist, inc, phase = (
        value.expand(batch_shape)
        for value in (m1, m2, dist, inc, phase)
    )

    # The two spin-weighted harmonics at azimuth zero are
    # sqrt(5/(64*pi))*(1 +/- cos(iota))**2.  Taking Re(H) and -Im(H)
    # reproduces the established single-mode projection when h2m2=conj(h22).
    cosine = torch.cos(inc).unsqueeze(-1)
    plus_weight = (1.0 + cosine).square()
    minus_weight = (1.0 - cosine).square()
    phase22 = torch.exp((-2j) * phase.unsqueeze(-1))
    phasem2 = torch.exp((2j) * phase.unsqueeze(-1))
    pair = plus_weight * h22 * phase22 + minus_weight * h2m2 * phasem2
    normalization = h22.real.new_tensor(math.sqrt(5.0 / (64.0 * math.pi)))
    pair = normalization * pair
    m_seconds = (m1 + m2) * h22.real.new_tensor(MSUN_SEC)
    scale = (
        m_seconds * h22.real.new_tensor(C_SI)
        / (dist * h22.real.new_tensor(MPC_SI))
    ).unsqueeze(-1)
    pair = pair * scale
    return pair.real, -pair.imag


# Short aliases make the projection helper discoverable without introducing
# another implementation path.
project_22_modes = project_22_pair_to_polarizations
project_quadrupole_pair = project_22_pair_to_polarizations


@dataclass(frozen=True)
class ConditionedSEOBNRv4FD:
    """Frequency-domain waveform plus the discrete choices and TD metadata."""

    frequencies: Tensor
    h_plus: Tensor
    h_cross: Tensor
    sampling: FourierSampling
    conditioning: Any
    support: Any
    td_time: Any
    mode_22: Any
    mode_2m2: Any
    h_plus_td: Any
    h_cross_td: Any
    omega_start: Tensor
    dt_geom: Tensor
    dt_seconds: Tensor
    physical_epoch: Tensor
    phic: Tensor
    tc: Tensor
    distance: Tensor
    inclination: Tensor
    generator_result: Any
    choices: Mapping[str, Any]
    fd_result: Any

    def __iter__(self):
        yield self.h_plus
        yield self.h_cross

    @property
    def hp(self) -> Tensor:
        return self.h_plus

    @property
    def hc(self) -> Tensor:
        return self.h_cross

    @property
    def f(self) -> Tensor:
        return self.frequencies

    @property
    def epoch(self) -> Tensor:
        """Physical time of the first TD sample relative to the peak chart."""

        return self.physical_epoch

    @property
    def fd(self) -> Any:
        return self.fd_result


ConditionedWaveform = ConditionedSEOBNRv4FD
ConditionedEOBFD = ConditionedSEOBNRv4FD


def _peak_time_from_result(result: Any, *, dtype: torch.dtype,
                           device: torch.device) -> Tensor:
    trajectory = _attribute_or_key(result, ("trajectory",))
    peak = _attribute_or_key(trajectory, ("peak",)) if trajectory is not None else None
    value = _attribute_or_key(peak, ("time",)) if peak is not None else None
    if value is None:
        attachment = _attribute_or_key(result, ("attachment",))
        value = _attribute_or_key(attachment, ("match_time",)) if attachment is not None else None
    if value is None:
        return torch.zeros((), dtype=dtype, device=device)
    peak_t = _finite_scalar(value, dtype=dtype, device=device, name="peak time")
    return peak_t


def _resolve_positional(
    args: Tuple[Any, ...], *, chi1z: Any, chi2z: Any, f_low: Any,
    f_max: Any, delta_f: Any,
) -> Tuple[Any, Any, Any, Any, Any]:
    """Accept the two natural positional forms while keeping keywords clear."""

    if not args:
        return chi1z, chi2z, f_low, f_max, delta_f
    if len(args) == 2:
        if chi1z is not None or chi2z is not None:
            raise TypeError("chi1z/chi2z were supplied both positionally and by keyword")
        return args[0], args[1], f_low, f_max, delta_f
    if len(args) == 3:
        if any(value is not None for value in (f_low, f_max, delta_f)):
            raise TypeError("frequency bounds were supplied both positionally and by keyword")
        return chi1z, chi2z, args[0], args[1], args[2]
    if len(args) == 5:
        if any(value is not None for value in (chi1z, chi2z, f_low, f_max, delta_f)):
            raise TypeError("adapter positional and keyword arguments overlap")
        return args[0], args[1], args[2], args[3], args[4]
    raise TypeError(
        "expected either (f_low, f_max, delta_f) or "
        "(chi1z, chi2z, f_low, f_max, delta_f) after mass1 and mass2"
    )


def _num_steps_for_support(support: TDSupportPlan, dt_seconds: Tensor) -> int:
    # num_steps is a controller budget and therefore necessarily discrete.
    # The support duration is the reviewed allocation bound; four extra
    # accepted steps leave room for the initial event stencil.
    estimate = torch.ceil(support.support_time / dt_seconds).detach()
    return max(4, int(estimate.cpu().item()) + 4)


def _run_scalar(
    *,
    mass1: Tensor,
    mass2: Tensor,
    chi1z: Tensor,
    chi2z: Tensor,
    distance: Tensor,
    inclination: Tensor,
    phic: Tensor,
    tc: Tensor,
    f_low: Tensor,
    f_max: Tensor,
    delta_f: Tensor,
    sampling: FourierSampling,
    td_generator: Callable[..., Any],
    num_steps: Optional[int],
    ringdown_duration: float,
) -> dict[str, Any]:
    support = plan_td_support(mass1, mass2, f_low, chi1z, chi2z,
                              dtype=mass1.dtype, device=mass1.device)
    dt_seconds = sampling.dt.to(device=mass1.device, dtype=mass1.dtype)
    dt_geom = dt_seconds / support.total_mass_seconds
    omega_start = mass1.new_tensor(PI) * support.total_mass_seconds * support.start_frequency
    n_steps = _num_steps_for_support(support, dt_seconds) if num_steps is None else num_steps
    result = td_generator(
        omega_start,
        mass1,
        mass2,
        chi1z,
        chi2z,
        **SEOBNRV4_ADAPTIVE_CHOICES,
        dt_geom=dt_geom,
        num_steps=n_steps,
        ringdown_duration=ringdown_duration,
    )
    time, mode22, mode2m2, result = _extract_generated_modes(
        result, dtype=mass1.dtype, device=mass1.device
    )
    _validate_generated_clock(time, dt_geom)
    h_plus_td, h_cross_td = project_22_pair_to_polarizations(
        mode22,
        mode2m2,
        mass1=mass1,
        mass2=mass2,
        distance=distance,
        inclination=inclination,
        phic=phic,
    )
    fd_result = condition_td_to_fd(
        h_plus_td,
        h_cross_td,
        dt=dt_seconds,
        f_max=f_max,
        delta_f=delta_f,
        support=support,
        tc=tc,
        return_metadata=True,
    )
    if not isinstance(fd_result, ConditionedFD):  # pragma: no cover - guard
        raise RuntimeError("condition_td_to_fd did not return metadata")
    peak_time = _peak_time_from_result(result, dtype=mass1.dtype, device=mass1.device)
    # The first generated sample's physical location is useful to consumers
    # reconstructing an absolute chart, but the FD array intentionally starts
    # at its own sample zero and receives only the explicit tc phase.
    physical_epoch = (time[0] - peak_time) * support.total_mass_seconds
    return {
        "support": support,
        "sampling": sampling,
        "dt_seconds": dt_seconds,
        "dt_geom": dt_geom,
        "omega_start": omega_start,
        "result": result,
        "time": time,
        "mode22": mode22,
        "mode2m2": mode2m2,
        "h_plus_td": h_plus_td,
        "h_cross_td": h_cross_td,
        "physical_epoch": physical_epoch,
        "fd": fd_result,
        "num_steps": n_steps,
    }


def conditioned_seobnrv4_fd(
    mass1: Any,
    mass2: Any,
    *args: Any,
    chi1z: Any = None,
    chi2z: Any = None,
    chi1: Any = None,
    chi2: Any = None,
    f_low: Any = None,
    f_max: Any = None,
    delta_f: Any = None,
    distance: Any = 100.0,
    inclination: Any = 0.0,
    phic: Any = 0.0,
    tc: Any = 0.0,
    fref: Any = None,
    f_ref: Any = None,
    return_metadata: bool = False,
    num_steps: Optional[int] = None,
    ringdown_duration: Any = DEFAULT_RINGDOWN_DURATION,
    generator: Optional[Callable[..., Any]] = None,
    td_generator: Optional[Callable[..., Any]] = None,
) -> Tuple[Tensor, Tensor] | ConditionedSEOBNRv4FD:
    """Generate the bounded clean-room SEOBNRv4 TD→FD waveform.

    The preferred call uses keyword frequency bounds::

        conditioned_seobnrv4_fd(
            30., 20., f_low=20., f_max=512., delta_f=.25,
            chi1z=0., chi2z=0., distance=100., inclination=.4,
        )

    For convenience, after the two masses either ``(f_low, f_max,
    delta_f)`` or ``(chi1z, chi2z, f_low, f_max, delta_f)`` can be supplied
    positionally.  Intrinsic rows are integrated independently; a batch is
    accepted when all physical parameters broadcast, and every row uses the
    same explicit Fourier grid.  The generator receives the exact reviewed
    choices in :data:`SEOBNRV4_ADAPTIVE_CHOICES` and a full ``150 M`` tail by
    default.

    The default return is ``(h_plus_fd, h_cross_fd)``.  With
    ``return_metadata=True``, a :class:`ConditionedSEOBNRv4FD` object exposes
    the common frequency grid, conditioned TD data, support plan, start
    frequency, geometric and physical sample spacings, physical epoch, and
    generator result.
    """

    chi1z, chi2z, f_low, f_max, delta_f = _resolve_positional(
        tuple(args), chi1z=chi1z, chi2z=chi2z, f_low=f_low, f_max=f_max,
        delta_f=delta_f,
    )
    if chi1 is not None:
        if chi1z is not None:
            raise TypeError("chi1 and chi1z were both supplied")
        chi1z = chi1
    if chi2 is not None:
        if chi2z is not None:
            raise TypeError("chi2 and chi2z were both supplied")
        chi2z = chi2
    chi1z = 0.0 if chi1z is None else chi1z
    chi2z = 0.0 if chi2z is None else chi2z
    if f_ref is not None:
        if fref is not None:
            raise TypeError("fref and f_ref were both supplied")
        fref = f_ref
    if fref is not None:
        raise ValueError(
            "fref is unsupported by the clean TD-to-FD adapter: "
            "phic is applied at the generated zero-phase chart"
        )
    if f_low is None or f_max is None or delta_f is None:
        raise ValueError("f_low, f_max and delta_f are required scalar frequencies")
    if generator is not None and td_generator is not None:
        raise TypeError("generator and td_generator were both supplied")
    td_generator = td_generator or generator or reconstruct_seobnrv4_22_adaptive
    if not callable(td_generator):
        raise TypeError("generator must be callable")
    if not isinstance(return_metadata, bool):
        raise TypeError("return_metadata must be a bool")
    if num_steps is not None and (
        isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 4
    ):
        raise ValueError("num_steps must be an integer of at least four")
    if isinstance(ringdown_duration, bool):
        raise TypeError("ringdown_duration must be finite and positive")
    try:
        ringdown_duration_float = float(ringdown_duration)
    except (TypeError, ValueError) as exc:
        raise TypeError("ringdown_duration must be finite and positive") from exc
    if not math.isfinite(ringdown_duration_float) or ringdown_duration_float <= 0:
        raise ValueError("ringdown_duration must be finite and positive")

    dtype, device = _infer_dtype_device(
        (mass1, mass2, chi1z, chi2z, f_low, f_max, delta_f, distance,
         inclination, phic, tc)
    )
    flow = _positive_scalar(f_low, dtype=dtype, device=device, name="f_low")
    fmax = _positive_scalar(f_max, dtype=dtype, device=device, name="f_max")
    df = _positive_scalar(delta_f, dtype=dtype, device=device, name="delta_f")
    if not bool(fmax > flow):
        raise ValueError("f_max must be greater than f_low")
    params = _parameter_broadcast(
        (("mass1", mass1), ("mass2", mass2), ("chi1z", chi1z),
         ("chi2z", chi2z), ("distance", distance),
         ("inclination", inclination), ("phic", phic), ("tc", tc)),
        dtype=dtype,
        device=device,
    )
    for name in ("mass1", "mass2", "chi1z", "chi2z", "distance",
                 "inclination", "phic", "tc"):
        _finite(params[name], name)
    if not bool((params["mass1"] > 0).all()) or not bool((params["mass2"] > 0).all()):
        raise ValueError("masses must be finite and positive")
    if not bool((params["distance"] > 0).all()):
        raise ValueError("distance must be finite and positive")

    # This call is intentionally before any expensive TD work and determines
    # both the common fixed grid and the exact physical sample spacing.
    sampling = fft_length_and_sampling(fmax, df)
    batch_shape = params["mass1"].shape
    flat = {name: value.reshape(-1) for name, value in params.items()}
    rows = []
    for index in range(math.prod(batch_shape) if batch_shape else 1):
        rows.append(_run_scalar(
            mass1=flat["mass1"][index], mass2=flat["mass2"][index],
            chi1z=flat["chi1z"][index], chi2z=flat["chi2z"][index],
            distance=flat["distance"][index],
            inclination=flat["inclination"][index],
            phic=flat["phic"][index], tc=flat["tc"][index],
            f_low=flow, f_max=fmax, delta_f=df, sampling=sampling,
            td_generator=td_generator, num_steps=num_steps,
            ringdown_duration=ringdown_duration_float,
        ))

    if not batch_shape:
        row = rows[0]
        fd = row["fd"]
        metadata = ConditionedSEOBNRv4FD(
            frequencies=fd.frequencies,
            h_plus=fd.h_plus,
            h_cross=fd.h_cross,
            sampling=fd.sampling,
            conditioning=fd.conditioning,
            support=row["support"],
            td_time=row["time"],
            mode_22=row["mode22"],
            mode_2m2=row["mode2m2"],
            h_plus_td=row["h_plus_td"],
            h_cross_td=row["h_cross_td"],
            omega_start=row["omega_start"],
            dt_geom=row["dt_geom"],
            dt_seconds=row["dt_seconds"],
            physical_epoch=row["physical_epoch"],
            phic=params["phic"],
            tc=fd.tc,
            distance=params["distance"],
            inclination=params["inclination"],
            generator_result=row["result"],
            choices=dict(SEOBNRV4_ADAPTIVE_CHOICES,
                         ringdown_duration=ringdown_duration_float,
                         num_steps=row["num_steps"]),
            fd_result=fd,
        )
    else:
        fd = rows[0]["fd"]
        metadata = ConditionedSEOBNRv4FD(
            frequencies=fd.frequencies,
            h_plus=torch.stack([row["fd"].h_plus for row in rows]).reshape(batch_shape + (-1,)),
            h_cross=torch.stack([row["fd"].h_cross for row in rows]).reshape(batch_shape + (-1,)),
            sampling=fd.sampling,
            conditioning=tuple(row["fd"].conditioning for row in rows),
            support=tuple(row["support"] for row in rows),
            td_time=tuple(row["time"] for row in rows),
            mode_22=tuple(row["mode22"] for row in rows),
            mode_2m2=tuple(row["mode2m2"] for row in rows),
            h_plus_td=tuple(row["h_plus_td"] for row in rows),
            h_cross_td=tuple(row["h_cross_td"] for row in rows),
            omega_start=torch.stack([row["omega_start"] for row in rows]).reshape(batch_shape),
            dt_geom=torch.stack([row["dt_geom"] for row in rows]).reshape(batch_shape),
            dt_seconds=rows[0]["dt_seconds"],
            physical_epoch=torch.stack([row["physical_epoch"] for row in rows]).reshape(batch_shape),
            phic=params["phic"],
            tc=params["tc"],
            distance=params["distance"],
            inclination=params["inclination"],
            generator_result=tuple(row["result"] for row in rows),
            choices=dict(SEOBNRV4_ADAPTIVE_CHOICES,
                         ringdown_duration=ringdown_duration_float,
                         num_steps=tuple(row["num_steps"] for row in rows)),
            fd_result=tuple(row["fd"] for row in rows),
        )
    return metadata if return_metadata else (metadata.h_plus, metadata.h_cross)


# Names used by callers that describe the same explicit route.
seobnrv4_conditioned_fd = conditioned_seobnrv4_fd
conditioned_waveform = conditioned_seobnrv4_fd
generate_conditioned_seobnrv4_fd = conditioned_seobnrv4_fd


__all__ = [
    "SEOBNRV4_ADAPTIVE_CHOICES",
    "DEFAULT_RINGDOWN_DURATION",
    "ConditionedSEOBNRv4FD",
    "ConditionedWaveform",
    "ConditionedEOBFD",
    "project_22_pair_to_polarizations",
    "project_22_modes",
    "project_quadrupole_pair",
    "conditioned_seobnrv4_fd",
    "seobnrv4_conditioned_fd",
    "conditioned_waveform",
    "generate_conditioned_seobnrv4_fd",
]

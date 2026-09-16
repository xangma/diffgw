"""Clean-room TEOB merger/ringdown and sampling assembly.

This module consumes already corrected waveform samples.  It owns the
audited sample join, the orbital-frequency peak polynomial, the merger and
attachment-time bookkeeping, and the published nonspinning TEOBResumS
postmerger extension.  NQC targets, NQC coefficients, and trajectory
evolution remain caller responsibilities.

The phase convention is ``h=A*exp(-i*phase)``.  Amplitude and phase are kept
as separate real arrays throughout the join, so a signed corrected amplitude
is never replaced by its absolute value.  The explicit array arguments are
the hand-off surface for an auxiliary-NQC implementation::

    assemble_teob_ringdown_22(
        full_time, full_amplitude, full_phase,
        auxiliary_time, auxiliary_amplitude, auxiliary_phase,
        auxiliary_orbital_frequency, nu, final_mass, final_spin,
    )

No production route is changed by this opt-in component.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import Tensor

from .teob_postmerger import postmerger_22


_REAL_DTYPES = (torch.float32, torch.float64)
_EXTENSION_SAMPLES = 1500


class CorrectedWaveformSeries(NamedTuple):
    """Named array bundle exchanged with the preceding NQC component."""

    time: Tensor
    amplitude: Tensor
    phase: Tensor


class RingdownAssemblyEvents(NamedTuple):
    """Peak, merger, and final-mass-scaled attachment metadata.

    In continuous mode ``peak_time`` is the caller-supplied accepted-series
    event.  The derivative and coefficient fields are unavailable auxiliary
    peak-fit diagnostics and are returned as NaNs; they are never used to
    define the event.
    """

    peak_knot_index: Tensor
    peak_time: Tensor
    peak_first_derivative: Tensor
    peak_second_derivative: Tensor
    peak_polynomial_coefficients: Tensor
    merger_time: Tensor
    match_time: Tensor
    match_time_final_mass: Tensor
    strictly_earlier_index: Tensor
    attachment_index: Tensor
    attachment_time: Tensor
    attachment_time_final_mass: Tensor

    @property
    def t_peak(self) -> Tensor:
        """Alias for the selected peak event time."""

        return self.peak_time

    @property
    def t_merger(self) -> Tensor:
        """Alias for the merger time in initial-total-mass units."""

        return self.merger_time

    @property
    def t_match(self) -> Tensor:
        """Alias for the matching time in initial-total-mass units."""

        return self.match_time


class RingdownAssemblyResult(NamedTuple):
    """Assembled mode and all sampling metadata.

    The ``mode``, ``amplitude``, ``phase`` and ``time`` fields are the final
    output (possibly on the optional uniform output grid).  ``joined_*`` are
    the pre-extension joined arrays, and ``extended_*`` retain the complete
    extension support before optional output interpolation.  The latter
    fields make the support and the replacement boundary directly auditable.

    ``epoch`` is the coordinate start of the returned support (the first
    output time).  It is not an observer epoch and is not shifted to the
    orbital-frequency peak.
    """

    mode: Tensor
    amplitude: Tensor
    phase: Tensor
    time: Tensor
    joined_mode: Tensor
    joined_time: Tensor
    joined_amplitude: Tensor
    joined_phase: Tensor
    extended_mode: Tensor
    extended_time: Tensor
    extended_amplitude: Tensor
    extended_phase: Tensor
    ringdown_mode: Tensor
    auxiliary_time: Tensor
    auxiliary_amplitude: Tensor
    auxiliary_phase: Tensor
    auxiliary_orbital_frequency: Tensor
    events: RingdownAssemblyEvents
    matching_time: Optional[Tensor]
    matching_state: Optional[Tensor]
    epoch: Tensor
    support_end: Tensor
    output_support_end: Tensor
    extension_samples: Tensor
    auxiliary_spacing: Tensor
    ringdown_phase0: Tensor
    auxiliary_amplitude_negative: Tensor

    @property
    def output_time(self) -> Tensor:
        """Alias for the final output time grid."""

        return self.time

    @property
    def coordinate_start(self) -> Tensor:
        """Return ``epoch`` as an explicit coordinate-start alias."""

        return self.epoch

    @property
    def t_peak(self) -> Tensor:
        """Alias for the selected orbital-frequency peak event time."""

        return self.events.peak_time

    @property
    def t_merger(self) -> Tensor:
        """Alias for the merger time in initial-total-mass units."""

        return self.events.merger_time

    @property
    def t_match(self) -> Tensor:
        """Alias for the matching time in initial-total-mass units."""

        return self.events.match_time

    @property
    def attachment_index(self) -> Tensor:
        """Alias for the selected replacement index in joined samples."""

        return self.events.attachment_index


def _check_real(name: str, value: Tensor, *, dtype=None, device=None) -> None:
    if not isinstance(value, Tensor) or value.dtype not in _REAL_DTYPES:
        raise TypeError(f"{name} must be a float32/float64 tensor")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must use the same floating dtype as time")
    if device is not None and value.device != device:
        raise TypeError(f"{name} must use the same device as time")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _parameter(name: str, value, like: Tensor) -> Tensor:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number or tensor")
    try:
        result = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    except (TypeError, RuntimeError) as exc:
        raise TypeError(f"{name} must be a real number or tensor") from exc
    if result.is_complex():
        raise TypeError(f"{name} must be real")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result


def _natural_cubic_interpolate(x: Tensor, y: Tensor, query: Tensor) -> Tensor:
    """Evaluate a natural cubic interpolant, derived from its spline system.

    ``x`` and ``query`` are one-dimensional.  The endpoint second
    derivatives are zero.  Keeping this small solver here avoids selecting a
    hidden interpolation policy from a trajectory or external library.
    """

    _check_real("spline knots", x)
    _check_real("spline values", y, dtype=x.dtype, device=x.device)
    _check_real("spline query", query, dtype=x.dtype, device=x.device)
    if x.ndim != 1 or y.ndim != 1 or query.ndim != 1:
        raise ValueError("natural cubic interpolation expects one-dimensional arrays")
    if x.shape[-1] < 2:
        raise ValueError("natural cubic interpolation needs at least two knots")
    if bool((x[1:] <= x[:-1]).any()):
        raise ValueError("spline knots must be strictly increasing")
    if bool((query < x[0]).any()) or bool((query > x[-1]).any()):
        raise ValueError("spline query lies outside its knot domain")

    n = x.shape[-1]
    h = x[1:] - x[:-1]
    if n == 2:
        interval = (torch.searchsorted(x, query, right=True) - 1).clamp(0, 0)
        dx = query - x[interval]
        return y[interval] + dx * (y[1] - y[0]) / h[0]

    interior_count = n - 2
    diagonal = 2.0 * (h[:-1] + h[1:])
    off_diagonal = h[1:-1]
    slopes = (y[1:] - y[:-1]) / h
    rhs = 6.0 * (slopes[1:] - slopes[:-1])

    # The natural-spline equations are tridiagonal.  A dense solve has
    # quadratic storage in the number of joined knots (the physical charts
    # contain O(10^4) samples), so solve the same system with Thomas forward
    # elimination/back substitution.  The no-grad branch uses compact
    # in-place work arrays; the differentiable branch keeps each recurrence
    # as a tensor expression so gradients through x and y remain available.
    if interior_count == 1:
        interior_second = rhs / diagonal
    elif not x.requires_grad and not y.requires_grad:
        reduced_diagonal = diagonal.clone()
        reduced_rhs = rhs.clone()
        for index in range(1, interior_count):
            factor = off_diagonal[index - 1] / reduced_diagonal[index - 1]
            reduced_diagonal[index] -= factor * off_diagonal[index - 1]
            reduced_rhs[index] -= factor * reduced_rhs[index - 1]
        interior_second = torch.empty_like(rhs)
        interior_second[-1] = reduced_rhs[-1] / reduced_diagonal[-1]
        for index in range(interior_count - 2, -1, -1):
            interior_second[index] = (
                reduced_rhs[index] - off_diagonal[index] * interior_second[index + 1]
            ) / reduced_diagonal[index]
    else:
        reduced_diagonal = [diagonal[0]]
        reduced_rhs = [rhs[0]]
        for index in range(1, interior_count):
            factor = off_diagonal[index - 1] / reduced_diagonal[index - 1]
            reduced_diagonal.append(
                diagonal[index] - factor * off_diagonal[index - 1]
            )
            reduced_rhs.append(rhs[index] - factor * reduced_rhs[index - 1])
        solved = [reduced_rhs[-1] / reduced_diagonal[-1]]
        for index in range(interior_count - 2, -1, -1):
            solved.append(
                (reduced_rhs[index] - off_diagonal[index] * solved[-1])
                / reduced_diagonal[index]
            )
        interior_second = torch.stack(tuple(reversed(solved)))
    second = torch.cat((
        torch.zeros(1, dtype=x.dtype, device=x.device),
        interior_second,
        torch.zeros(1, dtype=x.dtype, device=x.device),
    ))

    interval = (torch.searchsorted(x, query, right=True) - 1).clamp(0, n - 2)
    dx = query - x[interval]
    a = y[interval]
    b = (y[interval + 1] - y[interval]) / h[interval] - h[interval] * (
        2.0 * second[interval] + second[interval + 1]
    ) / 6.0
    c = second[interval] / 2.0
    d = (second[interval + 1] - second[interval]) / (6.0 * h[interval])
    return a + dx * (b + dx * (c + dx * d))


def natural_cubic_interpolate(x: Tensor, y: Tensor, query: Tensor) -> Tensor:
    """Public one-dimensional natural-cubic interpolation helper."""

    return _natural_cubic_interpolate(x, y, query)


def _uniform_spacing(time: Tensor) -> Tensor:
    if time.shape[-1] < 2:
        raise ValueError("auxiliary_time needs at least two knots")
    differences = time[1:] - time[:-1]
    if bool((differences <= 0).any()):
        raise ValueError("auxiliary_time must be strictly increasing")
    reference = differences[0]
    tolerance = 128.0 * torch.finfo(time.dtype).eps * torch.maximum(
        torch.ones_like(reference), torch.abs(time).max()
    )
    if bool((differences - reference).abs().max() > tolerance):
        raise ValueError("auxiliary_time must be uniformly spaced")
    return reference


def _single_peak_index(orbital_frequency: Tensor) -> int:
    """Return the unique interior local/global maximum index."""

    if orbital_frequency.shape[-1] < 4:
        raise ValueError("auxiliary_orbital_frequency needs at least four samples")
    left = orbital_frequency[:-2]
    center = orbital_frequency[1:-1]
    right = orbital_frequency[2:]
    maxima = (center >= left) & (center >= right) & ((center > left) | (center > right))
    candidates = torch.nonzero(maxima, as_tuple=False).flatten() + 1
    if candidates.numel() == 0:
        raise ValueError("auxiliary orbital-frequency maximum is absent")
    if candidates.numel() > 1:
        raise ValueError("multiple auxiliary orbital-frequency maxima are unsupported")
    candidate = int(candidates[0].item())
    global_index = int(torch.argmax(orbital_frequency).item())
    if global_index != candidate:
        raise ValueError("auxiliary orbital-frequency maximum must be a single interior peak")
    if candidate < 3:
        raise ValueError("orbital-frequency peak needs three left-hand neighbors")
    if candidate >= orbital_frequency.shape[-1] - 1:
        raise ValueError("orbital-frequency peak cannot be at an edge")
    return candidate


def _peak_polynomial(
    time: Tensor,
    orbital_frequency: Tensor,
    index: int,
    spacing: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fit the seven-knot degree-six peak polynomial and take one Newton step."""

    right_count = time.shape[-1] - index - 1
    left_time = time[index - 3:index]
    left_frequency = orbital_frequency[index - 3:index]
    if right_count >= 3:
        right_time = time[index + 1:index + 4]
        right_frequency = orbital_frequency[index + 1:index + 4]
    else:
        existing = max(right_count, 0)
        needed = 3 - existing
        extra_offsets = torch.arange(
            1, needed + 1, dtype=time.dtype, device=time.device
        )
        extra_time = time[-1] + spacing * extra_offsets
        last_slope = (orbital_frequency[-1] - orbital_frequency[-2]) / spacing
        extra_frequency = orbital_frequency[-1] + last_slope * spacing * extra_offsets
        right_time = torch.cat((time[index + 1:], extra_time))
        right_frequency = torch.cat((orbital_frequency[index + 1:], extra_frequency))
    knots = torch.cat((left_time, time[index:index + 1], right_time))
    ordinates = torch.cat((left_frequency, orbital_frequency[index:index + 1], right_frequency))
    if knots.shape[-1] != 7:
        raise ValueError("peak polynomial needs seven support knots")
    origin = time[index]
    local = knots - origin
    vandermonde = torch.stack(tuple(local.pow(power) for power in range(7)), dim=-1)
    coefficients = torch.linalg.solve(vandermonde, ordinates)
    first = coefficients[1]
    second = 2.0 * coefficients[2]
    safe_second = torch.where(second == 0, torch.ones_like(second), second)
    peak_candidate = origin - first / safe_second
    peak = torch.where(second == 0, origin, peak_candidate)
    return peak, first, second, coefficients


def _join_row(
    full_time: Tensor,
    full_amplitude: Tensor,
    full_phase: Tensor,
    auxiliary_time: Tensor,
    auxiliary_amplitude: Tensor,
    auxiliary_phase: Tensor,
    auxiliary_orbital_frequency: Tensor,
):
    """Apply the strict-before/second-auxiliary-knot join for one row."""

    if full_time.shape[-1] < 1 or auxiliary_time.shape[-1] < 2:
        raise ValueError("full and auxiliary series need samples for the audited join")
    if bool((full_time[1:] <= full_time[:-1]).any()):
        raise ValueError("full_time must be strictly increasing")
    if bool((auxiliary_time[1:] <= auxiliary_time[:-1]).any()):
        raise ValueError("auxiliary_time must be strictly increasing")
    cut = int((full_time < auxiliary_time[0]).sum().item())
    joined_time = torch.cat((full_time[:cut], auxiliary_time[1:]))
    joined_amplitude = torch.cat((full_amplitude[:cut], auxiliary_amplitude[1:]))
    joined_phase = torch.cat((full_phase[:cut], auxiliary_phase[1:]))
    if joined_time.shape[-1] < 4:
        raise ValueError("joined waveform needs at least four samples")
    if bool((joined_time[1:] <= joined_time[:-1]).any()):
        raise ValueError("strict waveform join must remain increasing")
    return joined_time, joined_amplitude, joined_phase, cut


def _attachment_indices(
    joined_time: Tensor,
    match_time: Tensor,
    final_mass: Tensor,
    *,
    scaled_match_time: Optional[Tensor] = None,
) -> tuple[int, int]:
    """Compare in final-mass units and select strict-earlier then previous.

    ``scaled_match_time`` is accepted so callers can retain the audited
    left-to-right arithmetic ``t_peak/final_mass - 3/final_mass +
    2/final_mass`` at the comparison boundary.
    """

    scaled_time = joined_time / final_mass
    scaled_match = (
        match_time / final_mass
        if scaled_match_time is None else scaled_match_time
    )
    strict_index = int(torch.searchsorted(scaled_time, scaled_match, right=False).item()) - 1
    attachment_index = strict_index - 1
    if strict_index < 0:
        raise ValueError("matching time has no strictly earlier joined knot")
    if attachment_index < 0:
        raise ValueError("attachment requires one knot before the strict-earlier knot")
    return strict_index, attachment_index


def _continuous_attachment_indices(
    joined_time: Tensor, match_time: Tensor, final_mass: Tensor,
) -> tuple[int, int]:
    """Return the knots bracketing a continuous matching time.

    The left index is the last strictly earlier knot.  The right index is the
    first knot at or after the physical boundary and is used only to select
    samples for analytic replacement; the event itself remains ``match_time``.
    """

    left = int(torch.searchsorted(joined_time, match_time, right=False).item()) - 1
    right = left + 1
    if left < 0:
        raise ValueError("matching time has no strictly earlier joined knot")
    if right >= joined_time.numel():
        raise ValueError("continuous matching time has no following joined knot")
    return left, right


def _unwrap_ringdown_phase(mode: Tensor, seed: Tensor) -> Tensor:
    """Unwrap the ringdown phase from a supplied phase at its first sample."""

    if mode.shape[-1] < 1:
        raise ValueError("ringdown mode needs a sample")
    if not bool(torch.isfinite(mode.real).all()) or not bool(torch.isfinite(mode.imag).all()):
        raise ValueError("ringdown mode must be finite")
    if bool((mode.abs() == 0).any()):
        raise ValueError("ringdown phase cannot be unwrapped through zero amplitude")
    if mode.shape[-1] == 1:
        return seed.reshape(1)
    increments = -torch.angle(mode[1:] * mode[:-1].conj())
    return torch.cat((seed.reshape(1), seed + increments.cumsum(dim=-1)))


def _assemble_row(
    full_time: Tensor,
    full_amplitude: Tensor,
    full_phase: Tensor,
    auxiliary_time: Tensor,
    auxiliary_amplitude: Tensor,
    auxiliary_phase: Tensor,
    auxiliary_orbital_frequency: Tensor,
    nu: Tensor,
    final_mass: Tensor,
    final_spin: Tensor,
    merger_frequency_value: Optional[Tensor],
    output_step: Optional[Tensor],
    matching_method: str,
    explicit_peak_time: Optional[Tensor],
    support_end: Optional[Tensor],
):
    spacing = _uniform_spacing(auxiliary_time)
    joined_time, joined_amplitude, joined_phase, cut = _join_row(
        full_time,
        full_amplitude,
        full_phase,
        auxiliary_time,
        auxiliary_amplitude,
        auxiliary_phase,
        auxiliary_orbital_frequency,
    )
    if matching_method == "grid":
        peak_index = _single_peak_index(auxiliary_orbital_frequency)
        peak_time, peak_first, peak_second, peak_coefficients = _peak_polynomial(
            auxiliary_time, auxiliary_orbital_frequency, peak_index, spacing
        )
    else:
        peak_time = explicit_peak_time
        # These are diagnostics only in continuous mode: the explicit
        # accepted-trajectory peak is authoritative and may be used with a
        # flat or tied auxiliary frequency curve.  A bracketing knot index is
        # retained; auxiliary polynomial diagnostics are unavailable.
        peak_index = int(torch.searchsorted(auxiliary_time, peak_time, right=False).item())
        if peak_index <= 0 or peak_index >= auxiliary_time.numel():
            peak_index = -1
        peak_first = torch.full_like(peak_time, torch.nan)
        peak_second = torch.full_like(peak_time, torch.nan)
        peak_coefficients = torch.full((7,), torch.nan, dtype=peak_time.dtype,
                                       device=peak_time.device)
    if matching_method == "continuous":
        if peak_time is None:  # defensive; validated at the public boundary
            raise ValueError("continuous matching requires peak_time")
        if bool((peak_time < full_time[0]) | (peak_time > full_time[-1])):
            raise ValueError("peak_time lies outside full-series support")
    merger_time = peak_time - torch.ones_like(peak_time) * 3.0
    match_time = (merger_time + torch.ones_like(merger_time) * 2.0
                  if matching_method == "grid" else
                  peak_time - torch.ones_like(peak_time))
    if matching_method == "continuous":
        if bool((match_time < full_time[0]) | (match_time > full_time[-1])):
            raise ValueError("continuous matching time lies outside full-series support")
    # Keep the attachment threshold's specified floating-point operation
    # order.  ``match_time/final_mass`` is algebraically equivalent but can
    # choose a different side of an exact knot in finite precision.
    if matching_method == "grid":
        match_time_final_mass = (
            peak_time / final_mass
            - torch.ones_like(peak_time) * 3.0 / final_mass
            + torch.ones_like(peak_time) * 2.0 / final_mass
        )
    else:
        match_time_final_mass = match_time / final_mass
    if matching_method == "grid":
        strict_index, attachment_index = _attachment_indices(
            joined_time, match_time, final_mass,
            scaled_match_time=match_time_final_mass,
        )
        attachment_time = joined_time[attachment_index]
    else:
        strict_index, attachment_index = _continuous_attachment_indices(
            joined_time, match_time, final_mass,
        )
        attachment_time = match_time

    if support_end is None:
        # Keep the legacy 1500-knot path byte-for-byte equivalent when no
        # explicit support is requested.
        extension_samples = _EXTENSION_SAMPLES
        extension_offsets = torch.arange(
            1, _EXTENSION_SAMPLES + 1, dtype=joined_time.dtype, device=joined_time.device
        )
    else:
        if not bool(torch.isfinite(support_end)) or not bool(support_end > 0):
            raise ValueError("support_end must be a positive finite scalar")
        if not bool(support_end > joined_time[-1]):
            raise ValueError("support_end must be later than the joined endpoint")
        span = support_end - joined_time[-1]
        extension_samples = int(torch.ceil(span / spacing).item())
        if extension_samples < 1:
            raise ValueError("support_end must leave at least one extension knot")
        extension_offsets = torch.arange(
            1, extension_samples + 1, dtype=joined_time.dtype, device=joined_time.device
        )
    extended_time = torch.cat((
        joined_time,
        joined_time[-1] + spacing * extension_offsets,
    ))
    if support_end is not None:
        # The last interval may be shorter than the auxiliary spacing.  The
        # ceil above ensures this clamp remains strictly after its predecessor.
        extended_time = torch.cat((extended_time[:-1], support_end.reshape(1)))
        if bool((extended_time[1:] <= extended_time[:-1]).any()):
            raise ValueError("support_end must produce strictly increasing extension knots")
    evaluate_time = torch.maximum(extended_time, merger_time)
    postmerger_kwargs = dict(
        nu=nu,
        final_mass=final_mass,
        final_spin=final_spin,
        merger_time=merger_time,
    )
    if merger_frequency_value is not None:
        postmerger_kwargs["merger_frequency_value"] = merger_frequency_value
    zero_phase = torch.zeros_like(merger_time)
    if matching_method == "continuous":
        corrected_anchor = (
            _natural_cubic_interpolate(
                full_time, full_amplitude, match_time.reshape(1)
            )[0]
            * torch.exp(-1j * _natural_cubic_interpolate(
                full_time, full_phase, match_time.reshape(1)
            )[0])
        )
    else:
        corrected_anchor = joined_amplitude[attachment_index] * torch.exp(
            -1j * joined_phase[attachment_index]
        )
    baseline_anchor = postmerger_22(
        (attachment_time if matching_method == "grid" else match_time).reshape(1),
        phase0=zero_phase, **postmerger_kwargs
    )[0]
    if bool((corrected_anchor.abs() == 0).any()):
        raise ValueError("attachment phase cannot be anchored at zero amplitude")
    ringdown_phase0 = torch.angle(baseline_anchor) - torch.angle(corrected_anchor)
    ringdown_mode = postmerger_22(
        evaluate_time, phase0=ringdown_phase0, **postmerger_kwargs
    )

    if matching_method == "continuous":
        pre_amplitude = _natural_cubic_interpolate(
            full_time, full_amplitude, joined_time[:attachment_index]
        )
        pre_phase = _natural_cubic_interpolate(
            full_time, full_phase, joined_time[:attachment_index]
        )
        joined_for_replacement_amplitude = torch.cat(
            (pre_amplitude, joined_amplitude[attachment_index:])
        )
        joined_for_replacement_phase = torch.cat(
            (pre_phase, joined_phase[attachment_index:])
        )
    else:
        joined_for_replacement_amplitude = joined_amplitude
        joined_for_replacement_phase = joined_phase
    joined_mode = joined_amplitude * torch.exp(-1j * joined_phase)
    replacement_base_mode = joined_for_replacement_amplitude * torch.exp(
        -1j * joined_for_replacement_phase
    )
    base_extended_mode = torch.cat((
        replacement_base_mode,
        torch.zeros(extension_samples, dtype=ringdown_mode.dtype, device=ringdown_mode.device),
    ))
    replacement = torch.arange(
        extended_time.shape[-1], device=extended_time.device
    ) >= attachment_index
    extended_mode = torch.where(replacement, ringdown_mode, base_extended_mode)
    base_extended_amplitude = torch.cat((
        joined_for_replacement_amplitude,
        torch.zeros(extension_samples, dtype=joined_amplitude.dtype, device=joined_amplitude.device),
    ))
    # The ansatz carries its own physical amplitude.  No anchor rescaling is
    # applied, and the signed pre-attachment input amplitude stays signed.
    ringdown_amplitude = ringdown_mode.abs()
    extended_amplitude = torch.where(replacement, ringdown_amplitude, base_extended_amplitude)

    ringdown_anchor = ringdown_mode[attachment_index]
    ringdown_seed_mod = -torch.angle(ringdown_anchor)
    two_pi = torch.as_tensor(2.0 * torch.pi, dtype=joined_phase.dtype, device=joined_phase.device)
    if matching_method == "continuous":
        phase_at_match = _natural_cubic_interpolate(
            full_time, full_phase, match_time.reshape(1)
        )[0]
        anchor_mode = postmerger_22(
            match_time.reshape(1), phase0=ringdown_phase0, **postmerger_kwargs
        )
        anchor_phase_mod = -torch.angle(anchor_mode[0])
        ringdown_seed = anchor_phase_mod + two_pi * torch.round(
            (phase_at_match - anchor_phase_mod) / two_pi
        )
        ringdown_phase_tail = _unwrap_ringdown_phase(
            torch.cat((anchor_mode, ringdown_mode[attachment_index:])),
            ringdown_seed,
        )[1:]
    else:
        ringdown_seed = ringdown_seed_mod + two_pi * torch.round(
            (joined_phase[attachment_index] - ringdown_seed_mod) / two_pi
        )
        ringdown_phase_tail = _unwrap_ringdown_phase(
            ringdown_mode[attachment_index:], ringdown_seed
        )
    extended_phase = torch.cat((
        joined_for_replacement_phase[:attachment_index], ringdown_phase_tail
    ))
    if extended_phase.shape != extended_time.shape:
        raise RuntimeError("internal ringdown phase support shape mismatch")

    if output_step is None:
        output_time = extended_time
        output_amplitude = extended_amplitude
        output_phase = extended_phase
    else:
        if bool((output_step <= 0).any()):
            raise ValueError("output_step must be positive")
        span = extended_time[-1] - extended_time[0]
        count = int(torch.floor(span / output_step).item()) + 1
        if count < 1:
            raise ValueError("output grid has no support")
        output_time = extended_time[0] + output_step * torch.arange(
            count, dtype=extended_time.dtype, device=extended_time.device
        )
        if matching_method == "grid":
            output_amplitude = _natural_cubic_interpolate(
                extended_time, extended_amplitude, output_time
            )
            output_phase = _natural_cubic_interpolate(
                extended_time, extended_phase, output_time
            )
        else:
            before = output_time < match_time
            output_amplitude = torch.empty_like(output_time)
            output_phase = torch.empty_like(output_time)
            if bool(before.any()):
                pre_time = output_time[before]
                output_amplitude[before] = _natural_cubic_interpolate(
                    full_time, full_amplitude, pre_time
                )
                output_phase[before] = _natural_cubic_interpolate(
                    full_time, full_phase, pre_time
                )
            if bool((~before).any()):
                post_time = output_time[~before]
                post_mode = postmerger_22(
                    torch.maximum(post_time, merger_time),
                    phase0=ringdown_phase0, **postmerger_kwargs
                )
                output_amplitude[~before] = post_mode.abs()
                phase_at_match = _natural_cubic_interpolate(
                    full_time, full_phase, match_time.reshape(1)
                )[0]
                anchor_mode = postmerger_22(
                    match_time.reshape(1), phase0=ringdown_phase0,
                    **postmerger_kwargs
                )
                anchor_phase_mod = -torch.angle(anchor_mode[0])
                ringdown_seed = anchor_phase_mod + two_pi * torch.round(
                    (phase_at_match - anchor_phase_mod) / two_pi
                )
                output_phase[~before] = _unwrap_ringdown_phase(
                    torch.cat((anchor_mode, post_mode)), ringdown_seed
                )[1:]
    output_mode = output_amplitude * torch.exp(-1j * output_phase)

    events = RingdownAssemblyEvents(
        torch.as_tensor(peak_index, dtype=torch.long, device=joined_time.device),
        peak_time,
        peak_first,
        peak_second,
        peak_coefficients,
        merger_time,
        match_time,
        match_time_final_mass,
        torch.as_tensor(strict_index, dtype=torch.long, device=joined_time.device),
        torch.as_tensor(attachment_index, dtype=torch.long, device=joined_time.device),
        attachment_time,
        attachment_time / final_mass,
    )
    return {
        "mode": output_mode,
        "amplitude": output_amplitude,
        "phase": output_phase,
        "time": output_time,
        "joined_mode": joined_mode,
        "joined_time": joined_time,
        "joined_amplitude": joined_amplitude,
        "joined_phase": joined_phase,
        "extended_mode": extended_mode,
        "extended_time": extended_time,
        "extended_amplitude": extended_amplitude,
        "extended_phase": extended_phase,
        "ringdown_mode": ringdown_mode,
        "events": events,
        "cut_count": cut,
        "spacing": spacing,
        "extension_samples": torch.as_tensor(
            extension_samples, dtype=torch.long, device=joined_time.device
        ),
        "ringdown_phase0": ringdown_phase0,
        "auxiliary_amplitude_negative": (auxiliary_amplitude < 0).any(),
    }


def _reshape_rows(values, batch_shape: tuple[int, ...]):
    stacked = torch.stack(values, dim=0)
    if not batch_shape:
        return stacked[0]
    return stacked.reshape(batch_shape + stacked.shape[1:])


def assemble_teob_ringdown_22(
    full_time: Tensor,
    full_amplitude: Tensor,
    full_phase: Tensor,
    auxiliary_time: Tensor,
    auxiliary_amplitude: Tensor,
    auxiliary_phase: Tensor,
    auxiliary_orbital_frequency: Tensor,
    nu,
    final_mass,
    final_spin,
    *,
    matching_time: Optional[Tensor] = None,
    matching_state: Optional[Tensor] = None,
    output_step=None,
    merger_frequency_value=None,
    matching_method: str = "grid",
    peak_time: Optional[Tensor] = None,
    support_end: Optional[Tensor] = None,
) -> RingdownAssemblyResult:
    """Join corrected samples and append the published TEOB postmerger.

    All seven series arguments use their last axis for samples and must share
    a float32/float64 dtype and device.  The full series keeps samples with
    ``full_time < auxiliary_time[0]``.  The auxiliary series starts at its
    second knot, so the common first knot is deliberately omitted.  The grid
    policy fits a unique interior orbital-frequency peak with seven knots;
    its one-step Newton estimate defines ``t_merger=t_peak-3`` and
    ``t_match=t_merger+2``.  Continuous policy uses the explicit
    accepted-series ``peak_time`` and does not fit the auxiliary frequency
    peak; its derivative/coefficient diagnostics are NaN.
    Attachment comparison scales both joined and matching times by
    ``1/final_mass`` and uses the last strict-earlier knot followed by one
    knot before it.

    ``output_step`` optionally requests a uniform output grid.  It is a
    scalar across a batch because a rectangular output tensor is required;
    amplitude and phase are interpolated independently with natural cubics.
    ``matching_time`` and ``matching_state`` are opaque metadata from the
    NQC stage and are returned unchanged.  ``matching_method="grid"`` keeps
    the legacy grid-snapped attachment.  ``matching_method="continuous"``
    requires the accepted-trajectory ``peak_time``; it sets
    ``t_merger=peak_time-3`` and ``t_match=peak_time-1``, evaluates the full
    corrected series at that exact match, and uses that value only to phase
    anchor the analytic ansatz.  Its attachment index identifies the first
    joined knot at or after the physical boundary, while attachment_time is
    the continuous boundary itself.

    ``support_end`` optionally sets the absolute end of the extended support
    in initial-total-mass time units.  It must be a positive finite scalar (or
    one scalar per batch row), later than the joined endpoint.  The final
    extension knot is clamped exactly to this value; this is an end coordinate,
    rather than a duration.  With ``support_end=None`` the legacy 1500
    auxiliary-spacing knots are retained exactly.
    """

    series = (
        ("full_time", full_time),
        ("full_amplitude", full_amplitude),
        ("full_phase", full_phase),
        ("auxiliary_time", auxiliary_time),
        ("auxiliary_amplitude", auxiliary_amplitude),
        ("auxiliary_phase", auxiliary_phase),
        ("auxiliary_orbital_frequency", auxiliary_orbital_frequency),
    )
    if not all(isinstance(value, Tensor) for _, value in series):
        raise TypeError("waveform series must be supplied as torch tensors")
    if matching_method not in ("grid", "continuous"):
        raise ValueError("matching_method must be 'grid' or 'continuous'")
    dtype = full_time.dtype
    device = full_time.device
    for name, value in series:
        _check_real(name, value, dtype=dtype, device=device)
        if value.ndim < 1:
            raise ValueError(f"{name} must have a sample axis")
    full_count = full_time.shape[-1]
    auxiliary_count = auxiliary_time.shape[-1]
    if full_amplitude.shape[-1] != full_count or full_phase.shape[-1] != full_count:
        raise ValueError("full time, amplitude, and phase must share their sample count")
    if any(value.shape[-1] != auxiliary_count for value in (
        auxiliary_amplitude, auxiliary_phase, auxiliary_orbital_frequency
    )):
        raise ValueError("auxiliary arrays must share their sample count")

    nu_tensor = _parameter("nu", nu, full_time)
    mass_tensor = _parameter("final_mass", final_mass, full_time)
    spin_tensor = _parameter("final_spin", final_spin, full_time)
    if bool((mass_tensor <= 0).any()):
        raise ValueError("final_mass must be positive")
    if nu_tensor.ndim > 0:
        parameter_batch = nu_tensor.shape
    else:
        parameter_batch = ()
    if matching_method == "continuous":
        if peak_time is None:
            raise ValueError("continuous matching requires explicit peak_time")
        peak_tensor = _parameter("peak_time", peak_time, full_time)
        parameter_batch = torch.broadcast_shapes(parameter_batch, peak_tensor.shape)
    else:
        peak_tensor = None
    if support_end is not None:
        support_end_tensor = _parameter("support_end", support_end, full_time)
        parameter_batch = torch.broadcast_shapes(parameter_batch, support_end_tensor.shape)
    else:
        support_end_tensor = None
    batch_shape = torch.broadcast_shapes(
        *(value.shape[:-1] for _, value in series),
        parameter_batch,
        mass_tensor.shape,
        spin_tensor.shape,
    )
    if output_step is not None:
        output_step_tensor = _parameter("output_step", output_step, full_time)
        if output_step_tensor.ndim > 0:
            if not bool(torch.all(output_step_tensor == output_step_tensor.reshape(-1)[0])):
                raise ValueError("output_step must be one scalar across a batch")
            output_step_tensor = output_step_tensor.reshape(-1)[0]
        if bool((output_step_tensor <= 0).any()):
            raise ValueError("output_step must be positive")
    else:
        output_step_tensor = None
    if merger_frequency_value is not None:
        merger_frequency_tensor = _parameter(
            "merger_frequency_value", merger_frequency_value, full_time
        )
    else:
        merger_frequency_tensor = None

    expanded_series = [
        value.expand(batch_shape + (value.shape[-1],)) for _, value in series
    ]
    nu_tensor = nu_tensor.expand(batch_shape)
    mass_tensor = mass_tensor.expand(batch_shape)
    spin_tensor = spin_tensor.expand(batch_shape)
    if peak_tensor is not None:
        peak_tensor = peak_tensor.expand(batch_shape)
    if support_end_tensor is not None:
        support_end_tensor = support_end_tensor.expand(batch_shape)
    if merger_frequency_tensor is not None:
        merger_frequency_tensor = merger_frequency_tensor.expand(batch_shape)
    row_count = 1
    for dimension in batch_shape:
        row_count *= dimension
    flat_series = [value.reshape((row_count, value.shape[-1])) for value in expanded_series]
    flat_nu = nu_tensor.reshape(row_count)
    flat_mass = mass_tensor.reshape(row_count)
    flat_spin = spin_tensor.reshape(row_count)
    flat_support_end = (
        None if support_end_tensor is None else support_end_tensor.reshape(row_count)
    )
    flat_merger_frequency = (
        None if merger_frequency_tensor is None else merger_frequency_tensor.reshape(row_count)
    )

    rows = []
    for row in range(row_count):
        rows.append(_assemble_row(
            *(value[row] for value in flat_series),
            flat_nu[row],
            flat_mass[row],
            flat_spin[row],
            None if flat_merger_frequency is None else flat_merger_frequency[row],
            output_step_tensor,
            matching_method,
            None if peak_tensor is None else peak_tensor.reshape(row_count)[row],
            None if flat_support_end is None else flat_support_end[row],
        ))
    joined_lengths = {item["joined_time"].shape[-1] for item in rows}
    if len(joined_lengths) != 1:
        raise ValueError("batch rows must use the same strict waveform join length")
    extended_lengths = {item["extended_time"].shape[-1] for item in rows}
    output_lengths = {item["time"].shape[-1] for item in rows}
    if len(extended_lengths) != 1 or len(output_lengths) != 1:
        raise ValueError(
            "batch rows must use the same extended/output support length; "
            "use a common support_end and output_step"
        )

    events = RingdownAssemblyEvents(*(
        _reshape_rows([item["events"][field] for item in rows], batch_shape)
        for field in range(len(RingdownAssemblyEvents._fields))
    ))
    # The explicit metadata is opaque to this component.  Preserve the
    # caller's objects unchanged; the NQC stage owns their shape and units.
    return RingdownAssemblyResult(
        _reshape_rows([item["mode"] for item in rows], batch_shape),
        _reshape_rows([item["amplitude"] for item in rows], batch_shape),
        _reshape_rows([item["phase"] for item in rows], batch_shape),
        _reshape_rows([item["time"] for item in rows], batch_shape),
        _reshape_rows([item["joined_mode"] for item in rows], batch_shape),
        _reshape_rows([item["joined_time"] for item in rows], batch_shape),
        _reshape_rows([item["joined_amplitude"] for item in rows], batch_shape),
        _reshape_rows([item["joined_phase"] for item in rows], batch_shape),
        _reshape_rows([item["extended_mode"] for item in rows], batch_shape),
        _reshape_rows([item["extended_time"] for item in rows], batch_shape),
        _reshape_rows([item["extended_amplitude"] for item in rows], batch_shape),
        _reshape_rows([item["extended_phase"] for item in rows], batch_shape),
        _reshape_rows([item["ringdown_mode"] for item in rows], batch_shape),
        expanded_series[3].reshape(batch_shape + (auxiliary_count,)),
        expanded_series[4].reshape(batch_shape + (auxiliary_count,)),
        expanded_series[5].reshape(batch_shape + (auxiliary_count,)),
        expanded_series[6].reshape(batch_shape + (auxiliary_count,)),
        events,
        matching_time,
        matching_state,
        _reshape_rows([item["time"][0] for item in rows], batch_shape),
        _reshape_rows([item["extended_time"][-1] for item in rows], batch_shape),
        _reshape_rows([item["time"][-1] for item in rows], batch_shape),
        (
            torch.as_tensor(_EXTENSION_SAMPLES, dtype=torch.long, device=device)
            if support_end_tensor is None else
            _reshape_rows([item["extension_samples"] for item in rows], batch_shape)
        ),
        _reshape_rows([item["spacing"] for item in rows], batch_shape),
        _reshape_rows([item["ringdown_phase0"] for item in rows], batch_shape),
        _reshape_rows([item["auxiliary_amplitude_negative"] for item in rows], batch_shape),
    )


# Short names keep the hand-off discoverable without creating another path.
assemble_ringdown_22 = assemble_teob_ringdown_22
teob_ringdown_assembly = assemble_teob_ringdown_22


__all__ = [
    "CorrectedWaveformSeries",
    "RingdownAssemblyEvents",
    "RingdownAssemblyResult",
    "natural_cubic_interpolate",
    "assemble_teob_ringdown_22",
    "assemble_ringdown_22",
    "teob_ringdown_assembly",
]

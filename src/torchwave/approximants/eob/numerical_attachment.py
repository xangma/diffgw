"""Numerical SEOBNRv4 attachment conventions.

This module contains the bounded numerical part of the clean-room v4
attachment contract.  It is deliberately separate from :mod:`attachment`,
whose local polynomial routine remains the default experimental path.

The input chart is dimensionless ``tau=t/M``.  Radii and momenta have the
usual EOB dimensionless conventions, ``A=abs(h22)`` is ``(D/M)`` times the
strain amplitude, and ``omega22=-d arg(h22)/d tau`` is positive.  A single
natural cubic spline is built over the complete supplied high-resolution
series for every sampled product ``A*q_j`` and every phase basis ``p_j``.
The base amplitude and the continuously unwrapped waveform phase get their
own splines as required by the numerical convention.  No SciPy or external
calibration data are used at runtime.

The ringdown helper is opt-in.  It attaches at the greatest grid sample no
later than the continuous NQC matching time (an exact match uses that knot).
Its boundary values come from
the corrected sampled mode, its amplitude derivative is zero, and its
positive frequency is the backward phase difference over one input-grid
interval.  ``internal_step`` records the geometric oversampling rule used by
the published convention::

    dt_internal = dt_output / 2**max(0, ceil(log2(50*dt_output/M))).

The free ringdown amplitude rate is projected to the ansatz's non-increasing
peak boundary if the printed fit and supplied QNM damping would bend the
attachment upward. The helper does not alter the default production EOB route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .events import OrbitalFrequencyPeak
from .merger_ringdown import (merger_ringdown_22, seobnrv4_merger_shape,
                              seobnrv4_peak_preserving_shape)
from .nqc import (
    NQCCalibration,
    NQCCoefficients,
    NQCMatchingTargets,
    _matching_solve,
    nqc_22_basis,
    nqc_22_factor,
    seobnrv4_nqc_targets,
    seobnrv4_nqc_time_offset_numerical_compatibility,
)


def _check_real(name: str, value: Tensor, *, dtype=None, device=None) -> None:
    if not isinstance(value, Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be a float32/float64 tensor")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must use the same floating dtype as time")
    if device is not None and value.device != device:
        raise TypeError(f"{name} must use the same device as time")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _natural_second_derivatives(x: Tensor, values: Tensor) -> Tensor:
    """Return natural-cubic knot second derivatives.

    ``values`` is ``(..., n, channels)`` and ``x`` is ``(..., n)``.  The
    interior tridiagonal system is solved by the Thomas recurrence.  The
    recurrence uses only Torch operations, so derivatives through ordinates,
    knots, and the resulting matching systems remain available.
    """
    n = x.shape[-1]
    result = torch.zeros_like(values)
    if n <= 2:
        return result

    h = x[..., 1:] - x[..., :-1]
    slopes = (values[..., 1:, :] - values[..., :-1, :]) / h.unsqueeze(-1)
    # There are m=n-2 unknown interior second derivatives.
    m = n - 2
    # For interior row ``j`` (unknown ``M[j+1]``), the lower coefficient
    # multiplies ``M[j]`` and is ``h[j]``.  Thomas uses ``lower[j-1]`` while
    # eliminating row ``j``, hence this is the same h[1:-1] sequence as the
    # upper diagonal.  Using h[:-1] here is harmless on uniform grids but
    # solves the wrong system whenever adjacent intervals differ.
    lower = h[..., 1:-1]
    diagonal = 2 * (h[..., :-1] + h[..., 1:])
    upper = h[..., 1:-1]
    rhs = 6 * (slopes[..., 1:, :] - slopes[..., :-1, :])

    # Forward Thomas elimination.  Keeping the short recurrence in Python
    # avoids constructing an O(n^2) dense matrix for public trajectories.
    cprime = []
    dprime = []
    denominator = diagonal[..., 0]
    if m > 1:
        cprime.append(upper[..., 0] / denominator)
    dprime.append(rhs[..., 0, :] / denominator.unsqueeze(-1))
    for j in range(1, m):
        denominator = diagonal[..., j] - lower[..., j - 1] * cprime[j - 1]
        if m > j + 1:
            cprime.append(upper[..., j] / denominator)
        dprime.append((rhs[..., j, :] - lower[..., j - 1].unsqueeze(-1) * dprime[-1])
                      / denominator.unsqueeze(-1))

    interior = [None] * m
    interior[-1] = dprime[-1]
    for j in range(m - 2, -1, -1):
        interior[j] = dprime[j] - cprime[j].unsqueeze(-1) * interior[j + 1]
    result[..., 1:-1, :] = torch.stack(interior, dim=-2)
    return result


@dataclass(frozen=True)
class NaturalCubicSpline:
    """A differentiable natural cubic interpolant on the last sample axis.

    ``knots`` has shape ``(..., n)``.  Internally, values and all four
    interval coefficient arrays have shape ``(..., n-1, channels)``.  A
    one-channel input returns the channel axis squeezed; a multichannel input
    returns a final channel axis.  Query coordinates are allowed to carry
    leading batch axes and are broadcast with the spline batch.
    """

    knots: Tensor
    values: Tensor
    a: Tensor
    b: Tensor
    c: Tensor
    d: Tensor
    scalar_output: bool

    @property
    def x(self) -> Tensor:
        """Alias for the knot coordinates."""
        return self.knots

    @property
    def y(self) -> Tensor:
        """Values with the sample axis last (and no synthetic channel axis)."""
        return self.values[..., 0] if self.scalar_output else self.values

    @property
    def channels(self) -> int:
        return self.values.shape[-1]

    def evaluate(self, query: Tensor, derivative: int = 0, *, extrapolate: bool = False) -> Tensor:
        """Evaluate value, first derivative, or second derivative.

        The derivative is with respect to the same ``tau=t/M`` coordinate as
        the knots.  Out-of-domain queries raise by default; attachment code
        clamps masked queries before evaluation so it never extrapolates.
        """
        if not isinstance(query, Tensor):
            raise TypeError("spline query must be a tensor")
        if query.dtype != self.knots.dtype or query.device != self.knots.device:
            raise TypeError("spline query must share knots' dtype and device")
        if derivative not in (0, 1, 2):
            raise ValueError("spline derivative must be 0, 1, or 2")
        if not bool(torch.isfinite(query).all()):
            raise ValueError("spline query must be finite")

        scalar_query = query.ndim == 0
        q = query.reshape(1) if scalar_query else query
        batch = torch.broadcast_shapes(self.knots.shape[:-1], q.shape[:-1])
        n = self.knots.shape[-1]
        channels = self.channels
        knots = self.knots.expand(batch + (n,))
        q = q.expand(batch + (q.shape[-1],))
        if not extrapolate:
            inside = (q >= knots[..., :1]) & (q <= knots[..., -1:])
            if not bool(inside.all()):
                raise ValueError("spline query lies outside its knot domain")
        # For extrapolation, retain the endpoint interval and let its cubic
        # polynomial continue beyond the endpoint.  The default attachment
        # path clips masked queries explicitly and never requests this mode.
        q_eval = q if extrapolate else q.clamp(min=knots[..., :1], max=knots[..., -1:])

        # Batched searchsorted is supported by Torch and keeps interval
        # selection discrete while dx and the polynomial remain differentiable.
        interval = (torch.searchsorted(knots.contiguous(), q_eval.contiguous(), right=True)
                    - 1).clamp(min=0, max=n - 2)
        dx = q_eval - torch.gather(knots, -1, interval)
        gather_index = interval.unsqueeze(-1).expand(batch + (q.shape[-1], channels))
        a = torch.gather(self.a.expand(batch + (n - 1, channels)), -2, gather_index)
        b = torch.gather(self.b.expand(batch + (n - 1, channels)), -2, gather_index)
        c = torch.gather(self.c.expand(batch + (n - 1, channels)), -2, gather_index)
        d = torch.gather(self.d.expand(batch + (n - 1, channels)), -2, gather_index)
        dx = dx.unsqueeze(-1)
        if derivative == 0:
            result = a + dx * (b + dx * (c + dx * d))
        elif derivative == 1:
            result = b + dx * (2 * c + dx * 3 * d)
        else:
            result = 2 * c + dx * 6 * d
        if self.scalar_output:
            result = result[..., 0]
        if scalar_query:
            result = result[..., 0, :] if not self.scalar_output else result[..., 0]
        return result

    __call__ = evaluate

    def derivative(self, query: Tensor, order: int = 1, *, extrapolate: bool = False) -> Tensor:
        """Convenience alias for :meth:`evaluate`."""
        return self.evaluate(query, derivative=order, extrapolate=extrapolate)


def natural_cubic_spline(
    x: Tensor,
    values: Tensor,
    *,
    scalar_output: Optional[bool] = None,
    scalar: Optional[bool] = None,
) -> NaturalCubicSpline:
    """Construct a full-series natural cubic spline without SciPy.

    ``x`` uses its last axis for samples.  Scalar ordinates use shape
    ``(..., n)`` and multichannel ordinates use ``(..., n, C)``.  Leading
    axes broadcast.  With the default ``scalar_output=None``, the historical
    ``(..., n, C)`` interpretation wins when a one-dimensional knot vector
    and an ``(n, n)`` value array are ambiguous; pass ``scalar_output=True``
    to interpret that array as ``n`` batched scalar series, or
    ``scalar_output=False`` to force channel semantics.  A one-channel
    ``(..., n, 1)`` input keeps the historical squeezed scalar result unless
    ``scalar_output=False`` is requested.  ``scalar`` is a short alias for
    ``scalar_output``.

    At least two strictly increasing knots are required; natural endpoint
    second derivatives are exactly zero.
    """
    _check_real("x", x)
    if x.ndim < 1 or x.shape[-1] < 2:
        raise ValueError("natural cubic spline needs at least two knots")
    if not isinstance(values, Tensor) or values.dtype != x.dtype or values.device != x.device:
        raise TypeError("spline values must share x's floating dtype and device")
    if scalar is not None:
        if not isinstance(scalar, bool):
            raise TypeError("scalar must be a bool when provided")
        if scalar_output is not None and scalar_output != scalar:
            raise ValueError("scalar and scalar_output disagree")
        scalar_output = scalar
    if scalar_output is not None and not isinstance(scalar_output, bool):
        raise TypeError("scalar_output must be a bool when provided")

    n = x.shape[-1]
    has_sample_axis = values.ndim >= 1 and values.shape[-1] == n
    has_channel_axis = values.ndim >= 2 and values.shape[-2] == n
    if scalar_output is True:
        # Explicit scalar mode always treats the final axis as samples.  An
        # explicitly supplied singleton channel is accepted as a convenience
        # and still returns the scalar result promised by this mode.
        if has_sample_axis:
            ordinate = values.unsqueeze(-1)
        elif has_channel_axis and values.shape[-1] == 1:
            ordinate = values
        else:
            raise ValueError("scalar spline values must end in the knot axis")
    elif scalar_output is False:
        if not has_channel_axis:
            raise ValueError("channel spline values must end in (..., knots, channels)")
        ordinate = values
    elif values.ndim == x.ndim and has_sample_axis:
        # Existing batched scalar callers with matching x/value rank.
        ordinate = values.unsqueeze(-1)
        scalar_output = True
    elif has_channel_axis:
        # Preserve existing channel callers, including x=(n), values=(n,C).
        ordinate = values
        scalar_output = values.shape[-1] == 1
    elif has_sample_axis:
        # A shared one-dimensional knot vector plus (..., n) values is a
        # batched scalar family.  This branch fixes the formerly rejected
        # shared-knot shape while retaining the explicit ambiguity option.
        ordinate = values.unsqueeze(-1)
        scalar_output = True
    else:
        raise ValueError("spline values must end in the knot axis or (..., knots, channels)")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("spline values must be finite")
    if bool((x.diff(dim=-1) <= 0).any()):
        raise ValueError("spline knots must be strictly increasing")

    batch = torch.broadcast_shapes(x.shape[:-1], ordinate.shape[:-2])
    channels = ordinate.shape[-1]
    knots = x.expand(batch + (n,))
    ordinate = ordinate.expand(batch + (n, channels))
    second = _natural_second_derivatives(knots, ordinate)
    h = knots[..., 1:] - knots[..., :-1]
    a = ordinate[..., :-1, :]
    b = ((ordinate[..., 1:, :] - ordinate[..., :-1, :]) / h.unsqueeze(-1)
         - h.unsqueeze(-1) * (2 * second[..., :-1, :] + second[..., 1:, :]) / 6)
    c = second[..., :-1, :] / 2
    d = (second[..., 1:, :] - second[..., :-1, :]) / (6 * h.unsqueeze(-1))
    return NaturalCubicSpline(knots, ordinate, a, b, c, d, scalar_output)


def natural_cubic_coefficients(
    x: Tensor,
    values: Tensor,
    *,
    scalar_output: Optional[bool] = None,
    scalar: Optional[bool] = None,
):
    """Return ``(a,b,c,d)`` interval coefficients for a natural spline."""
    spline = natural_cubic_spline(x, values, scalar_output=scalar_output, scalar=scalar)
    return spline.a, spline.b, spline.c, spline.d


def seobnrv4_internal_step(dt_output: Tensor, mass_scale: Tensor | float = 1.0) -> Tensor:
    """Return the documented power-of-two internal geometric step.

    ``dt_output`` and ``mass_scale`` use the same time unit.  In the usual
    dimensionless chart both are in units of the initial total mass, so the
    default ``mass_scale=1`` applies.  The returned step is always positive
    and no larger than ``0.02*mass_scale`` whenever ``dt_output`` is positive.
    """
    _check_real("dt_output", dt_output)
    mass = torch.as_tensor(mass_scale, dtype=dt_output.dtype, device=dt_output.device)
    _check_real("mass_scale", mass, dtype=dt_output.dtype, device=dt_output.device)
    dt_output, mass = torch.broadcast_tensors(dt_output, mass)
    if bool((dt_output <= 0).any()) or bool((mass <= 0).any()):
        raise ValueError("dt_output and mass_scale must be positive")
    ratio = 50 * dt_output / mass
    exponent = torch.ceil(torch.log2(ratio)).clamp_min(0)
    return dt_output / torch.pow(torch.as_tensor(2., dtype=dt_output.dtype, device=dt_output.device), exponent)


def seobnrv4_ringdown_support_samples(damping: Tensor, dt_high: Tensor) -> Tensor:
    """Return the discrete high-grid ringdown support count.

    The supported samples are elapsed times ``j*dt_high`` for
    ``j=0,...,count-1`` where ``count=floor(10/(damping*dt_high))``.  The
    first sample at ``count*dt_high`` is therefore zero.  ``damping`` and
    ``dt_high`` are broadcastable positive real tensors; the integer result is
    intentionally a discrete quantity and has no useful derivative.
    """
    _check_real("damping", damping)
    _check_real("dt_high", dt_high, dtype=damping.dtype, device=damping.device)
    damping, dt_high = torch.broadcast_tensors(damping, dt_high)
    if bool((damping <= 0).any()) or bool((dt_high <= 0).any()):
        raise ValueError("damping and dt_high must be positive")
    return torch.floor(10 / (damping * dt_high)).to(torch.long)


def _unwrap_mode_phase(mode: Tensor) -> Tensor:
    """Unwrap the positive ``phi`` convention for ``h=A exp(-i phi)``."""
    first = -torch.angle(mode[..., :1])
    if mode.shape[-1] == 1:
        return first
    increment = -torch.angle(mode[..., 1:] * mode[..., :-1].conj())
    return torch.cat((first, first + increment.cumsum(dim=-1)), dim=-1)


def _eval_batch_scalar(spline: NaturalCubicSpline, coordinate: Tensor, derivative: int = 0) -> Tensor:
    """Evaluate one coordinate per spline batch row."""
    value = spline.evaluate(coordinate.unsqueeze(-1), derivative=derivative)
    return value[..., 0, :] if not spline.scalar_output else value[..., 0]


@dataclass(frozen=True)
class FullSeriesNQC:
    """Full-series splines and solved coefficients used by the attachment."""

    calibration: NQCCalibration
    amplitude_spline: NaturalCubicSpline
    product_spline: NaturalCubicSpline
    phase_basis_spline: NaturalCubicSpline
    frequency_spline: NaturalCubicSpline
    waveform_phase_spline: NaturalCubicSpline
    sampled_amplitude: Tensor
    sampled_products: Tensor
    sampled_phase_basis: Tensor
    sampled_waveform_phase: Tensor
    sampled_corrected_amplitude: Tensor
    sampled_corrected_phase: Tensor
    amplitude_matching_matrix: Tensor
    phase_matching_matrix: Tensor

    @property
    def coefficients(self):
        """Alias to the usual ``NQCCalibration.coefficients`` pair."""
        return self.calibration.coefficients

    def evaluate(self, query: Tensor):
        """Evaluate the corrected complex inspiral mode on the full spline."""
        # A query with exactly the spline batch shape denotes one coordinate
        # per batch row.  NaturalCubicSpline's ordinary vector convention
        # treats its final axis as a shared sample axis, so add an explicit
        # singleton sample axis for this unambiguous batched-scalar case.
        per_batch_scalar = query.ndim > 0 and query.shape == self.amplitude_spline.knots.shape[:-1]
        spline_query = query.unsqueeze(-1) if per_batch_scalar else query
        amplitude = self.amplitude_spline.evaluate(spline_query)
        products = self.product_spline.evaluate(spline_query)
        phase_basis = self.phase_basis_spline.evaluate(spline_query)
        phase = self.waveform_phase_spline.evaluate(spline_query)
        a, b = self.calibration.coefficients
        # For a scalar query, ``products`` and the coefficient vector have
        # the same rank.  Vector queries add a sample axis immediately before
        # the three basis columns; expand coefficients along that axis only
        # in the latter case so scalar/batched scalar calls stay scalar.
        a_view = a if products.ndim == a.ndim else a.unsqueeze(-2)
        b_view = b if phase_basis.ndim == b.ndim else b.unsqueeze(-2)
        corrected_amplitude = amplitude + (products * a_view).sum(-1)
        corrected_phase = phase - (phase_basis * b_view).sum(-1)
        result = corrected_amplitude * torch.exp(-1j * corrected_phase)
        return result[..., 0] if per_batch_scalar else result


def _full_series_nqc(
    time: Tensor,
    r: Tensor,
    pr_star: Tensor,
    orbital_frequency: Tensor,
    base_amplitude: Tensor,
    base_frequency: Tensor,
    waveform_phase: Tensor,
    match_time: Tensor,
    targets: NQCMatchingTargets,
    *,
    max_condition: float,
    diagnostic_amplitude_derivatives: Optional[Tensor] = None,
    diagnostic_phase_matrix: Optional[Tensor] = None,
    diagnostic_matching_second: Optional[Tensor] = None,
    diagnostic_matching_rhs_derivatives: Optional[Tensor] = None,
) -> FullSeriesNQC:
    q, p = nqc_22_basis(r, pr_star, orbital_frequency)
    products = base_amplitude.unsqueeze(-1) * q
    # Every sampled amplitude product and phase basis is passed independently
    # through a full natural spline.  The analytic frequency samples are kept
    # as a separate diagnostic spline below; they are deliberately absent from
    # the phase matching system.
    combined = torch.cat((base_amplitude.unsqueeze(-1), products, p), dim=-1)
    combined_spline = natural_cubic_spline(time, combined)
    value = _eval_batch_scalar(combined_spline, match_time)
    first = _eval_batch_scalar(combined_spline, match_time, derivative=1)
    second = _eval_batch_scalar(combined_spline, match_time, derivative=2)
    if diagnostic_matching_second is not None:
        if (not isinstance(diagnostic_matching_second, Tensor)
                or diagnostic_matching_second.shape != second[..., :5].shape
                or diagnostic_matching_second.dtype != second.dtype
                or diagnostic_matching_second.device != second.device
                or not bool(torch.isfinite(diagnostic_matching_second).all())):
            raise ValueError("diagnostic_matching_second must contain five finite second derivatives")
        second = torch.cat((diagnostic_matching_second[..., :4], second[..., 4:]), dim=-1)
    if bool((value[..., 0] <= 0).any()):
        raise ValueError("full-series base amplitude is nonpositive at match_time")
    amplitude_matrix = torch.stack((value[..., 1:4], first[..., 1:4],
                                    second[..., 1:4]), dim=-2)
    if diagnostic_amplitude_derivatives is not None:
        if (not isinstance(diagnostic_amplitude_derivatives, Tensor)
                or diagnostic_amplitude_derivatives.shape != amplitude_matrix[..., 1:, :].shape
                or diagnostic_amplitude_derivatives.dtype != amplitude_matrix.dtype
                or diagnostic_amplitude_derivatives.device != amplitude_matrix.device
                or not bool(torch.isfinite(diagnostic_amplitude_derivatives).all())):
            raise ValueError("diagnostic_amplitude_derivatives must contain two finite matching rows")
        amplitude_matrix = torch.cat((amplitude_matrix[..., :1, :],
                                      diagnostic_amplitude_derivatives), dim=-2)
    amplitude_rhs = torch.stack(targets[:3], dim=-1) - torch.stack(
        (value[..., 0], first[..., 0], second[..., 0]), dim=-1)
    if diagnostic_matching_rhs_derivatives is not None:
        if diagnostic_matching_second is not None:
            raise ValueError("diagnostic RHS derivatives conflict with matching-second control")
        if (not isinstance(diagnostic_matching_rhs_derivatives, Tensor)
                or diagnostic_matching_rhs_derivatives.shape != amplitude_matrix.shape[:-2] + (2, 2)
                or diagnostic_matching_rhs_derivatives.dtype != amplitude_matrix.dtype
                or diagnostic_matching_rhs_derivatives.device != amplitude_matrix.device
                or not bool(torch.isfinite(diagnostic_matching_rhs_derivatives).all())):
            raise ValueError("diagnostic_matching_rhs_derivatives must contain four finite derivatives")
        amplitude_rhs = torch.stack(targets[:3], dim=-1) - torch.stack((
            value[..., 0], diagnostic_matching_rhs_derivatives[..., 0, 0],
            diagnostic_matching_rhs_derivatives[..., 0, 1]), dim=-1)
    phase_matrix = torch.stack((first[..., 4:6], second[..., 4:6]), dim=-2)
    if diagnostic_phase_matrix is not None:
        if (not isinstance(diagnostic_phase_matrix, Tensor)
                or diagnostic_phase_matrix.shape != phase_matrix.shape
                or diagnostic_phase_matrix.dtype != phase_matrix.dtype
                or diagnostic_phase_matrix.device != phase_matrix.device
                or not bool(torch.isfinite(diagnostic_phase_matrix).all())):
            raise ValueError("diagnostic_phase_matrix must be a finite matching-matrix tensor")
        phase_matrix = diagnostic_phase_matrix
    # The phase RHS is based on the first and second derivatives of the full
    # unwrapped waveform-phase spline.  In the h=A*exp(-i*phi) convention
    # these are the positive GW frequency and its derivative.  A separately
    # splined analytic-frequency series is a different numerical operation and
    # must not stand in for this derivative pair.
    waveform_phase_spline = natural_cubic_spline(time, waveform_phase)
    phase_first = _eval_batch_scalar(waveform_phase_spline, match_time, derivative=1)
    phase_second = _eval_batch_scalar(waveform_phase_spline, match_time, derivative=2)
    if diagnostic_matching_second is not None:
        phase_second = diagnostic_matching_second[..., 4]
    phase_rhs = torch.stack((phase_first - targets.frequency,
                             phase_second - targets.frequency_derivative), dim=-1)
    if diagnostic_matching_rhs_derivatives is not None:
        phase_rhs = torch.stack((
            diagnostic_matching_rhs_derivatives[..., 1, 0] - targets.frequency,
            diagnostic_matching_rhs_derivatives[..., 1, 1]
            - targets.frequency_derivative), dim=-1)
    a, amplitude_condition, amplitude_residual = _matching_solve(
        amplitude_matrix, amplitude_rhs, max_condition, "full-series amplitude matching system")
    b, phase_condition, phase_residual = _matching_solve(
        phase_matrix, phase_rhs, max_condition, "full-series phase matching system")

    # Keep separately constructed splines for A, A*q_j, p_j, base frequency,
    # and unwrapped waveform phase.  This is intentionally not a product rule
    # applied to independently interpolated A and q.
    amplitude_spline = natural_cubic_spline(time, base_amplitude)
    product_spline = natural_cubic_spline(time, products)
    phase_basis_spline = natural_cubic_spline(time, p)
    frequency_spline = natural_cubic_spline(time, base_frequency)
    spacing = time[..., 1:] - time[..., :-1]
    sampling_condition = spacing.amax(-1) / spacing.amin(-1)
    calibration = NQCCalibration(NQCCoefficients(a, b), sampling_condition,
                                 amplitude_condition, phase_condition,
                                 amplitude_residual, phase_residual)
    corrected_amplitude = base_amplitude + (products*a.unsqueeze(-2)).sum(-1)
    corrected_phase = waveform_phase - (p*b.unsqueeze(-2)).sum(-1)
    # The signed amplitude multiplier can, in principle, cross zero.  The
    # ringdown boundary convention uses the argument of the corrected complex
    # waveform, so retain an unwrapped phase of that actual sampled mode
    # rather than assuming the multiplier is positive everywhere.
    corrected_series_mode = corrected_amplitude * torch.exp(-1j * corrected_phase)
    sampled_corrected_phase = _unwrap_mode_phase(corrected_series_mode)
    return FullSeriesNQC(
        calibration, amplitude_spline, product_spline, phase_basis_spline,
        frequency_spline, waveform_phase_spline, base_amplitude, products, p,
        waveform_phase, corrected_amplitude, sampled_corrected_phase,
        amplitude_matrix, phase_matrix,
    )


def calibrate_nqc_22_full_series(
    time: Tensor,
    r: Tensor,
    pr_star: Tensor,
    orbital_frequency: Tensor,
    amplitude: Tensor,
    gw_frequency: Tensor,
    match_time: Tensor,
    targets: Optional[NQCMatchingTargets] = None,
    *,
    waveform_phase: Optional[Tensor] = None,
    base_mode: Optional[Tensor] = None,
    nu: Optional[Tensor] = None,
    chi: Optional[Tensor] = None,
    max_condition: float = 1.0e10,
) -> FullSeriesNQC:
    """Solve v4 NQC conditions using full-series natural cubic splines.

    ``time,r,pr_star,orbital_frequency,amplitude,gw_frequency`` all use the
    last axis for samples.  ``amplitude`` is ``abs(h22)`` and ``gw_frequency``
    is ``-d arg(h22)/d tau``.  ``base_mode`` may be supplied to derive the
    continuously unwrapped waveform argument; alternatively callers can pass
    ``waveform_phase`` directly.  ``nu`` and ``chi`` are required only when
    ``targets`` is omitted.  The returned object exposes ``.calibration`` and
    ``.coefficients`` as well as each independent spline.

    The sampled products are formed first and splined as their own columns.
    In particular, derivatives are not obtained by differentiating a product
    of separately interpolated factors.
    """
    values = (time, r, pr_star, orbital_frequency, amplitude, gw_frequency, match_time)
    if not all(isinstance(value, Tensor) for value in values):
        raise TypeError("full-series NQC inputs must be tensors")
    _check_real("time", time)
    dtype, device = time.dtype, time.device
    for name, value in zip(("r", "pr_star", "orbital_frequency", "amplitude",
                            "gw_frequency", "match_time"), values[1:]):
        _check_real(name, value, dtype=dtype, device=device)
    if time.ndim < 1 or time.shape[-1] < 3:
        raise ValueError("full-series NQC needs at least three samples")
    n = time.shape[-1]
    if any(value.ndim < 1 or value.shape[-1] != n for value in values[:-1]):
        raise ValueError("full-series NQC sample arrays must share their last axis")
    if waveform_phase is not None and base_mode is not None:
        raise ValueError("provide waveform_phase or base_mode, not both")
    if base_mode is not None:
        if not isinstance(base_mode, Tensor):
            raise TypeError("base_mode must be a tensor")
        expected = torch.complex128 if dtype == torch.float64 else torch.complex64
        if base_mode.dtype != expected or base_mode.device != device:
            raise TypeError("base_mode must use time's corresponding complex dtype/device")
        if base_mode.ndim < 1 or base_mode.shape[-1] != n:
            raise ValueError("base_mode must share full-series NQC's sample axis")
        if not bool(torch.isfinite(base_mode).all()) or bool((base_mode.abs() <= 0).any()):
            raise ValueError("base_mode must be finite and have positive amplitude")
        waveform_phase = _unwrap_mode_phase(base_mode)
    if waveform_phase is None:
        raise ValueError("full-series NQC requires base_mode or waveform_phase")
    _check_real("waveform_phase", waveform_phase, dtype=dtype, device=device)
    if waveform_phase.shape[-1] != n:
        raise ValueError("waveform_phase must share the full-series sample axis")
    if not 1.0 <= max_condition < float("inf"):
        raise ValueError("max_condition must be finite and >= 1")

    samples = torch.broadcast_tensors(time, r, pr_star, orbital_frequency,
                                      amplitude, gw_frequency, waveform_phase)
    batch = torch.broadcast_shapes(*(sample.shape[:-1] for sample in samples),
                                   match_time.shape)
    time, r, pr_star, orbital_frequency, amplitude, gw_frequency, waveform_phase = (
        sample.expand(batch + (n,)) for sample in samples)
    match_time = match_time.expand(batch)
    if bool((time.diff(dim=-1) <= 0).any()):
        raise ValueError("full-series NQC times must be strictly increasing")
    if bool(((match_time < time[..., 0]) | (match_time > time[..., -1])).any()):
        raise ValueError("match_time must lie inside the full-series spline domain")
    if bool((r <= 0).any()) or bool((orbital_frequency <= 0).any()):
        raise ValueError("full-series NQC requires r and orbital_frequency positive")
    if bool((amplitude <= 0).any()) or bool((gw_frequency <= 0).any()):
        raise ValueError("full-series NQC requires positive amplitude and GW frequency")

    if targets is None:
        if nu is None or chi is None:
            raise ValueError("nu and chi are required when targets are omitted")
        nu = torch.as_tensor(nu, dtype=dtype, device=device)
        chi = torch.as_tensor(chi, dtype=dtype, device=device)
        _check_real("nu", nu, dtype=dtype, device=device)
        _check_real("chi", chi, dtype=dtype, device=device)
        targets = seobnrv4_nqc_targets(nu.expand(batch), chi.expand(batch))
    elif not isinstance(targets, NQCMatchingTargets):
        raise TypeError("targets must be NQCMatchingTargets")
    target_values = tuple(targets)
    if any(not isinstance(value, Tensor) for value in target_values):
        raise TypeError("NQC target fields must be tensors")
    targets = NQCMatchingTargets(*(value.to(dtype=dtype, device=device).expand(batch)
                                   for value in target_values))
    return _full_series_nqc(time, r, pr_star, orbital_frequency, amplitude,
                            gw_frequency, waveform_phase, match_time,
                            targets, max_condition=max_condition)


def _full_orbital_peak(time: Tensor, frequency: Tensor) -> OrbitalFrequencyPeak:
    """Apply the published full-spline/sign-product orbital-peak rule.

    The first non-increasing sample determines the adjacent interval.  The
    interval is on the following side when the spline derivative at that
    sample is positive and on the preceding side otherwise.  The update is
    intentionally a generic derivative-sign-product contraction: it is
    defined even when the endpoint derivatives have the same sign and may
    therefore return a sample boundary rather than a physical maximum.
    """
    spline = natural_cubic_spline(time, frequency)
    decreasing = frequency[..., 1:] <= frequency[..., :-1]
    if not bool(decreasing.any(-1).all()):
        raise ValueError("orbital-frequency peak is unresolved: no non-increasing sample")
    index = decreasing.to(torch.int64).argmax(-1) + 1
    sample_time = torch.gather(time, -1, index.unsqueeze(-1)).squeeze(-1)
    sample_slope = _eval_batch_scalar(spline, sample_time, derivative=1)
    following = sample_slope > 0
    if bool((following & (index + 1 >= time.shape[-1])).any()):
        raise ValueError("orbital-frequency peak has no following interval")
    left_index = torch.where(following, index, index - 1)
    right_index = torch.where(following, index + 1, index)
    left = torch.gather(time, -1, left_index.unsqueeze(-1)).squeeze(-1)
    right = torch.gather(time, -1, right_index.unsqueeze(-1)).squeeze(-1)
    left_slope = _eval_batch_scalar(spline, left, derivative=1)
    # The tolerance and sign-product update are part of the numerical
    # convention; no negative-curvature or endpoint sign check is added.
    tolerance = torch.as_tensor(1.0e-5, dtype=time.dtype, device=time.device)
    active = (right - left) > tolerance
    middle = (left + right) / 2
    middle_slope = _eval_batch_scalar(spline, middle, derivative=1)
    last_middle = middle
    while bool(active.any()):
        # Only a strictly negative product moves the right endpoint.  Equality
        # (including an exactly zero endpoint or midpoint derivative) belongs
        # to the left move by the published numerical convention.
        move_right = left_slope * middle_slope < 0
        move_left = ~move_right
        left = torch.where(active & move_left, middle, left)
        right = torch.where(active & move_right, middle, right)
        left_slope = torch.where(active & move_left, middle_slope, left_slope)
        last_middle = torch.where(active, middle, last_middle)
        active = (right - left) > tolerance
        if bool(active.any()):
            middle = (left + right) / 2
            middle_slope = _eval_batch_scalar(spline, middle, derivative=1)
    peak_time = last_middle
    peak_frequency = _eval_batch_scalar(spline, peak_time)
    curvature = _eval_batch_scalar(spline, peak_time, derivative=2)
    return OrbitalFrequencyPeak(peak_time, peak_frequency, curvature, index)


def _peak_record_at_time(time: Tensor, frequency: Tensor, coordinate: Tensor) -> OrbitalFrequencyPeak:
    spline = natural_cubic_spline(time, frequency)
    # Event sample indices use the same floor convention as the attachment:
    # an exact knot belongs to that knot, while an in-between coordinate uses
    # the greatest earlier grid sample.
    index = (torch.searchsorted(time.contiguous(), coordinate.unsqueeze(-1).contiguous(), right=True)
             .squeeze(-1) - 1).clamp(min=0, max=time.shape[-1] - 1)
    return OrbitalFrequencyPeak(coordinate, _eval_batch_scalar(spline, coordinate),
                                _eval_batch_scalar(spline, coordinate, derivative=2), index)


class NumericalAttachedMode22(tuple):
    """Tuple-like result from :func:`attach_seobnrv4_22_numerical_compatibility`.

    The first six fields mirror :class:`attachment.AttachedMode22`; the
    remaining fields make the discrete boundary and geometric-step choices
    inspectable without parsing a diagnostic log.
    """

    __slots__ = ()
    _fields = (
        "mode", "corrected_inspiral", "peak", "match_time", "match_phase",
        "calibration", "boundary_time", "boundary_index", "boundary_phase",
        "boundary_frequency", "boundary_amplitude", "internal_step", "nqc",
    )

    def __new__(cls, mode, corrected_inspiral, peak, match_time, match_phase,
                calibration, boundary_time, boundary_index, boundary_phase,
                boundary_frequency, boundary_amplitude, internal_step, nqc):
        return tuple.__new__(cls, (mode, corrected_inspiral, peak, match_time,
                                   match_phase, calibration, boundary_time,
                                   boundary_index, boundary_phase,
                                   boundary_frequency, boundary_amplitude,
                                   internal_step, nqc))

    def __getnewargs__(self):
        return tuple(self)

    @property
    def mode(self): return self[0]
    @property
    def corrected_inspiral(self): return self[1]
    @property
    def peak(self): return self[2]
    @property
    def match_time(self): return self[3]
    @property
    def match_phase(self): return self[4]
    @property
    def calibration(self): return self[5]
    @property
    def boundary_time(self): return self[6]
    @property
    def boundary_index(self): return self[7]
    @property
    def boundary_phase(self): return self[8]
    @property
    def boundary_frequency(self): return self[9]
    @property
    def boundary_amplitude(self): return self[10]
    @property
    def internal_step(self): return self[11]
    @property
    def nqc(self): return self[12]


def attach_seobnrv4_22_numerical_compatibility(
    time: Tensor,
    state: Tensor,
    orbital_frequency: Tensor,
    base_mode: Tensor,
    base_frequency: Tensor,
    nu: Tensor,
    chi: Tensor,
    sigma: Tensor,
    output_time: Tensor,
    *,
    valid: Optional[Tensor] = None,
    match_time: Optional[Tensor] = None,
    mass_scale: Tensor | float = 1.0,
    max_condition: float = 1.0e10,
    diagnostic_amplitude_derivatives: Optional[Tensor] = None,
    diagnostic_phase_matrix: Optional[Tensor] = None,
    diagnostic_matching_second: Optional[Tensor] = None,
    diagnostic_matching_rhs_derivatives: Optional[Tensor] = None,
) -> NumericalAttachedMode22:
    """Attach the full-series NQC and discrete-boundary v4 ringdown.

    ``time`` is the high-resolution input grid and all its sample arrays end
    in that grid.  ``output_time`` is an increasing output chart in the same
    ``tau=t/M`` origin; arbitrary output queries are accepted.  If
    ``match_time`` is omitted, the continuous orbital peak is found from a
    natural spline and the printed v4 NQC offset is
    applied.  Supplying it is useful when an upstream event detector owns the
    published matching-time choice; no amplitude-maximum search is performed
    in either case.

    The returned ``boundary_time`` is the greatest input grid sample no
    later than ``match_time`` (an exact-knot match therefore uses that knot).
    Ringdown uses the corrected sampled amplitude and phase there,
    sets ``A'=0``, and obtains ``boundary_frequency`` from the corrected
    backward phase difference.  ``internal_step`` is the power-of-two
    geometric step required for the internal ringdown grid; the analytic
    ansatz is evaluated at output samples here.  Ringdown support is counted
    on that high-resolution grid before values are selected at output queries.
    """
    if not isinstance(state, Tensor) or state.ndim < 2 or state.shape[-1] != 4:
        raise ValueError("state must end in (r, phi, p_r*, p_phi)")
    _check_real("time", time)
    dtype, device = time.dtype, time.device
    for name, value in (("orbital_frequency", orbital_frequency), ("base_frequency", base_frequency),
                        ("nu", nu), ("chi", chi), ("output_time", output_time)):
        _check_real(name, value, dtype=dtype, device=device)
    if state.dtype != dtype or state.device != device:
        raise TypeError("state must share time's floating dtype and device")
    expected_complex = torch.complex128 if dtype == torch.float64 else torch.complex64
    if not isinstance(base_mode, Tensor) or base_mode.dtype != expected_complex or base_mode.device != device:
        raise TypeError("base_mode must use time's corresponding complex dtype/device")
    if not isinstance(sigma, Tensor) or sigma.dtype != expected_complex or sigma.device != device:
        raise TypeError("sigma must use time's corresponding complex dtype/device")
    if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(base_mode).all()):
        raise ValueError("state and base_mode must be finite")
    if not bool(torch.isfinite(sigma).all()):
        raise ValueError("sigma must be finite")
    if time.ndim < 1 or time.shape[-1] < 4 or output_time.ndim < 1 or output_time.shape[-1] < 2:
        raise ValueError("attachment grids need at least four input and two output samples")
    n = time.shape[-1]
    sample_values = (time, orbital_frequency, base_mode, base_frequency)
    if any(value.shape[-1] != n for value in sample_values) or state.shape[-2] != n:
        raise ValueError("attachment sample arrays must share time's last axis")
    batch = torch.broadcast_shapes(time.shape[:-1], state.shape[:-2],
                                   orbital_frequency.shape[:-1], base_mode.shape[:-1],
                                   base_frequency.shape[:-1], nu.shape, chi.shape,
                                   sigma.shape, output_time.shape[:-1])
    time, orbital_frequency, base_mode, base_frequency = (
        value.expand(batch + (n,)) for value in sample_values)
    state = state.expand(batch + (n, 4))
    nu, chi, sigma = (value.expand(batch) for value in (nu, chi, sigma))
    query = output_time.expand(batch + (output_time.shape[-1],))
    if valid is None:
        valid = torch.ones_like(time, dtype=torch.bool)
    elif not isinstance(valid, Tensor) or valid.dtype != torch.bool or valid.device != device:
        raise TypeError("valid must be a boolean tensor on time's device")
    valid = valid.expand(batch + (n,))
    if not bool(valid[..., 0].all()) or bool((valid[..., 1:] & ~valid[..., :-1]).any()):
        raise ValueError("valid samples must form contiguous nonempty prefixes")
    count = valid.sum(-1)
    if (count.numel() > 1 and bool((count != count.reshape(-1)[0]).any())) or bool((count < 4).any()):
        raise ValueError("all attachment rows must have the same at-least-four-sample valid prefix")
    n_valid = int(count.reshape(-1)[0].detach())
    time, orbital_frequency, base_mode, base_frequency, state = (
        value[..., :n_valid] if value is not state else value[..., :n_valid, :]
        for value in (time, orbital_frequency, base_mode, base_frequency, state)
    )
    valid = valid[..., :n_valid]
    if bool((time.diff(dim=-1) <= 0).any()):
        raise ValueError("attachment input times must be strictly increasing")
    if bool((query.diff(dim=-1) <= 0).any()):
        raise ValueError("attachment output times must be strictly increasing")
    dt_output = query[..., 1] - query[..., 0]
    # Output queries may be downsampled or otherwise irregular.  The first
    # interval is retained as metadata for the documented internal-step rule;
    # ringdown support itself is determined on that internal high-resolution
    # grid before selecting output samples.
    internal_step = seobnrv4_internal_step(dt_output, mass_scale)
    dt_high = internal_step

    if match_time is None:
        peak = _full_orbital_peak(time, orbital_frequency)
        match = peak.time + seobnrv4_nqc_time_offset_numerical_compatibility(nu, chi)
    else:
        match = torch.as_tensor(match_time, dtype=dtype, device=device).expand(batch)
        if not bool(torch.isfinite(match).all()):
            raise ValueError("match_time must be finite")
        peak = _peak_record_at_time(time, orbital_frequency,
                                    match - seobnrv4_nqc_time_offset_numerical_compatibility(nu, chi))
    if bool(((match <= time[..., 0]) | (match >= time[..., -1])).any()):
        raise ValueError("continuous match_time must lie inside the full input grid")
    # ``right=True`` gives the floor in the grid chart: an exact match knot is
    # the boundary itself, while an in-between match selects its predecessor.
    # Keep the floor intact.  A floor of zero has no backward phase sample and
    # is invalid; advancing it to index one changes the boundary convention.
    boundary_index = (torch.searchsorted(time.contiguous(), match.unsqueeze(-1).contiguous(), right=True)
                      .squeeze(-1) - 1)
    if bool((boundary_index < 1).any()):
        raise ValueError("attachment floor sample has no preceding phase sample")
    boundary_time = torch.gather(time, -1, boundary_index.unsqueeze(-1)).squeeze(-1)
    step = torch.gather(time, -1, boundary_index.unsqueeze(-1)).squeeze(-1) - torch.gather(
        time, -1, (boundary_index - 1).unsqueeze(-1)).squeeze(-1)

    amplitude = base_mode.abs()
    frequency = base_frequency
    targets = seobnrv4_nqc_targets(nu, chi)
    nqc = _full_series_nqc(
        time, state[..., 0], state[..., 2], orbital_frequency, amplitude,
        frequency, _unwrap_mode_phase(base_mode), match, targets,
        max_condition=max_condition,
        diagnostic_amplitude_derivatives=diagnostic_amplitude_derivatives,
        diagnostic_phase_matrix=diagnostic_phase_matrix,
        diagnostic_matching_second=diagnostic_matching_second,
        diagnostic_matching_rhs_derivatives=diagnostic_matching_rhs_derivatives,
    )
    coefficients = nqc.calibration.coefficients
    corrected = base_mode * nqc_22_factor(
        state[..., 0], state[..., 2], orbital_frequency, coefficients, sample_dims=1)
    boundary_mode = torch.gather(corrected, -1, boundary_index.unsqueeze(-1)).squeeze(-1)
    boundary_amplitude = boundary_mode.abs()
    # Keep the phase origin on the continuously unwrapped corrected series.
    # Taking ``angle`` of one complex sample would reset the ringdown phase at
    # every branch cut.  The one-step backward difference below is the
    # positive-frequency h=A*exp(-i*phi) convention in the same unwrapped chart.
    corrected_phase = nqc.sampled_corrected_phase
    boundary_phase = torch.gather(
        corrected_phase, -1, boundary_index.unsqueeze(-1)
    ).squeeze(-1)
    previous_phase = torch.gather(
        corrected_phase, -1, (boundary_index - 1).unsqueeze(-1)
    ).squeeze(-1)
    boundary_frequency = (boundary_phase - previous_phase) / step
    if bool((boundary_amplitude <= 0).any()) or bool((boundary_frequency <= 0).any()):
        raise ValueError("corrected boundary must have positive amplitude and backward frequency")
    match_phase = _eval_batch_scalar(
        nqc.waveform_phase_spline, match) - (
            _eval_batch_scalar(nqc.phase_basis_spline, match)
            * coefficients.phase
        ).sum(-1)
    ringdown_time = (query - boundary_time.unsqueeze(-1)).clamp_min(0)
    ringdown = merger_ringdown_22(
        ringdown_time, sigma, boundary_amplitude, torch.zeros_like(boundary_amplitude),
        boundary_frequency, boundary_phase,
        seobnrv4_peak_preserving_shape(sigma, seobnrv4_merger_shape(nu, chi)),
    )
    # Determine support on the high-resolution grid before selecting output
    # queries.  Samples j=0,...,count-1 are retained; elapsed count*dt_high
    # is the first zero-padded sample, hence the strict comparison.
    damping = -sigma.imag
    support_count = seobnrv4_ringdown_support_samples(damping, dt_high)
    support_end = support_count.to(dtype=dtype) * dt_high
    ringdown = torch.where(
        ringdown_time < support_end.unsqueeze(-1), ringdown, torch.zeros_like(ringdown)
    )
    spline_query = query.clamp(min=time[..., :1], max=boundary_time.unsqueeze(-1))
    inspiral = nqc.evaluate(spline_query)
    inspiral = torch.where(query < time[..., :1], torch.zeros_like(inspiral), inspiral)
    mode = torch.where(query < boundary_time.unsqueeze(-1), inspiral, ringdown)
    return NumericalAttachedMode22(
        mode, corrected, peak, match, match_phase, nqc.calibration,
        boundary_time, boundary_index, boundary_phase, boundary_frequency,
        boundary_amplitude, internal_step, nqc,
    )


# Short aliases make the opt-in nature visible while accommodating callers
# that refer to the contract as either "compatibility" or "numerical".
attach_seobnrv4_22_compatibility = attach_seobnrv4_22_numerical_compatibility
attach_seobnrv4_22_numerical = attach_seobnrv4_22_numerical_compatibility


__all__ = [
    "NaturalCubicSpline", "natural_cubic_spline", "natural_cubic_coefficients",
    "FullSeriesNQC", "NumericalAttachedMode22",
    "calibrate_nqc_22_full_series", "seobnrv4_internal_step",
    "seobnrv4_ringdown_support_samples",
    "attach_seobnrv4_22_numerical_compatibility",
    "attach_seobnrv4_22_compatibility", "attach_seobnrv4_22_numerical",
]

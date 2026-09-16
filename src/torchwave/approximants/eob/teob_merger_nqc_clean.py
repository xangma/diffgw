"""Clean-room TEOB auxiliary-grid and explicit-target NQC assembly.

This module owns only the numerical NQC assembly seam described in the
merger contract.  It accepts an already sampled, scalar nonspinning
trajectory and a caller-owned set of four matching targets.  The target
values are never inferred from a mass-ratio fit or from a global provider.

The input waveform convention is ``h22 = A exp(-i phase)``.  The auxiliary
grid is built from the accepted samples before the final accepted knot, and
all sampled quantities are interpolated independently with natural cubic
splines. By default the local NQC systems use a degree-four interpolating
derivative on five uniform-grid knots. The opt-in continuous policy uses
accepted-series spline derivatives at a continuous physical event instead.
The implementation is Torch-native so derivatives
through sampled data, targets, and the two solves remain available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, NamedTuple, Optional, Sequence

import torch
from torch import Tensor

from .numerical_attachment import natural_cubic_spline
from .continuous_matching import continuous_spline_peak
from .teob_nqc_clean import (
    NQCCoefficients,
    NQCMatchResult,
    NQCMatchingTargets,
    nqc_basis_values,
)


class TEOBMergerNQCError(ValueError):
    """Raised when the audited auxiliary-grid NQC seam is unsupported."""


class TEOBMergerNQCTargets(NamedTuple):
    """Explicit NQC targets in the normalized amplitude/time convention.

    ``amplitude`` and ``amplitude_derivative`` refer to the RW-normalized
    amplitude ``A/(nu*sqrt(24))`` used by the matching equations.  Frequency
    targets use the phase convention of ``h=A exp(-i phase)``.  No provenance
    is stored here because it is a caller assertion attached to the complete
    result; this record remains a compact four-array boundary.
    """

    amplitude: Tensor
    amplitude_derivative: Tensor
    frequency: Tensor
    frequency_derivative: Tensor

    @property
    def amplitude_rate(self) -> Tensor:
        return self.amplitude_derivative

    @property
    def frequency_rate(self) -> Tensor:
        return self.frequency_derivative

    @property
    def A(self) -> Tensor:
        return self.amplitude

    @property
    def Adot(self) -> Tensor:
        return self.amplitude_derivative

    @property
    def omega(self) -> Tensor:
        return self.frequency

    @property
    def omegadot(self) -> Tensor:
        return self.frequency_derivative


# This label is intentionally separate from the 2018 global provider in
# ``teob_postmerger.py``.  It is suitable as an explicit provenance string
# when a caller elects to use the formulas below; the assembler never selects
# it implicitly.
PUBLISHED_2017_NQC_PROVENANCE = (
    "Nagar, Riemenschneider & Pratten (2017), arXiv:1703.06814v1, "
    "Appendix A Table II (q=1) and Table IV + Eq. (A3) (q>1); "
    "A_RW=A/(nu*sqrt(24))=|Psi_22|/nu"
)


@dataclass(frozen=True)
class TEOBMergerNQCSeries:
    """Named sampled arrays on one of the two NQC grids.

    The ``basis_amplitude`` and ``basis_phase`` fields are the sampled
    ``q`` and ``p`` arrays from the clean NQC equations.  The correction is
    applied to each grid from that grid's own basis, rather than evaluating a
    basis on an interpolated corrected series.
    """

    time: Tensor
    state: Tensor
    amplitude: Tensor
    phase: Tensor
    orbital_frequency: Tensor
    radial_acceleration: Tensor
    basis_amplitude: Tensor
    basis_phase: Tensor
    corrected_amplitude: Tensor
    corrected_phase: Tensor

    @property
    def q(self) -> Tensor:
        return self.basis_amplitude

    @property
    def p(self) -> Tensor:
        return self.basis_phase

    @property
    def omega(self) -> Tensor:
        return self.orbital_frequency

    @property
    def radial_acceleration_conservative(self) -> Tensor:
        return self.radial_acceleration

    @property
    def mode(self) -> Tensor:
        """Reconstruct the corrected complex mode in the signed chart."""

        imaginary = torch.zeros_like(self.corrected_phase)
        return torch.complex(self.corrected_amplitude, imaginary) * torch.exp(
            torch.complex(imaginary, -self.corrected_phase)
        )


@dataclass(frozen=True)
class TEOBMergerNQCResult:
    """Auxiliary-grid NQC result with a compact named-array interface.

    ``original`` retains every caller sample, including the final accepted
    knot.  ``auxiliary`` starts at the contract cut and excludes that final
    knot from its spline support.  Downstream assembly can therefore consume
    ``result.original`` and ``result.auxiliary`` without guessing which grid
    a basis or correction belongs to.
    """

    original: TEOBMergerNQCSeries
    auxiliary: TEOBMergerNQCSeries
    coefficients: NQCCoefficients
    match: NQCMatchResult
    targets: TEOBMergerNQCTargets
    target_provenance: str
    cut_start_index: int
    cut_last_index: int
    auxiliary_spacing: Tensor
    t_peak_grid: Tensor
    t_nqc: Tensor
    matching_index: int
    amplitude_rw: Tensor
    auxiliary_amplitude_rw: Tensor
    auxiliary_negative_amplitude: Tensor
    matching_method: str = "grid"
    continuous_peak_time: Optional[Tensor] = None

    @property
    def t_peak(self) -> Tensor:
        """Physical event used by the selected matching policy."""
        return self.t_peak_grid if self.continuous_peak_time is None else self.continuous_peak_time

    @property
    def time(self) -> Tensor:
        return self.original.time

    @property
    def state(self) -> Tensor:
        return self.original.state

    @property
    def amplitude(self) -> Tensor:
        return self.original.amplitude

    @property
    def phase(self) -> Tensor:
        return self.original.phase

    @property
    def orbital_frequency(self) -> Tensor:
        return self.original.orbital_frequency

    @property
    def radial_acceleration(self) -> Tensor:
        return self.original.radial_acceleration

    @property
    def corrected_amplitude(self) -> Tensor:
        return self.original.corrected_amplitude

    @property
    def corrected_phase(self) -> Tensor:
        return self.original.corrected_phase

    @property
    def q(self) -> Tensor:
        return self.original.basis_amplitude

    @property
    def p(self) -> Tensor:
        return self.original.basis_phase

    @property
    def aux_time(self) -> Tensor:
        return self.auxiliary.time

    @property
    def aux_state(self) -> Tensor:
        return self.auxiliary.state

    @property
    def aux_amplitude(self) -> Tensor:
        return self.auxiliary.amplitude

    @property
    def aux_phase(self) -> Tensor:
        return self.auxiliary.phase

    @property
    def aux_orbital_frequency(self) -> Tensor:
        return self.auxiliary.orbital_frequency

    @property
    def aux_radial_acceleration(self) -> Tensor:
        return self.auxiliary.radial_acceleration

    @property
    def aux_corrected_amplitude(self) -> Tensor:
        return self.auxiliary.corrected_amplitude

    @property
    def aux_corrected_phase(self) -> Tensor:
        return self.auxiliary.corrected_phase

    @property
    def aux_amplitude_rw(self) -> Tensor:
        """RW-normalized amplitude on the auxiliary grid."""

        return self.auxiliary_amplitude_rw

    @property
    def aux_q(self) -> Tensor:
        return self.auxiliary.basis_amplitude

    @property
    def aux_p(self) -> Tensor:
        return self.auxiliary.basis_phase

    @property
    def amplitude_matrix(self) -> Tensor:
        return self.match.amplitude_matrix

    @property
    def phase_matrix(self) -> Tensor:
        return self.match.phase_matrix

    @property
    def amplitude_condition(self) -> Tensor:
        return self.match.amplitude_condition

    @property
    def phase_condition(self) -> Tensor:
        return self.match.phase_condition

    @property
    def amplitude_residual(self) -> Tensor:
        return self.match.amplitude_residual

    @property
    def phase_residual(self) -> Tensor:
        return self.match.phase_residual

    @property
    def negative_auxiliary_amplitude(self) -> Tensor:
        """Boolean mask exposing signed auxiliary corrected amplitudes."""

        return self.auxiliary_negative_amplitude

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "target_provenance": self.target_provenance,
            "cut_start_index": self.cut_start_index,
            "cut_last_index": self.cut_last_index,
            "auxiliary_spacing": self.auxiliary_spacing,
            "t_peak_grid": self.t_peak_grid,
            "t_nqc": self.t_nqc,
            "matching_index": self.matching_index,
            "amplitude_condition": self.amplitude_condition,
            "phase_condition": self.phase_condition,
            "auxiliary_negative_amplitude": self.auxiliary_negative_amplitude,
        }


def _as_real(value: object, *, name: str, dtype: torch.dtype,
             device: torch.device, scalar: bool = False) -> Tensor:
    if isinstance(value, Tensor):
        result = value
        if result.dtype != dtype or result.device != device:
            raise TypeError(f"{name} must share time's floating dtype and device")
    else:
        result = torch.as_tensor(value, dtype=dtype, device=device)
    if result.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    if scalar and result.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result


def published_2017_nqc_targets(
    nu: Tensor | float,
    effective_spin: Tensor | float,
) -> TEOBMergerNQCTargets:
    """Return the explicitly named 2017 NQC target fit.

    The fit is the primary-source family in Nagar, Riemenschneider and
    Pratten, arXiv:1703.06814v1, Appendix A.  Equal mass uses Table II's
    fourth-order polynomials.  The asymmetric branch uses Table IV and its
    rational frequency in Eq. (A3).  Its supported domain is
    ``0.16 < nu <= 0.25`` (``1 <= q < 4``); values at ``nu=0.25`` select the
    Table II branch.  The returned amplitude and amplitude derivative are
    already in the assembler's ``A_RW=|Psi_22|/nu`` convention.

    ``effective_spin`` is required even for the zero-spin use case so the
    source choice remains explicit.  Tensor inputs may carry matching
    leading shapes; no target family or provenance is inferred by the
    assembler from this helper.
    """

    # Infer a common floating dtype/device from an explicit tensor argument.
    # Python-only calls use the current Torch default, while preserving the
    # 8-decimal published coefficients in float64 when the caller requests it.
    tensor_args = [value for value in (nu, effective_spin) if isinstance(value, Tensor)]
    if tensor_args:
        first = tensor_args[0]
        if first.dtype not in (torch.float32, torch.float64):
            raise TypeError("nu and effective_spin must use float32 or float64")
        dtype, device = first.dtype, first.device
        if any(value.dtype != dtype or value.device != device for value in tensor_args):
            raise TypeError("nu and effective_spin tensors must share dtype and device")
    else:
        dtype, device = torch.get_default_dtype(), torch.device("cpu")
    nu_t = _as_real(nu, name="nu", dtype=dtype, device=device)
    spin_t = _as_real(effective_spin, name="effective_spin", dtype=dtype, device=device)
    try:
        nu_t, spin_t = torch.broadcast_tensors(nu_t, spin_t)
    except RuntimeError as exc:
        raise ValueError("nu and effective_spin must be broadcast-compatible") from exc
    if bool((nu_t <= 0.16).any()) or bool((nu_t > 0.25).any()):
        raise ValueError("published 2017 NQC targets require 0.16 < nu <= 0.25")
    if bool((spin_t.abs() > 1.0).any()):
        raise ValueError("effective_spin must satisfy |effective_spin| <= 1")

    def coeff(value: float) -> Tensor:
        return torch.as_tensor(value, dtype=dtype, device=device)

    # Table II, q=1.  Horner evaluation keeps the printed decimal constants
    # visible and gives a compact differentiable expression.
    a0 = spin_t
    table_ii_amplitude = (((coeff(0.00178195) * a0 + coeff(0.00435589)) * a0
                           + coeff(0.00344489)) * a0 - coeff(0.00076165)) * a0 + coeff(0.31973334)
    table_ii_amplitude_rate = (((coeff(0.00000927) * a0 - coeff(0.00024550)) * a0
                                + coeff(0.00012469)) * a0 + coeff(0.00123845)) * a0 - coeff(0.00195014)
    table_ii_frequency = (((coeff(0.00603482) * a0 + coeff(0.01604555)) * a0
                           + coeff(0.02290799)) * a0 + coeff(0.07084587)) * a0 + coeff(0.38321834)
    table_ii_frequency_rate = (((coeff(0.00024066) * a0 + coeff(0.00038123)) * a0
                                - coeff(0.00049714)) * a0 + coeff(0.00041219)) * a0 + coeff(0.01190548)

    # Table IV, q>1, and Eq. (A3) for the frequency itself.
    table_iv_amplitude = (
        (coeff(0.04680896) * nu_t - coeff(0.00632114)) * a0.pow(3)
        + (coeff(0.06586192) * nu_t - coeff(0.01180039)) * a0.square()
        + (-coeff(0.11617413) * nu_t + coeff(0.02704959)) * a0
        + coeff(0.15597465) * nu_t + coeff(0.28034978)
    )
    table_iv_amplitude_rate = (
        (-coeff(0.00130824) * nu_t + coeff(0.00006202)) * a0.pow(3)
        + (coeff(0.00199855) * nu_t - coeff(0.00027474)) * a0.square()
        + (coeff(0.00218838) * nu_t + coeff(0.00071540)) * a0
        - coeff(0.00362779) * nu_t - coeff(0.00105397)
    )
    frequency_denominator = 1.0 + (coeff(0.64131115) * nu_t - coeff(0.37878384)) * a0
    if bool((frequency_denominator == 0).any()):
        raise ValueError("published 2017 Eq. (A3) frequency denominator is singular")
    table_iv_frequency = (
        coeff(0.46908067) * nu_t + coeff(0.27022141)
    ) / frequency_denominator
    table_iv_frequency_rate = (
        (coeff(0.00061175) * nu_t + coeff(0.00074001)) * a0
        + coeff(0.02504442) * nu_t + coeff(0.00548217)
    )

    equal_mass = nu_t == coeff(0.25)
    return TEOBMergerNQCTargets(
        torch.where(equal_mass, table_ii_amplitude, table_iv_amplitude),
        torch.where(equal_mass, table_ii_amplitude_rate, table_iv_amplitude_rate),
        torch.where(equal_mass, table_ii_frequency, table_iv_frequency),
        torch.where(equal_mass, table_ii_frequency_rate, table_iv_frequency_rate),
    )


# Explicit aliases retain the source identity in introspection and call sites.
nqc_targets_2017 = published_2017_nqc_targets
nagar_2017_nqc_targets = published_2017_nqc_targets


def _check_primary_series(time: Tensor, state: Tensor, arrays: Sequence[tuple[str, Tensor]]) -> None:
    if not isinstance(time, Tensor) or time.dtype not in (torch.float32, torch.float64):
        raise TypeError("time must be a float32/float64 tensor")
    if time.ndim != 1 or time.numel() < 5:
        raise ValueError("merger NQC time needs at least five one-dimensional samples")
    if not bool(torch.isfinite(time).all()) or bool((time.diff() <= 0).any()):
        raise ValueError("merger NQC time must be finite and strictly increasing")
    if not isinstance(state, Tensor) or state.ndim != 2 or state.shape != (time.numel(), 4):
        raise ValueError("state must have shape (samples, 4) for (r, phi, p_r*, p_phi)")
    if state.dtype != time.dtype or state.device != time.device:
        raise TypeError("state must share time's floating dtype and device")
    if not bool(torch.isfinite(state).all()):
        raise ValueError("state must be finite")
    for name, value in arrays:
        if not isinstance(value, Tensor) or value.ndim != 1 or value.shape != time.shape:
            raise ValueError(f"{name} must be a one-dimensional series matching time")
        if value.dtype != time.dtype or value.device != time.device:
            raise TypeError(f"{name} must share time's floating dtype and device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")


def _quartic_weight_table(*, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Derive first-derivative weights for all five-node offsets.

    The five node coordinates are ``(-2,-1,0,1,2)`` in units of the local
    spacing.  Solving the Vandermonde interpolation equations for the
    derivative rows gives weights for evaluation offsets ``-2..2``.  This is
    a small fixed linear algebra derivation rather than a fitted stencil.
    """

    nodes = torch.arange(-2, 3, dtype=dtype, device=device)
    powers = torch.arange(5, dtype=dtype, device=device)
    vandermonde = nodes.unsqueeze(0).pow(powers.unsqueeze(-1))
    evaluation_offsets = nodes
    derivative_rows = torch.zeros((5, 5), dtype=dtype, device=device)
    for row, offset in enumerate(evaluation_offsets):
        derivative_rows[row, 1:] = powers[1:] * offset.pow(powers[1:] - 1)
    # ``vandermonde`` is arranged as powers-by-nodes, i.e. it is the
    # transpose of the usual node-by-powers interpolation matrix.  If ``c``
    # solves ``vandermonde.T @ c=f``, a derivative row is ``drow @ c`` and
    # the nodal weights therefore solve ``vandermonde @ w=drow``.
    return torch.linalg.solve(vandermonde, derivative_rows.transpose(-2, -1)).transpose(-2, -1)


def quartic_derivative(values: Tensor, spacing: Tensor | float, *, sample_dim: int = -1) -> Tensor:
    """Apply the degree-four five-knot derivative on a uniform grid.

    Interior rows use centered knots.  The first two rows use knots 0..4 and
    the last two use the final five knots, which is the contract's nearest
    five-node endpoint rule.  ``sample_dim`` selects the sample axis and all
    other axes are preserved.
    """

    if not isinstance(values, Tensor) or values.dtype not in (torch.float32, torch.float64):
        raise TypeError("values must be a float32/float64 tensor")
    if values.ndim < 1:
        raise ValueError("values must have a sample axis")
    if isinstance(sample_dim, bool) or not isinstance(sample_dim, int):
        raise TypeError("sample_dim must be an integer")
    dim = sample_dim % values.ndim
    moved = values.movedim(dim, -1)
    n = moved.shape[-1]
    if n < 5:
        raise ValueError("quartic derivative needs at least five samples")
    step = _as_real(spacing, name="spacing", dtype=values.dtype, device=values.device, scalar=True)
    if bool(step <= 0):
        raise ValueError("spacing must be finite and positive")

    centers = torch.arange(n, dtype=torch.long, device=values.device)
    center = centers.clamp(min=2, max=n - 3)
    indices = center.unsqueeze(-1) + torch.arange(-2, 3, dtype=torch.long, device=values.device)
    windows = moved[..., indices]
    relative = centers - center
    weights = _quartic_weight_table(dtype=values.dtype, device=values.device)[relative + 2]
    result = (windows * weights).sum(dim=-1) / step
    return result.movedim(-1, dim)


# Equation-friendly names used by diagnostics and downstream reviewers.
degree_four_derivative = quartic_derivative
five_point_quartic_derivative = quartic_derivative
D = quartic_derivative


def matching_index(auxiliary_time: Tensor, t_nqc: Tensor | float) -> int:
    """Return ``searchsorted(aux_time, t_nqc, right=True)-2`` exactly."""

    if not isinstance(auxiliary_time, Tensor) or auxiliary_time.ndim != 1:
        raise ValueError("auxiliary_time must be one-dimensional")
    if auxiliary_time.numel() < 5:
        raise ValueError("auxiliary_time needs at least five samples")
    if auxiliary_time.dtype not in (torch.float32, torch.float64):
        raise TypeError("auxiliary_time must use float32/float64")
    if not bool(torch.isfinite(auxiliary_time).all()) or bool((auxiliary_time.diff() <= 0).any()):
        raise ValueError("auxiliary_time must be finite and strictly increasing")
    coordinate = _as_real(t_nqc, name="t_nqc", dtype=auxiliary_time.dtype,
                          device=auxiliary_time.device, scalar=True)
    if bool((coordinate < auxiliary_time[0]) | (coordinate > auxiliary_time[-1])):
        raise TEOBMergerNQCError("NQC time lies outside auxiliary support")
    index = int(torch.searchsorted(auxiliary_time.contiguous(), coordinate, right=True).item()) - 2
    if index < 0 or index >= auxiliary_time.numel():
        raise TEOBMergerNQCError("NQC matching index is outside derivative support")
    return index


def _unwrap_angle(angle: Tensor) -> Tensor:
    """Unwrap a one-dimensional phase while retaining Torch gradients."""

    if angle.numel() == 0:
        return angle
    two_pi = torch.as_tensor(2.0 * math.pi, dtype=angle.dtype, device=angle.device)
    pi = torch.as_tensor(math.pi, dtype=angle.dtype, device=angle.device)
    jumps = torch.remainder(angle[1:] - angle[:-1] + pi, two_pi) - pi
    return torch.cat((angle[:1], angle[:1] + torch.cumsum(jumps, dim=0)))


def phase_from_mode(mode: Tensor, orbital_phase: Tensor) -> Tensor:
    """Recover ``phase`` by unwrapping the residual after removing ``2 phi``.

    A sparse full mode can cross several ``pi`` intervals between accepted
    knots.  Unwrapping its raw argument directly is therefore ambiguous.  We
    first multiply by ``exp(+2 i orbital_phase)``, unwrap the residual mode,
    negate its argument for the ``exp(-i phase)`` convention, and restore the
    orbital phase.
    """

    if not isinstance(mode, Tensor) or mode.ndim != 1:
        raise ValueError("mode must be a one-dimensional complex series")
    if mode.dtype not in (torch.complex64, torch.complex128):
        raise TypeError("mode must be complex64 or complex128")
    real_dtype = torch.float32 if mode.dtype == torch.complex64 else torch.float64
    orbital_phase = _as_real(orbital_phase, name="orbital_phase", dtype=real_dtype,
                             device=mode.device)
    if orbital_phase.shape != mode.shape:
        raise ValueError("orbital_phase must match mode")
    if not bool(torch.isfinite(mode).all()) or bool((mode.abs() <= 0).any()):
        raise ValueError("mode must be finite and nonzero")
    imaginary = torch.zeros_like(orbital_phase)
    rotation = torch.exp(torch.complex(imaginary, 2.0 * orbital_phase))
    residual_angle = torch.angle(mode * rotation)
    residual_phase = -_unwrap_angle(residual_angle)
    return residual_phase + 2.0 * orbital_phase


lift_mode_phase = phase_from_mode
unwrap_mode_phase_after_orbital_carrier = phase_from_mode


def _coerce_targets(targets: object, *, dtype: torch.dtype, device: torch.device) -> TEOBMergerNQCTargets:
    if targets is None:
        raise ValueError("four explicit NQC targets are required; no default provider is used")
    if isinstance(targets, (TEOBMergerNQCTargets, NQCMatchingTargets)):
        values = tuple(targets)
    elif isinstance(targets, Mapping):
        names = (
            ("amplitude", "A", "A_t"),
            ("amplitude_derivative", "amplitude_rate", "Adot", "Adot_t"),
            ("frequency", "omega", "omega_t"),
            ("frequency_derivative", "frequency_rate", "omegadot", "omegadot_t"),
        )
        selected = []
        for choices in names:
            found = next((targets[key] for key in choices if key in targets), None)
            if found is None:
                raise ValueError("targets mapping must provide amplitude, amplitude rate, frequency, and frequency rate")
            selected.append(found)
        values = tuple(selected)
    else:
        try:
            values = tuple(targets)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("targets must be a four-array target record or sequence") from exc
    if len(values) != 4:
        raise ValueError("targets must contain exactly four scalar values")
    result = []
    for name, value in zip(("amplitude target", "amplitude-rate target",
                            "frequency target", "frequency-rate target"), values):
        result.append(_as_real(value, name=name, dtype=dtype, device=device, scalar=True))
    return TEOBMergerNQCTargets(*result)


def _solve_two_by_two(matrix: Tensor, rhs: Tensor, *, max_condition: float, name: str):
    if isinstance(max_condition, bool) or not isinstance(max_condition, (float, int)):
        raise TypeError("max_condition must be a finite number greater than one")
    if not math.isfinite(float(max_condition)) or max_condition <= 1.0:
        raise ValueError("max_condition must be finite and greater than one")
    if matrix.shape != (2, 2) or rhs.shape != (2,):
        raise ValueError(f"{name} matching system must be 2x2 with a two-entry RHS")
    if not bool(torch.isfinite(matrix).all() & torch.isfinite(rhs).all()):
        raise TEOBMergerNQCError(f"{name} matching system must be finite")

    row_scale = torch.linalg.vector_norm(matrix, dim=-1)
    if bool((row_scale == 0).any()):
        raise TEOBMergerNQCError(f"{name} matching system is singular")
    row_scaled = matrix / row_scale.unsqueeze(-1)
    column_scale = torch.linalg.vector_norm(row_scaled, dim=-2)
    if bool((column_scale == 0).any()):
        raise TEOBMergerNQCError(f"{name} matching system is singular")
    scaled = row_scaled / column_scale.unsqueeze(-2)
    singular_values = torch.linalg.svdvals(scaled)
    condition = singular_values[0] / singular_values[-1]
    if bool((~torch.isfinite(condition)) | (condition > float(max_condition))):
        raise TEOBMergerNQCError(f"{name} matching system is singular or ill-conditioned")
    scaled_rhs = rhs / row_scale
    solved_scaled = torch.linalg.solve(scaled, scaled_rhs.unsqueeze(-1)).squeeze(-1)
    solution = solved_scaled / column_scale
    residual = (matrix * solution.unsqueeze(-2)).sum(dim=-1) - rhs
    return solution, condition, residual


def _make_series(
    time: Tensor,
    state: Tensor,
    amplitude: Tensor,
    phase: Tensor,
    orbital_frequency: Tensor,
    radial_acceleration: Tensor,
    coefficients: NQCCoefficients,
) -> TEOBMergerNQCSeries:
    basis = nqc_basis_values(
        state[..., 0], state[..., 2], orbital_frequency, radial_acceleration,
    )
    corrected_amplitude = amplitude * (1.0 + (basis.amplitude * coefficients.amplitude).sum(-1))
    corrected_phase = phase - (basis.phase * coefficients.phase).sum(-1)
    return TEOBMergerNQCSeries(
        time, state, amplitude, phase, orbital_frequency, radial_acceleration,
        basis.amplitude, basis.phase, corrected_amplitude, corrected_phase,
    )


def _resample_scalar(knots: Tensor, values: Tensor, query: Tensor, *, name: str) -> Tensor:
    # Construct one independent natural spline per sampled derived quantity.
    try:
        spline = natural_cubic_spline(knots, values, scalar_output=True)
        return spline.evaluate(query)
    except (TypeError, ValueError) as exc:
        raise TEOBMergerNQCError(f"cannot resample {name} on the auxiliary grid: {exc}") from exc


def assemble_teob_merger_nqc(
    time: Tensor,
    state: Tensor,
    amplitude: Optional[Tensor] = None,
    phase: Optional[Tensor] = None,
    orbital_frequency: Optional[Tensor] = None,
    radial_acceleration: Optional[Tensor] = None,
    nu: Optional[Tensor | float] = None,
    fixed_step: Optional[Tensor | float] = None,
    targets: object = None,
    target_provenance: Optional[str] = None,
    *,
    mode: Optional[Tensor] = None,
    orbital_phase: Optional[Tensor] = None,
    matching_method: str = "grid",
    max_condition: float = 1.0e10,
) -> TEOBMergerNQCResult:
    """Build the auxiliary grid and solve the four explicit NQC conditions.

    ``time`` and all supplied series are one-dimensional scalar-intrinsic
    arrays.  ``state`` ends in ``(r,phi,p_r*,p_phi)``.  Callers may provide
    ``amplitude`` and an already unwrapped ``phase`` directly, or provide a
    complex ``mode`` together with ``orbital_phase``; the latter uses
    :func:`phase_from_mode` to lift the residual phase before restoring the
    carrier.  Four target values and a nonempty provenance label are always
    required.

    ``matching_method="continuous"`` locates the orbital-frequency spline
    maximum on all accepted samples and evaluates the four conditions at
    that time minus one mass unit. Derivatives are of the sampled amplitude
    products and phase bases, ensuring that the corrected accepted-series
    spline satisfies the targets. Auxiliary knots only materialize output;
    their spacing does not enter the solve. ``"grid"`` retains the audited
    discrete matching convention. Gradients are local to the selected peak.
    """

    if matching_method not in ("grid", "continuous"):
        raise ValueError("matching_method must be 'grid' or 'continuous'")
    if not isinstance(time, Tensor):
        raise TypeError("time must be a torch tensor")
    if time.dtype not in (torch.float32, torch.float64):
        raise TypeError("time must use float32 or float64")
    if time.ndim != 1:
        raise ValueError("merger NQC is scalar-intrinsic and time must be one-dimensional")
    dtype, device = time.dtype, time.device

    if mode is not None:
        if not isinstance(mode, Tensor):
            raise TypeError("mode must be a complex tensor")
        expected = torch.complex64 if dtype == torch.float32 else torch.complex128
        if mode.dtype != expected or mode.device != device or mode.shape != time.shape:
            raise TypeError("mode must match time with its corresponding complex dtype/device")
        if amplitude is None:
            amplitude = mode.abs()
        if phase is None:
            if orbital_phase is None:
                raise ValueError("orbital_phase is required when phase is recovered from mode")
            phase = phase_from_mode(mode, orbital_phase)
        elif orbital_phase is not None:
            raise ValueError("provide phase or mode/orbital_phase, not both phase charts")
    elif orbital_phase is not None:
        raise ValueError("orbital_phase requires mode")
    if amplitude is None or phase is None or orbital_frequency is None or radial_acceleration is None:
        raise ValueError("amplitude, phase, orbital_frequency, and radial_acceleration are required")
    _check_primary_series(
        time, state,
        (("amplitude", amplitude), ("phase", phase),
         ("orbital_frequency", orbital_frequency),
         ("radial_acceleration", radial_acceleration)),
    )
    if bool((state[:, 0] <= 0).any()):
        raise TEOBMergerNQCError("merger NQC requires positive radius")
    if bool((orbital_frequency <= 0).any()):
        raise TEOBMergerNQCError("merger NQC requires positive orbital frequency")
    if bool((amplitude <= 0).any()):
        raise TEOBMergerNQCError("merger NQC requires positive input amplitude")
    if target_provenance is None or not isinstance(target_provenance, str) or not target_provenance.strip():
        raise ValueError("target_provenance must be a descriptive nonempty label")
    if nu is None:
        raise ValueError("nu is required for RW amplitude normalization")
    nu_t = _as_real(nu, name="nu", dtype=dtype, device=device, scalar=True)
    if bool((nu_t <= 0).any()) or bool((nu_t > 0.25).any()):
        raise ValueError("nu must lie in (0, 0.25]")
    if fixed_step is None:
        raise ValueError("fixed_step is required for auxiliary spacing")
    fixed_t = _as_real(fixed_step, name="fixed_step", dtype=dtype, device=device, scalar=True)
    if bool(fixed_t <= 0):
        raise ValueError("fixed_step must be finite and positive")
    target_record = _coerce_targets(targets, dtype=dtype, device=device)

    n = time.numel()
    cut_last_index = n - 2  # the final accepted knot is deliberately excluded
    threshold = time[-1] - torch.as_tensor(20.0, dtype=dtype, device=device)
    cut_start_index = int(torch.searchsorted(time.contiguous(), threshold, right=True).item()) - 1
    cut_start_index = max(0, min(cut_start_index, cut_last_index))
    support_time = time[cut_start_index:cut_last_index + 1]
    if support_time.numel() < 5:
        raise TEOBMergerNQCError("auxiliary spline support needs at least five accepted knots")

    half_mass = torch.as_tensor(0.5, dtype=dtype, device=device)
    spacing = torch.minimum(half_mass, fixed_t)
    support_width = support_time[-1] - support_time[0]
    node_count_tensor = torch.floor(support_width / spacing) + 1.0
    if not bool(torch.isfinite(node_count_tensor)):
        raise TEOBMergerNQCError("auxiliary node count is not finite")
    node_count = int(node_count_tensor.detach().item())
    if node_count < 5:
        raise TEOBMergerNQCError("auxiliary grid has insufficient five-knot derivative support")
    aux_time = support_time[0] + torch.arange(node_count, dtype=dtype, device=device) * spacing

    support_slices = {
        "state": state[cut_start_index:cut_last_index + 1],
        "amplitude": amplitude[cut_start_index:cut_last_index + 1],
        "phase": phase[cut_start_index:cut_last_index + 1],
        "orbital_frequency": orbital_frequency[cut_start_index:cut_last_index + 1],
        "radial_acceleration": radial_acceleration[cut_start_index:cut_last_index + 1],
    }
    aux_state_spline = natural_cubic_spline(
        support_time, support_slices["state"], scalar_output=False,
    )
    aux_state = aux_state_spline.evaluate(aux_time)
    aux_amplitude = _resample_scalar(support_time, support_slices["amplitude"], aux_time, name="amplitude")
    aux_phase = _resample_scalar(support_time, support_slices["phase"], aux_time, name="phase")
    aux_frequency = _resample_scalar(
        support_time, support_slices["orbital_frequency"], aux_time,
        name="orbital frequency",
    )
    aux_acceleration = _resample_scalar(
        support_time, support_slices["radial_acceleration"], aux_time,
        name="radial acceleration",
    )
    if not bool(torch.isfinite(aux_state).all() & torch.isfinite(aux_amplitude).all()
                & torch.isfinite(aux_phase).all() & torch.isfinite(aux_frequency).all()
                & torch.isfinite(aux_acceleration).all()):
        raise TEOBMergerNQCError("auxiliary interpolation produced nonfinite samples")
    if bool((aux_state[:, 0] <= 0).any()) or bool((aux_frequency <= 0).any()):
        raise TEOBMergerNQCError("auxiliary interpolation left the NQC domain")
    if bool((aux_amplitude <= 0).any()):
        raise TEOBMergerNQCError("auxiliary interpolation produced nonpositive amplitude")

    sqrt24 = torch.sqrt(torch.as_tensor(24.0, dtype=dtype, device=device))
    amplitude_rw = amplitude / (nu_t * sqrt24)
    aux_amplitude_rw = aux_amplitude / (nu_t * sqrt24)
    aux_basis_values = nqc_basis_values(
        aux_state[:, 0], aux_state[:, 2], aux_frequency, aux_acceleration,
    )
    aux_q, aux_p = aux_basis_values.amplitude, aux_basis_values.phase
    if matching_method == "grid":
        d_amplitude_rw = quartic_derivative(aux_amplitude_rw, spacing)
        d_product = quartic_derivative(aux_amplitude_rw.unsqueeze(-1) * aux_q, spacing, sample_dim=-2)
        d_phase = quartic_derivative(aux_phase, spacing)
        d2_phase = quartic_derivative(d_phase, spacing)
        d_p = quartic_derivative(aux_p, spacing, sample_dim=-2)
        d2_p = quartic_derivative(d_p, spacing, sample_dim=-2)

    peak_index = int(torch.argmax(aux_frequency).item())
    t_peak_grid = aux_time[peak_index]
    t_nqc = t_peak_grid - torch.as_tensor(1.0, dtype=dtype, device=device)
    continuous_peak_time = None
    if matching_method == "continuous":
        try:
            continuous_peak_time = continuous_spline_peak(time, orbital_frequency)
            t_nqc = continuous_peak_time - 1.0
            basis = nqc_basis_values(state[:, 0], state[:, 2], orbital_frequency, radial_acceleration)
            amplitude_spline = natural_cubic_spline(time, amplitude_rw)
            product_spline = natural_cubic_spline(
                time, amplitude_rw[:, None] * basis.amplitude, scalar_output=False,
            )
            phase_spline = natural_cubic_spline(time, phase)
            basis_spline = natural_cubic_spline(time, basis.phase, scalar_output=False)
            amp_matrix = torch.stack((product_spline(t_nqc), product_spline.derivative(t_nqc)))
            amp_rhs = torch.stack((target_record.amplitude - amplitude_spline(t_nqc),
                                   target_record.amplitude_derivative - amplitude_spline.derivative(t_nqc)))
            phase_matrix = torch.stack((basis_spline.derivative(t_nqc), basis_spline.derivative(t_nqc, 2)))
            phase_rhs = torch.stack((phase_spline.derivative(t_nqc) - target_record.frequency,
                                     phase_spline.derivative(t_nqc, 2) - target_record.frequency_derivative))
        except ValueError as exc:
            raise TEOBMergerNQCError(f"continuous matching is unsupported: {exc}") from exc
        # A bracketing auxiliary index is diagnostic only, never an input to
        # continuous matching. -1 means the event precedes auxiliary support.
        selected = int(torch.searchsorted(aux_time, t_nqc, right=True).item()) - 1
    else:
        selected = matching_index(aux_time, t_nqc)
        amp_matrix = torch.stack((aux_amplitude_rw[selected] * aux_q[selected], d_product[selected]), dim=0)
        amp_rhs = torch.stack((target_record.amplitude - aux_amplitude_rw[selected],
                               target_record.amplitude_derivative - d_amplitude_rw[selected]))
        phase_matrix = torch.stack((d_p[selected], d2_p[selected]), dim=0)
        phase_rhs = torch.stack((d_phase[selected] - target_record.frequency,
                                 d2_phase[selected] - target_record.frequency_derivative))
    a, amp_condition, amp_residual = _solve_two_by_two(
        amp_matrix, amp_rhs, max_condition=max_condition, name="amplitude",
    )
    b, phase_condition, phase_residual = _solve_two_by_two(
        phase_matrix, phase_rhs, max_condition=max_condition, name="phase",
    )
    coefficients = NQCCoefficients(a, b)
    match = NQCMatchResult(
        coefficients, amp_matrix, phase_matrix, amp_condition, phase_condition,
        amp_residual, phase_residual,
    )
    original = _make_series(
        time, state, amplitude, phase, orbital_frequency, radial_acceleration,
        coefficients,
    )
    auxiliary = _make_series(
        aux_time, aux_state, aux_amplitude, aux_phase, aux_frequency,
        aux_acceleration, coefficients,
    )
    negative = auxiliary.corrected_amplitude < 0
    return TEOBMergerNQCResult(
        original=original,
        auxiliary=auxiliary,
        coefficients=coefficients,
        match=match,
        targets=target_record,
        target_provenance=target_provenance.strip(),
        cut_start_index=cut_start_index,
        cut_last_index=cut_last_index,
        auxiliary_spacing=spacing,
        t_peak_grid=t_peak_grid,
        t_nqc=t_nqc,
        matching_index=selected,
        amplitude_rw=amplitude_rw,
        auxiliary_amplitude_rw=aux_amplitude_rw,
        auxiliary_negative_amplitude=negative,
        matching_method=matching_method,
        continuous_peak_time=continuous_peak_time,
    )


# Explicit aliases keep the function discoverable without adding another
# implementation or a production dispatch path.
calibrate_teob_merger_nqc = assemble_teob_merger_nqc
build_teob_merger_nqc = assemble_teob_merger_nqc
construct_teob_merger_nqc = assemble_teob_merger_nqc
teob_merger_nqc = assemble_teob_merger_nqc


__all__ = [
    "TEOBMergerNQCError",
    "TEOBMergerNQCTargets",
    "PUBLISHED_2017_NQC_PROVENANCE",
    "published_2017_nqc_targets",
    "nqc_targets_2017",
    "nagar_2017_nqc_targets",
    "TEOBMergerNQCSeries",
    "TEOBMergerNQCResult",
    "quartic_derivative",
    "degree_four_derivative",
    "five_point_quartic_derivative",
    "D",
    "matching_index",
    "phase_from_mode",
    "lift_mode_phase",
    "unwrap_mode_phase_after_orbital_carrier",
    "assemble_teob_merger_nqc",
    "calibrate_teob_merger_nqc",
    "build_teob_merger_nqc",
    "construct_teob_merger_nqc",
    "teob_merger_nqc",
]

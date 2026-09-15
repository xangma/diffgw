"""Clean-room, nonspinning TEOBResumS NQC matching equations.

The four parameter factor is the one printed in Nagar et al.,
arXiv:1406.6913v1, Eqs. (96)--(97), and in Damour et al.,
arXiv:1806.01772v2, Eq. (14)::

    h_NQC = (1 + a_1 n_1 + a_2 n_2)
            exp(i (b_1 n'_1 + b_2 n'_2)).

This module only evaluates that basis, solves the four local osculation
conditions, and applies the factor.  A trajectory, a peak finder, and a
waveform normalization are deliberately not implicit.  In particular, the
time derivatives of the basis are caller supplied.  This keeps the numerical
choice used to obtain derivatives visible when a future EOB trajectory is
connected to this component.

The phase convention used throughout is ``h = A exp(-i phase)``.  Therefore
the factor's ``+i`` phase means ``phase_corrected = phase - b.n_prime`` and
``frequency_corrected = frequency - b.dn_prime_dt``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class NQCBasisValues(NamedTuple):
    """The four dimensionless basis values at one matching state.

    ``amplitude`` has final axis ``(n1, n2)`` and ``phase`` has final axis
    ``(n1_prime, n2_prime)``.  Leading axes are arbitrary batch axes.
    """

    amplitude: torch.Tensor
    phase: torch.Tensor

    @property
    def n(self) -> torch.Tensor:
        """Alias for the amplitude basis ``(n1, n2)``."""

        return self.amplitude

    @property
    def n_prime(self) -> torch.Tensor:
        """Alias for the phase basis ``(n1_prime, n2_prime)``."""

        return self.phase


class NQCBasis(NamedTuple):
    """NQC basis values and the derivatives needed by the local solve.

    Every field has final axis of length two.  The derivative fields mean
    ``d(amplitude)/dt``, ``d(phase)/dt`` and ``d²(phase)/dt²`` respectively,
    in the same dimensionless time coordinate as the supplied waveform
    conditions.  No finite-difference or interpolation scheme is selected by
    this class.
    """

    amplitude: torch.Tensor
    amplitude_derivative: torch.Tensor
    phase: torch.Tensor
    phase_derivative: torch.Tensor
    phase_second_derivative: torch.Tensor

    @classmethod
    def from_values(
        cls,
        values: NQCBasisValues,
        amplitude_derivative: torch.Tensor,
        phase_derivative: torch.Tensor,
        phase_second_derivative: torch.Tensor,
    ) -> "NQCBasis":
        """Attach explicitly supplied derivatives to :class:`NQCBasisValues`."""

        if not isinstance(values, NQCBasisValues):
            raise TypeError("values must be an NQCBasisValues instance")
        return cls(values.amplitude, amplitude_derivative, values.phase,
                   phase_derivative, phase_second_derivative)

    @property
    def n(self) -> torch.Tensor:
        """Alias for the amplitude basis ``(n1, n2)``."""

        return self.amplitude

    @property
    def dn_dt(self) -> torch.Tensor:
        """Alias for ``d(n1, n2)/dt``."""

        return self.amplitude_derivative

    @property
    def n_prime(self) -> torch.Tensor:
        """Alias for the phase basis ``(n1_prime, n2_prime)``."""

        return self.phase

    @property
    def dn_prime_dt(self) -> torch.Tensor:
        """Alias for ``d(n1_prime, n2_prime)/dt``."""

        return self.phase_derivative

    @property
    def d2n_prime_dt2(self) -> torch.Tensor:
        """Alias for ``d²(n1_prime, n2_prime)/dt²``."""

        return self.phase_second_derivative


class NQCSignal(NamedTuple):
    """Uncorrected or corrected local waveform conditions.

    ``frequency`` means the positive gravitational-wave frequency in the
    convention ``h=A exp(-i phase)``.  The two derivative fields use the same
    time coordinate as the basis derivatives.
    """

    amplitude: torch.Tensor
    amplitude_derivative: torch.Tensor
    frequency: torch.Tensor
    frequency_derivative: torch.Tensor

    @property
    def amplitude_rate(self) -> torch.Tensor:
        """Alias for ``amplitude_derivative``."""

        return self.amplitude_derivative

    @property
    def frequency_rate(self) -> torch.Tensor:
        """Alias for ``frequency_derivative``."""

        return self.frequency_derivative


class NQCMatchingTargets(NamedTuple):
    """The four target values imposed by the osculation conditions."""

    amplitude: torch.Tensor
    amplitude_derivative: torch.Tensor
    frequency: torch.Tensor
    frequency_derivative: torch.Tensor

    @property
    def amplitude_rate(self) -> torch.Tensor:
        """Alias for ``amplitude_derivative``."""

        return self.amplitude_derivative

    @property
    def frequency_rate(self) -> torch.Tensor:
        """Alias for ``frequency_derivative``."""

        return self.frequency_derivative


class NQCCoefficients(NamedTuple):
    """Solved coefficients with final axes ``(a1, a2)`` and ``(b1, b2)``."""

    amplitude: torch.Tensor
    phase: torch.Tensor

    @property
    def a(self) -> torch.Tensor:
        """Alias for the amplitude coefficients."""

        return self.amplitude

    @property
    def b(self) -> torch.Tensor:
        """Alias for the phase coefficients."""

        return self.phase


class NQCMatchResult(NamedTuple):
    """Coefficients and compact diagnostics from a four-condition solve."""

    coefficients: NQCCoefficients
    amplitude_matrix: torch.Tensor
    phase_matrix: torch.Tensor
    amplitude_condition: torch.Tensor
    phase_condition: torch.Tensor
    amplitude_residual: torch.Tensor
    phase_residual: torch.Tensor

    @property
    def sampling_condition(self) -> torch.Tensor:
        """The larger of the two scaled matching condition numbers."""

        return torch.maximum(self.amplitude_condition, self.phase_condition)


# Names used by neighbouring EOB components and by callers that regard a
# match result as a calibration are harmless aliases, rather than separate
# implementations.
NQCTargets = NQCMatchingTargets
NQCCalibration = NQCMatchResult


def _validate_real_tensors(values, name: str) -> torch.Tensor:
    """Check a collection of same-device, same-dtype real Torch tensors."""

    values = tuple(values)
    if not values or not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError(f"{name} must be supplied as torch tensors")
    reference = values[0]
    if reference.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64 tensors")
    if any(value.dtype != reference.dtype or value.device != reference.device
           for value in values):
        raise TypeError(f"{name} tensors must share dtype and device")
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError(f"{name} tensors must be finite")
    return reference


def _validate_two_columns(values, name: str) -> None:
    for value in values:
        if value.ndim < 1 or value.shape[-1] != 2:
            raise ValueError(f"{name} fields must have final axis of length 2")


def nqc_basis_values(
    r: torch.Tensor,
    pr_star: torch.Tensor,
    orbital_frequency: torch.Tensor,
    radial_acceleration: torch.Tensor,
) -> NQCBasisValues:
    """Evaluate the clean-room TEOBResumS basis in Eqs. (96)--(97).

    The returned values are, in equation order,

    ``n1=(p_r*/(r Omega))²``, ``n2=rddot_conservative/(r Omega²)``,
    ``n1_prime=p_r*/(r Omega)``, and ``n2_prime=p_r* r Omega``.

    ``radial_acceleration`` is explicitly the conservative ``r̈``.  This
    function does not differentiate or interpolate its inputs.  Inputs may be
    broadcast over arbitrary leading axes and must be real float32/float64
    tensors on one device.
    """

    _validate_real_tensors(
        (r, pr_star, orbital_frequency, radial_acceleration), "NQC basis inputs")
    r, pr_star, orbital_frequency, radial_acceleration = torch.broadcast_tensors(
        r, pr_star, orbital_frequency, radial_acceleration)
    if bool((r <= 0).any()) or bool((orbital_frequency == 0).any()):
        raise ValueError("NQC basis requires r > 0 and nonzero orbital frequency")

    ratio = pr_star / (r * orbital_frequency)
    amplitude = torch.stack(
        (ratio.square(), radial_acceleration / (r * orbital_frequency.square())),
        dim=-1,
    )
    phase = torch.stack((ratio, pr_star * r * orbital_frequency), dim=-1)
    return NQCBasisValues(amplitude, phase)


def make_nqc_basis(
    values: NQCBasisValues,
    amplitude_derivative: torch.Tensor,
    phase_derivative: torch.Tensor,
    phase_second_derivative: torch.Tensor,
) -> NQCBasis:
    """Construct a full basis from values and caller-supplied derivatives."""

    return NQCBasis.from_values(values, amplitude_derivative, phase_derivative,
                                phase_second_derivative)


# The equation-facing name is useful when this helper is used beside a
# trajectory module.  It intentionally returns values only; derivatives stay
# an explicit argument to ``NQCBasis``/``make_nqc_basis``.
teob_nqc_basis_values = nqc_basis_values
teob_nqc_basis = nqc_basis_values
nqc_basis = nqc_basis_values


def _validate_match_objects(
    base: NQCSignal,
    targets: NQCMatchingTargets,
    basis: NQCBasis,
    coefficients: NQCCoefficients | None = None,
):
    # Conditions have identical fields whether they describe the uncorrected
    # signal or the requested target.  Accepting the two record types in
    # either position keeps ``corrected_matching_signal(...)`` directly
    # usable as the next call's targets without weakening tensor validation.
    if isinstance(base, NQCMatchingTargets):
        base = NQCSignal(*base)
    if isinstance(targets, NQCSignal):
        targets = NQCMatchingTargets(*targets)
    if not isinstance(base, NQCSignal):
        raise TypeError("base must be an NQCSignal instance")
    if not isinstance(targets, NQCMatchingTargets):
        raise TypeError("targets must be an NQCMatchingTargets instance")
    if not isinstance(basis, NQCBasis):
        raise TypeError("basis must be an NQCBasis instance")
    if coefficients is not None and not isinstance(coefficients, NQCCoefficients):
        raise TypeError("coefficients must be an NQCCoefficients instance")

    real_values = (*base, *targets, *basis)
    if coefficients is not None:
        real_values = (*real_values, *coefficients)
    reference = _validate_real_tensors(real_values, "NQC matching inputs")
    _validate_two_columns(basis, "NQC basis")
    if coefficients is not None:
        _validate_two_columns(coefficients, "NQC coefficients")

    signal_shapes = tuple(value.shape for value in (*base, *targets))
    basis_shapes = tuple(value.shape[:-1] for value in basis)
    coefficient_shapes = () if coefficients is None else tuple(
        value.shape[:-1] for value in coefficients)
    batch_shape = torch.broadcast_shapes(
        *signal_shapes, *basis_shapes, *coefficient_shapes)

    def expand_signal(signal):
        return type(signal)(*(value.expand(batch_shape) for value in signal))

    def expand_basis(value):
        return type(value)(*(field.expand(batch_shape + (2,)) for field in value))

    expanded_base = expand_signal(base)
    expanded_targets = expand_signal(targets)
    expanded_basis = expand_basis(basis)
    if coefficients is None:
        return reference, expanded_base, expanded_targets, expanded_basis
    return reference, expanded_base, expanded_targets, expanded_basis, NQCCoefficients(
        *(field.expand(batch_shape + (2,)) for field in coefficients))


def _solve_two_by_two(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    max_condition: float,
    name: str,
):
    """Solve a batched 2x2 system after transparent row/column equilibration."""

    if not isinstance(max_condition, (int, float)) or not torch.isfinite(
        torch.as_tensor(float(max_condition))
    ) or max_condition <= 1:
        raise ValueError("max_condition must be finite and greater than one")
    if not bool(torch.isfinite(matrix).all()) or not bool(torch.isfinite(rhs).all()):
        raise ValueError(f"{name} matching system must be finite")

    # Equilibration changes neither the exact solution nor the residual.  It
    # makes the diagnostic useful when the two basis columns carry different
    # units or magnitudes, while retaining a direct torch.linalg.solve call.
    row_scale = torch.linalg.vector_norm(matrix, dim=-1)
    if bool((row_scale == 0).any()):
        raise ValueError(f"{name} system is rank deficient (zero matching row)")
    scaled = matrix / row_scale.unsqueeze(-1)
    column_scale = torch.linalg.vector_norm(scaled, dim=-2)
    if bool((column_scale == 0).any()):
        raise ValueError(f"{name} system is rank deficient (zero basis column)")
    scaled = scaled / column_scale.unsqueeze(-2)

    singular = torch.linalg.svdvals(scaled)
    condition = singular[..., 0] / singular[..., -1]
    numerical_limit = 1.0 / (torch.finfo(matrix.dtype).eps * 2)
    threshold = min(float(max_condition), numerical_limit)
    if bool((~torch.isfinite(condition) | (condition > threshold)).any()):
        raise ValueError(f"{name} system is rank deficient or ill-conditioned")

    scaled_rhs = rhs / row_scale
    solution_scaled = torch.linalg.solve(scaled, scaled_rhs.unsqueeze(-1)).squeeze(-1)
    solution = solution_scaled / column_scale
    residual = (matrix * solution.unsqueeze(-2)).sum(dim=-1) - rhs
    return solution, condition, residual


def match_nqc(
    base: NQCSignal,
    targets: NQCMatchingTargets,
    basis: NQCBasis,
    *,
    max_condition: float = 1.0e10,
) -> NQCMatchResult:
    """Solve the four TEOBResumS osculation conditions.

    The amplitude and phase coefficients are solved as independent 2x2
    systems.  With ``N=1+a.n`` and ``B=b.n_prime``, the systems are

    ``[A*n; Adot*n + A*dn] a = [At-A; Adot_t-Adot]`` and
    ``[dnp; d2np] b = [omega-omega_t; omegadot-omegadot_t]``.

    The second right-hand side follows directly from ``h=A exp(-i phase)``
    and the ``exp(+i B)`` in the published NQC factor.  No fitted constants,
    pseudoinverse, regularizer, or coefficient clipping is used.
    """

    _, base, targets, basis = _validate_match_objects(base, targets, basis)
    if bool((base.amplitude <= 0).any()) or bool((targets.amplitude <= 0).any()):
        raise ValueError("NQC matching amplitudes must be positive")
    if bool((base.frequency <= 0).any()) or bool((targets.frequency <= 0).any()):
        raise ValueError("NQC matching frequencies must be positive")

    amplitude_matrix = torch.stack(
        (
            base.amplitude.unsqueeze(-1) * basis.amplitude,
            base.amplitude_derivative.unsqueeze(-1) * basis.amplitude
            + base.amplitude.unsqueeze(-1) * basis.amplitude_derivative,
        ),
        dim=-2,
    )
    amplitude_rhs = torch.stack(
        (targets.amplitude - base.amplitude,
         targets.amplitude_derivative - base.amplitude_derivative),
        dim=-1,
    )
    amplitude, amplitude_condition, amplitude_residual = _solve_two_by_two(
        amplitude_matrix, amplitude_rhs, max_condition, "amplitude")

    phase_matrix = torch.stack(
        (basis.phase_derivative, basis.phase_second_derivative), dim=-2)
    phase_rhs = torch.stack(
        (base.frequency - targets.frequency,
         base.frequency_derivative - targets.frequency_derivative),
        dim=-1,
    )
    phase, phase_condition, phase_residual = _solve_two_by_two(
        phase_matrix, phase_rhs, max_condition, "phase")
    return NQCMatchResult(
        NQCCoefficients(amplitude, phase),
        amplitude_matrix,
        phase_matrix,
        amplitude_condition,
        phase_condition,
        amplitude_residual,
        phase_residual,
    )


def solve_nqc_coefficients(
    base: NQCSignal,
    targets: NQCMatchingTargets,
    basis: NQCBasis,
    *,
    max_condition: float = 1.0e10,
) -> NQCCoefficients:
    """Return only the four coefficients from :func:`match_nqc`."""

    return match_nqc(base, targets, basis,
                     max_condition=max_condition).coefficients


solve_nqc = solve_nqc_coefficients
calibrate_nqc = match_nqc


def _basis_value_pair(basis):
    if isinstance(basis, NQCBasis):
        return basis.amplitude, basis.phase
    if isinstance(basis, NQCBasisValues):
        return basis.amplitude, basis.phase
    raise TypeError("basis must be an NQCBasis or NQCBasisValues instance")


def nqc_correction_factor(
    basis: NQCBasis | NQCBasisValues,
    coefficients: NQCCoefficients,
    *,
    sample_dims: int = 0,
) -> torch.Tensor:
    """Evaluate the published complex NQC factor.

    Coefficient batch axes precede ``sample_dims`` sample axes.  For example,
    coefficients shaped ``(B, 2)`` and basis shaped ``(B, N, 2)`` require
    ``sample_dims=1``.  The real amplitude multiplier is intentionally signed,
    exactly as printed in Eq. (96)/(14); it is not clipped.
    """

    if not isinstance(coefficients, NQCCoefficients):
        raise TypeError("coefficients must be an NQCCoefficients instance")
    amplitude_basis, phase_basis = _basis_value_pair(basis)
    _validate_real_tensors(
        (amplitude_basis, phase_basis, *coefficients), "NQC correction inputs")
    _validate_two_columns((amplitude_basis, phase_basis), "NQC basis values")
    _validate_two_columns(coefficients, "NQC coefficients")
    if not isinstance(sample_dims, int) or isinstance(sample_dims, bool) or sample_dims < 0:
        raise ValueError("sample_dims must be a nonnegative integer")

    a = coefficients.amplitude.reshape(
        coefficients.amplitude.shape[:-1] + (1,) * sample_dims + (2,))
    b = coefficients.phase.reshape(
        coefficients.phase.shape[:-1] + (1,) * sample_dims + (2,))
    amplitude_basis, phase_basis, a, b = torch.broadcast_tensors(
        amplitude_basis, phase_basis, a, b)
    multiplier = 1.0 + (amplitude_basis * a).sum(dim=-1)
    phase_shift = (phase_basis * b).sum(dim=-1)
    factor = multiplier * torch.exp(torch.complex(torch.zeros_like(phase_shift), phase_shift))
    return factor


def apply_nqc_correction(
    waveform: torch.Tensor,
    basis: NQCBasis | NQCBasisValues,
    coefficients: NQCCoefficients,
    *,
    sample_dims: int = 0,
) -> torch.Tensor:
    """Multiply a complex ``h=A exp(-i phase)`` waveform by the NQC factor."""

    if not isinstance(waveform, torch.Tensor) or waveform.dtype not in (
        torch.complex64, torch.complex128
    ):
        raise TypeError("waveform must be a complex64 or complex128 torch tensor")
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("waveform must be finite")
    amplitude_basis, _ = _basis_value_pair(basis)
    expected_real_dtype = torch.float64 if waveform.dtype == torch.complex128 else torch.float32
    if amplitude_basis.dtype != expected_real_dtype or amplitude_basis.device != waveform.device:
        raise TypeError("waveform and NQC basis must share corresponding dtype and device")
    factor = nqc_correction_factor(basis, coefficients, sample_dims=sample_dims)
    return waveform * factor


nqc_factor = nqc_correction_factor


def corrected_matching_signal(
    base: NQCSignal,
    basis: NQCBasis,
    coefficients: NQCCoefficients,
) -> NQCSignal:
    """Evaluate local amplitude/frequency conditions after applying NQC."""

    dummy_targets = NQCMatchingTargets(
        base.amplitude, base.amplitude_derivative,
        base.frequency, base.frequency_derivative)
    _, base, _, basis, coefficients = _validate_match_objects(
        base, dummy_targets, basis, coefficients)
    multiplier = 1.0 + (basis.amplitude * coefficients.amplitude).sum(dim=-1)
    multiplier_rate = (
        basis.amplitude_derivative * coefficients.amplitude).sum(dim=-1)
    phase_rate = (basis.phase_derivative * coefficients.phase).sum(dim=-1)
    phase_second_rate = (
        basis.phase_second_derivative * coefficients.phase).sum(dim=-1)
    return NQCSignal(
        base.amplitude * multiplier,
        base.amplitude_derivative * multiplier + base.amplitude * multiplier_rate,
        base.frequency - phase_rate,
        base.frequency_derivative - phase_second_rate,
    )


def rw_normalize_22(waveform: torch.Tensor) -> torch.Tensor:
    """Convert ``h_22`` to the paper's RW-normalized ``Psi_22=h_22/sqrt(24)``."""

    if not isinstance(waveform, torch.Tensor) or waveform.dtype not in (
        torch.complex64, torch.complex128
    ):
        raise TypeError("waveform must be complex64 or complex128")
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("waveform must be finite")
    real_dtype = torch.float64 if waveform.dtype == torch.complex128 else torch.float32
    sqrt24 = torch.sqrt(torch.as_tensor(24.0, dtype=real_dtype, device=waveform.device))
    return waveform / sqrt24


def rw_denormalize_22(waveform: torch.Tensor) -> torch.Tensor:
    """Convert RW-normalized ``Psi_22`` back to ``h_22``."""

    if not isinstance(waveform, torch.Tensor) or waveform.dtype not in (
        torch.complex64, torch.complex128
    ):
        raise TypeError("waveform must be complex64 or complex128")
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError("waveform must be finite")
    real_dtype = torch.float64 if waveform.dtype == torch.complex128 else torch.float32
    sqrt24 = torch.sqrt(torch.as_tensor(24.0, dtype=real_dtype, device=waveform.device))
    return waveform * sqrt24


zerilli_normalize_22 = rw_normalize_22
zerilli_denormalize_22 = rw_denormalize_22


def nqc_match_time_from_orbital_peak(orbital_frequency_peak: torch.Tensor) -> torch.Tensor:
    """Apply the nonspinning paper convention ``t_NQC=t_peak_Omega-1M``."""

    _validate_real_tensors((orbital_frequency_peak,), "orbital peak time")
    return orbital_frequency_peak - torch.ones_like(orbital_frequency_peak)


def merger_chart_time_from_nqc(nqc_time: torch.Tensor) -> torch.Tensor:
    """Return the chart merger time ``t_NQC-2M`` used for the NR+2M anchor."""

    _validate_real_tensors((nqc_time,), "NQC time")
    return nqc_time - 2.0 * torch.ones_like(nqc_time)


__all__ = [
    "NQCBasisValues",
    "NQCBasis",
    "NQCSignal",
    "NQCMatchingTargets",
    "NQCTargets",
    "NQCCoefficients",
    "NQCMatchResult",
    "NQCCalibration",
    "nqc_basis_values",
    "teob_nqc_basis_values",
    "teob_nqc_basis",
    "nqc_basis",
    "make_nqc_basis",
    "match_nqc",
    "solve_nqc_coefficients",
    "solve_nqc",
    "calibrate_nqc",
    "nqc_correction_factor",
    "nqc_factor",
    "apply_nqc_correction",
    "corrected_matching_signal",
    "rw_normalize_22",
    "rw_denormalize_22",
    "zerilli_normalize_22",
    "zerilli_denormalize_22",
    "nqc_match_time_from_orbital_peak",
    "merger_chart_time_from_nqc",
]

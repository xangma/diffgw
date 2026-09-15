"""Standalone (2,2) NQC equations from Bohé et al., arXiv:1611.03703v3.

All times are tau=t/M, radii are R/M, radial momenta are p_r*/mu, and
orbital frequencies are M*Omega. Mode amplitudes use (D_L/M)*abs(h22).
Positive GW frequency means -d(arg(h22))/d tau: h22=A*exp(-i*phi22).
No trajectory evolution, peak search, or waveform attachment is performed.
"""

from typing import NamedTuple

import torch


class NQCCoefficients(NamedTuple):
    amplitude: torch.Tensor  # (..., 3), a1, a2, a3
    phase: torch.Tensor  # (..., 2), b1, b2


class NQCMatchingTargets(NamedTuple):
    amplitude: torch.Tensor
    amplitude_derivative: torch.Tensor
    amplitude_second_derivative: torch.Tensor
    frequency: torch.Tensor
    frequency_derivative: torch.Tensor


class NQCCalibration(NamedTuple):
    coefficients: NQCCoefficients
    sampling_condition: torch.Tensor
    amplitude_condition: torch.Tensor
    phase_condition: torch.Tensor
    amplitude_residual: torch.Tensor
    phase_residual: torch.Tensor


def _real_tensors(*values):
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("NQC inputs must be torch tensors")
    reference = values[0]
    if reference.dtype not in (torch.float32, torch.float64):
        raise TypeError("NQC inputs must use float32 or float64")
    if any(value.dtype != reference.dtype or value.device != reference.device
           for value in values):
        raise TypeError("NQC inputs must share a floating dtype and device")
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("NQC inputs must be finite")


def _fit_domain(nu, chi):
    _real_tensors(nu, chi)
    nu, chi = torch.broadcast_tensors(nu, chi)
    if bool(((nu < 0) | (nu > 0.25) | (chi.abs() > 1)).any()):
        raise ValueError("published NQC fits require 0 <= nu <= 1/4 and |chi| <= 1")
    return nu, chi


def seobnrv4_nqc_parameters(m1, m2, chi1, chi2):
    """Return nu and Eq. (2.8)'s chi; invariant under mass scaling and exchange."""
    _real_tensors(m1, m2, chi1, chi2)
    m1, m2, chi1, chi2 = torch.broadcast_tensors(m1, m2, chi1, chi2)
    if bool(((m1 <= 0) | (m2 <= 0) | (chi1.abs() > 1) | (chi2.abs() > 1)).any()):
        raise ValueError("masses must be positive and component spins within [-1, 1]")
    x1, x2 = m1/(m1 + m2), m2/(m1 + m2)
    return x1*x2, (x1.square()*chi1 + x2.square()*chi2)/(x1.square() + x2.square())


def _endpoint_quadratic(nu, test_particle, equal_mass, linear):
    # Appendix A fixes A1 independently, then imposes endpoints at .001 and .25.
    lo, hi = 0.001, 0.25
    quadratic = (equal_mass - test_particle - linear*(hi - lo))/(hi**2 - lo**2)
    return test_particle + linear*(nu - lo) + quadratic*(nu.square() - lo**2)


def seobnrv4_nqc_targets(nu, chi):
    """Published peak targets, Eqs. (A1)-(A12); first amplitude derivative is zero.

    A9's frequency interpolation literally uses nu=0 for its test-particle
    endpoint. A2, A5 and A12 instead impose their endpoint at nu=0.001.
    The returned values have no distance factor and derivatives use tau=t/M.
    These printed fits are not a new NR validation or a full waveform model.
    """
    nu, chi = _fit_domain(nu, chi)
    amplitude_tpl = 1.452857 + chi*(0.166134 + chi*(0.027356 - 0.020073*chi))
    amplitude_eq = 1.577458 + chi*(-0.007695 + chi*(0.021887 + 0.023268*chi))
    amplitude_linear = -0.034424 + chi*(-1.218066 + chi*(-0.568373 + 0.401114*chi))
    amplitude = nu*_endpoint_quadratic(nu, amplitude_tpl, amplitude_eq, amplitude_linear)

    shifted = chi - 1.0
    curvature_tpl = shifted*(0.00239561 + shifted*(-0.00019274 - 0.00029666*shifted))
    curvature_eq = -0.00412651 + 0.00222400*chi
    curvature = nu*_endpoint_quadratic(nu, curvature_tpl, curvature_eq,
                                       -0.00577654 + 0.00103086*chi)

    a3 = 10.262073 + 4.0*(10.262073 - 25.850378)*(nu - 0.25)
    a4 = 7.629922 + 4.0*(7.629922 - 25.819795)*(nu - 0.25)
    frequency = 0.562679 + (-0.087062 + 0.001743*chi)*torch.log(a3 - a4*chi)

    slope_tpl = -0.011210 + (0.004087 + 0.000633*chi)*torch.log(68.474666 - 58.301488*chi)
    slope_eq = 0.011282 + 0.000287*chi
    slope = _endpoint_quadratic(nu, slope_tpl, slope_eq, 0.015743 + 0.022442*chi)
    return NQCMatchingTargets(amplitude, torch.zeros_like(amplitude), curvature,
                              frequency, slope)


def seobnrv4_nqc_time_offset(nu, chi):
    """Delta t22_peak/M, Eqs. (4.9), (4.15): t22_peak=tOmega_peak+Delta t22_peak."""
    nu, chi = _fit_domain(nu, chi)
    nonspinning = 716.044155*nu.pow(3) - 13.087878*nu.square() - 45.883834*nu - 2.504992
    return (nonspinning - 0.192775*chi.pow(3)*nu.square() + 19.053803*chi.pow(3)*nu
            - 11.543497*chi.square() + 40.318332*chi*nu - 13.006363*chi)


# Independent clean-room transcription of the coefficients printed in Eqs.
# (4.9) and (4.15) of Bohé et al., arXiv:1611.03703v3.  The numerical
# compatibility convention rounds each scalar coefficient to six significant
# decimal digits once, before evaluating the polynomial.  Keeping these as
# literals (rather than rounding tensor results) preserves autograd through
# nu and chi and makes the rounding rule explicit.
_NQC_TIME_OFFSET_SIX_SIG = {
    "nu3": 716.044,
    "nu2": -13.0879,
    "nu1": -45.8838,
    "constant": -2.50499,
    "chi3_nu2": -0.192775,
    "chi3_nu": 19.0538,
    "chi2": -11.5435,
    "chi_nu": 40.3183,
    "chi": -13.0064,
}


def seobnrv4_nqc_time_offset_numerical_compatibility(nu, chi):
    """Evaluate the NQC peak offset with six-significant-digit coefficients.

    This is an explicit opt-in numerical compatibility convention.  It uses
    only the independently published polynomial in Eqs. (4.9) and (4.15) of
    Bohé et al. (arXiv:1611.03703v3), with every printed coefficient rounded
    to six significant decimal digits before evaluation.  The published
    :func:`seobnrv4_nqc_time_offset` helper retains its exact printed-six-place
    values and is unchanged.
    """
    nu, chi = _fit_domain(nu, chi)
    c = _NQC_TIME_OFFSET_SIX_SIG
    nonspinning = c["nu3"]*nu.pow(3) + c["nu2"]*nu.square() + c["nu1"]*nu + c["constant"]
    return (nonspinning + c["chi3_nu2"]*chi.pow(3)*nu.square()
            + c["chi3_nu"]*chi.pow(3)*nu + c["chi2"]*chi.square()
            + c["chi_nu"]*chi*nu + c["chi"]*chi)




def nqc_22_basis(r, pr_star, orbital_frequency):
    """Eq. (2.7)'s real amplitude (last axis 3) and phase (last axis 2) bases."""
    _real_tensors(r, pr_star, orbital_frequency)
    r, pr_star, orbital_frequency = torch.broadcast_tensors(r, pr_star, orbital_frequency)
    if bool(((r <= 0) | (orbital_frequency == 0)).any()):
        raise ValueError("NQC basis requires r > 0 and nonzero orbital frequency")
    ratio = pr_star/(r*orbital_frequency)
    amplitude = torch.stack((ratio.square(), ratio.square()/r,
                             ratio.square()/r.pow(1.5)), dim=-1)
    phase = torch.stack((ratio, ratio*pr_star.square()), dim=-1)
    return amplitude, phase


def nqc_22_factor(r, pr_star, orbital_frequency, coefficients, *, sample_dims=0):
    """Complex N22; coefficient batch axes precede ``sample_dims`` sample axes.

    As in Eq. (2.7), the real amplitude multiplier is signed. This function
    evaluates the equation and does not clip the correction away from matching.
    Use sample_dims=1 for trajectories (..., N), including a shared (N,) stencil
    evaluated with a batch of coefficients. The default evaluates individual
    states with ordinary broadcasting and does not guess which axes are time.
    """
    _real_tensors(r, pr_star, orbital_frequency, *coefficients)
    amplitude, phase = nqc_22_basis(r, pr_star, orbital_frequency)
    a, b = coefficients
    if a.ndim < 1 or b.ndim < 1 or a.shape[-1] != 3 or b.shape[-1] != 2:
        raise ValueError("NQC coefficient final axes must have lengths 3 and 2")
    if not isinstance(sample_dims, int) or sample_dims < 0:
        raise ValueError("sample_dims must be a nonnegative integer")
    a = a.reshape(a.shape[:-1] + (1,)*sample_dims + (3,))
    b = b.reshape(b.shape[:-1] + (1,)*sample_dims + (2,))
    return (1.0 + (amplitude*a).sum(-1))*torch.exp(1j*(phase*b).sum(-1))


def _condition(matrix, maximum, name):
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} contains nonfinite values")
    singular = torch.linalg.svdvals(matrix)
    condition = singular[..., 0]/singular[..., -1]
    numerical_limit = 1.0/(torch.finfo(matrix.dtype).eps*max(matrix.shape[-2:]))
    if bool((~torch.isfinite(condition) | (condition > min(maximum, numerical_limit))).any()):
        raise ValueError(f"{name} is rank deficient or ill-conditioned")
    return condition


def _matching_solve(matrix, rhs, maximum, name):
    # Row and column equilibration prevents units or basis size alone from
    # driving the diagnostic. No ridge, pseudoinverse, or coefficient clipping.
    row_scale = torch.linalg.vector_norm(matrix, dim=-1)
    if bool((row_scale == 0).any()):
        raise ValueError(f"{name} is rank deficient (zero matching row)")
    scaled = matrix/row_scale.unsqueeze(-1)
    column_scale = torch.linalg.vector_norm(scaled, dim=-2)
    if bool((column_scale == 0).any()):
        raise ValueError(f"{name} is rank deficient (zero basis column)")
    scaled = scaled/column_scale.unsqueeze(-2)
    condition = _condition(scaled, maximum, name)
    solution = torch.linalg.solve(scaled, (rhs/row_scale).unsqueeze(-1)).squeeze(-1)/column_scale
    residual = (matrix@solution.unsqueeze(-1)).squeeze(-1) - rhs
    return solution, condition, residual


def calibrate_nqc_22(time, r, pr_star, orbital_frequency, amplitude, gw_frequency,
                     match_time, targets, *, degree=4, max_condition=1.0e10):
    """Fit NQC coefficients from a caller-selected local trajectory stencil.

    Samples occupy the last axis; preceding batch axes, scalar matching times,
    and target batch axes broadcast. Supply the *uncorrected* positive mode
    amplitude and positive frequency -d(arg h22)/d tau. Targets may describe a
    general matching point; the published peak targets have A'=0.

    A least-squares polynomial of the specified degree is fitted to A, each
    A*q_i, each phase basis p_i, and the base GW frequency on the entire given
    stencil. Its derivatives at match_time define the two matching systems.
    The caller must choose a sufficiently local and resolved stencil and check
    sampling convergence: the paper does not prescribe this interpolation.
    No peak selection or extrapolation is implicit. Derivatives and the solves
    retain autograd dependence on successful, well-conditioned input batches.
    """
    _real_tensors(time, r, pr_star, orbital_frequency, amplitude, gw_frequency,
                  match_time, *targets)
    if not isinstance(degree, int) or degree < 2:
        raise ValueError("local polynomial degree must be an integer >= 2")
    if not 1.0 <= max_condition < float("inf"):
        raise ValueError("max_condition must be finite and >= 1")
    if time.ndim < 1 or time.shape[-1] < degree + 1:
        raise ValueError("time must contain at least degree+1 samples on its last axis")
    count = time.shape[-1]
    samples = torch.broadcast_tensors(time, r, pr_star, orbital_frequency, amplitude, gw_frequency)
    if samples[0].shape[-1] != count:
        raise ValueError("trajectory samples must share time's last-axis length")
    batch = torch.broadcast_shapes(samples[0].shape[:-1], match_time.shape,
                                   *(target.shape for target in targets))
    time, r, pr_star, orbital_frequency, amplitude, gw_frequency = (
        sample.expand(batch + (count,)) for sample in samples)
    match_time = match_time.expand(batch)
    targets = NQCMatchingTargets(*(target.expand(batch) for target in targets))
    if bool((time.diff(dim=-1) <= 0).any()):
        raise ValueError("sample times must be strictly increasing")
    if bool(((match_time < time[..., 0]) | (match_time > time[..., -1])).any()):
        raise ValueError("match_time must lie inside each supplied stencil")
    if bool(((amplitude <= 0) | (gw_frequency <= 0)).any()) or bool(
            ((targets.amplitude <= 0) | (targets.frequency <= 0)).any()):
        raise ValueError("base and target amplitudes and GW frequencies must be positive")

    q, p = nqc_22_basis(r, pr_star, orbital_frequency)
    span = time[..., -1] - time[..., 0]
    x = (time - match_time.unsqueeze(-1))/span.unsqueeze(-1)
    vandermonde = torch.stack(tuple(x.pow(k) for k in range(degree + 1)), dim=-1)
    sampling_condition = _condition(vandermonde, max_condition, "local sample polynomial")
    ordinates = torch.cat((amplitude.unsqueeze(-1), amplitude.unsqueeze(-1)*q,
                           p, gw_frequency.unsqueeze(-1)), dim=-1)
    polynomial = torch.linalg.lstsq(vandermonde, ordinates, driver="gels").solution
    value = polynomial[..., 0, :]
    first = polynomial[..., 1, :]/span.unsqueeze(-1)
    second = 2.0*polynomial[..., 2, :]/span.square().unsqueeze(-1)
    if bool((value[..., 0] <= 0).any()):
        raise ValueError("interpolated base amplitude is nonpositive at match_time")

    amplitude_matrix = torch.stack((value[..., 1:4], first[..., 1:4], second[..., 1:4]), dim=-2)
    amplitude_rhs = torch.stack(targets[:3], dim=-1) - torch.stack(
        (value[..., 0], first[..., 0], second[..., 0]), dim=-1)
    phase_matrix = torch.stack((first[..., 4:6], second[..., 4:6]), dim=-2)
    # h=A exp(-i phi), N=N_A exp(+i b.p): omega_corrected=omega_base-b.p'.
    phase_rhs = torch.stack((value[..., 6] - targets.frequency,
                             first[..., 6] - targets.frequency_derivative), dim=-1)
    a, amplitude_condition, amplitude_residual = _matching_solve(
        amplitude_matrix, amplitude_rhs, max_condition, "amplitude matching system")
    b, phase_condition, phase_residual = _matching_solve(
        phase_matrix, phase_rhs, max_condition, "phase matching system")
    return NQCCalibration(NQCCoefficients(a, b), sampling_condition,
                          amplitude_condition, phase_condition,
                          amplitude_residual, phase_residual)

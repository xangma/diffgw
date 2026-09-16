"""Primary endpoint checks and independent synthetic-signal NQC matching."""

import math

import pytest
import torch

from torchwave.approximants.eob.nqc import (
    NQCCoefficients,
    NQCMatchingTargets,
    calibrate_nqc_22,
    nqc_22_factor,
    seobnrv4_nqc_parameters,
    seobnrv4_nqc_targets,
    seobnrv4_nqc_time_offset,
)


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def derivative(value, time):
    return torch.autograd.grad(value, time, create_graph=True)[0]


def synthetic_signal(time, a, b, parameters=None):
    """An analytic complex signal; its amplitude and phase are polynomials.

    This does not call the production NQC basis, factor, derivatives or solver.
    The synthetic trajectory is chosen to permit exact polynomial recovery.
    """
    if parameters is None:
        parameters = tensor([0.6, -0.3, -0.15, 0.3, 0.4])
    u0, p0, x0, amplitude, omega0 = parameters
    u = u0 + 0.12*time
    p = p0 + 0.04*time
    x = x0 + 0.025*time
    corrected_amplitude = amplitude*(1 + x*x*(a[0] + a[1]*u*u + a[2]*u*u*u))
    phase = -omega0*time - 0.01*time*time - (0.0001/3)*time**3 + x*(b[0] + b[1]*p*p)
    return corrected_amplitude*torch.exp(1j*phase)


def signal_targets(match_time, a, b):
    time = match_time.clone().requires_grad_(True)
    h = synthetic_signal(time, a, b)
    amplitude = h.abs()
    amplitude_prime = derivative(amplitude, time)
    h_prime = derivative(h.real, time) + 1j*derivative(h.imag, time)
    frequency = -(h_prime/h).imag
    return NQCMatchingTargets(amplitude, amplitude_prime, derivative(amplitude_prime, time),
                              frequency, derivative(frequency, time))


def trajectory(time, parameters=None):
    if parameters is None:
        parameters = tensor([0.6, -0.3, -0.15, 0.3, 0.4])
    u0, p0, x0, amplitude, omega0 = parameters
    r = (u0 + 0.12*time).pow(-2)
    p = p0 + 0.04*time
    orbital_frequency = p/(r*(x0 + 0.025*time))
    return r, p, orbital_frequency, amplitude*torch.ones_like(time), omega0 + 0.02*time + 0.0001*time*time


def fixture():
    time = torch.linspace(-2, 2, 9, dtype=torch.float64)
    a, b, match_time = tensor([0.4, -0.2, 0.1]), tensor([-0.3, 0.25]), tensor(0.13)
    targets = signal_targets(match_time, a, b)
    return time, a, b, match_time, targets


def test_primary_published_target_endpoints():
    # Appendix A prints these normalized, nonspinning endpoint amplitudes and
    # curvatures. The curvature TPL value evaluates A3 at chi=0.
    targets = seobnrv4_nqc_targets(tensor([0.001, 0.25]), tensor(0.0))
    torch.testing.assert_close(targets.amplitude, tensor([0.001*1.452857, 0.25*1.577458]))
    torch.testing.assert_close(targets.amplitude_second_derivative,
                               tensor([0.001*(-0.00229169), 0.25*(-0.00412651)]))
    torch.testing.assert_close(targets.amplitude_derivative, tensor([0.0, 0.0]))
    torch.testing.assert_close(targets.frequency_derivative,
                               tensor([-0.011210 + 0.004087*math.log(68.474666), 0.011282]))
    # A9 explicitly anchors the frequency's TPL endpoint at nu=0, unlike A2/A5/A12.
    frequency = seobnrv4_nqc_targets(tensor([0.0, 0.25]), tensor(0.0)).frequency
    torch.testing.assert_close(frequency,
                               tensor([0.562679 - 0.087062*math.log(25.850378),
                                       0.562679 - 0.087062*math.log(10.262073)]))


def test_mass_exchange_scaling_and_equal_mass_spin_symmetry():
    m1, m2 = tensor([30.0, 12.0, 15.0]), tensor([10.0, 9.0, 15.0])
    chi1, chi2 = tensor([0.7, -0.8, 0.9]), tensor([-0.2, 0.6, -0.5])
    parameters = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    for changed in (seobnrv4_nqc_parameters(m2, m1, chi2, chi1),
                    seobnrv4_nqc_parameters(1e4*m1, 1e4*m2, chi1, chi2)):
        for original, new in zip(parameters, changed):
            torch.testing.assert_close(original, new)
        for original, new in zip(seobnrv4_nqc_targets(*parameters), seobnrv4_nqc_targets(*changed)):
            torch.testing.assert_close(original, new)
        torch.testing.assert_close(seobnrv4_nqc_time_offset(*parameters), seobnrv4_nqc_time_offset(*changed))
    torch.testing.assert_close(parameters[1][-1], tensor(0.2))
    # Eqs. (4.9), (4.15), zero-spin test-particle limit, retaining the offset sign.
    torch.testing.assert_close(seobnrv4_nqc_time_offset(tensor(0.0), tensor(0.0)), tensor(-2.504992))


def test_factor_circular_limit_and_radial_momentum_reversal():
    r, omega, p = tensor([5.0, 8.0]), tensor([0.09, 0.06]), tensor([-0.2, -0.1])
    coefficients = NQCCoefficients(tensor([0.4, -0.2, 0.1]), tensor([-0.3, 0.25]))
    torch.testing.assert_close(nqc_22_factor(r, torch.zeros_like(p), omega, coefficients),
                               torch.ones(2, dtype=torch.complex128))
    torch.testing.assert_close(nqc_22_factor(r, -p, omega, coefficients),
                               nqc_22_factor(r, p, omega, coefficients).conj())


def test_independent_complex_signal_recovers_coefficients_and_five_conditions():
    time, a, b, match_time, targets = fixture()
    result = calibrate_nqc_22(time, *trajectory(time), match_time, targets, degree=5)
    torch.testing.assert_close(result.coefficients.amplitude, a, rtol=2e-8, atol=2e-9)
    torch.testing.assert_close(result.coefficients.phase, b, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(result.amplitude_residual, torch.zeros(3, dtype=time.dtype), atol=1e-15, rtol=0)
    torch.testing.assert_close(result.phase_residual, torch.zeros(2, dtype=time.dtype), atol=1e-15, rtol=0)

    # Verify all five conditions by differentiating the *complex corrected mode*.
    t = match_time.clone().requires_grad_(True)
    r, p, omega, amplitude, _ = trajectory(t)
    base = amplitude*torch.exp(-1j*(0.4*t + 0.01*t*t + (0.0001/3)*t**3))
    h = base*nqc_22_factor(r, p, omega, result.coefficients)
    a_value = h.abs()
    a_prime = derivative(a_value, t)
    h_prime = derivative(h.real, t) + 1j*derivative(h.imag, t)
    w = -(h_prime/h).imag
    actual = (a_value, a_prime, derivative(a_prime, t), w, derivative(w, t))
    for found, target in zip(actual, targets):
        torch.testing.assert_close(found, target, rtol=2e-8, atol=2e-12)


def test_batch_matches_scalar_and_amplitude_normalization_cancels():
    time, a, b, match_time, targets = fixture()
    scales = tensor([1.0, 7.0])
    r, p, omega, amplitude, frequency = trajectory(time)
    batch_targets = NQCMatchingTargets(*(value*scales if index < 3 else value.expand(2)
                                         for index, value in enumerate(targets)))
    result = calibrate_nqc_22(time, r, p, omega, scales[:, None]*amplitude,
                              frequency, match_time, batch_targets, degree=5)
    torch.testing.assert_close(result.coefficients.amplitude, a.expand(2, 3), rtol=2e-8, atol=2e-9)
    torch.testing.assert_close(result.coefficients.phase, b.expand(2, 2), rtol=2e-9, atol=2e-10)
    batch_factor = nqc_22_factor(r, p, omega, result.coefficients, sample_dims=1)
    for index in range(2):
        scalar_targets = NQCMatchingTargets(*(value[index] for value in batch_targets))
        scalar = calibrate_nqc_22(time, r, p, omega, scales[index]*amplitude,
                                  frequency, match_time, scalar_targets, degree=5)
        torch.testing.assert_close(result.coefficients.amplitude[index], scalar.coefficients.amplitude)
        torch.testing.assert_close(batch_factor[index], nqc_22_factor(r, p, omega, scalar.coefficients))


def test_known_peak_is_recovered_from_uncorrected_signal_samples():
    # Independently choose a quadratic amplitude maximum and a linear chirp.
    # Remove an analytic NQC correction to construct the baseline samples.
    # Recovering it must restore the known peak, including its zero derivative.
    match_time = tensor(0.13)
    time = match_time + torch.linspace(-0.1, 0.1, 9, dtype=torch.float64)
    a, b = tensor([0.4, -0.2, 0.1]), tensor([-0.3, 0.25])
    r, p, omega, _, _ = trajectory(time)
    u, x, dt = 0.6 + 0.12*time, -0.15 + 0.025*time, time - match_time
    multiplier = 1 + x*x*(a[0] + a[1]*u*u + a[2]*u*u*u)
    baseline_amplitude = (0.31 - 0.0008*dt*dt)/multiplier
    phase_prime = 0.025*(b[0] + b[1]*p*p) + x*(2*b[1]*p*0.04)
    baseline_frequency = 0.42 + 0.03*dt + phase_prime
    targets = NQCMatchingTargets(*(tensor(value) for value in (0.31, 0.0, -0.0016, 0.42, 0.03)))
    recovered = calibrate_nqc_22(time, r, p, omega, baseline_amplitude,
                                 baseline_frequency, match_time, targets, degree=5)
    torch.testing.assert_close(recovered.coefficients.amplitude, a, rtol=3e-6, atol=3e-7)
    torch.testing.assert_close(recovered.coefficients.phase, b, rtol=2e-8, atol=2e-9)


def test_time_translation_and_stencil_refinement():
    time, a, b, match_time, targets = fixture()
    shifted = calibrate_nqc_22(time + 100, *trajectory(time), match_time + 100, targets, degree=5)
    torch.testing.assert_close(shifted.coefficients.amplitude, a, rtol=2e-8, atol=2e-9)
    # Degree four omits a genuine fifth-degree amplitude term. A narrower
    # stencil must approach the analytic coefficient solution.
    errors = []
    for half_width in (1.0, 0.5, 0.25):
        local_time = match_time + torch.linspace(-half_width, half_width, 9, dtype=time.dtype)
        solved = calibrate_nqc_22(local_time, *trajectory(local_time), match_time, targets, degree=4)
        errors.append(torch.linalg.vector_norm(solved.coefficients.amplitude - a))
    assert errors[1] < errors[0]/3
    assert errors[2] < errors[1]/3


def test_calibration_gradients_include_trajectory_inputs():
    time, _, _, match_time, targets = fixture()
    targets = NQCMatchingTargets(*(value.detach() for value in targets))
    parameters = tensor([0.6, -0.3, -0.15, 0.3, 0.4]).requires_grad_(True)

    def solve(values):
        result = calibrate_nqc_22(time, *trajectory(time, values), match_time, targets, degree=5)
        return torch.cat(result.coefficients)

    assert torch.autograd.gradcheck(solve, (parameters,), eps=1e-6, atol=2e-5, rtol=2e-4)
    fit_nu, fit_chi = tensor(0.18).requires_grad_(True), tensor(0.3).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda n, c: (*seobnrv4_nqc_targets(n, c),
                                                seobnrv4_nqc_time_offset(n, c)),
                                    (fit_nu, fit_chi))


@pytest.mark.parametrize("failure", ["circular", "duplicate_time", "outside", "conditioning", "nonfinite"])
def test_invalid_or_unidentifiable_calibration_is_rejected(failure):
    time, _, _, match_time, targets = fixture()
    r, p, omega, amplitude, frequency = trajectory(time)
    maximum = 1e10
    if failure == "circular":
        p = torch.zeros_like(p)
    elif failure == "duplicate_time":
        time[1] = time[0]
    elif failure == "outside":
        match_time = tensor(4.0)
    elif failure == "conditioning":
        maximum = 1.01
    elif failure == "nonfinite":
        amplitude[0] = torch.nan
    with pytest.raises(ValueError):
        calibrate_nqc_22(time, r, p, omega, amplitude, frequency, match_time, targets,
                         degree=5, max_condition=maximum)

"""Independent matching and asymptotic tests for the published merger ansatz."""

import pytest
import torch

from torchwave.approximants.eob.merger_ringdown import (
    MergerRingdownShape,
    merger_ringdown_22,
    seobnrv4_merger_shape,
    seobnrv4_peak_preserving_shape,
)


def real(value):
    return torch.tensor(value, dtype=torch.float64)


def data():
    return (torch.tensor(.52 - .08j, dtype=torch.complex128), real(.36),
            real(0.), real(.31), real(.7), seobnrv4_merger_shape(real(.2), real(.3)))


def complex_derivative(value, variable):
    return (torch.autograd.grad(value.real, variable, create_graph=True, retain_graph=True)[0]
            + 1j*torch.autograd.grad(value.imag, variable, create_graph=True)[0])


def test_printed_fit_test_particle_nonspinning_values():
    # Constant terms in Eqs. (2.13),(2.14),(2.16),(2.17).
    actual = seobnrv4_merger_shape(real(0.), real(0.))
    torch.testing.assert_close(torch.stack(actual), real([.0830664, -.623953, .147584, 2.46654]), rtol=0, atol=0)


@pytest.mark.parametrize("amplitude_derivative", [0., -.02, .03])
def test_complex_value_and_first_derivative_match_supplied_inspiral(amplitude_derivative):
    sigma, amplitude, _, frequency, phase, shape = data()
    slope = real(amplitude_derivative)
    t = real([0.]).requires_grad_(True)
    h = merger_ringdown_22(t, sigma, amplitude, slope, frequency, phase, shape)[0]
    expected = amplitude*torch.exp(-1j*phase)
    torch.testing.assert_close(h, expected, rtol=1e-15, atol=1e-16)
    derivative = complex_derivative(h, t)[0]
    # Directly differentiating A exp(-i phi) independently fixes both signs.
    torch.testing.assert_close(derivative, (slope/amplitude - 1j*frequency)*expected,
                               rtol=2e-14, atol=1e-16)


def test_late_signal_approaches_input_qnm_and_monotonic_frequency():
    sigma, amplitude, slope, frequency, phase, shape = data()
    t = real([0., 1., 5., 20., 100., 400.]).requires_grad_(True)
    h = merger_ringdown_22(t, sigma, amplitude, slope, frequency, phase, shape)
    logarithmic_derivative = complex_derivative(h.sum(), t)/h
    instantaneous_frequency = -logarithmic_derivative.imag
    assert torch.all(instantaneous_frequency.diff() > 0)
    torch.testing.assert_close(logarithmic_derivative[-1], -1j*sigma, rtol=3e-13, atol=2e-15)


def test_broadcasting_matches_scalar_and_amplitude_units_cancel():
    sigma, amplitude, slope, frequency, phase, _ = data()
    nu, chi = real([[.1], [.24]]), real([-.5, 0., .8])
    shape = seobnrv4_merger_shape(nu, chi)
    t = torch.linspace(0, 70, 19, dtype=torch.float64)
    h = merger_ringdown_22(t, sigma, amplitude, slope, frequency, phase, shape)
    assert h.shape == (2, 3, 19)
    for i in range(2):
        for j in range(3):
            scalar = merger_ringdown_22(t, sigma, amplitude, slope, frequency, phase,
                                         seobnrv4_merger_shape(nu[i, 0], chi[j]))
            torch.testing.assert_close(h[i, j], scalar)
    scaled = merger_ringdown_22(t, sigma, amplitude*7, slope*7, frequency, phase, shape)
    torch.testing.assert_close(scaled, h*7, rtol=2e-15, atol=2e-16)


def test_zero_phase_offset_has_analytic_limit_and_finite_gradient():
    sigma, amplitude, slope, frequency, phase, source_shape = data()
    t = real([0., 1., 20.])
    d2 = real(0.).requires_grad_(True)
    shape = MergerRingdownShape(*source_shape[:3], d2)
    h = merger_ringdown_22(t, sigma, amplitude, slope, frequency, phase, shape)
    # At d2=0, the residual phase is (omega_QNM-omega0)/d1 * (exp(-d1*t)-1).
    expected_phase = phase + sigma.real*t + (sigma.real-frequency)/shape.phase_rate*torch.expm1(-shape.phase_rate*t)
    torch.testing.assert_close(h/h.abs(), torch.exp(-1j*expected_phase), rtol=2e-15, atol=2e-15)
    assert torch.autograd.gradcheck(lambda value: merger_ringdown_22(
        t, sigma, amplitude, slope, frequency, phase,
        MergerRingdownShape(*source_shape[:3], value)), (d2,))


def test_gradients_include_qnm_matching_and_intrinsic_parameters():
    t = real([0., .2, 5., 50.])
    parameters = real([.52, .08, .36, .0, .31, .7, .2, .3]).requires_grad_(True)

    def evaluate(values):
        w, damping, amplitude, slope, frequency, phase, nu, chi = values
        return merger_ringdown_22(t, torch.complex(w, -damping), amplitude, slope,
                                   frequency, phase, seobnrv4_merger_shape(nu, chi))

    assert torch.autograd.gradcheck(evaluate, (parameters,), rtol=2e-4, atol=2e-6)


def test_peak_preserving_continuation_fixes_only_rising_amplitude_rate():
    # The q=8, chi=.8 state samples a printed-fit region where a sufficiently
    # slow fundamental decay makes the C1 stationary join bend upward.
    raw = seobnrv4_merger_shape(real(8/81), real(.8))
    sigma = torch.tensor(.52-.04j, dtype=torch.complex128)
    adjusted = seobnrv4_peak_preserving_shape(sigma, raw)
    k = torch.tanh(raw.amplitude_offset)
    assert bool((.04 + 2*raw.amplitude_rate*k < 0))
    torch.testing.assert_close(adjusted.amplitude_rate, -.04/(2*k), rtol=2e-15, atol=0)
    for actual, original in zip(adjusted[1:], raw[1:]):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)

    t = real([0.]).requires_grad_()
    def amplitude(shape):
        return merger_ringdown_22(t, sigma, real(.36), real(0.), real(.31), real(.7), shape).abs()[0]
    raw_second = torch.autograd.grad(
        torch.autograd.grad(amplitude(raw), t, create_graph=True)[0], t)[0]
    assert raw_second > 0
    projected = amplitude(adjusted)
    first = torch.autograd.grad(projected, t, create_graph=True)[0]
    second = torch.autograd.grad(first, t, create_graph=True)[0]
    third = torch.autograd.grad(second, t)[0]
    torch.testing.assert_close(first, real([0.]), atol=2e-16, rtol=0)
    torch.testing.assert_close(second, real([0.]), atol=2e-16, rtol=0)
    assert third < 0

    damping = real(.04).requires_grad_()
    rate = seobnrv4_peak_preserving_shape(torch.complex(real(.52), -damping), raw).amplitude_rate
    torch.testing.assert_close(torch.autograd.grad(rate, damping)[0], -1/(2*k),
                               rtol=2e-15, atol=0)

    # The printed fit is left exactly intact when the damping already bends
    # the attachment downward.
    safe = seobnrv4_peak_preserving_shape(
        torch.tensor(.52-.12j, dtype=torch.complex128), raw)
    for actual, original in zip(safe, raw):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["negative_time", "growing_qnm", "bad_shape", "nonfinite"])
def test_invalid_domains_fail_explicitly(failure):
    sigma, amplitude, slope, frequency, phase, shape = data()
    t = real([0., 2.])
    if failure == "negative_time":
        t[0] = -1.
    elif failure == "growing_qnm":
        sigma = sigma.conj()
    elif failure == "bad_shape":
        shape = MergerRingdownShape(*shape[:3], real(-1.))
    else:
        amplitude = real(float("nan"))
    with pytest.raises(ValueError):
        merger_ringdown_22(t, sigma, amplitude, slope, frequency, phase, shape)

"""Continuous NQC/merger conditions checked through the attached complex mode."""

import pytest
import torch

from torchwave.approximants.eob.attachment import attach_seobnrv4_22
from torchwave.approximants.eob.nqc import seobnrv4_nqc_targets, seobnrv4_nqc_time_offset


def inputs(dt=.1):
    t = torch.arange(0, 32+dt/2, dt, dtype=torch.float64)
    r = 8-.1*t-.003*t.square()
    pr = -.01-.003*t-.0007*t.square()
    omega = .14-.00005*(t-25).square()
    state = torch.stack((r, t*.1, pr, t*0+3), -1)
    amp = .12+.004*t+.00002*t.square()
    frequency = .2+.005*t+.00004*t.square()
    phase = .3+.2*t+.0025*t.square()+(.00004/3)*t.pow(3)
    mode = amp*torch.exp(-1j*phase)
    nu, chi = torch.tensor(.22, dtype=t.dtype), torch.tensor(.1, dtype=t.dtype)
    sigma = torch.tensor(.5-.08j, dtype=torch.complex128)
    return t, state, omega, mode, frequency, nu, chi, sigma


def test_all_five_conditions_and_complex_c1_join_from_output_derivatives():
    args = inputs()
    nu, chi = args[-3:-1]
    targets = seobnrv4_nqc_targets(nu, chi)
    match = 25+seobnrv4_nqc_time_offset(nu, chi)
    epsilon = 1e-6
    points = torch.stack((match-epsilon, match, match+epsilon)).requires_grad_()
    result = attach_seobnrv4_22(*args, points)
    h = result.mode
    real_prime = torch.autograd.grad(h.real.sum(), points, create_graph=True, retain_graph=True)[0]
    imag_prime = torch.autograd.grad(h.imag.sum(), points, create_graph=True, retain_graph=True)[0]
    hprime = torch.complex(real_prime, imag_prime)
    amplitude_prime = torch.autograd.grad(h.abs().sum(), points, create_graph=True, retain_graph=True)[0]
    amplitude_second = torch.autograd.grad(amplitude_prime.sum(), points, retain_graph=True)[0]
    frequency = -(h.conj()*hprime).imag/h.abs().square()
    frequency_prime = torch.autograd.grad(frequency.sum(), points)[0]
    torch.testing.assert_close(result.match_time, match, atol=2e-13, rtol=0)
    torch.testing.assert_close(h.abs()[1], targets.amplitude, atol=2e-13, rtol=0)
    torch.testing.assert_close(amplitude_prime, torch.zeros_like(points), atol=2e-8, rtol=0)
    torch.testing.assert_close(amplitude_second[0], targets.amplitude_second_derivative, atol=3e-8, rtol=0)
    torch.testing.assert_close(frequency, targets.frequency.expand_as(points), atol=2e-8, rtol=0)
    torch.testing.assert_close(frequency_prime[0], targets.frequency_derivative, atol=3e-8, rtol=0)
    torch.testing.assert_close(hprime[0], hprime[-1], atol=2e-7, rtol=0)


def test_attachment_time_origin_and_phase_covariance():
    args = list(inputs())
    query = torch.linspace(0, 100, 201, dtype=torch.float64)
    original = attach_seobnrv4_22(*args, query)
    phase = torch.tensor(.73, dtype=query.dtype, requires_grad=True)
    args[0] = args[0]+103
    args[3] = args[3]*torch.exp(-1j*phase)
    shifted = attach_seobnrv4_22(*args, query+103)
    torch.testing.assert_close(shifted.mode, original.mode*torch.exp(-1j*phase), atol=3e-10, rtol=3e-10)
    derivative = torch.autograd.grad(shifted.mode.real.sum(), phase)[0]
    torch.testing.assert_close(derivative, shifted.mode.imag.sum(), atol=3e-10, rtol=3e-10)


def test_attachment_batch_and_inference_equal_independent_calls():
    args = list(inputs())
    args[6] = torch.tensor([.1, -.2], dtype=torch.float64)
    query = torch.linspace(0, 70, 57, dtype=torch.float64)
    batch = attach_seobnrv4_22(*args, query)
    with torch.inference_mode():
        inference = attach_seobnrv4_22(*args, query)
    torch.testing.assert_close(batch.mode, inference.mode)
    for index in range(2):
        single_args = list(args)
        single_args[6] = args[6][index]
        single = attach_seobnrv4_22(*single_args, query)
        torch.testing.assert_close(batch.mode[index], single.mode, rtol=3e-10, atol=3e-10)


def test_attachment_insufficient_pre_peak_history_fails():
    args = list(inputs())
    for index in range(5):
        args[index] = args[index][-75:]
    with pytest.raises(ValueError, match="extend trajectory history"):
        attach_seobnrv4_22(*args, args[0])


def test_attachment_phase_at_match_converges_to_analytic_integral():
    # The input phase is an exactly integrable cubic and its supplied frequency
    # is quadratic. Subtract the independently evaluated phase-basis correction
    # at the match to check the integral sign and phase anchor.
    from torchwave.approximants.eob.nqc import nqc_22_basis
    errors = []
    for dt in (.4, .2, .1):
        args = inputs(dt)
        result = attach_seobnrv4_22(*args, args[0])
        t = result.match_time
        r = 8-.1*t-.003*t.square()
        pr = -.01-.003*t-.0007*t.square()
        om = .14-.00005*(t-25).square()
        _, basis = nqc_22_basis(r, pr, om)
        phase = .3+.2*t+.0025*t.square()+(.00004/3)*t.pow(3)
        expected = phase-(basis*result.calibration.coefficients.phase).sum()
        error = torch.angle(torch.exp(1j*(result.match_phase-expected))).abs().item()
        errors.append(error)
    assert errors[-1] < 2e-8
    assert errors[0] > 50*errors[-1]

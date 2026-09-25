"""Analytic phase/time gauge controls for development-only ROM inputs."""

import numpy as np
import pytest

from dev.native_rom.rom.conditioning import condition_modes, condition_modes_windowed
from dev.native_rom.rom.reduced_interpolant import ReducedTensorInterpolant


def _nodes(low, high, count):
    return (low + high)/2 - (high-low)/2 * np.cos(np.pi*np.arange(count)/(count-1))


def _analytic_mode(q, a, b, frequency, f_ref):
    x = frequency - f_ref
    amp = .03 + (q + a*b)*frequency + (q**2-a)*frequency**2
    intrinsic_phase = (q-a)*x**2 + (b+q*a)*x**3
    extrinsic_phase = 1.7*a - .8*b + (4*q-2*b)*x
    return np.exp(amp + 1j*(intrinsic_phase + extrinsic_phase)), amp, intrinsic_phase


def test_conditioning_removes_arbitrary_phase_and_time_gauge():
    frequency = np.linspace(.01, .2, 65)
    f_ref = .08
    mode, amp, intrinsic_phase = _analytic_mode(
        np.array([2., 4.])[:, None], np.array([-.3, .4])[:, None],
        np.array([.5, -.2])[:, None], frequency[None, :], f_ref)
    first = condition_modes(frequency, mode, f_ref)
    shifted = condition_modes(frequency, mode * np.exp(
        1j*(2.3 + 15*(frequency-f_ref))), f_ref)
    np.testing.assert_allclose(first.log_amplitude, amp, rtol=0, atol=2e-15)
    np.testing.assert_allclose(first.phase, intrinsic_phase, rtol=0, atol=2e-12)
    np.testing.assert_allclose(shifted.phase, first.phase, rtol=0, atol=2e-12)


def test_conditioned_complex_holdout_is_independent_of_training_tensor():
    grid = (_nodes(1., 6., 4), _nodes(-.8, .8, 4), _nodes(-.8, .8, 4))
    frequency = np.linspace(.01, .2, 65)
    f_ref = .08
    q, a, b, f = np.meshgrid(*grid, frequency, indexing="ij")
    modes, _, _ = _analytic_mode(q, a, b, f, f_ref)
    conditioned = condition_modes(frequency, modes, f_ref)
    model = ReducedTensorInterpolant.fit(
        grid, frequency, conditioned.log_amplitude, conditioned.phase,
        ranks=(2, 2))
    holdout = (2.17, -.37, .41)
    mode, _, _ = _analytic_mode(*holdout, frequency, f_ref)
    expected = condition_modes(frequency, mode, f_ref)
    actual = model.evaluate(*holdout)
    np.testing.assert_allclose(
        actual, np.exp(expected.log_amplitude + 1j*expected.phase),
        rtol=2e-10, atol=2e-10)


def test_conditioning_rejects_unresolved_input_shape_and_zero_amplitude():
    frequency = np.linspace(.01, .2, 65)
    with pytest.raises(ValueError, match="share the frequency grid"):
        condition_modes(frequency, np.ones(64, dtype=complex), .08)
    invalid = np.ones(65, dtype=complex)
    invalid[10] = 0
    with pytest.raises(ValueError, match="nonzero"):
        condition_modes(frequency, invalid, .08)
    with pytest.raises(ValueError, match="reference frequency"):
        condition_modes(frequency, np.ones(65, dtype=complex), .21)


def test_windowed_gauge_removes_independent_phase_and_time_shifts():
    frequency = np.linspace(90., 110., 641)
    x = frequency - 100.
    phase = np.stack((.3*x**2 + .001*x**3, -.2*x**2 + .002*x**3))
    modes = np.exp(.001*x + 1j*phase)
    shifted = modes * np.exp(1j*np.stack((.7 + .03*x, -.4 - .02*x)))
    baseline = condition_modes_windowed(frequency, modes, 100., half_width_hz=2.)
    candidate = condition_modes_windowed(frequency, shifted, 100., half_width_hz=2.)
    np.testing.assert_allclose(candidate.log_amplitude, baseline.log_amplitude, atol=1e-15)
    np.testing.assert_allclose(candidate.phase, baseline.phase, atol=3e-12)


def test_windowed_gauge_requires_resolved_two_sided_window():
    frequency = np.linspace(99., 101., 65)
    with pytest.raises(ValueError, match="eight nodes on each side"):
        condition_modes_windowed(frequency, np.ones(65, dtype=complex),
                                 100., half_width_hz=.1)

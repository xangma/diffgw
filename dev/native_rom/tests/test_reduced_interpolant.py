"""Independent analytic holdouts for the development-only ROM algebra."""

import numpy as np
import pytest

from dev.native_rom.rom.reduced_interpolant import ReducedTensorInterpolant


def nodes(low, high, count):
    return (low + high) / 2 - (high - low) / 2 * np.cos(
        np.pi * np.arange(count) / (count - 1))


def analytic_spectrum(q, chi1, chi2, frequency):
    f = frequency
    amp = (.01 + f * (q + chi1 * chi2)
           + f**2 * (q**2 - chi1) + np.sin(f) * chi2**2)
    phase = f * (q**2 + chi1) + f**3 * (chi2 + q * chi1)
    return amp, phase


def fitted_model(ranks=(3, 2)):
    grid = (nodes(1., 8., 5), nodes(-.8, .8, 4), nodes(-.8, .8, 4))
    frequency = np.linspace(.01, .15, 24)
    q, chi1, chi2, f = np.meshgrid(*grid, frequency, indexing="ij")
    amp, phase = analytic_spectrum(q, chi1, chi2, f)
    return ReducedTensorInterpolant.fit(grid, frequency, amp, phase, ranks=ranks)


def test_off_grid_waveform_and_parameter_derivatives():
    model = fitted_model()
    assert model.log_amplitude.training_relative_error < 2e-13
    assert model.phase.training_relative_error < 2e-13

    # Every row is withheld from the training tensor, including domain edges.
    q = np.array([2.3, 7.1, 1., 8.])
    chi1 = np.array([-.43, .59, -.8, .8])
    chi2 = np.array([.32, -.61, .8, -.8])
    waveform, jacobian = model.evaluate(q, chi1, chi2, derivatives=True)
    amp, phase = analytic_spectrum(q[:, None], chi1[:, None], chi2[:, None],
                                   model.frequency[None, :])
    expected = np.exp(amp + 1j * phase)
    np.testing.assert_allclose(waveform, expected, rtol=2e-12, atol=2e-12)

    f = model.frequency[None, :]
    damp = (f + 2*f**2*q[:, None],
            f*chi2[:, None] - f**2,
            f*chi1[:, None] + 2*np.sin(f)*chi2[:, None])
    dphase = (2*f*q[:, None] + f**3*chi1[:, None],
              f + f**3*q[:, None], f**3 + np.zeros_like(expected.real))
    for axis in range(3):
        np.testing.assert_allclose(
            jacobian[:, axis, :], expected * (damp[axis] + 1j*dphase[axis]),
            rtol=3e-11, atol=3e-11)


def test_rank_truncation_is_measured_and_out_of_domain_is_rejected():
    full = fitted_model()
    truncated = fitted_model(ranks=(1, 1))
    assert truncated.log_amplitude.training_relative_error > full.log_amplitude.training_relative_error
    assert truncated.phase.training_relative_error > full.phase.training_relative_error
    with pytest.raises(ValueError, match="outside"):
        full.evaluate(8.01, 0., 0.)
    with pytest.raises(ValueError, match="finite"):
        full.evaluate(np.nan, 0., 0.)


def test_non_chebyshev_grid_and_invalid_training_fields_are_rejected():
    grid = (np.linspace(1., 8., 5), nodes(-.8, .8, 4), nodes(-.8, .8, 4))
    frequency = np.linspace(.01, .15, 24)
    amp = np.zeros((5, 4, 4, 24))
    with pytest.raises(ValueError, match="Chebyshev"):
        ReducedTensorInterpolant.fit(grid, frequency, amp, amp, ranks=(1, 1))
    grid = (nodes(1., 8., 5), *grid[1:])
    amp[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ReducedTensorInterpolant.fit(grid, frequency, amp, amp, ranks=(1, 1))

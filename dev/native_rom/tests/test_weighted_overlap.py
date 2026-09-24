"""Numerical controls for the development-only frequency-domain match."""

import numpy as np
import pytest

from dev.native_rom.measure_sxs_2508_weighted import optimized_overlap


def test_time_and_phase_are_recovered_without_amplitude_fit():
    frequency = np.linspace(0.01, 0.2, 1001)
    reference = np.exp(-((frequency - 0.08) / 0.04)**2) * np.exp(2j * frequency**2)
    candidate = 1.3 * reference * np.exp(2j * np.pi * frequency * 3.37 + 0.8j)
    psd = 1 + 7 * frequency
    result = optimized_overlap(frequency, candidate, reference, psd,
                               bounds=(-10, 10), step=0.25)
    assert result["overlap"] == pytest.approx(1, abs=1e-12)
    assert result["best_lag_M"] == pytest.approx(3.37, abs=1e-5)
    assert not result["time_shift_at_boundary"]


def test_shape_change_is_not_absorbed_by_time_and_phase():
    frequency = np.linspace(0.01, 0.2, 1001)
    reference = np.exp(-((frequency - 0.08) / 0.04)**2)
    candidate = reference * np.exp(300j * (frequency - 0.08)**2)
    result = optimized_overlap(frequency, candidate, reference, np.ones_like(frequency),
                               bounds=(-10, 10), step=0.25)
    assert result["overlap"] < 0.99

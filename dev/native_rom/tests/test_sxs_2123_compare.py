"""Check that the NR diagnostic's alignment does not fit the pair residual."""

import numpy as np
import pytest

from dev.native_rom.compare_sxs_2123 import anchor, compare
from dev.native_rom.optimize_sxs_2123_alignment import optimized_pair


def test_fixed_frequency_anchor_removes_coordinate_offsets_only():
    time = np.arange(0., 500.25, 0.25)
    phase = -(0.08*time + 0.00005*time**2)
    amplitude = np.exp(-0.5*((time-360.)/90.)**2)
    mode = amplitude*np.exp(1j*phase)
    reference = anchor(dict(time=time, mode=mode))
    shifted = anchor(dict(time=time+7., mode=mode*np.exp(0.4j)))
    assert reference["anchor_time_M"] == pytest.approx(100., abs=1e-3)
    result = compare(shifted, reference, {"window": [-100., 100.]}, 0.25)
    assert result["metrics"]["complex_relative_l2"] < 1e-12
    assert abs(result["peak_lag_difference_M"]) < 1e-12
    scaled = anchor(dict(time=time, mode=mode*1.2))
    result = compare(scaled, reference, {"window": [-100., 100.]}, 0.25)
    assert result["peak_amplitude_ratio"] == pytest.approx(1.2)
    assert result["metrics"]["amplitude_relative_l2"] == pytest.approx(0.2)


def test_secondary_optimizer_recovers_only_global_time_and_phase():
    time = np.arange(0., 500.25, 0.25)
    phase = -(0.08*time + 0.00005*time**2)
    mode = np.exp(-0.5*((time-360.)/90.)**2)*np.exp(1j*phase)
    reference = anchor(dict(time=time, mode=mode))
    candidate = dict(reference, time=reference["time"]+2.,
                     mode=reference["mode"]*1.2*np.exp(0.3j))
    result = optimized_pair(candidate, reference, window=(-200., 100.),
                            dt=0.25, lag_bounds=(-10., 10.))
    assert result["best_lag_M"] == pytest.approx(2., abs=1e-5)
    assert result["best_phase_rad"] == pytest.approx(-0.3, abs=1e-5)
    assert result["optimized_overlap"] == pytest.approx(1., abs=1e-12)
    assert result["time_phase_adjusted_metrics"]["amplitude_relative_l2"] == pytest.approx(0.2)

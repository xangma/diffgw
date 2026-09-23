"""Check that sub-sample anchoring resolves a known sampling bias."""

import numpy as np

from dev.native_rom.compare_whole_waveforms import aligned_metrics
from dev.native_rom.diagnose_peak_anchor import METHODS, fitted_metrics, fitted_peak


def test_local_fits_find_fractional_amplitude_peak():
    time = np.arange(-5, 6)*0.125
    mode = np.exp(-0.5*((time-0.03)/0.8)**2)*np.exp(-2j*time)
    for method in METHODS:
        assert abs(fitted_peak(time, mode, method)["time_s"]-0.03) < 0.001


def test_peak_fit_reduces_grid_origin_bias_for_same_waveform():
    native_time = np.arange(-100, 101)*0.125
    reference_time = native_time+0.06

    def waveform(time):
        return np.exp(-0.5*((time-0.03)/4)**2)*np.exp(-1j*(0.2*time+0.01*time**2))

    native, reference = waveform(native_time), waveform(reference_time)
    sampled = aligned_metrics(native_time, native, reference_time, reference, 1.)
    fitted = fitted_metrics(native_time, native, reference_time, reference, 1.,
                            "quadratic_power")
    assert sampled["metrics"]["complex_relative_l2"] > 0.005
    assert fitted["metrics"]["complex_relative_l2"] < 0.001

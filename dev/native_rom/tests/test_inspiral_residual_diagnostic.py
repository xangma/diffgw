"""Check the independent peak anchor and exact complex-error split."""

import numpy as np

from dev.native_rom.diagnose_inspiral_residual import residual_profile


def test_inspiral_residual_profile_recovers_known_phase_slope_and_amplitude():
    time = np.arange(-320., 151., 0.125)
    amplitude = np.exp(-(time/100.)**2)
    reference = amplitude*np.exp(-0.2j*time)
    candidate = 1.01*amplitude*np.exp(1j*(-0.2*time+0.001*time))

    result = residual_profile(time, candidate, time, reference, 1.)

    at_minus_100 = next(item for item in result["bins"] if item["center_M"] == -100.)
    mean_time = sum(at_minus_100["time_M"])/2
    assert abs(at_minus_100["phase_mean_rad"] - 0.001*mean_time) < 1e-7
    assert abs(at_minus_100["mode_frequency_difference_mean_per_M"] - 0.001) < 1e-7
    assert abs(at_minus_100["amplitude_ratio_mean"] - 1.01) < 1e-7
    for window in result["windows"].values():
        assert abs(window["amplitude_error_squared_fraction"]
                   + window["phase_error_squared_fraction"] - 1) < 1e-8

"""Control the phase-fit and amplitude-peak clocks independently."""

import numpy as np
import pytest

from dev.native_rom.measure_sxs_five_case_peak_timing import phase_aligned_peak_lag


def test_phase_alignment_exposes_independent_peak_delay():
    time = np.arange(0., 600.25, 0.25)
    def phase(t):
        return -(0.08 * t + 0.00005 * t**2)
    reference = dict(time=time,
                     mode=np.exp(-0.5 * ((time - 360.) / 80.)**2)
                     * np.exp(1j * phase(time)),
                     peak_after_anchor_M=360.)
    # The candidate phase is delayed by 3 M; its amplitude peak is delayed
    # by 8 M. One constant phase is free, but amplitude is not fitted.
    candidate = dict(time=time,
                     mode=np.exp(-0.5 * ((time - 368.) / 80.)**2)
                     * np.exp(1j * (phase(time - 3.) + 0.4)),
                     peak_after_anchor_M=368.)
    result = phase_aligned_peak_lag(candidate, reference, start=100.,
                                    duration=300., dt=0.25,
                                    bounds=(-10., 10.), step=0.25)
    assert result["fitted_time_shift_M"] == pytest.approx(3., abs=1e-4)
    assert result["phase_residual_rms_rad"] == pytest.approx(0., abs=1e-8)
    assert result["peak_lag_after_alignment_M"] == pytest.approx(5., abs=1e-4)
    assert not result["time_shift_at_boundary"]

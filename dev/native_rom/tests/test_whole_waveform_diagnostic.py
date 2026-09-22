"""Coordinate alignment checks; these do not establish waveform fidelity."""

import numpy as np
import pytest

from dev.native_rom.compare_whole_waveforms import aligned_metrics


def test_peak_anchor_removes_known_coordinate_offsets_but_preserves_amplitude_error():
    time = np.arange(-50., 50.01, .1)
    mode = np.exp(-(time/22)**2)*np.exp(-1j*(.1*time+.0002*time**2))
    equal = aligned_metrics(time, mode, time+4, mode*np.exp(1.2j), 1.)
    assert equal["metrics"]["complex_relative_l2"] < 1e-12
    assert equal["anchor"]["reference_amplitude_peak_s"] == pytest.approx(4.)
    assert set(equal["windows"]) == {"inspiral", "late_inspiral", "post_peak"}
    scaled = aligned_metrics(time, mode*1.1, time+4, mode*np.exp(1.2j), 1.)
    assert scaled["metrics"]["amplitude_relative_l2"] == pytest.approx(.1)


def test_peak_anchor_requires_common_nonzero_support():
    time = np.arange(10.)
    with pytest.raises(ValueError, match="nonzero complex waveforms"):
        aligned_metrics(time, np.zeros(10), time+100, np.ones(10), 1.)

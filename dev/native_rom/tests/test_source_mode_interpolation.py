"""Guard the full-mode diagnostic against complex-grid interpolation bias."""

import numpy as np

from dev.native_rom.diagnose_source_q4_q8_mode import _comparison


def test_phase_aware_interpolation_preserves_shifted_ringdown_amplitude():
    source_time = np.arange(-120.0, 101.0, 0.5)
    native_time = source_time + 0.19

    def mode(time):
        return np.exp(-0.072 * (time + 120.0)) * np.exp(-0.66j * time)

    comparison = _comparison(
        source_time, mode(source_time), native_time, mode(native_time), -100.0,
        [dict(name="ringdown", start_M=-120.0, end_M=101.0)],
    )["windows"]["ringdown"]
    componentwise = comparison["metrics"]["amplitude_relative_l2"]
    phase_aware = comparison["polar_interpolation_metrics"]["amplitude_relative_l2"]
    assert componentwise > 0.01
    assert phase_aware < 0.0002

"""The comparison sampler must preserve an oscillating mode's amplitude."""

import numpy as np

from dev.native_rom.diagnose_frequency_anchor_polar import polar_interpolate


def test_polar_interpolation_avoids_complex_chord_attenuation():
    time = np.arange(6, dtype=float)
    frequency = 0.4
    mode = np.exp(1j * frequency * time)
    query = time[:-1] + 0.5
    expected = np.exp(1j * frequency * query)
    sampled = polar_interpolate(query, time, mode)
    linear = np.interp(query, time, mode.real) + 1j * np.interp(
        query, time, mode.imag)
    np.testing.assert_allclose(sampled, expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.abs(linear), np.cos(frequency / 2),
                               rtol=0, atol=1e-15)

"""Analytic tests for the isolated complex NR33 phase-data preparation."""

from pathlib import Path

import numpy as np
import pytest

from scripts.fit_nr33_amplitude_pilot import FFTControls
from scripts.nr33_phase_data import (
    interpolate_complex_common_grid,
    load_complex_case,
    transform_complex,
)


FAST_CONTROLS = FFTControls(
    pad_factor=4, start_taper_duration=10.0, end_taper_duration=10.0,
    fmax=0.3, sample_dt=0.1,
)


def test_load_preserves_complex_mode_and_h22_peak(tmp_path: Path):
    time = np.arange(0.0, 100.0, 0.1)
    h33 = np.exp(-2j * np.pi * 0.05 * time) * (1 + 0.2j)
    h22 = np.exp(-0.5 * ((time - 63.4) / 2.0) ** 2)
    path = tmp_path / "case.npz"
    np.savez(path, time=time, h33=h33, h22=h22)

    case = load_complex_case(path)

    assert case.mode_name == "h33"
    np.testing.assert_array_equal(case.time, time)
    np.testing.assert_allclose(case.mode, h33)
    assert case.h22 is not None
    assert case.metadata["h22_peak_time"] == time[np.argmax(abs(h22))]
    assert "input time coordinate" in case.metadata["h22_peak_time_convention"]


def test_tone_has_positive_frequency_and_epoch_correction():
    time = np.arange(0.0, 200.0, 0.1)
    peak = 80.0
    f0 = 0.05
    h33 = np.exp(-2j * np.pi * f0 * (time - peak))

    result = transform_complex(
        time, h33, FAST_CONTROLS, relaxation_time=20.0,
        h22_peak_time=peak,
    )
    dominant = int(np.argmax(abs(result.spectrum)))
    assert abs(result.frequency_Mf[dominant] - f0) <= 1.0 / result.metadata["nfft"]
    assert np.all(result.frequency_Mf > 0)
    expected = np.exp(
        -2j * np.pi * result.frequency_Mf
        * (result.metadata["fft_epoch"] - peak)
    )
    np.testing.assert_allclose(result.spectrum, result.spectrum_raw * expected,
                               rtol=2e-13, atol=2e-13)
    assert result.metadata["positive_mode_mapping"] == "h_(3,-3) = -conj(h33)"


def test_time_label_translation_is_removed_by_h22_peak_epoch():
    time = np.arange(0.0, 200.0, 0.1)
    shift = 37.3
    h33 = np.exp(-2j * np.pi * 0.047 * (time - 80.0))

    first = transform_complex(
        time, h33, FAST_CONTROLS, relaxation_time=20.0,
        h22_peak_time=80.0,
    )
    second = transform_complex(
        time + shift, h33, FAST_CONTROLS, relaxation_time=20.0 + shift,
        h22_peak_time=80.0 + shift,
    )

    np.testing.assert_allclose(first.frequency_Mf, second.frequency_Mf, rtol=0, atol=0)
    np.testing.assert_allclose(first.spectrum, second.spectrum, rtol=2e-12, atol=2e-12)
    assert second.metadata["fft_epoch"] - (80.0 + shift) == pytest.approx(
        first.metadata["fft_epoch"] - 80.0
    )


def test_complex_chirp_retains_positive_band():
    time = np.arange(0.0, 160.0, 0.1)
    peak = 70.0
    u = time - peak
    # The stored SXS sign is negative; conjugation makes this an increasing-
    # phase chirp, whose Fourier power is at positive frequencies.
    h33 = np.exp(-2j * np.pi * (0.045 * u + 0.00008 * u * u))
    result = transform_complex(
        time, h33, FAST_CONTROLS, relaxation_time=10.0,
        h22_peak_time=peak,
    )
    assert np.iscomplexobj(result.spectrum)
    assert np.isfinite(result.spectrum).all()
    peak_frequency = result.frequency_Mf[np.argmax(abs(result.spectrum))]
    assert 0.03 < peak_frequency < 0.07


def test_common_grid_interpolates_complex_values_without_extrapolation():
    frequency = np.array([0.1, 0.2, 0.3])
    spectrum = np.array([1 + 2j, 3 + 6j, 5 + 10j])
    grid = np.array([0.15, 0.25])
    np.testing.assert_allclose(
        interpolate_complex_common_grid(frequency, spectrum, grid),
        np.array([2 + 4j, 4 + 8j]),
    )
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_complex_common_grid(frequency, spectrum, [0.09, 0.2])

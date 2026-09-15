"""Tests for the cache-backed public TEOB FD comparison helpers."""

import numpy as np
import pytest

from diagnose_teob_public_fd_clean import _band_comparison, _metrics


def test_exact_frequency_band_and_unaligned_metrics_are_explicit():
    frequencies = np.arange(0.0, 10.0, 0.5)
    native_hp = 1.0 + 0.0j * frequencies
    native_hc = 2.0 + 0.0j * frequencies
    public_hp = native_hp.copy()
    public_hc = native_hc.copy()
    public_hp[4] += 1.0
    comparison = _band_comparison(
        frequencies,
        native_hp,
        native_hc,
        frequencies.copy(),
        public_hp,
        public_hc,
        lower_hz=2.0,
        upper_hz=4.0,
        delta_f_hz=0.5,
    )
    assert comparison["axis"]["selected_samples"] == 5
    assert comparison["axis"]["first_frequency_hz"] == 2.0
    assert comparison["axis"]["last_frequency_hz"] == 4.0
    assert comparison["alignment"] is False
    assert comparison["optimization"] is False
    assert comparison["h_plus"]["signal_norm_native"] == pytest.approx(np.sqrt(5.0))
    assert comparison["h_plus"]["signal_norm_public"] == pytest.approx(np.sqrt(8.0))
    assert comparison["h_plus"]["relative_l2_unaligned"] == pytest.approx(1.0 / np.sqrt(8.0))


def test_combined_metrics_use_public_norm_as_relative_l2_denominator():
    native = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    public = np.array([1.0 + 0.0j, 1.0 + 0.0j])
    result = _metrics(native, public)
    assert result["signal_norm_native"] == pytest.approx(np.sqrt(5.0))
    assert result["signal_norm_public"] == pytest.approx(np.sqrt(2.0))
    assert result["residual_norm_unaligned"] == 1.0
    assert result["relative_l2_unaligned"] == pytest.approx(1.0 / np.sqrt(2.0))
    assert result["complex_norm_ratio_native_over_public"] == pytest.approx(np.sqrt(2.5))
    assert result["mismatch_unaligned"] == pytest.approx(1.0 - 3.0 / np.sqrt(10.0))

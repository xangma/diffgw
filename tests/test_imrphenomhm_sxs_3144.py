"""Data-free numerical invariants for the frozen SXS:BBH:3144 campaign."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts import run_imrphenomhm_sxs_3144 as campaign
from scripts.run_imrphenomhm_sxs_3144 import (
    ANGLES,
    FREQUENCIES,
    NR_MODES,
    direct_match,
    max_time_match,
    norm_squared,
    rotated_nr_basis,
    synthetic_fft_check,
)


def test_exact_network_shift_and_direct_inner_product():
    result = synthetic_fft_check()
    assert np.isclose(result["match"], 1.0, rtol=0, atol=1e-12)
    assert result["direct_recheck_error"] < 1e-12


def test_network_norm_and_fft_agree_with_direct_channels():
    frequencies = FREQUENCIES
    reference = np.stack((np.exp(-((frequencies - 160) / 90) ** 2),
                          0.4j * np.exp(-((frequencies - 190) / 95) ** 2)))
    candidate = np.stack((0.8 * reference[0], 1.2 * reference[1]))
    psd = 1 + (frequencies / 100) ** 2
    expected_ref_power = sum(4 * np.sum(np.abs(channel) ** 2 / psd)
                             for channel in reference)
    expected_candidate_power = sum(4 * np.sum(np.abs(channel) ** 2 / psd)
                                   for channel in candidate)
    assert np.isclose(norm_squared(reference, psd), expected_ref_power, rtol=0, atol=1e-12)
    result = max_time_match(reference, candidate, psd)
    assert result["shift_seconds"] == 0
    assert np.isclose(result["norm_ratio"],
                      np.sqrt(expected_candidate_power / expected_ref_power),
                      rtol=0, atol=1e-14)
    assert np.isclose(result["match"], direct_match(reference, candidate, psd, 0),
                      rtol=0, atol=1e-12)


def test_orbital_rotation_matches_time_domain_direct_sum():
    """Rotate signed m values in time, then compare independent Fourier sums."""
    sample_rate = 8192
    n = 64
    time = np.arange(n) / sample_rate
    frequencies = FREQUENCIES
    kernel = np.exp(-2j * np.pi * frequencies[:, None] * time)
    phase = 0.37
    modes = np.zeros((len(ANGLES), 2, len(NR_MODES), frequencies.size), complex)
    directly_rotated = np.zeros((len(ANGLES), 2, frequencies.size), complex)
    for view in range(len(ANGLES)):
        for j, (_, m) in enumerate(NR_MODES):
            amplitude = (j + 1) * (1 + 0.2 * view)
            complex_mode = amplitude * np.exp(2j * np.pi * (100 + 3 * j) * time)
            modes[view, 0, j] = kernel @ complex_mode.real / sample_rate
            modes[view, 1, j] = kernel @ (-complex_mode.imag) / sample_rate
            rotated_time = complex_mode * np.exp(-1j * m * phase)
            directly_rotated[view, 0] += kernel @ rotated_time.real / sample_rate
            directly_rotated[view, 1] += kernel @ (-rotated_time.imag) / sample_rate
    np.testing.assert_allclose(rotated_nr_basis(modes, phase), directly_rotated,
                               rtol=2e-13, atol=1e-14)


def test_evaluation_rejects_a_different_preparation_script(tmp_path, monkeypatch):
    fixture = tmp_path / "hm_sxs_3144_reference.npz"
    fixture.write_bytes(b"fixture")
    preparation = {
        "source_hashes": {},
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "script_sha256": "0" * 64,
    }
    (tmp_path / "hm_sxs_3144_preparation.json").write_text(json.dumps(preparation))
    monkeypatch.setattr(campaign, "verify_sources", lambda _: {})
    with pytest.raises(RuntimeError, match="preparation script differ"):
        campaign.evaluate_fixture(tmp_path, "test")

"""Tests for the source-isolated XHM (3,3) ringdown calibration pilot."""

import json
from pathlib import Path

import numpy as np

from scripts.fit_nr33_amplitude_pilot import (
    Case,
    FFTControls,
    frequency_amplitude,
    fit_direct,
    fit_manifest,
    ringdown_shape,
    split_by_q,
)


def _synthetic_spectrum(n=4096, dt=0.5, f_ring=0.1, f_damp=0.02,
                        log_a=2.0, lam=0.35):
    freq = np.fft.fftfreq(n, dt)
    positive = freq > 0
    spectrum = np.zeros(n, dtype=complex)
    physical = freq < 0
    spectrum[physical] = np.exp(log_a) * ringdown_shape(
        -freq[physical], f_ring, f_damp, lam
    )
    # This is an h33(t) fixture.  The fitter's conjugation maps its SXS-style
    # negative-frequency convention back to the positive f bins above.
    time = np.arange(n) * dt
    h33 = np.fft.ifft(spectrum) / dt
    return time, h33


def test_complex_fft_uses_sxs_negative_frequency_convention():
    time, h33 = _synthetic_spectrum()
    f, amp = frequency_amplitude(time, h33, FFTControls(1, 0, 0, 0.3, 0.5))
    expected = np.exp(2.0) * ringdown_shape(f, 0.1, 0.02, 0.35)
    # Frequency amplitude is recovered from the raw complex fixture without
    # silently dropping the h33 quadrature or selecting the wrong carrier sign.
    assert np.sqrt(np.mean((amp - expected) ** 2)) / np.sqrt(np.mean(expected ** 2)) < 2e-10


def test_direct_fit_recovers_eq42_parameters_from_raw_fixture():
    time, h33 = _synthetic_spectrum()
    f, amp = frequency_amplitude(time, h33, FFTControls(1, 0, 0, 0.3, 0.5))
    mask = (f >= 0.095) & (f <= 0.3)
    result = fit_direct(f[mask], amp[mask], 0.1, 0.02)
    assert abs(result["log_abs_a_lambda"] - 2.0) < 1e-6
    assert abs(result["lambda"] - 0.35) < 1e-6
    assert result["relative_rms"] < 1e-8


def test_q_split_is_frozen_by_distinct_mass_ratio():
    cases = [Case("q%.1f" % q, "unused", q, {}) for q in (1.5, 2, 3, 4, 5, 6)]
    train, holdout = split_by_q(cases)
    assert [c.q for c in train] == [1.5, 2, 4, 5]
    assert [c.q for c in holdout] == [3, 6]
    assert {c.q for c in train}.isdisjoint(c.q for c in holdout)


def test_manifest_runs_with_six_cases_and_reports_holdout(tmp_path: Path):
    entries = []
    for i, q in enumerate((1.5, 2, 3, 4, 5, 6)):
        time, h33 = _synthetic_spectrum(log_a=1.0 + 0.1 * q, lam=0.2 + 0.01 * q)
        path = tmp_path / ("case%d.npz" % i)
        np.savez(path, time=time, h33=h33)
        entries.append({"id": "case%d" % i, "path": path.name, "q": q,
                        "metadata": {"f_ring_33": 0.1, "f_damp_33": 0.02,
                                     "relaxation_time": 0.0, "peak_time": 1000.0}})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": entries}))
    result = fit_manifest(manifest, FFTControls(1, 0, 0, 0.3, 0.5))
    assert result["train_q"] == [1.5, 2, 4, 5]
    assert result["holdout_q"] == [3, 6]
    assert result["parameter_fit"]["degree"] == 2
    assert result["limitations"] == ["ringdown subcomponent only", "no full-XHM or LAL parity claim"]

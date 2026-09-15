from __future__ import annotations

import numpy as np

from scripts.teob_no_contact_nqc_attribution import complex_metrics, threshold_first


def test_complex_metrics_has_no_alignment_and_reports_phase() -> None:
    reference = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    candidate = reference * np.exp(1j * 1.0e-3)
    result = complex_metrics(candidate, reference)
    assert result["alignment"] == "none"
    assert result["samples"] == 2
    np.testing.assert_allclose(result["phase_max_abs_radians"], 1.0e-3, rtol=0.0, atol=1.0e-12)


def test_threshold_first_uses_first_crossing_without_fit() -> None:
    q = np.array([0.0, 0.5, 1.0])
    ratio = np.array([1.0e-8, 2.0e-4, 1.0e-5])
    assert threshold_first(q, ratio, 1.0e-4) == {
        "index": 1,
        "td_time_dimensionless": 0.5,
        "ratio": 2.0e-4,
    }


def test_threshold_first_returns_none_when_unreached() -> None:
    assert threshold_first(np.arange(2.0), np.array([0.0, 1.0e-8]), 1.0e-4) is None

"""Checks for the bounded no-contact SEOB first-divergence evidence."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.seob_no_contact_first_divergence import validate_common_fine_chart

REPORT = Path("docs/reviews/2026-09-15-no-contact/seob-first-divergence.json")

pytestmark = pytest.mark.skipif(
    not REPORT.exists(),
    reason="Review audit artifact not present (archived)",
)


def test_seob_first_divergence_report_contains_state_swap_evidence():
    report = json.loads(REPORT.read_text())
    assert report["parameters"]["pre_attachment_samples"] > 100
    assert report["parameters"]["sample_count"] >= report["parameters"]["pre_attachment_samples"]
    pre = report["pre_attachment"]
    assert pre["clean_factorized_on_clean_state_vs_public"]["samples"] == report["parameters"]["pre_attachment_samples"]
    assert pre["clean_factorized_on_public_state_vs_public"]["samples"] == report["parameters"]["pre_attachment_samples"]
    assert pre["clean_nqc_on_public_state_vs_public"]["complex_relative_l2"] < pre["clean_nqc_on_clean_state_vs_public"]["complex_relative_l2"]
    assert pre["state_substitution_improvement"]["complex_relative_l2_percent_reduction"] > 50.0
    arrays = Path(report["arrays"]["path"])
    with np.load(arrays, allow_pickle=False) as data:
        assert data["time_fine_global"].shape[0] == report["parameters"]["fine_sample_count"]
        assert data["clean_state"].shape == data["public_state"].shape


def test_seob_chart_validation_rejects_one_sample_drift():
    clean = np.arange(4, dtype=np.float64)
    public = clean.copy()
    public[2] += 1.0e-12
    with pytest.raises(RuntimeError, match="full fine time axes"):
        validate_common_fine_chart(clean, public)

"""Focused tests for the post-freeze NR (3,3) amplitude validator."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.validate_nr33_amplitude_pilot import (
    amplitude_metrics,
    geometric_to_physical,
    lal_one_mode_amplitude,
    validate_frozen_predictions,
    validate_frequency_grid,
)


def _payload(prediction, *, target=None, grid=None):
    grid = [0.02, 0.025, 0.03] if grid is None else grid
    target = prediction if target is None else target
    params = {
        "case_id": "heldout-01",
        "m1_msun": 35.0,
        "m2_msun": 15.0,
        "chi1": 0.0,
        "chi2": 0.0,
        "distance_mpc": 100.0,
    }
    return {
        "status": "frozen",
        "calibration_before_sha256": "a" * 64,
        "calibration_after_sha256": "a" * 64,
        "split": {"train_case_ids": ["training-01"], "holdout_case_ids": ["heldout-01"]},
        "predictions": [{**params, "frequency_Mf": grid, "amplitude_geometric": prediction}],
    }, {
        "targets": [{**params, "split": "heldout", "frequency_Mf": grid, "amplitude_geometric": target}],
    }


def test_geometric_physical_scale_is_explicit_and_reversible():
    physical = geometric_to_physical([1.0, 2.0], 2.0, 4.0)
    np.testing.assert_allclose(physical, [2.0 * (2.0 * 299792458.0 / 4.0), 4.0 * (2.0 * 299792458.0 / 4.0)])


def test_metrics_report_amplitude_l2_norm_and_maxerrors():
    result = amplitude_metrics([1.0, 2.0], [1.0, 1.0])
    assert result["relative_amp_l2"] == pytest.approx(1.0 / np.sqrt(2.0))
    assert result["norm"]["ratio_prediction_over_target"] == pytest.approx(np.sqrt(5.0 / 2.0))
    assert result["maxerrors"]["absolute"] == pytest.approx(1.0)
    assert result["maxerrors"]["relative"] == pytest.approx(1.0)


def test_frequency_grid_rejects_nonfinite_or_nonmonotonic_values():
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_frequency_grid([0.02, 0.02])
    with pytest.raises(ValueError, match="non-finite"):
        validate_frequency_grid([0.02, np.nan])


def test_validator_requires_matching_exact_target_grid():
    fit, target = _payload([1e-3, 2e-3, 3e-3], grid=[0.02, 0.025, 0.03])
    target["targets"][0]["frequency_Mf"] = [0.02, 0.025000000000000004, 0.03]
    with pytest.raises(ValueError, match="grids differ exactly"):
        validate_frozen_predictions(fit, target)


def test_validator_requires_frozen_calibration_hash_evidence():
    fit, target = _payload([1e-3, 2e-3, 3e-3])
    fit["status"] = "draft"
    with pytest.raises(ValueError, match="not frozen"):
        validate_frozen_predictions(fit, target)
    fit["status"] = "frozen"
    fit["calibration_after_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="before/after"):
        validate_frozen_predictions(fit, target)


def test_validator_reports_only_amplitude_metrics_and_heldout_case():
    fit, target = _payload([1.0e-3, 2.0e-3, 3.0e-3], target=[1.0e-3, 2.2e-3, 3.0e-3])
    report = validate_frozen_predictions(fit, target)
    assert report["status"] == "complete"
    assert report["amplitude_only"] is True
    assert report["complex_waveform_mismatch_evaluated"] is False
    assert report["cases"][0]["target_split"] == "heldout"
    assert report["cases"][0]["relative_amp_l2"] > 0.0


def test_validator_rejects_duplicate_or_unexpected_case_ids():
    fit, target = _payload([1.0e-3, 2.0e-3, 3.0e-3])
    target["targets"].append(dict(target["targets"][0], case_id="unexpected"))
    with pytest.raises(ValueError, match="held-out ID mismatch"):
        validate_frozen_predictions(fit, target)
    target["targets"][-1]["case_id"] = "heldout-01"
    with pytest.raises(ValueError, match="duplicate target case ID"):
        validate_frozen_predictions(fit, target)


def test_disk_calibration_tamper_is_rejected(tmp_path):
    fit, target = _payload([1.0e-3, 2.0e-3, 3.0e-3])
    fit_path = tmp_path / "fit.json"
    target_path = tmp_path / "targets.json"
    calibration_path = tmp_path / "coefficients.bin"
    digest_path = tmp_path / "coefficients.sha256"
    calibration_path.write_bytes(b"frozen coefficients")
    calibration_digest = __import__("hashlib").sha256(calibration_path.read_bytes()).hexdigest()
    digest_path.write_text(calibration_digest + "\n")
    fit["calibration_before_sha256"] = calibration_digest
    fit["calibration_after_sha256"] = calibration_digest
    # A digest embedded in the bytes it purports to hash is only a claim and
    # cannot be used as a self-consistent integrity check.
    fit["fit_artifact_sha256"] = "0" * 64
    fit_path.write_text(__import__("json").dumps(fit))
    target_path.write_text(__import__("json").dumps(target))
    report = validate_frozen_predictions(
        fit_path, target_path, calibration_path=calibration_path,
        calibration_digest_path=digest_path,
    )
    assert report["freeze"]["calibration_before_sha256"] == report["freeze"]["calibration_after_sha256"]
    calibration_path.write_bytes(b"tampered coefficients")
    with pytest.raises(ValueError, match="expected frozen digest"):
        validate_frozen_predictions(
            fit_path, target_path, calibration_path=calibration_path,
            calibration_digest_path=digest_path,
        )


class _FakeVector:
    def __init__(self, count):
        self.data = np.empty(count, dtype=np.float64)


class _FakeLAL:
    MSUN_SI = 2.0

    @staticmethod
    def CreateREAL8Vector(count):
        return _FakeVector(count)

    @staticmethod
    def CreateDict():
        return {"created": True}


class _FakeLALSim:
    __version__ = "fake"
    seen = None

    @staticmethod
    def SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(params, version):
        params["xhm_release_version"] = int(version)

    @staticmethod
    def SimInspiralWaveformParamsLookupPhenomXHMReleaseVersion(params):
        return params["xhm_release_version"]

    @classmethod
    def SimIMRPhenomXHMFrequencySequenceOneMode(cls, vector, m1, m2, chi1, chi2, ell, emm, distance, phi_ref, f_ref, extra):
        cls.seen = (vector.data.copy(), m1, m2, chi1, chi2, ell, emm, distance, phi_ref, f_ref, extra)
        return type("Series", (), {"data": type("Data", (), {"data": 2.0 + 0.0j * vector.data})()})()


def test_lal_adapter_uses_m_minus_three_and_exact_mf_to_hz_conversion():
    grid = np.array([0.02, 0.025, 0.03], dtype=np.float64)
    values, metadata = lal_one_mode_amplitude(
        grid,
        m1_msun=35.0,
        m2_msun=15.0,
        distance_m=100.0,
        lal_module=_FakeLAL,
        lalsimulation_module=_FakeLALSim,
    )
    expected_hz = grid / ((35.0 + 15.0) * 4.925490947641267e-6)
    np.testing.assert_array_equal(_FakeLALSim.seen[0], expected_hz)
    assert _FakeLALSim.seen[5:7] == (3, -3)
    np.testing.assert_array_equal(values, np.full(3, 2.0))
    assert metadata["mode_argument"] == [3, -3]
    assert metadata["xhm_release_version"] == 122022
    assert metadata["xhm_release_version_source"].startswith("explicit public setter")

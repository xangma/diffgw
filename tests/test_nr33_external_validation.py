"""Contract tests for validation on new, frozen external NR holdouts."""

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from scripts import run_nr33_external_validation as adapter


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "docs/reviews/2026-09-15-independent-calibration-v3-search/nr33-calibration.json"
PROTOCOL = ROOT / "docs/reviews/2026-09-15-independent-calibration-expanded/protocol.json"


def _npz(path: Path) -> str:
    np.savez(path, time=np.arange(0.0, 10.0, 0.1), h33=np.ones(100, dtype=complex))
    return adapter.sha256_file(path)


def _inputs(tmp_path: Path, *, sxs_id: str = "SXS:BBH:9999") -> tuple[Path, Path, dict]:
    waveform = tmp_path / "external.npz"
    digest = _npz(waveform)
    case_id = f"{sxs_id}/Lev5/N3"
    cases_payload = {"cases": [{
        "id": case_id, "path": str(waveform), "q": 7.0, "target_q": 7.0,
        "candidate_rank": 1,
        "sxs_id": sxs_id, "split": "holdout", "extrapolation_order": 3,
        "resolution": "Lev5", "npz_sha256": digest,
        "metadata": {"relaxation_time": 1.0, "peak_time": 5.0,
                      "final_mass_fraction": 0.95, "final_spin": [0.0, 0.0, 0.5],
                      "sxs_id": sxs_id, "split": "holdout",
                      "extrapolation_order": 3, "npz_sha256": digest},
    }]}
    screened_input = tmp_path / "external-input-manifest.json"
    screened_input.write_text(json.dumps(cases_payload))
    input_sha = adapter.sha256_file(screened_input)
    manifest = tmp_path / "external-manifest.json"
    qualified_payload = dict(cases_payload)
    qualified_payload["screening"] = {
        "policy_sha256": "1a863811b90c410cd5a9b0b52a0bd53696b869a0d503cb8fa61d9c054842138d",
        "protocol_sha256": adapter.FROZEN_PROTOCOL_SHA256,
        "manifest_sha256": input_sha,
        "selected_primary_case_ids": [case_id],
    }
    manifest.write_text(json.dumps(qualified_payload))
    qualified_sha = adapter.sha256_file(manifest)
    quality = tmp_path / "quality.json"
    report = {
        "overall_pass": True,
        "policy_sha256": "1a863811b90c410cd5a9b0b52a0bd53696b869a0d503cb8fa61d9c054842138d",
        "protocol_sha256": adapter.FROZEN_PROTOCOL_SHA256,
        "input_manifest_sha256": input_sha,
        "selected_primary_case_ids": [case_id],
        "qualified_manifest": {"path": str(manifest), "sha256": qualified_sha, "case_count": 1},
        "simulations": [{"primary_case_id": case_id, "passed": True, "evaluation_status": "passed"}],
    }
    quality.write_text(json.dumps(report))
    return manifest, quality, report


def _patch_fast_numerics(monkeypatch):
    freq = np.linspace(0.1, 0.3, 8)

    def fake_load(case):
        return np.arange(100.0), np.ones(100, dtype=complex), {
            "relaxation_time": 1.0, "peak_time": 5.0,
            "final_mass_fraction": 0.95, "final_spin": [0.0, 0.0, 0.5],
        }

    monkeypatch.setattr(adapter, "load_case", fake_load)
    monkeypatch.setattr(adapter, "independent_qnm33", lambda metadata: {
        "f_ring_33": 0.1, "f_damp_33": 0.02, "qnm_source": "test",
    })
    monkeypatch.setattr(adapter, "frequency_amplitude", lambda *args, **kwargs: (
        freq, np.ones(freq.size), {"test": True}))


def _prepare_args(tmp_path, manifest, quality):
    return (CALIBRATION, manifest, quality, tmp_path / "out",
            adapter.FROZEN_CALIBRATION_SHA256, PROTOCOL, adapter.FROZEN_PROTOCOL_SHA256)


def test_provenance_rejection(tmp_path):
    fit = json.loads(CALIBRATION.read_text())
    fit["provenance"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="original fit manifest"):
        adapter._check_frozen_inputs(CALIBRATION, fit, adapter.FROZEN_CALIBRATION_SHA256)


def test_accidental_training_reuse_is_rejected(tmp_path):
    manifest, quality, _ = _inputs(tmp_path, sxs_id="SXS:BBH:2331")
    with pytest.raises(ValueError, match="excluded by protocol"):
        adapter.prepare_inputs(*_prepare_args(tmp_path, manifest, quality))


def test_frozen_fit_overlap_guard_is_distinct(tmp_path):
    manifest, _, _ = _inputs(tmp_path, sxs_id="SXS:BBH:9999")
    cases = adapter._external_cases(manifest)
    fit = json.loads(CALIBRATION.read_text())
    fit["split"]["train_case_ids"].append(cases[0].case_id)
    protocol = json.loads(PROTOCOL.read_text())
    protocol["excluded_sxs_ids"] = ["SXS:BBH:0008"]
    with pytest.raises(ValueError, match="overlaps frozen fit"):
        adapter._check_external(cases, manifest, fit, protocol)


def test_existing_output_is_never_overwritten(tmp_path):
    manifest, quality, _ = _inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "nr33-external-predictions.json"
    existing.write_text("keep")
    args = list(_prepare_args(tmp_path, manifest, quality))
    args[3] = out
    with pytest.raises(FileExistsError, match="overwrite"):
        adapter.prepare_inputs(*args)
    assert existing.read_text() == "keep"


def test_individual_quality_failure_is_rejected(tmp_path):
    manifest, quality, report = _inputs(tmp_path)
    report["simulations"][0]["passed"] = False
    quality.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="individually pass quality gates"):
        adapter.prepare_inputs(*_prepare_args(tmp_path, manifest, quality))


def test_calibration_mutation_during_external_read_is_rejected(tmp_path, monkeypatch):
    calibration = tmp_path / "calibration.json"
    shutil.copyfile(CALIBRATION, calibration)
    manifest, quality, _ = _inputs(tmp_path)
    original_load = adapter.load_case

    def mutate_then_load(case):
        calibration.write_bytes(b"mutated coefficients")
        return original_load(case)

    _patch_fast_numerics(monkeypatch)
    monkeypatch.setattr(adapter, "load_case", mutate_then_load)
    args = list(_prepare_args(tmp_path, manifest, quality))
    args[0] = calibration
    with pytest.raises(ValueError, match="changed while preparing"):
        adapter.prepare_inputs(*args)
    assert not (tmp_path / "out" / "nr33-external-predictions.json").exists()


def test_valid_external_holdout_is_compatible_with_frozen_validator(tmp_path, monkeypatch):
    manifest, quality, _ = _inputs(tmp_path)
    _patch_fast_numerics(monkeypatch)
    prediction, target, provenance = adapter.prepare_inputs(*_prepare_args(tmp_path, manifest, quality))
    result = adapter.validate_frozen_predictions(
        prediction, target, calibration_path=CALIBRATION,
        expected_calibration_sha256=adapter.FROZEN_CALIBRATION_SHA256)
    assert result["status"] == "complete"
    assert result["cases"][0]["case_id"] == "SXS:BBH:9999/Lev5/N3"
    assert provenance["protocol_sha256"] == adapter.FROZEN_PROTOCOL_SHA256
    assert result["calibration_before_sha256"] == result["calibration_after_sha256"] == adapter.FROZEN_CALIBRATION_SHA256

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest
import torch
import numpy as np

from scripts.refit_teob_qnm_independent_clean import rational


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/replay_teob_qnm_refit_clean.py"
ARTIFACT = Path(__file__).resolve().parents[1] / "docs/reviews/2026-09-13-parity-review/verification/teob-qnm-recalibration-clean/refit-calibration.json"
SPEC = importlib.util.spec_from_file_location("replay_teob_qnm_refit_clean", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(replay)


def _coefficients(path: Path) -> None:
    path.write_text(json.dumps({"domain": [0.0, 0.95], "fits": {
        "omega220": {"coefficients": [1.0, 2.0, 0, 0, 0, 0, 0]},
        "alpha220": {"coefficients": [3.0, 4.0, 0, 0, 0, 0, 0]},
        "alpha21": {"coefficients": [5.0, 6.0, 0, 0, 0, 0, 0]},
    }}), encoding="utf-8")


def test_json_evaluator_is_tensor_polynomial_and_preserves_dtype(tmp_path):
    path = tmp_path / "coefficients.json"
    _coefficients(path)
    evaluator, _ = replay._load_evaluator(path)
    result = evaluator(torch.tensor([0.25], dtype=torch.float64))
    assert [float(x) for x in result] == [1.5, 6.0, 12.5]
    assert all(x.dtype == torch.float64 for x in result)


def test_frozen_artifact_matches_numpy_rational_on_2d_and_gradcheck():
    evaluator, raw = replay._load_evaluator(ARTIFACT)
    spin = torch.tensor([[0.17, 0.31], [0.53, 0.71]], dtype=torch.float64)
    values = evaluator(spin)
    for name, got in zip(("omega220", "alpha220", "alpha21"), values):
        expected = rational(spin.numpy(), raw["fits"][name]["coefficients"])
        np.testing.assert_allclose(got.detach().numpy(), expected, rtol=1e-13, atol=1e-14)
    probe = torch.tensor([0.21, 0.68], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: evaluator(x)[0], (probe,), eps=1e-6, atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("provider", ["direct", "refit"])
@pytest.mark.parametrize("bad", [complex(0.2, 0.0), [0.2 + 0.0j], torch.tensor(float("nan")), -0.01, 0.950001])
def test_both_providers_reject_complex_nan_and_out_of_domain(provider, bad):
    evaluator = replay._direct_evaluator
    if provider == "refit":
        evaluator, _ = replay._load_evaluator(ARTIFACT)
    with pytest.raises(ValueError):
        evaluator(bad)


def test_main_patches_qnm_only_and_restores_on_exception(monkeypatch, tmp_path):
    path = tmp_path / "coefficients.json"
    _coefficients(path)
    report = tmp_path / "report.json"
    original = replay.teob_postmerger.qnm_fits
    observed = {}

    class Delegated:
        @staticmethod
        def main():
            observed["patched"] = replay.teob_postmerger.qnm_fits is not original
            observed["value"] = replay.teob_postmerger.qnm_fits(torch.tensor(0.0))[0]
            raise RuntimeError("expected test failure")

    monkeypatch.setitem(__import__("sys").modules, "replay_teob_conditioned_waveform_public_clean", Delegated)
    monkeypatch.setattr(__import__("sys"), "argv", [str(SCRIPT), "--qnm-coefficients", str(path), "--report", str(report)])
    with pytest.raises(RuntimeError, match="expected test failure"):
        replay.main()
    assert observed["patched"] is True
    assert float(observed["value"]) == 1.0
    assert replay.teob_postmerger.qnm_fits is original
    assert not report.exists()


def test_stale_report_is_not_annotated_when_delegate_does_not_write(monkeypatch, tmp_path):
    path = tmp_path / "coefficients.json"
    _coefficients(path)
    report = tmp_path / "report.json"
    report.write_text('{"old": true}\n', encoding="utf-8")
    before = report.read_bytes()

    class Delegated:
        @staticmethod
        def main():
            return 0

    monkeypatch.setitem(__import__("sys").modules, "replay_teob_conditioned_waveform_public_clean", Delegated)
    monkeypatch.setattr(__import__("sys"), "argv", [str(SCRIPT), f"--qnm-coefficients={path}", f"--report={report}"])
    assert replay.main() == 0
    assert report.read_bytes() == before


def test_missing_coefficients_and_malformed_vectors_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        replay._load_evaluator(tmp_path / "missing.json")
    malformed = tmp_path / "bad.json"
    malformed.write_text(json.dumps({"domain": [0.0, 0.95], "fits": {"omega220": {"coefficients": [1]}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing seven rational coefficients for omega220"):
        replay._load_evaluator(malformed)


def test_direct_provider_is_bounded_and_returns_positive_damping():
    omega, alpha, alpha21 = replay._direct_evaluator(torch.tensor([0.0, 0.95], dtype=torch.float64))
    assert torch.all(omega > 0) and torch.all(alpha > 0) and torch.all(alpha21 > 0)
    with pytest.raises(ValueError, match="0 <= chi_final <= 0.95"):
        replay._direct_evaluator(torch.tensor(-0.1))
    with pytest.raises(ValueError, match="real"):
        replay._direct_evaluator(torch.tensor(0.1 + 0.0j))


def test_success_report_is_fresh_and_metadata_records_argv_and_restoration(monkeypatch, tmp_path):
    path = tmp_path / "coefficients.json"; _coefficients(path)
    report = tmp_path / "report.json"; original = replay.teob_postmerger.qnm_fits
    observed = {}
    class Delegated:
        @staticmethod
        def main():
            observed["patched"] = replay.teob_postmerger.qnm_fits is not original
            observed["argv"] = list(__import__("sys").argv)
            report.write_text(json.dumps({"run_control": {"pid": os.getpid(), "started_unix": time.time()}}))
            return 0
    monkeypatch.setitem(__import__("sys").modules, "replay_teob_conditioned_waveform_public_clean", Delegated)
    monkeypatch.setattr(__import__("sys"), "argv", [str(SCRIPT), "--qnm-provider=refit", f"--qnm-coefficients={path}", f"--report={report}", "--extra", "x"])
    assert replay.main() == 0
    data = json.loads(report.read_text())
    assert observed["patched"] and replay.teob_postmerger.qnm_fits is original
    assert "--qnm-provider=refit" not in observed["argv"] and "--qnm-coefficients=" not in " ".join(observed["argv"])
    assert data["qnm_refit_substitution"]["restored_in_finally"] is True
    assert data["qnm_refit_substitution"]["source_hashes_unchanged"] is True


def test_help_restores_argv(monkeypatch):
    observed = {}
    class Delegated:
        @staticmethod
        def main():
            observed["argv"] = list(__import__("sys").argv)
            return 7
    monkeypatch.setitem(__import__("sys").modules, "replay_teob_conditioned_waveform_public_clean", Delegated)
    original_argv = [str(SCRIPT), "--help"]
    monkeypatch.setattr(__import__("sys"), "argv", original_argv.copy())
    assert replay.main() == 7
    assert __import__("sys").argv == original_argv

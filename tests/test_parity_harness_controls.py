"""Focused negative controls and robustness tests for parity harness validation."""

import copy
import pytest
import torch

from tests.parity_harness import (
    get_canonical_analytic_cases,
    get_canonical_catalog_cases,
    run_full_parity_suite,
    run_oracle_comparison,
    validate_case_inventory,
    MANIFEST_MODELS,
)


def test_inventory_tamper_rejected():
    """Verify that duplicate, altered, or substituted cases fail inventory validation."""
    analytic_cases = get_canonical_analytic_cases()

    # 1. Duplicate case (replacement of cases with first case)
    tampered_cases = [copy.deepcopy(analytic_cases[0]) for _ in range(6)]
    for i, c in enumerate(tampered_cases):
        c["id"] = f"replacement_{i}"
    
    is_valid, errors = validate_case_inventory(tampered_cases, scope="analytic")
    assert not is_valid, "Inventory validation must reject substituted case parameters"
    assert any("mismatch" in e for e in errors)

    # 2. Run suite with tampered cases
    suite_res = run_full_parity_suite(scope="analytic", cases=tampered_cases)
    assert suite_res["inventory_valid"] is False
    assert suite_res["suite_passed"] is False


def test_inplace_parameter_corruption_rejected():
    """Verify that in-place mutation of mutable case lists is rejected against frozen specs."""
    corrupted_cases = get_canonical_analytic_cases()
    corrupted_cases[0]["params"]["mass1"] = 999.0

    is_valid, errors = validate_case_inventory(corrupted_cases, scope="analytic")
    assert not is_valid, "In-place corruption must be caught against deserialized frozen canonical specs"
    assert any("physical parameters do not match" in e for e in errors)


def test_unphysical_overlap_rejected():
    """Verify that unphysical overlap (> 1.0 + 1e-12 or non-finite) fails run_oracle_comparison."""
    case = copy.deepcopy(get_canonical_analytic_cases()[0])
    
    import tests.parity_harness as ph
    orig_eval = ph.evaluate_waveform_metrics

    def mock_eval(*args, **kwargs):
        res = orig_eval(*args, **kwargs)
        res["optimized_match"] = 2.0
        return res

    try:
        ph.evaluate_waveform_metrics = mock_eval
        res = run_oracle_comparison(case, strict=True)
        assert res["status"] == "FAILED", f"Unphysical overlap 2.0 must fail: {res}"
        assert "unphysical" in res.get("reason", "").lower()
    finally:
        ph.evaluate_waveform_metrics = orig_eval


def test_missing_and_unresolved_catalog_coverage(monkeypatch):
    """Exercise aggregation with synthetic results; physics acceptance is separate."""
    import tests.parity_harness as ph

    def synthetic_pass(case, strict=False):
        return {"status": "PASSED", "torch_name": case["torch_name"],
                "case_id": case["id"], "reference_kind": "lal_binary",
                "verification_scope": "full_polarizations"}

    monkeypatch.setattr(ph, "run_oracle_comparison", synthetic_pass)
    # Even apparently passing results cannot erase documented coverage gaps.
    unresolved = ph.run_full_parity_suite(scope="catalog")
    for name in ph.UNRESOLVED_ORACLE_REASONS:
        assert unresolved["catalog_coverage"][name]["status"] == "UNRESOLVED_ORACLE"

    assert not unresolved["catalog_passed"]
    for name in ("IMRPhenomXPHM-SpinTaylor", "IMRPhenomXPNR"):
        assert unresolved["catalog_coverage"][name]["status"] == "MISSING_COVERAGE"

    # Synthetic positive control only for the subset with frozen cases.
    monkeypatch.setattr(ph, "UNRESOLVED_ORACLE_REASONS", {})
    still_missing = ph.run_full_parity_suite(scope="catalog")
    assert not still_missing["catalog_passed"]
    covered = sorted({case["torch_name"] for case in ph.get_canonical_catalog_cases()})
    monkeypatch.setattr(ph, "MANIFEST_MODELS", covered)
    monkeypatch.setattr(ph, "get_manifest_models", lambda: covered)
    resolved = ph.run_full_parity_suite(scope="catalog")
    assert resolved["catalog_passed"] and resolved["suite_passed"]
    assert set(resolved["catalog_coverage"]) == set(covered)

    missing = ph.run_full_parity_suite(scope="analytic")
    assert not missing["catalog_passed"]
    assert missing["catalog_coverage"]["IMRPhenomD"]["status"] == "MISSING_COVERAGE"


def test_projection_named_psd_corruption_fails():
    """Verify that corrupting exclusively a projection PSD mismatch triggers verification failure."""
    case = copy.deepcopy(get_canonical_analytic_cases()[0])
    import tests.parity_harness as ph
    orig_psd_mismatch = ph.psd_mismatch

    call_count = [0]
    def corrupt_proj_psd(h1, h2, psd, df):
        call_count[0] += 1
        # Calls 1-6 are named polarization PSDs, 7-8 are polarization metric PSDs; projections follow.
        val = orig_psd_mismatch(h1, h2, psd, df)
        if call_count[0] > 8:
            return 0.5
        return val

    try:
        ph.psd_mismatch = corrupt_proj_psd
        res = ph.run_oracle_comparison(case, strict=True)
        assert res["status"] == "FAILED", "Projection PSD mismatch corruption must cause verification failure"
        assert res["plus"]["psd_mismatch"] <= 1.0e-8
        assert res["cross"]["psd_mismatch"] <= 1.0e-8
    finally:
        ph.psd_mismatch = orig_psd_mismatch


def test_invalid_waveform_inputs_fail():
    """Verify that invalid waveform tensors, bad shapes, invalid grids, or negative PSDs return FAILED with reason."""
    case = copy.deepcopy(get_canonical_analytic_cases()[0])
    import tests.parity_harness as ph

    # 1. Non-finite (NaN) waveform tensor
    orig_wf = ph.get_fd_waveform
    def nan_wf(*args, **kwargs):
        hp, hc = orig_wf(*args, **kwargs)
        hp = hp.clone()
        hp[len(hp) // 2] = float("nan")
        return hp, hc

    try:
        ph.get_fd_waveform = nan_wf
        res = ph.run_oracle_comparison(case, strict=True)
        assert res["status"] == "FAILED", "NaN waveform must fail verification"
        assert "non-finite" in res.get("reason", "").lower()
    finally:
        ph.get_fd_waveform = orig_wf

    # 2. Shape mismatch between waveform and grid
    def bad_shape_wf(*args, **kwargs):
        hp, hc = orig_wf(*args, **kwargs)
        return hp[:-5], hc[:-5]

    try:
        ph.get_fd_waveform = bad_shape_wf
        res = ph.run_oracle_comparison(case, strict=True)
        assert res["status"] == "FAILED", "Mismatched waveform shape must fail verification"
        assert "shape" in res.get("reason", "").lower()
    finally:
        ph.get_fd_waveform = orig_wf

    # 3. Invalid frequency grid (non-increasing frequencies / inverted bounds)
    bad_grid_case = copy.deepcopy(case)
    bad_grid_case["grid"]["flow"] = 1000.0
    bad_grid_case["grid"]["fhigh"] = 50.0
    res = ph.run_oracle_comparison(bad_grid_case, strict=True)
    assert res["status"] == "FAILED", "Inverted frequency grid must fail verification"
    assert "frequency bounds" in res.get("reason", "").lower() or "grid" in res.get("reason", "").lower()

    # 4. Non-positive / negative PSD values
    orig_psd = ph.aLIGOZeroDetHighPower
    def negative_psd(f):
        vals = orig_psd(f).clone()
        vals[10] = -1.0e-45
        return vals

    try:
        ph.aLIGOZeroDetHighPower = negative_psd
        res = ph.run_oracle_comparison(case, strict=True)
        assert res["status"] == "FAILED", "Negative PSD values must fail verification"
        assert "psd" in res.get("reason", "").lower()
    finally:
        ph.aLIGOZeroDetHighPower = orig_psd


def test_manifest_consistency_and_drift_detection(monkeypatch):
    """Detect catalog drift without running the waveform oracles."""
    import tests.parity_harness as ph
    actual_manifest_models = ph.get_manifest_models()
    assert len(actual_manifest_models) == 17
    assert set(actual_manifest_models) == set(ph.MANIFEST_MODELS)

    monkeypatch.setattr(ph, "run_oracle_comparison", lambda case, strict=False: {
        "status": "PASSED", "torch_name": case["torch_name"], "case_id": case["id"],
    })
    monkeypatch.setattr(ph, "get_manifest_models",
                        lambda: sorted(actual_manifest_models + ["DriftedModel_v1"]))
    suite_res = ph.run_full_parity_suite(scope="catalog", strict=False)
    assert suite_res["catalog_passed"] is False
    assert any("DriftedModel_v1" in err for err in suite_res["manifest_drift_errors"])


def test_manifest_validation_matches_referenced_artifacts():
    """Keep validation coverage and metrics tied to their declared strict artifacts."""
    import json
    import os
    import re
    import subprocess
    from pathlib import Path

    root = Path(__file__).parents[1]
    manifest = json.loads((root / "docs/model_manifest.json").read_text())
    references = manifest["validation_artifacts"]

    def read_artifact(reference):
        if "://" not in reference:
            return json.loads((root / reference).read_text())
        match = re.fullmatch(
            r"https://github\.com/xangma/diffgw/blob/([0-9a-f]{40})/([^?#]+)",
            reference,
        )
        assert match, f"Artifact URL must pin a same-repository commit: {reference}"
        commit, path = match.groups()
        git_env = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
        available = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root, env=git_env, capture_output=True, text=True,
        )
        if available.returncode:
            pytest.skip(f"Historical artifact commit {commit} is unavailable locally")
        result = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=root, env=git_env, capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)

    source_record = read_artifact(references["source_record"])
    assert source_record["changed_inputs"] == []
    assert source_record["working_tree_changes"] == []

    for scope in ("catalog", "holdout"):
        artifact = read_artifact(references[scope])
        assert artifact["scope"] == scope
        assert artifact["strict"] is True
        assert artifact["catalog_passed"] is artifact["suite_passed"]
        assert manifest["verification"][f"{scope}_passed"] is artifact["suite_passed"]
        results = artifact["results"]
        assert artifact["num_cases"] == len(results)
        assert artifact["num_passed"] == sum(
            result["status"] == "PASSED" for result in results
        )
        # Retired models remain part of historical evidence, not the release API.
        assert {result["torch_name"] for result in results} <= artifact["catalog_coverage"].keys()
        assert {result["case_id"] for result in results} == {
            case["id"] for case in artifact["metadata"]["executed_inventory"]
        }
        assert manifest["verification"][f"{scope}_cases_total"] == artifact["num_cases"]
        assert manifest["verification"][f"{scope}_cases_passed"] == artifact["num_passed"]

        for model_name, model in manifest["models"].items():
            model_results = [result for result in results if result["torch_name"] == model_name]
            if "validation" not in model:
                assert not model_results, f"Missing validation metadata for {model_name}"
                continue
            validation = model["validation"][scope]
            assert validation["case_ids"] == [result["case_id"] for result in model_results]
            assert validation["cases_total"] == len(model_results)
            assert validation["cases_passed"] == sum(
                result["status"] == "PASSED" for result in model_results
            )
            assert validation["verification_scopes"] == sorted(
                {result["verification_scope"] for result in model_results
                 if result.get("verification_scope") is not None}
            )
            polarization_errors = [
                result[polarization]["complex_relative_l2"]
                for result in model_results
                for polarization in ("plus", "cross")
                if polarization in result
            ]
            projection_errors = [
                projection["complex_relative_l2"]
                for result in model_results
                for projection in result.get("projections", {}).values()
            ]
            assert validation["max_polarization_complex_relative_l2"] == max(
                polarization_errors, default=None
            )
            assert validation["max_projection_complex_relative_l2"] == max(
                projection_errors, default=None
            )


def test_frozen_inventory_immutability():
    """Verify that mutating convenience globals does not alter execution or canonical specifications."""
    import tests.parity_harness as ph
    orig_val = ph.DETERMINISTIC_HELD_OUT_CASES[0]["params"]["mass1"]
    try:
        ph.DETERMINISTIC_HELD_OUT_CASES[0]["params"]["mass1"] = 9999.0
        # Freshly deserialized specs must remain unchanged
        canonical = ph.get_canonical_holdout_cases()
        assert canonical[0]["params"]["mass1"] == orig_val, "Frozen deserialized specs must not be affected by global mutation"

        # Default run must use clean canonical specs
        suite_res = ph.run_full_parity_suite(scope="holdout", strict=False)
        assert suite_res["inventory_valid"] is True
        assert suite_res["metadata"]["executed_inventory"][0]["params"]["mass1"] == orig_val
    finally:
        ph.DETERMINISTIC_HELD_OUT_CASES[0]["params"]["mass1"] = orig_val


@pytest.mark.parametrize("corruption", ["nan", "shape"])
def test_invalid_projection_returns_failure(monkeypatch, corruption):
    import tests.parity_harness as ph
    original = ph.project_strain
    def corrupt(*args, **kwargs):
        strain = original(*args, **kwargs)
        if corruption == "shape":
            return strain[:-1]
        strain = strain.clone()
        strain[len(strain) // 2] = complex(float("nan"), 0)
        return strain
    monkeypatch.setattr(ph, "project_strain", corrupt)
    result = ph.run_oracle_comparison(ph.get_canonical_analytic_cases()[0], strict=True)
    assert result["status"] == "FAILED"
    assert "Projection validation failed" in result["reason"]

@pytest.mark.parametrize("df", [0, -1, float("nan"), float("inf")])
def test_invalid_grid_rejected_before_oracle(monkeypatch, df):
    import tests.parity_harness as ph
    import lalsimulation
    def unexpected_oracle(*args, **kwargs):
        pytest.fail("Invalid frequency grid reached oracle")
    monkeypatch.setattr(lalsimulation, "SimInspiralChooseFDWaveform", unexpected_oracle)
    case = ph.get_canonical_analytic_cases()[0]
    case["grid"]["df"] = df
    result = ph.run_oracle_comparison(case, strict=True)
    assert result["status"] == "FAILED"
    assert "grid" in result["reason"]


@pytest.mark.parametrize("field,value", [
    ("oracle_type", "native_eob_reference"),
    ("verification_scope", "dominant_mode_only"),
    ("oracle_options", {"mode": [2, 2]}),
    ("oracle_api", "SimInspiralFD"),
    ("oracle_fhigh", 2048.0),
])
def test_reference_specification_tampering_rejected(field, value):
    cases = get_canonical_analytic_cases()
    cases[0][field] = value
    result = run_full_parity_suite(scope="analytic", cases=cases)
    assert not result["inventory_valid"]
    assert result["results"] == []


@pytest.mark.parametrize("value", [1.e-5, .35, float("inf"), float("nan"), 0, -1, True])
def test_weakened_or_invalid_gate_rejected_before_generation(monkeypatch, value):
    import tests.parity_harness as ph
    monkeypatch.setattr(ph, "get_fd_waveform", lambda *a, **k: pytest.fail("Reached generator"))
    case = get_canonical_analytic_cases()[0]
    case["expected_gate"]["complex_relative_l2"] = value
    result = run_oracle_comparison(case, strict=True)
    assert result["status"] == "FAILED"
    assert "parity gate" in result["reason"]


def test_native_references_never_count_as_external_parity(monkeypatch):
    import tests.parity_harness as ph
    monkeypatch.setattr(ph, "get_fd_waveform", lambda *a, **k: pytest.fail("Reached generator"))
    native = [c for c in get_canonical_catalog_cases()
              if c.get("oracle_type", "lal") not in {"lal", "lal_ringdown_td"}]
    assert native == []
    old_eob = get_canonical_analytic_cases()[0]
    old_eob["oracle_type"] = "native_eob_reference"
    native.append(old_eob)
    for case in native:
        result = run_oracle_comparison(case, strict=True)
        assert result["status"] == "UNRESOLVED_ORACLE"
        assert "plus" not in result


def test_kerr_uses_independent_td_reference_contract():
    case = next(c for c in get_canonical_catalog_cases()
                if c["torch_name"] == "KerrRingdown")
    assert case["oracle_type"] == "lal_ringdown_td"
    assert case["oracle_api"] == "SimBlackHoleRingdown"


def test_kerr_convergence_failure_is_not_accepted(monkeypatch):
    from tests import kerr_public_oracle
    import tests.parity_harness as ph
    monkeypatch.setattr(kerr_public_oracle, "public_ringdown_fd_converged",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forced convergence failure")))
    case = next(c for c in ph.get_canonical_catalog_cases()
                if c["torch_name"] == "KerrRingdown")
    result = ph.run_oracle_comparison(case, strict=True)
    assert result["status"] == "FAILED"
    assert "convergence failure" in result["reason"]


@pytest.mark.parametrize("field,value", [
    ("oracle_api", "untrusted_callable"),
    ("oracle_fhigh", 1.0),
    ("oracle_fhigh", float("nan")),
    ("oracle_fhigh", "bad"),
])
def test_invalid_external_adapter_rejected(monkeypatch, field, value):
    import tests.parity_harness as ph
    monkeypatch.setattr(ph, "get_fd_waveform", lambda *a, **k: pytest.fail("Reached generator"))
    case = get_canonical_analytic_cases()[0]
    case[field] = value
    result = run_oracle_comparison(case, strict=True)
    assert result["status"] == "FAILED"
    assert "oracle" in result["reason"]


def test_holdout_inventory_is_distinct_and_frozen():
    import tests.parity_harness as ph
    holdouts = ph.get_canonical_holdout_cases()
    canonical = ph.get_canonical_catalog_cases()
    assert len(holdouts) == 16
    assert not {c["id"] for c in holdouts} & {c["id"] for c in canonical}
    assert all(c["params"] != base["params"] for c in holdouts for base in canonical)
    assert ph.validate_case_inventory(holdouts, scope="holdout")[0]
    holdouts[0]["params"]["mass1"] += 1
    assert not ph.validate_case_inventory(holdouts, scope="holdout")[0]
    assert all(c["expected_gate"] == ph.PARITY_GATE
               for c in canonical + ph.get_canonical_holdout_cases())


def test_finite_time_reference_dc_does_not_bypass_native_or_inband_checks():
    import tests.parity_harness as ph
    f = torch.arange(8, dtype=torch.float64)
    native = torch.ones(8, dtype=torch.complex128)
    native[0] = 0
    reference = torch.ones_like(native)
    args = (native, reference, f, 1., 1., 7.)
    assert not ph.validate_waveform_and_grid(*args)[0]
    assert ph.validate_waveform_and_grid(*args, require_reference_zero_dc=False)[0]
    native[0] = 1
    assert not ph.validate_waveform_and_grid(*args, require_reference_zero_dc=False)[0]
    native[0], native[3] = 0, complex(float('nan'), 0)
    assert not ph.validate_waveform_and_grid(*args, require_reference_zero_dc=False)[0]


@pytest.mark.parametrize('value', [True, 102.5, 999])
def test_invalid_precession_version_rejected(monkeypatch, value):
    import tests.parity_harness as ph
    monkeypatch.setattr(ph, 'get_fd_waveform', lambda *a, **k: pytest.fail('Reached generator'))
    case = next(c for c in ph.get_canonical_model_cases() if c['torch_name'] == 'IMRPhenomXP')
    case['params']['prec_version'] = value
    result = ph.run_oracle_comparison(case, strict=True)
    assert result['status'] == 'FAILED'
    assert 'precession version' in result['reason']

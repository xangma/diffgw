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


def test_missing_and_unresolved_catalog_coverage():
    """Verify that full catalog parity suite fails honestly when unresolved models or math gaps exist."""
    suite_res = run_full_parity_suite(scope="catalog", strict=False)
    assert suite_res["catalog_passed"] is False, "Catalog parity must not certify while gaps remain"
    assert suite_res["suite_passed"] is False, "suite_passed must equal catalog_passed for catalog scope"
    assert len(suite_res["catalog_coverage"]) >= 17
    assert set(MANIFEST_MODELS).issubset(set(suite_res["catalog_coverage"].keys()))
    assert suite_res["catalog_coverage"]["SEOBNR_ROM"]["status"] == "UNRESOLVED_ORACLE"
    assert suite_res["catalog_coverage"]["KerrRingdown"]["status"] == "UNRESOLVED_ORACLE"


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


def test_manifest_consistency_and_drift_detection():
    """Verify that docs/model_manifest.json matches harness models, and simulated drift causes catalog failure."""
    import tests.parity_harness as ph
    actual_manifest_models = ph.get_manifest_models()
    assert len(actual_manifest_models) == 17, f"Expected 17 models in docs/model_manifest.json, found {len(actual_manifest_models)}"
    assert set(actual_manifest_models) == set(ph.MANIFEST_MODELS), "Manifest models must match harness MANIFEST_MODELS"

    # Injected manifest drift: add a fake model to manifest output
    orig_get = ph.get_manifest_models
    try:
        ph.get_manifest_models = lambda: sorted(actual_manifest_models + ["DriftedModel_v1"])
        suite_res = ph.run_full_parity_suite(scope="catalog", strict=False)
        assert suite_res["catalog_passed"] is False, "Catalog drift must fail catalog acceptance"
        assert len(suite_res.get("manifest_drift_errors", [])) > 0, "Drift error must be recorded"
        assert any("DriftedModel_v1" in err for err in suite_res["manifest_drift_errors"])
    finally:
        ph.get_manifest_models = orig_get


def test_frozen_inventory_immutability():
    """Verify that mutating convenience globals does not alter execution or canonical specifications."""
    import tests.parity_harness as ph
    orig_val = ph.DETERMINISTIC_HELD_OUT_CASES[0]["params"]["mass1"]
    try:
        ph.DETERMINISTIC_HELD_OUT_CASES[0]["params"]["mass1"] = 9999.0
        # Freshly deserialized specs must remain unchanged
        canonical = ph.get_canonical_analytic_cases()
        assert canonical[0]["params"]["mass1"] == orig_val, "Frozen deserialized specs must not be affected by global mutation"

        # Default run must use clean canonical specs
        suite_res = ph.run_full_parity_suite(scope="analytic", strict=False)
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

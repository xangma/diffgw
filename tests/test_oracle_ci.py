"""The CI gate must distinguish passing comparisons from catalog coverage."""

from scripts.run_oracle_ci import assess_report


def _report():
    results = [{"case_id": f"case-{index}", "status": "PASSED"} for index in range(19)]
    return {
        "scope": "catalog",
        "strict": True,
        "inventory_valid": True,
        "inventory_errors": [],
        "manifest_drift_errors": [],
        "results": results,
        "num_cases": 19,
        "num_passed": 19,
        "cases_passed": True,
        "catalog_passed": False,
        "catalog_coverage": {
            "TaylorF2": {"status": "PASSED"},
            "TaylorF2_HM": {"status": "UNRESOLVED_ORACLE"},
        },
    }


def test_unresolved_catalog_is_reported_without_hiding_passing_cases():
    errors, gaps = assess_report(_report(), "catalog")
    assert errors == []
    assert gaps == {"TaylorF2_HM": "UNRESOLVED_ORACLE"}


def test_failed_case_or_inventory_change_fails_ci_gate():
    report = _report()
    report["results"][0]["status"] = "FAILED"
    report["results"].pop()
    report["manifest_drift_errors"] = ["new model"]
    errors, _ = assess_report(report, "catalog")
    assert any("strict comparison failed" in error for error in errors)
    assert any("frozen cases" in error for error in errors)
    assert any("manifest drift" in error for error in errors)

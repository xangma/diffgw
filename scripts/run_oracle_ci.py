#!/usr/bin/env python3
"""Gate frozen LAL comparisons while reporting unresolved catalog coverage.

The catalog's ``catalog_passed`` field describes whole-model coverage. It can
remain false when every executable comparison passes, so CI gates the frozen
case inventory and its strict results separately. The raw reports retain the
whole-catalog status and every coverage gap.
"""

import json
from pathlib import Path

from tests.parity_harness import SOVEREIGN_STATUSES, run_full_parity_suite


# Deliberate inventory sizes. Changing a frozen campaign requires review here.
EXPECTED_CASES = {"catalog": 19, "holdout": 14}
ACCEPTED_CASE_STATUSES = {"PASSED", *SOVEREIGN_STATUSES}
REPORTED_COVERAGE_GAPS = {"UNRESOLVED_ORACLE", "MISSING_COVERAGE"}


def assess_report(report, scope):
    """Return gate errors and separately disclosed catalog coverage gaps."""
    errors = []
    if report.get("scope") != scope or report.get("strict") is not True:
        errors.append(f"{scope}: wrong scope or non-strict execution")
    if not report.get("inventory_valid") or report.get("inventory_errors"):
        errors.append(f"{scope}: frozen inventory invalid: {report.get('inventory_errors')}")
    if report.get("manifest_drift_errors"):
        errors.append(f"{scope}: model manifest drift: {report['manifest_drift_errors']}")

    results = report.get("results", [])
    expected = EXPECTED_CASES[scope]
    if report.get("num_cases") != expected or len(results) != expected:
        errors.append(f"{scope}: expected {expected} frozen cases, got {report.get('num_cases')} / {len(results)} results")
    failed = [row.get("case_id") for row in results if row.get("status") not in ACCEPTED_CASE_STATUSES]
    if failed or report.get("cases_passed") is not True:
        errors.append(f"{scope}: strict comparison failed: {failed}")
    if report.get("num_passed") != len(results):
        errors.append(f"{scope}: passed-case count does not match results")

    gaps = {}
    if scope == "catalog":
        coverage = report.get("catalog_coverage", {})
        if not coverage:
            errors.append("catalog: missing coverage report")
        for model, entry in coverage.items():
            status = entry.get("status")
            if status in REPORTED_COVERAGE_GAPS:
                gaps[model] = status
            elif status not in ACCEPTED_CASE_STATUSES:
                errors.append(f"catalog: {model} has unexpected coverage status {status}")
    elif report.get("suite_passed") is not True:
        errors.append("holdout: suite did not pass")
    return errors, gaps


def main():
    summaries = {}
    failed = False
    for scope, filename in (("catalog", "parity_results.json"), ("holdout", "parity_holdout_results.json")):
        report = run_full_parity_suite(scope=scope, strict=True)
        Path(filename).write_text(json.dumps(report, indent=2) + "\n")
        errors, gaps = assess_report(report, scope)
        summaries[scope] = {
            "case_gate_passed": not errors,
            "cases": report.get("num_cases"),
            "strict_cases_passed": report.get("num_passed"),
            "exact_lal_cases": report.get("num_exact_parity"),
            "sovereign_cases": report.get("num_sovereign"),
            "whole_catalog_passed": report.get("catalog_passed") if scope == "catalog" else None,
            "coverage_gaps": gaps,
            "errors": errors,
        }
        failed |= bool(errors)
        print(f"{scope}: {report.get('num_passed')}/{report.get('num_cases')} strict cases; "
              f"case gate {'FAILED' if errors else 'passed'}; coverage gaps: {gaps}")
        for error in errors:
            print(error)
    Path("parity_ci_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

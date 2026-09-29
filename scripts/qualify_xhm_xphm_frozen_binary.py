#!/usr/bin/env python3
"""Run the four frozen XHM/XPHM catalog and holdout LAL binary checks.

This compares waveform outputs through public installed LALSimulation APIs.
It does not inspect implementation source or fit native coefficients.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

from tests.parity_harness import (
    get_canonical_model_cases,
    get_canonical_holdout_cases,
    run_oracle_comparison,
)


CASE_IDS = (
    "IMRPhenomXHM_diagnostic",
    "IMRPhenomXPHM_diagnostic",
    "IMRPhenomXHM_holdout_6",
    "IMRPhenomXPHM_holdout",
)
PINNED_DISTRIBUTION = "7.26.15"
GATE = {"complex_relative_l2": 1e-6, "mismatch": 1e-8}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    installed = importlib.metadata.version("lalsuite")
    if installed != PINNED_DISTRIBUTION:
        parser.error(f"requires lalsuite=={PINNED_DISTRIBUTION}; found {installed}")
    inventory = get_canonical_model_cases() + get_canonical_holdout_cases()
    selected = {case["id"]: case for case in inventory if case["id"] in CASE_IDS}
    if tuple(case_id for case_id in CASE_IDS if case_id in selected) != CASE_IDS:
        parser.error("frozen XHM/XPHM case inventory is incomplete")
    if any(selected[case_id]["expected_gate"] != GATE for case_id in CASE_IDS):
        parser.error("frozen XHM/XPHM acceptance gate drifted")
    rows = [run_oracle_comparison(selected[case_id], strict=True) for case_id in CASE_IDS]
    report = {
        "schema": "diffgw.xhm-xphm.frozen-binary/v1",
        "host": platform.node(),
        "python": platform.python_version(),
        "lalsuite_distribution": installed,
        "gate": GATE,
        "cases": rows,
        # The parity harness marks some failed NR-calibrated models with a
        # sovereign status.  Only the actual strict binary verdict counts.
        "passed": sum(row.get("legacy_oracle_passed") is True for row in rows),
        "total": len(rows),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(row["case_id"], "binary_passed=", row.get("legacy_oracle_passed"),
              "harness_status=", row["status"],
              {polarization: {key: row.get(polarization, {}).get(key)
                              for key in ("complex_relative_l2", "flat_mismatch")}
               for polarization in ("plus", "cross")}, flush=True)
    print(f'{report["passed"]}/{report["total"]} passed', flush=True)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

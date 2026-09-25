"""Isolate Hamiltonian derivative and calibration effects in selected public FD cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from unittest.mock import patch

from . import diagnose_hamiltonian_public_fd_intervention as combined
from . import qualify_generator as q
from .diagnose_native_public_fd import FROZEN_ROWS
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/hamiltonian_fd_factorial_plan.json"
FD_PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/native_public_fd_plan.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    expected_arms = {
        "lal_log_derivative_only": {
            "diagnostic_hamiltonian_log_derivative": "lal_v4",
            "diagnostic_hamiltonian_calibration": "published"},
        "lal_rounded_calibration_only": {
            "diagnostic_hamiltonian_log_derivative": "analytic",
            "diagnostic_hamiltonian_calibration": "lal_v4_rounded"},
    }
    if (plan["schema"] != "diffgw.seobnrv4.hamiltonian-fd-factorial-plan.v1"
            or plan["release_qualified"] is not False
            or plan["rows"] != ["safe_frequency_cap", "strong_antialigned"]
            or plan["arms"] != expected_arms):
        parser.error("frozen factorial plan changed")
    baseline_path = q.ROOT / plan["baseline_report"]
    combined_path = q.ROOT / plan["combined_report"]
    for path, digest in (
        (baseline_path, plan["baseline_report_sha256"]),
        (combined_path, plan["combined_report_sha256"]),
        (Path(combined.__file__), plan["combined_script_sha256"]),
    ):
        if q.sha256(path) != digest:
            parser.error(f"frozen input differs: {path}")
    baseline = json.loads(baseline_path.read_text())
    prior_combined = json.loads(combined_path.read_text())
    if prior_combined["status"] != "measured" or prior_combined["errors"]:
        parser.error("combined intervention did not complete")
    frozen_binary = baseline["lal_binary"]
    if q.lal_binary_identity(frozen_binary["lal_version"],
                             frozen_binary["lalsimulation_version"]) != frozen_binary:
        parser.error("pinned public binary differs")
    if q.sha256(FD_PLAN) != baseline["plan_sha256"]:
        parser.error("frozen public FD plan differs")
    fd_plan = json.loads(FD_PLAN.read_text())
    rows = {row["name"]: row for row in FROZEN_ROWS}
    prior = {row["name"]: row for row in baseline["rows"]}
    both = {row["name"]: row for row in prior_combined["rows"]}
    output = {
        "schema": "diffgw.seobnrv4.hamiltonian-fd-factorial.v1",
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "baseline_report_sha256": q.sha256(baseline_path),
        "combined_report_sha256": q.sha256(combined_path),
        "process": {"host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid()},
        "rows": [], "errors": [],
    }
    print(json.dumps({"process": output["process"], "status": "started"}), flush=True)
    for name in plan["rows"]:
        result = {
            "name": name,
            "baseline_physical_fixed_distance_relative_l2": prior[name][
                "physical_fixed_distance_fit"]["relative_l2"],
            "both_physical_fixed_distance_relative_l2": both[name][
                "physical_fixed_distance_relative_l2"],
            "both_native_fd_sha256": both[name]["native_fd_sha256"],
            "arms": {},
        }
        for label, choice in plan["arms"].items():
            def generator(*gen_args, _choice=choice, **gen_kwargs):
                return reconstruct_seobnrv4_22_adaptive(
                    *gen_args, **gen_kwargs, **_choice)

            try:
                # The pinned comparison helper owns the physical fit and oracle
                # hash checks; only its development generator callback changes.
                with patch.object(combined, "_intervention_generator", generator):
                    arm = combined._row(rows[name], prior[name], fd_plan)
                result["arms"][label] = arm
                print(json.dumps({"name": name, "arm": label,
                                  "relative_l2": arm["physical_fixed_distance_relative_l2"],
                                  "native_fd_sha256": arm["native_fd_sha256"]}), flush=True)
            except Exception as exc:
                output["errors"].append({"name": name, "arm": label, "error": repr(exc)})
                print(f"{name}/{label}: {exc!r}", flush=True)
        output["rows"].append(result)
    output["status"] = "measured" if not output["errors"] else "failed"
    args.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0 if not output["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

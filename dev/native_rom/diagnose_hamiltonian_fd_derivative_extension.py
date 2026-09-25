"""Extend the historical Hamiltonian-derivative FD probe to two frozen rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from unittest.mock import patch

from . import diagnose_hamiltonian_public_fd_intervention as comparison
from . import qualify_generator as q
from .diagnose_native_public_fd import FROZEN_ROWS
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/hamiltonian_fd_derivative_extension_plan.json"
FD_PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/native_public_fd_plan.json"


def _generator(*args, **kwargs):
    return reconstruct_seobnrv4_22_adaptive(
        *args, **kwargs,
        diagnostic_hamiltonian_log_derivative="lal_v4",
        diagnostic_hamiltonian_calibration="published",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.hamiltonian-fd-derivative-extension-plan.v1"
            or plan["release_qualified"] is not False
            or plan["rows"] != ["mixed_spin", "q4_nonspinning"]
            or plan["choice"] != {
                "diagnostic_hamiltonian_log_derivative": "lal_v4",
                "diagnostic_hamiltonian_calibration": "published"}):
        parser.error("frozen derivative-extension plan changed")
    baseline_path = q.ROOT / plan["baseline_report"]
    factorial_path = q.ROOT / plan["factorial_report"]
    for path, digest in (
        (baseline_path, plan["baseline_report_sha256"]),
        (factorial_path, plan["factorial_report_sha256"]),
        (Path(comparison.__file__), plan["comparison_helper_sha256"]),
    ):
        if q.sha256(path) != digest:
            parser.error(f"frozen input differs: {path}")
    baseline = json.loads(baseline_path.read_text())
    factorial = json.loads(factorial_path.read_text())
    if factorial["status"] != "measured" or factorial["errors"]:
        parser.error("previous two-row factorial did not complete")
    frozen_binary = baseline["lal_binary"]
    if q.lal_binary_identity(frozen_binary["lal_version"],
                             frozen_binary["lalsimulation_version"]) != frozen_binary:
        parser.error("pinned public binary differs")
    if q.sha256(FD_PLAN) != baseline["plan_sha256"]:
        parser.error("frozen public FD plan differs")
    fd_plan = json.loads(FD_PLAN.read_text())
    rows = {row["name"]: row for row in FROZEN_ROWS}
    prior = {row["name"]: row for row in baseline["rows"]}
    output = {
        "schema": "diffgw.seobnrv4.hamiltonian-fd-derivative-extension.v1",
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "baseline_report_sha256": q.sha256(baseline_path),
        "factorial_report_sha256": q.sha256(factorial_path),
        "process": {"host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid()},
        "rows": [], "errors": [],
    }
    print(json.dumps({"process": output["process"], "status": "started"}), flush=True)
    for name in plan["rows"]:
        try:
            with patch.object(comparison, "_intervention_generator", _generator):
                result = comparison._row(rows[name], prior[name], fd_plan)
            output["rows"].append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        except Exception as exc:
            output["errors"].append({"name": name, "error": repr(exc)})
            print(f"{name}: {exc!r}", flush=True)
    output["status"] = "measured" if not output["errors"] else "failed"
    args.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0 if not output["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

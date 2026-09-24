"""Compare public v4 q10 NQC coefficients across the two finest output grids."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

from . import qualify_generator as q
from .diagnose_public_nqc_mode_boundary import public_call


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/near_extreme_public_nqc_grid_plan.json"
ROW = "q10_both_near_extreme_aligned"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("evidence path already exists")
    plan = json.loads(PLAN.read_text())
    dynamics_path = q.ROOT / plan["dynamics_result"]["path"]
    baseline_path = q.ROOT / plan["baseline_parity"]["path"]
    if (plan["schema"] != "diffgw.seobnrv4.near-extreme-public-nqc-grid-plan.v1"
            or plan["row"] != ROW
            or plan["dt_levels_M"] != [0.03125, 0.015625]
            or plan["release_qualified"] is not False
            or q.sha256(dynamics_path) != plan["dynamics_result"]["sha256"]
            or q.sha256(baseline_path) != plan["baseline_parity"]["sha256"]):
        parser.error("frozen plan or inputs changed")
    prior = json.loads(dynamics_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    pin = prior["lal_binary"]
    if (prior["status"] != "diagnostic_complete"
            or baseline["status"] != "diagnostic_complete"
            or pin != baseline["lal_binary"]
            or q.lal_binary_identity(pin["lal_version"],
                                     pin["lalsimulation_version"]) != pin):
        parser.error("pinned public binary or prerequisites changed")
    report = dict(schema="diffgw.seobnrv4.near-extreme-public-nqc-grid.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  dynamics_result_sha256=q.sha256(dynamics_path),
                  lal_binary=pin, row=ROW, levels={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        row = baseline["rows"][ROW]["row"]
        coefficient_sets = []
        for dt in plan["dt_levels_M"]:
            output, coefficients = public_call(row, dt, 1)
            if list(output) != [None, None, None] or not np.isfinite(coefficients).all():
                raise ValueError("invalid public NQC coefficient-only output")
            coefficient_sets.append(coefficients)
            report["levels"][f"dt_{dt:.6f}M"] = coefficients.tolist()
            q.write_evidence(args.json_out, report)
        coarse, fine = coefficient_sets
        absolute = np.abs(coarse - fine)
        relative = absolute / np.maximum(np.abs(fine), 1e-12)
        report["comparison"] = dict(
            absolute_difference=absolute.tolist(),
            relative_difference=relative.tolist(),
            max_absolute_difference=float(np.max(absolute)),
            max_relative_difference=float(np.max(relative)),
            direct_postpeak_complex_relative_l2=prior["comparison"][
                "direct_postpeak_complex_relative_l2"])
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])))
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

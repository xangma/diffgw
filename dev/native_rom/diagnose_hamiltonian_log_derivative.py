"""Ablate the historical radial log-derivative convention on frozen public states."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_basis_trace import _physics
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from .eob.torch.historical_hamiltonian import HistoricalLogDerivative


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/hamiltonian_log_derivative_plan.json"
COORDINATES = ("r", "phi", "pr_star", "pphi")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_late_field"]
    prior = json.loads(prior_path.read_text())
    source = q.ROOT / "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py"
    if (plan["schema"] != "diffgw.seobnrv4.hamiltonian-log-derivative-plan.v1"
            or q.sha256(prior_path) != plan["prior_late_field_sha256"]
            or q.sha256(source) != plan["native_hamiltonian_sha256"]
            or prior["status"] != "diagnostic_complete"
            or len(prior["grids"]) != 2):
        parser.error("frozen plan, native source, or prior evidence differs")
    row = prior["row"]
    baseline, _ = _physics(row)
    base_ham = baseline.__self__.hamiltonian
    historical_ham = HistoricalLogDerivative(
        base_ham.nu, base_ham.chi1, base_ham.chi2, delta=base_ham.delta)
    historical = EOBTrajectoryIntegrator(
        historical_ham, baseline.__self__.rr).vector_field
    report = dict(
        schema="diffgw.seobnrv4.hamiltonian-log-derivative.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        historical_hamiltonian_parity_established=False,
        plan_sha256=q.sha256(PLAN), prior_sha256=q.sha256(prior_path),
        script_sha256=q.sha256(__file__), source_sha256=q.sha256(source),
        diagnostic_hamiltonian_sha256=q.sha256(
            q.ROOT / "dev/native_rom/eob/torch/historical_hamiltonian.py"),
        historical_source=plan["historical_source"],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                     pid=os.getpid(), argv=sys.argv), row=row, grids=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for prior_grid in prior["grids"]:
            grid = dict(step_M=prior_grid["step_M"],
                        public_array_sha256=prior_grid["array_sha256"], points=[])
            for point in prior_grid["points"]:
                state = torch.tensor([point["state"][key] for key in COORDINATES],
                                     dtype=torch.float64)
                with torch.no_grad():
                    base = baseline(state).numpy()
                    changed = historical(state).numpy()
                stored = np.array([point["native_derivative"][key]
                                   for key in COORDINATES])
                if not np.allclose(base, stored, rtol=2e-11, atol=2e-12):
                    raise ValueError("baseline vector field no longer replays")
                sampled = np.array([point["sampled_derivative"][key]
                                    for key in COORDINATES])
                grid["points"].append(dict(
                    target_Momega=point["target_Momega"], state=point["state"],
                    sampled_derivative=point["sampled_derivative"],
                    native_baseline=dict(zip(COORDINATES, base.tolist())),
                    native_historical_derivative=dict(zip(COORDINATES, changed.tolist())),
                    baseline_relative_to_sampled=dict(zip(
                        COORDINATES, ((base-sampled)/np.maximum(abs(sampled), 1e-15)).tolist())),
                    historical_relative_to_sampled=dict(zip(
                        COORDINATES, ((changed-sampled)/np.maximum(abs(sampled), 1e-15)).tolist()))))
            report["grids"].append(grid)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(step_M=grid["step_M"], late=[
                {"target": item["target_Momega"], "baseline_phi": item["baseline_relative_to_sampled"]["phi"],
                 "historical_phi": item["historical_relative_to_sampled"]["phi"],
                 "baseline_pr": item["baseline_relative_to_sampled"]["pr_star"],
                 "historical_pr": item["historical_relative_to_sampled"]["pr_star"]}
                for item in grid["points"][-3:]])), flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

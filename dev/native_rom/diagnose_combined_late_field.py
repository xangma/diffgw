"""Test both source-motivated late SEOBNRv4 choices at frozen public states."""

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
from .eob.torch.historical_hamiltonian import HistoricalLogDerivative
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/combined_late_field_plan.json"
COORDINATES = ("r", "phi", "pr_star", "pphi")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    ham_path = q.ROOT / plan["prior_hamiltonian"]
    mode_path = q.ROOT / plan["prior_mode_33"]
    if (plan["schema"] != "diffgw.seobnrv4.combined-late-field-plan.v1"
            or q.sha256(ham_path) != plan["prior_hamiltonian_sha256"]
            or q.sha256(mode_path) != plan["prior_mode_33_sha256"]
            or plan["mode_33_higher_spin"] != "lal_v4_rho"):
        parser.error("frozen plan or prior evidence differs")
    ham_prior, mode_prior = (json.loads(path.read_text()) for path in (ham_path, mode_path))
    if (ham_prior["status"] != "diagnostic_complete"
            or mode_prior["status"] != "diagnostic_complete"
            or ham_prior["row"] != mode_prior["row"]):
        parser.error("prior diagnostics differ")
    row = ham_prior["row"]
    mass1, mass2, chi1, chi2 = (torch.tensor(row[key], dtype=torch.float64)
                                for key in ("mass1", "mass2", "chi1", "chi2"))
    x1, x2 = mass1/(mass1+mass2), mass2/(mass1+mass2)
    ham = HistoricalLogDerivative(x1*x2, chi1, chi2, delta=x1-x2)
    choice = mode_prior["choice"]
    flux = SEOBNRv4PublishedFlux(
        spin_truncation=choice["spin_truncation"],
        retain_pan_cubic_spin=choice["retain_pan_cubic_spin"],
        mode_21_higher_spin=choice["mode_21_higher_spin"],
        mode_33_higher_spin="lal_v4_rho")
    field = EOBTrajectoryIntegrator(ham, flux).vector_field
    report = dict(
        schema="diffgw.seobnrv4.combined-late-field.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        historical_vector_field_parity_established=False,
        plan_sha256=q.sha256(PLAN), hamiltonian_sha256=q.sha256(ham_path),
        mode_33_sha256=q.sha256(mode_path), script_sha256=q.sha256(__file__),
        source_sha256={name: q.sha256(q.ROOT / name) for name in (
            "dev/native_rom/eob/torch/historical_hamiltonian.py",
            "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
            "src/torchwave/approximants/eob/seobnrv4_flux.py")},
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv), row=row, grids=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for ham_grid, mode_grid in zip(ham_prior["grids"], mode_prior["grids"]):
            if (ham_grid["step_M"] != mode_grid["step_M"]
                    or len(ham_grid["points"]) != len(mode_grid["points"])):
                raise ValueError("prior grid alignment differs")
            grid = dict(step_M=ham_grid["step_M"], points=[])
            for ham_point, mode_point in zip(ham_grid["points"], mode_grid["points"]):
                if ham_point["state"] != mode_point["state"]:
                    raise ValueError("prior state alignment differs")
                state = torch.tensor([ham_point["state"][key] for key in COORDINATES],
                                     dtype=torch.float64)
                with torch.no_grad():
                    current = field(state).numpy()
                sampled = np.array([ham_point["sampled_derivative"][key]
                                    for key in COORDINATES])
                grid["points"].append(dict(
                    target_Momega=ham_point["target_Momega"], state=ham_point["state"],
                    native_derivative=dict(zip(COORDINATES, current.tolist())),
                    relative_to_sampled=dict(zip(
                        COORDINATES, ((current-sampled)/np.maximum(abs(sampled), 1e-15)).tolist()))))
            report["grids"].append(grid)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(step_M=grid["step_M"], relative=[
                {"target": point["target_Momega"], **point["relative_to_sampled"]}
                for point in grid["points"]])), flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen same-state source placement check after correcting (3,2) rho."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import torch

from . import qualify_generator as q
from .diagnose_strong_spin_inspiral_field import COORDINATES, _physics
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/mode_21_31_32_flux_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior"]
    if (plan["schema"] != "diffgw.seobnrv4.mode-21-31-32-flux-plan.v1"
            or q.sha256(prior_path) != plan["prior_sha256"]):
        parser.error("frozen plan or prior differs")
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "diagnostic_complete":
        parser.error("prior flux check failed")
    torch.set_num_threads(1)
    ham, _, field = _physics(prior["row"])
    report = dict(
        schema="diffgw.seobnrv4.mode-21-31-32-flux.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), prior_sha256=q.sha256(prior_path),
        script_sha256=q.sha256(__file__),
        source_sha256={"src/torchwave/approximants/eob/seobnrv4_flux.py":
                       q.sha256(q.ROOT / "src/torchwave/approximants/eob/seobnrv4_flux.py")},
        historical_source=plan["historical_source"], row=prior["row"],
        points=[], errors=[], process=dict(host=socket.gethostname(),
            cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        for point in prior["points"]:
            state = torch.tensor([point["state"][key] for key in COORDINATES],
                                 dtype=torch.float64)
            with torch.no_grad():
                omega = field(state)[1]
                quantities = SEOBNRv4PublishedFlux.from_hamiltonian(
                    state, ham, omega=omega)
                nu = float(quantities["x1"]*quantities["x2"])
                sample = point["sampled_public_torque"]
                choices = {}
                for m21 in ("taylor_f", "rho_hybrid"):
                    for m31 in ("taylor_f", "lal_v4_rho"):
                        flux = SEOBNRv4PublishedFlux(
                            spin_truncation="generic-leading-ell8",
                            retain_pan_cubic_spin=True,
                            mode_21_higher_spin=m21,
                            mode_31_higher_spin=m31,
                            mode_32_spin_square="lal_v4_zero",
                            mode_33_higher_spin="lal_v4_rho")
                        torque = -float(flux.energy_flux(state, **quantities))/(nu*float(omega))
                        choices[f"{m21},{m31}"] = dict(
                            torque=torque, relative_to_sampled=torque/sample-1)
                baseline = choices["taylor_f,taylor_f"]["torque"]
                if abs(baseline-point["placements"]["lal_v4_zero"]["torque"]) > 2e-13*abs(baseline):
                    raise ValueError("rho32-only baseline failed exact replay")
                report["points"].append(dict(
                    target_Momega=point["target_Momega"],
                    actual_Momega=point["actual_Momega"],
                    sampled_public_torque=sample, choices=choices))
            q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

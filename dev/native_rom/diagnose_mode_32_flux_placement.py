"""Frozen same-state strong-spin (3,2) v4 rho spin-square diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import torch

from . import qualify_generator as q
from .diagnose_early_aligned_flux_choices import _mode_fluxes
from .diagnose_strong_spin_inspiral_field import COORDINATES, _physics
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/mode_32_flux_placement_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_field"]
    if (plan["schema"] != "diffgw.seobnrv4.mode-32-flux-placement-plan.v1"
            or q.sha256(prior_path) != plan["prior_field_sha256"]
            or plan["grid_step_M"] != 0.0625
            or plan["frequency_targets_Momega"] != [0.045, 0.06, 0.09, 0.12]):
        parser.error("frozen plan or prior field differs")
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "diagnostic_complete":
        parser.error("prior field incomplete")
    torch.set_num_threads(1)
    ham, _, field = _physics(prior["row"])
    points = prior["grids"][str(plan["grid_step_M"])]["targets"]
    report = dict(
        schema="diffgw.seobnrv4.mode-32-flux-placement.v1", scope=q.SCOPE,
        status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
        prior_sha256=q.sha256(prior_path), script_sha256=q.sha256(__file__),
        source_sha256={name: q.sha256(q.ROOT/name) for name in (
            "src/torchwave/approximants/eob/seobnrv4_flux.py",
            "src/torchwave/approximants/eob/factorized_modes.py")},
        historical_source=plan["historical_source"], row=prior["row"],
        points=[], errors=[], process=dict(host=socket.gethostname(),
            cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        for target, point in zip(plan["frequency_targets_Momega"], points, strict=True):
            if abs(point["frequency_Momega"]-target) > 1e-4:
                raise ValueError("public frequency target differs")
            state = torch.tensor([point["state"][key] for key in COORDINATES],
                                 dtype=torch.float64)
            with torch.no_grad():
                omega = field(state)[1]
                quantities = SEOBNRv4PublishedFlux.from_hamiltonian(
                    state, ham, omega=omega)
                nu = float(quantities["x1"]*quantities["x2"])
                rows = {}
                for placement in ("pan_f26", "lal_v4_zero"):
                    choice = dict(plan["baseline"], mode_32_spin_square=placement)
                    flux = SEOBNRv4PublishedFlux(**choice)
                    parts = _mode_fluxes(flux, state, quantities)
                    energy_flux = float(flux.energy_flux(state, **quantities))
                    rows[placement] = dict(
                        energy_flux=energy_flux,
                        torque=-energy_flux/(nu*float(omega)),
                        mode_32_flux=parts["3,2"],
                        mode_32_flux_fraction=parts["3,2"]/energy_flux)
                baseline = rows["pan_f26"]["torque"]
                sampled = point["sampled_derivative"]["pphi"]
                if abs(baseline-point["native_derivative"]["pphi"]) > 2e-13*abs(baseline):
                    raise ValueError("native baseline torque did not replay")
                report["points"].append(dict(
                    target_Momega=target, actual_Momega=point["frequency_Momega"],
                    state=point["state"], sampled_public_torque=sampled,
                    placements=rows,
                    baseline_relative_to_sampled=baseline/sampled-1,
                    historical_relative_to_sampled=rows["lal_v4_zero"]["torque"]/sampled-1,
                    historical_relative_to_baseline=rows["lal_v4_zero"]["torque"]/baseline-1))
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

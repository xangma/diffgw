"""Frozen strong-spin same-state mode-flux choice sensitivity; diagnostic only."""

from __future__ import annotations

import argparse
from itertools import product
import json
import math
import os
from pathlib import Path
import socket
import sys

import torch

from . import qualify_generator as q
from .diagnose_early_aligned_flux_choices import _mode_fluxes
from .diagnose_strong_spin_inspiral_field import COORDINATES, _physics
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/strong_spin_flux_choices_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["inspiral_field"]
    if (plan["schema"] != "diffgw.seobnrv4.strong-spin-flux-choices-plan.v1"
            or q.sha256(prior_path) != plan["inspiral_field_sha256"]
            or plan["grid_step_M"] != 0.0625
            or plan["frequency_targets_Momega"] != [0.045, 0.06, 0.09, 0.12]
            or plan["baseline"] != dict(
                spin_truncation=q.CHOICES["spin_truncation"],
                retain_pan_cubic_spin=True, mode_21_higher_spin="taylor_f",
                mode_33_higher_spin="lal_v4_rho")):
        parser.error("frozen plan or prior evidence differs")
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "diagnostic_complete":
        parser.error("prior field diagnostic incomplete")
    points = prior["grids"][str(plan["grid_step_M"])]["targets"]
    if len(points) != len(plan["frequency_targets_Momega"]):
        parser.error("prior target count differs")
    torch.set_num_threads(1)
    ham, _, field = _physics(prior["row"])
    source = "src/torchwave/approximants/eob/seobnrv4_flux.py"
    report = dict(
        schema="diffgw.seobnrv4.strong-spin-flux-choices.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), prior_sha256=q.sha256(prior_path),
        script_sha256=q.sha256(__file__), source_sha256={source: q.sha256(q.ROOT/source)},
        row=prior["row"], points=[], errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv))
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
                rows = []
                values = [plan["choices"][key] for key in plan["baseline"]]
                for combination in product(*values):
                    choice = dict(zip(plan["baseline"], combination, strict=True))
                    flux = SEOBNRv4PublishedFlux(**choice)
                    parts = _mode_fluxes(flux, state, quantities)
                    total = float(flux.energy_flux(state, **quantities))
                    if not math.isclose(total, sum(parts.values()), rel_tol=5e-14):
                        raise ValueError("mode flux contributions do not sum to total")
                    rows.append(dict(choice=choice, energy_flux=total,
                                     torque=-total/(nu*float(omega)), mode_flux=parts))
                baseline = next(item for item in rows
                                if item["choice"] == plan["baseline"])
                sampled = point["sampled_derivative"]["pphi"]
                if not math.isclose(baseline["torque"],
                                    point["native_derivative"]["pphi"],
                                    rel_tol=2e-13, abs_tol=1e-15):
                    raise ValueError("same-state baseline torque does not replay")
                for item in rows:
                    item["relative_to_baseline"] = item["torque"]/baseline["torque"]-1
                    item["relative_to_sampled_public"] = item["torque"]/sampled-1
                    item["mode_flux_delta_from_baseline"] = {
                        mode: value-baseline["mode_flux"][mode]
                        for mode, value in item["mode_flux"].items()
                        if value != baseline["mode_flux"][mode]}
                report["points"].append(dict(
                    target_Momega=target, actual_Momega=point["frequency_Momega"],
                    state=point["state"], sampled_public_torque=sampled,
                    baseline_mode_flux_fraction={
                        mode: value/baseline["energy_flux"]
                        for mode, value in baseline["mode_flux"].items()},
                    choices=rows))
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(
                target=target,
                baseline_relative=baseline["torque"]/sampled-1,
                closest=min(rows, key=lambda item: abs(
                    item["relative_to_sampled_public"]))["choice"])), flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

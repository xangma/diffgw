"""Frozen same-state sensitivity to explicit native aligned-spin flux choices."""

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
from .diagnose_boundary_strong_spin_basis_trace import _physics
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/early_aligned_flux_choices_plan.json"


def _mode_fluxes(flux, state, quantities):
    omega = float(quantities["omega"])
    modes = flux._signed_amplitudes(state, **quantities)
    return {f"{ell},{m}": float((m*omega)**2*amplitude.square()/(8*math.pi))
            for (ell, m), amplitude in modes.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_vector_field"]
    prior = json.loads(prior_path.read_text())
    baseline_choice = plan["baseline"]
    if (plan["schema"] != "diffgw.seobnrv4.early-aligned-flux-choices-plan.v1"
            or q.sha256(prior_path) != plan["prior_vector_field_sha256"]
            or prior["status"] != "diagnostic_complete"
            or plan["public_output_step_M"] != prior["grids"][1]["step_M"]
            or len(prior["grids"][1]["points"]) != len(plan["frequency_targets_Momega"])
            or baseline_choice != dict(spin_truncation=q.CHOICES["spin_truncation"],
                                       retain_pan_cubic_spin=True,
                                       mode_21_higher_spin="taylor_f")):
        parser.error("frozen flux-choice plan or prior vector-field evidence differs")
    field, _ = _physics(prior["row"])
    ham = field.__self__.hamiltonian
    report = dict(schema="diffgw.seobnrv4.early-aligned-flux-choices.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  historical_flux_parity_established=False,
                  plan_sha256=q.sha256(PLAN), prior_sha256=q.sha256(prior_path),
                  script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT/path) for path in (
                      "src/torchwave/approximants/eob/seobnrv4_flux.py",
                      "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py")},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv),
                  row=prior["row"], points=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for target, point in zip(plan["frequency_targets_Momega"],
                                 prior["grids"][1]["points"], strict=True):
            if abs(point["frequency_Momega"]-target) > 1e-5:
                raise ValueError("frozen public frequency target differs")
            s = point["state"]
            state = torch.tensor([s[k] for k in ("r", "phi", "pr_star", "pphi")],
                                 dtype=torch.float64)
            with torch.no_grad():
                omega = field(state)[1]
                quantities = SEOBNRv4PublishedFlux.from_hamiltonian(
                    state, ham, omega=omega)
                nu = float(quantities["x1"]*quantities["x2"])
                rows = []
                for spin, cubic, mode_21 in product(
                        plan["spin_truncation"], plan["retain_pan_cubic_spin"],
                        plan["mode_21_higher_spin"]):
                    choice = dict(spin_truncation=spin,
                                  retain_pan_cubic_spin=cubic,
                                  mode_21_higher_spin=mode_21)
                    flux = SEOBNRv4PublishedFlux(**choice)
                    parts = _mode_fluxes(flux, state, quantities)
                    total = float(flux.energy_flux(state, **quantities))
                    if not math.isclose(total, sum(parts.values()), rel_tol=5e-14):
                        raise ValueError("mode flux contributions do not sum to total")
                    rows.append(dict(choice=choice, energy_flux=total,
                                     torque=-total/(nu*float(omega)),
                                     mode_energy_flux=parts))
                baseline = next(row for row in rows if row["choice"] == baseline_choice)
                sampled = point["sampled_derivative"]["pphi"]
                if not math.isclose(baseline["torque"],
                                    point["native_derivative"]["pphi"],
                                    rel_tol=2e-13, abs_tol=1e-15):
                    raise ValueError("baseline native torque does not replay")
                for row in rows:
                    row["relative_to_baseline"] = row["torque"]/baseline["torque"]-1
                    row["relative_to_sampled_public"] = row["torque"]/sampled-1
                    row["mode_flux_delta_from_baseline"] = {
                        mode: value-baseline["mode_energy_flux"][mode]
                        for mode, value in row["mode_energy_flux"].items()
                        if value != baseline["mode_energy_flux"][mode]}
                report["points"].append(dict(target_Momega=target,
                                             actual_Momega=point["frequency_Momega"],
                                             state=s, sampled_public_torque=sampled,
                                             native_omega=float(omega),
                                             baseline_mode_flux_fraction={
                                                 mode: value/baseline["energy_flux"]
                                                 for mode, value in baseline["mode_energy_flux"].items()},
                                             choices=rows))
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(target=target, baseline_relative=
                                  baseline["torque"]/sampled-1,
                                  closest=min(rows, key=lambda row: abs(
                                      row["relative_to_sampled_public"]))["choice"])),
                  flush=True)
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

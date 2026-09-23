"""Frozen same-state sensitivity to orbital and circular-frequency inputs."""

from __future__ import annotations

import argparse
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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/early_aligned_flux_inputs_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_flux_choices"]
    prior = json.loads(prior_path.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.early-aligned-flux-inputs-plan.v1"
            or q.sha256(prior_path) != plan["prior_flux_choices_sha256"]
            or prior["status"] != "diagnostic_complete"
            or len(prior["points"]) != len(plan["frequency_targets_Momega"])):
        parser.error("frozen plan or prior flux-choice evidence differs")
    field, _ = _physics(prior["row"])
    ham = field.__self__.hamiltonian
    baseline_choice = dict(spin_truncation=q.CHOICES["spin_truncation"],
                           retain_pan_cubic_spin=True,
                           mode_21_higher_spin="taylor_f")
    flux = SEOBNRv4PublishedFlux(**baseline_choice)
    report = dict(schema="diffgw.seobnrv4.early-aligned-flux-inputs.v1",
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
                                 prior["points"], strict=True):
            if target != point["target_Momega"]:
                raise ValueError("frozen frequency targets differ")
            s = point["state"]
            state = torch.tensor([s[k] for k in ("r", "phi", "pr_star", "pphi")],
                                 dtype=torch.float64)
            with torch.no_grad():
                omega_native = field(state)[1]
                omega_sampled = torch.tensor(point["actual_Momega"], dtype=state.dtype)
                quantities = SEOBNRv4PublishedFlux.from_hamiltonian(
                    state, ham, omega=omega_native)
                r, j = state[0], state[3]
                h = plan["circular_derivative_step"]
                nu = float(quantities["x1"]*quantities["x2"])
                omega0_fd = (ham.real_hamiltonian(r, torch.zeros_like(r), j+h)
                             -ham.real_hamiltonian(r, torch.zeros_like(r), j-h))/(2*h*nu)
                if not math.isfinite(float(omega0_fd)) or float(omega0_fd) <= 0:
                    raise ValueError("circular finite derivative is invalid")
                def torque(*, omega, omega0):
                    use = dict(quantities, omega=omega, omega_zero_radial=omega0)
                    return -float(flux.energy_flux(state, **use))/(nu*float(omega))
                baseline = torque(omega=omega_native,
                                  omega0=quantities["omega_zero_radial"])
                sampled_omega = torque(omega=omega_sampled,
                                       omega0=quantities["omega_zero_radial"])
                finite_omega0 = torque(omega=omega_native, omega0=omega0_fd)
                combined = torque(omega=omega_sampled, omega0=omega0_fd)
                prior_baseline = next(row for row in point["choices"]
                                      if row["choice"] == baseline_choice)
                if not math.isclose(baseline, prior_baseline["torque"], rel_tol=2e-13):
                    raise ValueError("baseline torque does not replay")
                report["points"].append(dict(
                    target_Momega=target,
                    omega_native=float(omega_native), omega_sampled=float(omega_sampled),
                    omega_zero_radial_native=float(quantities["omega_zero_radial"]),
                    omega_zero_radial_finite=float(omega0_fd),
                    omega_zero_radial_finite_relative=float(
                        omega0_fd/quantities["omega_zero_radial"]-1),
                    sampled_public_torque=point["sampled_public_torque"],
                    baseline_torque=baseline, sampled_omega_torque=sampled_omega,
                    finite_omega0_torque=finite_omega0, combined_torque=combined,
                    baseline_relative_to_public=baseline/point["sampled_public_torque"]-1,
                    sampled_omega_relative_to_baseline=sampled_omega/baseline-1,
                    finite_omega0_relative_to_baseline=finite_omega0/baseline-1,
                    combined_relative_to_baseline=combined/baseline-1))
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(target=target,
                                  sampled_omega_relative=sampled_omega/baseline-1,
                                  finite_omega0_relative=finite_omega0/baseline-1)),
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

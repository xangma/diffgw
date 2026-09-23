"""Ablate the historical finite radial-response probe in native initial data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
from scipy.integrate import solve_ivp
import torch

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_basis_trace import _physics


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/initial_radial_probe_plan.json"
COORDINATES = ("r", "phi", "pr_star", "pphi")


def _finite_probe_state(field, initial, probe):
    ham = field.__self__.hamiltonian
    r, j = (torch.tensor(initial[i], dtype=torch.float64) for i in (0, 3))
    xi = ham.metric_potentials(r)[2]
    pstar = (xi*probe).detach().requires_grad_(True)
    real = ham.real_hamiltonian_from_eff(ham.effective_hamiltonian(r, pstar, j))
    finite_derivative = torch.autograd.grad(real, pstar)[0]
    zero = torch.zeros_like(pstar).requires_grad_(True)
    real_zero = ham.real_hamiltonian_from_eff(ham.effective_hamiltonian(r, zero, j))
    first = torch.autograd.grad(real_zero, zero, create_graph=True)[0]
    linear_response = torch.autograd.grad(first, zero)[0]
    finite_response = finite_derivative/pstar
    if not bool(torch.isfinite(finite_response) & (finite_response > 0)):
        raise ValueError("finite radial response is not positive and finite")
    candidate = initial.copy()
    candidate[2] *= float((linear_response/finite_response).detach())
    return candidate, dict(tortoise_factor=float(xi.detach()),
                           linear_response=float(linear_response.detach()),
                           finite_response=float(finite_response.detach()),
                           pr_star_delta=float(candidate[2]-initial[2]))


def _evolve(field, initial, stop, plan):
    def rhs(_time, state):
        with torch.no_grad():
            return field(torch.from_numpy(state)).detach().numpy()

    result = solve_ivp(rhs, (0., stop), initial, method=plan["ode_method"],
                       rtol=plan["ode_rtol"], atol=plan["ode_atol"],
                       first_step=plan["ode_first_step_M"])
    if not result.success:
        raise RuntimeError(result.message)
    return result.y[:, -1], result.nfev


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_native_evolution"]
    prior = json.loads(prior_path.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.initial-radial-probe-plan.v1"
            or q.sha256(prior_path) != plan["prior_native_evolution_sha256"]
            or prior["status"] != "diagnostic_complete"
            or len(prior["results"]) != 2 or plan["row_indices"] != [0, 1]
            or plan["cartesian_radial_probe"] != 1e-3
            or (plan["ode_method"], plan["ode_rtol"], plan["ode_atol"],
                plan["ode_first_step_M"]) != ("DOP853", 1e-12, 1e-13, 0.125)):
        parser.error("frozen radial-probe plan or input mismatch")
    report = dict(schema="diffgw.seobnrv4.initial-radial-probe.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  prior_sha256=q.sha256(prior_path), plan_sha256=q.sha256(PLAN),
                  script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT/path) for path in (
                      "src/torchwave/approximants/eob/initial_conditions.py",
                      "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
                      "src/torchwave/approximants/eob/seobnrv4_flux.py")},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), results=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for index in plan["row_indices"]:
            reference = prior["results"][index]
            row = dict(reference["row"], omega_start=reference["native_start_Momega"])
            field, initial = _physics(row)
            candidate, response = _finite_probe_state(
                field, initial, plan["cartesian_radial_probe"])
            stop = reference["reference_cutoff_time_M"]
            base, base_calls = _evolve(field, initial, stop, plan)
            altered, altered_calls = _evolve(field, candidate, stop, plan)
            baseline_record = np.array([reference["native_state_at_reference_time"][key]
                                        for key in COORDINATES])
            if np.max(np.abs(base-baseline_record)) > 2e-7:
                raise ValueError("baseline pre-evolution differs from frozen reference")
            compiled = np.array([reference["reference_initial"][key]
                                 for key in COORDINATES])
            compiled_cutoff = baseline_record-np.array([
                reference["native_minus_reference_at_same_time"][key]
                for key in COORDINATES])
            report["results"].append(dict(row=reference["row"],
                                          native_start_Momega=row["omega_start"],
                                          response=response,
                                          initial_native_minus_compiled=(initial-compiled).tolist(),
                                          initial_probe_minus_compiled=(candidate-compiled).tolist(),
                                          cutoff_time_M=stop,
                                          baseline_replay_max_abs_error=float(np.max(np.abs(
                                              base-baseline_record))),
                                          cutoff_probe_minus_baseline=dict(zip(
                                              COORDINATES, (altered-base).tolist())),
                                          cutoff_baseline_minus_compiled=dict(zip(
                                              COORDINATES, (base-compiled_cutoff).tolist())),
                                          cutoff_probe_minus_compiled=dict(zip(
                                              COORDINATES, (altered-compiled_cutoff).tolist())),
                                          evaluations=dict(baseline=base_calls,
                                                           probe=altered_calls)))
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(index=index, initial_delta=response["pr_star_delta"],
                                  cutoff_delta=report["results"][-1][
                                      "cutoff_probe_minus_baseline"])), flush=True)
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

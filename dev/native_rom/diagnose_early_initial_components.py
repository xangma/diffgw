"""Ablate early public-binary initial-state differences by component."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/early_initial_component_plan.json"
COORDINATES = ("r", "phi", "pr_star", "pphi")
SUBSTITUTIONS = ("r", "pr_star", "pphi", "all")


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
    if (plan["schema"] != "diffgw.seobnrv4.early-initial-component-plan.v1"
            or q.sha256(prior_path) != plan["prior_native_evolution_sha256"]
            or prior["status"] != "diagnostic_complete"
            or len(prior["results"]) != 2 or plan["row_indices"] != [0, 1]
            or plan["initial_component_substitutions"] != list(SUBSTITUTIONS)
            or (plan["ode_method"], plan["ode_rtol"], plan["ode_atol"],
                plan["ode_first_step_M"]) != ("DOP853", 1e-12, 1e-13, 0.125)):
        parser.error("frozen component-ablation plan or input mismatch")
    report = dict(schema="diffgw.seobnrv4.early-initial-component.v1",
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
            record = prior["results"][index]
            row = dict(record["row"], omega_start=record["native_start_Momega"])
            field, native = _physics(row)
            reference = np.array([record["reference_initial"][key]
                                  for key in COORDINATES])
            frozen_baseline = np.array([record["native_state_at_reference_time"][key]
                                        for key in COORDINATES])
            compiled_cutoff = frozen_baseline - np.array([
                record["native_minus_reference_at_same_time"][key]
                for key in COORDINATES])
            stop = record["reference_cutoff_time_M"]
            baseline, baseline_calls = _evolve(field, native, stop, plan)
            if np.max(np.abs(baseline-frozen_baseline)) > 2e-7:
                raise ValueError("native baseline differs from frozen pre-evolution")
            trials = {}
            for name in SUBSTITUTIONS:
                substituted = native.copy()
                indices = [COORDINATES.index(name)] if name != "all" else [0, 2, 3]
                substituted[indices] = reference[indices]
                endpoint, calls = _evolve(field, substituted, stop, plan)
                trials[name] = dict(initial_minus_native=dict(zip(
                                        COORDINATES, (substituted-native).tolist())),
                                    cutoff_minus_native=dict(zip(
                                        COORDINATES, (endpoint-baseline).tolist())),
                                    cutoff_minus_compiled=dict(zip(
                                        COORDINATES, (endpoint-compiled_cutoff).tolist())),
                                    evaluations=calls)
            sum_singles = sum((np.array([trials[name]["cutoff_minus_native"][key]
                                         for key in COORDINATES])
                               for name in SUBSTITUTIONS[:-1]), np.zeros(4))
            all_delta = np.array([trials["all"]["cutoff_minus_native"][key]
                                  for key in COORDINATES])
            report["results"].append(dict(
                row=record["row"], cutoff_time_M=stop,
                baseline_replay_max_abs_error=float(np.max(np.abs(baseline-frozen_baseline))),
                baseline_minus_compiled=dict(zip(COORDINATES,
                                                 (baseline-compiled_cutoff).tolist())),
                baseline_evaluations=baseline_calls, trials=trials,
                linear_superposition_remainder=dict(zip(COORDINATES,
                                                         (all_delta-sum_singles).tolist()))))
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(index=index, baseline=report["results"][-1][
                "baseline_minus_compiled"], all=trials["all"]["cutoff_minus_compiled"])),
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

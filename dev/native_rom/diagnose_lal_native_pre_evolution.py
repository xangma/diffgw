"""Development-only native pre-evolution against sampled compiled v4 states."""

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
from .diagnose_boundary_strong_spin_ode import array, array_hash
from .diagnose_lal_dynamics import _reference


EVIDENCE = q.ROOT / "dev/native_rom/evidence/2026-09-23"
PLAN = EVIDENCE / "lal_native_pre_evolution_plan.json"
COORDINATES = ("r", "phi", "pr_star", "pphi")


def _difference(left, right):
    return dict(zip(COORDINATES, (np.asarray(left)-np.asarray(right)).tolist()))


def _run(row, dynamics_row, plan):
    early = dict(row, omega_start=plan["native_start_Momega"])
    field, initial = _physics(early)
    values = _reference(row, plan["reference_step_M"])
    expected_hash = dynamics_row["sampled_references"][str(plan["reference_step_M"])][
        "reference_array_sha256"]
    if array_hash(values) != expected_hash:
        raise ValueError("compiled reference changed since frozen dynamics diagnostic")
    frequency = np.gradient(values[2], values[0], edge_order=2)
    index = int(np.argmin(abs(frequency-plan["cutoff_Momega"])))
    time = float(values[0, index])

    def rhs(_time, state):
        with torch.no_grad():
            return array(field(torch.from_numpy(state)))

    def crossing(_time, state):
        return rhs(_time, state)[1]-plan["cutoff_Momega"]

    crossing.direction = 1
    crossing.terminal = False
    solution = solve_ivp(rhs, (0., time+10.), initial.copy(), method="DOP853",
                         rtol=1e-12, atol=1e-13, first_step=0.125,
                         dense_output=True, events=crossing)
    if not solution.success or len(solution.t_events[0]) != 1:
        raise RuntimeError("native pre-evolution did not cross the requested frequency once")
    same_time = solution.sol(time)
    cross_time = float(solution.t_events[0][0])
    cross_state = solution.y_events[0][0]
    reference_initial = values[1:, 0]
    reference_cutoff = values[1:, index]
    with torch.no_grad():
        native_frequency_at_reference_time = float(field(torch.from_numpy(same_time.copy()))[1])
    return dict(row=row, native_start_Momega=plan["native_start_Momega"],
                reference_start_Momega=float(frequency[0]),
                reference_initial=dict(zip(COORDINATES, reference_initial.tolist())),
                native_initial=dict(zip(COORDINATES, initial.tolist())),
                initial_native_minus_reference=_difference(initial, reference_initial),
                reference_cutoff_index=index, reference_cutoff_time_M=time,
                reference_cutoff_Momega=float(frequency[index]),
                native_frequency_at_reference_time=native_frequency_at_reference_time,
                native_state_at_reference_time=dict(zip(COORDINATES, same_time.tolist())),
                native_minus_reference_at_same_time=_difference(same_time, reference_cutoff),
                native_cutoff_crossing_time_M=cross_time,
                native_cutoff_crossing_state=dict(zip(COORDINATES, cross_state.tolist())),
                native_minus_reference_at_frequency_crossing=_difference(
                    cross_state, reference_cutoff),
                cutoff_time_difference_M=cross_time-time,
                evaluations=solution.nfev)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    path = q.ROOT / plan["dynamics_diagnostic"]
    if (plan["schema"] != "diffgw.seobnrv4.lal-native-pre-evolution-plan.v1"
            or plan["rows"] != [0, 1]
            or plan["reference_step_M"] != 0.125
            or q.sha256(path) != plan["dynamics_sha256"]):
        parser.error("frozen native pre-evolution plan or input mismatch")
    dynamics = json.loads(path.read_text())
    if dynamics["status"] != "diagnostic_complete" or len(dynamics["results"]) != 2:
        parser.error("frozen dynamics diagnostic incomplete")
    report = dict(schema="diffgw.seobnrv4.lal-native-pre-evolution.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  whole_waveform_parity_validated=False,
                  plan_sha256=q.sha256(PLAN), dynamics_sha256=q.sha256(path),
                  script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in (
                      "src/torchwave/approximants/eob/initial_conditions.py",
                      "src/torchwave/approximants/eob/integrator.py",
                      "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
                      "src/torchwave/approximants/eob/seobnrv4_flux.py")},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), results=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for index in plan["rows"]:
            result = _run(dynamics["results"][index]["row"],
                          dynamics["results"][index], plan)
            report["results"].append(result)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(index=index,
                                  initial_difference=result["initial_native_minus_reference"],
                                  cutoff_difference=result["native_minus_reference_at_same_time"])),
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

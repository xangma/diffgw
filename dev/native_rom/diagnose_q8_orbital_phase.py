"""Compare native and public q8 orbital phase spans at equal frequencies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import torch

from . import qualify_generator as q
from .diagnose_lal_dynamics import _reference
from .diagnose_q8_initial_alignment import _physics


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q8_orbital_phase_plan.json"


def _public_crossings(values, targets):
    frequency = np.gradient(values[2], values[0], edge_order=2)
    reached = np.flatnonzero(frequency >= targets[-1])
    if not len(reached):
        raise ValueError("public orbital frequency never reaches the final target")
    end = int(reached[0]) + 4
    if end >= len(frequency) - 2:
        raise ValueError("public frequency crossing lacks interpolation support")
    frequency = frequency[2:end]
    if not np.all(np.diff(frequency) > 0):
        raise ValueError("public orbital frequency is not monotone")
    if not (frequency[0] < targets[0] < targets[-1] < frequency[-1]):
        raise ValueError("public frequencies do not cover the frozen targets")
    return [dict(time_M=float(PchipInterpolator(frequency, values[0, 2:end])(w)),
                 phi_rad=float(PchipInterpolator(frequency, values[2, 2:end])(w)))
            for w in targets]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign_path = q.ROOT / plan["campaign"]
    alignment_path = q.ROOT / plan["initial_alignment"]
    targets = plan["frequencies_Momega"]
    solver = plan["solver"]
    if (plan["schema"] != "diffgw.seobnrv4.q8-orbital-phase-plan.v1"
            or targets != [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13]
            or plan["reference_output_steps_M"] != [0.125, 0.0625]
            or solver != dict(method="DOP853", rtol=1e-12, atol=1e-13,
                              first_step_M=0.125, max_step_M=10.0)
            or plan["release_qualified"] is not False
            or q.sha256(campaign_path) != plan["campaign_sha256"]
            or q.sha256(alignment_path) != plan["initial_alignment_sha256"]):
        parser.error("frozen plan or inputs differ")
    campaign = json.loads(campaign_path.read_text())
    alignment = json.loads(alignment_path.read_text())
    row = campaign["rows"]["q8_high_aligned"]["row"]
    choice = campaign["choice"]
    pin = campaign["lal_binary"]
    if (alignment["status"] != "diagnostic_complete" or alignment["row"] != row
            or alignment["choice"] != choice
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("candidate or public binary differs")
    torch.set_num_threads(1)
    _, _, field = _physics(row, choice)
    initial = np.array([alignment["native_warm_start"]["target_state"][name]
                        for name in ("r", "phi", "pr_star", "pphi")])

    def rhs(_time, state):
        with torch.no_grad():
            return field(torch.from_numpy(state)).detach().numpy()

    def event(_time, state):
        return rhs(0, state)[1] - targets[-1]

    event.terminal = True
    event.direction = 1
    sources = (
        "dev/native_rom/eob/torch/source_calibration_hamiltonian.py",
        "src/torchwave/approximants/eob/seobnrv4_flux.py",
        "src/torchwave/approximants/eob/integrator.py",
        "dev/native_rom/diagnose_lal_dynamics.py",
    )
    report = dict(
        schema="diffgw.seobnrv4.q8-orbital-phase.v1", scope=q.SCOPE,
        status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), campaign_sha256=q.sha256(campaign_path),
        initial_alignment_sha256=q.sha256(alignment_path),
        source_sha256={path: q.sha256(q.ROOT / path) for path in sources},
        lal_binary=pin, row=row, choice=choice, solver=solver, native={},
        public_grids={}, comparisons={}, errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                     pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        sol = solve_ivp(rhs, (0., 3000.), initial, method=solver["method"],
                        rtol=solver["rtol"], atol=solver["atol"],
                        first_step=solver["first_step_M"],
                        max_step=solver["max_step_M"], dense_output=True, events=event)
        if not sol.success or len(sol.t_events[0]) != 1:
            raise RuntimeError(f"native q8 crossing not found: {sol.message}")
        native = []
        t_left = 0.
        for target in targets:
            if target == targets[0]:
                crossing_t = 0.
            elif target == targets[-1]:
                crossing_t = float(sol.t_events[0][0])
            else:
                crossing_t = brentq(
                    lambda t: rhs(0, sol.sol(t))[1] - target,
                    t_left, float(sol.t_events[0][0]), xtol=1e-9)
            state = sol.sol(crossing_t)
            native.append(dict(time_M=float(crossing_t), phi_rad=float(state[1]),
                               radius=float(state[0]), pr_star=float(state[2]),
                               pphi=float(state[3])))
            t_left = crossing_t
        report["native"] = dict(crossings=native, accepted_steps=len(sol.t)-1,
                                evaluations=sol.nfev,
                                elapsed_seconds=time.monotonic()-started)
        q.write_evidence(args.json_out, report)
        print("native_trajectory_complete", flush=True)
        for step in plan["reference_output_steps_M"]:
            started = time.monotonic()
            values = _reference(row, step)
            expected = alignment["grids"][str(step)]["reference_array_sha256"]
            from .diagnose_boundary_strong_spin_ode import array_hash
            if array_hash(values) != expected:
                raise ValueError("public grid differs from frozen alignment")
            public = _public_crossings(values, targets)
            report["public_grids"][str(step)] = dict(
                reference_array_sha256=expected, crossings=public,
                elapsed_seconds=time.monotonic()-started)
            report["comparisons"][str(step)] = [dict(
                frequency_Momega=target,
                native_minus_public_phase_span_rad=float(
                    (native[i]["phi_rad"]-native[0]["phi_rad"])
                    -(public[i]["phi_rad"]-public[0]["phi_rad"])),
                native_minus_public_time_span_M=float(
                    (native[i]["time_M"]-native[0]["time_M"])
                    -(public[i]["time_M"]-public[0]["time_M"])))
                for i, target in enumerate(targets)]
            q.write_evidence(args.json_out, report)
            print(f"public_step_M={step}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

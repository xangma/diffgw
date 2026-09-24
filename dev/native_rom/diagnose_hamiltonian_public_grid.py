"""Vary the public sampling grid at fixed states for the late Hamiltonian check."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_ode import array_hash
from .diagnose_lal_dynamics import _five_point, _reference


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/hamiltonian_public_grid_plan.json"
COORDINATES = ("r", "phi", "pr_star", "pphi")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    late_path = q.ROOT / plan["prior_late_field"]
    ham_path = q.ROOT / plan["prior_hamiltonian_ablation"]
    if (plan["schema"] != "diffgw.seobnrv4.hamiltonian-public-grid-plan.v1"
            or q.sha256(late_path) != plan["prior_late_field_sha256"]
            or q.sha256(ham_path) != plan["prior_hamiltonian_ablation_sha256"]
            or plan["target_frequencies_Momega"] != [0.1, 0.12, 0.15]
            or plan["public_output_steps_M"] != [0.125, 0.0625, 0.03125, 0.015625]):
        parser.error("frozen plan or prior evidence differs")
    late = json.loads(late_path.read_text())
    ham = json.loads(ham_path.read_text())
    if late["status"] != "diagnostic_complete" or ham["status"] != "diagnostic_complete":
        parser.error("prior diagnostic did not complete")
    points = {}
    for public, native in zip(late["grids"][0]["points"], ham["grids"][0]["points"]):
        if public["target_Momega"] != native["target_Momega"]:
            parser.error("prior point order differs")
        if public["target_Momega"] in plan["target_frequencies_Momega"]:
            points[public["target_Momega"]] = (public, native)
    report = dict(
        schema="diffgw.seobnrv4.hamiltonian-public-grid.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        direct_historical_derivative_established=False,
        plan_sha256=q.sha256(PLAN), late_sha256=q.sha256(late_path),
        hamiltonian_sha256=q.sha256(ham_path), script_sha256=q.sha256(__file__),
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv), row=late["row"], grids=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for step in plan["public_output_steps_M"]:
            values = _reference(late["row"], step)
            grid = dict(step_M=step, shape=list(values.shape),
                        public_array_sha256=array_hash(values), points=[])
            for target in plan["target_frequencies_Momega"]:
                public, native = points[target]
                expected_time = public["time_M"]
                index = int(round(expected_time/step))
                if (index < 2 or index+2 >= values.shape[1]
                        or abs(values[0, index]-expected_time) > 1e-7):
                    raise ValueError("fixed time outside public derivative support")
                state = values[1:, index]
                stored_state = np.array([public["state"][key] for key in COORDINATES])
                sampled = _five_point(values[1:], index, step)
                historical = native["native_historical_derivative"]
                pr_star = public["state"]["pr_star"]
                pphi = public["state"]["pphi"]
                radial_with_public_torque = (historical["pr_star"]
                                             + pr_star/pphi*(sampled[3]-historical["pphi"]))
                grid["points"].append(dict(
                    target_Momega=target, time_M=expected_time, index=index,
                    max_state_difference=float(np.max(abs(state-stored_state))),
                    sampled_derivative=dict(zip(COORDINATES, sampled.tolist())),
                    historical_phi_relative_to_public=historical["phi"]/sampled[1]-1,
                    historical_pr_relative_to_public=(historical["pr_star"]-sampled[2])/abs(sampled[2]),
                    public_torque_adjusted_pr_relative_to_public=(
                        radial_with_public_torque-sampled[2])/abs(sampled[2])))
            report["grids"].append(grid)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(step_M=step, relative=[{
                "target": point["target_Momega"],
                "phi": point["historical_phi_relative_to_public"],
                "pr_torque_adjusted": point["public_torque_adjusted_pr_relative_to_public"]}
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

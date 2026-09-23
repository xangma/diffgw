"""Frozen later-inspiral same-state v4 torque sensitivity to (3,3) rho.

Public torque is a five-point derivative of sampled p_phi, not a direct mode
flux. The two public grids are separate numerical diagnostics, not an error
estimate for historical flux parity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_basis_trace import _physics
from .diagnose_boundary_strong_spin_ode import array_hash
from .diagnose_lal_dynamics import _five_point, _reference
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/mode_33_late_flux_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / plan[name] for name in (
        "prior_mode_33", "prior_vector_field", "prior_vector_field_plan")}
    if (plan["schema"] != "diffgw.seobnrv4.mode-33-late-flux-plan.v1"
            or any(q.sha256(path) != plan[name + "_sha256"]
                   for name, path in paths.items())
            or plan["public_output_steps_M"] != [0.125, 0.0625]
            or plan["frequency_targets_Momega"] != [
                0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15]):
        parser.error("frozen plan or prior evidence differs")
    prior = json.loads(paths["prior_mode_33"].read_text())
    field_prior = json.loads(paths["prior_vector_field"].read_text())
    field_plan = json.loads(paths["prior_vector_field_plan"].read_text())
    binary_path = q.ROOT / field_plan["pinned_public_waveform"]
    binary = json.loads(binary_path.read_text())
    pin = binary["lal_binary"]
    if (prior["status"] != "diagnostic_complete"
            or field_prior["status"] != "diagnostic_complete"
            or prior["row"] != field_prior["row"]
            or field_prior["binary_sha256"] != q.sha256(binary_path)
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("prior row or compiled binary identity differs")
    base = dict(plan["native_choices"])
    choices = base.pop("mode_33_higher_spin")
    if choices != ["taylor_f", "lal_v4_rho"]:
        parser.error("native choices differ from frozen plan")
    row = prior["row"]
    field, _ = _physics(row)
    ham = field.__self__.hamiltonian
    models = {choice: SEOBNRv4PublishedFlux(**base, mode_33_higher_spin=choice)
              for choice in choices}
    report = dict(
        schema="diffgw.seobnrv4.mode-33-late-flux.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        historical_mode_flux_parity_established=False,
        plan_sha256=q.sha256(PLAN),
        prior_sha256={name: q.sha256(path) for name, path in paths.items()},
        binary_sha256=q.sha256(binary_path),
        binary_manifest_sha256=q.canonical_hash(pin["binaries"]),
        script_sha256=q.sha256(__file__),
        source_sha256={path: q.sha256(q.ROOT / path) for path in (
            "src/torchwave/approximants/eob/seobnrv4_flux.py",
            "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
            "dev/native_rom/diagnose_lal_dynamics.py")},
        process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                     pid=os.getpid(), argv=sys.argv),
        row=row, choice=base, grids=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for step in plan["public_output_steps_M"]:
            values = _reference(row, step)
            frequencies = np.gradient(values[2], values[0], edge_order=2)
            indices = [int(np.argmin(abs(frequencies-target)))
                       for target in plan["frequency_targets_Momega"]]
            if not (4 <= indices[0] < indices[-1] <= values.shape[1]-5
                    and all(a < b for a, b in zip(indices, indices[1:]))):
                raise ValueError("frequency targets outside derivative support")
            grid = dict(step_M=step, shape=list(values.shape),
                        array_sha256=array_hash(values), points=[])
            for target, index in zip(plan["frequency_targets_Momega"], indices):
                state = torch.from_numpy(values[1:, index].copy())
                sampled_derivative = _five_point(values[1:], index, step)
                public = float(sampled_derivative[3])
                if not public < 0:
                    raise ValueError("public sampled torque is not negative")
                with torch.no_grad():
                    native_derivative = field(state).numpy()
                    omega = torch.as_tensor(native_derivative[1])
                    quantities = models["taylor_f"].from_hamiltonian(
                        state, ham, omega=omega)
                    nu = quantities["x1"]*quantities["x2"]
                    torques = {
                        name: float(-model.energy_flux(state, **quantities)/(nu*omega))
                        for name, model in models.items()}
                grid["points"].append(dict(
                    target_Momega=target, index=index, time_M=float(values[0, index]),
                    sampled_phase_derivative_Momega=float(frequencies[index]),
                    native_state_Momega=float(omega),
                    state=dict(zip(("r", "phi", "pr_star", "pphi"),
                                   values[1:, index].tolist())),
                    sampled_derivative=dict(zip(("r", "phi", "pr_star", "pphi"),
                                                sampled_derivative.tolist())),
                    native_derivative=dict(zip(("r", "phi", "pr_star", "pphi"),
                                               native_derivative.tolist())),
                    native_derivative_relative_to_sampled=dict(zip(
                        ("r", "phi", "pr_star", "pphi"),
                        ((native_derivative-sampled_derivative)
                         / np.maximum(abs(sampled_derivative), 1e-15)).tolist())),
                    sampled_public_torque=public, native_torque=torques,
                    native_torque_relative_to_public={
                        name: torque/public-1 for name, torque in torques.items()}))
            report["grids"].append(grid)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(step_M=step, relative=[
                point["native_torque_relative_to_public"] for point in grid["points"]])),
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

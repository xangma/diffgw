"""Frozen output-grid sensitivity of the sampled public strong-spin torque."""

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
from .diagnose_boundary_strong_spin_ode import array_hash
from .diagnose_lal_dynamics import _five_point, _reference
from .diagnose_strong_spin_inspiral_field import _physics
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/public_torque_sampling_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior"]
    if (plan["schema"] != "diffgw.seobnrv4.public-torque-sampling-plan.v1"
            or q.sha256(prior_path) != plan["prior_sha256"]
            or plan["steps_M"] != [0.125, 0.0625, 0.03125]
            or plan["targets_Momega"] != [0.045, 0.06, 0.09, 0.12]):
        parser.error("frozen plan or prior differs")
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "diagnostic_complete":
        parser.error("prior diagnostic is incomplete")
    if q.lal_binary_identity(prior["lal_binary"]["lal_version"],
                             prior["lal_binary"]["lalsimulation_version"]) != prior["lal_binary"]:
        parser.error("public binary identity changed")
    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.public-torque-sampling.v1", scope=q.SCOPE,
        status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
        prior_sha256=q.sha256(prior_path), script_sha256=q.sha256(__file__),
        source_sha256={"src/torchwave/approximants/eob/seobnrv4_flux.py":
                       q.sha256(q.ROOT / "src/torchwave/approximants/eob/seobnrv4_flux.py")},
        lal_binary=prior["lal_binary"], row=prior["row"],
        native_choice=plan["native_choice"], grids={}, points=[], errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        values = {}
        for step in plan["steps_M"]:
            grid = _reference(prior["row"], step)
            digest = array_hash(grid)
            if step in (0.125, 0.0625):
                expected = prior["grids"][str(step)]["reference_array_sha256"]
                if digest != expected:
                    raise ValueError(f"public reference at step {step} failed exact replay")
            values[step] = grid
            report["grids"][str(step)] = dict(shape=list(grid.shape), array_sha256=digest)
            q.write_evidence(args.json_out, report)
        ham, _, field = _physics(prior["row"])
        flux = SEOBNRv4PublishedFlux(**plan["native_choice"])
        coarse = values[0.125]
        frequency = np.gradient(coarse[2], coarse[0], edge_order=2)
        for target in plan["targets_Momega"]:
            index = int(np.argmin(abs(frequency-target)))
            if not 4 <= index < coarse.shape[1]-4:
                raise ValueError("frequency target outside safe derivative support")
            time_m = float(coarse[0, index])
            samples = {}
            for step in plan["steps_M"]:
                grid = values[step]
                at = int(np.argmin(abs(grid[0]-time_m)))
                if abs(float(grid[0, at])-time_m) > 1e-8:
                    raise ValueError("public grids have no shared sample")
                samples[str(step)] = dict(
                    time_M=float(grid[0, at]),
                    sampled_torque=float(_five_point(grid[4], at, step)),
                    max_state_delta_from_coarse=float(np.max(abs(
                        grid[1:, at]-coarse[1:, index]))))
            state = torch.from_numpy(values[0.03125][1:, int(np.argmin(
                abs(values[0.03125][0]-time_m)))].copy())
            with torch.no_grad():
                omega = field(state)[1]
                quantities = flux.from_hamiltonian(state, ham, omega=omega)
                nu = float(quantities["x1"]*quantities["x2"])
                native = -float(flux.energy_flux(state, **quantities))/(nu*float(omega))
            sampled = samples["0.03125"]["sampled_torque"]
            report["points"].append(dict(
                target_Momega=target, time_M=time_m, native_Momega=float(omega),
                native_torque=native, samples=samples,
                native_relative_to_finest=native/sampled-1,
                coarse_relative_to_finest=samples["0.125"]["sampled_torque"]/sampled-1,
                middle_relative_to_finest=samples["0.0625"]["sampled_torque"]/sampled-1))
            q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

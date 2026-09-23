"""Trace native versus pinned-public dynamics from the historical early start."""

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
from .diagnose_lal_dynamics import _point, _reference


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/early_aligned_vector_field_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_native_evolution"]
    binary_path = q.ROOT / plan["pinned_public_waveform"]
    prior = json.loads(prior_path.read_text())
    binary = json.loads(binary_path.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.early-aligned-vector-field-plan.v1"
            or q.sha256(prior_path) != plan["prior_native_evolution_sha256"]
            or q.sha256(binary_path) != plan["pinned_public_waveform_sha256"]
            or prior["status"] != "diagnostic_complete"
            or binary["status"] != "diagnostic_complete"
            or plan["row_index"] != 1
            or plan["early_start_Momega"] != prior["results"][1]["native_start_Momega"]
            or plan["frequency_targets_Momega"] != [0.032, 0.035, 0.04, 0.05]
            or plan["public_output_steps_M"] != [0.125, 0.0625]
            or q.lal_binary_identity(binary["lal_binary"]["lal_version"],
                                     binary["lal_binary"]["lalsimulation_version"])
            != binary["lal_binary"]):
        parser.error("frozen early vector-field plan or reference differs")
    row = dict(prior["results"][1]["row"], omega_start=plan["early_start_Momega"])
    field, _ = _physics(row)
    report = dict(schema="diffgw.seobnrv4.early-aligned-vector-field.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN),
                  prior_sha256=q.sha256(prior_path), binary_sha256=q.sha256(binary_path),
                  script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT/path) for path in (
                      "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
                      "src/torchwave/approximants/eob/seobnrv4_flux.py",
                      "dev/native_rom/diagnose_lal_dynamics.py")},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), row=row,
                  grids=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for step in plan["public_output_steps_M"]:
            values = _reference(row, step)
            frequencies = np.gradient(values[2], values[0], edge_order=2)
            indices = [int(np.argmin(abs(frequencies-target)))
                       for target in plan["frequency_targets_Momega"]]
            if not (4 <= indices[0] < indices[1] < indices[2] < indices[3]
                    <= values.shape[1]-5):
                raise ValueError("early frequency targets outside derivative support")
            points = [_point(field, values, index, step) for index in indices]
            report["grids"].append(dict(step_M=step, shape=list(values.shape),
                                        array_sha256=array_hash(values), points=points))
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(step_M=step, pphi_relative=[
                point["relative_difference"]["pphi"] for point in points],
                pr_star_relative=[point["relative_difference"]["pr_star"]
                                  for point in points])), flush=True)
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

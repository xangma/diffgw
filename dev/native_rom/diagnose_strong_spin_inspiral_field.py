"""Frozen same-state strong-spin inspiral field and initial-state diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_ode import array, array_hash
from .diagnose_lal_dynamics import _continuation, _point, _reference
from .eob.torch.historical_hamiltonian import HistoricalLogDerivative
from .eob.torch.warm_start import native_frequency_warm_start
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/strong_spin_inspiral_field_plan.json"
COORDINATES = ("r", "phi", "pr_star", "pphi")


def _physics(row):
    m1, m2, s1, s2 = (torch.tensor(row[key], dtype=torch.float64)
                      for key in ("mass1", "mass2", "chi1", "chi2"))
    x1, x2 = m1/(m1+m2), m2/(m1+m2)
    ham = HistoricalLogDerivative(x1*x2, s1, s2, delta=x1-x2)
    flux = SEOBNRv4PublishedFlux(
        spin_truncation=q.CHOICES["spin_truncation"],
        retain_pan_cubic_spin=True, mode_33_higher_spin="lal_v4_rho")
    return ham, flux, EOBTrajectoryIntegrator(ham, flux).vector_field


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign_path, waveform_path = (q.ROOT / plan[key] for key in ("campaign", "waveform"))
    if (plan["schema"] != "diffgw.seobnrv4.strong-spin-inspiral-field-plan.v1"
            or q.sha256(campaign_path) != plan["campaign_sha256"]
            or q.sha256(waveform_path) != plan["waveform_sha256"]
            or plan["row_name"] != "strong_aligned_spin"
            or plan["reference_output_steps_M"] != [0.125, 0.0625]
            or plan["frequency_targets_Momega"] != [0.045, 0.06, 0.09, 0.12]):
        parser.error("frozen plan or prior evidence differs")
    campaign, waveform = (json.loads(path.read_text()) for path in
                          (campaign_path, waveform_path))
    if (campaign["rows"][plan["row_name"]]["row"] != waveform["row"]
            or waveform["status"] != "diagnostic_complete"
            or campaign["candidate_choice"]["hamiltonian_log_derivative"] != "lal_v4"
            or campaign["candidate_choice"]["mode_33_higher_spin"] != "lal_v4_rho"):
        parser.error("frozen row, candidate, or prior result differs")
    pin = campaign["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity changed")
    torch.set_num_threads(1)
    row = waveform["row"]
    ham, flux, field = _physics(row)
    sources = ("dev/native_rom/eob/torch/historical_hamiltonian.py",
               "dev/native_rom/eob/torch/warm_start.py",
               "src/torchwave/approximants/eob/seobnrv4_flux.py",
               "src/torchwave/approximants/eob/integrator.py",
               "src/torchwave/approximants/eob/initial_conditions.py")
    report = dict(
        schema="diffgw.seobnrv4.strong-spin-inspiral-field.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), campaign_sha256=q.sha256(campaign_path),
        waveform_sha256=q.sha256(waveform_path), script_sha256=q.sha256(__file__),
        source_sha256={name: q.sha256(q.ROOT / name) for name in sources},
        lal_binary=pin, row=row, grids={}, errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        with torch.no_grad():
            warm = native_frequency_warm_start(
                ham, flux, torch.tensor(0.03162277660168379, dtype=torch.float64),
                torch.tensor(row["omega_start"], dtype=torch.float64),
                atol=1e-14, rtol=1e-13, first_step_M=0.125)
        report["native_warm_start"] = dict(
            state=dict(zip(COORDINATES, array(warm.state).tolist())),
            crossing_time_M=float(warm.crossing_time),
            frequency_residual=float(warm.frequency_residual),
            accepted_steps=warm.accepted_steps, evaluations=warm.evaluations)
        q.write_evidence(args.json_out, report)
        for step in plan["reference_output_steps_M"]:
            print(f"public_step_M={step}", flush=True)
            started = time.monotonic()
            values = _reference(row, step)
            frequency = np.gradient(values[2], values[0], edge_order=2)
            selected = [int(np.argmin(abs(frequency-target)))
                        for target in plan["frequency_targets_Momega"]]
            if not all(4 <= index < values.shape[1]-4 for index in selected):
                raise ValueError("a frequency target lies outside safe derivative support")
            if selected != sorted(set(selected)):
                raise ValueError("frequency targets are not distinct and ordered")
            points = [_point(field, values, index, step) for index in selected]
            public_initial = values[1:, 0].copy()
            public_initial[1] = 0.0
            initial_difference = array(warm.state)-public_initial
            grid = dict(
                step_M=step, shape=list(values.shape),
                reference_array_sha256=array_hash(values),
                public_initial=dict(zip(COORDINATES, public_initial.tolist())),
                native_minus_public_initial=dict(zip(COORDINATES, initial_difference.tolist())),
                targets=points,
                continuation=_continuation(field, values, selected[0], selected[1]),
                elapsed_seconds=time.monotonic()-started)
            report["grids"][str(step)] = grid
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(step_M=step, elapsed_seconds=grid["elapsed_seconds"],
                                  relative=[point["relative_difference"] for point in points],
                                  continuation=grid["continuation"][
                                      "endpoint_native_minus_reference"])), flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

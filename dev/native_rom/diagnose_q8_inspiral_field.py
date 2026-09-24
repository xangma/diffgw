"""Compare the frozen q8 candidate field with sampled public v4 dynamics."""

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
from .diagnose_boundary_strong_spin_ode import array_hash
from .diagnose_lal_dynamics import _point, _reference
from .eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian,
)
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q8_inspiral_field_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign_path = q.ROOT / plan["campaign"]
    if (plan["schema"] != "diffgw.seobnrv4.q8-inspiral-field-plan.v1"
            or plan["row_name"] != "q8_high_aligned"
            or plan["reference_output_steps_M"] != [0.125, 0.0625]
            or plan["frequency_targets_Momega"] != [0.045, 0.06, 0.09, 0.12]
            or plan["release_qualified"] is not False
            or q.sha256(campaign_path) != plan["campaign_sha256"]):
        parser.error("frozen plan or campaign differs")
    campaign = json.loads(campaign_path.read_text())
    row = campaign["rows"][plan["row_name"]]["row"]
    choice = campaign["choice"]
    if (campaign["status"] != "diagnostic_complete"
            or campaign["rows"][plan["row_name"]]["check"]["numerical_check_passed"]
            or choice["calibration"] != "lal_v4_rounded"
            or choice["physical"]["hamiltonian_log_derivative"] != "lal_v4"
            or choice["mode_43_higher_spin"] != "lal_v4_rho"
            or choice["spin_truncation"] != "tabulated"):
        parser.error("q8 failure or candidate choice differs")
    pin = campaign["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity differs")
    torch.set_num_threads(1)
    def scalar(value):
        return torch.tensor(value, dtype=torch.float64)
    x1 = row["mass1"]/(row["mass1"]+row["mass2"])
    x2 = 1.-x1
    ham = SourceCalibratedHistoricalHamiltonian(
        scalar(x1*x2), scalar(row["chi1"]), scalar(row["chi2"]),
        delta=scalar(x1-x2))
    flux = SEOBNRv4PublishedFlux(
        spin_truncation="tabulated", retain_pan_cubic_spin=True,
        mode_21_higher_spin=choice["flux"]["mode_21_higher_spin"],
        mode_31_higher_spin=choice["flux"]["mode_31_higher_spin"],
        mode_32_spin_square=choice["flux"]["mode_32_spin_square"],
        mode_33_higher_spin=choice["physical"]["mode_33_higher_spin"],
        mode_43_higher_spin="lal_v4_rho")
    field = EOBTrajectoryIntegrator(ham, flux).vector_field
    sources = (
        "dev/native_rom/eob/torch/source_calibration_hamiltonian.py",
        "dev/native_rom/diagnose_lal_dynamics.py",
        "src/torchwave/approximants/eob/seobnrv4_flux.py",
        "src/torchwave/approximants/eob/integrator.py",
    )
    report = dict(
        schema="diffgw.seobnrv4.q8-inspiral-field.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        campaign_sha256=q.sha256(campaign_path),
        source_sha256={path: q.sha256(q.ROOT / path) for path in sources},
        lal_binary=pin, row=row, choice=choice, grids={}, errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                     pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        for step in plan["reference_output_steps_M"]:
            started = time.monotonic()
            values = _reference(row, step)
            frequency = np.gradient(values[2], values[0], edge_order=2)
            selected = [int(np.argmin(abs(frequency-target)))
                        for target in plan["frequency_targets_Momega"]]
            if (selected != sorted(set(selected))
                    or not all(4 <= index < values.shape[1]-4 for index in selected)):
                raise ValueError("frequency probes outside ordered derivative support")
            points = [_point(field, values, index, step) for index in selected]
            report["grids"][str(step)] = dict(
                step_M=step, shape=list(values.shape),
                reference_array_sha256=array_hash(values),
                frequency_targets=points,
                elapsed_seconds=time.monotonic()-started)
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

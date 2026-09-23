"""Compare sampled pinned SEOBNRv4 dynamics with the native vector field.

The LAL binary is a diagnostic oracle. Its sampled derivatives are numerical
estimates, and this script does not establish historical waveform parity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import lal
import lalsimulation as lalsim
import numpy as np
from scipy.integrate import solve_ivp
import torch

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_basis_trace import _physics
from .diagnose_boundary_strong_spin_ode import array, array_hash


EVIDENCE = q.ROOT / "dev/native_rom/evidence/2026-09-23"
PLAN = EVIDENCE / "lal_dynamics_plan.json"
COORDINATES = ("r", "phi", "pr_star", "pphi")


def _five_point(values: np.ndarray, index: int, step: float) -> np.ndarray:
    return (-values[..., index+2] + 8*values[..., index+1]
            - 8*values[..., index-1] + values[..., index-2]) / (12*step)


def _reference(row: dict, step_m: float) -> np.ndarray:
    mass_s = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
    result = lalsim.SimIMRSpinAlignedEOBModes(
        step_m*mass_s, row["mass1"]*lal.MSUN_SI,
        row["mass2"]*lal.MSUN_SI, row["omega_start"]/(math.pi*mass_s),
        100e6*lal.PC_SI, row["chi1"], row["chi2"], 4,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
        lal.CreateREAL8Vector(10), 0, lal.CreateDict(), lal.CreateDict())
    values = np.asarray(result[1].data).reshape(5, -1).copy()
    if (values.shape[1] < 10 or not np.isfinite(values).all()
            or not np.all(np.diff(values[0]) > 0)
            or not np.allclose(np.diff(values[0]), step_m, rtol=0, atol=2e-8)):
        raise ValueError("compiled low-sampling dynamics has unexpected layout or grid")
    return values


def _point(field, values: np.ndarray, index: int, step_m: float) -> dict:
    state = values[1:, index]
    sampled = _five_point(values[1:], index, step_m)
    with torch.no_grad():
        native = array(field(torch.from_numpy(state.copy())))
    error = native-sampled
    return dict(index=index, time_M=float(values[0, index]),
                frequency_Momega=float(sampled[1]),
                state=dict(zip(COORDINATES, state.tolist())),
                sampled_derivative=dict(zip(COORDINATES, sampled.tolist())),
                native_derivative=dict(zip(COORDINATES, native.tolist())),
                native_minus_sampled=dict(zip(COORDINATES, error.tolist())),
                relative_difference=dict(zip(
                    COORDINATES, (error/np.maximum(abs(sampled), 1e-15)).tolist())))


def _continuation(field, values: np.ndarray, start: int, end: int) -> dict:
    times = values[0, start:end+1]
    reference = values[1:, start:end+1].T

    def rhs(_time, state):
        with torch.no_grad():
            return array(field(torch.from_numpy(state)))

    solution = solve_ivp(rhs, (float(times[0]), float(times[-1])),
                         reference[0].copy(), method="DOP853", rtol=1e-12,
                         atol=1e-13, first_step=0.125, t_eval=times)
    if not solution.success or solution.y.shape != reference.T.shape:
        raise RuntimeError(f"native same-state continuation failed: {solution.message}")
    difference = solution.y.T-reference
    return dict(duration_M=float(times[-1]-times[0]), samples=len(times),
                evaluations=solution.nfev,
                endpoint_native_minus_reference=dict(zip(COORDINATES, difference[-1].tolist())),
                max_abs_native_minus_reference=dict(zip(
                    COORDINATES, np.max(abs(difference), axis=0).tolist())))


def _run_row(row: dict, plan: dict) -> dict:
    field, initial = _physics(row)
    output = dict(row=row, native_quasicircular_initial=dict(zip(COORDINATES, initial.tolist())),
                  sampled_references={})
    for step in plan["reference_output_steps_M"]:
        values = _reference(row, step)
        frequency = np.gradient(values[2], values[0], edge_order=2)
        selected = [int(np.argmin(abs(frequency-target)))
                    for target in plan["frequency_targets_Momega"]]
        if not (4 <= selected[0] < selected[1] < values.shape[1]-4):
            raise ValueError("frequency probes outside safe low-sampling support")
        points = [_point(field, values, index, step) for index in selected]
        initial_difference = initial-values[1:, selected[0]]
        result = dict(reference_shape=list(values.shape),
                      reference_array_sha256=array_hash(values),
                      frequency_targets=points,
                      initial_minus_reference_at_first_target=dict(zip(
                          ("r", "pr_star", "pphi"),
                          initial_difference[[0, 2, 3]].tolist())))
        if step == 0.125:
            result["same_state_continuation"] = _continuation(
                field, values, selected[0], selected[1])
        output["sampled_references"][str(step)] = result
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_whole_waveform"]
    if q.sha256(prior_path) != plan["prior_sha256"]:
        parser.error("prior evidence differs from frozen plan")
    prior = json.loads(prior_path.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.lal-dynamics-plan.v1"
            or plan["rows"] != [0, 1]
            or plan["reference_output_steps_M"] != [0.125, 0.0625]
            or plan["frequency_targets_Momega"] != [0.04, 0.05]
            or prior["status"] != "diagnostic_complete"
            or prior["rows"] != [prior["results"][i]["row"] for i in plan["rows"]]):
        parser.error("frozen plan or prior evidence incompatible")
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("compiled LALSuite binary differs from frozen evidence")
    sources = ["dev/native_rom/qualify_generator.py",
               "dev/native_rom/diagnose_boundary_strong_spin_basis_trace.py",
               "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
               "src/torchwave/approximants/eob/seobnrv4_flux.py",
               "src/torchwave/approximants/eob/integrator.py",
               "src/torchwave/approximants/eob/initial_conditions.py"]
    report = dict(schema="diffgw.seobnrv4.lal-dynamics-diagnostic.v1",
                  scope=dict(q.SCOPE), status="running", release_qualified=False,
                  historical_parity_established=False,
                  plan_sha256=q.sha256(PLAN), prior_sha256=q.sha256(prior_path),
                  script_sha256=q.sha256(__file__),
                  source_sha256={path: q.sha256(q.ROOT / path) for path in sources},
                  binary_manifest_sha256=q.canonical_hash(pin["binaries"]),
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), results=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for index in plan["rows"]:
            started = time.monotonic()
            row = _run_row(prior["rows"][index], plan)
            row["elapsed_seconds"] = time.monotonic()-started
            report["results"].append(row)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(index=index, elapsed_seconds=row["elapsed_seconds"])), flush=True)
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

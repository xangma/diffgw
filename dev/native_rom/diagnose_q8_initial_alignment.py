"""Compare native q8 warm-start and public v4 states at equal orbital frequency."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
import torch

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_ode import array, array_hash
from .diagnose_lal_dynamics import COORDINATES, _five_point, _reference
from .eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian,
)
from .eob.torch.warm_start import native_frequency_warm_start
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/q8_initial_alignment_plan.json"


def _physics(row, choice):
    def scalar(value):
        return torch.tensor(value, dtype=torch.float64)

    x1 = row["mass1"] / (row["mass1"] + row["mass2"])
    x2 = 1 - x1
    ham = SourceCalibratedHistoricalHamiltonian(
        scalar(x1 * x2), scalar(row["chi1"]), scalar(row["chi2"]),
        delta=scalar(x1 - x2))
    flux = SEOBNRv4PublishedFlux(
        spin_truncation="tabulated", retain_pan_cubic_spin=True,
        mode_21_higher_spin=choice["flux"]["mode_21_higher_spin"],
        mode_31_higher_spin=choice["flux"]["mode_31_higher_spin"],
        mode_32_spin_square=choice["flux"]["mode_32_spin_square"],
        mode_33_higher_spin=choice["physical"]["mode_33_higher_spin"],
        mode_43_higher_spin=choice["mode_43_higher_spin"])
    return ham, flux, EOBTrajectoryIntegrator(ham, flux).vector_field


def _crossing(values, step, target, field, expected_hash):
    if array_hash(values) != expected_hash:
        raise ValueError("public dynamics differs from frozen field comparison")
    times, states = values[0], values[1:]
    sampled_frequency = np.full(len(times), np.nan)
    sampled_frequency[2:-2] = [
        _five_point(states[1], i, step) for i in range(2, len(times) - 2)]
    index = int(np.nanargmin(abs(sampled_frequency - target)))
    if not 8 <= index < len(times) - 8:
        raise ValueError("frequency crossing outside safe interpolation support")
    indices = np.arange(index - 6, index + 7)
    with torch.no_grad():
        native_frequency = np.array([
            float(field(torch.from_numpy(states[:, i].copy()))[1])
            for i in indices])
    sampled_local = sampled_frequency[indices]
    if not (np.all(np.diff(native_frequency) > 0)
            and np.all(np.diff(sampled_local) > 0)
            and native_frequency[0] < target < native_frequency[-1]
            and sampled_local[0] < target < sampled_local[-1]):
        raise ValueError("public frequency crossing is not monotone and bracketed")

    def interpolate(frequency):
        state = np.array([
            float(PchipInterpolator(frequency, states[k, indices])(target))
            for k in range(4)])
        state[1] = 0  # Orbital phase origin is arbitrary at the crossing.
        return dict(state=dict(zip(COORDINATES, state.tolist())),
                    time_M=float(PchipInterpolator(frequency, times[indices])(target)))

    return dict(index=index, sampled_nearest_frequency=float(sampled_frequency[index]),
                reference_array_sha256=expected_hash,
                native_field_frequency=interpolate(native_frequency),
                sampled_phase_frequency=interpolate(sampled_local),
                max_local_frequency_difference=float(np.max(abs(native_frequency-sampled_local))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign_path = q.ROOT / plan["campaign"]
    field_path = q.ROOT / plan["field_evidence"]
    if (plan["schema"] != "diffgw.seobnrv4.q8-initial-alignment-plan.v1"
            or plan["row_name"] != "q8_high_aligned"
            or plan["warm_start_Momega"] != 0.03162277660168379
            or plan["target_Momega"] != 0.04
            or plan["reference_output_steps_M"] != [0.125, 0.0625]
            or plan["release_qualified"] is not False
            or q.sha256(campaign_path) != plan["campaign_sha256"]
            or q.sha256(field_path) != plan["field_evidence_sha256"]):
        parser.error("frozen inputs differ")
    campaign = json.loads(campaign_path.read_text())
    prior_field = json.loads(field_path.read_text())
    row = campaign["rows"][plan["row_name"]]["row"]
    choice = campaign["choice"]
    pin = campaign["lal_binary"]
    if (campaign["status"] != "diagnostic_complete"
            or prior_field["status"] != "diagnostic_complete"
            or prior_field["row"] != row or prior_field["choice"] != choice
            or choice["physical"]["hamiltonian_log_derivative"] != "lal_v4"
            or choice["calibration"] != "lal_v4_rounded"
            or q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("campaign, public wheel, or candidate choice differs")
    torch.set_num_threads(1)
    ham, flux, vector_field = _physics(row, choice)
    sources = (
        "dev/native_rom/eob/torch/warm_start.py",
        "dev/native_rom/eob/torch/source_calibration_hamiltonian.py",
        "dev/native_rom/diagnose_lal_dynamics.py",
        "src/torchwave/approximants/eob/initial_conditions.py",
        "src/torchwave/approximants/eob/seobnrv4_flux.py",
    )
    report = dict(
        schema="diffgw.seobnrv4.q8-initial-alignment.v1", scope=q.SCOPE,
        status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), campaign_sha256=q.sha256(campaign_path),
        field_evidence_sha256=q.sha256(field_path),
        source_sha256={path: q.sha256(q.ROOT / path) for path in sources},
        lal_binary=pin, row=row, choice=choice,
        process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                     pid=os.getpid(), argv=sys.argv), grids={}, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        with torch.no_grad():
            warm = native_frequency_warm_start(
                ham, flux, torch.tensor(plan["warm_start_Momega"], dtype=torch.float64),
                torch.tensor(plan["target_Momega"], dtype=torch.float64))
            initial = array(warm.initial_state)
            target = array(warm.state)
            initial_frequency = float(vector_field(warm.initial_state)[1])
        report["native_warm_start"] = dict(
            initial_state=dict(zip(COORDINATES, initial.tolist())),
            initial_Momega=initial_frequency,
            target_state=dict(zip(COORDINATES, target.tolist())),
            crossing_time_M=float(warm.crossing_time),
            frequency_residual=float(warm.frequency_residual),
            accepted_steps=warm.accepted_steps, evaluations=warm.evaluations,
            elapsed_seconds=time.monotonic()-started)
        q.write_evidence(args.json_out, report)
        for step in plan["reference_output_steps_M"]:
            started = time.monotonic()
            values = _reference(row, step)
            expected = prior_field["grids"][str(step)]["reference_array_sha256"]
            crossing = _crossing(values, step, plan["target_Momega"],
                                 vector_field, expected)
            with torch.no_grad():
                public_initial_frequency = float(vector_field(
                    torch.from_numpy(values[1:, 0].copy()))[1])
            crossing["first_public_state"] = dict(zip(COORDINATES, values[1:, 0].tolist()))
            crossing["first_public_state_native_Momega"] = public_initial_frequency
            crossing["native_minus_first_public_state"] = dict(zip(
                ("r", "pr_star", "pphi"),
                (initial-values[1:, 0])[[0, 2, 3]].tolist()))
            for method in ("native_field_frequency", "sampled_phase_frequency"):
                public_state = crossing[method]["state"]
                crossing[method]["native_minus_public_state"] = {
                    name: float(target[i]-public_state[name])
                    for i, name in enumerate(COORDINATES) if name != "phi"}
            crossing["elapsed_seconds"] = time.monotonic()-started
            report["grids"][str(step)] = crossing
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

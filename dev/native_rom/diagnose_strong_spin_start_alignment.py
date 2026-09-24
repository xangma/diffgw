"""Correct the frozen strong-spin initial-state comparison at matched frequencies."""

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
from .diagnose_boundary_strong_spin_ode import array, array_hash
from .diagnose_lal_dynamics import _five_point, _reference
from .diagnose_strong_spin_inspiral_field import COORDINATES, _physics
from .eob.torch.warm_start import native_frequency_warm_start


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/strong_spin_start_alignment_plan.json"


def _coordinates(state):
    return dict(zip(COORDINATES, state.tolist(), strict=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_field"]
    if (plan["schema"] != "diffgw.seobnrv4.strong-spin-start-alignment-plan.v1"
            or q.sha256(prior_path) != plan["prior_field_sha256"]
            or plan["reference_output_steps_M"] != [0.125, 0.0625]
            or plan["native_start_Momega"] != 0.03162277660168379
            or plan["requested_Momega"] != 0.04):
        parser.error("frozen plan or prior field differs")
    prior = json.loads(prior_path.read_text())
    if prior["status"] != "diagnostic_complete":
        parser.error("prior field incomplete")
    pin = prior["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary identity changed")
    torch.set_num_threads(1)
    row = prior["row"]
    ham, flux, field = _physics(row)
    report = dict(
        schema="diffgw.seobnrv4.strong-spin-start-alignment.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), prior_sha256=q.sha256(prior_path),
        script_sha256=q.sha256(__file__), lal_binary=pin, row=row,
        comparisons={}, errors=[], process=dict(host=socket.gethostname(),
            cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        with torch.no_grad():
            warm = native_frequency_warm_start(
                ham, flux,
                torch.tensor(plan["native_start_Momega"], dtype=torch.float64),
                torch.tensor(plan["requested_Momega"], dtype=torch.float64),
                atol=1e-14, rtol=1e-13, first_step_M=0.125)
        native_initial, native_crossing = array(warm.initial_state), array(warm.state)
        if not np.allclose(native_crossing, np.array([
                prior["native_warm_start"]["state"][key] for key in COORDINATES]),
                rtol=0, atol=2e-12):
            raise ValueError("native warm-start crossing did not replay")
        report["native"] = dict(initial=_coordinates(native_initial),
                                crossing=_coordinates(native_crossing),
                                crossing_time_M=float(warm.crossing_time))
        q.write_evidence(args.json_out, report)
        for step in plan["reference_output_steps_M"]:
            print(f"public_step_M={step}", flush=True)
            values = _reference(row, step)
            if array_hash(values) != prior["grids"][str(step)]["reference_array_sha256"]:
                raise ValueError("public dynamics grid did not replay")
            frequency = np.gradient(values[2], values[0], edge_order=2)
            crossing_index = int(np.argmin(abs(frequency-plan["requested_Momega"])))
            if crossing_index < 4 or crossing_index >= values.shape[1]-4:
                raise ValueError("requested-frequency index outside safe support")
            public_initial = values[1:, 0].copy()
            public_crossing = values[1:, crossing_index].copy()
            public_crossing[1] = 0.0
            public_initial_frequency = float(_five_point(values[2], 2, step))
            crossing_frequency = float(_five_point(values[2], crossing_index, step))
            report["comparisons"][str(step)] = dict(
                reference_array_sha256=array_hash(values),
                public_first_state_frequency_Momega=public_initial_frequency,
                public_crossing_frequency_Momega=crossing_frequency,
                public_crossing_index=crossing_index,
                public_crossing_time_M=float(values[0, crossing_index]),
                native_initial=_coordinates(native_initial),
                public_initial=_coordinates(public_initial),
                native_minus_public_initial=_coordinates(native_initial-public_initial),
                native_crossing=_coordinates(native_crossing),
                public_crossing_phase_zero=_coordinates(public_crossing),
                native_minus_public_crossing=_coordinates(native_crossing-public_crossing))
            q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

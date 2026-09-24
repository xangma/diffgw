"""Post-hoc public-SEOBNRv4 output-grid self-comparison for the q10 boundary row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

from . import qualify_generator as q
from .diagnose_frequency_anchor_polar import _polar_metrics
from .diagnose_peak_anchor import public_mode
from .diagnose_q8_ringdown_peak_bound import _frequency_metrics


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/near_extreme_public_grid_plan.json"
ROW = "q10_both_near_extreme_aligned"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("evidence path already exists")
    plan = json.loads(PLAN.read_text())
    grid_path = q.ROOT / plan["grid_result"]["path"]
    screen_path = q.ROOT / plan["screen"]["path"]
    baseline_path = q.ROOT / plan["baseline_parity"]["path"]
    if (plan["schema"] != "diffgw.seobnrv4.near-extreme-public-grid-plan.v1"
            or plan["row"] != ROW
            or plan["levels"] != ["ode_2", "grid_1", "grid_2"]
            or plan["reference_level"] != "grid_2"
            or plan["anchor_angular_frequency_M"] != 0.1
            or plan["release_qualified"] is not False
            or any(q.sha256(path) != plan[name]["sha256"]
                   for name, path in (("grid_result", grid_path),
                                      ("screen", screen_path),
                                      ("baseline_parity", baseline_path)))):
        parser.error("frozen plan or inputs changed")
    grid = json.loads(grid_path.read_text())
    screen = json.loads(screen_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    pin = grid["lal_binary"]
    if (grid["status"] != "diagnostic_complete"
            or baseline["status"] != "diagnostic_complete"
            or screen["status"] != "diagnostic_numerical_gates_passed"
            or pin != baseline["lal_binary"]
            or q.lal_binary_identity(pin["lal_version"],
                                     pin["lalsimulation_version"]) != pin):
        parser.error("pinned public binary or prerequisites changed")

    report = dict(schema="diffgw.seobnrv4.near-extreme-public-grid.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  grid_result_sha256=q.sha256(grid_path), row=ROW,
                  lal_binary=pin, levels={}, comparisons={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        row = baseline["rows"][ROW]["row"]
        modes = {}
        for name in plan["levels"]:
            frozen = next(run for run in screen["rows"][ROW]["runs"]
                          if run["name"] == name)
            # public_mode ignores the native arrays when constructing the LAL
            # waveform; these one-sample placeholders keep its exact call path.
            dummy = dict(dt_geom=frozen["dt_geom"], peak_time_M=0.,
                         time=np.array([0.]), mode=np.array([0.j]))
            _, _, time, mode, mass_seconds = public_mode(row, dummy)
            self_anchor = _frequency_metrics(
                time, mode, time, mode, mass_seconds,
                plan["anchor_angular_frequency_M"])["reference_anchor"]
            prior = (baseline["rows"][ROW]["linear"]["reference_anchor"]
                     if name == "ode_2"
                     else grid["levels"][name]["linear"]["reference_anchor"])
            if (abs(self_anchor["time_s"] - prior["time_s"]) > 1e-10
                    or abs(self_anchor["peak_lag_M"] - prior["peak_lag_M"]) > 1e-7):
                raise ValueError(f"{name} public waveform does not reproduce frozen anchor")
            modes[name] = time, mode
            report["levels"][name] = dict(
                dt_geom=frozen["dt_geom"], samples=len(time),
                time_sha256=hashlib.sha256(np.ascontiguousarray(time).tobytes()).hexdigest(),
                mode_sha256=hashlib.sha256(np.ascontiguousarray(mode).tobytes()).hexdigest(),
                anchor=self_anchor)
            q.write_evidence(args.json_out, report)
        reference_time, reference_mode = modes[plan["reference_level"]]
        for name in plan["levels"][:-1]:
            candidate_time, candidate_mode = modes[name]
            linear = _frequency_metrics(
                candidate_time, candidate_mode, reference_time, reference_mode,
                mass_seconds, plan["anchor_angular_frequency_M"])
            polar = _polar_metrics(candidate_time, candidate_mode,
                                   reference_time, reference_mode, mass_seconds,
                                   linear)
            report["comparisons"][name] = dict(linear=linear, polar=polar)
            q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])))
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

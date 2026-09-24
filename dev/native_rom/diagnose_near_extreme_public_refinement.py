"""Check pinned public SEOBNRv4 post-peak convergence below 0.125 M sampling."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/near_extreme_public_refinement_plan.json"
ROW = "q10_both_near_extreme_aligned"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("evidence path already exists")
    plan = json.loads(PLAN.read_text())
    path = q.ROOT / plan["public_grid_result"]["path"]
    baseline_path = q.ROOT / plan["baseline_parity"]["path"]
    if (plan["schema"] != "diffgw.seobnrv4.near-extreme-public-refinement-plan.v1"
            or plan["row"] != ROW
            or plan["dt_levels_M"] != [0.125, 0.0625, 0.03125, 0.015625]
            or plan["anchor_angular_frequency_M"] != 0.1
            or plan["release_qualified"] is not False
            or q.sha256(path) != plan["public_grid_result"]["sha256"]
            or q.sha256(baseline_path) != plan["baseline_parity"]["sha256"]):
        parser.error("frozen plan or inputs changed")
    prior = json.loads(path.read_text())
    baseline = json.loads(baseline_path.read_text())
    pin = prior["lal_binary"]
    if (prior["status"] != "diagnostic_complete"
            or baseline["status"] != "diagnostic_complete"
            or pin != baseline["lal_binary"]
            or q.lal_binary_identity(pin["lal_version"],
                                     pin["lalsimulation_version"]) != pin):
        parser.error("pinned public binary or prerequisites changed")
    report = dict(schema="diffgw.seobnrv4.near-extreme-public-refinement.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  public_grid_result_sha256=q.sha256(path), lal_binary=pin,
                  row=ROW, levels={}, comparisons={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        row = baseline["rows"][ROW]["row"]
        modes = {}
        for dt in plan["dt_levels_M"]:
            name = f"dt_{dt:.6f}M"
            dummy = dict(dt_geom=dt, peak_time_M=0., time=np.array([0.]),
                         mode=np.array([0.j]))
            _, _, time, mode, mass_seconds = public_mode(row, dummy)
            modes[name] = time, mode
            anchor = _frequency_metrics(time, mode, time, mode, mass_seconds,
                                        plan["anchor_angular_frequency_M"])[
                                            "reference_anchor"]
            if dt == 0.125:
                frozen = prior["levels"]["grid_2"]
                if (hashlib.sha256(np.ascontiguousarray(time).tobytes()).hexdigest()
                        != frozen["time_sha256"]
                        or hashlib.sha256(np.ascontiguousarray(mode).tobytes()).hexdigest()
                        != frozen["mode_sha256"]):
                    raise ValueError("0.125 M pinned public waveform changed")
            report["levels"][name] = dict(dt_geom=dt, samples=len(time),
                                           anchor=anchor)
            q.write_evidence(args.json_out, report)
        names = list(modes)
        pairs = [(names[i], names[i+1]) for i in range(len(names)-1)]
        pairs += [(name, names[-1]) for name in names[:-2]]
        for candidate, reference in pairs:
            nt, nm = modes[candidate]
            rt, rm = modes[reference]
            linear = _frequency_metrics(nt, nm, rt, rm, mass_seconds,
                                        plan["anchor_angular_frequency_M"])
            polar = _polar_metrics(nt, nm, rt, rm, mass_seconds, linear)
            report["comparisons"][f"{candidate}_vs_{reference}"] = dict(
                linear=linear, polar=polar)
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

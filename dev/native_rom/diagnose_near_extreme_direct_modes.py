"""Compare public v4 direct 22 modes and polarization output-grid behavior."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import lal
import lalsimulation as lalsim
import numpy as np

from . import qualify_generator as q
from .diagnose_frequency_anchor_polar import _polar_metrics
from .diagnose_public_nqc_mode_boundary import public_call
from .diagnose_q8_ringdown_peak_bound import _frequency_metrics


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/near_extreme_direct_modes_plan.json"
ROW = "q10_both_near_extreme_aligned"


def direct_mode(row, step_M):
    modes, _, _ = public_call(row, step_M, 0)[0]
    series = lalsim.SphHarmTimeSeriesGetMode(modes, 2, 2)
    time = float(series.epoch) + np.arange(len(series.data.data)) * float(series.deltaT)
    mode = np.asarray(series.data.data, dtype=np.complex128)
    mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    if (not np.isfinite(time).all() or not np.isfinite(mode).all()
            or abs(float(series.deltaT) / mass_seconds - step_M) > 1e-10):
        raise ValueError("invalid direct public mode grid")
    return time, mode, mass_seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("evidence path already exists")
    plan = json.loads(PLAN.read_text())
    refinement_path = q.ROOT / plan["refinement_result"]["path"]
    baseline_path = q.ROOT / plan["baseline_parity"]["path"]
    if (plan["schema"] != "diffgw.seobnrv4.near-extreme-direct-modes-plan.v1"
            or plan["row"] != ROW
            or plan["pairs"] != [[0.0625, 0.03125], [0.03125, 0.015625]]
            or plan["anchor_angular_frequency_M"] != 0.1
            or plan["release_qualified"] is not False
            or q.sha256(refinement_path) != plan["refinement_result"]["sha256"]
            or q.sha256(baseline_path) != plan["baseline_parity"]["sha256"]):
        parser.error("frozen plan or inputs changed")
    prior = json.loads(refinement_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    pin = prior["lal_binary"]
    if (prior["status"] != "diagnostic_complete"
            or baseline["status"] != "diagnostic_complete"
            or pin != baseline["lal_binary"]
            or q.lal_binary_identity(pin["lal_version"],
                                     pin["lalsimulation_version"]) != pin):
        parser.error("pinned public binary or prerequisites changed")
    report = dict(schema="diffgw.seobnrv4.near-extreme-direct-modes.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  refinement_result_sha256=q.sha256(refinement_path),
                  lal_binary=pin, row=ROW, levels={}, comparisons={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        row = baseline["rows"][ROW]["row"]
        modes = {}
        for dt in sorted({dt for pair in plan["pairs"] for dt in pair}):
            time, mode, mass_seconds = direct_mode(row, dt)
            modes[dt] = time, mode
            report["levels"][f"dt_{dt:.6f}M"] = dict(
                dt_geom=dt, samples=len(time), epoch_s=float(time[0]))
            q.write_evidence(args.json_out, report)
        for coarse, fine in plan["pairs"]:
            nt, nm = modes[coarse]
            rt, rm = modes[fine]
            linear = _frequency_metrics(nt, nm, rt, rm, mass_seconds,
                                        plan["anchor_angular_frequency_M"])
            polar = _polar_metrics(nt, nm, rt, rm, mass_seconds, linear)
            key = f"dt_{coarse:.6f}M_vs_dt_{fine:.6f}M"
            report["comparisons"][key] = dict(
                direct_mode=dict(linear=linear, polar=polar),
                polarization_postpeak_complex_relative_l2=prior[
                    "comparisons"][key]["polar"]["windows"]["post_peak"][
                        "complex_relative_l2"])
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

"""Check whether q10 public high-rate EOB states vary with output spacing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

from . import qualify_generator as q
from .diagnose_public_nqc_mode_boundary import public_call


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/near_extreme_public_dynamics_grid_plan.json"
ROW = "q10_both_near_extreme_aligned"


def public_high(row, step):
    _, _, high = public_call(row, step, 0)[0]
    values = np.asarray(high.data, dtype=np.float64).reshape(5, -1)
    time, radius = values[:2]
    if (not np.isfinite(values).all() or len(time) < 200
            or not np.all(np.diff(time) > 0)):
        raise ValueError("invalid public high-rate dynamics")
    orbital_frequency = np.gradient(values[2], time, edge_order=2)
    peak = int(np.argmax(orbital_frequency))
    pre = values[:, :peak+1]
    if len(pre[0]) < 100 or not np.all(np.diff(pre[1]) < 0):
        raise ValueError("pre-peak public radius is not monotonically decreasing")
    return pre, dict(samples=len(time), pre_peak_samples=pre.shape[1],
                     radius_range=[float(pre[1, -1]), float(pre[1, 0])],
                     orbital_peak_time_M=float(time[peak]),
                     orbital_peak_frequency_M=float(orbital_frequency[peak]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("evidence path already exists")
    plan = json.loads(PLAN.read_text())
    direct_path = q.ROOT / plan["direct_mode_result"]["path"]
    baseline_path = q.ROOT / plan["baseline_parity"]["path"]
    if (plan["schema"] != "diffgw.seobnrv4.near-extreme-public-dynamics-grid-plan.v1"
            or plan["row"] != ROW
            or plan["dt_levels_M"] != [0.03125, 0.015625]
            or plan["common_radius_samples"] != 200
            or plan["release_qualified"] is not False
            or q.sha256(direct_path) != plan["direct_mode_result"]["sha256"]
            or q.sha256(baseline_path) != plan["baseline_parity"]["sha256"]):
        parser.error("frozen plan or inputs changed")
    prior = json.loads(direct_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    pin = prior["lal_binary"]
    if (prior["status"] != "diagnostic_complete"
            or baseline["status"] != "diagnostic_complete"
            or pin != baseline["lal_binary"]
            or q.lal_binary_identity(pin["lal_version"],
                                     pin["lalsimulation_version"]) != pin):
        parser.error("pinned public binary or prerequisites changed")
    report = dict(schema="diffgw.seobnrv4.near-extreme-public-dynamics-grid.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  direct_mode_result_sha256=q.sha256(direct_path),
                  lal_binary=pin, row=ROW, levels={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        row = baseline["rows"][ROW]["row"]
        states = {}
        for dt in plan["dt_levels_M"]:
            states[dt], summary = public_high(row, dt)
            report["levels"][f"dt_{dt:.6f}M"] = summary
            q.write_evidence(args.json_out, report)
        coarse, fine = [states[dt] for dt in plan["dt_levels_M"]]
        lower = max(coarse[1, -1], fine[1, -1])
        upper = min(coarse[1, 0], fine[1, 0])
        if not lower < upper:
            raise ValueError("no common pre-peak radius support")
        radius = np.linspace(lower, upper, plan["common_radius_samples"])
        comparison = {}
        for name, index in (("time_M", 0), ("orbital_phase", 2),
                            ("radial_momentum", 3), ("angular_momentum", 4)):
            a = np.interp(radius, coarse[1, ::-1], coarse[index, ::-1])
            b = np.interp(radius, fine[1, ::-1], fine[index, ::-1])
            if name in {"time_M", "orbital_phase"}:
                a -= a[-1]
                b -= b[-1]
            delta = a - b
            comparison[name] = dict(max_absolute=float(np.max(np.abs(delta))),
                                    rms_absolute=float(np.sqrt(np.mean(delta**2))))
        report["comparison"] = dict(common_radius_range=[lower, upper],
                                    samples=len(radius), fields=comparison,
                                    direct_postpeak_complex_relative_l2=prior[
                                        "comparisons"]["dt_0.031250M_vs_dt_0.015625M"][
                                            "direct_mode"]["polar"]["windows"][
                                                "post_peak"]["complex_relative_l2"])
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])))
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Separate public-output phase-anchor effects from post-peak shape changes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

from . import qualify_generator as q
from .diagnose_frequency_anchor_polar import _polar_metrics, polar_interpolate
from .diagnose_peak_anchor import fitted_peak, public_mode
from .diagnose_q8_ringdown_peak_bound import _frequency_metrics


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/near_extreme_public_alignment_plan.json"
ROW = "q10_both_near_extreme_aligned"


def peak_polar_metrics(nt, nm, rt, rm, mass_seconds):
    """Compare post-peak modes after each waveform's own local peak fit."""
    npk = fitted_peak(nt, nm, "quadratic_power")
    rpk = fitted_peak(rt, rm, "quadratic_power")
    ntime = (nt - npk["time_s"]) / mass_seconds
    rtime = (rt - rpk["time_s"]) / mass_seconds
    nmode = nm * np.exp(-1j * npk["phase_rad"])
    rmode = rm * np.exp(-1j * rpk["phase_rad"])
    common = (rtime >= ntime[0]) & (rtime <= ntime[-1])
    times = rtime[common]
    if times.size < 3 or (times >= 0).sum() < 3:
        raise ValueError("too few common post-peak samples")
    candidate = polar_interpolate(times, ntime, nmode)
    reference = rmode[common]
    return dict(native_peak=npk, reference_peak=rpk,
                whole=q.mode_metrics(candidate, reference),
                post_peak=q.mode_metrics(candidate[times >= 0],
                                         reference[times >= 0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("evidence path already exists")
    plan = json.loads(PLAN.read_text())
    path = q.ROOT / plan["refinement_result"]["path"]
    baseline_path = q.ROOT / plan["baseline_parity"]["path"]
    if (plan["schema"] != "diffgw.seobnrv4.near-extreme-public-alignment-plan.v1"
            or plan["row"] != ROW
            or plan["pairs"] != [[0.0625, 0.03125], [0.03125, 0.015625]]
            or plan["anchor_angular_frequency_M"] != 0.1
            or plan["release_qualified"] is not False
            or q.sha256(path) != plan["refinement_result"]["sha256"]
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
    report = dict(schema="diffgw.seobnrv4.near-extreme-public-alignment.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  refinement_result_sha256=q.sha256(path), lal_binary=pin,
                  row=ROW, comparisons={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        row = baseline["rows"][ROW]["row"]
        modes = {}
        for dt in sorted({dt for pair in plan["pairs"] for dt in pair}):
            dummy = dict(dt_geom=dt, peak_time_M=0., time=np.array([0.]),
                         mode=np.array([0.j]))
            _, _, time, mode, mass_seconds = public_mode(row, dummy)
            modes[dt] = time, mode
            name = f"dt_{dt:.6f}M"
            if len(time) != prior["levels"][name]["samples"]:
                raise ValueError(f"{name} public sample count changed")
        for coarse, fine in plan["pairs"]:
            nt, nm = modes[coarse]
            rt, rm = modes[fine]
            linear = _frequency_metrics(nt, nm, rt, rm, mass_seconds,
                                        plan["anchor_angular_frequency_M"])
            frequency_polar = _polar_metrics(nt, nm, rt, rm, mass_seconds,
                                             linear)
            key = f"dt_{coarse:.6f}M_vs_dt_{fine:.6f}M"
            frozen = prior["comparisons"][key]["polar"]["windows"][
                "post_peak"]["complex_relative_l2"]
            actual = frequency_polar["windows"]["post_peak"]["complex_relative_l2"]
            if abs(actual - frozen) > 1e-12:
                raise ValueError(f"{key} public frequency comparison changed")
            report["comparisons"][key] = dict(
                frequency_postpeak_complex_relative_l2=actual,
                peak_polar=peak_polar_metrics(nt, nm, rt, rm, mass_seconds))
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

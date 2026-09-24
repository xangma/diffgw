"""Confirm ultrafine public q10 discrepancy in directly returned (2,2) modes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

from . import qualify_generator as q
from .diagnose_frequency_anchor_polar import _polar_metrics
from .diagnose_near_extreme_direct_modes import direct_mode
from .diagnose_q8_ringdown_peak_bound import _frequency_metrics


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/near_extreme_public_ultrafine_direct_plan.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("evidence path already exists")
    plan = json.loads(PLAN.read_text())
    path = q.ROOT / plan["ultrafine"]["path"]
    if (plan["schema"] != "diffgw.seobnrv4.near-extreme-public-ultrafine-direct-plan.v1"
            or plan["row"] != "q10_both_near_extreme_aligned"
            or plan["pair_M"] != [0.0078125, 0.00390625]
            or plan["anchor_angular_frequency_M"] != 0.1
            or plan["release_qualified"] is not False
            or q.sha256(path) != plan["ultrafine"]["sha256"]):
        parser.error("frozen ultrafine direct-mode plan differs")
    prior = json.loads(path.read_text())
    pin = prior["lal_binary"]
    if (prior["status"] != "diagnostic_complete"
            or q.lal_binary_identity(pin["lal_version"],
                                     pin["lalsimulation_version"]) != pin):
        parser.error("pinned public result differs")
    report = dict(
        schema="diffgw.seobnrv4.near-extreme-public-ultrafine-direct.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        input_sha256=q.sha256(path), lal_binary=pin, row=prior["row"],
        levels={}, comparison={}, errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        modes = {}
        for dt in plan["pair_M"]:
            time, mode, mass_seconds = direct_mode(prior["row"], dt)
            modes[dt] = time, mode
            report["levels"][f"dt_{dt:.6f}M"] = dict(samples=len(time))
            q.write_evidence(args.json_out, report)
        coarse, fine = plan["pair_M"]
        nt, nm = modes[coarse]
        rt, rm = modes[fine]
        linear = _frequency_metrics(nt, nm, rt, rm, mass_seconds,
                                    plan["anchor_angular_frequency_M"])
        polar = _polar_metrics(nt, nm, rt, rm, mass_seconds, linear)
        key = f"dt_{coarse:.6f}M_vs_dt_{fine:.6f}M"
        inverted = prior["comparisons"][key]["polar"]["windows"]["post_peak"][
            "complex_relative_l2"]
        direct = polar["windows"]["post_peak"]["complex_relative_l2"]
        if abs(direct-inverted) > 1e-7:
            raise ValueError("direct and polarization-inverted grid changes differ")
        report["comparison"] = dict(
            direct_mode=dict(linear=linear, polar=polar),
            polarization_postpeak_complex_relative_l2=inverted,
            postpeak_l2_difference=float(direct-inverted))
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

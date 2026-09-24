"""Extend pinned public q10 output-cadence study to ultrafine steps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

from . import qualify_generator as q
from .diagnose_frequency_anchor_polar import _polar_metrics
from .diagnose_peak_anchor import public_mode
from .diagnose_public_nqc_mode_boundary import public_call
from .diagnose_q8_ringdown_peak_bound import _frequency_metrics


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/near_extreme_public_ultrafine_plan.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("evidence path already exists")
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / item["path"] for name, item in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.near-extreme-public-ultrafine-plan.v1"
            or plan["row"] != "q10_both_near_extreme_aligned"
            or plan["dt_levels_M"] != [0.03125, 0.015625, 0.0078125, 0.00390625]
            or plan["anchor_angular_frequency_M"] != 0.1
            or plan["release_qualified"] is not False
            or any(q.sha256(paths[name]) != item["sha256"]
                   for name, item in plan["inputs"].items())):
        parser.error("frozen ultrafine plan or prerequisites changed")
    prior = json.loads(paths["refinement"].read_text())
    nqc = json.loads(paths["nqc"].read_text())
    parity = json.loads(paths["parity"].read_text())
    pin = prior["lal_binary"]
    if (prior["status"] != "diagnostic_complete"
            or nqc["status"] != "diagnostic_complete"
            or prior["row"] != plan["row"] or nqc["row"] != plan["row"]
            or pin != nqc["lal_binary"] or pin != parity["lal_binary"]
            or q.lal_binary_identity(pin["lal_version"],
                                     pin["lalsimulation_version"]) != pin):
        parser.error("pinned public prerequisite differs")
    row = parity["rows"][plan["row"]]["row"]
    report = dict(
        schema="diffgw.seobnrv4.near-extreme-public-ultrafine.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        input_sha256={name: q.sha256(path) for name, path in paths.items()},
        lal_binary=pin, row=row, levels={}, comparisons={}, errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        modes = {}
        coefficients = {}
        for dt in plan["dt_levels_M"]:
            key = f"dt_{dt:.6f}M"
            dummy = dict(dt_geom=dt, peak_time_M=0., time=np.array([0.]),
                         mode=np.array([0.j]))
            _, _, time, mode, mass_seconds = public_mode(row, dummy)
            output, coefficient = public_call(row, dt, 1)
            if list(output) != [None, None, None] or not np.isfinite(coefficient).all():
                raise ValueError(f"invalid public NQC output at {key}")
            if dt in (0.03125, 0.015625):
                expected = np.asarray(nqc["levels"][key])
                if not np.array_equal(coefficient, expected):
                    raise ValueError(f"pinned public NQC coefficients changed: {key}")
            modes[key] = time, mode
            coefficients[key] = coefficient
            report["levels"][key] = dict(
                samples=len(time), nqc_coefficients=coefficient.tolist(),
                anchor=_frequency_metrics(time, mode, time, mode, mass_seconds,
                    plan["anchor_angular_frequency_M"])["reference_anchor"])
            q.write_evidence(args.json_out, report)
        names = list(modes)
        for candidate, reference in zip(names[:-1], names[1:]):
            nt, nm = modes[candidate]
            rt, rm = modes[reference]
            linear = _frequency_metrics(nt, nm, rt, rm, mass_seconds,
                                        plan["anchor_angular_frequency_M"])
            polar = _polar_metrics(nt, nm, rt, rm, mass_seconds, linear)
            if candidate == names[0]:
                expected = prior["comparisons"][f"{candidate}_vs_{reference}"]["polar"]
                if abs(polar["windows"]["post_peak"]["complex_relative_l2"]
                       - expected["windows"]["post_peak"]["complex_relative_l2"]) > 1e-9:
                    raise ValueError("original fine-grid public comparison changed")
            report["comparisons"][f"{candidate}_vs_{reference}"] = dict(
                polar=polar, linear=linear,
                nqc_active_l2=float(np.linalg.norm(
                    (coefficients[candidate]-coefficients[reference])[[0, 1, 2, 6, 7]])))
            q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

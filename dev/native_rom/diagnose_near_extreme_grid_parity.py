"""Test finer output sampling after the frozen q10 post-peak parity miss."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch

from . import qualify_generator as q
from .diagnose_frequency_anchor_polar import _polar_metrics, _source_rkf_run
from .diagnose_peak_anchor import public_mode
from .diagnose_q8_ringdown_peak_bound import _frequency_metrics


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/near_extreme_grid_parity_plan.json"
ROW = "q10_both_near_extreme_aligned"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("evidence path already exists")
    plan = json.loads(PLAN.read_text())
    screen_path = q.ROOT / plan["screen"]["path"]
    parity_path = q.ROOT / plan["baseline_parity"]["path"]
    if (plan["schema"] != "diffgw.seobnrv4.near-extreme-grid-parity-plan.v1"
            or plan["row"] != ROW or plan["levels"] != ["grid_1", "grid_2"]
            or plan["release_qualified"] is not False
            or plan["anchor_angular_frequency_M"] != 0.1
            or q.sha256(screen_path) != plan["screen"]["sha256"]
            or q.sha256(parity_path) != plan["baseline_parity"]["sha256"]):
        parser.error("frozen plan or inputs changed")
    screen = json.loads(screen_path.read_text())
    baseline = json.loads(parity_path.read_text())
    if (screen["status"] != "diagnostic_numerical_gates_passed"
            or not screen["all_numerical_checks_passed"]
            or baseline["status"] != "diagnostic_complete"
            or baseline["rows"][ROW]["gates"] != dict(
                whole_complex=True, postpeak_complex=False, peak_lag=True)):
        parser.error("expected completed numerical screen and frozen parity miss")
    pin = plan["lal_binary"]
    if (pin != baseline["lal_binary"] or q.lal_binary_identity(
            pin["lal_version"], pin["lalsimulation_version"]) != pin):
        parser.error("public binary differs from frozen baseline")
    current = q.source_snapshot()
    runner = q.ROOT / "dev/native_rom/diagnose_source_rkf_domain.py"
    current[str(runner.relative_to(q.ROOT))] = q.sha256(runner)
    if current != screen["source_sha256_after"]:
        parser.error("generator sources differ from frozen screen")

    torch.set_num_threads(1)
    report = dict(schema="diffgw.seobnrv4.near-extreme-grid-parity.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  screen_sha256=q.sha256(screen_path),
                  baseline_parity_sha256=q.sha256(parity_path),
                  lal_binary=pin, row=ROW, levels={}, errors=[], process=dict(
                      host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                      argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        row = screen["rows"][ROW]["inputs"]
        settings = q.Settings(**screen["settings"])
        physical = screen["physical_choice"]
        for name in plan["levels"]:
            frozen = next(run for run in screen["rows"][ROW]["runs"]
                          if run["name"] == name)
            level = {key: frozen[key] for key in ("name", "atol", "rtol", "dt_geom")}
            started = time.monotonic()
            run = _source_rkf_run(row, level, settings, physical,
                                  physical["flux"], screen["numerical_choice"])
            summary = q.summarize_run(run)
            if summary["array_sha256"] != frozen["array_sha256"]:
                raise ValueError(f"{name} native waveform differs from frozen screen")
            result = dict(native_array_sha256=frozen["array_sha256"],
                          native_elapsed_seconds=time.monotonic() - started)
            report["levels"][name] = result
            q.write_evidence(args.json_out, report)
            print(f"{name} native waveform replayed", flush=True)
            nt, nm, rt, rm, mass_seconds = public_mode(row, run)
            linear = _frequency_metrics(nt, nm, rt, rm, mass_seconds,
                                        plan["anchor_angular_frequency_M"])
            polar = _polar_metrics(nt, nm, rt, rm, mass_seconds, linear)
            post = polar["windows"]["post_peak"]["complex_relative_l2"]
            result.update(linear=linear, polar=polar,
                          postpeak_complex_relative_l2=post,
                          postpeak_limit_passed=post <= plan["frozen_postpeak_limit"],
                          public_samples=len(rt))
            q.write_evidence(args.json_out, report)
            print(f"{name} public comparison complete: postpeak={post}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

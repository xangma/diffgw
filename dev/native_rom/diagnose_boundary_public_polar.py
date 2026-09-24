"""Compare frozen antialigned numerical rows with the pinned public v4 binary."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/boundary_public_polar_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("evidence path already exists")
    plan = json.loads(PLAN.read_text())
    names = ["q8_strong_antialigned", "q1_extreme_antialigned"]
    if (plan["schema"] != "diffgw.seobnrv4.boundary-public-polar-plan.v1"
            or plan["rows"] != names or plan["release_qualified"] is not False
            or plan["anchor_angular_frequency_M"] != 0.1):
        parser.error("unexpected frozen plan")
    paths = {name: q.ROOT / plan["screens"][name]["path"] for name in names}
    control_path = q.ROOT / plan["public_control"]["path"]
    if (any(q.sha256(paths[name]) != plan["screens"][name]["sha256"] for name in names)
            or q.sha256(control_path) != plan["public_control"]["sha256"]):
        parser.error("frozen inputs changed")
    screens = {name: json.loads(paths[name].read_text()) for name in names}
    if any(screen["status"] != "diagnostic_numerical_gates_passed"
           or not screen["all_numerical_checks_passed"]
           or screen["selected_rows"] != [name] for name, screen in screens.items()):
        parser.error("numerical screens did not pass")
    if len({screen["source_manifest_sha256"] for screen in screens.values()}) != 1:
        parser.error("screens use different source manifests")
    control = json.loads(control_path.read_text())
    pin = control["lal_binary"]
    if q.lal_binary_identity(pin["lal_version"], pin["lalsimulation_version"]) != pin:
        parser.error("public binary differs from frozen control")
    torch.set_num_threads(1)
    report = dict(schema="diffgw.seobnrv4.boundary-public-polar.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={name: plan["screens"][name]["sha256"] for name in names},
                  public_binary_sha256=q.canonical_hash(pin), rows={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        for name in names:
            screen = screens[name]
            row = screen["rows"][name]["inputs"]
            settings = q.Settings(**screen["settings"])
            level = dict(name="ode_2", atol=settings.atol_levels[2],
                         rtol=settings.rtol_levels[2], dt_geom=settings.dt_levels_M[0])
            started = time.monotonic()
            physical = screen["physical_choice"]
            run = _source_rkf_run(row, level, settings, physical,
                                  physical["flux"], screen["numerical_choice"])
            summary = q.summarize_run(run)
            expected = screen["rows"][name]["runs"][2]["array_sha256"]
            if summary["array_sha256"] != expected:
                raise ValueError(f"{name} native waveform differs from frozen screen")
            result = dict(row=row, native_array_sha256=expected,
                          native_elapsed_seconds=time.monotonic() - started)
            report["rows"][name] = result
            q.write_evidence(args.json_out, report)
            print(f"{name} native waveform replayed", flush=True)
            nt, nm, rt, rm, mass_seconds = public_mode(row, run)
            linear = _frequency_metrics(nt, nm, rt, rm, mass_seconds,
                                        plan["anchor_angular_frequency_M"])
            polar = _polar_metrics(nt, nm, rt, rm, mass_seconds, linear)
            gates = dict(
                whole_complex=polar["metrics"]["complex_relative_l2"]
                <= plan["max_whole_complex_relative_l2"],
                postpeak_complex=polar["windows"]["post_peak"]["complex_relative_l2"]
                <= plan["max_postpeak_complex_relative_l2"],
                peak_lag=abs(linear["native_minus_reference_peak_lag_M"])
                <= plan["max_abs_peak_lag_M"])
            result.update(linear=linear, polar=polar, gates=gates,
                          parity_passed=all(gates.values()), public_samples=len(rt))
            q.write_evidence(args.json_out, report)
            print(f"{name} public comparison complete: {gates}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

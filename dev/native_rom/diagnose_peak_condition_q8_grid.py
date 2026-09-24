"""Test requested output-grid sensitivity of the current q8 waveform."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import torch

from . import qualify_generator as q
from .diagnose_source_q4_q8_mode import _run


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/peak_condition_q8_grid_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / path for name, path in plan["inputs"].items()}
    baseline = dict(name="ode_3", atol=3e-15, rtol=3e-14, dt_geom=0.5)
    finer = dict(name="grid_1", atol=3e-15, rtol=3e-14, dt_geom=0.25)
    if (plan["schema"] != "diffgw.seobnrv4.peak-condition-q8-grid-plan.v1"
            or plan["row"] != "q8_high_aligned"
            or plan["baseline_level"] != baseline
            or plan["finer_level"] != finer
            or plan["relative_l2_limit"] != 1e-6
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen q8 grid plan or inputs differ")
    campaign = json.loads(paths["campaign"].read_text())
    prior = json.loads(paths["tighter_ode"].read_text())
    item = campaign["rows"][plan["row"]]
    row = item["row"]
    if (campaign["status"] != "diagnostic_complete"
            or prior["status"] != "diagnostic_complete"
            or prior["additional_gate_passed"] is not True
            or prior["row"] != row
            or prior["runs"]["ode_3"]["level"] != baseline):
        parser.error("pinned q8 predecessors changed")
    scope = dict(q.SCOPE)
    scope["description"] = "Requested output-grid sensitivity of current peak-preserving q8 waveform."
    scope["known_limitations"] = [
        "Only one output-step refinement is tested; this does not establish asymptotic convergence.",
        "The requested output step may also change the native internal high-rate step.",
        "A one-row numerical check does not establish physical or domain-wide accuracy.",
    ]
    report = dict(schema="diffgw.seobnrv4.peak-condition-q8-grid.v1",
                  status="running", release_qualified=False, scope=scope,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], row=row,
                  levels=dict(baseline=baseline, finer=finer), runs={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        torch.set_num_threads(1)
        settings = q.Settings(**campaign["settings"])
        runs = []
        for name, level in (("ode_3", baseline), ("grid_1", finer)):
            started = time.monotonic()
            run = _run(row, level, settings, campaign["choice"],
                       item["fixed_restart_time_M"])
            summary = q.summarize_run(run)
            if name == "ode_3" and summary["array_sha256"] != (
                    prior["runs"]["ode_3"]["native"]["array_sha256"]):
                raise ValueError("current peak-preserving q8 baseline changed")
            report["runs"][name] = dict(level=level, native=summary,
                                         elapsed_seconds=time.monotonic()-started)
            q.write_evidence(args.json_out, report)
            print(f"level={name} complete", flush=True)
            runs.append(run)
        comparison = q.compare_runs(*runs)
        limit = plan["relative_l2_limit"]
        gate = dict(
            complex_relative_l2=comparison["complex_relative_l2"] <= limit,
            amplitude_relative_l2=comparison["amplitude_relative_l2"] <= limit,
            match_time_M=comparison["match_time_M_abs_difference"] <= settings.event_time_limit_M,
            peak_time_M=comparison["peak_time_M_abs_difference"] <= settings.event_time_limit_M,
            boundary_time_M=comparison["boundary_time_M_abs_difference"] <= settings.event_time_limit_M,
        )
        report.update(comparison=comparison, gates=gate,
                      grid_gate_passed=all(gate.values()),
                      status="diagnostic_complete")
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc),
                                     traceback=traceback.format_exc()))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          grid_gate_passed=report.get("grid_gate_passed"))),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

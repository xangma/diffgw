"""Extend current peak-preserving q8 mode by one frozen ODE tolerance."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/peak_condition_q8_tighter_ode_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / path for name, path in plan["inputs"].items()}
    expected = dict(name="ode_3", atol=3e-15, rtol=3e-14, dt_geom=0.5)
    if (plan["schema"] != "diffgw.seobnrv4.peak-condition-q8-tighter-ode-plan.v1"
            or plan["row"] != "q8_high_aligned"
            or plan["baseline_level"] != "ode_2"
            or plan["additional_level"] != expected
            or plan["relative_l2_limit"] != 1e-6
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen q8 ODE plan or inputs differ")
    campaign = json.loads(paths["campaign"].read_text())
    integrated = json.loads(paths["integrated_candidate"].read_text())
    prior = json.loads(paths["prior_tighter_ode"].read_text())
    item = campaign["rows"][plan["row"]]
    row = item["row"]
    if (any(x["status"] != "diagnostic_complete"
            for x in (campaign, integrated, prior))
            or integrated["rows"][plan["row"]]["row"] != row
            or prior["row"] != row
            or prior["additional_gate_passed"] is not True
            or prior["runs"]["ode_3"]["level"] != expected):
        parser.error("pinned q8 predecessors changed")
    scope = dict(q.SCOPE)
    scope["description"] = "Numerical tolerance extension on the current peak-preserving q8 waveform."
    scope["known_limitations"] = [
        "The chosen tolerance passed on the earlier ringdown variant; this is not an independent choice.",
        "A one-row ODE check cannot establish physical or domain-wide convergence.",
        "No JAX or differentiability claim follows from this scalar Torch replay.",
    ]
    report = dict(schema="diffgw.seobnrv4.peak-condition-q8-tighter-ode.v1",
                  status="running", release_qualified=False, scope=scope,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], row=row,
                  levels=dict(baseline=item["runs"]["ode_2"]["level"], additional=expected),
                  runs={}, errors=[], process=dict(host=socket.gethostname(),
                  cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        torch.set_num_threads(1)
        settings = q.Settings(**campaign["settings"])
        runs = []
        for name, level in (("ode_2", item["runs"]["ode_2"]["level"]),
                            ("ode_3", expected)):
            started = time.monotonic()
            run = _run(row, level, settings, campaign["choice"],
                       item["fixed_restart_time_M"])
            summary = q.summarize_run(run)
            if name == "ode_2" and summary["array_sha256"] != (
                    integrated["rows"][plan["row"]]["array_sha256"]):
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
                      additional_gate_passed=all(gate.values()),
                      status="diagnostic_complete")
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc),
                                     traceback=traceback.format_exc()))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          additional_gate_passed=report.get("additional_gate_passed"))),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

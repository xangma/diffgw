"""Replay and refine the native q2 development mode's ODE tolerance."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_q2_ode_refinement_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / path for name, path in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.source-q2-ode-refinement-plan.v1"
            or plan["baseline_level"] != dict(
                name="ode_3", atol=3e-15, rtol=3e-14, dt_geom=0.5)
            or plan["refined_level"] != dict(
                name="ode_4", atol=1e-15, rtol=1e-14, dt_geom=0.5)
            or plan["relative_l2_limit"] != 1e-6
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen q2 ODE plan or inputs differ")
    prior = json.loads(paths["q2_probe"].read_text())
    campaign = json.loads(paths["campaign"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or campaign["status"] != "diagnostic_complete"
            or prior["level"] != plan["baseline_level"]):
        parser.error("pinned q2 probe or campaign changed")
    scope = dict(q.SCOPE)
    scope["description"] = "One-row ODE tolerance refinement of the q2 development mode."
    scope["known_limitations"] = [
        "The q2 row was examined before choosing this tighter level; this is not an untouched release holdout.",
        "ODE tolerance convergence does not establish source parity or domain-wide convergence.",
        "No independent JAX or differentiability validation follows from this Torch run.",
    ]
    report = dict(schema="diffgw.seobnrv4.source-q2-ode-refinement.v1",
                  status="running", release_qualified=False, scope=scope,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], row=prior["row"],
                  levels=dict(baseline=plan["baseline_level"],
                              refined=plan["refined_level"]),
                  runs={}, errors=[], process=dict(host=socket.gethostname(),
                  cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        torch.set_num_threads(1)
        settings = q.Settings(**campaign["settings"])
        runs = []
        for name, level in (("baseline", plan["baseline_level"]),
                            ("refined", plan["refined_level"])):
            started = time.monotonic()
            run = _run(prior["row"], level, settings, campaign["choice"],
                       prior["native"]["restart_time_M"])
            summary = q.summarize_run(run)
            if (name == "baseline"
                    and summary["array_sha256"] != prior["native"]["array_sha256"]):
                raise ValueError("q2 native baseline changed")
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

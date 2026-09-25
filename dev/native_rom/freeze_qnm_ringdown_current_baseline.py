"""Archive historical-QNM modes on two pinned current-source development rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_q8_mode_phase import _replay


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/qnm_ringdown_current_baseline_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.qnm-ringdown-current-baseline-plan.v1"
            or plan["rows"] != ["q4_mixed_spin", "q8_high_aligned"]
            or plan["level"] != "ode_2" or plan["qnm_source"] != "historical_spline"
            or plan["release_qualified"] is not False
            or plan["production_change"] is not False):
        parser.error("baseline plan differs")
    for key in ("campaign", "wheel_qnm_evidence", "supersedes_attempt_plan"):
        if q.sha256(q.ROOT / plan[key]) != plan[key+"_sha256"]:
            parser.error(f"pinned {key} differs")
    for path, expected in plan["source_sha256"].items():
        if q.sha256(q.ROOT / path) != expected:
            parser.error(f"pinned source changed: {path}")
    campaign = json.loads((q.ROOT / plan["campaign"]).read_text())
    wheel = json.loads((q.ROOT / plan["wheel_qnm_evidence"]).read_text())
    if (campaign["status"] != "diagnostic_complete"
            or wheel["status"] != "diagnostic_complete"
            or wheel["campaign_sha256"] != plan["campaign_sha256"]
            or wheel["lal_binary"] != campaign["lal_binary"]):
        parser.error("campaign or wheel witness differs")
    torch.set_num_threads(1)
    report = dict(schema="diffgw.seobnrv4.qnm-ringdown-current-baseline.v1",
                  status="running", scope=q.SCOPE, release_qualified=False,
                  production_change=False, plan_sha256=q.sha256(PLAN),
                  script_sha256=q.sha256(__file__), source_sha256=plan["source_sha256"],
                  campaign_sha256=plan["campaign_sha256"],
                  wheel_qnm_evidence_sha256=plan["wheel_qnm_evidence_sha256"],
                  archive=plan["archive"], rows={}, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    arrays = {}
    try:
        for name in plan["rows"]:
            item = campaign["rows"][name]
            row = item["row"]
            started = time.monotonic()
            run = _replay(row, item["runs"]["ode_2"]["level"],
                          q.Settings(**campaign["settings"]), campaign["choice"],
                          item["fixed_restart_time_M"], return_boundary_data=True,
                          diagnostic_qnm_source=plan["qnm_source"])
            summary = q.summarize_run(run)
            witness = wheel["rows"][name]
            if (run["final_spin"] != witness["final_spin"]
                    or run["final_mass_fraction"] != witness["final_mass_fraction"]
                    or row != witness["row"]
                    or abs(run["boundary_data"]["qnm_frequency"]
                           - witness["public_binary"]["frequency_M"]) > 3e-15
                    or abs(run["boundary_data"]["qnm_damping"]
                           - witness["public_binary"]["damping_M"]) > 3e-15):
                raise ValueError(f"{name}: remnant or historical QNM differs from wheel witness")
            if not np.any(run["time"] < run["boundary_time_M"]):
                raise ValueError(f"{name}: missing preattachment samples")
            if not np.any(run["time"] >= run["boundary_time_M"]+100):
                raise ValueError(f"{name}: missing full 100 M ringdown window")
            arrays[name+"_time"] = run["time"]
            arrays[name+"_mode"] = run["mode"]
            report["rows"][name] = dict(
                row=row, level=item["runs"]["ode_2"]["level"],
                fixed_restart_time_M=item["fixed_restart_time_M"],
                elapsed_seconds=time.monotonic()-started,
                historical_campaign_array_sha256=item["runs"]["ode_2"]["native"]["array_sha256"],
                current_array_sha256=summary["array_sha256"],
                historical_campaign_exact_replay=(summary["array_sha256"] ==
                                                  item["runs"]["ode_2"]["native"]["array_sha256"]),
                samples=len(run["time"]), match_time_M=run["match_time_M"],
                boundary_time_M=run["boundary_time_M"],
                final_spin=run["final_spin"],
                final_mass_fraction=run["final_mass_fraction"],
                internal_step_M=run["internal_step_M"],
                boundary_data=run["boundary_data"])
            q.write_evidence(args.json_out, report)
            print(f"{name} historical baseline archived in memory", flush=True)
        archive = q.ROOT / plan["archive"]
        np.savez_compressed(archive, **arrays)
        report["archive_sha256"] = q.sha256(archive)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

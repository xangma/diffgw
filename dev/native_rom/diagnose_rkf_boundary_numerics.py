"""Screen fine-stage RKF45 tolerance stability on predeclared boundary rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import torch

from . import qualify_generator as q


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    rows_path = q.ROOT / plan["prior_rows_path"]
    old = json.loads(rows_path.read_text())
    levels = plan["levels"]
    if (plan["schema"] != "diffgw.seobnrv4.rkf-boundary-numerics-plan.v1"
            or plan["prior_rows_sha256"] != q.sha256(rows_path)
            or old["schema"] != "diffgw.seobnrv4.parameter-boundary-plan.v1"
            or plan["row_names"] != list(old["rows"])
            or [level["name"] for level in levels] != ["tighter", "tightest"]
            or [(level["atol"], level["rtol"], level["dt_geom"]) for level in levels]
            != [(1e-14, 1e-13, 0.25), (1e-15, 1e-14, 0.25)]
            or plan["sampler"] != ("diagnostic_fine_rkf45_resampling=True; "
                                   "coarse output remains natural cubic")):
        raise ValueError("boundary numerical screen does not match frozen inputs")
    torch.set_num_threads(1)
    source = q.canonical_hash(q.source_snapshot())
    report = dict(schema="diffgw.seobnrv4.rkf-boundary-numerics.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(args.plan), prior_rows_sha256=q.sha256(rows_path),
                  script_sha256=q.sha256(__file__), generator_source_manifest_sha256=source,
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                   text=True).strip(),
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                               argv=sys.argv), rows={})
    q.write_evidence(args.json_out, report)
    for name in plan["row_names"]:
        row = old["rows"][name]
        runs = {}
        entry = dict(row=row, runs={}, comparison=None, checks=None,
                     numerical_check_passed=False, errors=[])
        report["rows"][name] = entry
        for level in levels:
            started = time.monotonic()
            try:
                run = q.native_run(row, level, q.Settings(max_run_seconds=600.),
                                   diagnostic_fine_rkf45_resampling=True)
                runs[level["name"]] = run
                entry["runs"][level["name"]] = dict(
                    native=q.summarize_run(run), elapsed_seconds=time.monotonic()-started)
                print(f"{name} {level['name']} complete", flush=True)
            except Exception as exc:
                entry["errors"].append(dict(level=level["name"], type=type(exc).__name__,
                                            message=str(exc),
                                            elapsed_seconds=time.monotonic()-started))
                print(f"{name} {level['name']} failed: {exc}", flush=True)
                break
            finally:
                q.write_evidence(args.json_out, report)
        if len(runs) == 2:
            try:
                metrics = q.compare_runs(runs["tighter"], runs["tightest"])
                budget = plan["stability_budget"]
                checks = dict(
                    complex_error_budget=(metrics["complex_relative_l2"]
                                          <= budget["complex_relative_l2_max"]),
                    amplitude_error_budget=(metrics["amplitude_relative_l2"]
                                            <= budget["amplitude_relative_l2_max"]),
                    **{event: (metrics[event + "_abs_difference"]
                               <= budget["event_time_difference_M_max"])
                       for event in q.EVENTS})
                entry.update(comparison=metrics, checks=checks,
                             numerical_check_passed=all(checks.values()))
                print(json.dumps(dict(row=name,
                                      complex_relative_l2=metrics["complex_relative_l2"],
                                      numerical_check_passed=entry["numerical_check_passed"])),
                      flush=True)
            except Exception as exc:
                entry["errors"].append(dict(check="compare_runs", type=type(exc).__name__,
                                            message=str(exc)))
                print(f"{name} comparison failed: {exc}", flush=True)
        q.write_evidence(args.json_out, report)
    report["generator_source_manifest_sha256_after"] = q.canonical_hash(q.source_snapshot())
    report["sources_unchanged"] = report["generator_source_manifest_sha256_after"] == source
    report["all_numerical_checks_passed"] = all(
        item["numerical_check_passed"] for item in report["rows"].values())
    report["status"] = "diagnostic_complete" if report["sources_unchanged"] else "source_changed"
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"],
                          all_numerical_checks_passed=report["all_numerical_checks_passed"])),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

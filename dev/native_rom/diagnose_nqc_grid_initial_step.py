"""Isolate the first-step coupling in the frozen aligned-spin NQC grid test.

This opt-in control fixes the coarse ODE first step at the frozen baseline's
0.5 M, keeping the output grid and all candidate rules otherwise unchanged.
It does not alter the predeclared campaign or qualify the generator.
"""

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
from .diagnose_trajectory_sensitivity import _array, _hash
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    campaign = json.loads(args.campaign.read_text())
    if (campaign.get("schema") != "diffgw.seobnrv4.nqc-second-derivative-campaign.v1"
            or not campaign.get("sources_unchanged")
            or campaign["rows"]["aligned_spin"]["checks"]["output_grid"]["gates"]["error_decreased"] is not False):
        parser.error("expected the frozen aligned-spin candidate grid failure")
    sources = q.source_snapshot()
    if any(campaign["source_sha256_before"].get(key) != value
           for key, value in sources.items()):
        parser.error("generator sources differ from frozen campaign")
    torch.set_num_threads(1)
    row = campaign["rows"]["aligned_spin"]["inputs"]
    settings = q.Settings(atol_levels=tuple(campaign["settings"]["atol_levels"]),
                          rtol_levels=tuple(campaign["settings"]["rtol_levels"]))
    stored = {run["name"]: run for run in campaign["rows"]["aligned_spin"]["runs"]}
    report = dict(schema="diffgw.seobnrv4.nqc-grid-initial-step.v1",
                  scope=dict(**q.SCOPE, development_only=True,
                             numerical_generator_qualified=False),
                  status="running", campaign_sha256=q.sha256(args.campaign),
                  script_sha256=q.sha256(__file__), source_sha256_before=sources,
                  process=dict(argv=sys.argv, cwd=os.getcwd(),
                               host=socket.gethostname(), pid=os.getpid()),
                  coarse_initial_step_M=0.5,
                  cases={}, comparisons={},
                  frozen_grid_comparison={
                      "coarse_to_medium": campaign["rows"]["aligned_spin"]["checks"]["output_grid"]["coarse_to_medium"],
                      "medium_to_fine": campaign["rows"]["aligned_spin"]["checks"]["output_grid"]["medium_to_fine"],
                  })
    q.write_evidence(args.json_out, report)
    runs = {}
    for name in ("ode_2", "grid_1", "grid_2"):
        level = stored[name]
        started = time.monotonic()
        with torch.no_grad():
            result = reconstruct_seobnrv4_22_adaptive(
                torch.tensor(row["omega_start"], dtype=torch.float64),
                **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
                **q.CHOICES, **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
                diagnostic_hermite_resampling=True,
                diagnostic_local_nqc_derivatives=True,
                diagnostic_matching_second_derivatives=True,
                diagnostic_restart_time_M=165.,
                diagnostic_coarse_initial_step_M=0.5,
                ringdown_duration=settings.ringdown_duration_M,
                num_steps=settings.max_accepted_steps,
                max_accepted_steps=settings.max_accepted_steps,
                max_evaluations=settings.max_evaluations,
                max_rejected_steps=settings.max_rejected_steps)
        run = dict(name=name, dt_geom=level["dt_geom"],
                   time=_array(result.time), mode=_array(result.mode),
                   peak_time_M=float(_array(result.trajectory.peak.time)),
                   match_time_M=float(_array(result.attachment.match_time)),
                   boundary_time_M=float(_array(result.attachment.boundary_time)))
        hashes = {key: _hash(run[key]) for key in ("time", "mode")}
        if name == "ode_2" and hashes != level["array_sha256"]:
            raise ValueError("fixed-step baseline differs from frozen waveform")
        runs[name] = run
        report["cases"][name] = dict(dt_geom=level["dt_geom"], array_sha256=hashes,
                                     matches_frozen_hash=hashes == level["array_sha256"],
                                     samples=len(run["time"]),
                                     peak_time_M=run["peak_time_M"],
                                     match_time_M=run["match_time_M"],
                                     boundary_time_M=run["boundary_time_M"],
                                     restart_time_M=float(_array(result.trajectory.restart_time).reshape(-1)[0]),
                                     coarse_accepted=result.trajectory.coarse.statistics.total_accepted,
                                     fine_accepted=result.trajectory.fine.statistics.total_accepted,
                                     elapsed_seconds=time.monotonic()-started)
        q.write_evidence(args.json_out, report)
        print(json.dumps(dict(run=name, seconds=report["cases"][name]["elapsed_seconds"],
                              matches_frozen_hash=report["cases"][name]["matches_frozen_hash"])), flush=True)
    report["comparisons"] = dict(
        coarse_to_medium=q.compare_runs(runs["ode_2"], runs["grid_1"]),
        medium_to_fine=q.compare_runs(runs["grid_1"], runs["grid_2"]),
        coarse_to_fine=q.compare_runs(runs["ode_2"], runs["grid_2"]))
    report["source_sha256_after"] = q.source_snapshot()
    report["sources_unchanged"] = sources == report["source_sha256_after"]
    report["status"] = "diagnostic_complete" if report["sources_unchanged"] else "failed"
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"],
                          errors={key: value["complex_relative_l2"]
                                  for key, value in report["comparisons"].items()})), flush=True)
    return 0 if report["sources_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Replay frozen grid-failing rows with only the coarse first step fixed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

from . import qualify_generator as q

import torch

from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/boundary_grid_first_step_plan.json"
CAMPAIGN = q.ROOT / "dev/native_rom/evidence/2026-09-23/parameter_boundary_campaign.json"


def array(value):
    result = value.detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite trajectory or waveform")
    return result


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign = json.loads(CAMPAIGN.read_text())
    if (plan["frozen_campaign_sha256"] != q.sha256(CAMPAIGN)
            or campaign["schema"] != "diffgw.seobnrv4.parameter-boundary-campaign.v1"
            or not campaign["sources_unchanged"]
            or plan["levels"] != ["ode_2", "grid_1", "grid_2"]
            or plan["coarse_initial_step_M"] != 0.5
            or plan["unchanged_choices"] != dict(
                hermite_resampling=True, local_nqc_phase_derivatives=True,
                local_matching_second_derivatives=True,
                fixed_restart_time_M=165.0,
                ode_tolerances=[1e-14, 1e-13],
                output_grid_M=[0.5, 0.25, 0.125])):
        parser.error("plan or frozen campaign differs from the predeclaration")
    settings = q.Settings(atol_levels=tuple(campaign["settings"]["atol_levels"]),
                          rtol_levels=tuple(campaign["settings"]["rtol_levels"]))
    sources = q.source_snapshot()
    if any(campaign["source_sha256_before"].get(key) != value
           for key, value in sources.items()):
        parser.error("generator sources differ from the frozen campaign")
    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.boundary-grid-first-step.v1",
        scope=dict(**q.SCOPE, development_only=True,
                   numerical_generator_qualified=False,
                   same_provisional_equations=True),
        status="running", plan_sha256=q.sha256(PLAN),
        campaign_sha256=q.sha256(CAMPAIGN), script_sha256=q.sha256(__file__),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256_before=sources,
        process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid()),
        coarse_initial_step_M=0.5, rows={})
    q.write_evidence(args.json_out, report)
    for name in plan["rows"]:
        frozen = campaign["rows"][name]
        frozen_grid = frozen["checks"]["output_grid"]
        if frozen_grid["gates"]["error_decreased"] is not False:
            raise ValueError(f"{name} lacks a frozen grid monotonicity failure")
        row = frozen["inputs"]
        stored = {run["name"]: run for run in frozen["runs"]}
        outcome = report["rows"][name] = dict(
            inputs=row, frozen_grid_comparison={
                "coarse_to_medium": frozen_grid["coarse_to_medium"],
                "medium_to_fine": frozen_grid["medium_to_fine"]},
            cases={}, comparisons={}, errors=[])
        runs = {}
        for level_name in plan["levels"]:
            level = stored[level_name]
            started = time.monotonic()
            try:
                with torch.no_grad():
                    result = reconstruct_seobnrv4_22_adaptive(
                        torch.tensor(row["omega_start"], dtype=torch.float64),
                        **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
                        **q.CHOICES,
                        **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
                        diagnostic_hermite_resampling=True,
                        diagnostic_local_nqc_derivatives=True,
                        diagnostic_matching_second_derivatives=True,
                        diagnostic_restart_time_M=165.0,
                        diagnostic_coarse_initial_step_M=0.5,
                        ringdown_duration=settings.ringdown_duration_M,
                        num_steps=settings.max_accepted_steps,
                        max_accepted_steps=settings.max_accepted_steps,
                        max_evaluations=settings.max_evaluations,
                        max_rejected_steps=settings.max_rejected_steps)
                run = dict(name=level_name, dt_geom=level["dt_geom"],
                           time=array(result.time), mode=array(result.mode),
                           peak_time_M=float(array(result.trajectory.peak.time)),
                           match_time_M=float(array(result.attachment.match_time)),
                           boundary_time_M=float(array(result.attachment.boundary_time)))
                q.compare_runs(run, run)
                hashes = {key: array_hash(run[key]) for key in ("time", "mode")}
                if level_name == "ode_2" and hashes != level["array_sha256"]:
                    raise ValueError("fixed-step baseline differs from frozen waveform")
                stages = {}
                for stage_name in ("coarse", "fine"):
                    stage = getattr(result.trajectory, stage_name)
                    stages[stage_name] = dict(
                        accepted=stage.statistics.total_accepted,
                        rejected=stage.statistics.total_rejected,
                        evaluations=stage.statistics.total_evaluations,
                        hashes={field: array_hash(array(getattr(stage, field)))
                                for field in ("time", "state", "derivative")})
                runs[level_name] = run
                outcome["cases"][level_name] = dict(
                    dt_geom=level["dt_geom"], array_sha256=hashes,
                    matches_frozen_hash=hashes == level["array_sha256"],
                    samples=len(run["time"]), stages=stages,
                    peak_time_M=run["peak_time_M"],
                    match_time_M=run["match_time_M"],
                    boundary_time_M=run["boundary_time_M"],
                    restart_time_M=float(array(result.trajectory.restart_time).reshape(-1)[0]),
                    elapsed_seconds=time.monotonic()-started)
                q.write_evidence(args.json_out, report)
                print(json.dumps(dict(row=name, run=level_name,
                                      seconds=outcome["cases"][level_name]["elapsed_seconds"],
                                      matches_frozen_hash=outcome["cases"][level_name]["matches_frozen_hash"])),
                      flush=True)
            except Exception as exc:
                outcome["errors"].append(dict(run=level_name, type=type(exc).__name__,
                                              message=str(exc), elapsed_seconds=time.monotonic()-started))
                break
        if len(runs) == 3:
            outcome["comparisons"] = dict(
                coarse_to_medium=q.compare_runs(runs["ode_2"], runs["grid_1"]),
                medium_to_fine=q.compare_runs(runs["grid_1"], runs["grid_2"]),
                coarse_to_fine=q.compare_runs(runs["ode_2"], runs["grid_2"]))
            outcome["accepted_histories_identical"] = all(
                outcome["cases"][level]["stages"][stage]["hashes"]
                == outcome["cases"]["ode_2"]["stages"][stage]["hashes"]
                for level in ("grid_1", "grid_2") for stage in ("coarse", "fine"))
            outcome["nested_waveforms_identical"] = all(
                np.array_equal(runs[a]["mode"], runs[b]["mode"][::round(
                    runs[a]["dt_geom"]/runs[b]["dt_geom"])])
                for a, b in (("ode_2", "grid_1"), ("grid_1", "grid_2")))
        q.write_evidence(args.json_out, report)
        print(json.dumps(dict(row=name, complete=len(runs)==3,
                              accepted_histories_identical=outcome.get("accepted_histories_identical"),
                              nested_waveforms_identical=outcome.get("nested_waveforms_identical"))),
              flush=True)
    report["source_sha256_after"] = q.source_snapshot()
    report["sources_unchanged"] = sources == report["source_sha256_after"]
    report["status"] = ("diagnostic_complete" if report["sources_unchanged"]
                        and all(not row["errors"] and len(row["cases"]) == 3
                                for row in report["rows"].values()) else "failed")
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], evidence=str(args.json_out.resolve()))),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

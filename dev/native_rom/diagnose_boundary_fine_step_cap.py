"""Predeclared accepted-step cap control on frozen boundary failures; dev only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_nqc_inputs import _system
from .diagnose_boundary_strong_spin_ode import array, array_hash, stages_of
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from torchwave.approximants.eob.nqc import (
    seobnrv4_nqc_parameters, seobnrv4_nqc_targets,
)


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/boundary_fine_step_cap_plan.json"
CAMPAIGN = q.ROOT / "dev/native_rom/evidence/2026-09-23/parameter_boundary_campaign.json"
GENERATOR = "dev/native_rom/eob/torch/adaptive_driver_clean.py"


def _run(row, level, settings, cap):
    started = time.monotonic()
    with q.bounded_run(settings.max_run_seconds), torch.no_grad():
        result = reconstruct_seobnrv4_22_adaptive(
            torch.tensor(row["omega_start"], dtype=torch.float64),
            **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
            **q.CHOICES,
            **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
            diagnostic_hermite_resampling=True,
            diagnostic_local_nqc_derivatives=True,
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=165.0,
            diagnostic_fine_max_step_M=cap,
            ringdown_duration=settings.ringdown_duration_M,
            num_steps=settings.max_accepted_steps,
            max_accepted_steps=settings.max_accepted_steps,
            max_evaluations=settings.max_evaluations,
            max_rejected_steps=settings.max_rejected_steps)
    parameters = [torch.tensor(row[key], dtype=torch.float64)
                  for key in ("mass1", "mass2", "chi1", "chi2")]
    targets = seobnrv4_nqc_targets(*seobnrv4_nqc_parameters(*parameters))
    system = _system(result, targets)
    trajectory = result.trajectory
    fine = trajectory.fine
    accepted = array(fine.time)[array(fine.valid).astype(bool)]
    steps = np.diff(accepted)
    if len(steps) < 1 or (cap is not None and max(steps) > cap + 1e-12):
        raise ValueError("fine accepted steps do not obey diagnostic cap")
    stages = stages_of(trajectory)
    run = dict(**level, time=array(result.time), mode=array(result.mode),
               peak_time_M=float(array(trajectory.peak.time)),
               match_time_M=float(array(result.attachment.match_time)),
               boundary_time_M=float(array(result.attachment.boundary_time)))
    q.compare_runs(run, run)
    details = dict(summary=q.summarize_run(run), stages=stages,
                   fine_step=dict(max_M=float(max(steps)), min_M=float(min(steps)),
                                  median_M=float(np.median(steps)), count=len(steps)),
                   nqc_phase=dict(matrix=system["phase"]["matrix"].tolist(),
                                  rhs=system["phase"]["rhs"].tolist(),
                                  coefficients=system["phase"]["coefficients"].tolist(),
                                  condition=system["phase"]["condition"],
                                  local_fit_samples=system["local_fit_samples"]),
                   elapsed_seconds=time.monotonic()-started)
    return run, details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    campaign = json.loads(CAMPAIGN.read_text())
    expected_rows = ["equal_mass_unequal_spin", "mass_ratio_four_nonspinning",
                     "strong_aligned_spin", "strong_antialigned_spin"]
    if (plan["schema"] != "diffgw.seobnrv4.boundary-fine-step-cap-plan.v1"
            or plan["frozen_campaign_sha256"] != q.sha256(CAMPAIGN)
            or plan["rows"] != expected_rows
            or plan["levels"] != ["ode_0", "ode_1", "ode_2"]
            or plan["fine_accepted_max_step_M"] != 0.125
            or plan["baseline_replay"] != dict(
                row="strong_aligned_spin", level="ode_1",
                require_exact_frozen_waveform_and_stage_statistics=True)
            or plan["unchanged_choices"] != dict(
                hermite_resampling=True, local_nqc_phase_derivatives=True,
                local_matching_second_derivatives=True, local_second_window_M=0.5,
                local_second_degree=6, fixed_restart_time_M=165.0,
                output_grid_M=0.5, coarse_adaptive_driver="unchanged",
                fine_vector_field="unchanged", nqc_matching="unchanged")
            or campaign["schema"] != "diffgw.seobnrv4.parameter-boundary-campaign.v1"
            or not campaign["sources_unchanged"]):
        parser.error("predeclared plan or frozen campaign mismatch")
    sources = q.source_snapshot()
    changed = {key for key in sources.keys() | campaign["source_sha256_before"].keys()
               if sources.get(key) != campaign["source_sha256_before"].get(key)}
    if changed != {GENERATOR}:
        parser.error(f"unexpected generator source changes since frozen campaign: {sorted(changed)}")
    settings = q.Settings(atol_levels=tuple(campaign["settings"]["atol_levels"]),
                          rtol_levels=tuple(campaign["settings"]["rtol_levels"]),
                          max_run_seconds=campaign["settings"]["max_run_seconds"],
                          relative_l2_limit=campaign["settings"]["relative_l2_limit"],
                          event_time_limit_M=campaign["settings"]["event_time_limit_M"])
    report = dict(schema="diffgw.seobnrv4.boundary-fine-step-cap.v1",
                  scope=dict(**q.SCOPE, development_only=True,
                             same_provisional_equations=True,
                             numerical_generator_qualified=False),
                  status="running", plan_sha256=q.sha256(PLAN),
                  campaign_sha256=q.sha256(CAMPAIGN), script_sha256=q.sha256(__file__),
                  source_sha256_before=sources,
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid()),
                  cap_M=plan["fine_accepted_max_step_M"], baseline_replay={}, rows={})
    q.write_evidence(args.json_out, report)
    baseline_name = plan["baseline_replay"]["row"]
    baseline_level = plan["baseline_replay"]["level"]
    baseline_row = campaign["rows"][baseline_name]
    frozen_baseline = next(run for run in baseline_row["runs"] if run["name"] == baseline_level)
    baseline_run, baseline_details = _run(baseline_row["inputs"], frozen_baseline,
                                          settings, None)
    baseline_stages = {name: {key: val for key, val in stage.items()
                              if key != "array_sha256"}
                       for name, stage in baseline_details["stages"].items()}
    baseline_exact = (baseline_details["summary"]["array_sha256"] ==
                      frozen_baseline["array_sha256"] and
                      baseline_stages == frozen_baseline["stages"])
    report["baseline_replay"] = dict(row=baseline_name, level=baseline_level,
                                     exact_frozen_waveform_and_stage_statistics=baseline_exact,
                                     details=baseline_details)
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(baseline=baseline_name, level=baseline_level,
                          exact=baseline_exact,
                          seconds=baseline_details["elapsed_seconds"])), flush=True)
    if not baseline_exact:
        raise ValueError("uncapped baseline did not reproduce frozen campaign")
    del baseline_run
    for name in plan["rows"]:
        frozen_row = campaign["rows"][name]
        if frozen_row["numerical_checks_passed"]:
            raise ValueError(f"{name} is not a failing frozen row")
        prior = {run["name"]: run for run in frozen_row["runs"]}
        outcome = report["rows"][name] = dict(
            inputs=frozen_row["inputs"], cases={}, errors=[],
            frozen_ode_check=frozen_row["checks"]["ode_tolerance"])
        runs = {}
        for level_name in plan["levels"]:
            started = time.monotonic()
            try:
                runs[level_name], outcome["cases"][level_name] = _run(
                    frozen_row["inputs"], prior[level_name], settings, report["cap_M"])
                q.write_evidence(args.json_out, report)
                print(json.dumps(dict(row=name, level=level_name,
                                      seconds=outcome["cases"][level_name]["elapsed_seconds"],
                                      fine_max_step_M=outcome["cases"][level_name]["fine_step"]["max_M"])),
                      flush=True)
            except Exception as exc:
                outcome["errors"].append(dict(level=level_name, type=type(exc).__name__,
                                              message=str(exc),
                                              elapsed_seconds=time.monotonic()-started))
                q.write_evidence(args.json_out, report)
                print(json.dumps(dict(row=name, level=level_name, error=str(exc))), flush=True)
                break
        if len(runs) == 3:
            controlled = q.convergence_check(*(runs[level] for level in plan["levels"]), settings)
            outcome["controlled_ode_check"] = controlled
            frozen_check = outcome["frozen_ode_check"]
            outcome["comparison"] = dict(
                medium_to_fine_complex_error_ratio=(
                    controlled["medium_to_fine"]["complex_relative_l2"] /
                    frozen_check["medium_to_fine"]["complex_relative_l2"]),
                frozen_gates=frozen_check["gates"], controlled_gates=controlled["gates"],
                phase_basis_second_delta=[
                    outcome["cases"]["ode_2"]["nqc_phase"]["matrix"][1][column] -
                    outcome["cases"]["ode_1"]["nqc_phase"]["matrix"][1][column]
                    for column in range(2)],
                phase_coefficient_delta=[
                    outcome["cases"]["ode_2"]["nqc_phase"]["coefficients"][column] -
                    outcome["cases"]["ode_1"]["nqc_phase"]["coefficients"][column]
                    for column in range(2)])
        q.write_evidence(args.json_out, report)
        print(json.dumps(dict(row=name, completed=len(runs) == 3,
                              controlled_gates=outcome.get("comparison", {}).get("controlled_gates"),
                              error_ratio=outcome.get("comparison", {}).get(
                                  "medium_to_fine_complex_error_ratio"))), flush=True)
    report["source_sha256_after"] = q.source_snapshot()
    report["sources_unchanged"] = sources == report["source_sha256_after"]
    report["status"] = ("diagnostic_complete" if report["sources_unchanged"] and
                        all(not row["errors"] and len(row["cases"]) == 3
                            for row in report["rows"].values()) else "failed")
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

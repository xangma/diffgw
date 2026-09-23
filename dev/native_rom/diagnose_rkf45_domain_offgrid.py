"""Frozen six-row off-grid sampler comparison for the provisional v4 generator."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
import torch
import torchwave

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_basis_trace import _physics
from .diagnose_boundary_strong_spin_ode import array, array_hash, stages_of
from .diagnose_rkf45_offgrid_reference import _errors, _locally_seeded_reference
from .eob.torch.adaptive_driver_clean import (
    _diagnostic_hermite_output, _diagnostic_rkf45_output,
    reconstruct_seobnrv4_22_adaptive,
)


EVIDENCE = q.ROOT / "dev/native_rom/evidence/2026-09-23"
PLAN = EVIDENCE / "rkf45_domain_offgrid_plan.json"


def _run(name, row, prior, plan):
    started = time.monotonic()
    with torch.no_grad():
        result = reconstruct_seobnrv4_22_adaptive(
            torch.tensor(row["omega_start"], dtype=torch.float64),
            **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
            **q.CHOICES,
            **{key: prior[key] for key in ("atol", "rtol", "dt_geom")},
            diagnostic_rkf45_resampling=True,
            diagnostic_local_nqc_derivatives=True,
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=165.0,
            diagnostic_fine_max_step_M=0.125,
            diagnostic_coarse_initial_step_M=0.5,
            ringdown_duration=150.0,
            num_steps=20000, max_accepted_steps=20000,
            max_evaluations=200000, max_rejected_steps=10000)
    hashes = dict(mode=array_hash(array(result.mode)), time=array_hash(array(result.time)))
    if hashes != prior["array_sha256"]:
        raise ValueError(f"{name} waveform differs from frozen boundary campaign")
    stages = stages_of(result.trajectory)
    for stage in ("coarse", "fine"):
        if any(stages[stage][key] != prior["stages"][stage][key]
               for key in ("accepted", "rejected", "evaluations", "termination", "event_time_M")):
            raise ValueError(f"{name} {stage} trajectory differs from frozen campaign")

    fine = result.trajectory.fine
    valid = array(fine.valid).astype(bool)
    accepted_times = array(fine.time)[valid]
    accepted_states = array(fine.state)[valid]
    intervals = len(accepted_times)-1
    match_local = float(result.attachment.match_time)-float(result.trajectory.restart_time)
    indices = [min(intervals-1, int(fraction*intervals))
               for fraction in plan["accepted_bracket_index_fraction"]]
    if plan["include_match_bracket"]:
        match_index = int(np.searchsorted(accepted_times, match_local, side="right")-1)
        indices.append(match_index)
    if min(indices) < 0 or max(indices) >= intervals or len(set(indices)) != len(indices):
        raise ValueError(f"{name} off-grid accepted brackets unavailable or duplicate")
    query = np.array([accepted_times[index]+fraction*(accepted_times[index+1]-accepted_times[index])
                      for index in indices for fraction in plan["within_bracket_fraction"]])
    field, _ = _physics(row)
    probe_run = replace(fine, output_time=torch.from_numpy(query))
    with torch.no_grad():
        direct = array(_diagnostic_rkf45_output(probe_run, field).output_state)
        hermite = array(_diagnostic_hermite_output(probe_run).output_state)
    references = []
    evaluations = []
    for rtol, atol in plan["dense_reference"]["rtol_atol"]:
        states, nfev = _locally_seeded_reference(
            field, accepted_times, accepted_states, query, rtol, atol)
        references.append(states)
        evaluations.append(nfev)
    tight = references[-1]
    return dict(name=name, elapsed_seconds=time.monotonic()-started,
                array_sha256=hashes, stages=stages, accepted_intervals=intervals,
                bracket_indices=indices, bracket_times_M=accepted_times[indices].tolist(),
                match_time_local_M=match_local, probe_time_local_M=query.tolist(),
                dense_evaluations=evaluations,
                state_error=dict(rkf45=_errors(direct, tight),
                                 hermite=_errors(hermite, tight),
                                 dense_reference_spread=_errors(references[0], tight)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    module_path = Path(torchwave.__file__).resolve()
    if not module_path.is_relative_to(q.ROOT / "src"):
        parser.error(f"torchwave loaded outside this checkout: {module_path}")
    plan = json.loads(PLAN.read_text())
    campaign_path = EVIDENCE / plan["combined_evidence"]
    campaign = json.loads(campaign_path.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.rkf45-domain-offgrid-plan.v1"
            or plan["level"] != "ode_2"
            or set(plan["rows"]) != set(campaign["rows"])
            or campaign["status"] != "diagnostic_numerical_gates_passed"
            or not campaign["all_numerical_checks_passed"]):
        parser.error("frozen plan or campaign mismatch")
    source = q.source_snapshot()
    prior_source = campaign["source_sha256_verified"]
    drift = sorted(key for key, value in prior_source.items() if source.get(key) != value)
    if drift != sorted(plan["permitted_prior_source_drift"]):
        parser.error(f"source drift differs from predeclared list: {drift}")
    report = dict(schema="diffgw.seobnrv4.rkf45-domain-offgrid.v1",
                  scope=dict(**q.SCOPE, development_only=True,
                             numerical_generator_qualified=False, independent_physics=False),
                  status="running", plan_sha256=q.sha256(PLAN),
                  campaign_sha256=q.sha256(campaign_path),
                  source_sha256_before=source, prior_source_drift=drift,
                  script_sha256=q.sha256(__file__),
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(),
                               pid=os.getpid(), torchwave_module=str(module_path)), rows={})
    q.write_evidence(args.json_out, report)
    try:
        for name in plan["rows"]:
            row = campaign["rows"][name]
            prior = next(run for run in row["runs"] if run["name"] == plan["level"])
            report["rows"][name] = _run(name, row["inputs"], prior, plan)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(row=name,
                                  elapsed_seconds=report["rows"][name]["elapsed_seconds"],
                                  max_abs_state={key: value["max_abs_any"] for key, value in
                                                 report["rows"][name]["state_error"].items()})), flush=True)
        report["source_sha256_after"] = q.source_snapshot()
        report["sources_unchanged"] = report["source_sha256_after"] == source
        if not report["sources_unchanged"]:
            raise ValueError("generator sources changed during off-grid reference")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        q.write_evidence(args.json_out, report)
        raise
    q.write_evidence(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compare frozen fine-stage accepted states with whole-trajectory DOP853; dev only."""

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
import torchwave

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_basis_trace import _physics, _solve
from .diagnose_boundary_strong_spin_ode import array, array_hash, stages_of
from .diagnose_rkf45_offgrid_reference import _errors
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive


EVIDENCE = q.ROOT / "dev/native_rom/evidence/2026-09-23"
PLAN = EVIDENCE / "rkf45_full_trajectory_plan.json"


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
    native_times = array(fine.time)[valid]
    native_states = array(fine.state)[valid]
    if len(native_times) < 2 or native_times[0] != 0 or not np.all(np.diff(native_times) > 0):
        raise ValueError(f"{name} fine accepted history is invalid")
    field, _ = _physics(row)
    references = []
    metadata = []
    for rtol, atol in plan["reference"]["rtol_atol"]:
        solution, meta = _solve(field, native_states[0], float(native_times[0]),
                                float(native_times[-1]), rtol, atol,
                                plan["reference"]["first_step_M"])
        references.append(solution.sol(native_times).T)
        metadata.append(meta)
    tight = references[-1]
    native_error = _errors(native_states, tight)
    spread = _errors(references[0], tight)
    ceiling = plan["diagnostic_ceiling"]
    checks = dict(native_max_abs_any=native_error["max_abs_any"] <= ceiling["native_max_abs_any"],
                  reference_spread_max_abs_any=(spread["max_abs_any"]
                                                <= ceiling["reference_spread_max_abs_any"]))
    return dict(name=name, elapsed_seconds=time.monotonic()-started,
                array_sha256=hashes, stages=stages,
                accepted_knots=len(native_times), start_M=float(native_times[0]),
                endpoint_M=float(native_times[-1]),
                initial_state_sha256=array_hash(native_states[0]),
                reference_runs=metadata,
                state_error=dict(native=native_error, reference_spread=spread),
                endpoint_error=(native_states[-1]-tight[-1]).tolist(),
                checks=checks, diagnostic_checks_passed=all(checks.values()))


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
    if (plan["schema"] != "diffgw.seobnrv4.rkf45-full-trajectory-plan.v1"
            or plan["stage"] != "fine" or plan["level"] != "ode_2"
            or set(plan["rows"]) != set(campaign["rows"])
            or campaign["status"] != "diagnostic_numerical_gates_passed"
            or not campaign["all_numerical_checks_passed"]):
        parser.error("frozen plan or campaign mismatch")
    source = q.source_snapshot()
    prior_source = campaign["source_sha256_verified"]
    drift = sorted(key for key, value in prior_source.items() if source.get(key) != value)
    added = sorted(key for key in source if key not in prior_source)
    if (drift != sorted(plan["permitted_prior_source_drift"])
            or added != sorted(plan["permitted_added_sources"])):
        parser.error(f"source drift differs from predeclared list: {drift}, {added}")
    report = dict(schema="diffgw.seobnrv4.rkf45-full-trajectory.v1",
                  scope=dict(**q.SCOPE, development_only=True,
                             numerical_generator_qualified=False, independent_physics=False),
                  status="running", plan_sha256=q.sha256(PLAN),
                  campaign_sha256=q.sha256(campaign_path),
                  source_sha256_before=source, prior_source_drift=drift,
                  prior_source_additions=added, script_sha256=q.sha256(__file__),
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
            entry = report["rows"][name]
            print(json.dumps(dict(row=name, elapsed_seconds=entry["elapsed_seconds"],
                                  native_max_abs_any=entry["state_error"]["native"]["max_abs_any"],
                                  reference_spread_max_abs_any=(entry["state_error"][
                                      "reference_spread"]["max_abs_any"]),
                                  checks=entry["checks"])), flush=True)
        report["source_sha256_after"] = q.source_snapshot()
        report["sources_unchanged"] = report["source_sha256_after"] == source
        if not report["sources_unchanged"]:
            raise ValueError("generator sources changed during trajectory reference")
        report["all_diagnostic_checks_passed"] = all(
            item["diagnostic_checks_passed"] for item in report["rows"].values())
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

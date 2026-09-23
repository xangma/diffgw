"""Trace the capped mass-ratio-four NQC phase-basis derivative; dev only."""

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
from .diagnose_boundary_strong_spin_basis_trace import (
    _continuous, _hermite, _local_fit, _partition, _physics, _sampled, _solve,
)
from .diagnose_boundary_strong_spin_nqc_inputs import _system
from .diagnose_boundary_strong_spin_ode import array, array_hash, stages_of
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from torchwave.approximants.eob.nqc import (
    seobnrv4_nqc_parameters, seobnrv4_nqc_targets,
)


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/boundary_q4_capped_basis_trace_plan.json"
ATTRIBUTION = q.ROOT / "dev/native_rom/evidence/2026-09-23/boundary_q4_capped_attribution.json"
CONTROL = q.ROOT / "dev/native_rom/evidence/2026-09-23/boundary_fine_step_cap.json"
CAMPAIGN_SCRIPT = "dev/native_rom/diagnose_parameter_boundaries.py"


def _replay(name, row, prior, targets, plan):
    started = time.monotonic()
    with torch.no_grad():
        result = reconstruct_seobnrv4_22_adaptive(
            torch.tensor(row["omega_start"], dtype=torch.float64),
            **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
            **q.CHOICES,
            **{key: prior[key] for key in ("atol", "rtol", "dt_geom")},
            diagnostic_hermite_resampling=True,
            diagnostic_local_nqc_derivatives=True,
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=165.0,
            diagnostic_fine_max_step_M=plan["fine_accepted_max_step_M"],
            ringdown_duration=150.0,
            num_steps=20000, max_accepted_steps=20000,
            max_evaluations=200000, max_rejected_steps=10000)
    hashes = dict(mode=array_hash(array(result.mode)),
                  time=array_hash(array(result.time)))
    if hashes != prior["array_sha256"]:
        raise ValueError(f"{name} waveform differs from capped attribution")
    stages = stages_of(result.trajectory)
    if stages != prior["stages"]:
        raise ValueError(f"{name} accepted trajectory differs from capped attribution")
    system = _system(result, targets)
    native_second = system["phase"]["matrix"][1]
    match = float(result.attachment.match_time)
    restart = float(result.trajectory.restart_time)
    if restart != 165.0:
        raise ValueError(f"{name} restart time differs from frozen setting")
    fine = result.trajectory.fine
    local_times = array(fine.output_time[:int(fine.output_valid.sum())])
    local_match = system["match_time_local_M"]
    window = plan["dense_reference"]["local_fit_window_M"]
    selected = abs(local_times-local_match) <= window
    chart = (local_times[selected]-local_match)/window
    native_states = array(fine.output_state[:len(local_times)])[selected]
    native_basis = array(result.attachment.fine.nqc.sampled_phase_basis)[selected]
    if len(chart) != system["local_fit_samples"]:
        raise ValueError("local sample count differs from matching system")
    recovered = _local_fit(native_basis, chart, window)
    if np.max(abs(recovered-native_second)) > 1e-12:
        raise ValueError("local phase fit does not recover NQC matrix")
    restart_state = array(fine.state[0])
    if not np.array_equal(restart_state, array(fine.output_state[0])):
        raise ValueError("accepted and output restart states differ")
    valid = array(fine.valid).astype(bool)
    accepted_times = array(fine.time)[valid]
    accepted_states = array(fine.state)[valid]
    accepted_derivatives = array(fine.derivative)[valid]
    left = np.searchsorted(accepted_times, local_times[selected], side="right")-1
    left = np.clip(left, 0, len(accepted_times)-2)
    lo, hi = int(left.min()), int(left.max())+2
    accepted_times = accepted_times[lo:hi]
    accepted_states = accepted_states[lo:hi]
    accepted_derivatives = accepted_derivatives[lo:hi]
    reconstructed = _hermite(accepted_times, accepted_states,
                              accepted_derivatives, local_times[selected])
    reconstruction_error = float(np.max(abs(reconstructed-native_states)))
    if reconstruction_error > 1e-12:
        raise ValueError("accepted-state Hermite map does not recover native output")
    serial = dict(name=name, array_sha256=hashes, stages=stages,
                  match_time_M=match, restart_time_M=restart,
                  restart_state=restart_state.tolist(),
                  native_second=native_second.tolist(),
                  local_fit_samples=len(chart), accepted_knots=len(accepted_times),
                  native_hermite_reconstruction_error=reconstruction_error,
                  elapsed_seconds=time.monotonic()-started)
    local = dict(restart_state=restart_state,
                 global_times=restart+local_times[selected], chart=chart,
                 native_states=native_states, native_basis=native_basis,
                 accepted_times=restart+accepted_times,
                 accepted_states=accepted_states,
                 accepted_derivatives=accepted_derivatives)
    return serial, local


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    attribution = json.loads(ATTRIBUTION.read_text())
    control = json.loads(CONTROL.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.boundary-q4-capped-basis-trace-plan.v1"
            or plan["attribution_sha256"] != q.sha256(ATTRIBUTION)
            or plan["row"] != "mass_ratio_four_nonspinning"
            or plan["pair"] != ["ode_1", "ode_2"]
            or plan["fine_accepted_max_step_M"] != 0.125
            or attribution["schema"] != "diffgw.seobnrv4.boundary-q4-capped-attribution.v1"
            or attribution["status"] != "diagnostic_complete"
            or attribution["control_sha256"] != q.sha256(CONTROL)
            or not attribution["sources_unchanged"]):
        parser.error("predeclared plan or attribution mismatch")
    sources = q.source_snapshot()
    sources[CAMPAIGN_SCRIPT] = q.sha256(q.ROOT / CAMPAIGN_SCRIPT)
    if sources != attribution["source_sha256_before"]:
        parser.error("generator sources differ from capped attribution")
    row = attribution["row"]
    parameters = [torch.tensor(row[key], dtype=torch.float64)
                  for key in ("mass1", "mass2", "chi1", "chi2")]
    targets = seobnrv4_nqc_targets(*seobnrv4_nqc_parameters(*parameters))
    report = dict(
        schema="diffgw.seobnrv4.boundary-q4-capped-basis-trace.v1",
        scope=dict(**q.SCOPE, development_only=True,
                   same_provisional_equations=True,
                   numerical_generator_qualified=False),
        status="running", plan_sha256=q.sha256(PLAN),
        attribution_sha256=q.sha256(ATTRIBUTION), script_sha256=q.sha256(__file__),
        source_sha256_before=sources,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid()),
        row=row, pair=plan["pair"], runs=[], dense={}, comparison={})
    q.write_evidence(args.json_out, report)
    try:
        locals_ = []
        for name in plan["pair"]:
            prior = attribution["levels"][name]
            frozen = control["rows"][plan["row"]]["cases"][name]
            level = dict(**{key: frozen["summary"][key]
                            for key in ("atol", "rtol", "dt_geom")},
                         array_sha256=prior["waveform_sha256"], stages=prior["stages"])
            serial, local = _replay(name, row, level, targets, plan)
            if np.max(abs(np.asarray(serial["native_second"])
                          - np.asarray(prior["phase_system"]["matrix"][1]))) > 1e-12:
                raise ValueError(f"{name} phase derivative differs from prior attribution")
            report["runs"].append(serial)
            locals_.append(local)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(run=name, seconds=serial["elapsed_seconds"],
                                  frozen_hash_verified=True)), flush=True)
        if report["runs"][0]["match_time_M"] != report["runs"][1]["match_time_M"]:
            raise ValueError("pair has distinct match times")
        field, initial = _physics(row)
        match = report["runs"][0]["match_time_M"]
        restart = report["runs"][0]["restart_time_M"]
        config = plan["dense_reference"]
        steps = config["five_point_steps_M"]
        reference_key = f"{min(steps):g}"
        dense_end = max(match+2*max(steps),
                        *(max(local["global_times"][-1], local["accepted_times"][-1])
                          for local in locals_)) + 1e-9
        report["dense_end_M"] = dense_end
        dense_levels = []
        for rtol, atol in config["rtol_atol"]:
            global_solution, global_meta = _solve(
                field, initial, 0., dense_end, rtol, atol,
                config["global_first_step_M"])
            global_meta.update(stencils=_continuous(field, global_solution, match, steps),
                               restart_state=global_solution.sol(restart).tolist())
            dense = dict(global_reference=global_meta, restarted={})
            for run, local in zip(report["runs"], locals_):
                solution, meta = _solve(
                    field, local["restart_state"], restart, dense_end,
                    rtol, atol, config["restart_first_step_M"])
                meta.update(stencils=_continuous(field, solution, match, steps),
                            sampled=_sampled(
                                field, solution, local["global_times"], local["chart"],
                                config["local_fit_window_M"], local["native_states"],
                                local["native_basis"], local["accepted_times"],
                                local["accepted_states"], local["accepted_derivatives"]))
                dense["restarted"][run["name"]] = meta
                print(json.dumps(dict(dop_rtol=rtol, run=run["name"],
                                      seconds=meta["elapsed_seconds"],
                                      evaluations=meta["evaluations"])), flush=True)
            report["dense"][f"{rtol:g}"] = dense
            dense_levels.append(dense)
            q.write_evidence(args.json_out, report)
        tight = dense_levels[-1]
        reference = tight["global_reference"]["stencils"][reference_key]["second"]
        partitions = {}
        for run in report["runs"]:
            dense = tight["restarted"][run["name"]]
            partitions[run["name"]] = _partition(
                run["native_second"], dense["sampled"]["accepted_hermite_second"],
                dense["sampled"]["direct_second"],
                dense["stencils"][reference_key]["second"], reference)
        first, second = plan["pair"]
        native_delta = np.asarray(report["runs"][1]["native_second"])-np.asarray(
            report["runs"][0]["native_second"])
        categories = ("fine_accepted_state_evolution", "hermite_output_interpolation",
                      "local_derivative_fit", "restart_state")
        category_delta = {key: (np.asarray(partitions[second][key])
                                - np.asarray(partitions[first][key])).tolist()
                          for key in categories}
        closure = sum(np.asarray(value) for value in category_delta.values())-native_delta
        if np.max(abs(closure)) > 1e-15:
            raise ValueError("between-run error partition does not close")
        loose = dense_levels[0]["global_reference"]["stencils"][reference_key]["second"]
        report["comparison"] = dict(
            partition=partitions, native_delta=native_delta.tolist(),
            category_delta=category_delta,
            between_run_closure_max_abs=float(np.max(abs(closure))),
            restart_state_delta=(locals_[1]["restart_state"]
                                 -locals_[0]["restart_state"]).tolist(),
            tight_vs_loose_reference_spread=(np.asarray(reference)
                                             -np.asarray(loose)).tolist())
        report["source_sha256_after"] = q.source_snapshot()
        report["source_sha256_after"][CAMPAIGN_SCRIPT] = q.sha256(q.ROOT / CAMPAIGN_SCRIPT)
        report["sources_unchanged"] = report["source_sha256_after"] == sources
        if not report["sources_unchanged"]:
            raise ValueError("generator sources changed during trace")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        q.write_evidence(args.json_out, report)
        raise
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"],
                          native_delta=report["comparison"]["native_delta"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

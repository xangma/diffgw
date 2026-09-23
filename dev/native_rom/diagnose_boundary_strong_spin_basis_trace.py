"""Trace the frozen strong-spin NQC phase-basis derivative error; development only."""

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

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_ode import array, array_hash, stages_of
from .diagnose_boundary_strong_spin_nqc_inputs import _system

import torch

from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.nqc import (
    nqc_22_basis, seobnrv4_nqc_parameters, seobnrv4_nqc_targets,
)
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/boundary_strong_spin_basis_trace_plan.json"
CAMPAIGN = q.ROOT / "dev/native_rom/evidence/2026-09-23/parameter_boundary_campaign.json"
ATTRIBUTION = q.ROOT / "dev/native_rom/evidence/2026-09-23/boundary_strong_spin_nqc_inputs.json"


def _physics(row):
    m1, m2, s1, s2 = (torch.tensor(row[key], dtype=torch.float64)
                      for key in ("mass1", "mass2", "chi1", "chi2"))
    x1, x2 = m1/(m1+m2), m2/(m1+m2)
    ham = SEOBNRv4Hamiltonian(x1*x2, s1, s2, delta=x1-x2)
    flux = SEOBNRv4PublishedFlux(spin_truncation=q.CHOICES["spin_truncation"],
                                 retain_pan_cubic_spin=True)
    with torch.no_grad():
        r0, pr0, pphi0 = find_quasicircular_initial_conditions(
            ham, flux, torch.tensor(row["omega_start"], dtype=torch.float64))
        initial = torch.stack((r0, torch.zeros_like(r0), pr0, pphi0))
    return EOBTrajectoryIntegrator(ham, flux).vector_field, array(initial)


def _solve(field, initial, start, end, rtol, atol, first_step):
    from scipy.integrate import solve_ivp

    def rhs(_time, state):
        with torch.no_grad():
            return array(field(torch.from_numpy(state)))

    started = time.monotonic()
    solution = solve_ivp(rhs, (start, end), initial, method="DOP853",
                         rtol=rtol, atol=atol, first_step=first_step,
                         dense_output=True)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution, dict(rtol=rtol, atol=atol, evaluations=solution.nfev,
                          elapsed_seconds=time.monotonic()-started)


def _basis(field, states):
    values = []
    with torch.no_grad():
        for state in np.atleast_2d(states):
            tensor = torch.from_numpy(state)
            omega = field(tensor)[1]
            _, phase = nqc_22_basis(tensor[0], tensor[2], omega)
            values.append(array(phase))
    return np.asarray(values)


def _continuous(field, solution, match, steps):
    stencils = {}
    for step in steps:
        times = match + np.arange(-2, 3)*step
        values = _basis(field, solution.sol(times).T)
        first = (values[0]-8*values[1]+8*values[3]-values[4])/(12*step)
        second = (-values[0]+16*values[1]-30*values[2]
                  +16*values[3]-values[4])/(12*step**2)
        stencils[f"{step:g}"] = dict(first=first.tolist(), second=second.tolist())
    return stencils


def _local_fit(values, chart, window):
    design = torch.stack([torch.from_numpy(chart).pow(k) for k in range(7)], -1)
    coefficients = torch.linalg.lstsq(design, torch.from_numpy(values), driver="gels").solution
    return array(2*coefficients[2]/window**2)


def _hermite(knot_times, states, derivatives, queries):
    left = np.searchsorted(knot_times, queries, side="right")-1
    left = np.clip(left, 0, len(knot_times)-2)
    t0, t1 = knot_times[left], knot_times[left+1]
    span = t1-t0
    u = (queries-t0)/span
    return ((2*u**3-3*u**2+1)[:, None]*states[left]
            + ((u**3-2*u**2+u)*span)[:, None]*derivatives[left]
            + (-2*u**3+3*u**2)[:, None]*states[left+1]
            + ((u**3-u**2)*span)[:, None]*derivatives[left+1])


def _sampled(field, solution, global_times, chart, window, native_states,
             native_basis, accepted_times, accepted_states, accepted_derivatives):
    if min(global_times) < solution.t[0] or max(global_times) > solution.t[-1]:
        raise ValueError("dense solution does not cover the full native output stencil")
    if min(accepted_times) < solution.t[0] or max(accepted_times) > solution.t[-1]:
        raise ValueError("dense solution does not cover the bracketing accepted knots")
    states = solution.sol(global_times).T
    values = _basis(field, states)
    dense_accepted_states = solution.sol(accepted_times).T
    with torch.no_grad():
        dense_accepted_derivatives = array(field(torch.from_numpy(dense_accepted_states)))
    dense_hermite_states = _hermite(accepted_times, dense_accepted_states,
                                    dense_accepted_derivatives, global_times)
    dense_hermite_basis = _basis(field, dense_hermite_states)
    return dict(direct_second=_local_fit(values, chart, window).tolist(),
                accepted_hermite_second=_local_fit(dense_hermite_basis, chart, window).tolist(),
                max_accepted_state_error=np.max(abs(accepted_states-dense_accepted_states), axis=0).tolist(),
                max_accepted_derivative_error=np.max(abs(accepted_derivatives-dense_accepted_derivatives), axis=0).tolist(),
                max_hermite_interpolation_state_error=np.max(abs(dense_hermite_states-states), axis=0).tolist(),
                max_state_error=np.max(abs(native_states-states), axis=0).tolist(),
                max_basis_error=np.max(abs(native_basis-values), axis=0).tolist())


def _replay(name, row, prior, settings, targets, plan):
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
            ringdown_duration=settings.ringdown_duration_M,
            num_steps=settings.max_accepted_steps,
            max_accepted_steps=settings.max_accepted_steps,
            max_evaluations=settings.max_evaluations,
            max_rejected_steps=settings.max_rejected_steps)
    hashes = dict(mode=array_hash(array(result.mode)),
                  time=array_hash(array(result.time)))
    if hashes != prior["array_sha256"]:
        raise ValueError(f"{name} waveform differs from frozen campaign")
    stages = stages_of(result.trajectory)
    if any({key: value for key, value in stages[stage].items() if key != "array_sha256"}
           != prior["stages"][stage] for stage in ("coarse", "fine")):
        raise ValueError(f"{name} stage statistics differ from frozen campaign")
    system = _system(result, targets)
    match = float(result.attachment.match_time)
    restart = float(result.trajectory.restart_time)
    if abs(match-prior["match_time_M"]) > 1e-12 or restart != 165.0:
        raise ValueError(f"{name} event time differs from frozen campaign")
    fine = result.trajectory.fine
    local_times = array(fine.output_time[:int(fine.output_valid.sum())])
    local_match = system["match_time_local_M"]
    window = plan["dense_reference"]["local_fit_window_M"]
    selected = abs(local_times-local_match) <= window
    chart = (local_times[selected]-local_match)/window
    native_states = array(fine.output_state[:len(local_times)])[selected]
    native_basis = array(result.attachment.fine.nqc.sampled_phase_basis)[selected]
    if len(chart) != system["local_fit_samples"]:
        raise ValueError("local sample count differs from exact phase system")
    native_second = _local_fit(native_basis, chart, window)
    if np.max(abs(native_second-system["phase"]["matrix"][1])) > 1e-12:
        raise ValueError("sampled phase basis does not recover frozen NQC matrix")
    restart_state = array(fine.state[0])
    if not np.array_equal(restart_state, array(fine.output_state[0])):
        raise ValueError("fine accepted and output restart states differ")
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
    native_reconstructed = _hermite(accepted_times, accepted_states,
                                    accepted_derivatives, local_times[selected])
    native_hermite_error = float(np.max(abs(native_reconstructed-native_states)))
    if native_hermite_error > 1e-12:
        raise ValueError("accepted-state Hermite map does not recover native output")
    return dict(name=name, array_sha256=hashes, level={key: prior[key] for key in
                                                        ("atol", "rtol", "dt_geom")},
                match_time_M=match, restart_time_M=restart,
                restart_state=restart_state.tolist(),
                native_second=native_second.tolist(),
                local_fit_samples=len(chart), accepted_knots=len(accepted_times),
                native_hermite_reconstruction_error=native_hermite_error,
                elapsed_seconds=time.monotonic()-started), dict(
                    restart_state=restart_state, global_times=restart+local_times[selected],
                    chart=chart, native_states=native_states, native_basis=native_basis,
                    accepted_times=restart+accepted_times,
                    accepted_states=accepted_states,
                    accepted_derivatives=accepted_derivatives)


def _partition(native, accepted_hermite, sampled, continuous, global_reference):
    native, accepted_hermite = np.asarray(native), np.asarray(accepted_hermite)
    sampled = np.asarray(sampled)
    continuous, global_reference = np.asarray(continuous), np.asarray(global_reference)
    accepted_error = native-accepted_hermite
    interpolation = accepted_hermite-sampled
    local_fit = sampled-continuous
    restart = continuous-global_reference
    total = native-global_reference
    closure = accepted_error+interpolation+local_fit+restart-total
    if np.max(abs(closure)) > 1e-15:
        raise ValueError("derivative error partition does not close")
    return dict(fine_accepted_state_evolution=accepted_error.tolist(),
                hermite_output_interpolation=interpolation.tolist(),
                local_derivative_fit=local_fit.tolist(),
                restart_state=restart.tolist(), total=total.tolist(),
                closure_max_abs=float(np.max(abs(closure))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign = json.loads(CAMPAIGN.read_text())
    attribution = json.loads(ATTRIBUTION.read_text())
    if (plan["frozen_campaign_sha256"] != q.sha256(CAMPAIGN)
            or plan["row"] != "strong_aligned_spin"
            or plan["pair"] != ["ode_1", "ode_2"]
            or plan["unchanged_choices"] != dict(
                hermite_resampling=True, local_nqc_phase_derivatives=True,
                local_matching_second_derivatives=True,
                fixed_restart_time_M=165.0, output_grid_M=0.5)
            or campaign["schema"] != "diffgw.seobnrv4.parameter-boundary-campaign.v1"
            or not campaign["sources_unchanged"]
            or attribution["status"] != "diagnostic_complete"
            or attribution["campaign_sha256"] != q.sha256(CAMPAIGN)):
        parser.error("plan, frozen campaign, or NQC attribution differs from predeclaration")
    sources = q.source_snapshot()
    campaign_script = "dev/native_rom/diagnose_parameter_boundaries.py"
    sources[campaign_script] = q.sha256(q.ROOT / campaign_script)
    if (sources != campaign["source_sha256_before"]
            or q.canonical_hash(sources) != campaign["source_manifest_sha256"]):
        parser.error("generator sources differ from frozen campaign")
    row = campaign["rows"][plan["row"]]
    frozen = {item["name"]: item for item in row["runs"]}
    settings = q.Settings(atol_levels=tuple(campaign["settings"]["atol_levels"]),
                          rtol_levels=tuple(campaign["settings"]["rtol_levels"]))
    parameters = [torch.tensor(row["inputs"][key], dtype=torch.float64)
                  for key in ("mass1", "mass2", "chi1", "chi2")]
    targets = seobnrv4_nqc_targets(*seobnrv4_nqc_parameters(*parameters))
    torch.set_num_threads(1)
    report = dict(schema="diffgw.seobnrv4.boundary-strong-spin-basis-trace.v1",
                  scope=dict(**q.SCOPE, development_only=True,
                             same_provisional_equations=True,
                             numerical_generator_qualified=False),
                  status="running", plan_sha256=q.sha256(PLAN),
                  campaign_sha256=q.sha256(CAMPAIGN), attribution_sha256=q.sha256(ATTRIBUTION),
                  script_sha256=q.sha256(__file__),
                  helper_sha256=dict(
                      replay=q.sha256(q.ROOT / "dev/native_rom/diagnose_boundary_strong_spin_ode.py"),
                      matching=q.sha256(q.ROOT / "dev/native_rom/diagnose_boundary_strong_spin_nqc_inputs.py")),
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_sha256_before=sources,
                  process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid()),
                  row=row["inputs"], pair=plan["pair"], runs=[], dense={}, comparison={})
    q.write_evidence(args.json_out, report)
    try:
        run_data = []
        for name in plan["pair"]:
            serial, local = _replay(name, row["inputs"], frozen[name], settings, targets, plan)
            prior = next(item for item in attribution["runs"] if item["name"] == name)
            if (serial["array_sha256"] != prior["array_sha256"]
                    or np.max(abs(np.asarray(serial["native_second"])
                                  - np.asarray(prior["phase"]["matrix"][1]))) > 1e-12):
                raise ValueError(f"{name} NQC derivative differs from prior attribution")
            report["runs"].append(serial)
            run_data.append(local)
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(run=name, seconds=serial["elapsed_seconds"],
                                  frozen_hash_verified=True)), flush=True)
        if report["runs"][0]["match_time_M"] != report["runs"][1]["match_time_M"]:
            raise ValueError("pair has distinct match times")
        field, initial = _physics(row["inputs"])
        match = report["runs"][0]["match_time_M"]
        restart = report["runs"][0]["restart_time_M"]
        config = plan["dense_reference"]
        steps = config["five_point_steps_M"]
        reference_key = f"{min(steps):g}"
        dense_end = max(match+2*max(steps),
                        *(max(local["global_times"][-1], local["accepted_times"][-1])
                          for local in run_data)) + 1e-9
        report["dense_end_M"] = dense_end
        dense_levels = []
        for rtol, atol in config["rtol_atol"]:
            global_solution, global_meta = _solve(
                field, initial, 0., dense_end, rtol, atol,
                config["global_first_step_M"])
            global_stencils = _continuous(field, global_solution, match, steps)
            global_meta.update(stencils=global_stencils,
                               restart_state=global_solution.sol(restart).tolist(),
                               match_state=global_solution.sol(match).tolist())
            level = dict(global_reference=global_meta, restarted={})
            for run, local in zip(report["runs"], run_data):
                solution, meta = _solve(
                    field, local["restart_state"], restart, dense_end,
                    rtol, atol, config["restart_first_step_M"])
                meta.update(stencils=_continuous(field, solution, match, steps),
                            match_state=solution.sol(match).tolist(),
                            sampled=_sampled(
                                field, solution, local["global_times"], local["chart"],
                                config["local_fit_window_M"], local["native_states"],
                                local["native_basis"], local["accepted_times"],
                                local["accepted_states"], local["accepted_derivatives"]))
                level["restarted"][run["name"]] = meta
                print(json.dumps(dict(dop_rtol=rtol, run=run["name"],
                                      seconds=meta["elapsed_seconds"],
                                      evaluations=meta["evaluations"])), flush=True)
            dense_levels.append(level)
            report["dense"][f"{rtol:g}"] = level
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
        delta = np.asarray(report["runs"][1]["native_second"])-np.asarray(
            report["runs"][0]["native_second"])
        category_delta = {
            key: (np.asarray(partitions["ode_2"][key])
                  -np.asarray(partitions["ode_1"][key])).tolist()
            for key in ("fine_accepted_state_evolution", "hermite_output_interpolation",
                        "local_derivative_fit", "restart_state")}
        closure = sum(np.asarray(item) for item in category_delta.values())-delta
        if np.max(abs(closure)) > 1e-15:
            raise ValueError("between-run error partition does not close")
        # Ordered coordinate swaps isolate the restart-state difference. The
        # order is frozen in the plan; no linearity or order independence is assumed.
        medium = run_data[0]["restart_state"]
        fine = run_data[1]["restart_state"]
        current = medium.copy()
        previous = np.asarray(tight["restarted"]["ode_1"]["stencils"][reference_key]["second"])
        substitutions = []
        for coordinate in config["state_coordinate_order"]:
            index = {"r": 0, "phi": 1, "pr_star": 2, "pphi": 3}[coordinate]
            current[index] = fine[index]
            if np.array_equal(current, fine):
                value = np.asarray(tight["restarted"]["ode_2"]["stencils"][reference_key]["second"])
                meta = None
            else:
                solution, meta = _solve(field, current, restart, dense_end,
                                        *config["rtol_atol"][-1],
                                        config["restart_first_step_M"])
                value = np.asarray(_continuous(field, solution, match, steps)[reference_key]["second"])
            substitutions.append(dict(coordinate=coordinate,
                                      state=current.tolist(), contribution=(value-previous).tolist(),
                                      solve=meta))
            previous = value
        restart_delta = (np.asarray(tight["restarted"]["ode_2"]["stencils"][reference_key]["second"])
                         -np.asarray(tight["restarted"]["ode_1"]["stencils"][reference_key]["second"]))
        substitution_closure = sum(np.asarray(item["contribution"]) for item in substitutions)-restart_delta
        if np.max(abs(substitution_closure)) > 1e-15:
            raise ValueError("restart-state substitution does not close")
        report["comparison"] = dict(partition=partitions,
                                    native_delta=delta.tolist(), category_delta=category_delta,
                                    between_run_closure_max_abs=float(np.max(abs(closure))),
                                    restart_state_delta=(fine-medium).tolist(),
                                    restart_substitutions=substitutions,
                                    restart_substitution_closure_max_abs=float(np.max(abs(substitution_closure))),
                                    tight_vs_loose_reference_spread=(
                                        np.asarray(dense_levels[-1]["global_reference"]["stencils"][reference_key]["second"])
                                        -np.asarray(dense_levels[0]["global_reference"]["stencils"][reference_key]["second"])).tolist())
        report["source_sha256_after"] = q.source_snapshot()
        report["source_sha256_after"][campaign_script] = q.sha256(q.ROOT / campaign_script)
        report["sources_unchanged"] = report["source_sha256_after"] == sources
        if not report["sources_unchanged"]:
            raise ValueError("generator sources changed during diagnostic")
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

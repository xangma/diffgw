"""Attribute the frozen strong-spin NQC phase solve; development only."""

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
from .diagnose_nqc_matching_inputs import _decompose

import torch

from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from torchwave.approximants.eob.numerical_attachment import _eval_batch_scalar, natural_cubic_spline
from torchwave.approximants.eob.nqc import (
    nqc_22_basis, seobnrv4_nqc_parameters, seobnrv4_nqc_targets,
)


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/boundary_strong_spin_nqc_inputs_plan.json"
CAMPAIGN = q.ROOT / "dev/native_rom/evidence/2026-09-23/parameter_boundary_campaign.json"
WINDOW_M = 0.5
DEGREE = 6


def _system(result, targets):
    fine = result.attachment.fine
    nqc = fine.nqc
    run = result.trajectory.fine
    count = int(run.output_valid.sum().detach())
    axis = run.output_time[:count]
    match = fine.match_time
    if nqc.sampled_phase_basis.shape != (count, 2):
        raise ValueError("fine NQC basis differs from trajectory support")
    selected = (axis - match).detach().abs() <= WINDOW_M
    chart = (axis[selected] - match) / WINDOW_M
    if chart.numel() < DEGREE + 3 or not bool((chart.min() < 0) & (chart.max() > 0)):
        raise ValueError("local phase fit lacks two-sided samples")
    design = torch.stack([chart.pow(k) for k in range(DEGREE + 1)], -1)
    basis_fit = torch.linalg.lstsq(design, nqc.sampled_phase_basis[selected],
                                   driver="gels").solution
    matrix = torch.stack((basis_fit[1] / WINDOW_M,
                          2 * basis_fit[2] / WINDOW_M**2))
    phase_spline = natural_cubic_spline(axis, nqc.sampled_waveform_phase)
    phase_first = _eval_batch_scalar(phase_spline, match, derivative=1)
    phase_values = nqc.sampled_waveform_phase[selected]
    phase_fit = torch.linalg.lstsq(
        design, phase_values - phase_values[chart.numel() // 2],
        driver="gels").solution
    phase_second = 2 * phase_fit[2] / WINDOW_M**2
    base = torch.stack((phase_first, phase_second))
    target = torch.stack(targets[3:])
    rhs = base - target
    reconstructed = torch.linalg.solve(matrix, rhs)
    stored = fine.calibration.coefficients.phase
    error = float((reconstructed - stored).abs().max())
    if error > 1e-7:
        raise ValueError(f"phase system does not recover stored coefficients: {error}")
    condition = float(torch.linalg.cond(matrix))
    if abs(condition - float(fine.calibration.phase_condition)) > 1e-7:
        raise ValueError("phase condition differs from stored calibration")
    return dict(match_time_local_M=float(match), local_fit_samples=int(chart.numel()),
                phase=dict(matrix=array(matrix), rhs=array(rhs), base=array(base),
                           target=array(target), coefficients=array(stored),
                           reconstruction_max_abs_error=error, condition=condition,
                           singular_values=array(torch.linalg.svdvals(matrix))))


def _dense_reference(row, match, plan):
    from scipy.integrate import solve_ivp
    from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
    from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
    from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
    from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
    from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

    m1, m2, chi1, chi2 = (torch.tensor(row[key], dtype=torch.float64)
                          for key in ("mass1", "mass2", "chi1", "chi2"))
    x1, x2 = m1/(m1+m2), m2/(m1+m2)
    ham = SEOBNRv4Hamiltonian(x1*x2, chi1, chi2, delta=x1-x2)
    flux = SEOBNRv4PublishedFlux(spin_truncation=q.CHOICES["spin_truncation"],
                                 retain_pan_cubic_spin=True)
    with torch.no_grad():
        r0, pr0, pphi0 = find_quasicircular_initial_conditions(
            ham, flux, torch.tensor(row["omega_start"], dtype=torch.float64))
        initial = torch.stack((r0, torch.zeros_like(r0), pr0, pphi0))
    field = EOBTrajectoryIntegrator(ham, flux).vector_field

    def scipy_field(_t, state):
        with torch.no_grad():
            return array(field(torch.from_numpy(state)))

    def quantities(solution, times):
        basis_values, modes = [], []
        for coordinate in times:
            state = torch.from_numpy(solution.sol(coordinate))
            with torch.no_grad():
                omega = field(state)[1]
                mode, _ = published_mode22_and_frequency(
                    state, ham, flux, spin_v6=q.CHOICES["spin_v6"],
                    include_pan_v8=False, include_positive_chi_540=False,
                    retain_pan_cubic_spin=True)
                _, basis = nqc_22_basis(state[0], state[2], omega)
            basis_values.append(array(basis))
            modes.append(complex(array(mode)))
        phase = -np.angle(np.asarray(modes)*np.conj(modes[len(modes)//2]))
        if np.max(abs(np.diff(phase))) >= np.pi:
            raise ValueError("dense phase crosses a local branch")
        return np.column_stack((phase, np.asarray(basis_values)))

    config = plan["dense_reference"]
    levels = []
    for rtol, atol in config["tolerances_rtol_atol"]:
        started = time.monotonic()
        solution = solve_ivp(
            scipy_field, (0., match + 2*max(config["five_point_steps_M"])),
            array(initial), method="DOP853", rtol=rtol, atol=atol,
            first_step=config["first_step_M"], dense_output=True)
        if not solution.success:
            raise RuntimeError(solution.message)
        stencils = {}
        for step in config["five_point_steps_M"]:
            values = quantities(solution, match + np.arange(-2, 3)*step)
            first = (values[0]-8*values[1]+8*values[3]-values[4])/(12*step)
            second = (-values[0]+16*values[1]-30*values[2]
                      +16*values[3]-values[4])/(12*step**2)
            stencils[f"{step:g}"] = dict(first=first.tolist(), second=second.tolist())
        levels.append(dict(rtol=rtol, atol=atol, evaluations=solution.nfev,
                           elapsed_seconds=time.monotonic()-started, stencils=stencils))
        print(json.dumps(dict(dop_rtol=rtol, seconds=levels[-1]["elapsed_seconds"],
                              evaluations=solution.nfev)), flush=True)
    return levels


def _dense_comparison(runs, dense, plan):
    steps = plan["dense_reference"]["five_point_steps_M"]
    key = f"{min(steps):g}"
    reference = dense[-1]["stencils"][key]
    reference_first = np.asarray(reference["first"])
    reference_second = np.asarray(reference["second"])
    results = {}
    for run in runs:
        system = run["phase"]
        native_first = np.r_[system["base"][0], system["matrix"][0]]
        native_second = np.r_[system["base"][1], system["matrix"][1]]
        results[run["name"]] = dict(first_error=(native_first-reference_first).tolist(),
                                    second_error=(native_second-reference_second).tolist())
    earlier = dense[0]["stencils"][key]
    coarser = dense[-1]["stencils"][f"{steps[-2]:g}"]
    return dict(reference_step_M=min(steps),
                first_reference=reference["first"], second_reference=reference["second"],
                tolerance_spread=dict(first=(np.asarray(earlier["first"])-reference_first).tolist(),
                                      second=(np.asarray(earlier["second"])-reference_second).tolist()),
                step_spread=dict(first=(np.asarray(coarser["first"])-reference_first).tolist(),
                                 second=(np.asarray(coarser["second"])-reference_second).tolist()),
                native_error=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign = json.loads(CAMPAIGN.read_text())
    if (plan["frozen_campaign_sha256"] != q.sha256(CAMPAIGN)
            or plan["row"] != "strong_aligned_spin"
            or plan["pair"] != ["ode_1", "ode_2"]
            or plan["unchanged_choices"] != dict(
                hermite_resampling=True, local_nqc_phase_derivatives=True,
                local_matching_second_derivatives=True,
                fixed_restart_time_M=165.0, output_grid_M=0.5)
            or campaign["schema"] != "diffgw.seobnrv4.parameter-boundary-campaign.v1"
            or not campaign["sources_unchanged"]):
        parser.error("plan or frozen campaign differs from predeclaration")
    sources = q.source_snapshot()
    campaign_script = "dev/native_rom/diagnose_parameter_boundaries.py"
    sources[campaign_script] = q.sha256(q.ROOT / campaign_script)
    if (sources != campaign["source_sha256_before"]
            or q.canonical_hash(sources) != campaign["source_manifest_sha256"]):
        parser.error("generator sources differ from frozen campaign")
    row = campaign["rows"][plan["row"]]
    if row["checks"]["ode_tolerance"]["gates"]["complex_error_budget"] is not False:
        parser.error("frozen strong-spin ODE failure absent")
    frozen = {item["name"]: item for item in row["runs"]}
    settings = q.Settings(atol_levels=tuple(campaign["settings"]["atol_levels"]),
                          rtol_levels=tuple(campaign["settings"]["rtol_levels"]))
    parameters = [torch.tensor(row["inputs"][key], dtype=torch.float64)
                  for key in ("mass1", "mass2", "chi1", "chi2")]
    targets = seobnrv4_nqc_targets(*seobnrv4_nqc_parameters(*parameters))
    torch.set_num_threads(1)
    report = dict(schema="diffgw.seobnrv4.boundary-strong-spin-nqc-inputs.v1",
                  scope=dict(**q.SCOPE, development_only=True,
                             same_provisional_equations=True,
                             numerical_generator_qualified=False),
                  status="running", plan_sha256=q.sha256(PLAN),
                  campaign_sha256=q.sha256(CAMPAIGN), script_sha256=q.sha256(__file__),
                  helper_sha256=dict(
                      replay=q.sha256(q.ROOT / "dev/native_rom/diagnose_boundary_strong_spin_ode.py"),
                      attribution=q.sha256(q.ROOT / "dev/native_rom/diagnose_nqc_matching_inputs.py")),
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_sha256_before=sources,
                  process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid()),
                  row=row["inputs"], pair=plan["pair"], runs=[], comparison={})
    q.write_evidence(args.json_out, report)
    try:
        runs = []
        with torch.no_grad():
            for name in plan["pair"]:
                prior = frozen[name]
                started = time.monotonic()
                result = reconstruct_seobnrv4_22_adaptive(
                    torch.tensor(row["inputs"]["omega_start"], dtype=torch.float64),
                    **{key: row["inputs"][key] for key in ("mass1", "mass2", "chi1", "chi2")},
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
                if any({key: value for key, value in stages[stage].items()
                        if key != "array_sha256"} != prior["stages"][stage]
                       for stage in ("coarse", "fine")):
                    raise ValueError(f"{name} stage statistics differ from frozen campaign")
                system = _system(result, targets)
                match_global = float(result.attachment.match_time)
                if abs(match_global - prior["match_time_M"]) > 1e-12:
                    raise ValueError(f"{name} match time differs from frozen campaign")
                run = dict(name=name, level={key: prior[key] for key in ("atol", "rtol", "dt_geom")},
                           array_sha256=hashes, match_time_global_M=match_global,
                           **system, elapsed_seconds=time.monotonic()-started)
                runs.append(run)
                serial = dict(run)
                serial["phase"] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                   for key, value in system["phase"].items()}
                report["runs"].append(serial)
                q.write_evidence(args.json_out, report)
                print(json.dumps(dict(run=name, seconds=run["elapsed_seconds"],
                                      frozen_hash_verified=True)), flush=True)
        if runs[0]["match_time_global_M"] != runs[1]["match_time_global_M"]:
            raise ValueError("pair has distinct match times")
        report["comparison"]["phase_attribution"] = _decompose(
            runs[0]["phase"], runs[1]["phase"])
        q.write_evidence(args.json_out, report)
        dense = _dense_reference(row["inputs"], runs[0]["match_time_global_M"], plan)
        report["dense_dop853"] = dense
        report["comparison"]["dense"] = _dense_comparison(runs, dense, plan)
        report["source_sha256_after"] = q.source_snapshot()
        report["source_sha256_after"][campaign_script] = q.sha256(q.ROOT / campaign_script)
        report["sources_unchanged"] = report["source_sha256_after"] == sources
        if not report["sources_unchanged"]:
            raise ValueError("generator sources changed during replay")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        q.write_evidence(args.json_out, report)
        raise
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"],
                          phase_delta=report["comparison"]["phase_attribution"]["delta_coefficients"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

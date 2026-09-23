"""Locate the frozen strong aligned-spin ODE discrepancy without retuning it."""

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
from .diagnose_hermite_error import phase_attribution

import torch

from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-23/boundary_strong_spin_ode_plan.json"
CAMPAIGN = q.ROOT / "dev/native_rom/evidence/2026-09-23/parameter_boundary_campaign.json"


def array(value):
    result = value.detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("nonfinite trajectory or waveform")
    return result


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def stages_of(trajectory):
    stages = {}
    for name in ("coarse", "fine"):
        stage = getattr(trajectory, name)
        stats = stage.statistics
        stages[name] = dict(
            accepted=stats.total_accepted,
            rejected=stats.total_rejected,
            evaluations=stats.total_evaluations,
            termination=list(stats.termination),
            event_time_M=float(array(stats.event_time)),
            array_sha256={key: array_hash(array(getattr(stage, key)))
                          for key in ("time", "state", "derivative")})
    return stages


def raw_mode_of(trajectory, row, length):
    m1, m2 = row["mass1"], row["mass2"]
    total = m1 + m2
    hamiltonian = SEOBNRv4Hamiltonian(
        torch.tensor(m1*m2/total**2, dtype=torch.float64),
        torch.tensor(row["chi1"], dtype=torch.float64),
        torch.tensor(row["chi2"], dtype=torch.float64),
        delta=torch.tensor((m1-m2)/total, dtype=torch.float64))
    flux = SEOBNRv4PublishedFlux(spin_truncation=q.CHOICES["spin_truncation"],
                                 retain_pan_cubic_spin=True)
    valid = array(trajectory.valid).astype(bool)
    if valid.shape != (length,):
        raise ValueError("trajectory validity differs from waveform support")
    raw, _ = published_mode22_and_frequency(
        trajectory.state[trajectory.valid], hamiltonian, flux,
        spin_v6=q.CHOICES["spin_v6"], include_pan_v8=False,
        include_positive_chi_540=False, retain_pan_cubic_spin=True)
    result = np.zeros(length, dtype=np.complex128)
    result[valid] = array(raw)
    return result, valid


def regional_decomposition(candidate, reference, plan):
    if not np.array_equal(candidate["time"], reference["time"]):
        raise ValueError("ODE pair does not share an output grid")
    time_axis = reference["time"]
    a, b = candidate["mode"], reference["mode"]
    amplitude_error = (abs(a)-abs(b))**2
    phase_error = 4*abs(a)*abs(b)*np.sin(np.angle(a*np.conj(b))/2)**2
    complex_error = abs(a-b)**2
    if not np.allclose(complex_error, amplitude_error+phase_error,
                       rtol=1e-8, atol=1e-20):
        raise ValueError("amplitude/phase decomposition identity failed")
    total_error = float(complex_error.sum())
    total_power = float(np.square(abs(b)).sum())
    if min(total_error, total_power) <= 0:
        raise ValueError("zero waveform error or reference norm")
    boundaries = dict(restart=reference["restart_time_M"],
                      boundary_minus_30=reference["boundary_time_M"]-30,
                      boundary=reference["boundary_time_M"],
                      peak=reference["peak_time_M"])
    regions = {}
    for item in plan["regions"]:
        start = boundaries[item["start"]] if isinstance(item["start"], str) else item["start"]
        end = (boundaries[item["end"]] if isinstance(item["end"], str)
               else float("inf") if item["end"] is None else item["end"])
        mask = (time_axis >= start) & (time_axis < end)
        error = float(complex_error[mask].sum())
        power = float(np.square(abs(b[mask])).sum())
        regions[item["name"]] = dict(
            start_M=start, end_M=end if np.isfinite(end) else None,
            samples=int(mask.sum()),
            error_energy_fraction=error/total_error,
            reference_energy_fraction=power/total_power,
            contribution_to_whole_relative_l2=float(np.sqrt(error/total_power)),
            local_relative_l2=float(np.sqrt(error/power)) if power else None,
            amplitude_error_energy_fraction=float(amplitude_error[mask].sum()/total_error),
            phase_error_energy_fraction=float(phase_error[mask].sum()/total_error))
    if sum(item["samples"] for item in regions.values()) != len(time_axis):
        raise ValueError("regions do not partition the waveform")
    return dict(
        total_relative_l2=float(np.sqrt(total_error/total_power)),
        amplitude_error_energy_fraction=float(amplitude_error.sum()/total_error),
        phase_error_energy_fraction=float(phase_error.sum()/total_error),
        regions=regions)


def pre_attachment(candidate, reference):
    if not np.array_equal(candidate["time"], reference["time"]):
        raise ValueError("pre-attachment grids differ")
    axis = reference["time"]
    mask = (candidate["valid"] & reference["valid"]
            & (axis < reference["boundary_time_M"]))
    if not mask.any():
        raise ValueError("no common pre-attachment support")
    delta = candidate["state"][mask]-reference["state"][mask]
    frequency_delta = candidate["frequency"][mask]-reference["frequency"][mask]
    return dict(samples=int(mask.sum()), last_common_time_M=float(axis[mask][-1]),
                state_max_abs=dict(zip(("r", "phi", "pr_star", "pphi"),
                                       np.max(abs(delta), axis=0).tolist())),
                state_rms=dict(zip(("r", "phi", "pr_star", "pphi"),
                                   np.sqrt(np.mean(delta**2, axis=0)).tolist())),
                frequency_max_abs=float(np.max(abs(frequency_delta))),
                raw_mode=q.mode_metrics(candidate["raw_mode"][mask],
                                        reference["raw_mode"][mask]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    campaign = json.loads(CAMPAIGN.read_text())
    if (plan["frozen_campaign_sha256"] != q.sha256(CAMPAIGN)
            or campaign["schema"] != "diffgw.seobnrv4.parameter-boundary-campaign.v1"
            or not campaign["sources_unchanged"]
            or plan["row"] != "strong_aligned_spin"
            or plan["pair"] != ["ode_1", "ode_2"]
            or plan["unchanged_choices"] != dict(
                hermite_resampling=True, local_nqc_phase_derivatives=True,
                local_matching_second_derivatives=True,
                fixed_restart_time_M=165.0, output_grid_M=0.5)):
        parser.error("plan or frozen campaign differs from the predeclaration")
    sources = q.source_snapshot()
    if (sources != campaign["source_sha256_before"]
            or q.canonical_hash(sources) != campaign["source_manifest_sha256"]):
        parser.error("generator sources differ from the frozen campaign")
    row = campaign["rows"][plan["row"]]
    frozen = {item["name"]: item for item in row["runs"]}
    if row["checks"]["ode_tolerance"]["gates"]["complex_error_budget"] is not False:
        parser.error("row lacks the frozen strong-spin ODE failure")
    settings = q.Settings(atol_levels=tuple(campaign["settings"]["atol_levels"]),
                          rtol_levels=tuple(campaign["settings"]["rtol_levels"]))
    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.boundary-strong-spin-ode.v1",
        scope=dict(**q.SCOPE, development_only=True,
                   numerical_generator_qualified=False,
                   same_provisional_equations=True),
        status="running", plan_sha256=q.sha256(PLAN),
        campaign_sha256=q.sha256(CAMPAIGN), script_sha256=q.sha256(__file__),
        helper_sha256=q.sha256(q.ROOT / "dev/native_rom/diagnose_hermite_error.py"),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256_before=sources,
        process=dict(argv=sys.argv, cwd=os.getcwd(), host=socket.gethostname(), pid=os.getpid()),
        row=row["inputs"], pair=plan["pair"], levels={}, comparison={})
    q.write_evidence(args.json_out, report)
    try:
        runs = []
        results = []
        with torch.no_grad():
            for name in plan["pair"]:
                prior = frozen[name]
                started = time.monotonic()
                result = reconstruct_seobnrv4_22_adaptive(
                    torch.tensor(row["inputs"]["omega_start"], dtype=torch.float64),
                    **{key: row["inputs"][key]
                       for key in ("mass1", "mass2", "chi1", "chi2")},
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
                trajectory = result.trajectory
                mode, axis = array(result.mode), array(result.time)
                hashes = dict(mode=array_hash(mode), time=array_hash(axis))
                if hashes != prior["array_sha256"]:
                    raise ValueError(f"{name} waveform differs from frozen campaign")
                stages = stages_of(trajectory)
                if any({key: value for key, value in stages[stage].items()
                        if key != "array_sha256"} != prior["stages"][stage]
                       for stage in ("coarse", "fine")):
                    raise ValueError(f"{name} stage statistics differ from frozen campaign")
                raw_mode, valid = raw_mode_of(trajectory, row["inputs"], len(mode))
                calibration = result.attachment.fine.calibration
                coefficients = calibration.coefficients
                run = dict(name=name, dt_geom=prior["dt_geom"], time=axis, mode=mode,
                           state=array(trajectory.state),
                           frequency=array(trajectory.frequency),
                           valid=valid, raw_mode=raw_mode,
                           peak_time_M=float(array(trajectory.peak.time)),
                           match_time_M=float(array(result.attachment.match_time)),
                           boundary_time_M=float(array(result.attachment.boundary_time)),
                           restart_time_M=float(array(trajectory.restart_time).reshape(-1)[0]))
                report["levels"][name] = dict(
                    array_sha256=hashes, stages=stages, samples=len(axis),
                    events_M={key: run[key] for key in (
                        "peak_time_M", "match_time_M", "boundary_time_M", "restart_time_M")},
                    nqc_coefficients=dict(amplitude=array(coefficients.amplitude).tolist(),
                                          phase=array(coefficients.phase).tolist()),
                    nqc_matching=dict(
                        sampling_condition=float(array(calibration.sampling_condition)),
                        amplitude_condition=float(array(calibration.amplitude_condition)),
                        phase_condition=float(array(calibration.phase_condition)),
                        amplitude_residual=array(calibration.amplitude_residual).tolist(),
                        phase_residual=array(calibration.phase_residual).tolist()),
                    elapsed_seconds=time.monotonic()-started)
                runs.append(run)
                results.append(result)
                q.write_evidence(args.json_out, report)
                print(json.dumps(dict(run=name, seconds=report["levels"][name]["elapsed_seconds"],
                                      frozen_hash_verified=True)), flush=True)
        candidate, reference = runs
        report["comparison"] = dict(
            whole=q.compare_runs(candidate, reference),
            regional=regional_decomposition(candidate, reference, plan),
            pre_attachment=pre_attachment(candidate, reference),
            nqc_phase_attribution=phase_attribution(*results))
        if report["comparison"]["whole"] != row["checks"]["ode_tolerance"]["medium_to_fine"]:
            raise ValueError("whole-waveform comparison differs from frozen campaign")
        report["source_sha256_after"] = q.source_snapshot()
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
                          complex_relative_l2=report["comparison"]["whole"]["complex_relative_l2"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

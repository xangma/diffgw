"""Exact-hash trace of a failed boundary-row ODE tolerance comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_trajectory_sensitivity import _array, _hash, _inspiral, _regions
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from .eob.torch.historical_hamiltonian import HistoricalLogDerivative
from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/boundary_tolerance_trace_plan.json"


def replay(row, level, settings, numerical, candidate, warm_frequency,
           first_step, expected):
    started = time.monotonic()
    with torch.no_grad():
        result = reconstruct_seobnrv4_22_adaptive(
            torch.tensor(row["omega_start"], dtype=torch.float64),
            **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
            **q.CHOICES,
            **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
            diagnostic_hermite_resampling=numerical["hermite_resampling"],
            diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=numerical["fixed_restart_time_M"],
            diagnostic_coarse_initial_step_M=first_step,
            diagnostic_warm_start_Momega=warm_frequency,
            diagnostic_mode_33_higher_spin=candidate["mode_33_higher_spin"],
            diagnostic_hamiltonian_log_derivative=candidate["hamiltonian_log_derivative"],
            ringdown_duration=settings.ringdown_duration_M,
            num_steps=settings.max_accepted_steps,
            max_accepted_steps=settings.max_accepted_steps,
            max_evaluations=settings.max_evaluations,
            max_rejected_steps=settings.max_rejected_steps)
        axis, mode = _array(result.time), _array(result.mode)
        hashes = {"time": _hash(axis), "mode": _hash(mode)}
        if hashes != expected["array_sha256"]:
            raise ValueError(f"{level['name']} waveform differs from frozen boundary run")
        trajectory = result.trajectory
        m1, m2 = row["mass1"], row["mass2"]
        nu = m1*m2/(m1+m2)**2
        ham = HistoricalLogDerivative(
            torch.tensor(nu, dtype=torch.float64),
            torch.tensor(row["chi1"], dtype=torch.float64),
            torch.tensor(row["chi2"], dtype=torch.float64),
            delta=torch.tensor((m1-m2)/(m1+m2), dtype=torch.float64))
        flux = SEOBNRv4PublishedFlux(
            spin_truncation=q.CHOICES["spin_truncation"],
            retain_pan_cubic_spin=True,
            mode_33_higher_spin=candidate["mode_33_higher_spin"])
        valid = _array(trajectory.valid).astype(bool)
        raw_mode = np.zeros_like(mode)
        raw, _ = published_mode22_and_frequency(
            trajectory.state[trajectory.valid], ham, flux,
            spin_v6=q.CHOICES["spin_v6"], include_pan_v8=False,
            include_positive_chi_540=False, retain_pan_cubic_spin=True)
        raw_mode[valid] = _array(raw)
        coefficients = result.attachment.fine.calibration.coefficients
        return dict(**level, time=axis, mode=mode, state=_array(trajectory.state),
                    frequency=_array(trajectory.frequency), valid=valid,
                    raw_mode=raw_mode, hashes=hashes,
                    peak_time_M=float(_array(trajectory.peak.time)),
                    match_time_M=float(_array(result.attachment.match_time)),
                    boundary_time_M=float(_array(result.attachment.boundary_time)),
                    restart_time_M=float(_array(trajectory.restart_time).reshape(-1)[0]),
                    nqc_coefficients=dict(
                        amplitude=_array(coefficients.amplitude).tolist(),
                        phase=_array(coefficients.phase).tolist()),
                    elapsed_seconds=time.monotonic()-started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_campaign"]
    if (plan["schema"] != "diffgw.seobnrv4.boundary-tolerance-trace-plan.v1"
            or q.sha256(prior_path) != plan["prior_campaign_sha256"]
            or plan["row_name"] != "strong_aligned_spin"
            or plan["levels"] != ["ode_1", "ode_2"]):
        parser.error("frozen boundary trace plan differs")
    prior = json.loads(prior_path.read_text())
    row_report = prior["rows"][plan["row_name"]]
    if (prior["status"] != "failed"
            or row_report["check"]["numerical_check_passed"]
            or prior["candidate_choice"]["hamiltonian_log_derivative"] != "lal_v4"):
        parser.error("expected frozen strong-aligned boundary failure")
    torch.set_num_threads(1)
    settings = q.Settings(**prior["settings"])
    source_paths = [
        "src/torchwave/approximants/eob/published_waveform.py",
        "src/torchwave/approximants/eob/seobnrv4_flux.py",
        "dev/native_rom/eob/torch/historical_hamiltonian.py",
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/qualify_generator.py",
    ]
    report = dict(
        schema="diffgw.seobnrv4.boundary-tolerance-trace.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), prior_sha256=q.sha256(prior_path),
        script_sha256=q.sha256(__file__),
        source_sha256={p: q.sha256(q.ROOT / p) for p in source_paths},
        row_name=plan["row_name"], row=row_report["row"], settings=asdict(settings),
        levels=[], comparison=None, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        runs = []
        for name in plan["levels"]:
            print(f"replay={name}", flush=True)
            frozen = row_report["runs"][name]
            run = replay(row_report["row"], frozen["level"], settings,
                         prior["numerical_choice"], prior["candidate_choice"],
                         plan["warm_start_Momega"], plan["fixed_coarse_first_step_M"],
                         frozen["native"])
            runs.append(run)
            report["levels"].append({k: v for k, v in run.items() if k not in
                                     ("time", "mode", "state", "frequency", "valid", "raw_mode")})
            q.write_evidence(args.json_out, report)
        medium, fine = runs
        boundary, restart = fine["boundary_time_M"], fine["restart_time_M"]
        windows = (("coarse_prefix", 0., restart),
                   ("fine_early", restart, boundary-30.),
                   ("fine_late", boundary-30., boundary),
                   ("attachment", boundary, boundary+10.),
                   ("ringdown", boundary+10., float("inf")))
        report["comparison"] = dict(
            whole=q.compare_runs(medium, fine),
            regions=_regions(medium, fine, windows),
            pre_attachment=_inspiral(medium, fine, boundary))
        target = row_report["check"]["medium_to_fine"]["complex_relative_l2"]
        if abs(report["comparison"]["whole"]["complex_relative_l2"]-target) > 1e-12:
            raise ValueError("whole-waveform replay metric differs")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Attribute the frozen inline warm-start aligned-spin tolerance failure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_nqc_matching_inputs import _decompose, _system
from .diagnose_trajectory_sensitivity import _array, _hash, _inspiral, _regions


EVIDENCE = q.ROOT / "dev/native_rom/evidence/2026-09-23"
PLAN = EVIDENCE / "inline_warmstart_spin_tolerance_plan.json"
CONVERGENCE_PLAN = EVIDENCE / "inline_warmstart_convergence_plan.json"


def _replay(row, level, frozen, choices, settings, targets):
    from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
    from torchwave.approximants.eob.published_waveform import published_mode22_and_frequency
    from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
    from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

    started = time.monotonic()
    with torch.no_grad():
        result = reconstruct_seobnrv4_22_adaptive(
            torch.tensor(row["omega_start"], dtype=torch.float64),
            **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
            **q.CHOICES,
            **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
            diagnostic_hermite_resampling=choices["hermite_resampling"],
            diagnostic_local_nqc_derivatives=choices["local_nqc_phase_derivatives"],
            diagnostic_restart_time_M=choices["fixed_restart_time_M"],
            diagnostic_coarse_initial_step_M=choices["coarse_first_step_M"],
            diagnostic_warm_start_Momega=choices["native_start_Momega"],
            ringdown_duration=settings.ringdown_duration_M,
            num_steps=settings.max_accepted_steps,
            max_accepted_steps=settings.max_accepted_steps,
            max_evaluations=settings.max_evaluations,
            max_rejected_steps=settings.max_rejected_steps,
        )
        time_axis, mode = _array(result.time), _array(result.mode)
        hashes = {"time": _hash(time_axis), "mode": _hash(mode)}
        if hashes != frozen["array_sha256"]:
            raise ValueError(f"{level['name']} replay differs from frozen waveform")
        trajectory = result.trajectory
        m1, m2 = row["mass1"], row["mass2"]
        nu = m1*m2/(m1+m2)**2
        hamiltonian = SEOBNRv4Hamiltonian(
            torch.tensor(nu, dtype=torch.float64),
            torch.tensor(row["chi1"], dtype=torch.float64),
            torch.tensor(row["chi2"], dtype=torch.float64),
            delta=torch.tensor((m1-m2)/(m1+m2), dtype=torch.float64))
        flux = SEOBNRv4PublishedFlux(
            spin_truncation=q.CHOICES["spin_truncation"], retain_pan_cubic_spin=True)
        valid = _array(trajectory.valid).astype(bool)
        raw_mode = np.zeros_like(mode)
        raw, _ = published_mode22_and_frequency(
            trajectory.state[trajectory.valid], hamiltonian, flux,
            spin_v6=q.CHOICES["spin_v6"], include_pan_v8=False,
            include_positive_chi_540=False, retain_pan_cubic_spin=True)
        raw_mode[valid] = _array(raw)
        systems = _system(result, targets)
        coefficients = result.attachment.fine.calibration.coefficients
        record = dict(**level, time=time_axis, mode=mode,
                      state=_array(trajectory.state),
                      frequency=_array(trajectory.frequency),
                      valid=valid, raw_mode=raw_mode, hashes=hashes,
                      peak_time_M=float(_array(trajectory.peak.time)),
                      match_time_M=float(_array(result.attachment.match_time)),
                      boundary_time_M=float(_array(result.attachment.boundary_time)),
                      restart_time_M=float(_array(trajectory.restart_time).reshape(-1)[0]),
                      nqc_coefficients=dict(
                          amplitude=_array(coefficients.amplitude).tolist(),
                          phase=_array(coefficients.phase).tolist()),
                      systems=systems,
                      elapsed_seconds=time.monotonic()-started)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    plan = json.loads(PLAN.read_text())
    prior_path = q.ROOT / plan["prior_convergence"]
    prior = json.loads(prior_path.read_text())
    choices = json.loads(CONVERGENCE_PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.inline-warmstart-spin-tolerance-plan.v1"
            or q.sha256(prior_path) != plan["prior_convergence_sha256"]
            or prior["status"] != "diagnostic_complete"
            or q.sha256(CONVERGENCE_PLAN) != prior["plan_sha256"]
            or plan["row_index"] != 1
            or plan["levels"] != ["ode_1", "ode_2"]
            or prior["results"][1]["numerical_check_passed"]
            or plan["phase_fit_window_M"] != 0.5
            or plan["phase_fit_degree"] != 6):
        parser.error("frozen spin-tolerance plan or input mismatch")
    from torchwave.approximants.eob.nqc import (
        seobnrv4_nqc_parameters, seobnrv4_nqc_targets,
    )

    row = prior["results"][1]["row"]
    parameters = [torch.tensor(row[key], dtype=torch.float64)
                  for key in ("mass1", "mass2", "chi1", "chi2")]
    with torch.no_grad():
        targets = seobnrv4_nqc_targets(*seobnrv4_nqc_parameters(*parameters))
    settings = q.Settings(
        relative_l2_limit=choices["gates"]["complex_relative_l2"],
        event_time_limit_M=choices["gates"]["event_time_difference_M"])
    frozen = prior["results"][1]["runs"]
    source_paths = (
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/eob/torch/warm_start.py",
        "dev/native_rom/qualify_generator.py",
    )
    report = dict(schema="diffgw.seobnrv4.inline-warmstart-spin-tolerance.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  prior_sha256=q.sha256(prior_path), plan_sha256=q.sha256(PLAN),
                  script_sha256=q.sha256(__file__),
                  frozen_source_sha256=prior["source_sha256"],
                  replay_source_sha256={path: q.sha256(q.ROOT/path)
                                        for path in source_paths},
                  row=row, levels=[], comparison=None, errors=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        runs = []
        for name in plan["levels"]:
            level = next(level for level in choices["ode_levels"] if level["name"] == name)
            run = _replay(row, level, frozen[name], choices, settings, targets)
            runs.append(run)
            print(json.dumps(dict(name=name, seconds=run["elapsed_seconds"],
                                  hashes_verified=True)), flush=True)
        coarse, fine = runs
        boundary = fine["boundary_time_M"]
        restart = fine["restart_time_M"]
        windows = (("coarse_prefix", 0., restart),
                   ("fine_early", restart, boundary-30.),
                   ("fine_late", boundary-30., boundary),
                   ("attachment", boundary, boundary+10.),
                   ("ringdown", boundary+10., float("inf")))
        comparison = dict(whole=q.compare_runs(coarse, fine),
                          regions=_regions(coarse, fine, windows),
                          pre_attachment=_inspiral(coarse, fine, boundary),
                          nqc_systems={name: _decompose(
                              coarse["systems"]["systems"][name],
                              fine["systems"]["systems"][name])
                              for name in ("amplitude", "phase")})
        if (abs(comparison["whole"]["complex_relative_l2"]
                - prior["results"][1]["ode"]["medium_to_fine"]
                ["complex_relative_l2"]) > 1e-12):
            raise ValueError("replay whole-waveform metric differs from frozen pair")
        report["comparison"] = comparison
        report["levels"] = [dict(name=run["name"], hashes=run["hashes"],
                                 elapsed_seconds=run["elapsed_seconds"],
                                 peak_time_M=run["peak_time_M"],
                                 match_time_M=run["match_time_M"],
                                 boundary_time_M=run["boundary_time_M"],
                                 restart_time_M=run["restart_time_M"],
                                 nqc_coefficients=run["nqc_coefficients"],
                                 nqc_condition={name: float(system["condition"])
                                                for name, system in run["systems"]["systems"].items()})
                             for run in runs]
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          evidence=str(args.json_out.resolve()))), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

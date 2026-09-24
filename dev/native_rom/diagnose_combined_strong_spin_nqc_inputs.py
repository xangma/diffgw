"""Attribute the frozen combined-candidate strong-spin NQC phase drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_nqc_inputs import _system
from .diagnose_nqc_matching_inputs import _decompose
from .diagnose_trajectory_sensitivity import _array, _hash
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive
from torchwave.approximants.eob.nqc import (
    seobnrv4_nqc_parameters, seobnrv4_nqc_targets,
)


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/combined_strong_spin_nqc_inputs_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.combined-strong-spin-nqc-inputs-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["levels"] != ["ode_1", "ode_2"]):
        parser.error("unexpected frozen plan")
    campaign_path = q.ROOT / plan["campaign"]
    trace_path = q.ROOT / plan["tolerance_trace"]
    if (q.sha256(campaign_path) != plan["campaign_sha256"]
            or q.sha256(trace_path) != plan["tolerance_trace_sha256"]):
        parser.error("frozen evidence hash differs")
    campaign = json.loads(campaign_path.read_text())
    trace = json.loads(trace_path.read_text())
    row = campaign["rows"][plan["row_name"]]
    if (campaign["status"] != "failed"
            or trace["status"] != "diagnostic_complete"
            or trace["comparison"]["whole"]["complex_relative_l2"]
            != row["check"]["medium_to_fine"]["complex_relative_l2"]):
        parser.error("frozen failure or replay is inconsistent")
    torch.set_num_threads(1)
    settings = q.Settings(**campaign["settings"])
    inputs = row["row"]
    parameters = [torch.tensor(inputs[key], dtype=torch.float64)
                  for key in ("mass1", "mass2", "chi1", "chi2")]
    targets = seobnrv4_nqc_targets(*seobnrv4_nqc_parameters(*parameters))
    source_paths = [
        "src/torchwave/approximants/eob/numerical_attachment.py",
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/eob/torch/historical_hamiltonian.py",
        "dev/native_rom/diagnose_boundary_strong_spin_nqc_inputs.py",
        "dev/native_rom/diagnose_nqc_matching_inputs.py",
    ]
    report = dict(
        schema="diffgw.seobnrv4.combined-strong-spin-nqc-inputs.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        campaign_sha256=q.sha256(campaign_path),
        tolerance_trace_sha256=q.sha256(trace_path),
        source_sha256={p: q.sha256(q.ROOT / p) for p in source_paths},
        row=inputs, runs=[], comparison=None, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        runs = []
        with torch.no_grad():
            for name in plan["levels"]:
                frozen = row["runs"][name]
                level = frozen["level"]
                started = time.monotonic()
                result = reconstruct_seobnrv4_22_adaptive(
                    torch.tensor(inputs["omega_start"], dtype=torch.float64),
                    **{key: inputs[key] for key in ("mass1", "mass2", "chi1", "chi2")},
                    **q.CHOICES,
                    **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
                    diagnostic_hermite_resampling=campaign["numerical_choice"]["hermite_resampling"],
                    diagnostic_local_nqc_derivatives=campaign["numerical_choice"]["local_nqc_phase_derivatives"],
                    diagnostic_matching_second_derivatives=True,
                    diagnostic_restart_time_M=campaign["numerical_choice"]["fixed_restart_time_M"],
                    diagnostic_coarse_initial_step_M=plan["fixed_coarse_first_step_M"],
                    diagnostic_warm_start_Momega=plan["warm_start_Momega"],
                    diagnostic_mode_33_higher_spin=campaign["candidate_choice"]["mode_33_higher_spin"],
                    diagnostic_hamiltonian_log_derivative=campaign["candidate_choice"]["hamiltonian_log_derivative"],
                    ringdown_duration=settings.ringdown_duration_M,
                    num_steps=settings.max_accepted_steps,
                    max_accepted_steps=settings.max_accepted_steps,
                    max_evaluations=settings.max_evaluations,
                    max_rejected_steps=settings.max_rejected_steps)
                hashes = dict(time=_hash(_array(result.time)), mode=_hash(_array(result.mode)))
                if hashes != frozen["native"]["array_sha256"]:
                    raise ValueError(f"{name} differs from frozen waveform")
                system = _system(result, targets)
                phase = system["phase"]
                runs.append(phase)
                report["runs"].append(dict(
                    name=name, level=level, array_sha256=hashes,
                    match_time_M=system["match_time_local_M"],
                    local_fit_samples=system["local_fit_samples"],
                    phase={key: value.tolist() if isinstance(value, np.ndarray) else value
                           for key, value in phase.items()},
                    elapsed_seconds=time.monotonic()-started))
                q.write_evidence(args.json_out, report)
                print(f"replay={name}", flush=True)
        report["comparison"] = _decompose(*runs)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

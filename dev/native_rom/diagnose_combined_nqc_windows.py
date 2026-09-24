"""Test local NQC phase-matrix window sensitivity on frozen strong-spin solves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_nqc_matching_inputs import _decompose
from .diagnose_trajectory_sensitivity import _array, _hash
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/combined_nqc_windows_plan.json"


def local_matrix(result, window):
    nqc = result.attachment.fine.nqc
    run = result.trajectory.fine
    count = int(run.output_valid.sum().detach())
    axis = run.output_time[:count]
    match = result.attachment.fine.match_time
    selected = (axis-match).detach().abs() <= window
    chart = (axis[selected]-match)/window
    if chart.numel() < 9 or not bool((chart.min() < 0) & (chart.max() > 0)):
        raise ValueError(f"window {window} lacks two-sided samples")
    design = torch.stack([chart.pow(k) for k in range(7)], -1)
    fit = torch.linalg.lstsq(
        design, nqc.sampled_phase_basis[selected], driver="gels").solution
    matrix = torch.stack((fit[1]/window, 2*fit[2]/window**2))
    return matrix, int(chart.numel())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.combined-nqc-windows-plan.v1"
            or plan["windows_M"] != [0.5, 1.0, 2.0]
            or plan["levels"] != ["ode_1", "ode_2"]):
        parser.error("unexpected frozen plan")
    campaign_path = q.ROOT / plan["campaign"]
    system_path = q.ROOT / plan["nqc_systems"]
    if (q.sha256(campaign_path) != plan["campaign_sha256"]
            or q.sha256(system_path) != plan["nqc_systems_sha256"]):
        parser.error("frozen evidence hash differs")
    campaign = json.loads(campaign_path.read_text())
    systems = json.loads(system_path.read_text())
    row = campaign["rows"]["strong_aligned_spin"]
    if systems["status"] != "diagnostic_complete":
        parser.error("NQC system attribution is incomplete")
    settings = q.Settings(**campaign["settings"])
    inputs = row["row"]
    torch.set_num_threads(1)
    source_paths = [
        "src/torchwave/approximants/eob/numerical_attachment.py",
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/eob/torch/historical_hamiltonian.py",
    ]
    report = dict(
        schema="diffgw.seobnrv4.combined-nqc-windows.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        campaign_sha256=q.sha256(campaign_path),
        nqc_systems_sha256=q.sha256(system_path),
        source_sha256={p: q.sha256(q.ROOT / p) for p in source_paths}, row=inputs,
        runs=[], comparison=None, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        for name, frozen_system in zip(plan["levels"], systems["runs"], strict=True):
            frozen = row["runs"][name]
            level = frozen["level"]
            with torch.no_grad():
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
                rhs = np.asarray(frozen_system["phase"]["rhs"])
                run = dict(name=name, array_sha256=hashes, windows={})
                for window in plan["windows_M"]:
                    matrix, samples = local_matrix(result, window)
                    matrix_np = _array(matrix)
                    coeff = np.linalg.solve(matrix_np, rhs)
                    if window == 0.5 and np.max(np.abs(
                            coeff-np.asarray(frozen_system["phase"]["coefficients"]))) > 1e-7:
                        raise ValueError(f"{name} default window does not recover coefficient")
                    run["windows"][str(window)] = dict(
                        matrix=matrix_np.tolist(), rhs=rhs.tolist(),
                        coefficients=coeff.tolist(), samples=samples,
                        condition=float(np.linalg.cond(matrix_np)))
                report["runs"].append(run)
                q.write_evidence(args.json_out, report)
                print(f"replay={name}", flush=True)
        report["comparison"] = {
            str(window): _decompose(*[
                {key: np.asarray(run["windows"][str(window)][key])
                 for key in ("matrix", "rhs", "coefficients")}
                for run in report["runs"]])
            for window in plan["windows_M"]}
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

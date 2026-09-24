"""Development-only DOP853 check of strong-spin NQC phase derivatives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
import torch

from . import qualify_generator as q
from .diagnose_nqc_dense_reference import five_point_matrix
from .eob.torch.adaptive_driver_clean import (
    _diagnostic_local_nqc_phase_matrix,
    reconstruct_seobnrv4_22_adaptive,
)
from .eob.torch.historical_hamiltonian import HistoricalLogDerivative
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.approximants.eob.nqc import nqc_22_basis
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from torchwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/strong_spin_nqc_dense_plan.json"


def array(value):
    return np.asarray(value.detach().cpu().numpy())


def max_absolute_difference(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.strong-spin-nqc-dense-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["level"] != "ode_2"
            or plan["native_window_M"] != 2.0
            or plan["comparison_windows_M"] != [0.5, 1.0, 2.0]):
        parser.error("unexpected frozen plan")
    campaign_path, window_path, sensitivity_path = (q.ROOT / plan[key]
        for key in ("campaign", "window_result", "window_sensitivity"))
    if (q.sha256(campaign_path) != plan["campaign_sha256"]
            or q.sha256(window_path) != plan["window_result_sha256"]
            or q.sha256(sensitivity_path) != plan["window_sensitivity_sha256"]):
        parser.error("frozen evidence changed")
    campaign, window = (json.loads(path.read_text())
                        for path in (campaign_path, window_path))
    sensitivity = json.loads(sensitivity_path.read_text())
    if (campaign["status"] != "failed"
            or window["status"] != "diagnostic_complete"
            or not window["check"]["numerical_check_passed"]):
        parser.error("frozen numerical evidence changed")
    row_report = campaign["rows"][plan["row_name"]]
    row = row_report["row"]
    level = row_report["runs"][plan["level"]]["level"]
    settings = q.Settings(**campaign["settings"])
    numerical, candidate = campaign["numerical_choice"], campaign["candidate_choice"]
    sources = {name: q.sha256(q.ROOT / name) for name in (
        "dev/native_rom/diagnose_nqc_dense_reference.py",
        "dev/native_rom/eob/torch/adaptive_driver_clean.py",
        "dev/native_rom/eob/torch/historical_hamiltonian.py",
        "src/torchwave/approximants/eob/integrator.py",
        "src/torchwave/approximants/eob/nqc.py",
        "src/torchwave/approximants/eob/seobnrv4_flux.py",
        "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
    )}
    report = dict(
        schema="diffgw.seobnrv4.strong-spin-nqc-dense.v1", status="running",
        release_qualified=False,
        scope=dict(**q.SCOPE, development_only=True, same_provisional_vector_field=True,
                   independent_integrator=True, independent_physics=False,
                   domain_qualified=False),
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        campaign_sha256=q.sha256(campaign_path), window_result_sha256=q.sha256(window_path),
        window_sensitivity_sha256=q.sha256(sensitivity_path),
        sources_before=sources, row=row, level=level, match_time_M=None,
        native={}, dop853=[], comparisons={}, errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        torch.set_num_threads(1)
        common = dict(
            diagnostic_hermite_resampling=numerical["hermite_resampling"],
            diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=numerical["fixed_restart_time_M"],
            diagnostic_coarse_initial_step_M=0.125,
            diagnostic_warm_start_Momega=0.03162277660168379,
            diagnostic_mode_33_higher_spin=candidate["mode_33_higher_spin"],
            diagnostic_hamiltonian_log_derivative=candidate["hamiltonian_log_derivative"],
            diagnostic_nqc_phase_window_M=plan["native_window_M"],
        )
        started = time.monotonic()
        with q.bounded_run(settings.max_run_seconds), torch.no_grad():
            result = reconstruct_seobnrv4_22_adaptive(
                torch.tensor(row["omega_start"], dtype=torch.float64),
                **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
                **q.CHOICES, **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
                **common, ringdown_duration=settings.ringdown_duration_M,
                num_steps=settings.max_accepted_steps,
                max_accepted_steps=settings.max_accepted_steps,
                max_rejected_steps=settings.max_rejected_steps,
                max_evaluations=settings.max_evaluations,
            )
        report["native"]["elapsed_seconds"] = time.monotonic() - started
        mode_hash = hashlib.sha256(np.ascontiguousarray(array(result.mode)).tobytes()).hexdigest()
        expected = window["candidate"][plan["level"]]["native"]["array_sha256"]["mode"]
        if mode_hash != expected:
            raise ValueError("native waveform no longer matches frozen candidate")
        fine = result.trajectory.fine
        time_grid = fine.output_time
        state_grid = fine.output_state
        match = float(result.attachment.fine.match_time)
        report["match_time_M"] = match
        if abs(match + float(result.trajectory.restart_time) - window["candidate"][
                plan["level"]]["native"]["match_time_M"]) > 1e-9:
            raise ValueError("native match time differs from frozen candidate")
        m1, m2 = (torch.tensor(row[key], dtype=torch.float64)
                  for key in ("mass1", "mass2"))
        x1, x2 = m1 / (m1 + m2), m2 / (m1 + m2)
        hamiltonian_type = (HistoricalLogDerivative
                            if candidate["hamiltonian_log_derivative"] == "lal_v4"
                            else SEOBNRv4Hamiltonian)
        ham = hamiltonian_type(x1*x2,
                               torch.tensor(row["chi1"], dtype=torch.float64),
                               torch.tensor(row["chi2"], dtype=torch.float64),
                               delta=x1-x2)
        flux = SEOBNRv4PublishedFlux(
            spin_truncation="generic-leading-ell8", retain_pan_cubic_spin=True,
            mode_33_higher_spin=candidate["mode_33_higher_spin"])
        field = EOBTrajectoryIntegrator(ham, flux).vector_field
        with torch.no_grad():
            orbital_frequency = field(state_grid)[..., 1]
            for window_M in plan["comparison_windows_M"]:
                matrix = _diagnostic_local_nqc_phase_matrix(
                    time_grid, state_grid, orbital_frequency, fine.output_time.new_tensor(match),
                    window_M)
                report["native"][f"window_{window_M:g}_matrix"] = array(matrix).tolist()
        frozen_windows = next(item["windows"] for item in sensitivity["runs"]
                              if item["name"] == plan["level"])
        for window_M in plan["comparison_windows_M"]:
            if max_absolute_difference(
                    report["native"][f"window_{window_M:g}_matrix"],
                    frozen_windows[str(window_M)]["matrix"]) > 1e-12:
                raise ValueError(f"native {window_M:g} M matrix changed from frozen evidence")
        embedded = array(result.attachment.fine.nqc.phase_matching_matrix)
        if max_absolute_difference(embedded, report["native"]["window_2_matrix"]) > 1e-12:
            raise ValueError("recomputed native NQC matrix differs from attached matrix")
        start_index = int(torch.argmin(abs(time_grid - (match - plan[
            "integration_start_before_match_M"]))).item())
        t0 = float(time_grid[start_index])
        t1 = match + plan["integration_end_after_match_M"]
        initial = array(state_grid[start_index])

        def scipy_field(_time, state):
            with torch.no_grad():
                return array(field(torch.from_numpy(state)))

        def phase_basis(state):
            tensor = torch.from_numpy(np.asarray(state))
            with torch.no_grad():
                frequency = field(tensor)[1]
                _, basis = nqc_22_basis(tensor[0], tensor[2], frequency)
            return array(basis)

        report["native"].update(mode_sha256=mode_hash, anchor_time_M=t0,
                                anchor_state=initial.tolist())
        q.write_evidence(args.json_out, report)
        for rtol, atol in plan["dop853_tolerances"]:
            started = time.monotonic()
            solution = solve_ivp(scipy_field, (t0, t1), initial,
                                 method="DOP853", rtol=rtol, atol=atol,
                                 dense_output=True)
            if not solution.success:
                raise RuntimeError(solution.message)
            matrices = {f"step_{step:g}": five_point_matrix(
                solution, match, step, phase_basis).tolist()
                for step in plan["five_point_steps_M"]}
            report["dop853"].append(dict(
                rtol=rtol, atol=atol, nfev=solution.nfev,
                elapsed_seconds=time.monotonic()-started,
                match_state=solution.sol(match).tolist(),
                match_basis=phase_basis(solution.sol(match)).tolist(),
                matrices=matrices))
            q.write_evidence(args.json_out, report)
        for step in plan["five_point_steps_M"]:
            name = f"step_{step:g}"
            matrices = [np.asarray(item["matrices"][name]) for item in report["dop853"]]
            report["comparisons"][name] = dict(
                dop853_tolerance_max_abs=max_absolute_difference(*matrices),
                native_window_max_abs={
                    str(window_M): [max_absolute_difference(
                        matrix, report["native"][f"window_{window_M:g}_matrix"])
                        for matrix in matrices]
                    for window_M in plan["comparison_windows_M"]},
            )
        report["sources_after"] = {name: q.sha256(q.ROOT / name) for name in sources}
        if report["sources_after"] != sources:
            raise RuntimeError("source changed during diagnostic")
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          comparisons=report["comparisons"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

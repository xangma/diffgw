"""Isolate the local versus full-spline NQC phase-matrix choice on native inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch

from . import qualify_generator as q


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/nqc_phase_matrix_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.nqc-phase-matrix-plan.v1"
            or plan["rows"] != ["strong_aligned_spin", "strong_antialigned_spin"]):
        parser.error("frozen phase-matrix plan differs")
    coefficient_path = q.ROOT / plan["coefficient_evidence"]
    if q.sha256(coefficient_path) != plan["coefficient_evidence_sha256"]:
        parser.error("coefficient evidence differs")
    coefficients = json.loads(coefficient_path.read_text())
    if (coefficients["status"] != "diagnostic_complete"
            or [row["name"] for row in coefficients["rows"]] != plan["rows"]):
        parser.error("coefficient prerequisites failed")
    waveform_plan = json.loads((q.ROOT / plan["waveform_plan"]).read_text())
    if [row["name"] for row in waveform_plan["rows"]] != plan["rows"]:
        parser.error("waveform plan differs")
    binary = coefficients["lal_binary"]
    if q.lal_binary_identity(binary["lal_version"], binary["lalsimulation_version"]) != binary:
        parser.error("public binary identity differs")

    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.nqc-phase-matrix.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        coefficient_evidence_sha256=q.sha256(coefficient_path),
        source_sha256={path: q.sha256(q.ROOT / path) for path in (
            "dev/native_rom/qualify_generator.py",
            "dev/native_rom/eob/torch/adaptive_driver_clean.py",
            "src/torchwave/approximants/eob/nqc.py",
            "src/torchwave/approximants/eob/numerical_attachment.py",
        )}, lal_binary=binary, rows=[], errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        for item, prior in zip(waveform_plan["rows"], coefficients["rows"]):
            waveform_path = q.ROOT / item["waveform"]
            if q.sha256(waveform_path) != item["waveform_sha256"]:
                raise ValueError(f"{item['name']} waveform evidence differs")
            waveform = json.loads(waveform_path.read_text())
            level = waveform["candidate"][waveform_plan["level"]]["level"]
            numerical = waveform["choice"]["numerical"]
            physical = waveform["choice"]["physical"]
            flux = waveform["choice"]["flux_placements"]
            run = q.native_run(
                waveform["row"], level, q.Settings(**waveform["settings"]),
                diagnostic_hermite_resampling=numerical["hermite_resampling"],
                diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
                diagnostic_matching_second_derivatives=True,
                diagnostic_restart_time_M=(
                    numerical["fixed_restart_time_M"] if item["name"] == "strong_aligned_spin"
                    else waveform["choice"]["fixed_restart_time_M"]),
                diagnostic_coarse_initial_step_M=0.125,
                diagnostic_warm_start_Momega=0.03162277660168379,
                diagnostic_nqc_phase_window_M=item["nqc_phase_window_M"],
                diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
                diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
                diagnostic_mode_21_higher_spin=flux["mode_21_higher_spin"],
                diagnostic_mode_31_higher_spin=flux["mode_31_higher_spin"],
                diagnostic_mode_32_spin_square=flux["mode_32_spin_square"],
                return_nqc_coefficients=True,
            )
            replay = q.summarize_run(run)["array_sha256"]
            if replay != prior["native_waveform_sha256"]:
                raise ValueError(f"{item['name']} native waveform no longer reproduces")
            native = np.asarray(run["nqc_coefficients"]["phase"])
            public = np.asarray(prior["public_coefficients"][3:])
            if not np.allclose(native, prior["native_coefficients"][3:], rtol=0, atol=1e-10):
                raise ValueError(f"{item['name']} native coefficients no longer reproduce")
            local_matrix = np.asarray(run["nqc_matching_matrices"]["phase"])
            if not np.allclose(local_matrix,
                               prior["target_counterfactual"]["native_matching_matrices"]["phase"],
                               rtol=0, atol=1e-10):
                raise ValueError(f"{item['name']} native phase matrix no longer reproduces")
            full_spline_matrix = np.asarray(run["nqc_full_spline_phase_matrix"])
            rhs = local_matrix @ native
            full_spline = np.linalg.solve(full_spline_matrix, rhs)
            report["rows"].append(dict(
                name=item["name"], waveform_sha256=replay,
                local_phase_matrix=local_matrix.tolist(),
                full_spline_phase_matrix=full_spline_matrix.tolist(),
                phase_rhs=rhs.tolist(),
                local_phase_coefficients=native.tolist(),
                full_spline_phase_coefficients=full_spline.tolist(),
                public_phase_coefficients=public.tolist(),
                local_minus_public=(native - public).tolist(),
                full_spline_minus_public=(full_spline - public).tolist(),
                local_public_gap_l2=float(np.linalg.norm(native - public)),
                full_spline_public_gap_l2=float(np.linalg.norm(full_spline - public)),
                local_matrix_condition=float(np.linalg.cond(local_matrix)),
                full_spline_matrix_condition=float(np.linalg.cond(full_spline_matrix)),
            ))
            q.write_evidence(args.json_out, report)
            print(f"row={item['name']}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

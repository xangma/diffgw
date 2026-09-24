"""Measure NQC coefficient output-step sensitivity on both prototype and public runs."""

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
from .diagnose_public_nqc_mode_boundary import public_call


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/native_nqc_output_grid_plan.json"
SCOPE = {
    **q.SCOPE,
    "description": "Native and pinned public NQC output-step comparison; no training or promotion.",
    "known_limitations": [
        limitation for limitation in q.SCOPE["known_limitations"]
        if not limitation.startswith("No time, phase, amplitude, or coefficient fitting")
    ] + [
        "Both generators solve their prescribed NQC matching equations; no coefficients "
        "are tuned to make the generators agree.",
        "The native fixed-restart control does not reproduce the public coarse-step "
        "evolution or establish same-input NQC parity.",
    ],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.native-nqc-output-grid-plan.v1"
            or plan["rows"] != ["strong_aligned_spin", "strong_antialigned_spin"]
            or plan["output_steps_M"] != [0.125, 0.0625, 0.03125, 0.015625]
            or plan["expected_native_internal_step_M"] != 0.015625):
        parser.error("frozen output-grid plan differs")
    coefficient_path = q.ROOT / plan["coefficient_evidence"]
    if q.sha256(coefficient_path) != plan["coefficient_evidence_sha256"]:
        parser.error("frozen coefficient evidence differs")
    coefficients = json.loads(coefficient_path.read_text())
    waveform_plan = json.loads((q.ROOT / plan["waveform_plan"]).read_text())
    if (coefficients["status"] != "diagnostic_complete"
            or [row["name"] for row in coefficients["rows"]] != plan["rows"]
            or [row["name"] for row in waveform_plan["rows"]] != plan["rows"]):
        parser.error("frozen prerequisites differ")
    binary = coefficients["lal_binary"]
    if q.lal_binary_identity(binary["lal_version"], binary["lalsimulation_version"]) != binary:
        parser.error("pinned public binary differs")
    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.native-nqc-output-grid.v1", scope=SCOPE,
        status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), lal_binary=binary, rows=[], errors=[],
        source_sha256={path: q.sha256(q.ROOT / path) for path in (
            "dev/native_rom/qualify_generator.py",
            "dev/native_rom/eob/torch/adaptive_driver_clean.py",
            "src/torchwave/approximants/eob/nqc.py",
            "src/torchwave/approximants/eob/numerical_attachment.py",
        )},
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        for item, prior in zip(waveform_plan["rows"], coefficients["rows"]):
            waveform_path = q.ROOT / item["waveform"]
            if q.sha256(waveform_path) != item["waveform_sha256"]:
                raise ValueError(f"{item['name']} frozen waveform differs")
            waveform = json.loads(waveform_path.read_text())
            level = waveform["candidate"][waveform_plan["level"]]["level"]
            numerical = waveform["choice"]["numerical"]
            physical = waveform["choice"]["physical"]
            flux = waveform["choice"]["flux_placements"]
            baseline_native = np.asarray(prior["native_coefficients"])
            baseline_public = np.asarray(prior["public_coefficients"])
            row = dict(name=item["name"], row=prior["row"], grids=[])
            report["rows"].append(row)
            for step in plan["output_steps_M"]:
                run = q.native_run(
                    waveform["row"], {**level, "dt_geom": step},
                    q.Settings(**waveform["settings"]),
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
                if run["internal_step_M"] != plan["expected_native_internal_step_M"]:
                    raise ValueError(f"{item['name']} native internal step changed")
                native = np.asarray(run["nqc_coefficients"]["amplitude"]
                                    + run["nqc_coefficients"]["phase"])
                output, public_all = public_call(prior["row"], step, 1)
                if list(output) != [None, None, None]:
                    raise ValueError(f"{item['name']} coefficient-only public call differs")
                public = public_all[[0, 1, 2, 6, 7]]
                if step == plan["output_steps_M"][0]:
                    replay = q.summarize_run(run)["array_sha256"]
                    if (replay != prior["native_waveform_sha256"]
                            or not np.array_equal(native, baseline_native)
                            or not np.array_equal(public, baseline_public)):
                        raise ValueError(f"{item['name']} frozen baseline does not reproduce")
                row["grids"].append(dict(
                    output_step_M=step, native_coefficients=native.tolist(),
                    public_coefficients=public.tolist(),
                    native_over_baseline_relative=(native / baseline_native - 1).tolist(),
                    public_over_baseline_relative=(public / baseline_public - 1).tolist(),
                    native_over_public_relative=(native / public - 1).tolist(),
                    native_peak_time_M=run["peak_time_M"],
                    native_match_time_M=run["match_time_M"],
                    native_restart_time_M=run["restart_time_M"],
                    native_internal_step_M=run["internal_step_M"],
                ))
                q.write_evidence(args.json_out, report)
                print(f"row={item['name']} output_step_M={step}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

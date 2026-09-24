"""Compare native NQC fit with pinned source and wheel on one frozen row."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/native_source_nqc_plan.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {key: q.ROOT / plan[key] for key in
             ("native_waveform", "source_wheel_mode", "prior_waveform")}
    if (plan["schema"] != "diffgw.seobnrv4.native-source-nqc-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["native_level"] != "ode_2"
            or any(q.sha256(paths[key]) != plan[key+"_sha256"] for key in paths)):
        parser.error("Frozen plan or prerequisites changed")
    native = json.loads(paths["native_waveform"].read_text())
    source = json.loads(paths["source_wheel_mode"].read_text())
    prior = json.loads(paths["prior_waveform"].read_text())
    source_row = next(row for row in source["rows"] if row["name"] == plan["row_name"])
    if (native["status"] != "diagnostic_complete"
            or source["status"] != "diagnostic_complete"
            or native["row"] != source_row["row"]
            or native["row"] != prior["row"]
            or native["choice"]["candidate_changes"] != {
                "spin_truncation": "tabulated", "mode_43_higher_spin": "lal_v4_rho"}):
        parser.error("Prerequisite row or candidate changed")
    row = native["row"]
    settings = q.Settings(**native["settings"])
    numerical = native["choice"]["numerical"]
    physical = native["choice"]["physical"]
    placements = native["choice"]["baseline_flux_placements"]
    level = native["candidate"]["ode_2"]["level"]
    torch.set_num_threads(1)
    report = dict(schema="diffgw.seobnrv4.native-source-nqc.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={key: q.sha256(path) for key, path in paths.items()},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv),
                  row_name=plan["row_name"], level=level, comparison=None, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        start = time.monotonic()
        run = q.native_run(
            row, level, settings,
            diagnostic_hermite_resampling=numerical["hermite_resampling"],
            diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=numerical["fixed_restart_time_M"],
            diagnostic_coarse_initial_step_M=0.125,
            diagnostic_warm_start_Momega=0.03162277660168379,
            diagnostic_nqc_phase_window_M=prior["candidate_window_M"],
            diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
            diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
            diagnostic_mode_21_higher_spin=placements["mode_21_higher_spin"],
            diagnostic_mode_31_higher_spin=placements["mode_31_higher_spin"],
            diagnostic_mode_32_spin_square=placements["mode_32_spin_square"],
            diagnostic_spin_truncation="tabulated",
            diagnostic_mode_43_higher_spin="lal_v4_rho",
            return_nqc_coefficients=True)
        summary = q.summarize_run(run)
        expected = native["candidate"]["ode_2"]["native"]["array_sha256"]
        if summary["array_sha256"] != expected:
            raise ValueError("Native waveform failed frozen replay")
        coefficients = run["nqc_coefficients"]
        native_active = np.asarray(coefficients["amplitude"]
                                   + coefficients["phase"], dtype=np.float64)
        source_active = np.asarray(source_row["source_nqc_coefficients"])[[0, 1, 2, 6, 7]]
        wheel_active = np.asarray(source_row["wheel_nqc_coefficients"])[[0, 1, 2, 6, 7]]
        if native_active.shape != (5,) or not np.isfinite(native_active).all():
            raise ValueError("Native NQC coefficients invalid")
        report["comparison"] = dict(
            elapsed_seconds=time.monotonic()-start,
            native_waveform_sha256=summary["array_sha256"],
            native_active=native_active.tolist(),
            source_active=source_active.tolist(),
            wheel_active=wheel_active.tolist(),
            native_minus_source=(native_active-source_active).tolist(),
            native_minus_wheel=(native_active-wheel_active).tolist(),
            source_minus_wheel=(source_active-wheel_active).tolist(),
            active_l2=dict(native_source=float(np.linalg.norm(native_active-source_active)),
                           native_wheel=float(np.linalg.norm(native_active-wheel_active)),
                           source_wheel=float(np.linalg.norm(source_active-wheel_active))))
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

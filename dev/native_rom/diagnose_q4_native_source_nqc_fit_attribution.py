"""Dev-only q4 source/native NQC input and fitting-rule attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_native_source_nqc_inputs import _fit, _native_samples
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_source_flux_initial import _libraries
from .diagnose_source_q4_ultrafine_chart import _native
from torchwave.approximants.eob import numerical_attachment as attachment


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/q4_native_source_nqc_fit_attribution_plan.json"


def _systems(fit):
    return (np.asarray(fit["amplitude_matrix"]), np.asarray(fit["amplitude_rhs"]),
            np.asarray(fit["phase_matrix"]), np.asarray(fit["phase_rhs"]))


def _crossed(native_fit, source_fit):
    na, nb, np_, nq = _systems(native_fit)
    sa, sb, sp, sq = _systems(source_fit)
    stages = []
    for name, matrix_n, rhs_n, matrix_s, rhs_s in (
            ("amplitude", na, nb, sa, sb),
            ("phase", np_, nq, sp, sq)):
        native = np.linalg.solve(matrix_n, rhs_n)
        source = np.linalg.solve(matrix_s, rhs_s)
        matrix_first = np.linalg.solve(matrix_s, rhs_n)
        rhs_first = np.linalg.solve(matrix_n, rhs_s)
        stages.append({
            "system": name,
            "matrix_max_abs_difference": float(np.max(abs(matrix_n-matrix_s))),
            "rhs_max_abs_difference": float(np.max(abs(rhs_n-rhs_s))),
            "native_to_source_coefficients_l2": float(np.linalg.norm(source-native)),
            "matrix_first_change_l2": float(np.linalg.norm(matrix_first-native)),
            "rhs_after_matrix_change_l2": float(np.linalg.norm(source-matrix_first)),
            "rhs_first_change_l2": float(np.linalg.norm(rhs_first-native)),
            "matrix_after_rhs_change_l2": float(np.linalg.norm(source-rhs_first)),
            "native_condition": float(np.linalg.cond(matrix_n)),
            "source_condition": float(np.linalg.cond(matrix_s)),
        })
    return stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    keys = ("source_inputs", "prior_intervention", "source_flux_plan", "settings")
    paths = {key: q.ROOT / plan[key] for key in keys}
    if (plan["schema"] != "diffgw.seobnrv4.q4-native-source-nqc-fit-attribution-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row_name"] != "q4_nonspinning"
            or plan["output_step_M"] != 0.015625
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in paths.items())):
        parser.error("Frozen plan or prerequisites changed")
    source = json.loads(paths["source_inputs"].read_text())
    prior = json.loads(paths["prior_intervention"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    key = str(plan["output_step_M"])
    if (source["status"] != "diagnostic_complete"
            or prior["status"] != "diagnostic_complete"
            or source["row"] != prior["row"]
            or np.max(abs(np.asarray(source["grids"][key]["source_active"])
                          - np.asarray(prior["grids"][key]["source_coefficients"]))) > 1e-12):
        parser.error("Pinned source NQC row or coefficients changed")
    report = {
        "schema": "diffgw.seobnrv4.q4-native-source-nqc-fit-attribution.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in paths.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": prior["row"], "output_step_M": plan["output_step_M"],
        "comparison": None, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        torch.set_num_threads(1)
        start = time.monotonic()
        targets_seen = []
        original = attachment._full_series_nqc

        def capture(*values, **kwargs):
            targets_seen.append([float(item.detach().cpu()) for item in values[8]])
            return original(*values, **kwargs)

        with patch.object(attachment, "_full_series_nqc", capture):
            run = _native(prior["row"], plan["output_step_M"],
                          {"atol": plan["atol"], "rtol": plan["rtol"]},
                          q.Settings(**json.loads(paths["settings"].read_text())["settings"]),
                          return_nqc_coefficients=True, return_nqc_samples=True)
        summary = q.summarize_run(run)
        if (summary["array_sha256"]
                != prior["grids"][key]["native_baseline"]["array_sha256"]
                or len(targets_seen) != 2):
            raise ValueError("Native waveform or NQC call count changed")
        native_time, native_values, native_match = _native_samples(run["nqc_samples"])
        source_targets = np.asarray(source["grids"][key]["targets"])
        native_targets = np.asarray(targets_seen[-1])
        source_fit = source["grids"][key]["python_gsl_fit"]
        active_source = np.asarray(source["grids"][key]["source_active"])
        if np.linalg.norm(np.asarray(source_fit["coefficients"])-active_source) > 1e-6:
            raise ValueError("Pinned-source GSL fit replay changed")
        gsl_path = _libraries(source_plan["pinned_wheel"])["libgsl.28.dylib"]
        native_fit = _fit(GSLCubic(gsl_path), native_time, *native_values,
                          native_match, source_targets)
        native_actual = np.asarray(run["nqc_coefficients"]["amplitude"]
                                   + run["nqc_coefficients"]["phase"])
        if np.max(abs(native_actual - prior["grids"][key]["native_coefficients"])) > 1e-12:
            raise ValueError("Native active NQC coefficients changed")
        amplitude_matrix, amplitude_rhs, phase_matrix, phase_rhs = _systems(native_fit)
        target_difference = native_targets - source_targets
        native_gsl_native_targets = np.concatenate((
            np.linalg.solve(amplitude_matrix, amplitude_rhs + target_difference[:3]),
            np.linalg.solve(phase_matrix, phase_rhs + target_difference[3:])))
        report["comparison"] = {
            "elapsed_seconds": time.monotonic() - start,
            "native_mode_sha256": summary["array_sha256"],
            "native_nqc_samples": len(native_time),
            "native_nqc_sample_sha256": hashlib.sha256(native_values.tobytes()).hexdigest(),
            "native_nqc_match_time_M": native_match,
            "source_nqc_match_time_M": source["grids"][key]["match_time_M"],
            "native_targets": native_targets.tolist(),
            "source_targets": source_targets.tolist(),
            "target_max_abs_difference": float(np.max(abs(native_targets-source_targets))),
            "native_active": native_actual.tolist(),
            "native_gsl_with_source_targets": native_fit,
            "native_gsl_with_native_targets": native_gsl_native_targets.tolist(),
            "source_gsl_with_source_targets": source_fit,
            "source_active": active_source.tolist(),
            "native_gsl_target_change_l2": float(np.linalg.norm(
                native_gsl_native_targets - np.asarray(native_fit["coefficients"]))),
            "native_gsl_native_targets_minus_native_active_l2": float(np.linalg.norm(
                native_gsl_native_targets - native_actual)),
            "native_gsl_minus_native_active_l2": float(np.linalg.norm(
                np.asarray(native_fit["coefficients"])-native_actual)),
            "native_gsl_minus_source_gsl_l2": float(np.linalg.norm(
                np.asarray(native_fit["coefficients"])
                - np.asarray(source_fit["coefficients"]))),
            "crossed_systems": _crossed(native_fit, source_fit),
        }
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

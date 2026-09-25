"""Retrospective strong-aligned NQC field-stencil continuation.

This reuses one previously captured trajectory, match event, target vector,
and independently integrated field stencil set. It is not a fresh holdout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

from . import qualify_generator as q
from .diagnose_source_aligned_derivative_only_pilot import _field_derivatives, _solve


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/source_aligned_finer_stencils_plan.json"


def _difference(first: np.ndarray, second: np.ndarray) -> dict:
    return {
        "coefficient_l2": float(np.linalg.norm(first - second)),
        "coefficient_linf": float(np.max(abs(first - second))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    source_path = q.ROOT / plan["dense_field_evidence"]
    prior_path = q.ROOT / plan["prior_pilot"]
    prior_script = q.ROOT / plan["prior_solver"]
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-finer-stencils-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row_name"] != "strong_aligned_spin"
            or plan["step_sequence_M"] != [0.2, 0.1, 0.05, 0.025]
            or plan["new_steps_M"] != [0.05, 0.025]
            or plan["prior_failed_gate"] != {"pair_M": [0.2, 0.1],
                                             "coefficient_l2_max": 0.01}
            or plan["historical_replay_linf_max"] != 0.001
            or plan["linear_residual_linf_max"] != 1e-12
            or q.sha256(source_path) != plan["dense_field_evidence_sha256"]
            or q.sha256(prior_path) != plan["prior_pilot_sha256"]
            or q.sha256(prior_script) != plan["prior_solver_sha256"]
            or q.sha256(__file__) != plan["script_sha256"]):
        parser.error("Frozen retrospective plan or source evidence changed")
    source = json.loads(source_path.read_text())
    prior = json.loads(prior_path.read_text())
    if (source["status"] != "diagnostic_complete"
            or prior["status"] != "diagnostic_complete"
            or source["row"] != plan["row"]
            or prior["row"] != plan["row"]
            or source["resample_factor"] != 32
            or source["source"]["meta"][1] != plan["internal_step_M"]
            or source["source"]["match_time_M"] != plan["match_time_M"]
            or prior["gates"]["coefficient_stencil"] is not False
            or prior["stencil_coefficient_l2"] <= 0.01
            or len(source["centered_dop853"]) != 2
            or [[x["rtol"], x["atol"]] for x in source["centered_dop853"]]
            != plan["field_tolerances"]):
        parser.error("Pinned field capture or failed pilot changed")
    captured = np.asarray(source["source"]["captured_gsl_derivatives"], dtype=np.float64)
    targets = np.asarray(source["source"]["meta"][3:], dtype=np.float64)
    active = np.asarray(source["source"]["active_nqc_coefficients"], dtype=np.float64)
    if captured.shape != (7, 3) or targets.shape != (5,) or active.shape != (5,):
        parser.error("Unexpected NQC capture dimensions")
    report = {
        "schema": "diffgw.seobnrv4.source-aligned-finer-stencils.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {"dense_field_evidence": q.sha256(source_path),
                         "prior_pilot": q.sha256(prior_path),
                         "prior_solver": q.sha256(prior_script)},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": source["row"], "resample_factor": 32,
        "fixed": {"trajectory_sha256": source["source"]["input_array_sha256"],
                  "match_time_M": source["source"]["match_time_M"],
                  "internal_step_M": source["source"]["meta"][1],
                  "targets": targets.tolist(),
                  "raw_values": "captured at historical match",
                  "ringdown": "not evaluated or modified"},
        "prior_failed_gate": {"pair_M": [0.2, 0.1],
                              "coefficient_l2": prior["stencil_coefficient_l2"],
                              "limit": 0.01,
                              "passed": prior["gates"]["coefficient_stencil"]},
        "baseline": {}, "steps": {}, "adjacent": {}, "order_ratios": {},
        "checks": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        baseline = _solve(captured, targets)
        replay = float(np.max(abs(np.asarray(baseline["coefficients"]) - active)))
        report["baseline"] = {"historical_gsl_fit": baseline,
                              "historical_active": active.tolist(),
                              "replay_linf": replay}
        tight, loose = source["centered_dop853"][1], source["centered_dop853"][0]
        arrays = {}
        for step in plan["step_sequence_M"]:
            key = f"{step:g}"
            tight_data = _field_derivatives(captured, tight["stencils"][key])
            loose_data = _field_derivatives(captured, loose["stencils"][key])
            tight_fit = _solve(tight_data, targets)
            loose_fit = _solve(loose_data, targets)
            tight_coeff = np.asarray(tight_fit["coefficients"])
            loose_coeff = np.asarray(loose_fit["coefficients"])
            arrays[key] = (tight_data, tight_coeff)
            report["steps"][key] = {
                "tight_derivatives_value_first_second": tight_data.tolist(),
                "tight_fit": tight_fit,
                "loose_fit_coefficients": loose_coeff.tolist(),
                "tight_minus_loose_derivative_linf": float(np.max(abs(
                    tight_data[:, 1:] - loose_data[:, 1:]))),
                "tight_minus_loose_coefficient_l2": float(np.linalg.norm(
                    tight_coeff - loose_coeff)),
                "tight_minus_historical_coefficient_l2": float(np.linalg.norm(
                    tight_coeff - active)),
            }
        pair_names = []
        for coarse, fine in zip(plan["step_sequence_M"], plan["step_sequence_M"][1:]):
            coarse_key, fine_key = f"{coarse:g}", f"{fine:g}"
            coarse_data, coarse_coeff = arrays[coarse_key]
            fine_data, fine_coeff = arrays[fine_key]
            pair_name = f"{coarse_key}_to_{fine_key}"
            pair_names.append(pair_name)
            report["adjacent"][pair_name] = {
                **_difference(coarse_coeff, fine_coeff),
                "derivative_linf": float(np.max(abs(coarse_data[:, 1:]
                                                      - fine_data[:, 1:]))),
            }
        for previous, current in zip(pair_names, pair_names[1:]):
            item = {}
            for metric in ("coefficient_l2", "derivative_linf"):
                numerator = report["adjacent"][previous][metric]
                denominator = report["adjacent"][current][metric]
                ratio = numerator / denominator if denominator > 0 else float("inf")
                item[metric + "_coarse_over_fine"] = ratio
                item[metric + "_observed_order_log2"] = float(np.log2(ratio))
            report["order_ratios"][previous + "__" + current] = item
        prior_pair = report["adjacent"]["0.2_to_0.1"]["coefficient_l2"]
        prior_step_error = max(float(np.max(abs(
                np.asarray(report["steps"][key]["tight_fit"]["coefficients"])
                - np.asarray(prior["field"][key]["all_rows"]["coefficients"]))))
            for key in ("0.2", "0.1"))
        residual = max(
            fit[metric]
            for fit in (baseline, *(report["steps"][f"{step:g}"]["tight_fit"]
                                   for step in plan["step_sequence_M"]))
            for metric in ("amplitude_residual_linf", "phase_residual_linf")
        )
        report["checks"] = {
            "historical_replay_linf": replay,
            "historical_replay_passed": replay <= plan["historical_replay_linf_max"],
            "prior_pair_replay_abs": abs(prior_pair - prior["stencil_coefficient_l2"]),
            "prior_step_coefficient_replay_linf": prior_step_error,
            "prior_replay_passed": max(abs(prior_pair - prior["stencil_coefficient_l2"]),
                                       prior_step_error) <= plan["prior_replay_linf_max"],
            "max_linear_residual_linf": residual,
            "linear_residual_passed": residual <= plan["linear_residual_linf_max"],
            "new_pairs_monotone_decreasing_coefficient_l2": (
                report["adjacent"]["0.1_to_0.05"]["coefficient_l2"]
                > report["adjacent"]["0.05_to_0.025"]["coefficient_l2"]),
            "new_pairs_monotone_decreasing_derivative_linf": (
                report["adjacent"]["0.1_to_0.05"]["derivative_linf"]
                > report["adjacent"]["0.05_to_0.025"]["derivative_linf"]),
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

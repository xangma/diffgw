"""Apply one source-amplitude algebraic proxy to frozen NQC stencil systems."""

from __future__ import annotations

import argparse
import ctypes
from decimal import Decimal, localcontext
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np

from . import qualify_generator as q
from .diagnose_source_aligned_derivative_only_pilot import _solve
from .diagnose_source_aligned_raw_amplitude_curvature import _key
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot


PLAN = (q.ROOT / "dev/native_rom/evidence/2026-09-25/"
        "source_aligned_exact_circ_nqc_counterfactual_plan.json")
ANCHOR = "  out[2] = Hreal;\n"
WEIGHTS_FIRST = (1, -8, 0, 8, -1)
WEIGHTS_SECOND = (-1, 16, -30, 16, -1)


def _stencil_delta(values: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    with localcontext() as context:
        context.prec = 80
        first = []
        second = []
        for col in range(values.shape[1]):
            column = [Decimal.from_float(float(value)) for value in values[:, col]]
            first.append(float(sum(Decimal(weight) * value for weight, value
                                   in zip(WEIGHTS_FIRST, column, strict=True))
                               / (Decimal(12) * Decimal.from_float(step))))
            second.append(float(sum(Decimal(weight) * value for weight, value
                                    in zip(WEIGHTS_SECOND, column, strict=True))
                                / (Decimal(12) * Decimal.from_float(step) ** 2)))
    return np.asarray(first), np.asarray(second)


def _delta_amplitude(amplitude: float, gsl: float, exact: float) -> float:
    return amplitude * math.expm1((4 / 3) * math.log1p((gsl - exact) / exact))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()

    plan = json.loads(PLAN.read_text())
    names = ("source_nqc_evidence", "pilot_evidence", "curvature_evidence",
             "operation_evidence", "source_flux_plan", "mode_adapter", "probe_include")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-exact-circ-nqc-counterfactual-plan.v1"
            or plan["release_qualified"] is not False
            or plan["event_M"] != 137.59240346159578
            or plan["row"] != {"mass1": 30.0, "mass2": 20.0, "chi1": 0.8,
                               "chi2": 0.8, "omega_start": 0.04}
            or plan["stencil_steps_M"] != [0.1, 0.2]
            or plan["sample_offsets_M"] != [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]
            or plan["sample_integrator"] != "bounded_dense_tight"
            or plan["historical_coefficient_stencil_l2_limit"] != 0.01
            or any(q.sha256(path) != plan[name + "_sha256"]
                   for name, path in paths.items())):
        parser.error("Frozen counterfactual plan or inputs changed")

    source = json.loads(paths["source_nqc_evidence"].read_text())
    pilot = json.loads(paths["pilot_evidence"].read_text())
    curvature = json.loads(paths["curvature_evidence"].read_text())
    operation = json.loads(paths["operation_evidence"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (source["status"] != "diagnostic_complete"
            or pilot["status"] != "diagnostic_complete"
            or curvature["status"] != "diagnostic_complete"
            or operation["status"] != "diagnostic_complete"
            or source["source"]["match_time_M"] != plan["event_M"]
            or pilot["match_time_M"] != plan["event_M"]
            or curvature["event_M"] != plan["event_M"]
            or operation["event_M"] != plan["event_M"]
            or pilot["row"] != plan["row"]
            or curvature["row"] != plan["row"]
            or operation["row"] != plan["row"]
            or pilot["gates"]["coefficient_stencil"] is not False
            or pilot["stencil_coefficient_l2"] <= 0.01
            or curvature["historical_0p01_gate"] is not False
            or operation["historical_0p01_gate"] is not False):
        parser.error("Frozen event or historical failed gate changed")

    report = {
        "schema": "diffgw.seobnrv4.source-aligned-exact-circ-nqc-counterfactual.v1",
        "scope": dict(**q.SCOPE, development_only=True,
                      post_hoc_algebraic_counterfactual=True),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": plan["row"], "event_M": plan["event_M"],
        "source_pin": source_plan["pinned_source"],
        "wheel_pin": source_plan["pinned_wheel"],
        "historical": {"coefficients": {str(step): pilot["field"][str(step)]
                                       ["all_rows"]["coefficients"]
                                        for step in plan["stencil_steps_M"]},
                       "coefficient_stencil_l2": pilot["stencil_coefficient_l2"],
                       "coefficient_stencil_gate": False},
        "samples": {}, "counterfactual": {}, "checks": {},
        "limitations": plan["limitation"], "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        samples = curvature["integrators"][plan["sample_integrator"]]["samples"]
        old_samples = operation["samples"][plan["sample_integrator"]]
        for offset in (-0.2, -0.1, 0.0, 0.1, 0.2):
            key = _key(offset)
            if (old_samples[key]["amplitude"] != samples[key]["amplitude"]
                    or old_samples[key]["saved_amplitude_replay_error"] != 0):
                raise ValueError("Saved amplitude operation replay changed")

        libraries = _libraries(source_plan["pinned_wheel"])
        with tempfile.TemporaryDirectory(prefix="diffgw-source-exact-circ-nqc-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            original = paths["mode_adapter"].read_text()
            if original.count(ANCHOR) != 1:
                raise ValueError("Pinned dev adapter insertion anchor changed")
            adapter = Path(tmp) / "mode_probe.c"
            adapter.write_text(original.replace(
                ANCHOR, ANCHOR + paths["probe_include"].read_text()))
            dylib = Path(tmp) / "mode_probe.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     adapter, dylib, source_plan["flux_source_includes"],
                     source_plan["flux_public_includes"])
            function = ctypes.CDLL(str(dylib)).diffgw_v4_mode22_same_state
            function.argtypes = [ctypes.c_double] * 9 + [ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            row = plan["row"]
            x1 = row["mass1"] / (row["mass1"] + row["mass2"])

            def evaluate(sample: dict) -> dict:
                output = (ctypes.c_double * 47)()
                rc = function(x1, 1 - x1, row["chi1"], row["chi2"],
                              *sample["state"], sample["omega"], output)
                if rc:
                    raise RuntimeError(f"Pinned raw mode probe failed: {rc}")
                if not all(math.isfinite(value) for value in output):
                    raise ValueError("Non-finite raw mode probe")
                amplitude = abs(complex(output[0], output[1]))
                gsl, exact = output[39], output[40]
                if gsl <= 0 or exact <= 0:
                    raise ValueError("Non-positive circular frequency")
                return {"amplitude": amplitude, "omega_circ_gsl": gsl,
                        "omega_circ_exact": exact,
                        "delta_amplitude": _delta_amplitude(amplitude, gsl, exact)}

            records = {}
            for offset in plan["sample_offsets_M"]:
                key = _key(offset)
                sample = samples[key]
                observed = evaluate(sample)
                replay_error = observed["amplitude"] - sample["amplitude"]
                if abs(replay_error) > 2e-12:
                    raise ValueError(f"Raw amplitude replay at {key}: {replay_error}")
                if key in old_samples:
                    old = old_samples[key]
                    if (observed["amplitude"] != old["amplitude"]
                            or observed["omega_circ_gsl"] != old["omega_circ_gsl"]
                            or observed["omega_circ_exact"] != old["omega_circ_exact"]):
                        raise ValueError(f"Saved operation replay at {key} changed")
                record = dict(observed, offset_M=offset,
                              raw_amplitude_replay_error=replay_error)
                radius, _phase, momentum, _pphi = sample["state"]
                p1 = momentum / (radius * sample["omega"])
                q1 = p1 * p1
                record["basis"] = [1.0, q1, q1 / radius, q1 / radius ** 1.5]
                records[key] = record
            report["samples"] = records
            q.write_evidence(args.json_out, report)

        captured = np.asarray(pilot["captured_derivatives"], dtype=float)
        targets = np.asarray(source["source"]["meta"][3:], dtype=float)
        if captured.shape != (7, 3) or targets.shape != (5,):
            raise ValueError("Unexpected frozen source system shape")
        solved = {}
        for step in plan["stencil_steps_M"]:
            key = str(step)
            historical = np.asarray(pilot["field"][key]["derivatives"], dtype=float)
            original_solve = _solve(historical, targets)
            saved_solve = pilot["field"][key]["all_rows"]
            replay_error = max(abs(np.asarray(original_solve["coefficients"])
                                   - np.asarray(saved_solve["coefficients"])))
            if replay_error > 1e-9:
                raise ValueError(f"Historical coefficient replay failed: {replay_error}")
            node_delta = np.asarray([
                np.asarray(records[_key(j * step)]["basis"])
                * records[_key(j * step)]["delta_amplitude"]
                for j in (-2, -1, 0, 1, 2)
            ])
            first, second = _stencil_delta(node_delta, step)
            perturbed = historical.copy()
            # Historical centers are the captured source values, while the saved
            # bounded-dense states supply only the differential stencil changes.
            center = records[_key(0.0)]
            factor_delta = center["delta_amplitude"] / center["amplitude"]
            center_delta = captured[:4, 0] * factor_delta
            perturbed[:4, 0] += center_delta
            perturbed[:4, 1] += first
            perturbed[:4, 2] += second
            counterfactual_solve = _solve(perturbed, targets)
            if (counterfactual_solve["phase_matrix"] != saved_solve["phase_matrix"]
                    or counterfactual_solve["phase_rhs"] != saved_solve["phase_rhs"]):
                raise ValueError("Phase system changed")
            solved[key] = {
                "historical_replay_coefficient_linf": float(replay_error),
                "amplitude_series_center_delta": center_delta.tolist(),
                "amplitude_series_first_delta": first.tolist(),
                "amplitude_series_second_delta": second.tolist(),
                "amplitude_matrix": counterfactual_solve["amplitude_matrix"],
                "amplitude_rhs": counterfactual_solve["amplitude_rhs"],
                "coefficients": counterfactual_solve["coefficients"],
                "coefficient_delta_from_historical": (
                    np.asarray(counterfactual_solve["coefficients"])
                    - np.asarray(saved_solve["coefficients"])).tolist(),
                "amplitude_matrix_condition": counterfactual_solve[
                    "amplitude_matrix_condition"],
                "amplitude_residual_linf": counterfactual_solve[
                    "amplitude_residual_linf"],
                "phase_system_unchanged": True,
            }
        first, second = (np.asarray(solved[str(step)]["coefficients"])
                         for step in plan["stencil_steps_M"])
        diagnostic_l2 = float(np.linalg.norm(second - first))
        report["counterfactual"] = {
            "systems": solved,
            "coefficient_stencil_l2": diagnostic_l2,
            "coefficient_stencil_below_historical_limit": diagnostic_l2 <= 0.01,
            "distance_change_from_historical": (
                diagnostic_l2 - pilot["stencil_coefficient_l2"]),
        }
        report["checks"] = {
            "historical_gate_still_failed": pilot["gates"]["coefficient_stencil"] is False,
            "phase_coefficients_unchanged_within_1e-10": all(
                np.max(np.abs(np.asarray(solved[str(step)]["coefficients"])[3:]
                              - np.asarray(pilot["field"][str(step)]["all_rows"]
                                           ["coefficients"])[3:])) < 1e-10
                for step in plan["stencil_steps_M"]),
            "max_saved_raw_amplitude_replay_error": max(
                abs(record["raw_amplitude_replay_error"])
                for record in records.values()),
            "max_historical_coefficient_replay_error": max(
                item["historical_replay_coefficient_linf"]
                for item in solved.values()),
        }
        report["status"] = "diagnostic_complete"
        q.write_evidence(args.json_out, report)
        print(json.dumps({"status": report["status"],
                          "historical_l2": pilot["stencil_coefficient_l2"],
                          "counterfactual_l2": diagnostic_l2,
                          "checks": report["checks"]}), flush=True)
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        q.write_evidence(args.json_out, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

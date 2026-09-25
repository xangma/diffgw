"""Attribute one frozen strong-spin raw-mode amplitude fluctuation to source operations."""

from __future__ import annotations

import argparse
import ctypes
from decimal import Decimal, localcontext
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np

from . import qualify_generator as q
from .diagnose_source_aligned_raw_amplitude_curvature import _key
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot


PLAN = (q.ROOT / "dev/native_rom/evidence/2026-09-25/"
        "source_aligned_raw_amplitude_operation_plan.json")
ANCHOR = "  out[2] = Hreal;\n"
FIELDS = ("omega_circ_gsl", "omega_circ_exact", "non_kepler", "v_phi_squared",
          "newton_amplitude", "residual_amplitude", "hamiltonian_full_difference",
          "hamiltonian_half_difference")
WEIGHTS = (-1, 16, -30, 16, -1)


def _decompose(first: dict, second: dict) -> dict:
    newton_delta = second["newton_amplitude"] - first["newton_amplitude"]
    residual_delta = second["residual_amplitude"] - first["residual_amplitude"]
    return {
        "amplitude_delta": second["amplitude"] - first["amplitude"],
        "amplitude_exact_circ_proxy_delta": (second["amplitude_exact_circ_proxy"]
                                             - first["amplitude_exact_circ_proxy"]),
        "newton_contribution": newton_delta * first["residual_amplitude"],
        "residual_contribution": second["newton_amplitude"] * residual_delta,
        "omega_circ_gsl_delta": second["omega_circ_gsl"] - first["omega_circ_gsl"],
        "omega_circ_exact_delta": second["omega_circ_exact"] - first["omega_circ_exact"],
        "hamiltonian_full_difference_delta": (second["hamiltonian_full_difference"]
                                               - first["hamiltonian_full_difference"]),
        "hamiltonian_half_difference_delta": (second["hamiltonian_half_difference"]
                                               - first["hamiltonian_half_difference"]),
    }


def _curvature(records: dict, field: str) -> float:
    with localcontext() as context:
        context.prec = 80
        numerator = sum(Decimal(weight) * Decimal.from_float(records[key][field])
                        for weight, key in zip(
                            WEIGHTS, ("-0.20", "-0.10", "0.00", "0.10", "0.20"),
                            strict=True))
        return float(numerator / (Decimal(12) * Decimal.from_float(0.1) ** 2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    names = ("curvature_evidence", "sensitivity_evidence", "source_flux_plan",
             "mode_adapter", "probe_include")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-raw-amplitude-operation-plan.v1"
            or plan["release_qualified"] is not False
            or plan["event_M"] != 137.59240346159578
            or plan["row"] != {"mass1": 30.0, "mass2": 20.0,
                               "chi1": 0.8, "chi2": 0.8, "omega_start": 0.04}
            or plan["integrators"] != ["bounded_dense_tight", "bounded_endpoint_tight"]
            or plan["offsets_M"] != [-0.2, -0.1, 0.0, 0.1, 0.2]
            or plan["center_perturbations"] != {
                "coordinates": ["radius", "angular_momentum", "radial_momentum", "omega"],
                "absolute_steps": [1e-8, 1e-9, 1e-10, 1e-11],
            }
            or not plan["algebraic_check"].startswith(
                "At each saved state replace only the Newtonian factor's numerical")
            or any(q.sha256(path) != plan[name + "_sha256"]
                   for name, path in paths.items())):
        parser.error("Frozen operation plan or inputs changed")
    prior = json.loads(paths["curvature_evidence"].read_text())
    sensitivity = json.loads(paths["sensitivity_evidence"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or sensitivity["status"] != "diagnostic_complete"
            or prior["event_M"] != plan["event_M"]
            or prior["row"] != plan["row"]
            or prior["historical_0p01_gate"] is not False
            or sensitivity["historical_0p01_gate"] is not False):
        parser.error("Frozen event or failed gate changed")
    report = {
        "schema": "diffgw.seobnrv4.source-aligned-raw-amplitude-operation.v1",
        "scope": dict(**q.SCOPE, development_only=True,
                      fixed_source_state_and_event=True),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": plan["row"], "event_M": plan["event_M"],
        "historical_0p01_gate": False,
        "samples": {}, "comparisons": {}, "curvature": {},
        "center_perturbations": {},
        "errors": [], "limitations": plan["interpretation"],
    }
    q.write_evidence(args.json_out, report)
    try:
        libraries = _libraries(source_plan["pinned_wheel"])
        with tempfile.TemporaryDirectory(prefix="diffgw-source-raw-amp-operation-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            original = paths["mode_adapter"].read_text()
            if original.count(ANCHOR) != 1:
                raise ValueError("Pinned dev adapter insertion anchor changed")
            adapter = Path(tmp) / "mode_probe.c"
            adapter.write_text(original.replace(ANCHOR, ANCHOR
                                               + paths["probe_include"].read_text()))
            dylib = Path(tmp) / "mode_probe.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     adapter, dylib, source_plan["flux_source_includes"],
                     source_plan["flux_public_includes"])
            function = ctypes.CDLL(str(dylib)).diffgw_v4_mode22_same_state
            function.argtypes = [ctypes.c_double] * 9 + [ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            row = plan["row"]
            x1 = row["mass1"] / (row["mass1"] + row["mass2"])

            def evaluate(state: np.ndarray, omega: float) -> dict:
                output = (ctypes.c_double * 47)()
                rc = function(x1, 1 - x1, row["chi1"], row["chi2"],
                              *state, omega, output)
                if rc:
                    raise RuntimeError(f"Pinned raw-mode operation failed: {rc}")
                values = np.asarray(output[:47])
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite operation trace")
                result = {name: float(value) for name, value in
                          zip(FIELDS, values[39:], strict=True)}
                result["amplitude"] = abs(complex(*values[:2]))
                result["amplitude_exact_circ_proxy"] = result["amplitude"] * (
                    result["omega_circ_gsl"] / result["omega_circ_exact"]) ** (4 / 3)
                result["newton_reconstruction_error"] = (
                    result["newton_amplitude"] * result["residual_amplitude"]
                    - result["amplitude"])
                return result

            for name in plan["integrators"]:
                records = {}
                for offset in plan["offsets_M"]:
                    key = _key(offset)
                    sample = prior["integrators"][name]["samples"][key]
                    record = evaluate(np.asarray(sample["state"]), sample["omega"])
                    record["saved_amplitude_replay_error"] = (
                        record["amplitude"] - sample["amplitude"])
                    records[key] = record
                report["samples"][name] = records
                q.write_evidence(args.json_out, report)
            dense = report["samples"]["bounded_dense_tight"]
            endpoint = report["samples"]["bounded_endpoint_tight"]
            comparisons = {}
            for offset in plan["offsets_M"]:
                key = _key(offset)
                dense_sample = prior["integrators"]["bounded_dense_tight"]["samples"][key]
                endpoint_sample = prior["integrators"]["bounded_endpoint_tight"]["samples"][key]
                endpoint_state_at_dense_omega = evaluate(
                    np.asarray(endpoint_sample["state"]), dense_sample["omega"])
                comparisons[key] = {
                    "state_at_fixed_omega": _decompose(
                        dense[key], endpoint_state_at_dense_omega),
                    "omega_at_fixed_state": _decompose(
                        endpoint_state_at_dense_omega, endpoint[key]),
                    "total": _decompose(dense[key], endpoint[key]),
                }
            report["comparisons"] = comparisons
            report["curvature"] = {
                field: {
                    "bounded_dense_tight": _curvature(dense, field),
                    "bounded_endpoint_tight": _curvature(endpoint, field),
                    "endpoint_minus_dense": (_curvature(endpoint, field)
                                             - _curvature(dense, field)),
                }
                for field in ("amplitude", "amplitude_exact_circ_proxy")
            }
            center = prior["integrators"]["bounded_dense_tight"]["samples"][_key(0.)]
            state = np.asarray(center["state"])
            omega = center["omega"]
            baseline = evaluate(state, omega)
            index = {"radius": 0, "angular_momentum": 3,
                     "radial_momentum": 2, "omega": None}
            for name in plan["center_perturbations"]["coordinates"]:
                records = {}
                for step in plan["center_perturbations"]["absolute_steps"]:
                    if index[name] is None:
                        plus, minus = evaluate(state, omega + step), evaluate(state,
                                                                               omega - step)
                    else:
                        forward, backward = state.copy(), state.copy()
                        forward[index[name]] += step
                        backward[index[name]] -= step
                        plus, minus = evaluate(forward, omega), evaluate(backward, omega)
                    records[str(step)] = {
                        "plus": _decompose(baseline, plus),
                        "minus": _decompose(baseline, minus),
                        "central_slope": {field: (plus[field]-minus[field])/(2*step)
                                          for field in ("amplitude", "omega_circ_gsl",
                                                        "omega_circ_exact", "newton_amplitude",
                                                        "residual_amplitude")},
                    }
                report["center_perturbations"][name] = records
                q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

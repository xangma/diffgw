"""Probe raw (2,2)-mode amplitude curvature along one pinned source ODE event.

This development-only diagnostic holds the captured NQC state and match event
fixed. It neither refits NQC coefficients nor generates a new waveform.
"""

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
from scipy.integrate import solve_ivp

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot


PLAN = (q.ROOT / "dev/native_rom/evidence/2026-09-25/"
        "source_aligned_raw_amplitude_curvature_plan.json")
WEIGHTS_FIVE = (-1, 16, -30, 16, -1)
WEIGHTS_SEVEN = (2, -27, 270, -490, 270, -27, 2)


def _key(offset: float) -> str:
    return f"{offset:.2f}"


def _curvature(samples: dict[str, dict], step: float, seven: bool = False) -> dict:
    offsets = np.arange(-3 if seven else -2, (3 if seven else 2) + 1) * step
    values = [samples[_key(float(offset))]["amplitude"] for offset in offsets]
    weights = WEIGHTS_SEVEN if seven else WEIGHTS_FIVE
    divisor = 180 if seven else 12
    floating = sum(weight * value for weight, value in zip(weights, values, strict=True))
    floating /= divisor * step * step
    with localcontext() as context:
        context.prec = 80
        precise = sum(Decimal(weight) * Decimal.from_float(value)
                      for weight, value in zip(weights, values, strict=True))
        precise /= Decimal(divisor) * Decimal.from_float(step) ** 2
    return {"value": float(floating), "decimal_on_float_samples": float(precise),
            "float_arithmetic_error": floating - float(precise), "samples": values}


def _integrate(field, state, event: float, extent: float, config: dict,
               offsets: list[float]) -> tuple[dict[str, np.ndarray], dict]:
    parameters = {"method": "DOP853", "rtol": config["rtol"],
                  "atol": config["atol"]}
    if config["max_step_M"] is not None:
        parameters["max_step"] = config["max_step_M"]
    def rhs(_time, value):
        return field(value)[:4]
    if config["sampling"] == "dense":
        backward = solve_ivp(rhs, (event, event - extent), state,
                             dense_output=True, **parameters)
        forward = solve_ivp(rhs, (event, event + extent), state,
                            dense_output=True, **parameters)
        if not backward.success or not forward.success:
            raise RuntimeError("Centered source integration failed")
        states = {_key(offset): (backward.sol(event + offset) if offset < 0
                                 else forward.sol(event + offset))
                  for offset in offsets}
        return states, {"nfev_backward": backward.nfev,
                        "nfev_forward": forward.nfev}
    if config["sampling"] != "independent_endpoint":
        raise ValueError("Invalid sampling method")
    states = {_key(0.0): state.copy()}
    nfev = 0
    for offset in offsets:
        if offset == 0:
            continue
        solution = solve_ivp(rhs, (event, event + offset), state, **parameters)
        if not solution.success:
            raise RuntimeError(f"Source endpoint integration failed at {offset}")
        states[_key(offset)] = solution.y[:, -1]
        nfev += solution.nfev
    return states, {"nfev_total": nfev}


def _compare(first: dict, second: dict, steps: list[float]) -> dict:
    states0 = np.asarray([sample["state"] for sample in first["samples"].values()])
    states1 = np.asarray([sample["state"] for sample in second["samples"].values()])
    amplitudes0 = np.asarray([sample["amplitude"] for sample in first["samples"].values()])
    amplitudes1 = np.asarray([sample["amplitude"] for sample in second["samples"].values()])
    return {
        "state_abs_linf": float(np.max(abs(states1 - states0))),
        "amplitude_abs_linf": float(np.max(abs(amplitudes1 - amplitudes0))),
        "curvature_deltas": {
            str(step): second["five_point"][str(step)]["value"]
            - first["five_point"][str(step)]["value"] for step in steps},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    names = ("source_evidence", "pilot_evidence", "solve_evidence",
             "source_flux_plan", "field_adapter", "mode_adapter")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-raw-amplitude-curvature-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row"] != {"mass1": 30.0, "mass2": 20.0,
                               "chi1": 0.8, "chi2": 0.8, "omega_start": 0.04}
            or plan["event_M"] != 137.59240346159578
            or plan["state_source"] != "captured_state_at_match"
            or plan["offsets_M"] != [-0.4, -0.3, -0.2, -0.15, -0.1, -0.05,
                                     0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
            or plan["five_point_steps_M"] != [0.05, 0.1, 0.2]
            or plan["seven_point_step_M"] != 0.1
            or any(q.sha256(path) != plan[name + "_sha256"]
                   for name, path in paths.items())):
        parser.error("Frozen plan or source inputs changed")
    source = json.loads(paths["source_evidence"].read_text())
    pilot = json.loads(paths["pilot_evidence"].read_text())
    solved = json.loads(paths["solve_evidence"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (source["status"] != "diagnostic_complete"
            or source["row"] != plan["row"]
            or source["source"]["match_time_M"] != plan["event_M"]
            or pilot["gates"]["coefficient_stencil"] is not False
            or solved["historical_0p01_gate"] is not False):
        parser.error("Frozen source event or failed gate changed")
    row = plan["row"]
    state = np.asarray(source["source"]["captured_state_at_match"], dtype=np.float64)
    event = plan["event_M"]
    if state.shape != (4,) or not np.isfinite(state).all():
        parser.error("Invalid captured source state")
    report = {
        "schema": "diffgw.seobnrv4.source-aligned-raw-amplitude-curvature.v1",
        "scope": dict(**q.SCOPE, development_only=True,
                      fixed_source_state_and_event=True),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": row, "event_M": event, "state": state.tolist(),
        "historical_0p01_gate": False,
        "historical_coefficient_stencil_l2": pilot["stencil_coefficient_l2"],
        "historical_amplitude_coefficient_delta_l2": solved["pairs"]
            ["pilot_0.1_to_pilot_0.2"]["amplitude"]["actual_coefficient_delta_l2"],
        "integrators": {}, "comparisons": {}, "errors": [],
        "limitations": plan["interpretation"],
    }
    q.write_evidence(args.json_out, report)
    try:
        libraries = _libraries(source_plan["pinned_wheel"])
        with tempfile.TemporaryDirectory(prefix="diffgw-source-raw-amp-curvature-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            field_path = Path(tmp) / "field.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["field_adapter"], field_path,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            field_function = ctypes.CDLL(str(field_path)).diffgw_v4_source_field
            field_function.argtypes = [ctypes.c_double] * 8 + [ctypes.POINTER(ctypes.c_double)]
            field_function.restype = ctypes.c_int
            mode_path = Path(tmp) / "mode.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["mode_adapter"], mode_path,
                     source_plan["flux_source_includes"],
                     source_plan["flux_public_includes"])
            mode_function = ctypes.CDLL(str(mode_path)).diffgw_v4_mode22_same_state
            mode_function.argtypes = [ctypes.c_double] * 9 + [ctypes.POINTER(ctypes.c_double)]
            mode_function.restype = ctypes.c_int
            masses = [row[key] for key in ("mass1", "mass2", "chi1", "chi2")]
            x1 = row["mass1"] / (row["mass1"] + row["mass2"])

            def field(value: np.ndarray) -> np.ndarray:
                output = (ctypes.c_double * 6)()
                rc = field_function(*masses, *value, output)
                if rc:
                    raise RuntimeError(f"Pinned source field failed: {rc}")
                return np.asarray(output[:6])

            def mode(value: np.ndarray, omega: float) -> complex:
                output = (ctypes.c_double * 39)()
                rc = mode_function(x1, 1 - x1, row["chi1"], row["chi2"],
                                   *value, omega, output)
                if rc:
                    raise RuntimeError(f"Pinned raw mode failed: {rc}")
                return complex(output[0], output[1])

            offsets = plan["offsets_M"]
            for config in plan["integrators"]:
                states, counts = _integrate(field, state, event, max(abs(x) for x in offsets),
                                             config, offsets)
                samples = {}
                for offset in offsets:
                    key = _key(offset)
                    value = states[key]
                    omega = field(value)[1]
                    raw = mode(value, omega)
                    gauge_value = value.copy()
                    gauge_value[1] = 0.0
                    gauge_raw = mode(gauge_value, omega)
                    samples[key] = {"offset_M": offset, "state": value.tolist(),
                                    "omega": float(omega),
                                    "mode": [raw.real, raw.imag],
                                    "amplitude": abs(raw),
                                    "amplitude_phi_zero": abs(gauge_raw)}
                five_point = {str(step): _curvature(samples, step)
                              for step in plan["five_point_steps_M"]}
                seven_point = _curvature(samples, plan["seven_point_step_M"],
                                         seven=True)
                gauge_samples = {key: {**sample,
                                       "amplitude": sample["amplitude_phi_zero"]}
                                 for key, sample in samples.items()}
                gauge_five_point = {str(step): _curvature(gauge_samples, step)["value"]
                                    for step in plan["five_point_steps_M"]}
                report["integrators"][config["name"]] = {
                    "config": config, "counts": counts, "samples": samples,
                    "five_point": five_point, "seven_point": seven_point,
                    "phi_zero_five_point": gauge_five_point,
                    "max_gauge_amplitude_abs_difference": float(max(
                        abs(item["amplitude"] - item["amplitude_phi_zero"])
                        for item in samples.values())),
                }
                q.write_evidence(args.json_out, report)
                print(json.dumps({"integrator": config["name"],
                                  "nfev": counts,
                                  "curvature": {key: value["value"]
                                                for key, value in five_point.items()}}),
                      flush=True)

        integrators = report["integrators"]
        baseline = integrators["adaptive_dense_tight"]
        historical = source["centered_dop853"][-1]["stencils"]
        report["comparisons"] = {
            "replay_minus_historical_curvature": {
                str(step): baseline["five_point"][str(step)]["value"]
                - historical[str(step)]["second"][0]
                for step in plan["five_point_steps_M"]},
            "historical_0p2_minus_0p1_curvature": (
                historical["0.2"]["second"][0] - historical["0.1"]["second"][0]),
            "baseline_0p2_minus_0p1_curvature": (
                baseline["five_point"]["0.2"]["value"]
                - baseline["five_point"]["0.1"]["value"]),
            "loose_vs_tight": _compare(integrators["adaptive_dense_loose"],
                                       baseline, plan["five_point_steps_M"]),
            "adaptive_vs_bounded_dense": _compare(
                baseline, integrators["bounded_dense_tight"],
                plan["five_point_steps_M"]),
            "bounded_dense_vs_endpoint": _compare(
                integrators["bounded_dense_tight"],
                integrators["bounded_endpoint_tight"],
                plan["five_point_steps_M"]),
            "center_explicit_minus_captured_amplitude": (
                baseline["samples"][_key(0.0)]["amplitude"]
                - source["source"]["captured_amplitude_at_match"]),
            "center_explicit_replay_error": (
                baseline["samples"][_key(0.0)]["amplitude"]
                - source["source"]["explicit_mode_amplitude_at_match"]),
        }
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"],
                      "comparisons": report["comparisons"],
                      "errors": report["errors"]}), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

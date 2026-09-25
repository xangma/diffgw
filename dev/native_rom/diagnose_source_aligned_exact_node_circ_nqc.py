"""Replay historical source NQC stencil nodes for a post-hoc amplitude audit."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
from scipy.integrate import solve_ivp

from . import qualify_generator as q
from .diagnose_source_aligned_derivative_only_pilot import _solve
from .diagnose_source_aligned_exact_circ_nqc_counterfactual import (
    _delta_amplitude,
    _stencil_delta,
)
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_dense_field import five_point


PLAN = (q.ROOT / "dev/native_rom/evidence/2026-09-25/"
        "source_aligned_exact_node_circ_nqc_plan.json")
WEIGHTS = (-2, -1, 0, 1, 2)


def _original_series(nodes: list[dict]) -> np.ndarray:
    values = []
    for node in nodes:
        radius, _phase, momentum, _pphi = node["state"]
        amplitude = node["amplitude"]
        omega = node["omega"]
        p1 = momentum / (radius * omega)
        q1 = p1 * p1
        values.append((amplitude, amplitude * q1, amplitude * q1 / radius,
                       amplitude * q1 / radius ** 1.5, node["mode_phase"],
                       p1, p1 * momentum * momentum))
    series = np.asarray(values)
    series[:, 4] = np.unwrap(series[:, 4])
    return series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)

    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / value["path"]
             for name, value in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-exact-node-circ-nqc-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row"] != {"mass1": 30.0, "mass2": 20.0, "chi1": 0.8,
                               "chi2": 0.8, "omega_start": 0.04}
            or plan["match_time_M"] != 137.59240346159578
            or plan["rtol"] != 1e-12 or plan["atol"] != 1e-13
            or plan["half_extent_M"] != 0.4
            or plan["stencil_steps_M"] != [0.1, 0.2]
            or plan["sample_offsets_M"] != [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]
            or plan["historical_coefficient_stencil_l2_limit"] != 0.01
            or any(q.sha256(path) != value["sha256"]
                   for name, value in plan["inputs"].items()
                   for path in (paths[name],))):
        parser.error("Frozen plan or inputs changed")
    source = json.loads(paths["source_nqc_evidence"].read_text())
    pilot = json.loads(paths["pilot_evidence"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    prior = source["centered_dop853"][-1]
    if (source["status"] != "diagnostic_complete"
            or pilot["status"] != "diagnostic_complete"
            or source["row"] != plan["row"] or pilot["row"] != plan["row"]
            or source["source"]["match_time_M"] != plan["match_time_M"]
            or prior["rtol"] != plan["rtol"] or prior["atol"] != plan["atol"]
            or source["source"]["meta"][1] != 0.00390625
            or pilot["stencil_coefficient_l2"] <= 0.01
            or pilot["gates"]["coefficient_stencil"] is not False):
        parser.error("Frozen source event or historical failed gate changed")

    report = {
        "schema": "diffgw.seobnrv4.source-aligned-exact-node-circ-nqc.v1",
        "scope": dict(**q.SCOPE, development_only=True,
                      post_hoc_algebraic_counterfactual=True),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
                    "argv": sys.argv},
        "source_pin": source_plan["pinned_source"],
        "wheel_pin": source_plan["pinned_wheel"],
        "row": plan["row"], "match_time_M": plan["match_time_M"],
        "historical": {"coefficient_stencil_l2": pilot["stencil_coefficient_l2"],
                       "coefficient_stencil_gate": False},
        "replay": {}, "systems": {}, "counterfactual": {}, "errors": [],
        "limitations": plan["limitations"],
    }
    q.write_evidence(args.json_out, report)
    try:
        libraries = _libraries(source_plan["pinned_wheel"])
        with tempfile.TemporaryDirectory(prefix="diffgw-source-exact-node-nqc-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            field_lib = Path(tmp) / "field.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["field_adapter"], field_lib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            original = paths["mode_adapter"].read_text()
            # The prior probe's own anchor is pinned in its input plan.
            anchor = "  out[2] = Hreal;\n"
            if original.count(anchor) != 1:
                raise ValueError("Pinned raw-mode probe insertion anchor changed")
            mode_adapter = Path(tmp) / "mode_probe.c"
            mode_adapter.write_text(original.replace(
                anchor, anchor + paths["probe_include"].read_text()))
            mode_lib = Path(tmp) / "mode.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     mode_adapter, mode_lib, source_plan["flux_source_includes"],
                     source_plan["flux_public_includes"])

            field = ctypes.CDLL(str(field_lib)).diffgw_v4_source_field
            field.argtypes = [ctypes.c_double] * 8 + [ctypes.POINTER(ctypes.c_double)]
            field.restype = ctypes.c_int
            mode = ctypes.CDLL(str(mode_lib)).diffgw_v4_mode22_same_state
            mode.argtypes = [ctypes.c_double] * 9 + [ctypes.POINTER(ctypes.c_double)]
            mode.restype = ctypes.c_int
            row = plan["row"]
            masses = [row[k] for k in ("mass1", "mass2", "chi1", "chi2")]
            x1 = row["mass1"] / (row["mass1"] + row["mass2"])

            def source_field(state: np.ndarray) -> np.ndarray:
                out = (ctypes.c_double * 6)()
                if field(*masses, *state, out):
                    raise RuntimeError("Pinned source field failed")
                result = np.asarray(out[:4])
                if not np.isfinite(result).all():
                    raise ValueError("Nonfinite pinned source field")
                return result

            def source_mode(state: np.ndarray, omega: float) -> dict:
                out = (ctypes.c_double * 47)()
                if mode(x1, 1 - x1, row["chi1"], row["chi2"], *state,
                        omega, out):
                    raise RuntimeError("Pinned source pre-NQC mode failed")
                if not all(math.isfinite(value) for value in out):
                    raise ValueError("Nonfinite pinned source pre-NQC mode")
                wave = complex(out[0], out[1])
                gsl, exact = out[39], out[40]
                if gsl <= 0 or exact <= 0:
                    raise ValueError("Nonpositive circular frequency")
                amplitude = abs(wave)
                return {"amplitude": amplitude, "mode_phase": np.angle(wave),
                        "omega_circ_gsl": gsl, "omega_circ_exact": exact,
                        "delta_amplitude": _delta_amplitude(amplitude, gsl, exact)}

            center_state = np.asarray(
                source["source"]["captured_state_at_match"], dtype=np.float64)
            match = plan["match_time_M"]
            backward = solve_ivp(lambda t, y: source_field(y),
                                 (match, match - plan["half_extent_M"]),
                                 center_state, method="DOP853",
                                 rtol=plan["rtol"], atol=plan["atol"],
                                 dense_output=True)
            forward = solve_ivp(lambda t, y: source_field(y),
                                (match, match + plan["half_extent_M"]),
                                center_state, method="DOP853",
                                rtol=plan["rtol"], atol=plan["atol"],
                                dense_output=True)
            if not backward.success or not forward.success:
                raise RuntimeError("Centered source field reintegration failed")
            nfev = {"backward": backward.nfev, "forward": forward.nfev}
            expected_nfev = {"backward": prior["nfev_backward"],
                             "forward": prior["nfev_forward"]}
            report["replay"]["nfev"] = nfev
            report["replay"]["expected_nfev"] = expected_nfev
            if nfev != expected_nfev:
                raise ValueError("Centered DOP853 evaluation counts changed")

            captured = np.asarray(pilot["captured_derivatives"], dtype=np.float64)
            targets = np.asarray(source["source"]["meta"][3:], dtype=np.float64)
            solved = {}
            max_derivative_error = 0.0
            max_center_amp_error = 0.0
            max_historical_coefficient_error = 0.0
            for step in plan["stencil_steps_M"]:
                key = str(step)
                times = match + np.arange(-2, 3) * step
                nodes = []
                for offset, coordinate in zip(WEIGHTS, times, strict=True):
                    state = (backward.sol(coordinate) if coordinate < match
                             else forward.sol(coordinate))
                    omega = source_field(state)[1]
                    node = source_mode(state, float(omega))
                    node.update(offset_M=offset * step, time_M=float(coordinate),
                                state=state.tolist(), omega=float(omega))
                    nodes.append(node)
                series = _original_series(nodes)
                stencil = five_point(series, step)
                historical_stencil = prior["stencils"][key]
                derivative_error = max(
                    float(np.max(np.abs(np.asarray(stencil[k])
                                        - np.asarray(historical_stencil[k]))))
                    for k in ("first", "second"))
                center_amp_error = abs(nodes[2]["amplitude"]
                                       - historical_stencil["center"][0])
                max_derivative_error = max(max_derivative_error, derivative_error)
                max_center_amp_error = max(max_center_amp_error, center_amp_error)
                historical = np.asarray(pilot["field"][key]["derivatives"])
                historical_solve = _solve(historical, targets)
                coefficient_error = float(np.max(np.abs(
                    np.asarray(historical_solve["coefficients"])
                    - np.asarray(pilot["field"][key]["all_rows"]["coefficients"]))))
                max_historical_coefficient_error = max(
                    max_historical_coefficient_error, coefficient_error)
                solved[key] = {"nodes": nodes, "original_stencil": stencil,
                               "historical_stencil_derivative_linf": derivative_error,
                               "historical_coefficient_replay_linf": coefficient_error}

                delta = np.asarray([
                    np.asarray((1.0, *(series[j, 1:4] / series[j, 0])))
                    * node["delta_amplitude"]
                    for j, node in enumerate(nodes)
                ])
                first, second = _stencil_delta(delta, step)
                perturbed = historical.copy()
                central_factor = nodes[2]["delta_amplitude"] / nodes[2]["amplitude"]
                center_delta = captured[:4, 0] * central_factor
                perturbed[:4, 0] += center_delta
                perturbed[:4, 1] += first
                perturbed[:4, 2] += second
                counterfactual = _solve(perturbed, targets)
                if (counterfactual["phase_matrix"]
                        != historical_solve["phase_matrix"]
                        or counterfactual["phase_rhs"]
                        != historical_solve["phase_rhs"]):
                    raise ValueError("Source phase matching system changed")
                solved[key]["counterfactual"] = {
                    "amplitude_series_center_delta": center_delta.tolist(),
                    "amplitude_series_first_delta": first.tolist(),
                    "amplitude_series_second_delta": second.tolist(),
                    "coefficients": counterfactual["coefficients"],
                    "amplitude_matrix_condition": counterfactual[
                        "amplitude_matrix_condition"],
                    "amplitude_residual_linf": counterfactual[
                        "amplitude_residual_linf"],
                    "phase_system_unchanged": True,
                }
            report["replay"].update(
                max_source_stencil_derivative_linf=max_derivative_error,
                max_source_center_amplitude_absolute=max_center_amp_error,
                max_historical_coefficient_replay_linf=max_historical_coefficient_error)
            report["systems"] = solved
            q.write_evidence(args.json_out, report)
            checks = plan["replay"]
            if (max_derivative_error > checks["source_stencil_derivative_linf"]
                    or max_center_amp_error > checks["source_center_amplitude_absolute"]
                    or max_historical_coefficient_error
                    > checks["historical_coefficient_replay_linf"]):
                raise ValueError("Historical NQC stencil replay misses frozen tolerance")
            first_coefficients, second_coefficients = (
                np.asarray(solved[str(step)]["counterfactual"]["coefficients"])
                for step in plan["stencil_steps_M"])
            corrected_l2 = float(np.linalg.norm(second_coefficients-first_coefficients))
            report["counterfactual"] = {
                "coefficient_stencil_l2": corrected_l2,
                "below_historical_limit": corrected_l2 <= 0.01,
                "distance_change_from_historical": (
                    corrected_l2 - pilot["stencil_coefficient_l2"]),
                "historical_gate_still_failed": True,
            }
            report["status"] = "diagnostic_complete"
            q.write_evidence(args.json_out, report)
            print(json.dumps({"status": report["status"],
                              "historical_l2": pilot["stencil_coefficient_l2"],
                              "counterfactual_l2": corrected_l2,
                              "replay": report["replay"]}), flush=True)
            return 0
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        q.write_evidence(args.json_out, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

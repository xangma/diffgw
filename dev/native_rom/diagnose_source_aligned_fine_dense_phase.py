"""Check corrected-source fine-grid NQC phase curvature against its dense field."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
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
from .diagnose_source_nqc_dense_field import CALL_ANCHOR, STATE_CAPTURE
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE
from .diagnose_source_q4_peak_bracket_correction import ANCHOR, CORRECTION


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/source_aligned_fine_dense_phase_plan.json"


def _pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _capture(function, row, step, capacity):
    count = ctypes.c_size_t()
    meta = np.empty(8, dtype=np.float64)
    values = np.empty((capacity, 5), dtype=np.float64)
    coefficients = np.empty(10, dtype=np.float64)
    extra = np.empty((capacity, 2), dtype=np.float64)
    rc = function(*[row[key] for key in ("mass1", "mass2", "chi1", "chi2")],
                  row["omega_start"], step, capacity, ctypes.byref(count),
                  _pointer(meta), _pointer(values), _pointer(coefficients),
                  _pointer(extra))
    if rc or count.value < 10 or count.value > capacity:
        raise RuntimeError(f"Pinned source state capture failed at {step:g} M: {rc}")
    return meta, values[:count.value].copy(), extra[:count.value].copy()


def _five_point(state_at, match, step, basis):
    sample = np.asarray([basis(state_at(match + k*step))
                         for k in (-2, -1, 0, 1, 2)])
    return (-sample[0] + 16*sample[1] - 30*sample[2]
            + 16*sample[3] - sample[4])/(12*step*step)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    names = ("prior_evidence", "prior_script", "source_flux_plan",
             "capture_adapter", "capture_base_adapter", "field_adapter",
             "dense_impl", "peak_impl")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-fine-dense-phase-plan.v1"
            or plan["release_qualified"] is not False
            or plan["steps_M"] != [0.00390625, 0.001953125]
            or plan["capture_capacity"] != 100000
            or plan["five_point_steps_M"] != [0.2, 0.1, 0.05, 0.025]
            or plan["dop853_tolerances"] != [[1e-11, 1e-12], [1e-12, 1e-13]]
            or any(q.sha256(path) != plan[name + "_sha256"]
                   for name, path in paths.items())):
        parser.error("Frozen plan or input changed")
    prior = json.loads(paths["prior_evidence"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or prior["release_qualified"] is not False
            or prior["row"] != {"mass1": 30.0, "mass2": 20.0,
                                "chi1": 0.8, "chi2": 0.8,
                                "omega_start": 0.04}):
        parser.error("Frozen prior result changed")
    row = prior["row"]
    report = {
        "schema": "diffgw.seobnrv4.source-aligned-fine-dense-phase.v1",
        "scope": dict(**q.SCOPE, development_only=True,
                      independent_source_field=True),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "instrumentation_sha256": hashlib.sha256(
            (CORRECTION + CAPTURE + STATE_CAPTURE).encode()).hexdigest(),
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": row, "grids": {}, "comparisons": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        libraries = _libraries(source_plan["pinned_wheel"])
        with tempfile.TemporaryDirectory(prefix="diffgw-source-aligned-fine-dense-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if any(original.count(anchor) != 1
                   for anchor in (ANCHOR, INCLUDE, CALL_ANCHOR)):
                raise ValueError("Pinned source diagnostic anchors changed")
            source_file.write_text(original.replace(ANCHOR, CORRECTION + ANCHOR)
                                   .replace(INCLUDE, INCLUDE + CAPTURE)
                                   .replace(CALL_ANCHOR, STATE_CAPTURE + CALL_ANCHOR))
            capture_path = Path(tmp) / "capture.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["capture_adapter"], capture_path,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            capture = ctypes.CDLL(str(capture_path)).diffgw_v4_full_source_nqc_states
            capture.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)]
            capture.restype = ctypes.c_int
            source_file.write_text(original)
            field_path = Path(tmp) / "field.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["field_adapter"], field_path,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            field_function = ctypes.CDLL(str(field_path)).diffgw_v4_source_field
            field_function.argtypes = [ctypes.c_double]*8 + [
                ctypes.POINTER(ctypes.c_double)]
            field_function.restype = ctypes.c_int

            def field(state):
                output = (ctypes.c_double*6)()
                rc = field_function(row["mass1"], row["mass2"], row["chi1"],
                                    row["chi2"], *state, output)
                if rc:
                    raise RuntimeError(f"Pinned source field failed: {rc}")
                return np.asarray(output[:4])

            def basis(state):
                radius, _phase, momentum, _angular_momentum = state
                omega = field(state)[1]
                p1 = momentum/(radius*omega)
                return np.asarray((p1, p1*momentum*momentum))

            for step in plan["steps_M"]:
                frozen = prior["grids"][str(step)]
                meta, values, extra = _capture(
                    capture, row, step, plan["capture_capacity"])
                input_hash = hashlib.sha256(values.tobytes()).hexdigest()
                if (input_hash != frozen["input_sha256"]
                        or len(values) != frozen["nqc_samples"]
                        or not np.isfinite(meta).all()
                        or not np.isfinite(values).all()
                        or not np.isfinite(extra).all()
                        or abs(meta[1]-step) > 1e-12
                        or abs(meta[2]-frozen["match_time_M"]) > 1e-12):
                    raise ValueError(f"Pinned source replay changed at {step:g} M")
                axis = np.arange(len(values), dtype=np.float64)*meta[1]
                match = float(meta[2])
                index = int(np.argmin(abs(axis-match)))
                state = np.asarray((values[index, 2], extra[index, 0],
                                    values[index, 3], extra[index, 1]))
                local_matrix = np.asarray(frozen["local_fit"]["phase_matrix"])
                local_rhs = np.asarray(frozen["local_fit"]["phase_rhs"])
                if np.max(abs(np.linalg.solve(local_matrix, local_rhs)
                              - frozen["local_fit"]["coefficients"][3:])) > 1e-8:
                    raise ValueError("Frozen phase system does not reconstruct")
                grid = {"input_sha256": input_hash, "samples": len(values),
                        "match_time_M": match, "nearest_time_M": float(axis[index]),
                        "nearest_state": state.tolist(),
                        "nearest_omega_difference": float(field(state)[1]-values[index, 4]),
                        "local_phase_matrix": local_matrix.tolist(),
                        "local_phase_rhs": local_rhs.tolist(),
                        "centered_dop853": []}
                report["grids"][str(step)] = grid
                q.write_evidence(args.json_out, report)
                extent = 2*max(plan["five_point_steps_M"])
                for rtol, atol in plan["dop853_tolerances"]:
                    backward = solve_ivp(lambda _t, s: field(s),
                                         (axis[index], match-extent), state,
                                         method="DOP853", rtol=rtol, atol=atol,
                                         dense_output=True)
                    forward = solve_ivp(lambda _t, s: field(s),
                                        (axis[index], match+extent), state,
                                        method="DOP853", rtol=rtol, atol=atol,
                                        dense_output=True)
                    if not backward.success or not forward.success:
                        raise RuntimeError("Centered source integration failed")
                    def state_at(t):
                        return backward.sol(t) if t < axis[index] else forward.sol(t)

                    # The source NQC phase matrix is minus the basis derivatives.
                    rows = {str(h): (-_five_point(state_at, match, h, basis)).tolist()
                            for h in plan["five_point_steps_M"]}
                    replacement = {}
                    for h, row_values in rows.items():
                        matrix = local_matrix.copy()
                        matrix[1] = row_values
                        replacement[h] = np.linalg.solve(matrix, local_rhs).tolist()
                    grid["centered_dop853"].append({
                        "rtol": rtol, "atol": atol, "nfev": backward.nfev+forward.nfev,
                        "phase_second_rows": rows,
                        "dense_row_replaced_coefficients": replacement,
                        "center_state": state_at(match).tolist(),
                    })
                    q.write_evidence(args.json_out, report)
                grid["local_minus_dense_phase_second_row_by_stencil"] = {}
                grid["dense_tolerance_spread_by_stencil"] = {}
                for h in plan["five_point_steps_M"]:
                    key = str(h)
                    low = np.asarray(grid["centered_dop853"][0]["phase_second_rows"][key])
                    high = np.asarray(grid["centered_dop853"][-1][
                        "phase_second_rows"][key])
                    grid["local_minus_dense_phase_second_row_by_stencil"][key] = (
                        local_matrix[1]-high).tolist()
                    grid["dense_tolerance_spread_by_stencil"][key] = (low-high).tolist()
                q.write_evidence(args.json_out, report)
                print(json.dumps({"step_M": step,
                                  "local_minus_dense_0.1_M": grid[
                                      "local_minus_dense_phase_second_row_by_stencil"][
                                          "0.1"]}),
                      flush=True)
        first, second = (report["grids"][str(step)] for step in plan["steps_M"])
        report["comparisons"]["local_row_shift"] = (
            np.asarray(second["local_phase_matrix"][1])
            - np.asarray(first["local_phase_matrix"][1])).tolist()
        report["comparisons"]["dense_row_shift_by_stencil"] = {}
        report["comparisons"]["dense_row_replaced_coefficient_shift_by_stencil"] = {}
        latest_first, latest_second = (grid["centered_dop853"][-1]
                                       for grid in (first, second))
        for h in plan["five_point_steps_M"]:
            key = str(h)
            report["comparisons"]["dense_row_shift_by_stencil"][key] = (
                np.asarray(latest_second["phase_second_rows"][key])
                - np.asarray(latest_first["phase_second_rows"][key])).tolist()
            report["comparisons"][
                "dense_row_replaced_coefficient_shift_by_stencil"][key] = (
                np.asarray(latest_second["dense_row_replaced_coefficients"][key])
                - np.asarray(latest_first["dense_row_replaced_coefficients"][key])
            ).tolist()
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

"""Replay corrected-source whole modes with dense-field NQC curvature rows."""

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

import lal
import numpy as np

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import INCLUDE
from .diagnose_source_nqc_whole_mode_override import OVERRIDE, override_source
from .diagnose_source_q4_peak_bracket_correction import ANCHOR, CORRECTION
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/source_aligned_dense_row_whole_mode_plan.json"


def _chart(mode):
    epoch, step, values = mode[:3]
    return epoch + np.arange(len(values))*step, values


def _fitted(candidate, reference, mass_seconds):
    return fitted_metrics(*_chart(candidate), *_chart(reference),
                          mass_seconds, "quadratic_power")


def _hash(mode):
    return hashlib.sha256(mode[2].tobytes()).hexdigest()


def _active(fit, dense_row):
    matrix = np.asarray(fit["phase_matrix"], dtype=np.float64).copy()
    matrix[1] = dense_row
    phase = np.linalg.solve(matrix, np.asarray(fit["phase_rhs"], dtype=np.float64))
    return np.ascontiguousarray([*fit["coefficients"][:3], *phase], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    names = ("dense_evidence", "dense_script", "prior_evidence",
             "source_flux_plan", "override_adapter", "override_impl", "peak_impl")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-dense-row-whole-mode-plan.v1"
            or plan["release_qualified"] is not False
            or plan["steps_M"] != [0.00390625, 0.001953125]
            or plan["dense_stencils_M"] != [0.2, 0.1]
            or plan["dense_tolerance_index"] != 1
            or plan["mode_capacity"] != 600000
            or plan["peak_fit"] != "quadratic_power"
            or any(q.sha256(path) != plan[name + "_sha256"]
                   for name, path in paths.items())):
        parser.error("Frozen plan or input changed")
    dense = json.loads(paths["dense_evidence"].read_text())
    prior = json.loads(paths["prior_evidence"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (dense["status"] != "diagnostic_complete"
            or prior["status"] != "diagnostic_complete"
            or dense["release_qualified"] is not False
            or prior["release_qualified"] is not False
            or dense["row"] != prior["row"]):
        parser.error("Frozen dense-field or source result changed")
    row = prior["row"]
    mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
    report = {
        "schema": "diffgw.seobnrv4.source-aligned-dense-row-whole-mode.v1",
        "scope": dict(**q.SCOPE, development_only=True, dense_field_counterfactual=True),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "instrumentation_sha256": hashlib.sha256(
            (CORRECTION+OVERRIDE).encode()).hexdigest(),
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": row, "grids": {}, "adjacent": {}, "fixed_fine": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        libraries = _libraries(source_plan["pinned_wheel"])
        with tempfile.TemporaryDirectory(prefix="diffgw-source-aligned-dense-row-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(ANCHOR) != 1 or original.count(INCLUDE) != 1:
                raise ValueError("Pinned source patch anchors changed")
            source_file.write_text(original.replace(ANCHOR, CORRECTION+ANCHOR)
                                   .replace(INCLUDE, INCLUDE+OVERRIDE))
            library_path = Path(tmp) / "source.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["override_adapter"], library_path,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            library = ctypes.CDLL(str(library_path))
            baseline_function = library.diffgw_v4_full_source_mode
            baseline_function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            baseline_function.restype = ctypes.c_int
            override_function = library.diffgw_v4_full_source_mode_with_nqc
            override_function.argtypes = baseline_function.argtypes + [
                ctypes.POINTER(ctypes.c_double)]
            override_function.restype = ctypes.c_int
            modes = {}
            coefficients = {}
            for step in plan["steps_M"]:
                key = str(step)
                frozen = prior["grids"][key]
                dense_grid = dense["grids"][key]
                if (dense_grid["input_sha256"] != frozen["input_sha256"]
                        or not np.array_equal(dense_grid["local_phase_matrix"],
                                              frozen["local_fit"]["phase_matrix"])
                        or not np.array_equal(dense_grid["local_phase_rhs"],
                                              frozen["local_fit"]["phase_rhs"])):
                    raise ValueError(f"Dense/local inputs differ at {step:g} M")
                baseline = _source(baseline_function, row, step,
                                   plan["mode_capacity"])
                local_active = np.ascontiguousarray(
                    frozen["local_fit"]["coefficients"], dtype=np.float64)
                own_active = np.ascontiguousarray(frozen["source_active"],
                                                 dtype=np.float64)
                own = override_source(override_function, row, step,
                                      plan["mode_capacity"], own_active)
                local = override_source(override_function, row, step,
                                        plan["mode_capacity"], local_active)
                wheel = _public(row, step)
                if (_hash(baseline) != frozen["source_mode_sha256"]
                        or _hash(own) != _hash(baseline)
                        or own[:2] != baseline[:2]
                        or _hash(local) != frozen["local_mode_sha256"]
                        or _hash(wheel) != frozen["public_mode_sha256"]):
                    raise ValueError(f"Frozen full modes fail replay at {step:g} M")
                output = {
                    "source_mode_sha256": _hash(baseline),
                    "own_noop_mode_sha256": _hash(own),
                    "local_mode_sha256": _hash(local),
                    "public_mode_sha256": _hash(wheel),
                    "stencils": {},
                }
                output["replayed_local_minus_public"] = _fitted(
                    local, wheel, mass_seconds)
                if abs(output["replayed_local_minus_public"]["metrics"][
                    "complex_relative_l2"] - frozen["local_minus_public_mode"][
                        "metrics"]["complex_relative_l2"]) > 1e-12:
                    raise ValueError("Frozen local/public metric fails replay")
                modes[step] = {}
                coefficients[step] = {}
                for stencil in plan["dense_stencils_M"]:
                    label = str(stencil)
                    dense_row = dense_grid["centered_dop853"][
                        plan["dense_tolerance_index"]]["phase_second_rows"][label]
                    active = _active(frozen["local_fit"], dense_row)
                    expected = dense_grid["centered_dop853"][
                        plan["dense_tolerance_index"]][
                        "dense_row_replaced_coefficients"][label]
                    if np.max(abs(active[3:]-expected)) > 1e-9:
                        raise ValueError("Dense-row coefficient solve differs from prior")
                    candidate = override_source(override_function, row, step,
                                                plan["mode_capacity"], active)
                    output["stencils"][label] = {
                        "active_coefficients": active.tolist(),
                        "mode_sha256": _hash(candidate),
                        "candidate_minus_source_epoch_M": float(
                            (candidate[0]-baseline[0])/mass_seconds),
                        "candidate_minus_source_samples": len(candidate[2])-len(baseline[2]),
                        "candidate_minus_source": _fitted(candidate, baseline,
                                                          mass_seconds),
                        "candidate_minus_local": _fitted(candidate, local,
                                                         mass_seconds),
                        "candidate_minus_public": _fitted(candidate, wheel,
                                                          mass_seconds),
                    }
                    modes[step][label] = candidate
                    coefficients[step][label] = active
                first_label, second_label = (str(h) for h in plan["dense_stencils_M"])
                output["stencil_spread"] = {
                    "whole_mode": _fitted(modes[step][first_label],
                                          modes[step][second_label], mass_seconds),
                    "coefficient_difference_l2": float(np.linalg.norm(
                        coefficients[step][first_label]
                        - coefficients[step][second_label])),
                }
                report["grids"][key] = output
                q.write_evidence(args.json_out, report)
                print(json.dumps({"step_M": step, "stencils": list(output["stencils"])}),
                      flush=True)
            coarse, fine = plan["steps_M"]
            for stencil in plan["dense_stencils_M"]:
                label = str(stencil)
                report["adjacent"][label] = _fitted(
                    modes[coarse][label], modes[fine][label], mass_seconds)
                report["adjacent"][label]["coefficient_difference_l2"] = float(
                    np.linalg.norm(coefficients[fine][label]
                                   - coefficients[coarse][label]))
                fixed = override_source(override_function, row, coarse,
                                        plan["mode_capacity"],
                                        coefficients[fine][label])
                report["fixed_fine"][label] = {
                    "coarse_mode_sha256": _hash(fixed),
                    "adjacent_fixed_fine": _fitted(
                        fixed, modes[fine][label], mass_seconds),
                }
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

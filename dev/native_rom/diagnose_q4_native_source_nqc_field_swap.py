"""Dev-only aligned q4 NQC input profiles and single-field counterfactual fits."""

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
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_native_source_nqc_inputs import _fit, _native_samples
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture
from .diagnose_source_q4_ultrafine_chart import _native


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/q4_native_source_nqc_field_swap_plan.json"
FIELDS = ("amplitude", "phase", "radius", "radial_momentum", "orbital_frequency")


def _fit_array(gsl, time, values, match, targets):
    return _fit(gsl, time, *values.T, match, targets)


def _profile(gsl, time, native, source, match, half_width):
    near = abs(time-match) <= half_width
    if near.sum() < 100:
        raise ValueError("Insufficient common samples around NQC matching")
    phase_shift = (gsl.derivatives(time, source[:, 1], match)[0]
                   - gsl.derivatives(time, native[:, 1], match)[0])
    result = {}
    for index, name in enumerate(FIELDS):
        difference = native[near, index]-source[near, index]
        if name == "phase":
            difference += phase_shift
        result[name] = {
            "max_abs_difference": float(np.max(abs(difference))),
            "rms_difference": float(np.sqrt(np.mean(difference**2))),
            "native_at_match": float(gsl.derivatives(time, native[:, index], match)[0]),
            "source_at_match": float(gsl.derivatives(time, source[:, index], match)[0]),
        }
    return result, int(near.sum()), float(phase_shift)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    keys = ("prior_fit", "source_inputs", "capture_adapter",
            "source_flux_plan", "settings")
    paths = {key: q.ROOT / plan[key] for key in keys}
    if (plan["schema"] != "diffgw.seobnrv4.q4-native-source-nqc-field-swap-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row_name"] != "q4_nonspinning"
            or plan["output_step_M"] != 0.015625
            or plan["capacity"] != 50000
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in paths.items())):
        parser.error("Frozen plan or prerequisites changed")
    previous = json.loads(paths["prior_fit"].read_text())
    source_inputs = json.loads(paths["source_inputs"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    key = str(plan["output_step_M"])
    if (previous["status"] != "diagnostic_complete"
            or source_inputs["status"] != "diagnostic_complete"
            or previous["row"] != source_inputs["row"]):
        parser.error("Pinned NQC input or prior fit changed")
    report = {
        "schema": "diffgw.seobnrv4.q4-native-source-nqc-field-swap.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in paths.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": previous["row"], "output_step_M": plan["output_step_M"],
        "comparison": None, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        torch.set_num_threads(1)
        libraries = _libraries(source_plan["pinned_wheel"])
        gsl = GSLCubic(libraries["libgsl.28.dylib"])
        with tempfile.TemporaryDirectory(prefix="diffgw-q4-nqc-field-swap-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("Pinned-source NQC instrumentation anchor changed")
            source_file.write_text(original.replace(INCLUDE, INCLUDE + CAPTURE))
            library = Path(tmp) / "source_nqc_inputs.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["capture_adapter"], library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            function = ctypes.CDLL(str(library)).diffgw_v4_full_source_nqc_inputs
            function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            meta, source_values, source_coefficients = _capture(
                function, previous["row"], plan["output_step_M"], plan["capacity"])
            expected = source_inputs["grids"][key]
            if (hashlib.sha256(source_values.tobytes()).hexdigest() != expected["input_sha256"]
                    or abs(meta[1]-expected["internal_step_M"]) > 1e-12
                    or abs(meta[2]-expected["match_time_M"]) > 1e-10
                    or np.max(abs(source_coefficients[[0, 1, 2, 6, 7]]
                                  - np.asarray(expected["source_active"]))) > 1e-9):
                raise ValueError("Pinned-source fine NQC inputs changed")
        run = _native(previous["row"], plan["output_step_M"],
                      {"atol": plan["atol"], "rtol": plan["rtol"]},
                      q.Settings(**json.loads(paths["settings"].read_text())["settings"]),
                      return_nqc_coefficients=True, return_nqc_samples=True)
        native_time, native_columns, native_match = _native_samples(run["nqc_samples"])
        comparison = previous["comparison"]
        if (q.summarize_run(run)["array_sha256"] != comparison["native_mode_sha256"]
                or hashlib.sha256(native_columns.tobytes()).hexdigest()
                != comparison["native_nqc_sample_sha256"]):
            raise ValueError("Native fine NQC replay changed")
        source_time = np.arange(len(source_values), dtype=np.float64)*meta[1]
        translation = native_match-meta[2]
        native_indices = np.rint((source_time+translation-native_time[0])/meta[1]).astype(int)
        valid = (native_indices >= 0) & (native_indices < len(native_time))
        source_axis = source_time[valid]
        native_indices = native_indices[valid]
        if (len(source_axis) < 1000
                or np.max(abs(native_time[native_indices]-source_axis-translation)) > 1e-8
                or not np.all(np.diff(native_indices) == 1)):
            raise ValueError("Source/native NQC grids do not align by match time")
        common_native = native_columns.T[native_indices]
        common_source = source_values[valid]
        targets = np.asarray(expected["targets"])
        native_fit = _fit_array(gsl, source_axis, common_native, meta[2], targets)
        source_fit = _fit_array(gsl, source_axis, common_source, meta[2], targets)
        native_full = np.asarray(comparison["native_gsl_with_source_targets"]["coefficients"])
        source_full = np.asarray(comparison["source_gsl_with_source_targets"]["coefficients"])
        native_replay_gap = float(np.linalg.norm(np.asarray(native_fit["coefficients"])-native_full))
        source_replay_gap = float(np.linalg.norm(np.asarray(source_fit["coefficients"])-source_full))
        if max(native_replay_gap, source_replay_gap) > 1e-5:
            raise ValueError("Aligned common-grid GSL fits do not replay full fits")
        profile, near_count, phase_shift = _profile(
            gsl, source_axis, common_native, common_source, meta[2],
            plan["near_match_half_width_M"])
        swaps = {}
        for name, columns in {
                "amplitude_only": (0,), "phase_only": (1,),
                "radius_only": (2,), "radial_momentum_only": (3,),
                "orbital_frequency_only": (4,), "raw_mode": (0, 1),
                "orbital_state": (2, 3, 4), "all_fields": tuple(range(5)),
        }.items():
            changed = common_native.copy()
            changed[:, columns] = common_source[:, columns]
            fit = _fit_array(gsl, source_axis, changed, meta[2], targets)
            coefficients = np.asarray(fit["coefficients"])
            swaps[name] = {
                "columns": [FIELDS[index] for index in columns],
                "coefficients": coefficients.tolist(),
                "distance_from_native_l2": float(np.linalg.norm(
                    coefficients-np.asarray(native_fit["coefficients"]))),
                "distance_to_source_l2": float(np.linalg.norm(
                    coefficients-np.asarray(source_fit["coefficients"]))),
                "amplitude_matrix": fit["amplitude_matrix"],
                "phase_matrix": fit["phase_matrix"],
            }
        if np.linalg.norm(np.asarray(swaps["all_fields"]["coefficients"])
                          - np.asarray(source_fit["coefficients"])) > 1e-10:
            raise ValueError("All-fields source swap failed common-grid control")
        report["comparison"] = {
            "elapsed_seconds": time.monotonic()-started,
            "source_input_sha256": hashlib.sha256(source_values.tobytes()).hexdigest(),
            "native_input_sha256": hashlib.sha256(native_columns.tobytes()).hexdigest(),
            "common_samples": len(source_axis),
            "common_first_relative_to_match_M": float(source_axis[0]-meta[2]),
            "common_last_relative_to_match_M": float(source_axis[-1]-meta[2]),
            "near_match_samples": near_count,
            "match_time_translation_M": float(translation),
            "max_grid_alignment_error_M": float(np.max(abs(
                native_time[native_indices]-source_axis-translation))),
            "phase_constant_alignment_rad": phase_shift,
            "near_match_input_profile": profile,
            "common_native_fit": native_fit,
            "common_source_fit": source_fit,
            "native_full_fit_replay_l2": native_replay_gap,
            "source_full_fit_replay_l2": source_replay_gap,
            "field_swaps": swaps,
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

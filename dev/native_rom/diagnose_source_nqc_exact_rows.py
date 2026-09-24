"""Replay pinned source NQC fitting on source and wheel public states for three rows."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import tempfile

import numpy as np

from . import qualify_generator as q
from .diagnose_nqc_c_same_input import bind, compile_wrapper
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_public_nqc_mode_boundary import public_call
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_wheel_state_fit import ACTIVE, local_indices, source_peak_rule


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_exact_rows_plan.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / value["path"] for name, value in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-exact-rows-plan.v1"
            or plan["release_qualified"] is not False
            or plan["rows"] != ["nonspinning", "mild_mixed_spin", "strong_aligned_spin"]
            or plan["step_M"] != 0.125 or plan["local_half_width_M"] != 4.0
            or plan["peak_half_width_M"] != 4.0
            or set(plan["public_high_sha256"]) != set(plan["rows"])
            or any(q.sha256(paths[name]) != value["sha256"]
                   for name, value in plan["inputs"].items())):
        parser.error("Frozen exact-row plan or prerequisites changed")
    frozen = json.loads(paths["source_wheel_mode"].read_text())
    public = json.loads(paths["matched_public"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    libraries = _libraries(source_plan["pinned_wheel"])
    gsl = GSLCubic(libraries["libgsl.28.dylib"])
    report = {
        "schema": "diffgw.seobnrv4.source-nqc-exact-rows.v1",
        "scope": dict(
            q.SCOPE, development_only=True,
            description="Three-row exact-source-call NQC fits on source and wheel public states.",
            known_limitations=[
                "The wheel's private raw-mode and NQC input arrays remain unavailable.",
                "The source C fit is separately compiled from the installed wheel.",
                "Three diagnostic rows do not qualify the model or its ROM.",
            ],
        ),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "pinned_libraries_sha256": {name: q.sha256(path) for name, path in libraries.items()},
        "rows": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-nqc-exact-rows-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
            exact_library = Path(tmp) / "exact.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["exact_state_adapter"], exact_library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            high_library = Path(tmp) / "high.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["high_adapter"], high_library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            nqc_library = Path(tmp) / "nqc.dylib"
            compile_wrapper(private, args.headers, args.gsl_include,
                            [str(path) for path in libraries.values()], nqc_library)
            c_fit, c_peak_offset = bind(nqc_library)
            exact = ctypes.CDLL(str(exact_library)).diffgw_v4_source_nqc_state
            exact.argtypes = [ctypes.c_double]*8 + [ctypes.POINTER(ctypes.c_double)]
            exact.restype = ctypes.c_int
            high_source = ctypes.CDLL(str(high_library)).diffgw_v4_full_source_high_dynamics
            high_source.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                                    ctypes.POINTER(ctypes.c_size_t),
                                    ctypes.POINTER(ctypes.c_double)]
            high_source.restype = ctypes.c_int
            pointer = ctypes.POINTER(ctypes.c_double)

            for name in plan["rows"]:
                source_row = next(item for item in frozen["rows"] if item["name"] == name)
                public_row = next((item for item in public["rows"]
                                   if item["name"] == name), None)
                dense = json.loads(paths[f"dense_{name}"].read_text())
                row = source_row["row"]
                meta = np.asarray(dense["source"]["meta"], dtype=np.float64)
                source_coefficients = np.asarray(source_row["source_nqc_coefficients"])[list(ACTIVE)]
                wheel_coefficients = np.asarray(source_row["wheel_nqc_coefficients"])[list(ACTIVE)]
                if (dense["row_name"] != name or dense["resample_factor"] != 8
                        or meta.shape != (8,) or abs(meta[1]-0.015625) > 1e-12
                        or np.max(abs(source_coefficients - dense["source"][
                            "active_nqc_coefficients"])) > 1e-9):
                    raise ValueError(f"Frozen source NQC metadata changed: {name}")
                masses = [row[key] for key in ("mass1", "mass2", "chi1", "chi2")]
                capacity = 50000
                buffer = np.empty(5*capacity, dtype=np.float64)
                count = ctypes.c_size_t()
                rc = high_source(*masses, row["omega_start"], plan["step_M"],
                                 capacity, ctypes.byref(count),
                                 buffer.ctypes.data_as(pointer))
                if rc or count.value < 100:
                    raise RuntimeError(f"Pinned source high dynamics failed: {name}, {rc}")
                source_high = buffer[:5*count.value].reshape(5, -1).copy()
                wheel_output, _ = public_call(row, plan["step_M"], 0)
                wheel_high = np.asarray(wheel_output[2].data, dtype=np.float64).reshape(5, -1)
                wheel_high_hash = q.canonical_hash(wheel_high.tolist())
                if (wheel_high_hash != plan["public_high_sha256"][name]
                        or public_row is not None
                        and (wheel_high_hash != public_row["public_high_dynamics_sha256"]
                             or wheel_high.shape[1] != public_row["public_high_dynamics_length"])):
                    raise ValueError(f"Frozen wheel high dynamics changed: {name}")
                _, wheel_reported = public_call(row, plan["step_M"], 1)
                if np.max(abs(np.asarray(wheel_reported)[list(ACTIVE)]
                              - wheel_coefficients)) > 1e-9:
                    raise ValueError(f"Frozen wheel NQC coefficients changed: {name}")

                def exact_inputs(high: np.ndarray, indices: np.ndarray) -> np.ndarray:
                    output = np.empty((len(indices), 5), dtype=np.float64)
                    modes = np.empty(len(indices), dtype=np.complex128)
                    values = (ctypes.c_double*3)()
                    for j, index in enumerate(indices):
                        if exact(*masses, *high[1:, index], values):
                            raise RuntimeError(f"Pinned exact raw mode failed: {name}")
                        output[j, 2:4] = high[[1, 3], index]
                        output[j, 4] = values[0]
                        modes[j] = complex(values[1], values[2])
                    output[:, 0] = abs(modes)
                    output[:, 1] = np.unwrap(np.angle(modes))
                    return output

                def c_nqc_fit(inputs: np.ndarray, peak: float, first: float) -> np.ndarray:
                    columns = [np.ascontiguousarray(inputs[:, i]) for i in range(5)]
                    result = np.empty(5, dtype=np.float64)
                    rc_fit = c_fit(*(column.ctypes.data_as(pointer) for column in columns),
                                   len(inputs), peak-first, float(meta[1]),
                                   row["mass1"], row["mass2"],
                                   (row["chi1"]-row["chi2"])/2,
                                   (row["chi1"]+row["chi2"])/2,
                                   result.ctypes.data_as(pointer))
                    if rc_fit or not np.isfinite(result).all():
                        raise RuntimeError(f"Pinned source C NQC fit failed: {name}, {rc_fit}")
                    return result

                offset = c_peak_offset(row["mass1"], row["mass2"],
                                       (row["chi1"]-row["chi2"])/2,
                                       (row["chi1"]+row["chi2"])/2)
                by_build = {}
                for build, high in (("source", source_high), ("wheel", wheel_high)):
                    axis = high[0]-high[0, 0]
                    if np.max(abs(np.diff(axis)-meta[1])) > 1e-9:
                        raise ValueError(f"Unexpected high-rate grid: {name}/{build}")
                    peak_indices = local_indices(axis, float(meta[0]),
                                                 plan["peak_half_width_M"])
                    peak_input = exact_inputs(high, peak_indices)
                    peak_result = source_peak_rule(gsl, axis[peak_indices],
                                                   peak_input[:, 4])
                    peak = peak_result["time_M"]
                    match = peak-offset
                    indices = local_indices(axis, match, plan["local_half_width_M"])
                    inputs = exact_inputs(high, indices)
                    fitted = c_nqc_fit(inputs, peak, float(axis[indices[0]]))
                    reported = source_coefficients if build == "source" else wheel_coefficients
                    by_build[build] = {
                        "high_samples": int(high.shape[1]),
                        "peak_rule": peak_result, "match_time_M": match,
                        "fitted_coefficients": fitted.tolist(),
                        "reported_coefficients": reported.tolist(),
                        "fitted_minus_reported_l2": float(np.linalg.norm(fitted-reported)),
                        "fitted_minus_reported_max_absolute": float(np.max(abs(fitted-reported))),
                        "sample_count": len(indices),
                    }
                if (abs(by_build["source"]["peak_rule"]["time_M"]-meta[0]) > 1e-5
                        or abs(by_build["source"]["match_time_M"]-meta[2]) > 1e-5
                        or by_build["source"]["fitted_minus_reported_max_absolute"] > 1e-3):
                    raise ValueError(f"Exact source NQC replay failed: {name}")
                report["rows"][name] = {
                    "row": row, "source": by_build["source"], "wheel": by_build["wheel"],
                    "wheel_public_high_dynamics_sha256": wheel_high_hash,
                    "source_to_wheel_reported_l2": float(np.linalg.norm(
                        source_coefficients-wheel_coefficients)),
                    "source_to_wheel_exact_fit_l2": float(np.linalg.norm(
                        np.asarray(by_build["source"]["fitted_coefficients"])
                        - np.asarray(by_build["wheel"]["fitted_coefficients"]))),
                }
                q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "diagnostic_failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        q.write_evidence(args.json_out, report)
        raise
    q.write_evidence(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

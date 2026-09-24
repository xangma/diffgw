"""Compare source and regular-field NQC factors on identical captured source states."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_high_rate import RESAMPLE_ANCHOR
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_factor_impact_plan.json"


def correction(values: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    _, _, radius, momentum, frequency = values.T
    p1 = momentum / (radius * frequency)
    q1 = p1**2
    amplitude_basis = np.column_stack((q1, q1/radius, q1/radius**1.5))
    phase_basis = np.column_stack((p1, p1*momentum**2))
    magnitude = 1 + amplitude_basis @ coefficients[:3]
    phase = phase_basis @ coefficients[3:]
    return magnitude * np.exp(1j*phase)


def window_summary(time: np.ndarray, match: float, baseline: np.ndarray,
                   counter: np.ndarray, half_width: float | None,
                   before_only: bool = False) -> dict:
    mask = np.ones(len(time), dtype=bool) if half_width is None else abs(time-match) <= half_width
    if before_only:
        mask &= time <= match
    if not np.any(mask):
        raise ValueError("Empty comparison window")
    source = baseline[mask]
    candidate = counter[mask]
    if np.min(abs(source)) < 1e-8:
        raise ValueError("Source correction is too close to zero for a relative comparison")
    ratio = candidate/source
    difference = candidate-source
    return {
        "samples": int(mask.sum()),
        "time_start_M": float(time[mask][0]),
        "time_end_M": float(time[mask][-1]),
        "max_relative_complex_difference": float(np.max(abs(ratio-1))),
        "rms_relative_complex_difference": float(np.sqrt(np.mean(abs(ratio-1)**2))),
        "max_absolute_complex_factor_difference": float(np.max(abs(difference))),
        "max_absolute_magnitude_ratio_minus_one": float(np.max(abs(abs(ratio)-1))),
        "max_absolute_wrapped_phase_difference_rad": float(np.max(abs(np.angle(ratio)))),
        "source_min_absolute_factor": float(np.min(abs(source))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    names = ("coefficient_field", "source_wheel_mode", "source_flux_plan", "adapter")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-factor-impact-plan.v1"
            or plan["release_qualified"] is not False
            or plan["rows"] != ["strong_aligned_spin", "nonspinning", "mild_mixed_spin"]
            or plan["factors"] != [8, 32]
            or plan["output_step_M"] != 0.125
            or plan["capacity"] != 50000
            or plan["window_half_widths_M"] != [2.0, 10.0]
            or any(q.sha256(paths[name]) != plan[name + "_sha256"] for name in names)):
        parser.error("Frozen factor-impact plan or prerequisites changed")
    coefficients = json.loads(paths["coefficient_field"].read_text())
    wheel = json.loads(paths["source_wheel_mode"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (coefficients["status"] != "diagnostic_complete"
            or wheel["status"] != "diagnostic_complete"
            or coefficients["release_qualified"] is not False):
        parser.error("Frozen source evidence is incomplete")
    rows = {item["name"]: item["row"] for item in wheel["rows"]}
    wheel_rows = {item["name"]: item for item in wheel["rows"]}
    libraries = _libraries(source_plan["pinned_wheel"])
    report = {
        "schema": "diffgw.seobnrv4.source-nqc-factor-impact.v1",
        "scope": dict(**q.SCOPE, development_only=True),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "pinned_libraries_sha256": {name: q.sha256(path) for name, path in libraries.items()},
        "instrumentation_sha256": hashlib.sha256(CAPTURE.encode()).hexdigest(),
        "rows": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-nqc-factor-impact-") as tmp:
            private = Path(tmp)/"source"
            _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
            source_file = private/"lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1 or original.count(RESAMPLE_ANCHOR) != 1:
                raise ValueError("Pinned source diagnostic anchors changed")
            for factor in plan["factors"]:
                source_file.write_text(original.replace(INCLUDE, INCLUDE+CAPTURE).replace(
                    RESAMPLE_ANCHOR,
                    f"\n\n  resampFac = {factor}u; /* diagnostic high-rate grid */"+RESAMPLE_ANCHOR))
                dylib = Path(tmp)/f"capture_f{factor}.dylib"
                _compile(private, args.headers, args.gsl_include, libraries,
                         paths["adapter"], dylib,
                         source_plan["full_generator_source_includes"],
                         source_plan["full_generator_public_includes"],
                         require_wheel_flux=False)
                function = ctypes.CDLL(str(dylib)).diffgw_v4_full_source_nqc_inputs
                function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
                function.restype = ctypes.c_int
                for name in plan["rows"]:
                    meta, values, all_coefficients = _capture(
                        function, rows[name], plan["output_step_M"], plan["capacity"])
                    reference = coefficients["rows"][name]["factors"][str(factor)]
                    if (hashlib.sha256(values.tobytes()).hexdigest() != reference["input_array_sha256"]
                            or abs(meta[2]-reference["match_time_M"]) > 1e-12
                            or max(abs(all_coefficients[[0, 1, 2, 6, 7]]
                                        -reference["source_active_coefficients"])) > 1e-9):
                        raise ValueError(f"Captured source arrays fail frozen replay: {name}, {factor}")
                    time = np.arange(len(values), dtype=float)*meta[1]
                    source_coeff = np.asarray(reference["source_active_coefficients"])
                    dense_coeff = np.asarray(reference["dense_field_coefficients"])
                    baseline = correction(values, source_coeff)
                    counter = correction(values, dense_coeff)
                    by_factor = report["rows"].setdefault(name, {})
                    by_factor[str(factor)] = {
                        "input_array_sha256": reference["input_array_sha256"],
                        "samples": len(values), "match_time_M": float(meta[2]),
                        "internal_step_M": float(meta[1]),
                        "windows": {str(width): window_summary(time, meta[2], baseline,
                                                                 counter, width)
                                    for width in plan["window_half_widths_M"]},
                        "pre_match_windows": {
                            str(width): window_summary(time, meta[2], baseline,
                                                       counter, width, before_only=True)
                            for width in plan["window_half_widths_M"]},
                        "full_high_rate": window_summary(time, meta[2], baseline,
                                                         counter, None),
                    }
                    if factor == 8:
                        wheel_row = wheel_rows[name]
                        if max(abs(np.asarray(wheel_row["source_nqc_coefficients"])[[0, 1, 2, 6, 7]]
                                   -source_coeff)) > 1e-9:
                            raise ValueError(f"Pinned wheel result source coefficients differ: {name}")
                        wheel_coeff = np.asarray(wheel_row["wheel_nqc_coefficients"])[[0, 1, 2, 6, 7]]
                        wheel_factor = correction(values, wheel_coeff)
                        by_factor[str(factor)]["default_wheel_reference"] = {
                            "wheel_active_coefficients": wheel_coeff.tolist(),
                            "source_coefficient_l2": float(np.linalg.norm(source_coeff-wheel_coeff)),
                            "dense_coefficient_l2": float(np.linalg.norm(dense_coeff-wheel_coeff)),
                            "pre_match_windows": {
                                str(width): {
                                    "source_vs_wheel": window_summary(
                                        time, meta[2], wheel_factor, baseline, width,
                                        before_only=True),
                                    "dense_vs_wheel": window_summary(
                                        time, meta[2], wheel_factor, counter, width,
                                        before_only=True),
                                }
                                for width in plan["window_half_widths_M"]},
                        }
                    q.write_evidence(args.json_out, report)
                source_file.write_text(original)
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

"""Isolate pinned-source v4 high-rate NQC grid convergence at fixed coarse output."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile

import numpy as np

from . import qualify_generator as q
from .diagnose_native_source_nqc_inputs import _fit
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_high_rate_plan.json"
RESAMPLE_ANCHOR = "\n\n  /* Allocate the values vector to contain the initial conditions */"


class LocalPolynomial:
    def __init__(self, half_width: float, degree: int):
        self.half_width = half_width
        self.degree = degree

    def derivatives(self, time, values, match):
        time = np.asarray(time)
        values = np.asarray(values)
        mask = abs(time - match) <= self.half_width
        x = (time[mask] - match) / self.half_width
        if len(x) < 2 * (self.degree + 1):
            raise ValueError("Insufficient local polynomial derivative samples")
        matrix = np.vander(x, N=self.degree + 1, increasing=True)
        coefficients = np.linalg.lstsq(matrix, values[mask], rcond=None)[0]
        return [float(coefficients[0]), float(coefficients[1] / self.half_width),
                float(2 * coefficients[2] / self.half_width**2)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    names = ("source_flux_plan", "source_wheel_mode", "source_peak_grid", "adapter")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-high-rate-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["output_step_M"] != 0.125
            or plan["resample_factors"] != [4, 8, 16, 32]
            or plan["polynomial_degree"] != 4
            or plan["polynomial_half_widths_M"] != [0.5, 1.0, 2.0]
            or plan["common_match_reference_factor"] != 32
            or plan["capacity"] != 50000
            or any(q.sha256(paths[name]) != plan[name + "_sha256"] for name in names)):
        parser.error("Frozen plan or prerequisites changed")
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    prior = json.loads(paths["source_wheel_mode"].read_text())
    peak = json.loads(paths["source_peak_grid"].read_text())
    if prior["status"] != "diagnostic_complete" or peak["status"] != "diagnostic_complete":
        parser.error("Pinned source results unavailable")
    row = next(item["row"] for item in prior["rows"]
               if item["name"] == plan["row_name"])
    baseline = next(item for item in peak["rows"]
                    if item["output_step_M"] == plan["output_step_M"])
    libraries = _libraries(source_plan["pinned_wheel"])
    gsl = GSLCubic(libraries["libgsl.28.dylib"])
    report = dict(schema="diffgw.seobnrv4.source-nqc-high-rate.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={name: q.sha256(path) for name, path in paths.items()},
                  instrumentation_sha256=hashlib.sha256(CAPTURE.encode()).hexdigest(),
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), rows=[], comparisons={}, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        captures = []
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-nqc-high-rate-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1 or original.count(RESAMPLE_ANCHOR) != 1:
                raise ValueError("Pinned source diagnostic anchors changed")
            for factor in plan["resample_factors"]:
                instrumented = original.replace(INCLUDE, INCLUDE + CAPTURE)
                instrumented = instrumented.replace(
                    RESAMPLE_ANCHOR,
                    f"\n\n  resampFac = {factor}u; /* diagnostic high-rate grid */"
                    + RESAMPLE_ANCHOR)
                source_file.write_text(instrumented)
                dylib = Path(tmp) / f"source_nqc_high_rate_{factor}.dylib"
                _compile(private, args.headers, args.gsl_include, libraries,
                         paths["adapter"], dylib,
                         source_plan["full_generator_source_includes"],
                         source_plan["full_generator_public_includes"],
                         require_wheel_flux=False)
                function = ctypes.CDLL(str(dylib)).diffgw_v4_full_source_nqc_inputs
                function.argtypes = [ctypes.c_double] * 6 + [ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
                function.restype = ctypes.c_int
                meta, values, coefficients = _capture(function, row,
                    plan["output_step_M"], plan["capacity"])
                time = np.arange(len(values), dtype=np.float64) * meta[1]
                captures.append((time, values, meta[3:]))
                match = float(meta[2])
                targets = meta[3:]
                fit = _fit(gsl, time, *values.T, match, targets)
                polynomial = {}
                for width in plan["polynomial_half_widths_M"]:
                    local = _fit(LocalPolynomial(width, plan["polynomial_degree"]),
                                 time, *values.T, match, targets)
                    polynomial[f"{width:g}M"] = dict(
                        coefficients=local["coefficients"],
                        amplitude_second_derivative=local[
                            "amplitude_derivatives"][2],
                        amplitude_condition=local["amplitude_condition"],
                        phase_condition=local["phase_condition"])
                input_derivatives = np.asarray([
                    gsl.derivatives(time, values[:, j], match) for j in range(5)])
                active = coefficients[[0, 1, 2, 6, 7]]
                replay_error = float(np.max(abs(np.asarray(fit["coefficients"]) - active)))
                if (replay_error > 1e-3 or not np.isfinite(values).all()
                        or abs(meta[1] - plan["output_step_M"] / factor) > 1e-12):
                    raise ValueError("High-rate source NQC capture or GSL replay changed")
                array_hash = hashlib.sha256(values.tobytes()).hexdigest()
                if factor == 8 and (array_hash != baseline["input_array_sha256"]
                        or np.max(abs(active - np.asarray(
                            baseline["active_nqc_coefficients"]))) > 1e-9):
                    raise ValueError("Unmodified high-rate control changed")
                report["rows"].append(dict(resample_factor=factor,
                    requested_output_step_M=plan["output_step_M"],
                    internal_step_M=float(meta[1]), samples=len(values),
                    time_peak_M=float(meta[0]), match_time_M=match,
                    match_grid_fraction=float(match / meta[1] % 1),
                    targets=targets.tolist(), input_array_sha256=array_hash,
                    instrumented_source_sha256=q.sha256(source_file),
                    active_nqc_coefficients=active.tolist(),
                    input_at_match_value_first_second=input_derivatives.tolist(),
                    amplitude_second_derivative=fit["amplitude_derivatives"][2],
                    phase_second_derivative=fit["frequency_derivatives"][1],
                    full_fit_replay_max_abs_error=replay_error,
                    amplitude_condition=fit["amplitude_condition"],
                    phase_condition=fit["phase_condition"],
                    amplitude_matrix=fit["amplitude_matrix"],
                    phase_matrix=fit["phase_matrix"],
                    amplitude_rhs=fit["amplitude_rhs"],
                    phase_rhs=fit["phase_rhs"]))
                report["rows"][-1]["local_polynomial"] = polynomial
                q.write_evidence(args.json_out, report)
                print(f"resample_factor={factor}", flush=True)
        common_match = report["rows"][-1]["match_time_M"]
        if report["rows"][-1]["resample_factor"] != plan["common_match_reference_factor"]:
            raise ValueError("Common-match reference factor changed")
        report["common_match_time_M"] = common_match
        for item, (time, values, targets) in zip(report["rows"], captures):
            local = _fit(LocalPolynomial(2.0, plan["polynomial_degree"]),
                         time, *values.T, common_match, targets)
            full = _fit(gsl, time, *values.T, common_match, targets)
            item["common_match_counterfactual"] = dict(
                polynomial_2M_coefficients=local["coefficients"],
                polynomial_2M_amplitude_second_derivative=local[
                    "amplitude_derivatives"][2],
                gsl_coefficients=full["coefficients"],
                gsl_amplitude_second_derivative=full["amplitude_derivatives"][2])
        for first, second in zip(report["rows"], report["rows"][1:]):
            key = f"{first['resample_factor']}_to_{second['resample_factor']}"
            a = np.asarray(first["active_nqc_coefficients"])
            b = np.asarray(second["active_nqc_coefficients"])
            first_input = np.asarray(first["input_at_match_value_first_second"])
            second_input = np.asarray(second["input_at_match_value_first_second"])
            report["comparisons"][key] = dict(
                coefficient_l2=float(np.linalg.norm(a - b)),
                coefficient_max_abs=float(np.max(abs(a - b))),
                amplitude_coefficient_l2=float(np.linalg.norm(a[:3] - b[:3])),
                phase_coefficient_l2=float(np.linalg.norm(a[3:] - b[3:])),
                nonphase_value_relative_change=(
                    second_input[[0, 2, 3, 4], 0] /
                    first_input[[0, 2, 3, 4], 0] - 1).tolist(),
                nonphase_second_derivative_relative_change=(
                    second_input[[0, 2, 3, 4], 2] /
                    first_input[[0, 2, 3, 4], 2] - 1).tolist(),
                peak_shift_M=second["time_peak_M"] - first["time_peak_M"],
                match_shift_M=second["match_time_M"] - first["match_time_M"],
                amplitude_curvature_relative_change=(
                    second["amplitude_second_derivative"]
                    / first["amplitude_second_derivative"] - 1))
            report["comparisons"][key]["local_polynomial"] = {
                window: dict(coefficient_l2=float(np.linalg.norm(
                    np.asarray(first["local_polynomial"][window]["coefficients"])
                    - np.asarray(second["local_polynomial"][window]["coefficients"]))),
                    amplitude_curvature_relative_change=(
                    second["local_polynomial"][window]["amplitude_second_derivative"] /
                    first["local_polynomial"][window]["amplitude_second_derivative"] - 1))
                for window in first["local_polynomial"]}
            report["comparisons"][key]["common_match_counterfactual"] = {
                method: float(np.linalg.norm(
                    np.asarray(first["common_match_counterfactual"][method]) -
                    np.asarray(second["common_match_counterfactual"][method])))
                for method in ("polynomial_2M_coefficients", "gsl_coefficients")}
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

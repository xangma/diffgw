"""Check fixed-coarse source NQC high-rate sensitivity on additional spin rows."""

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
from .diagnose_source_nqc_high_rate import LocalPolynomial, RESAMPLE_ANCHOR
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_high_rate_domain_plan.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    names = ("source_flux_plan", "source_wheel_mode", "high_rate_result", "adapter")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-high-rate-domain-plan.v1"
            or plan["rows"] != ["nonspinning", "mild_mixed_spin"]
            or plan["output_step_M"] != 0.125
            or plan["resample_factors"] != [8, 16, 32]
            or plan["polynomial_degree"] != 4
            or plan["polynomial_half_width_M"] != 2.0
            or plan["capacity"] != 50000
            or any(q.sha256(paths[name]) != plan[name + "_sha256"] for name in names)):
        parser.error("Frozen plan or prerequisites changed")
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    prior = json.loads(paths["source_wheel_mode"].read_text())
    previous = json.loads(paths["high_rate_result"].read_text())
    if prior["status"] != "diagnostic_complete" or previous["status"] != "diagnostic_complete":
        parser.error("Pinned source results unavailable")
    references = {item["name"]: item for item in prior["rows"]}
    libraries = _libraries(source_plan["pinned_wheel"])
    gsl = GSLCubic(libraries["libgsl.28.dylib"])
    polynomial = LocalPolynomial(plan["polynomial_half_width_M"],
                                 plan["polynomial_degree"])
    report = dict(schema="diffgw.seobnrv4.source-nqc-high-rate-domain.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={name: q.sha256(path) for name, path in paths.items()},
                  instrumentation_sha256=hashlib.sha256(CAPTURE.encode()).hexdigest(),
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), rows=[], comparisons={}, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        captures = {name: [] for name in plan["rows"]}
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-nqc-high-domain-") as tmp:
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
                dylib = Path(tmp) / f"source_nqc_high_rate_domain_{factor}.dylib"
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
                for name in plan["rows"]:
                    reference = references[name]
                    meta, values, coefficients = _capture(function, reference["row"],
                        plan["output_step_M"], plan["capacity"])
                    time = np.arange(len(values), dtype=np.float64) * meta[1]
                    match = float(meta[2])
                    targets = meta[3:]
                    captures[name].append((time, values, targets))
                    full = _fit(gsl, time, *values.T, match, targets)
                    local = _fit(polynomial, time, *values.T, match, targets)
                    active = coefficients[[0, 1, 2, 6, 7]]
                    replay_error = float(np.max(abs(np.asarray(full["coefficients"])
                                                    - active)))
                    if (replay_error > 1e-3 or not np.isfinite(values).all()
                            or abs(meta[1] - plan["output_step_M"] / factor) > 1e-12):
                        raise ValueError(f"{name}: source capture or fit replay changed")
                    if factor == 8 and np.max(abs(coefficients - np.asarray(
                            reference["source_nqc_coefficients"]))) > 1e-9:
                        raise ValueError(f"{name}: unmodified source control changed")
                    report["rows"].append(dict(name=name, resample_factor=factor,
                        internal_step_M=float(meta[1]), samples=len(values),
                        time_peak_M=float(meta[0]), match_time_M=match,
                        targets=targets.tolist(),
                        input_array_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                        active_nqc_coefficients=active.tolist(),
                        full_fit_replay_max_abs_error=replay_error,
                        gsl_amplitude_second_derivative=full[
                            "amplitude_derivatives"][2],
                        polynomial_2M_coefficients=local["coefficients"],
                        polynomial_2M_amplitude_second_derivative=local[
                            "amplitude_derivatives"][2]))
                    q.write_evidence(args.json_out, report)
                    print(f"{name} resample_factor={factor}", flush=True)
        for name in plan["rows"]:
            selected = [item for item in report["rows"] if item["name"] == name]
            common_match = selected[-1]["match_time_M"]
            for item, (time, values, targets) in zip(selected, captures[name]):
                full = _fit(gsl, time, *values.T, common_match, targets)
                local = _fit(polynomial, time, *values.T, common_match, targets)
                item["common_match_counterfactual"] = dict(
                    match_time_M=common_match, gsl_coefficients=full["coefficients"],
                    polynomial_2M_coefficients=local["coefficients"])
            comparisons = []
            for first, second in zip(selected, selected[1:]):
                comparisons.append(dict(first_factor=first["resample_factor"],
                    second_factor=second["resample_factor"],
                    peak_shift_M=second["time_peak_M"] - first["time_peak_M"],
                    source_gsl_coefficient_l2=float(np.linalg.norm(
                        np.asarray(first["active_nqc_coefficients"]) -
                        np.asarray(second["active_nqc_coefficients"]))),
                    source_polynomial_2M_coefficient_l2=float(np.linalg.norm(
                        np.asarray(first["polynomial_2M_coefficients"]) -
                        np.asarray(second["polynomial_2M_coefficients"]))),
                    common_match_gsl_coefficient_l2=float(np.linalg.norm(
                        np.asarray(first["common_match_counterfactual"]["gsl_coefficients"])
                        - np.asarray(second["common_match_counterfactual"]["gsl_coefficients"]))),
                    common_match_polynomial_2M_coefficient_l2=float(np.linalg.norm(
                        np.asarray(first["common_match_counterfactual"][
                            "polynomial_2M_coefficients"])
                        - np.asarray(second["common_match_counterfactual"][
                            "polynomial_2M_coefficients"])))))
            report["comparisons"][name] = comparisons
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

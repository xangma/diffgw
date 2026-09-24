"""Capture pinned-source q4/q8 orbital peaks and NQC matching times.

This uses a private, checksum-verified LAL source snapshot and the existing
development-only NQC input adapter. It does not modify production waveforms.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile

import numpy as np
import lal

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_peak
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture
from .diagnose_source_wheel_mode import _public, _public_nqc, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_q4_q8_attachment_plan.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / value for name, value in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-q8-attachment-plan.v1"
            or plan["rows"] != ["q4_mixed_spin", "q8_high_aligned"]
            or plan["output_step_M"] != 0.5 or plan["capacity"] != 50000
            or plan["release_qualified"] is not False
            or {name: q.sha256(path) for name, path in paths.items()}
               != plan["input_sha256"]):
        parser.error("frozen source attachment plan or inputs differ")
    campaign = json.loads(paths["campaign"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (campaign["status"] != "diagnostic_complete"
            or q.lal_binary_identity(campaign["lal_binary"]["lal_version"],
                                     campaign["lal_binary"]["lalsimulation_version"])
               != campaign["lal_binary"]):
        parser.error("pinned campaign or public binary differs")
    libraries = _libraries(source_plan["pinned_wheel"])
    scope = dict(q.SCOPE)
    scope["description"] = ("Peak-relative source/native attachment timing and "
                            "source/wheel NQC coefficients on two fixed rows; "
                            "absolute generator time origins are different.")
    scope["known_limitations"] = [
        item for item in q.SCOPE["known_limitations"]
        if item != "No time, phase, amplitude, or coefficient fitting is performed."]
    scope["known_limitations"].append(
        "A fixed three-sample quadratic-power fit locates each generated mode's "
        "own amplitude peak; no waveform is fit to another.")
    scope["known_limitations"].append(
        "The wheel mode epoch is interpreted using the pinned source's "
        "orbital-peak convention; the wheel's private peak state is not exposed.")
    scope["known_limitations"].append(
        "The frozen q8 native campaign predates the peak-preserving ringdown "
        "variant used by the later public inverse. Its native fitted peak is "
        "taken only from the frozen campaign; the later variant is reported "
        "separately and never combined with that campaign's boundary.")
    report = dict(schema="diffgw.seobnrv4.source-q4-q8-attachment.v1",
                  scope=scope, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256=plan["input_sha256"], lal_binary=campaign["lal_binary"],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv),
                  rows={}, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-q4-q8-attachment-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("pinned NQC instrumentation anchor changed")
            source_file.write_text(original.replace(INCLUDE, INCLUDE + CAPTURE))
            report["instrumented_source_sha256"] = q.sha256(source_file)
            dylib = Path(tmp) / "source_q4_q8_attachment.dylib"
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
            mode_dylib = Path(tmp) / "source_q4_q8_mode.dylib"
            # The mode adapter includes the same verified source as a separate
            # translation unit, without the NQC instrumentation.
            source_file.write_text(original)
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["mode_adapter"], mode_dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            mode_function = ctypes.CDLL(str(mode_dylib)).diffgw_v4_full_source_mode
            mode_function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            mode_function.restype = ctypes.c_int
            for name in plan["rows"]:
                item = campaign["rows"][name]
                row = item["row"]
                meta, values, coefficients = _capture(
                    function, row, plan["output_step_M"], plan["capacity"])
                wheel_coefficients = _public_nqc(row, plan["output_step_M"])
                if (not np.isfinite(meta).all() or not np.isfinite(values).all()
                        or not np.isfinite(coefficients).all()
                        or not np.isfinite(wheel_coefficients).all()
                        or abs(meta[1]-plan["output_step_M"]/32) > 1e-12):
                    raise ValueError(f"{name} source or wheel capture invalid")
                native = item["runs"]["ode_2"]["native"]
                source_boundary = meta[2]-math.fmod(meta[2], meta[1])
                native_boundary = native["boundary_time_M"]
                source_peak_to_match = float(meta[2]-meta[0])
                native_peak_to_match = native["match_time_M"]-native["peak_time_M"]
                source_boundary_from_peak = float(source_boundary-meta[0])
                native_boundary_from_peak = native_boundary-native["peak_time_M"]
                coefficient_difference = coefficients-wheel_coefficients
                mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
                source_epoch, source_dt, source_mode = _source(
                    mode_function, row, plan["output_step_M"], plan["capacity"])
                wheel_epoch, wheel_dt, wheel_mode, _ = _public(
                    row, plan["output_step_M"])
                source_peak_fit = fitted_peak(
                    source_epoch+np.arange(len(source_mode))*source_dt,
                    source_mode, "quadratic_power")
                wheel_peak_fit = fitted_peak(
                    wheel_epoch+np.arange(len(wheel_mode))*wheel_dt,
                    wheel_mode, "quadratic_power")
                source_mode_peak = source_peak_fit["time_s"]/mass_seconds
                wheel_mode_peak = wheel_peak_fit["time_s"]/mass_seconds
                public_inverse = json.loads(paths[
                    "q4_inverse" if name == "q4_mixed_spin" else "q8_inverse"
                ].read_text())
                if (public_inverse["status"] != "diagnostic_complete"
                        or public_inverse["row"] != row
                        or not public_inverse["ansatz_holdout_passed"]):
                    raise ValueError(f"{name} public inverse differs")
                public_inferred_boundary_from_fitted_peak = (
                    public_inverse["public_amplitude_fit"]["t0_M"])
                native_peak_fit = item["fitted_peak"]["native_peak"]
                if abs(native_peak_fit["offset_samples"]) > 1:
                    raise ValueError(f"{name} native peak fit is outside the peak sample")
                native_mode_peak = native_peak_fit["time_s"]/mass_seconds
                native_boundary_from_fitted_peak = (
                    native_boundary_from_peak-native_mode_peak)
                source_boundary_from_fitted_peak = (
                    source_boundary_from_peak-source_mode_peak)
                wheel_boundary_from_orbital_peak = (
                    public_inferred_boundary_from_fitted_peak+wheel_mode_peak)
                report["rows"][name] = dict(row=row,
                    source=dict(orbital_peak_time_M=float(meta[0]),
                                match_time_M=float(meta[2]),
                                boundary_time_M=float(source_boundary),
                                internal_step_M=float(meta[1]),
                                nqc_coefficients=coefficients.tolist(),
                                nqc_input_count=int(len(values))),
                    wheel=dict(nqc_coefficients=wheel_coefficients.tolist(),
                               source_minus_wheel_max_abs_nqc_coefficient=float(
                                   np.max(abs(coefficient_difference))),
                               source_minus_wheel_relative_l2_nqc_coefficient=float(
                                   np.linalg.norm(coefficient_difference)
                                   /np.linalg.norm(wheel_coefficients))),
                    native=dict(orbital_peak_time_M=native["peak_time_M"],
                                match_time_M=native["match_time_M"],
                                boundary_time_M=native_boundary),
                    peak_relative=dict(source_match_from_peak_M=source_peak_to_match,
                                       native_match_from_peak_M=native_peak_to_match,
                                       source_minus_native_match_offset_M=(
                                           source_peak_to_match-native_peak_to_match),
                                       source_boundary_from_peak_M=source_boundary_from_peak,
                                       native_boundary_from_peak_M=native_boundary_from_peak,
                                       source_minus_native_boundary_offset_M=(
                                           source_boundary_from_peak-native_boundary_from_peak)),
                    fitted_waveform_peak=dict(
                        source_mode_peak_from_orbital_peak_M=source_mode_peak,
                        wheel_mode_peak_from_orbital_peak_M=wheel_mode_peak,
                        native_mode_peak_from_orbital_peak_M=native_mode_peak,
                        source_fit_offset_samples=source_peak_fit["offset_samples"],
                        wheel_fit_offset_samples=wheel_peak_fit["offset_samples"],
                        native_fit_offset_samples=native_peak_fit["offset_samples"],
                        source_sampled_peak_from_boundary_M=(
                            source_mode_peak-source_peak_fit["offset_samples"]
                            *plan["output_step_M"]-source_boundary_from_peak),
                        public_sampled_peak_from_inferred_boundary_M=(
                            wheel_mode_peak-wheel_peak_fit["offset_samples"]
                            *plan["output_step_M"]-wheel_boundary_from_orbital_peak),
                        native_sampled_peak_from_boundary_M=(
                            native_mode_peak-native_peak_fit["offset_samples"]
                            *plan["output_step_M"]-native_boundary_from_peak),
                        source_boundary_from_own_mode_peak_M=source_boundary_from_fitted_peak,
                        public_inferred_boundary_from_own_mode_peak_M=(
                            public_inferred_boundary_from_fitted_peak),
                        native_boundary_from_own_mode_peak_M=native_boundary_from_fitted_peak,
                        later_native_variant_boundary_from_own_mode_peak_M=(
                            public_inverse["native_boundary"]["time_from_fitted_peak_M"]),
                        public_inferred_boundary_from_orbital_peak_M=(
                            wheel_boundary_from_orbital_peak),
                        source_minus_public_boundary_from_orbital_peak_M=(
                            source_boundary_from_peak-wheel_boundary_from_orbital_peak),
                        source_minus_public_boundary_in_own_peak_chart_M=(
                            source_boundary_from_fitted_peak
                            -public_inferred_boundary_from_fitted_peak),
                        source_minus_native_waveform_peak_from_orbital_peak_M=(
                            source_mode_peak-native_mode_peak)))
                q.write_evidence(args.json_out, report)
                print(f"captured={name}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Probe pinned-source/public-wheel cadence in an antialigned-spin sector.

This is a development diagnostic, not an NR comparison or release gate.
The parameter row, grids, peak convention, and source identities are frozen
before execution. Each mode is anchored only at its own local amplitude peak.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import traceback

import lal
import numpy as np

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_wheel_mode import _public, _public_nqc, _source, _source_nqc


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/source_antialigned_cadence_plan.json"
ACTIVE_NQC = (0, 1, 2, 6, 7)


def _time_mode(output):
    epoch, dt, mode = output[:3]
    if (not np.isfinite(mode).all() or mode.size < 10
            or not math.isfinite(epoch) or not math.isfinite(dt)):
        raise ValueError("Invalid generator mode")
    return epoch + np.arange(mode.size)*dt, mode


def _comparison(candidate, reference, mass_seconds, method):
    candidate_time, candidate_mode = _time_mode(candidate)
    reference_time, reference_mode = _time_mode(reference)
    return fitted_metrics(candidate_time, candidate_mode, reference_time,
                          reference_mode, mass_seconds, method)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    source_path = q.ROOT / plan["source_plan"]["path"]
    adapter_path = q.ROOT / plan["mode_adapter"]["path"]
    row = dict(mass1=30.0, mass2=10.0, chi1=-0.85, chi2=-0.85,
               omega_start=0.04)
    steps = [0.125, 0.0625, 0.03125]
    if (plan["schema"] != "diffgw.seobnrv4.source-antialigned-cadence-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row"] != row or plan["output_steps_M"] != steps
            or plan["capacity"] != 50000
            or plan["peak_fit"] != "quadratic_power"
            or q.sha256(source_path) != plan["source_plan"]["sha256"]
            or q.sha256(adapter_path) != plan["mode_adapter"]["sha256"]):
        parser.error("Frozen antialigned cadence plan or inputs differ")
    source_plan = json.loads(source_path.read_text())
    libraries = _libraries(source_plan["pinned_wheel"])
    mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
    report = dict(schema="diffgw.seobnrv4.source-antialigned-cadence.v1",
                  status="running", release_qualified=False,
                  scope={**q.SCOPE,
                         "description": "One development-only source/wheel antialigned cadence probe.",
                         "known_limitations": [
                             "The row starts at omega=0.04 and does not reproduce the NR initial frequency.",
                             "No native waveform, NR overlap, or domain-wide convergence is tested.",
                             "Both reference implementations share the historical model equations."]},
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={"source_plan":q.sha256(source_path),
                                "mode_adapter":q.sha256(adapter_path)},
                  pinned_wheel=source_plan["pinned_wheel"],
                  pinned_source={key:source_plan["pinned_source"][key] for key in
                                 ("tag", "tag_object", "peeled_commit")},
                  row=row, output_steps_M=steps, levels=[], adjacent=[],
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), errors=[])
    q.write_evidence(args.json_out, report)
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-anti-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            dylib = Path(tmp) / "source_mode.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     adapter_path, dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            library = ctypes.CDLL(str(dylib))
            mode_function = library.diffgw_v4_full_source_mode
            mode_function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            mode_function.restype = ctypes.c_int
            nqc_function = library.diffgw_v4_full_source_nqc
            nqc_function.argtypes = [ctypes.c_double]*6 + [
                ctypes.POINTER(ctypes.c_double)]
            nqc_function.restype = ctypes.c_int
            previous = None
            for step in steps:
                source = _source(mode_function, row, step, plan["capacity"])
                wheel = _public(row, step)
                source_coefficients = _source_nqc(nqc_function, row, step)
                wheel_coefficients = _public_nqc(row, step)
                for label, output in (("source",source), ("wheel",wheel)):
                    _time_mode(output)
                    if not math.isclose(output[1]/mass_seconds, step,
                                        rel_tol=0., abs_tol=1e-10):
                        raise ValueError(f"{label} output spacing changed")
                active_difference = (source_coefficients-wheel_coefficients)[
                    list(ACTIVE_NQC)]
                level = dict(step_M=step,
                             source_samples=int(source[2].size),
                             wheel_samples=int(wheel[2].size),
                             source_mode_sha256=hashlib.sha256(
                                 source[2].tobytes()).hexdigest(),
                             wheel_mode_sha256=hashlib.sha256(
                                 wheel[2].tobytes()).hexdigest(),
                             source_nqc=source_coefficients.tolist(),
                             wheel_nqc=wheel_coefficients.tolist(),
                             active_nqc_l2_difference=float(np.linalg.norm(
                                 active_difference)),
                             source_against_wheel=_comparison(
                                 source, wheel, mass_seconds, plan["peak_fit"]))
                report["levels"].append(level)
                if previous is not None:
                    prev_step, prev_source, prev_wheel, prev_source_nqc, prev_wheel_nqc = previous
                    report["adjacent"].append(dict(
                        coarse_step_M=prev_step, fine_step_M=step,
                        source=_comparison(prev_source, source, mass_seconds,
                                           plan["peak_fit"]),
                        wheel=_comparison(prev_wheel, wheel, mass_seconds,
                                          plan["peak_fit"]),
                        source_active_nqc_l2=float(np.linalg.norm(
                            (prev_source_nqc-source_coefficients)[list(ACTIVE_NQC)])),
                        wheel_active_nqc_l2=float(np.linalg.norm(
                            (prev_wheel_nqc-wheel_coefficients)[list(ACTIVE_NQC)]))))
                previous = (step, source, wheel, source_coefficients,
                            wheel_coefficients)
                q.write_evidence(args.json_out, report)
                print(f"step_M={step} complete", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc),
                                     traceback=traceback.format_exc()))
    report["elapsed_seconds"] = time.monotonic()-started
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

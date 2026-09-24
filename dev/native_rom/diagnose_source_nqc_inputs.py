"""Capture pre-fit NQC inputs from the pinned full v4 source generator."""

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
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_inputs_plan.json"
INCLUDE = '#include "LALSimIMREOBNQCCorrection.c"\n'
CAPTURE = """\
static int diffgw_capture_nqc_inputs(
    REAL8Vector *, REAL8Vector *, REAL8Vector *, REAL8Vector *, REAL8Vector *,
    INT4, INT4, REAL8, REAL8, REAL8, REAL8, REAL8, REAL8, EOBNonQCCoeffs *);
#define XLALSimIMRSpinEOBCalculateNQCCoefficientsV4 diffgw_capture_nqc_inputs
"""


def _capture(function, row: dict, step_M: float, capacity: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.empty((capacity, 5), dtype=np.float64)
    coefficients = np.empty(10, dtype=np.float64)
    count = ctypes.c_size_t()
    meta = np.empty(8, dtype=np.float64)
    rc = function(row["mass1"], row["mass2"], row["chi1"], row["chi2"],
                  row["omega_start"], step_M, capacity, ctypes.byref(count),
                  meta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  coefficients.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if rc or count.value < 10 or count.value > capacity:
        raise RuntimeError(f"Pinned-source NQC input capture failed: {rc}")
    return meta, values[:count.value].copy(), coefficients


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {key: q.ROOT / plan[key] for key in
             ("source_flux_plan", "source_wheel_mode", "adapter")}
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-inputs-plan.v1"
            or plan["rows"] != ["strong_aligned_spin"]
            or plan["step_M"] != 0.125 or plan["capacity"] != 50000
            or plan["array_asset"] != "dev/native_rom/evidence/2026-09-24/source_nqc_inputs.npz"
            or any(q.sha256(paths[key]) != plan[key+"_sha256"] for key in paths)):
        parser.error("Frozen plan or prerequisites changed")
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    prior = json.loads(paths["source_wheel_mode"].read_text())
    if prior["status"] != "diagnostic_complete":
        parser.error("Pinned source/wheel coefficient result unavailable")
    libraries = _libraries(source_plan["pinned_wheel"])
    report = dict(schema="diffgw.seobnrv4.source-nqc-inputs.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={key: q.sha256(path) for key, path in paths.items()},
                  instrumentation_sha256=hashlib.sha256(CAPTURE.encode()).hexdigest(),
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), rows=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-nqc-inputs-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("Pinned source instrumentation anchor changed")
            source_file.write_text(original.replace(INCLUDE, INCLUDE+CAPTURE))
            report["instrumented_source_sha256"] = q.sha256(source_file)
            dylib = Path(tmp) / "source_nqc_inputs.dylib"
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
                reference = next(item for item in prior["rows"] if item["name"] == name)
                row = reference["row"]
                meta, values, coefficients = _capture(function, row, plan["step_M"],
                                                       plan["capacity"])
                baseline = np.asarray(reference["source_nqc_coefficients"])
                if (not np.isfinite(values).all() or not np.isfinite(meta).all()
                        or np.max(abs(coefficients-baseline)) > 1e-9):
                    raise ValueError("Instrumented source changed the NQC fit")
                array_path = q.ROOT / plan["array_asset"]
                np.savez_compressed(array_path, values=values, meta=meta,
                                    coefficients=coefficients)
                report["array_asset"] = plan["array_asset"]
                report["array_asset_sha256"] = q.sha256(array_path)
                peak = meta[0]
                closest = int(round(peak/meta[1]))
                indices = sorted(set(np.clip(np.array((0, closest-800, closest-160,
                                                      closest-1, closest,
                                                      closest+1, len(values)-1)),
                                             0, len(values)-1).tolist()))
                report["rows"].append(dict(
                    name=name, row=row, samples=len(values),
                    time_peak_M=float(peak), internal_step_M=float(meta[1]),
                    match_time_M=float(meta[2]),
                    nqc_targets=meta[3:].tolist(),
                    amplitude_phase_r_pr_omega_sha256=hashlib.sha256(
                        values.tobytes()).hexdigest(),
                    source_nqc_coefficients=coefficients.tolist(),
                    max_coefficient_replay_error=float(np.max(abs(coefficients-baseline))),
                    sample_rows=[dict(index=i, time_from_peak_M=float(i*meta[1]-peak),
                                      amplitude_phase_r_pr_omega=values[i].tolist())
                                 for i in indices]))
                q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

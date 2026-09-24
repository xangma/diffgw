"""Check peak location and NQC match inputs across pinned-source output grids."""

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
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_peak_grid_plan.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    names = ("source_flux_plan", "source_wheel_mode", "source_wheel_grid", "adapter")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-peak-grid-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["output_steps_M"] != [0.125, 0.0625, 0.03125]
            or plan["capacity"] != 50000
            or any(q.sha256(paths[name]) != plan[name+"_sha256"] for name in names)):
        parser.error("Frozen plan or prerequisites changed")
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    prior = json.loads(paths["source_wheel_mode"].read_text())
    grid = json.loads(paths["source_wheel_grid"].read_text())
    if prior["status"] != "diagnostic_complete" or grid["status"] != "diagnostic_complete":
        parser.error("Pinned source results unavailable")
    row = next(item["row"] for item in prior["rows"]
               if item["name"] == plan["row_name"])
    libraries = _libraries(source_plan["pinned_wheel"])
    gsl = GSLCubic(libraries["libgsl.28.dylib"])
    report = dict(schema="diffgw.seobnrv4.source-nqc-peak-grid.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={name: q.sha256(path) for name, path in paths.items()},
                  instrumentation_sha256=hashlib.sha256(CAPTURE.encode()).hexdigest(),
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), rows=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-nqc-peak-grid-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("Pinned source instrumentation anchor changed")
            source_file.write_text(original.replace(INCLUDE, INCLUDE + CAPTURE))
            report["instrumented_source_sha256"] = q.sha256(source_file)
            dylib = Path(tmp) / "source_nqc_peak_grid.dylib"
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
            for step in plan["output_steps_M"]:
                meta, values, coefficients = _capture(function, row, step,
                                                       plan["capacity"])
                baseline = next(item for item in grid["rows"]
                                if item["output_step_M"] == step)
                if (not np.isfinite(values).all() or not np.isfinite(meta).all()
                        or np.max(abs(coefficients-np.asarray(
                            baseline["source_coefficients"]))) > 1e-9):
                    raise ValueError("Instrumented source changed the NQC fit")
                match = float(meta[2])
                time = np.arange(len(values), dtype=np.float64)*meta[1]
                sample = np.asarray([gsl.derivatives(time, values[:, j], match)
                                     for j in range(5)])
                report["rows"].append(dict(
                    output_step_M=step, internal_step_M=float(meta[1]),
                    samples=len(values), time_peak_M=float(meta[0]),
                    match_time_M=match, match_grid_fraction=float(match/meta[1] % 1),
                    input_at_match_value_first_second=sample.tolist(),
                    input_array_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                    active_nqc_coefficients=coefficients[[0, 1, 2, 6, 7]].tolist()))
                q.write_evidence(args.json_out, report)
                print(f"output_step_M={step}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

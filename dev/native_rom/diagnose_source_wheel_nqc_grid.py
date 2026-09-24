"""Measure pinned source and wheel NQC sensitivity to requested output spacing."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import socket
import sys
import tempfile

import numpy as np

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_wheel_mode import _public_nqc, _source_nqc


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_wheel_nqc_grid_plan.json"
ALLOWED_STEPS = ([0.125, 0.0625, 0.03125, 0.015625],
                 [0.125, 0.0078125, 0.00390625])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    names = ("source_flux_plan", "source_wheel_mode", "adapter")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-wheel-nqc-grid-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["output_steps_M"] not in ALLOWED_STEPS
            or any(q.sha256(paths[name]) != plan[name+"_sha256"] for name in names)):
        parser.error("Frozen plan or prerequisites changed")
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    prior = json.loads(paths["source_wheel_mode"].read_text())
    if prior["status"] != "diagnostic_complete":
        parser.error("Pinned source/wheel baseline unavailable")
    baseline = next(item for item in prior["rows"] if item["name"] == plan["row_name"])
    libraries = _libraries(source_plan["pinned_wheel"])
    report = dict(schema="diffgw.seobnrv4.source-wheel-nqc-grid.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(args.plan), script_sha256=q.sha256(__file__),
                  input_sha256={name: q.sha256(path) for name, path in paths.items()},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), rows=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-nqc-grid-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            dylib = Path(tmp) / "source_nqc_grid.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["adapter"], dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            function = ctypes.CDLL(str(dylib)).diffgw_v4_full_source_nqc
            function.argtypes = [ctypes.c_double]*6 + [ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            reference_source = np.asarray(baseline["source_nqc_coefficients"])
            reference_wheel = np.asarray(baseline["wheel_nqc_coefficients"])
            active = [0, 1, 2, 6, 7]
            for step in plan["output_steps_M"]:
                source = _source_nqc(function, baseline["row"], step)
                wheel = _public_nqc(baseline["row"], step)
                if step == plan["output_steps_M"][0] and (
                        np.max(abs(source-reference_source)) > 1e-9
                        or np.max(abs(wheel-reference_wheel)) > 1e-9):
                    raise ValueError("Pinned source/wheel NQC baseline failed replay")
                report["rows"].append(dict(
                    output_step_M=step,
                    source_coefficients=source.tolist(), wheel_coefficients=wheel.tolist(),
                    source_minus_baseline=(source-reference_source).tolist(),
                    wheel_minus_baseline=(wheel-reference_wheel).tolist(),
                    source_wheel_active_l2=float(np.linalg.norm((source-wheel)[active])),
                    source_baseline_active_l2=float(np.linalg.norm(
                        (source-reference_source)[active])),
                    wheel_baseline_active_l2=float(np.linalg.norm(
                        (wheel-reference_wheel)[active]))))
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

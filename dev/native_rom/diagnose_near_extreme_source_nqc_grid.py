"""Compare pinned-source pre-fit q10 NQC inputs at two output cadences."""

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
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/near_extreme_source_nqc_grid_v2_plan.json"
ACTIVE = (0, 1, 2, 6, 7)
COLUMNS = ("raw_amplitude", "raw_phase", "radius", "radial_momentum", "orbital_frequency")


def compare_inputs(first: tuple, second: tuple, half_width: float) -> dict:
    fmeta, fvalues, _ = first
    smeta, svalues, _ = second
    fstep, sstep = float(fmeta[1]), float(smeta[1])
    if abs(fstep - sstep) > 1e-12:
        raise ValueError("different internal NQC steps")
    ft = np.arange(len(fvalues)) * fstep - fmeta[2]
    st = np.arange(len(svalues)) * sstep - smeta[2]
    query = np.arange(-half_width, half_width + fstep/2, fstep)
    if query[0] < max(ft[0], st[0]) or query[-1] > min(ft[-1], st[-1]):
        raise ValueError("common NQC input window unavailable")
    controls = {}
    for index, name in enumerate(COLUMNS):
        a = np.interp(query, ft, fvalues[:, index])
        b = np.interp(query, st, svalues[:, index])
        if name == "raw_phase":
            midpoint = int(np.argmin(abs(query)))
            a -= a[midpoint]
            b -= b[midpoint]
        difference = a - b
        controls[name] = dict(
            max_absolute_difference=float(np.max(abs(difference))),
            rms_absolute_difference=float(np.sqrt(np.mean(difference**2))),
            scale=float(np.max(abs(b))),
            max_relative_to_scale=float(np.max(abs(difference)) / max(np.max(abs(b)), 1e-300)),
        )
    return dict(window_relative_to_NQC_match_M=[float(query[0]), float(query[-1])],
                samples=len(query), columns=controls)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.json_out.exists():
        parser.error("evidence path already exists")
    plan = json.loads(PLAN.read_text())
    paths = {name: q.ROOT / item["path"] for name, item in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.near-extreme-source-nqc-grid-plan.v2"
            or plan["row"] != "q10_both_near_extreme_aligned"
            or plan["output_steps_M"] != [0.03125, 0.015625]
            or plan["expected_internal_step_M"] != 0.015625
            or plan["local_half_width_M"] != 4.0
            or plan["capacity"] != 50000
            or plan["release_qualified"] is not False
            or any(q.sha256(paths[name]) != item["sha256"]
                   for name, item in plan["inputs"].items())):
        parser.error("frozen source NQC-grid plan or prerequisites changed")
    public = json.loads(paths["public_nqc"].read_text())
    parity = json.loads(paths["parity"].read_text())
    source_plan = json.loads(paths["source_plan"].read_text())
    if (public["status"] != "diagnostic_complete"
            or public["row"] != plan["row"]
            or public["lal_binary"] != parity["lal_binary"]
            or q.lal_binary_identity(public["lal_binary"]["lal_version"],
                                     public["lal_binary"]["lalsimulation_version"])
            != public["lal_binary"]):
        parser.error("pinned public evidence differs")
    row = parity["rows"][plan["row"]]["row"]
    libraries = _libraries(source_plan["pinned_wheel"])
    report = dict(
        schema="diffgw.seobnrv4.near-extreme-source-nqc-grid.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        input_sha256={name: q.sha256(path) for name, path in paths.items()},
        instrumentation_sha256=hashlib.sha256(CAPTURE.encode()).hexdigest(),
        pinned_libraries_sha256={name: q.sha256(path) for name, path in libraries.items()},
        lal_binary=public["lal_binary"], row=row, levels={}, comparison={}, errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-q10-source-nqc-grid-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("pinned source NQC include anchor changed")
            source_file.write_text(original.replace(INCLUDE, INCLUDE + CAPTURE))
            dylib = Path(tmp) / "source_nqc_inputs.dylib"
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
            captures = {}
            for step in plan["output_steps_M"]:
                meta, values, source_coefficients = _capture(
                    function, row, step, plan["capacity"])
                if (not np.isfinite(values).all() or not np.isfinite(meta).all()
                        or not np.isfinite(source_coefficients).all()
                        or abs(meta[1] - plan["expected_internal_step_M"]) > 1e-12):
                    raise ValueError("invalid pinned-source NQC inputs")
                wheel_coefficients = np.asarray(
                    public["levels"][f"dt_{step:.6f}M"], dtype=np.float64)
                captures[step] = meta, values, source_coefficients
                report["levels"][f"dt_{step:.6f}M"] = dict(
                    input_samples=len(values), input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                    meta=meta.tolist(), source_coefficients=source_coefficients.tolist(),
                    wheel_coefficients=wheel_coefficients.tolist(),
                    source_minus_wheel_active_l2=float(np.linalg.norm(
                        source_coefficients[list(ACTIVE)] - wheel_coefficients[list(ACTIVE)])),
                )
                q.write_evidence(args.json_out, report)
            coarse = captures[0.03125]
            fine = captures[0.015625]
            report["comparison"] = dict(
                source_coefficients_active_l2=float(np.linalg.norm(
                    coarse[2][list(ACTIVE)] - fine[2][list(ACTIVE)])),
                wheel_coefficients_active_l2=float(np.linalg.norm(
                    np.asarray(public["levels"]["dt_0.031250M"])[list(ACTIVE)]
                    - np.asarray(public["levels"]["dt_0.015625M"])[list(ACTIVE)])),
                peak_time_difference_M=float(coarse[0][0] - fine[0][0]),
                match_time_difference_M=float(coarse[0][2] - fine[0][2]),
                targets_max_absolute_difference=float(np.max(abs(coarse[0][3:] - fine[0][3:]))),
                inputs=compare_inputs(coarse, fine, plan["local_half_width_M"]),
            )
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

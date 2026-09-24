"""Compare low-rate v4 dynamics from the pinned source and compiled wheel."""

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

import lal
import lalsimulation as lalsim
import numpy as np

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_wheel_dynamics_plan.json"
FIELDS = ("r", "phi", "pr_star", "pphi")


def _public(row: dict, step_M: float) -> np.ndarray:
    mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
    output = lalsim.SimIMRSpinAlignedEOBModes(
        step_M*mass_seconds, row["mass1"]*lal.MSUN_SI,
        row["mass2"]*lal.MSUN_SI, row["omega_start"]/(math.pi*mass_seconds),
        100e6*lal.PC_SI, row["chi1"], row["chi2"], 4,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
        lal.CreateREAL8Vector(10), 0, lal.CreateDict(), lal.CreateDict())
    return np.asarray(output[1].data, dtype=np.float64).reshape(5, -1).copy()


def _source(function, row: dict, step_M: float, capacity: int) -> np.ndarray:
    values = np.empty(5*capacity, dtype=np.float64)
    count = ctypes.c_size_t()
    rc = function(row["mass1"], row["mass2"], row["chi1"], row["chi2"],
                  row["omega_start"], step_M, capacity, ctypes.byref(count),
                  values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if rc or count.value < 10 or count.value > capacity:
        raise RuntimeError(f"Source full-generator dynamics failed: {rc}")
    # The source writes five consecutive blocks of the *actual* length. The
    # provided capacity is only a bound, not a stride between blocks.
    return values[:5*count.value].reshape(5, count.value).copy()


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _compare(name: str, row: dict, source: np.ndarray, wheel: np.ndarray,
             plan: dict, frozen: dict | None) -> dict:
    count = min(source.shape[1], wheel.shape[1])
    if (count < 10 or not np.isfinite(source).all() or not np.isfinite(wheel).all()
            or np.max(abs(source[0, :count]-wheel[0, :count])) > 1e-8
            or abs(source[0, 0]) > 1e-12 or abs(wheel[0, 0]) > 1e-12):
        raise ValueError(f"Source/wheel grids cannot be compared by index: {name}")
    initial = dict(zip(FIELDS, (source[1:, 0]-wheel[1:, 0]).tolist()))
    src_omega = np.gradient(source[2, :count], plan["step_M"], edge_order=2)
    whl_omega = np.gradient(wheel[2, :count], plan["step_M"], edge_order=2)
    points = []
    for target in plan["frequency_targets_Momega"]:
        index = int(np.argmin(abs(whl_omega-target)))
        if not 3 <= index < count-3:
            raise ValueError(f"Frequency target {target} outside shared {name} grid")
        points.append(dict(
            target_Momega=target, index=index, time_M=float(wheel[0, index]),
            wheel_Momega=float(whl_omega[index]),
            source_minus_wheel_Momega=float(src_omega[index]-whl_omega[index]),
            source_minus_wheel=dict(zip(
                FIELDS, (source[1:, index]-wheel[1:, index]).tolist())),
            source_minus_wheel_phase_span_rad=float(
                (source[2, index]-source[2, 0])
                -(wheel[2, index]-wheel[2, 0]))))
    prior_first = None
    if frozen is not None:
        prior = frozen["source_full_generator_initial"]
        prior_first = dict(zip(
            ("r", "pr_star", "pphi"),
            (float(source[1, 0]-prior["r"]),
             float(source[3, 0]-prior["pr_star"]),
             float(source[4, 0]-prior["pphi"]))))
    return dict(name=name, row=row, step_M=plan["step_M"],
                source_shape=list(source.shape), wheel_shape=list(wheel.shape),
                source_array_sha256=_digest(source), wheel_array_sha256=_digest(wheel),
                common_samples=count,
                max_shared_time_difference_M=float(np.max(abs(
                    source[0, :count]-wheel[0, :count]))),
                initial_source_minus_wheel=initial,
                source_first_minus_prior_source=prior_first,
                frequency_points=points)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {key: q.ROOT / plan[key] for key in
             ("source_flux_result", "source_flux_plan", "aligned_waveform", "adapter")}
    if (plan["schema"] != "diffgw.seobnrv4.source-wheel-dynamics-plan.v1"
            or plan["rows"] != ["nonspinning", "mild_mixed_spin", "strong_aligned_spin"]
            or plan["step_M"] != 0.125
            or plan["max_samples"] != 50000
            or any(q.sha256(paths[key]) != plan[key+"_sha256"] for key in paths)):
        parser.error("Frozen plan or inputs changed")
    prior = json.loads(paths["source_flux_result"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    aligned = json.loads(paths["aligned_waveform"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or aligned["status"] != "diagnostic_complete"
            or not aligned["check"]["numerical_check_passed"]
            or q.lal_binary_identity(
                aligned["lal_binary"]["lal_version"],
                aligned["lal_binary"]["lalsimulation_version"])
            != aligned["lal_binary"]
            or any(aligned["lal_binary"][key] != source_plan["pinned_wheel"][key]
                   for key in ("lal_version", "lalsimulation_version",
                               "lalsuite_distribution"))):
        parser.error("Source or waveform prerequisite is incompatible")
    libraries = _libraries(source_plan["pinned_wheel"])
    rows = [item["row"] for item in prior["rows"]] + [aligned["row"]]
    report = dict(schema="diffgw.seobnrv4.source-wheel-dynamics.v1",
                  scope=q.SCOPE, status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={key: q.sha256(path) for key, path in paths.items()},
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), rows=[], errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-dynamics-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            dylib = Path(tmp) / "full_source_dynamics.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["adapter"], dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            function = ctypes.CDLL(str(dylib)).diffgw_v4_full_source_dynamics
            function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double)]
            function.restype = ctypes.c_int
            for name, row, frozen in zip(plan["rows"], rows,
                                         [*prior["rows"], None]):
                source = _source(function, row, plan["step_M"], plan["max_samples"])
                wheel = _public(row, plan["step_M"])
                report["rows"].append(_compare(name, row, source, wheel, plan, frozen))
                q.write_evidence(args.json_out, report)
                print(f"row={name}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])),
          flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

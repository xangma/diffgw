"""Test fixed local spline windows on captured pinned-source v4 NQC inputs."""

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


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_local_windows_plan.json"


def _local_fit(gsl, time, values, match, targets, half_width):
    mask = abs(time - match) <= half_width
    selected = time[mask]
    if len(selected) < 5 or not selected[0] < match < selected[-1]:
        raise ValueError("Local NQC matching window has insufficient support")
    fit = _fit(gsl, selected, *values[mask].T, match, targets)
    return fit, len(selected)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    names = ("source_flux_plan", "source_wheel_mode", "source_wheel_grid",
             "source_peak_grid", "adapter")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-local-windows-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["output_steps_M"] != [0.125, 0.0625, 0.03125]
            or plan["half_widths_M"] != [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
            or plan["capacity"] != 50000
            or any(q.sha256(paths[name]) != plan[name + "_sha256"] for name in names)):
        parser.error("Frozen plan or prerequisites changed")
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    prior = json.loads(paths["source_wheel_mode"].read_text())
    grid = json.loads(paths["source_wheel_grid"].read_text())
    peak = json.loads(paths["source_peak_grid"].read_text())
    if any(result["status"] != "diagnostic_complete" for result in (prior, grid, peak)):
        parser.error("Pinned source results unavailable")
    row = next(item["row"] for item in prior["rows"]
               if item["name"] == plan["row_name"])
    libraries = _libraries(source_plan["pinned_wheel"])
    gsl = GSLCubic(libraries["libgsl.28.dylib"])
    report = dict(schema="diffgw.seobnrv4.source-nqc-local-windows.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={name: q.sha256(path) for name, path in paths.items()},
                  instrumentation_sha256=hashlib.sha256(CAPTURE.encode()).hexdigest(),
                  process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                               pid=os.getpid(), argv=sys.argv), rows=[], comparisons={}, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-nqc-local-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"],
                      private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("Pinned source instrumentation anchor changed")
            source_file.write_text(original.replace(INCLUDE, INCLUDE + CAPTURE))
            report["instrumented_source_sha256"] = q.sha256(source_file)
            dylib = Path(tmp) / "source_nqc_local_windows.dylib"
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
            for step in plan["output_steps_M"]:
                meta, values, coefficients = _capture(function, row, step,
                                                       plan["capacity"])
                baseline = next(item for item in grid["rows"]
                                if item["output_step_M"] == step)
                peak_row = next(item for item in peak["rows"]
                                if item["output_step_M"] == step)
                array_hash = hashlib.sha256(values.tobytes()).hexdigest()
                if (not np.isfinite(values).all() or not np.isfinite(meta).all()
                        or np.max(abs(coefficients - np.asarray(
                            baseline["source_coefficients"]))) > 1e-9
                        or array_hash != peak_row["input_array_sha256"]):
                    raise ValueError("Captured source inputs changed")
                time = np.arange(len(values), dtype=np.float64) * meta[1]
                match = float(meta[2])
                targets = meta[3:]
                full = _fit(gsl, time, *values.T, match, targets)
                active = coefficients[[0, 1, 2, 6, 7]]
                replay_error = float(np.max(abs(np.asarray(full["coefficients"]) - active)))
                if replay_error > 1e-3:
                    raise ValueError("Pinned-source full-series GSL fit replay changed")
                variants = {"full": dict(coefficients=full["coefficients"],
                                         amplitude_second_derivative=full[
                                             "amplitude_derivatives"][2],
                                         samples=len(time))}
                for half_width in plan["half_widths_M"]:
                    local, count = _local_fit(gsl, time, values, match, targets,
                                              half_width)
                    local_coeff = np.asarray(local["coefficients"])
                    full_coeff = np.asarray(full["coefficients"])
                    key = f"{half_width:g}M"
                    variants[f"all_{key}"] = dict(
                        coefficients=local_coeff.tolist(),
                        amplitude_second_derivative=local[
                            "amplitude_derivatives"][2], samples=count,
                        amplitude_condition=local["amplitude_condition"],
                        phase_condition=local["phase_condition"])
                    variants[f"phase_only_{key}"] = dict(
                        coefficients=np.concatenate((full_coeff[:3],
                                                     local_coeff[3:])).tolist(),
                        samples=count)
                    variants[f"amplitude_only_{key}"] = dict(
                        coefficients=np.concatenate((local_coeff[:3],
                                                     full_coeff[3:])).tolist(),
                        samples=count)
                report["rows"].append(dict(output_step_M=step, samples=len(time),
                    internal_step_M=float(meta[1]), match_time_M=match,
                    time_peak_M=float(meta[0]), targets=targets.tolist(),
                    input_array_sha256=array_hash,
                    source_active_coefficients=active.tolist(),
                    full_fit_replay_max_abs_error=replay_error, variants=variants))
                q.write_evidence(args.json_out, report)
                print(f"output_step_M={step}", flush=True)
        for name in report["rows"][0]["variants"]:
            coeff = [np.asarray(item["variants"][name]["coefficients"])
                     for item in report["rows"]]
            pairwise = [float(np.linalg.norm(coeff[j] - coeff[i]))
                        for i, j in ((0, 1), (1, 2), (0, 2))]
            report["comparisons"][name] = dict(
                adjacent_and_endpoint_coefficient_l2=pairwise,
                maximum_coefficient_l2=max(pairwise),
                differences_from_source_coefficients_l2=[float(np.linalg.norm(
                    coeff[i] - np.asarray(report["rows"][i][
                        "source_active_coefficients"]))) for i in range(3)])
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

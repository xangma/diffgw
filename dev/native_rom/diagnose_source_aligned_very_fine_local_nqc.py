"""Dev-only fifth-grid corrected-source strong-spin local NQC check."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import lal
import numpy as np

from . import qualify_generator as q
from .diagnose_native_source_nqc_inputs import _fit
from .diagnose_peak_anchor import fitted_metrics
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_high_rate import LocalPolynomial
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture
from .diagnose_source_nqc_whole_mode_override import OVERRIDE, override_source
from .diagnose_source_q4_nqc_cadence_ladder import _local_peak
from .diagnose_source_q4_peak_bracket_correction import ANCHOR, CORRECTION, _function
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/source_aligned_very_fine_local_nqc_plan.json"


def _chart(mode):
    epoch, step, values = mode[:3]
    return epoch + np.arange(len(values))*step, values


def _fitted(candidate, reference, mass_seconds, peak_fit):
    return fitted_metrics(*_chart(candidate), *_chart(reference),
                          mass_seconds, peak_fit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    keys = ("previous_evidence", "previous_script", "peak_script",
            "source_flux_plan", "capture_adapter", "override_adapter",
            "mode_adapter", "polynomial_impl", "fit_impl")
    paths = {key: q.ROOT / plan[key] for key in keys}
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-very-fine-local-nqc-plan.v1"
            or plan["release_qualified"] is not False
            or plan["steps_M"] != [0.00390625, 0.001953125]
            or plan["fixed_coefficient_reference_step_M"] != plan["steps_M"][-1]
            or plan["polynomial_degree"] != 4
            or plan["polynomial_half_width_M"] != 2.0
            or plan["capture_capacity"] != 100000
            or plan["mode_capacity"] != 600000
            or plan["peak_fit"] != "quadratic_power"
            or any(q.sha256(path) != plan[key + "_sha256"]
                   for key, path in paths.items())):
        parser.error("Frozen plan or input changed")
    prior = json.loads(paths["previous_evidence"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or prior["release_qualified"] is not False
            or prior["row"] != {"mass1": 30.0, "mass2": 20.0,
                                "chi1": 0.8, "chi2": 0.8,
                                "omega_start": 0.04}
            or str(plan["steps_M"][0]) not in prior["grids"]):
        parser.error("Previous strong-spin evidence changed")
    row = prior["row"]
    mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    report = {
        "schema": "diffgw.seobnrv4.source-aligned-very-fine-local-nqc.v1",
        "scope": q.SCOPE, "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {key: q.sha256(path) for key, path in paths.items()},
        "correction_sha256": hashlib.sha256(CORRECTION.encode()).hexdigest(),
        "capture_sha256": hashlib.sha256(CAPTURE.encode()).hexdigest(),
        "override_sha256": hashlib.sha256(OVERRIDE.encode()).hexdigest(),
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": row, "grids": {}, "adjacent_local": {},
        "adjacent_source": {}, "adjacent_public": {},
        "adjacent_fixed_finest": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        started = time.monotonic()
        libraries = _libraries(source_plan["pinned_wheel"])
        polynomial = LocalPolynomial(plan["polynomial_half_width_M"],
                                     plan["polynomial_degree"])
        with tempfile.TemporaryDirectory(prefix="diffgw-source-aligned-very-fine-") as tmp:
            functions = {}
            for kind, patch, adapter in (
                ("capture", CAPTURE, "capture_adapter"),
                ("override", OVERRIDE, "override_adapter"),
            ):
                private = Path(tmp) / kind
                _snapshot(args.source_root, source_plan["pinned_source"],
                          private, args.headers)
                source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
                original = source_file.read_text()
                if original.count(ANCHOR) != 1 or original.count(INCLUDE) != 1:
                    raise ValueError("Pinned source patch anchors changed")
                source_file.write_text(original.replace(ANCHOR, CORRECTION+ANCHOR)
                                       .replace(INCLUDE, INCLUDE+patch))
                library_path = Path(tmp) / f"{kind}.dylib"
                _compile(private, args.headers, args.gsl_include, libraries,
                         paths[adapter], library_path,
                         source_plan["full_generator_source_includes"],
                         source_plan["full_generator_public_includes"],
                         require_wheel_flux=False)
                if kind == "capture":
                    functions["capture"] = _function(
                        library_path, "diffgw_v4_full_source_nqc_inputs")
                else:
                    library = ctypes.CDLL(str(library_path))
                    baseline = library.diffgw_v4_full_source_mode
                    baseline.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
                    baseline.restype = ctypes.c_int
                    override = library.diffgw_v4_full_source_mode_with_nqc
                    override.argtypes = baseline.argtypes + [ctypes.POINTER(ctypes.c_double)]
                    override.restype = ctypes.c_int
                    functions.update(baseline=baseline, override=override)
            previous = None
            modes = {}
            for step in plan["steps_M"]:
                frozen = prior["grids"].get(str(step))
                meta, values, coefficients = _capture(
                    functions["capture"], row, step, plan["capture_capacity"])
                own = np.ascontiguousarray(coefficients[[0, 1, 2, 6, 7]])
                input_hash = hashlib.sha256(values.tobytes()).hexdigest()
                if (not np.isfinite(values).all() or not np.isfinite(meta).all()
                        or abs(meta[1]-step) > 1e-12):
                    raise ValueError(f"Invalid NQC inputs at {step:g} M")
                if frozen is not None and (input_hash != frozen["input_sha256"]
                        or not np.array_equal(own, np.asarray(frozen["source_active"]))):
                    raise ValueError(f"Previous NQC inputs changed at {step:g} M")
                time_axis = np.arange(len(values), dtype=np.float64)*meta[1]
                peak = _local_peak(time_axis, values[:, 4], meta[0])
                if abs(peak["spline_minus_reported_time_M"]) > step:
                    raise ValueError(f"Corrected peak not stationary at {step:g} M")
                fit = _fit(polynomial, time_axis, *values.T, meta[2], meta[3:])
                local_active = np.ascontiguousarray(fit["coefficients"], dtype=np.float64)
                baseline_mode = _source(functions["baseline"], row, step,
                                        plan["mode_capacity"])
                baseline_hash = hashlib.sha256(baseline_mode[2].tobytes()).hexdigest()
                noop = override_source(functions["override"], row, step,
                                       plan["mode_capacity"], own)
                if noop[:2] != baseline_mode[:2] or not np.array_equal(
                        noop[2], baseline_mode[2]):
                    raise ValueError(f"Own-coefficient control failed at {step:g} M")
                local_mode = override_source(functions["override"], row, step,
                                             plan["mode_capacity"], local_active)
                local_hash = hashlib.sha256(local_mode[2].tobytes()).hexdigest()
                wheel = _public(row, step)
                wheel_hash = hashlib.sha256(wheel[2].tobytes()).hexdigest()
                if frozen is not None and (
                        baseline_hash != frozen["source_mode_sha256"]
                        or local_hash != frozen["local_mode_sha256"]
                        or wheel_hash != frozen["public_mode_sha256"]
                        or np.max(abs(local_active-np.asarray(frozen["local_fit"]["coefficients"])))
                        > 1e-10):
                    raise ValueError(f"Previous aligned-spin modes changed at {step:g} M")
                comparisons = {
                    "local_minus_source_mode": _fitted(
                        local_mode, baseline_mode, mass_seconds, plan["peak_fit"]),
                    "local_minus_public_mode": _fitted(
                        local_mode, wheel, mass_seconds, plan["peak_fit"]),
                    "source_minus_public_mode": _fitted(
                        baseline_mode, wheel, mass_seconds, plan["peak_fit"]),
                }
                if frozen is not None and abs(
                        comparisons["local_minus_public_mode"]["metrics"]["complex_relative_l2"]
                        - frozen["local_minus_public_mode"]["metrics"]["complex_relative_l2"]
                ) > 1e-12:
                    raise ValueError("Previous local/public comparison changed")
                report["grids"][str(step)] = {
                    "nqc_samples": len(values),
                    "source_mode_samples": len(baseline_mode[2]),
                    "public_mode_samples": len(wheel[2]),
                    "orbital_peak_M": float(meta[0]), "match_time_M": float(meta[2]),
                    "high_grid_peak": peak, "input_sha256": input_hash,
                    "source_mode_sha256": baseline_hash,
                    "noop_mode_sha256": hashlib.sha256(noop[2].tobytes()).hexdigest(),
                    "local_mode_sha256": local_hash,
                    "public_mode_sha256": wheel_hash,
                    "source_active": own.tolist(), "local_fit": fit,
                    "local_minus_source_active_l2": float(np.linalg.norm(local_active-own)),
                    "local_minus_source_epoch_M": (local_mode[0]-baseline_mode[0])/mass_seconds,
                    "local_minus_source_samples": len(local_mode[2])-len(baseline_mode[2]),
                    **comparisons,
                }
                if previous is not None:
                    old_step, old_source, old_local, old_wheel, old_active = previous
                    pair = f"{old_step:g}_to_{step:g}"
                    report["adjacent_local"][pair] = _fitted(
                        old_local, local_mode, mass_seconds, plan["peak_fit"])
                    report["adjacent_local"][pair]["coefficient_difference_l2"] = float(
                        np.linalg.norm(local_active-old_active))
                    report["adjacent_source"][pair] = _fitted(
                        old_source, baseline_mode, mass_seconds, plan["peak_fit"])
                    report["adjacent_public"][pair] = _fitted(
                        old_wheel, wheel, mass_seconds, plan["peak_fit"])
                previous = (step, baseline_mode, local_mode, wheel, local_active)
                modes[step] = (baseline_mode, local_mode, own, local_active)
                q.write_evidence(args.json_out, report)
                print(f"step={step:g} local/public="
                      f"{comparisons['local_minus_public_mode']['metrics']['complex_relative_l2']:.9g}",
                      flush=True)
            coarse, fine = plan["steps_M"]
            coarse_source, coarse_local, _, _ = modes[coarse]
            fine_source, fine_local, fine_source_active, fine_local_active = modes[fine]
            fixed_source = override_source(functions["override"], row, coarse,
                                           plan["mode_capacity"], fine_source_active)
            fixed_local = override_source(functions["override"], row, coarse,
                                          plan["mode_capacity"], fine_local_active)
            report["adjacent_fixed_finest"] = {
                "source": _fitted(fixed_source, fine_source, mass_seconds,
                                  plan["peak_fit"]),
                "local": _fitted(fixed_local, fine_local, mass_seconds,
                                 plan["peak_fit"]),
                "fixed_source_coarse_mode_sha256": hashlib.sha256(
                    fixed_source[2].tobytes()).hexdigest(),
                "fixed_local_coarse_mode_sha256": hashlib.sha256(
                    fixed_local[2].tobytes()).hexdigest(),
                "source_coarse_mode_sha256": hashlib.sha256(
                    coarse_source[2].tobytes()).hexdigest(),
                "local_coarse_mode_sha256": hashlib.sha256(
                    coarse_local[2].tobytes()).hexdigest(),
            }
        report["elapsed_seconds"] = time.monotonic()-started
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

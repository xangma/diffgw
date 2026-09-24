"""Bound source-versus-wheel mode mismatch with wheel (2,2) NQC coefficients."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import tempfile

import lal
import numpy as np

from . import qualify_generator as q
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import INCLUDE
from .diagnose_source_nqc_whole_mode_override import (
    OVERRIDE, direct_metrics, fitted, override_source,
)
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_wheel_override_plan.json"
ACTIVE_INDICES = (0, 1, 2, 6, 7)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    names = ("prior_override", "source_wheel_mode", "source_flux_plan", "adapter")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-wheel-override-plan.v1"
            or plan["release_qualified"] is not False
            or plan["rows"] != ["nonspinning", "mild_mixed_spin", "strong_aligned_spin"]
            or plan["step_M"] != 0.125 or plan["capacity"] != 50000
            or any(q.sha256(paths[name]) != plan[name + "_sha256"] for name in names)):
        parser.error("Frozen wheel override plan or prerequisites changed")
    previous = json.loads(paths["prior_override"].read_text())
    source_wheel = json.loads(paths["source_wheel_mode"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (previous["status"] != "diagnostic_complete"
            or source_wheel["status"] != "diagnostic_complete"
            or previous["release_qualified"] is not False):
        parser.error("Frozen prerequisite is incomplete")
    previous_rows = {item["name"]: item for item in previous["rows"]}
    wheel_rows = {item["name"]: item for item in source_wheel["rows"]}
    libraries = _libraries(source_plan["pinned_wheel"])
    report = {
        "schema": "diffgw.seobnrv4.source-nqc-wheel-override.v1",
        "scope": dict(
            q.SCOPE, development_only=True,
            description="Diagnostic source whole-mode sensitivity to wheel NQC coefficients.",
            known_limitations=[
                "Wheel NQC coefficients are reference outputs, not native model data.",
                "Three sampled rows do not qualify the generator or domain.",
                "No gradients, batching, accelerator, or ROM training are qualified.",
            ],
        ),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "instrumentation_sha256": hashlib.sha256(OVERRIDE.encode()).hexdigest(),
        "pinned_libraries_sha256": {name: q.sha256(path) for name, path in libraries.items()},
        "rows": [], "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-nqc-wheel-override-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("Pinned source NQC include anchor changed")
            source_file.write_text(original.replace(INCLUDE, INCLUDE + OVERRIDE))
            dylib = Path(tmp) / "source_nqc_override.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["adapter"], dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            library = ctypes.CDLL(str(dylib))
            baseline_function = library.diffgw_v4_full_source_mode
            baseline_function.argtypes = [ctypes.c_double] * 6 + [
                ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
            ]
            baseline_function.restype = ctypes.c_int
            override_function = library.diffgw_v4_full_source_mode_with_nqc
            override_function.argtypes = baseline_function.argtypes + [
                ctypes.POINTER(ctypes.c_double)]
            override_function.restype = ctypes.c_int
            for name in plan["rows"]:
                frozen = wheel_rows[name]
                prior = previous_rows[name]
                row = frozen["row"]
                baseline = _source(baseline_function, row, plan["step_M"], plan["capacity"])
                wheel = _public(row, plan["step_M"])
                if (hashlib.sha256(baseline[2].tobytes()).hexdigest() != frozen["source_mode_sha256"]
                        or hashlib.sha256(wheel[2].tobytes()).hexdigest() != frozen["wheel_mode_sha256"]):
                    raise ValueError(f"Baseline source or wheel mode fails frozen replay: {name}")
                full = frozen["wheel_nqc_coefficients"]
                if (len(full) != 10 or any(full[i] != 0 for i in (3, 4, 5, 8, 9))):
                    raise ValueError(f"Unexpected active NQC coefficient layout: {name}")
                active = np.asarray([full[i] for i in ACTIVE_INDICES], dtype=np.float64)
                modified = override_source(override_function, row, plan["step_M"],
                                           plan["capacity"], active)
                if (baseline[0] != modified[0] or baseline[1] != modified[1]
                        or len(baseline[2]) != len(modified[2])):
                    raise ValueError(f"NQC override changed mode time axis: {name}")
                mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
                baseline_direct = direct_metrics(baseline[2], wheel[2], frozen["wheel_peak_index"])
                baseline_fitted = fitted(baseline, wheel, mass_seconds)
                if (baseline_direct != prior["baseline"]["direct"]
                        or baseline_fitted != prior["baseline"]["fitted"]):
                    raise ValueError(f"Baseline comparison fails prior replay: {name}")
                report["rows"].append({
                    "name": name, "row": row,
                    "baseline_source_mode_sha256": frozen["source_mode_sha256"],
                    "wheel_mode_sha256": frozen["wheel_mode_sha256"],
                    "modified_mode_sha256": hashlib.sha256(modified[2].tobytes()).hexdigest(),
                    "wheel_active_coefficients": active.tolist(),
                    "baseline": {"direct": baseline_direct, "fitted": baseline_fitted},
                    "modified": {
                        "direct": direct_metrics(modified[2], wheel[2], frozen["wheel_peak_index"]),
                        "fitted": fitted(modified, wheel, mass_seconds),
                    },
                    "same_source_mode_change": q.mode_metrics(modified[2], baseline[2]),
                })
                q.write_evidence(args.json_out, report)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "diagnostic_failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        q.write_evidence(args.json_out, report)
        raise
    q.write_evidence(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

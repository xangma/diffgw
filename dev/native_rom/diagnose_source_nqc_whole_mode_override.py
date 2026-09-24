"""Test a dense-field NQC coefficient override in the pinned full v4 generator."""

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
from .diagnose_peak_anchor import fitted_metrics
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import INCLUDE
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_nqc_whole_mode_override_plan.json"
OVERRIDE = """\
static int diffgw_override_nqc_coefficients(
    REAL8Vector *, REAL8Vector *, REAL8Vector *, REAL8Vector *, REAL8Vector *,
    INT4, INT4, REAL8, REAL8, REAL8, REAL8, REAL8, REAL8, EOBNonQCCoeffs *);
#define XLALSimIMRSpinEOBCalculateNQCCoefficientsV4 diffgw_override_nqc_coefficients
"""


def override_source(function, row: dict, step_M: float, capacity: int,
                    active: np.ndarray) -> tuple[float, float, np.ndarray]:
    values = np.empty(2*capacity, dtype=np.float64)
    count = ctypes.c_size_t()
    epoch, delta_t = ctypes.c_double(), ctypes.c_double()
    rc = function(row["mass1"], row["mass2"], row["chi1"], row["chi2"],
                  row["omega_start"], step_M, capacity, ctypes.byref(count),
                  ctypes.byref(epoch), ctypes.byref(delta_t),
                  values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  active.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    if rc or count.value < 10 or count.value > capacity:
        raise RuntimeError(f"Overridden source mode failed: {rc}")
    mode = values[:2*count.value].reshape(-1, 2)
    return epoch.value, delta_t.value, mode[:, 0]+1j*mode[:, 1]


def direct_metrics(candidate: np.ndarray, wheel: np.ndarray, peak: int) -> dict:
    count = min(len(candidate), len(wheel))
    if peak < 160 or peak >= count:
        raise ValueError("Wheel peak outside common mode samples")
    return {
        "same_index": q.mode_metrics(candidate[:count], wheel[:count]),
        "pre_attachment_20M": q.mode_metrics(candidate[:peak-160], wheel[:peak-160]),
    }


def fitted(source: tuple[float, float, np.ndarray],
           wheel: tuple[float, float, np.ndarray, np.ndarray], mass_seconds: float) -> dict:
    source_epoch, source_dt, source_mode = source
    wheel_epoch, wheel_dt, wheel_mode, _ = wheel
    return fitted_metrics(source_epoch+np.arange(len(source_mode))*source_dt,
                          source_mode,
                          wheel_epoch+np.arange(len(wheel_mode))*wheel_dt,
                          wheel_mode, mass_seconds, "quadratic_power")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    names = ("coefficient_field", "source_wheel_mode", "source_flux_plan", "adapter")
    paths = {name: q.ROOT/plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-whole-mode-override-plan.v1"
            or plan["release_qualified"] is not False
            or plan["rows"] != ["nonspinning", "mild_mixed_spin", "strong_aligned_spin"]
            or plan["step_M"] != 0.125 or plan["capacity"] != 50000
            or any(q.sha256(paths[name]) != plan[name+"_sha256"] for name in names)):
        parser.error("Frozen whole-mode override plan or prerequisites changed")
    coefficients = json.loads(paths["coefficient_field"].read_text())
    source_wheel = json.loads(paths["source_wheel_mode"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (coefficients["status"] != "diagnostic_complete"
            or source_wheel["status"] != "diagnostic_complete"
            or coefficients["release_qualified"] is not False):
        parser.error("Frozen source evidence is incomplete")
    wheel_rows = {item["name"]: item for item in source_wheel["rows"]}
    libraries = _libraries(source_plan["pinned_wheel"])
    report = {
        "schema": "diffgw.seobnrv4.source-nqc-whole-mode-override.v1",
        "scope": dict(**q.SCOPE, development_only=True),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "instrumentation_sha256": hashlib.sha256(OVERRIDE.encode()).hexdigest(),
        "pinned_libraries_sha256": {name: q.sha256(path) for name, path in libraries.items()},
        "rows": [], "errors": [],
    }
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-nqc-whole-mode-") as tmp:
            private = Path(tmp)/"source"
            _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
            source_file = private/"lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("Pinned source NQC include anchor changed")
            source_file.write_text(original.replace(INCLUDE, INCLUDE+OVERRIDE))
            dylib = Path(tmp)/"source_nqc_override.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["adapter"], dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            library = ctypes.CDLL(str(dylib))
            baseline_function = library.diffgw_v4_full_source_mode
            baseline_function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            baseline_function.restype = ctypes.c_int
            override_function = library.diffgw_v4_full_source_mode_with_nqc
            override_function.argtypes = baseline_function.argtypes + [
                ctypes.POINTER(ctypes.c_double)]
            override_function.restype = ctypes.c_int
            for name in plan["rows"]:
                frozen = wheel_rows[name]
                row = frozen["row"]
                baseline = _source(baseline_function, row, plan["step_M"], plan["capacity"])
                wheel = _public(row, plan["step_M"])
                if (hashlib.sha256(baseline[2].tobytes()).hexdigest() != frozen["source_mode_sha256"]
                        or hashlib.sha256(wheel[2].tobytes()).hexdigest() != frozen["wheel_mode_sha256"]):
                    raise ValueError(f"Full source or wheel mode fails frozen replay: {name}")
                active = np.asarray(coefficients["rows"][name]["factors"]["8"]["dense_field_coefficients"],
                                    dtype=np.float64)
                modified = override_source(override_function, row, plan["step_M"],
                                           plan["capacity"], active)
                if (baseline[0] != modified[0] or baseline[1] != modified[1]
                        or len(baseline[2]) != len(modified[2])):
                    raise ValueError(f"NQC override changed mode time axis: {name}")
                wheel_peak = frozen["wheel_peak_index"]
                mass_seconds = (row["mass1"]+row["mass2"])*lal.MTSUN_SI
                result = {
                    "name": name, "row": row,
                    "baseline_source_mode_sha256": frozen["source_mode_sha256"],
                    "wheel_mode_sha256": frozen["wheel_mode_sha256"],
                    "modified_mode_sha256": hashlib.sha256(modified[2].tobytes()).hexdigest(),
                    "dense_field_active_coefficients": active.tolist(),
                    "source_nqc_active_coefficients": coefficients["rows"][name]["factors"]["8"]["source_active_coefficients"],
                    "baseline": {
                        "direct": direct_metrics(baseline[2], wheel[2], wheel_peak),
                        "fitted": fitted(baseline, wheel, mass_seconds),
                    },
                    "modified": {
                        "direct": direct_metrics(modified[2], wheel[2], wheel_peak),
                        "fitted": fitted(modified, wheel, mass_seconds),
                    },
                    "same_source_mode_change": q.mode_metrics(modified[2], baseline[2]),
                }
                report["rows"].append(result)
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

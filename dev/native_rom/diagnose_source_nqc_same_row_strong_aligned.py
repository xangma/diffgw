"""Dev-only pinned historical SEOBNRv4 NQC capture for the 0.8/0.6 row."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_adaptive_physical_event import CASES
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_nqc_same_row_strong_aligned_plan.json"
OUTPUT = EVIDENCE / "source_nqc_same_row_strong_aligned.json"
ROW = (30.0, 20.0, 0.8, 0.6, 0.04)
SOURCE_ROOT = Path("/Users/xangma/repos/lalsuite-git")
HEADERS = Path("/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include")
GSL_INCLUDE = Path("/usr/local/include")


def _checked_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-nqc-same-row-plan.v1"
            or plan["release_qualified"] is not False
            or tuple(plan["row_tuple"]) != ROW
            or tuple(CASES["strong_aligned"]) != ROW
            or plan["step_M"] != 0.125
            or plan["capacity"] != 50000
            or plan["gates"] != {
                "capture_repeat_bitwise": True,
                "finite_capture": True,
                "valid_shapes_and_times": True,
            }
            or plan["script_sha256"] != q.sha256(Path(__file__))
            or plan["instrumentation_sha256"] != hashlib.sha256(CAPTURE.encode()).hexdigest()):
        raise ValueError("Frozen same-row NQC capture plan changed")
    for name, item in plan["inputs"].items():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen same-row NQC input changed: {name}")
    return plan


def main() -> None:
    plan = _checked_plan()
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    if (plan["pinned_source"] != {
            "tag": source_plan["pinned_source"]["tag"],
            "tag_object": source_plan["pinned_source"]["tag_object"],
            "peeled_commit": source_plan["pinned_source"]["peeled_commit"],
            "aligned_source_sha256": source_plan["pinned_source"]["files_sha256"][
                "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"],
            }
            or plan["pinned_wheel_library_sha256"]
            != source_plan["pinned_wheel"]["libraries_sha256"]):
        raise ValueError("Pinned source or wheel identity changed")
    libraries = _libraries(source_plan["pinned_wheel"])
    row = dict(zip(("mass1", "mass2", "chi1", "chi2", "omega_start"), ROW))
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.source-nqc-same-row.v1",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
        "row": row, "row_tuple": list(ROW), "step_M": plan["step_M"],
        "source_tag": source_plan["pinned_source"]["tag"],
        "source_tag_object": source_plan["pinned_source"]["tag_object"],
        "source_peeled_commit": source_plan["pinned_source"]["peeled_commit"],
        "wheel_versions": {key: source_plan["pinned_wheel"][key] for key in
                           ("lalsuite_distribution", "lal_version", "lalsimulation_version")},
        "wheel_library_sha256": {name: q.sha256(path) for name, path in libraries.items()},
        "instrumentation_sha256": hashlib.sha256(CAPTURE.encode()).hexdigest(),
        "checks": {},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with tempfile.TemporaryDirectory(prefix="diffgw-v4-same-row-nqc-") as tmp:
        private = Path(tmp) / "source"
        _snapshot(SOURCE_ROOT, source_plan["pinned_source"], private, HEADERS)
        source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
        original = source_file.read_text()
        if original.count(INCLUDE) != 1:
            raise ValueError("Pinned source instrumentation anchor changed")
        source_file.write_text(original.replace(INCLUDE, INCLUDE + CAPTURE))
        result["instrumented_source_sha256"] = q.sha256(source_file)
        dylib = Path(tmp) / "source_nqc_same_row.dylib"
        _compile(private, HEADERS, GSL_INCLUDE, libraries,
                 ROOT / plan["inputs"]["adapter"]["path"], dylib,
                 source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"],
                 require_wheel_flux=False)
        function = ctypes.CDLL(str(dylib)).diffgw_v4_full_source_nqc_inputs
        function.argtypes = [ctypes.c_double] * 6 + [
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double)]
        function.restype = ctypes.c_int
        first = _capture(function, row, plan["step_M"], plan["capacity"])
        second = _capture(function, row, plan["step_M"], plan["capacity"])
    meta, values, coefficients = first
    checks = {
        "exact_requested_tuple": tuple(row.values()) == ROW,
        "capture_repeat_bitwise_meta": np.array_equal(first[0], second[0]),
        "capture_repeat_bitwise_input_array": np.array_equal(first[1], second[1]),
        "capture_repeat_bitwise_coefficients": np.array_equal(first[2], second[2]),
        "finite_capture": bool(np.isfinite(meta).all() and np.isfinite(values).all()
                               and np.isfinite(coefficients).all()),
        "input_shape": values.shape[1] == 5 and 10 <= len(values) <= plan["capacity"],
        "coefficient_shape": coefficients.shape == (10,),
        "positive_cadence": 0 < meta[1] <= plan["step_M"],
        "valid_match_time": 0 < meta[2] < meta[0] < len(values) * meta[1],
    }
    result.update({
        "status": "diagnostic_complete" if all(checks.values()) else "failed",
        "elapsed_s": time.monotonic() - started,
        "captured_samples": len(values),
        "time_peak_M": float(meta[0]), "internal_step_M": float(meta[1]),
        "match_time_M": float(meta[2]), "nqc_targets": meta[3:].tolist(),
        "prefit_input_array_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "coefficient_array_sha256": hashlib.sha256(coefficients.tobytes()).hexdigest(),
        "active_coefficient_indices": [0, 1, 2, 6, 7],
        "source_nqc_coefficients": coefficients.tolist(),
        "source_active_nqc_coefficients": coefficients[[0, 1, 2, 6, 7]].tolist(),
        "basis": "N22=(1+sum_j(a_j q_j))*exp(i sum_j(b_j p_j)); active order a1,a2,a3,b1,b2",
        "checks": {name: bool(ok) for name, ok in checks.items()},
    })
    result["diagnostic_checks_passed"] = all(checks.values())
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "source_active_nqc_coefficients": result["source_active_nqc_coefficients"]}),
          flush=True)
    if not result["diagnostic_checks_passed"]:
        raise ValueError("Same-row source NQC capture failed a frozen gate")


if __name__ == "__main__":
    main()

"""Cross source q10 NQC input fields with source orbital-peak times."""

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
from .diagnose_near_extreme_source_nqc_grid import ACTIVE
from .diagnose_nqc_c_same_input import bind, compile_wrapper
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/near_extreme_source_nqc_peak_field_plan.json"


def main(argv=None) -> int:
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
    if (plan["schema"] != "diffgw.seobnrv4.near-extreme-source-nqc-peak-field-plan.v1"
            or plan["row"] != "q10_both_near_extreme_aligned"
            or plan["output_steps_M"] != [0.03125, 0.015625]
            or plan["expected_internal_step_M"] != 0.015625
            or plan["capacity"] != 50000 or plan["release_qualified"] is not False
            or any(q.sha256(paths[name]) != item["sha256"]
                   for name, item in plan["inputs"].items())):
        parser.error("frozen peak/field plan or prerequisites changed")
    prior = json.loads(paths["prior"].read_text())
    capture_plan = json.loads(paths["capture_plan"].read_text())
    if (prior["status"] != "diagnostic_complete"
            or prior["row"] != json.loads((q.ROOT / capture_plan["inputs"]["parity"]["path"]).read_text())
            ["rows"][plan["row"]]["row"]):
        parser.error("prior q10 source capture differs")
    source_plan = json.loads((q.ROOT / capture_plan["inputs"]["source_plan"]["path"]).read_text())
    adapter = q.ROOT / capture_plan["inputs"]["adapter"]["path"]
    if any(q.sha256(q.ROOT / item["path"]) != item["sha256"]
           for item in capture_plan["inputs"].values()):
        parser.error("capture prerequisites changed")
    libraries = _libraries(source_plan["pinned_wheel"])
    report = dict(
        schema="diffgw.seobnrv4.near-extreme-source-nqc-peak-field.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        input_sha256={name: q.sha256(path) for name, path in paths.items()},
        instrumentation_sha256=hashlib.sha256(CAPTURE.encode()).hexdigest(),
        pinned_libraries_sha256={name: q.sha256(path) for name, path in libraries.items()},
        row=prior["row"], levels={}, crossed_fits={}, comparison={}, errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-q10-nqc-peak-field-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("pinned source NQC include anchor changed")
            source_file.write_text(original.replace(INCLUDE, INCLUDE + CAPTURE))
            capture_library = Path(tmp) / "capture.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     adapter, capture_library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            function = ctypes.CDLL(str(capture_library)).diffgw_v4_full_source_nqc_inputs
            pointer = ctypes.POINTER(ctypes.c_double)
            function.argtypes = [ctypes.c_double] * 6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), pointer, pointer, pointer]
            function.restype = ctypes.c_int
            fit_library = Path(tmp) / "fit.dylib"
            # The wrapper must include uninstrumented source, not the private
            # generator snapshot whose call site was redirected for capture.
            fit_private = Path(tmp) / "fit_source"
            _snapshot(args.source_root, source_plan["pinned_source"], fit_private, args.headers)
            compile_wrapper(fit_private, args.headers, args.gsl_include,
                            [str(path) for path in libraries.values()], fit_library)
            c_fit, _ = bind(fit_library)
            captures = {}
            for step in plan["output_steps_M"]:
                meta, values, coefficients = _capture(function, prior["row"], step,
                                                      plan["capacity"])
                key = f"dt_{step:.6f}M"
                expected = prior["levels"][key]
                if (len(values) != expected["input_samples"]
                        or hashlib.sha256(values.tobytes()).hexdigest() != expected["input_sha256"]
                        or not np.array_equal(meta, np.asarray(expected["meta"]))
                        or not np.array_equal(coefficients,
                                              np.asarray(expected["source_coefficients"]))
                        or abs(meta[1] - plan["expected_internal_step_M"]) > 1e-12):
                    raise ValueError(f"source capture no longer reproduces: {key}")
                captures[key] = meta, values, coefficients
                report["levels"][key] = dict(
                    input_samples=len(values), input_sha256=expected["input_sha256"],
                    peak_time_M=float(meta[0]), source_coefficients=coefficients.tolist())
                q.write_evidence(args.json_out, report)

            row = prior["row"]
            m1, m2 = row["mass1"], row["mass2"]
            chi_a = (row["chi1"] - row["chi2"]) / 2
            chi_s = (row["chi1"] + row["chi2"]) / 2
            coarse, fine = [f"dt_{step:.6f}M" for step in plan["output_steps_M"]]
            fits = {}
            for fields_key in (coarse, fine):
                fields_meta, values, _ = captures[fields_key]
                columns = [np.ascontiguousarray(values[:, i]) for i in range(5)]
                for peak_key in (coarse, fine):
                    peak = float(captures[peak_key][0][0])
                    result = np.empty(5, dtype=np.float64)
                    rc = c_fit(*(column.ctypes.data_as(pointer) for column in columns),
                               len(values), peak, float(fields_meta[1]),
                               m1, m2, chi_a, chi_s, result.ctypes.data_as(pointer))
                    if rc or not np.isfinite(result).all():
                        raise RuntimeError(f"C fit failed: {fields_key}/{peak_key}, {rc}")
                    fits[fields_key, peak_key] = result
                    report["crossed_fits"][f"fields_{fields_key}__peak_{peak_key}"] = result.tolist()
                    q.write_evidence(args.json_out, report)
            for key in (coarse, fine):
                replay = fits[key, key]
                expected = captures[key][2][list(ACTIVE)]
                if np.max(abs(replay - expected)) > 1e-8:
                    raise ValueError(f"same-input C fit does not replay source: {key}")
                report["levels"][key]["same_input_max_absolute_error"] = float(
                    np.max(abs(replay - expected)))
            base = fits[coarse, coarse]
            peak_only = fits[coarse, fine] - base
            field_only = fits[fine, coarse] - base
            interaction = fits[fine, fine] - base - peak_only - field_only
            total = fits[fine, fine] - base
            report["comparison"] = dict(
                peak_only_change=peak_only.tolist(),
                field_only_change=field_only.tolist(),
                interaction_change=interaction.tolist(),
                total_change=total.tolist(),
                peak_only_l2=float(np.linalg.norm(peak_only)),
                field_only_l2=float(np.linalg.norm(field_only)),
                interaction_l2=float(np.linalg.norm(interaction)),
                total_l2=float(np.linalg.norm(total)),
                peak_time_difference_M=float(captures[fine][0][0]-captures[coarse][0][0]),
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

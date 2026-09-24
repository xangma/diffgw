"""Test whether public q10 NQC-grid coefficient changes affect the full mode."""

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

import lal
import numpy as np

from . import qualify_generator as q
from .diagnose_peak_anchor import fitted_metrics
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import INCLUDE
from .diagnose_source_nqc_whole_mode_override import OVERRIDE, override_source
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/near_extreme_nqc_intervention_v2_plan.json"
ACTIVE = (0, 1, 2, 6, 7)


def postpeak_metrics(first: np.ndarray, second: np.ndarray) -> dict:
    if first.shape != second.shape:
        raise ValueError("intervention changed source output length")
    peak = int(np.argmax(np.abs(second)))
    return dict(
        all=q.mode_metrics(first, second),
        postpeak=q.mode_metrics(first[peak:], second[peak:]),
        reference_peak_index=peak,
    )


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
    paths = {key: q.ROOT / value["path"] for key, value in plan["inputs"].items()}
    if (plan["schema"] != "diffgw.seobnrv4.near-extreme-nqc-intervention-plan.v2"
            or plan["row"] != "q10_both_near_extreme_aligned"
            or plan["coefficient_steps_M"] != [0.03125, 0.015625]
            or plan["mode_step_M"] != 0.015625
            or plan["capacity"] != 150000
            or plan["release_qualified"] is not False
            or any(q.sha256(paths[key]) != value["sha256"]
                   for key, value in plan["inputs"].items())):
        parser.error("frozen plan or prerequisites changed")
    nqc = json.loads(paths["nqc"].read_text())
    parity = json.loads(paths["parity"].read_text())
    source_plan = json.loads(paths["source_plan"].read_text())
    if (nqc["status"] != "diagnostic_complete"
            or parity["status"] != "diagnostic_complete"
            or nqc["row"] != plan["row"]
            or nqc["lal_binary"] != parity["lal_binary"]
            or q.lal_binary_identity(nqc["lal_binary"]["lal_version"],
                                     nqc["lal_binary"]["lalsimulation_version"])
            != nqc["lal_binary"]):
        parser.error("pinned reference evidence differs")
    row = parity["rows"][plan["row"]]["row"]
    libraries = _libraries(source_plan["pinned_wheel"])
    coefficients = {step: np.asarray(
        nqc["levels"][f"dt_{step:.6f}M"], dtype=np.float64)[list(ACTIVE)]
        for step in plan["coefficient_steps_M"]}
    report = dict(
        schema="diffgw.seobnrv4.near-extreme-nqc-intervention.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        input_sha256={key: q.sha256(path) for key, path in paths.items()},
        instrumentation_sha256=hashlib.sha256(OVERRIDE.encode()).hexdigest(),
        pinned_libraries_sha256={name: q.sha256(path) for name, path in libraries.items()},
        lal_binary=nqc["lal_binary"], row=row,
        coefficient_values={f"dt_{step:.6f}M": active.tolist()
                            for step, active in coefficients.items()},
        modes={}, comparisons={}, errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-q10-nqc-swap-") as tmp:
            private = Path(tmp) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("pinned source NQC include anchor changed")
            source_file.write_text(original.replace(INCLUDE, INCLUDE + OVERRIDE))
            dylib = Path(tmp) / "source_nqc_override.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["adapter"], dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            library = ctypes.CDLL(str(dylib))
            baseline_fn = library.diffgw_v4_full_source_mode
            baseline_fn.argtypes = [ctypes.c_double] * 6 + [ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            baseline_fn.restype = ctypes.c_int
            override_fn = library.diffgw_v4_full_source_mode_with_nqc
            override_fn.argtypes = baseline_fn.argtypes + [ctypes.POINTER(ctypes.c_double)]
            override_fn.restype = ctypes.c_int
            step = plan["mode_step_M"]
            baseline = _source(baseline_fn, row, step, plan["capacity"])
            coarse = override_source(override_fn, row, step, plan["capacity"],
                                     coefficients[0.03125])
            fine = override_source(override_fn, row, step, plan["capacity"],
                                   coefficients[0.015625])
            wheel = _public(row, step)
            modes = {"baseline_source": baseline, "coarse_coefficients": coarse,
                     "fine_coefficients": fine}
            if any((mode[0] != baseline[0] or mode[1] != baseline[1]
                    or len(mode[2]) != len(baseline[2]))
                   for mode in modes.values()):
                raise ValueError("intervention changed source mode grid")
            mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
            for name, (epoch, delta_t, mode) in modes.items():
                report["modes"][name] = dict(
                    samples=len(mode), epoch_s=epoch, delta_t_s=delta_t,
                    sha256=hashlib.sha256(mode.tobytes()).hexdigest(),
                    vs_public_fitted=fitted_metrics(
                        epoch + np.arange(len(mode)) * delta_t, mode,
                        wheel[0] + np.arange(len(wheel[2])) * wheel[1],
                        wheel[2], mass_seconds, "quadratic_power"),
                )
                q.write_evidence(args.json_out, report)
            report["comparisons"] = dict(
                coarse_vs_fine=postpeak_metrics(coarse[2], fine[2]),
                baseline_vs_fine=postpeak_metrics(baseline[2], fine[2]),
                baseline_vs_coarse=postpeak_metrics(baseline[2], coarse[2]),
                observed_public_grid_postpeak_complex_relative_l2=nqc[
                    "comparison"]["direct_postpeak_complex_relative_l2"],
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

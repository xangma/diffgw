"""Replay native NQC samples through the pinned LALSuite V4 C matching routine.

This diagnostic compiles the pinned source locally; it does not call the
private routine inside the distributed LALSuite wheel.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

import numpy as np
import torch

from . import qualify_generator as q


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/nqc_c_same_input_plan.json"
WRAPPER = q.ROOT / "dev/native_rom/reference/nqc_v4_source_wrapper.c"
SCOPE = {
    **q.SCOPE,
    "description": "Source-compiled C NQC matching diagnostic on frozen native samples only.",
    "known_limitations": [
        "The private wheel C function is not callable; this compiles the pinned source locally.",
        "The orbital inputs are reconstructed from native NQC basis arrays with an explicit identity check.",
        "The public waveform fit has different trajectory and raw-mode inputs.",
        "No waveform, gradient, training, or production qualification follows from this check.",
    ],
}


def compile_wrapper(source_root: Path, headers: Path, gsl_include: Path,
                    binaries: dict[str, str], output: Path) -> list[str]:
    if sys.platform != "darwin":
        raise RuntimeError("the pinned-wheel C wrapper recipe is currently macOS-only")
    dylibs = [next((Path(path) for path in binaries if Path(path).name == name), None)
              for name in ("liblal.20.dylib", "libgsl.28.dylib", "libgslcblas.0.dylib")]
    if any(path is None for path in dylibs):
        raise ValueError("pinned LAL/GSL libraries are missing")
    source_dir = source_root / "lalsimulation/lib"
    command = ["clang", "-dynamiclib", "-O2", "-Wno-gnu-folding-constant",
               "-Wl,-headerpad_max_install_names", "-I", str(headers),
               "-I", str(source_dir), "-I", str(gsl_include),
               "-o", str(output), str(WRAPPER), *(str(path) for path in dylibs)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    changes = []
    for path in dylibs:
        changes += ["-change", f"/DLC/lal/.dylibs/{path.name}", str(path)]
    subprocess.run(["install_name_tool", *changes, str(output)], check=True,
                   capture_output=True, text=True)
    return command


def bind(path: Path):
    library = ctypes.CDLL(str(path))
    pointer = ctypes.POINTER(ctypes.c_double)
    function = library.diffgw_v4_nqc_same_input
    function.argtypes = [pointer] * 5 + [ctypes.c_uint] + [ctypes.c_double] * 6 + [pointer]
    function.restype = ctypes.c_int
    offset = library.diffgw_v4_nqc_delta_t
    offset.argtypes = [ctypes.c_double] * 4
    offset.restype = ctypes.c_double
    return function, offset


def replay(item, prior, waveform_plan):
    waveform_path = q.ROOT / item["waveform"]
    if q.sha256(waveform_path) != item["waveform_sha256"]:
        raise ValueError(f"{item['name']} waveform prerequisite changed")
    waveform = json.loads(waveform_path.read_text())
    level = waveform["candidate"][waveform_plan["level"]]["level"]
    numerical = waveform["choice"]["numerical"]
    physical = waveform["choice"]["physical"]
    flux = waveform["choice"]["flux_placements"]
    run = q.native_run(
        waveform["row"], level, q.Settings(**waveform["settings"]),
        diagnostic_hermite_resampling=numerical["hermite_resampling"],
        diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
        diagnostic_matching_second_derivatives=True,
        diagnostic_restart_time_M=(
            numerical["fixed_restart_time_M"] if item["name"] == "strong_aligned_spin"
            else waveform["choice"]["fixed_restart_time_M"]),
        diagnostic_coarse_initial_step_M=0.125,
        diagnostic_warm_start_Momega=0.03162277660168379,
        diagnostic_nqc_phase_window_M=item["nqc_phase_window_M"],
        diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
        diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
        diagnostic_mode_21_higher_spin=flux["mode_21_higher_spin"],
        diagnostic_mode_31_higher_spin=flux["mode_31_higher_spin"],
        diagnostic_mode_32_spin_square=flux["mode_32_spin_square"],
        return_nqc_coefficients=True, return_nqc_samples=True)
    replay_hash = q.summarize_run(run)["array_sha256"]
    if replay_hash != prior["native_waveform_sha256"]:
        raise ValueError(f"{item['name']} native waveform no longer reproduces")
    native = np.asarray(run["nqc_coefficients"]["amplitude"]
                        + run["nqc_coefficients"]["phase"])
    if not np.allclose(native, prior["native_coefficients"], rtol=0, atol=1e-10):
        raise ValueError(f"{item['name']} native coefficients no longer reproduce")
    return waveform["row"], run["nqc_samples"], replay_hash, native


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if plan["schema"] != "diffgw.seobnrv4.nqc-c-same-input-plan.v1":
        parser.error("unexpected C NQC plan")
    source = args.source_root / plan["source_path"]
    table = args.source_root / plan["table_path"]
    if q.sha256(source) != plan["source_sha256"] or q.sha256(table) != plan["table_sha256"]:
        parser.error("pinned NQC source differs")
    coefficient_path = q.ROOT / plan["coefficient_evidence"]
    gsl_path = q.ROOT / plan["gsl_evidence"]
    if (q.sha256(coefficient_path) != plan["coefficient_evidence_sha256"]
            or q.sha256(gsl_path) != plan["gsl_evidence_sha256"]):
        parser.error("frozen coefficient or GSL prerequisite differs")
    coefficients = json.loads(coefficient_path.read_text())
    gsl = json.loads(gsl_path.read_text())
    waveform_plan = json.loads((q.ROOT / plan["waveform_plan"]).read_text())
    binary = coefficients["lal_binary"]
    if q.lal_binary_identity(binary["lal_version"], binary["lalsimulation_version"]) != binary:
        parser.error("pinned wheel differs")
    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.nqc-c-same-input.v1", status="running",
        scope=SCOPE,
        release_qualified=False, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), wrapper_sha256=q.sha256(WRAPPER),
        source_revision=plan["source_revision"], source_sha256=q.sha256(source),
        table_sha256=q.sha256(table), lal_binary=binary,
        compiler=subprocess.run(["clang", "--version"], check=True, capture_output=True,
                                text=True).stdout.splitlines()[0],
        build_command=None, rows=[], errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        with tempfile.TemporaryDirectory(prefix="diffgw-v4-nqc-") as directory:
            library_path = Path(directory) / "nqc_same_input.dylib"
            report["build_command"] = compile_wrapper(
                args.source_root, args.headers, args.gsl_include,
                binary["binaries"], library_path)
            function, offset = bind(library_path)
            pointer = ctypes.POINTER(ctypes.c_double)
            for item, prior, baseline in zip(waveform_plan["rows"], coefficients["rows"], gsl["rows"]):
                if item["name"] != prior["name"] or item["name"] != baseline["name"]:
                    raise ValueError("frozen row order differs")
                row, samples, replay_hash, native = replay(item, prior, waveform_plan)
                time = np.asarray(samples["time"])
                amp = np.asarray(samples["amplitude"])
                product = np.asarray(samples["products"])
                basis = np.asarray(samples["phase_basis"])
                p1, p2 = basis[:, 0], basis[:, 1]
                # Invert the exact NQC basis identities to recover C inputs.
                radius = product[:, 0] / product[:, 1]
                prstar = np.copysign(np.sqrt(p2 / p1), p1)
                omega = prstar / (p1 * radius)
                q_discrepancy = float(np.max(np.abs(p1**2 - product[:, 0] / amp)))
                if not np.isfinite([*radius, *prstar, *omega]).all() or q_discrepancy > 1e-12:
                    raise ValueError(f"{item['name']} cannot recover same NQC basis samples")
                dt = float(time[1] - time[0])
                if not np.allclose(np.diff(time), dt, rtol=0, atol=1e-9):
                    raise ValueError(f"{item['name']} NQC time grid is not uniform")
                m1, m2 = row["mass1"], row["mass2"]
                chi_a, chi_s = (row["chi1"] - row["chi2"]) / 2, (row["chi1"] + row["chi2"]) / 2
                match = float(samples["match_time"])
                peak = match + offset(m1, m2, chi_a, chi_s)
                arrays = [np.ascontiguousarray(value, dtype=np.float64) for value in
                          (amp, -np.asarray(samples["waveform_phase"]), radius, prstar, omega)]
                result = np.zeros(5, dtype=np.float64)
                rc = function(*(value.ctypes.data_as(pointer) for value in arrays), len(time),
                              peak, dt, m1, m2, chi_a, chi_s,
                              result.ctypes.data_as(pointer))
                if rc != 0 or not np.isfinite(result).all():
                    raise RuntimeError(f"{item['name']} C matching failed: {rc}")
                c_minus_gsl = result - np.asarray(baseline["gsl_coefficients"])
                if np.max(np.abs(c_minus_gsl)) > 2e-6:
                    raise ValueError(f"{item['name']} C and GSL matching differ")
                public = np.asarray(prior["public_coefficients"])
                report["rows"].append(dict(
                    name=item["name"], waveform_sha256=replay_hash,
                    sample_count=len(time), match_time_M=match, time_peak_M=peak,
                    input_basis_identity_max_abs=q_discrepancy,
                    source_c_coefficients=result.tolist(),
                    gsl_coefficients=baseline["gsl_coefficients"],
                    native_coefficients=native.tolist(), public_coefficients=public.tolist(),
                    c_minus_gsl=c_minus_gsl.tolist(), c_minus_public=(result-public).tolist(),
                    c_public_gap_l2=float(np.linalg.norm(result-public)),
                ))
                q.write_evidence(args.json_out, report)
                print(f"row={item['name']}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

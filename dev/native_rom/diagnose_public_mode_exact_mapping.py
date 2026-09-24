"""Resolve a public v4 mode's high-rate index from source-backed lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

import lal
import lalsimulation as lalsim
import numpy as np

from . import qualify_generator as q
from .diagnose_public_nqc_mode_boundary import public_call


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/public_mode_exact_mapping_plan.json"


def source_identity(source_repo: Path, pin: dict) -> None:
    def git(*args: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(source_repo), *args], check=True,
            capture_output=True,
        ).stdout

    if (git("rev-parse", pin["tag"]).decode().strip() != pin["tag_object"]
            or git("cat-file", "-t", pin["tag_object"]).decode().strip() != "tag"
            or git("rev-parse", pin["tag_object"] + "^{commit}").decode().strip()
            != pin["peeled_commit"]):
        raise ValueError("pinned source tag identity differs")
    source = git("show", f"{pin['tag_object']}:{pin['path']}")
    if hashlib.sha256(source).hexdigest() != pin["sha256"]:
        raise ValueError("pinned source file differs")


def row_result(boundary: dict, grid: dict, step: float) -> dict:
    row = boundary["row"]
    output, _ = public_call(row, step, 0)
    modes, low_dynamics, high_dynamics = output
    high = np.asarray(high_dynamics.data, dtype=float).reshape(5, -1)
    low = np.asarray(low_dynamics.data, dtype=float).reshape(5, -1)
    mode = lalsim.SphHarmTimeSeriesGetMode(modes, 2, 2)
    strain = np.asarray(mode.data.data, dtype=np.complex128)
    if (q.canonical_hash(high.tolist()) != boundary["high_dynamics_sha256"]
            or q.canonical_hash(strain.view(np.float64).tolist())
            != boundary["mode_sha256"]
            or (len(low[0]), len(high[0]), len(strain))
            != (boundary["low_dynamics_length"],
                boundary["high_dynamics_length"], boundary["mode_length"])
            or q.canonical_hash(high.tolist()) != grid["high_dynamics_sha256"]
            or q.canonical_hash(strain.view(np.float64).tolist())
            != grid["mode_sha256"]):
        raise ValueError(f"{boundary['name']} pinned public output differs")

    # The tagged v4 path is non-PA. Its resampling factor stays one when
    # 50*deltaT/M <= 1. The full mode vector is low_len + high_len +
    # ceil(20/(Im(omega_QNM)*deltaT)) before the leading kMin samples are cut.
    if step * 50 > 1:
        raise ValueError("this diagnostic requires high-rate output with resampFac=1")
    qnm = lal.CreateCOMPLEX16Vector(1)
    status = lalsim.SimIMREOBGenerateQNMFreqV2(
        qnm, row["mass1"], row["mass2"],
        [0.0, 0.0, row["chi1"]], [0.0, 0.0, row["chi2"]],
        2, 2, 1, lalsim.SEOBNRv4,
    )
    omega_qnm = complex(qnm.data[0])
    if status != 0 or not np.isfinite(omega_qnm.imag) or omega_qnm.imag <= 0:
        raise ValueError("public v4 QNM call failed")
    step_seconds = step * (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    ringdown_extra = math.ceil(20.0 / (omega_qnm.imag * step_seconds))
    k_min = len(low[0]) + len(high[0]) + ringdown_extra - len(strain)
    high_start_index_real = high[0, 0] / step
    high_start_index = round(high_start_index_real)
    if (k_min < 0 or abs(high_start_index_real - high_start_index) > 1e-4
            or abs(low[0, 0]) > 1e-8):
        raise ValueError("public output does not satisfy pinned non-PA grid geometry")
    mode_start = high_start_index - k_min
    if mode_start < 0 or mode_start + len(high[0]) > len(strain):
        raise ValueError("exact high-rate mode slice lies outside public output")
    shift = mode_start - grid["nominal_mode_start_index"]
    if shift not in (-1, 0, 1):
        raise ValueError("exact source index lies outside frozen mapping bracket")
    old_values = [next(sample for sample in comparison["shifted"]
                       if sample["mode_start_shift_samples"] == 0)
                  for comparison in grid["comparisons"]]
    exact_values = [next(sample for sample in comparison["shifted"]
                         if sample["mode_start_shift_samples"] == shift)
                    for comparison in grid["comparisons"]]

    def residuals(values: list[dict]) -> dict:
        return dict(
            max_abs_raw_amplitude_relative=max(
                abs(sample["raw_amplitude_relative"]) for sample in values),
            max_abs_raw_gravitational_wave_frequency_relative=max(
                abs(sample["raw_gravitational_wave_frequency_relative"])
                for sample in values),
            max_abs_raw_amplitude_derivative_difference=max(
                abs(sample["raw_amplitude_derivative_difference"])
                for sample in values),
        )

    return dict(
        name=boundary["name"], output_step_M=step,
        mode_sha256=boundary["mode_sha256"],
        high_dynamics_sha256=boundary["high_dynamics_sha256"],
        low_dynamics_length=len(low[0]), high_dynamics_length=len(high[0]),
        mode_length=len(strain), qnm_frequency_SI=[omega_qnm.real, omega_qnm.imag],
        ringdown_extra_samples=ringdown_extra, inferred_k_min=k_min,
        high_start_index_in_uncut_mode=high_start_index,
        high_start_index_real=high_start_index_real,
        exact_high_mode_start_index=mode_start,
        previous_epoch_based_mode_start_index=grid["nominal_mode_start_index"],
        exact_minus_previous_samples=shift,
        frozen_orbital_frequencies=[item["orbital_frequency"]
                                    for item in grid["comparisons"]],
        previous_reconstructed_raw_mode_residuals=residuals(old_values),
        exact_mapping_reconstructed_raw_mode_residuals=residuals(exact_values),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.public-mode-exact-mapping-plan.v1"
            or plan["release_qualified"] is not False
            or plan["rows"] != ["strong_aligned_spin", "strong_antialigned_spin"]
            or plan["output_step_M"] != 0.015625):
        parser.error("frozen mapping plan differs")
    source_identity(args.source_repo.resolve(), plan["source"])
    paths = [q.ROOT / plan[key] for key in
             ("boundary_evidence", "grid_alignment_evidence")]
    for key, path in zip(("boundary_evidence", "grid_alignment_evidence"), paths):
        if q.sha256(path) != plan[key + "_sha256"]:
            parser.error(f"{key} differs")
    boundary, alignment = [json.loads(path.read_text()) for path in paths]
    binary = boundary["lal_binary"]
    if (q.lal_binary_identity(binary["lal_version"], binary["lalsimulation_version"])
            != binary or alignment["lal_binary"] != binary
            or [r["name"] for r in boundary["rows"]] != plan["rows"]
            or [r["name"] for r in alignment["rows"]] != plan["rows"]):
        parser.error("pinned public binary or row identity differs")
    report = dict(
        schema="diffgw.seobnrv4.public-mode-exact-mapping.v1",
        scope=q.SCOPE, status="running", release_qualified=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        pinned_source=plan["source"], lal_binary=binary, rows=[], errors=[],
        interpretation=("The source-backed integer mapping removes the output-grid "
                        "uncertainty for these 0.015625 M public runs. Raw-mode "
                        "residuals still compare separately evolved trajectories "
                        "and NQC reconstructions, not identical-state components."),
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(),
                     argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        for b, a in zip(boundary["rows"], alignment["rows"]):
            grid = next(g for g in a["grids"]
                        if g["output_step_M"] == plan["output_step_M"])
            report["rows"].append(row_result(b, grid, plan["output_step_M"]))
            q.write_evidence(args.json_out, report)
            print(f"row={b['name']}", flush=True)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Locate the public v4 mode's ringdown boundary relative to NQC matching."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys

import lal
import lalsimulation as lalsim
import numpy as np

from . import qualify_generator as q
from .diagnose_nqc_gsl_same_input import GSLCubic


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/public_nqc_mode_boundary_plan.json"


def public_call(row, step_M, nqc_flag):
    mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    coefficients = lal.CreateREAL8Vector(10)
    output = lalsim.SimIMRSpinAlignedEOBModes(
        step_M * mass_seconds, row["mass1"] * lal.MSUN_SI,
        row["mass2"] * lal.MSUN_SI, row["omega_start"] / (math.pi * mass_seconds),
        100e6 * lal.PC_SI, row["chi1"], row["chi2"], 4,
        0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
        coefficients, nqc_flag, lal.CreateDict(), lal.CreateDict(),
    )
    return output, np.asarray(coefficients.data, dtype=float).copy()


def row_result(prior, step_M, gsl):
    row = prior["row"]
    coefficient_output, coefficients = public_call(row, step_M, 1)
    modes, low_dynamics, high_dynamics = public_call(row, step_M, 0)[0]
    if list(coefficient_output) != [None, None, None]:
        raise ValueError("nqcFlag=1 did not return coefficient-only output")
    if np.any(coefficients[[3, 4, 5, 8, 9]] != 0):
        raise ValueError("unexpected active NQC coefficients")
    mode = lalsim.SphHarmTimeSeriesGetMode(modes, 2, 2)
    strain = np.asarray(mode.data.data, dtype=np.complex128)
    if len(high_dynamics.data) % 5 or len(low_dynamics.data) % 5:
        raise ValueError("public dynamics do not have five blocks")
    high = np.asarray(high_dynamics.data, dtype=float).reshape(5, -1)
    low = np.asarray(low_dynamics.data, dtype=float).reshape(5, -1)
    time = high[0] - high[0, 0]
    if (not np.isfinite(high).all() or not np.isfinite(strain).all()
            or not np.allclose(np.diff(time), step_M, rtol=0, atol=1e-9)
            or not np.allclose(np.diff(low[0]), step_M, rtol=0, atol=1e-9)):
        raise ValueError("public samples are not finite uniform grids")
    mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    if abs(float(mode.deltaT) / mass_seconds - step_M) > 1e-10:
        raise ValueError("mode and dynamics output steps differ")
    orbital_frequency_estimate = np.gradient(high[2], step_M, edge_order=2)
    orbital_peak_index = int(np.argmax(orbital_frequency_estimate))
    cut_real = (float(mode.epoch) / mass_seconds + high[0, 0]
                + time[orbital_peak_index]) / step_M
    cut_index = round(cut_real)
    if abs(cut_real - cut_index) > 0.5:
        raise ValueError("mode epoch and orbital peak imply no integer cut")
    mode_start = round(high[0, 0] / step_M) - cut_index
    if mode_start < 0 or mode_start + len(time) > len(strain):
        raise ValueError("high dynamics cannot be mapped into public mode")
    high_mode = strain[mode_start:mode_start + len(time)]
    # The epoch is GPS-rounded; once its integer cut is known it fixes the
    # source orbital peak more accurately than differentiating sampled phi.
    orbital_peak = cut_index * step_M - high[0, 0] - float(mode.epoch) / mass_seconds
    if abs(orbital_peak - time[orbital_peak_index]) > step_M:
        raise ValueError("epoch peak and sampled orbital peak disagree")
    source_delta = prior["peak_time_M"] - prior["match_time_M"]
    match_time = orbital_peak - source_delta
    amplitude = np.abs(high_mode) / (
        (row["mass1"] + row["mass2"]) * lal.MRSUN_SI / (100e6 * lal.PC_SI))
    amplitude_peak_index = int(np.argmax(amplitude))
    amplitude_peak_time = time[amplitude_peak_index]
    target = np.asarray(prior["target_counterfactual"]["source_targets"])
    matched_amplitude = gsl.derivatives(time, amplitude, match_time)
    phase = np.unwrap(np.angle(high_mode))
    matched_phase = gsl.derivatives(time, phase, match_time)
    return dict(
        name=prior["name"], row=row, output_step_M=step_M,
        coefficient_only_output=True,
        public_coefficients=coefficients[[0, 1, 2, 6, 7]].tolist(),
        mode_length=len(strain), low_dynamics_length=low.shape[1],
        high_dynamics_length=high.shape[1],
        mode_sha256=q.canonical_hash(strain.view(np.float64).tolist()),
        high_dynamics_sha256=q.canonical_hash(high.tolist()),
        cut_index=cut_index, cut_real=cut_real, high_mode_start_index=mode_start,
        orbital_peak_time_M=orbital_peak,
        orbital_phase_derivative_peak_time_M=float(time[orbital_peak_index]),
        source_nqc_peak_offset_M=source_delta,
        inferred_match_time_M=match_time,
        public_mode_amplitude_peak_time_M=float(amplitude_peak_time),
        public_mode_peak_minus_match_M=float(amplitude_peak_time - match_time),
        public_mode_peak_index_in_high=amplitude_peak_index,
        source_targets=target.tolist(),
        public_mode_amplitude_at_match=matched_amplitude,
        public_mode_phase_derivatives_at_match=matched_phase,
        public_mode_amplitude_target_residual=(np.asarray(matched_amplitude) - target[:3]).tolist(),
        public_mode_frequency_target_residual=[
            -matched_phase[1] - target[3], -matched_phase[2] - target[4]],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.public-nqc-mode-boundary-plan.v1"
            or plan["output_step_M"] != 0.015625
            or plan["rows"] != ["strong_aligned_spin", "strong_antialigned_spin"]):
        parser.error("frozen boundary plan differs")
    coefficient_path = q.ROOT / plan["coefficient_evidence"]
    if q.sha256(coefficient_path) != plan["coefficient_evidence_sha256"]:
        parser.error("coefficient evidence differs")
    evidence = json.loads(coefficient_path.read_text())
    binary = evidence["lal_binary"]
    if q.lal_binary_identity(binary["lal_version"], binary["lalsimulation_version"]) != binary:
        parser.error("public LAL binary differs")
    if [row["name"] for row in evidence["rows"]] != plan["rows"]:
        parser.error("coefficient rows differ")
    gsl_paths = [Path(path) for path in binary["binaries"] if Path(path).name.startswith("libgsl.")]
    if len(gsl_paths) != 1:
        parser.error("pinned GSL library is ambiguous")
    gsl = GSLCubic(gsl_paths[0])
    result = dict(
        schema="diffgw.seobnrv4.public-nqc-mode-boundary.v1", scope=q.SCOPE,
        status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), coefficient_evidence_sha256=q.sha256(coefficient_path),
        lal_binary=binary, rows=[], errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, result)
    try:
        for prior in evidence["rows"]:
            row = row_result(prior, plan["output_step_M"], gsl)
            if abs(row["public_mode_peak_minus_match_M"]) > plan["output_step_M"]:
                raise ValueError(f"{row['name']} mode amplitude peak is not at NQC matching")
            result["rows"].append(row)
            q.write_evidence(args.json_out, result)
            print(f"row={row['name']}", flush=True)
        result["status"] = "diagnostic_complete"
    except Exception as exc:
        result["status"] = "failed"
        result["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, result)
    print(json.dumps(dict(status=result["status"], errors=result["errors"])), flush=True)
    return 0 if result["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

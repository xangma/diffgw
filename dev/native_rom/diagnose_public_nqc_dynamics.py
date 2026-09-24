"""Compare native pre-NQC orbital scalars with pinned public high dynamics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import torch
from scipy.optimize import brentq

from . import qualify_generator as q
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_public_nqc_mode_boundary import public_call


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/public_nqc_dynamics_plan.json"


def nqc_coefficients_at(samples, match, targets, gsl):
    time = np.asarray(samples["time"], dtype=float)
    amplitude = np.asarray(samples["amplitude"], dtype=float)
    products = np.asarray(samples["products"], dtype=float)
    bases = np.asarray(samples["phase_basis"], dtype=float)
    phase = -np.asarray(samples["waveform_phase"], dtype=float)
    amp_matrix = np.stack(
        [gsl.derivatives(time, products[:, j], match) for j in range(3)], axis=-1)
    phase_matrix = -np.stack(
        [gsl.derivatives(time, bases[:, j], match)[1:] for j in range(2)], axis=-1)
    raw_omega, raw_omega_dot = gsl.derivatives(time, phase, match)[1:]
    omega = abs(raw_omega)
    omega_dot = (abs(raw_omega_dot) if raw_omega_dot * raw_omega > 0
                 else -abs(raw_omega_dot))
    return np.concatenate((
        np.linalg.solve(amp_matrix, targets[:3] - gsl.derivatives(time, amplitude, match)),
        np.linalg.solve(phase_matrix, targets[3:] - [omega, omega_dot]),
    ))


def replay_native(item, waveform, level, settings):
    numerical = waveform["choice"]["numerical"]
    physical = waveform["choice"]["physical"]
    flux = waveform["choice"]["flux_placements"]
    return q.native_run(
        waveform["row"], level, settings,
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
        return_nqc_coefficients=True, return_nqc_samples=True,
    )


def row_result(item, prior, boundary, same_input, plan, gsl):
    waveform_path = q.ROOT / item["waveform"]
    if q.sha256(waveform_path) != item["waveform_sha256"]:
        raise ValueError(f"{item['name']} frozen waveform differs")
    waveform = json.loads(waveform_path.read_text())
    level = waveform["candidate"][plan["waveform_level"]]["level"]
    run = replay_native(item, waveform, level, q.Settings(**waveform["settings"]))
    replay = q.summarize_run(run)["array_sha256"]
    if replay != prior["native_waveform_sha256"]:
        raise ValueError(f"{item['name']} native waveform does not reproduce")
    samples = run["nqc_samples"]
    time = np.asarray(samples["time"], dtype=float)
    products = np.asarray(samples["products"], dtype=float)
    bases = np.asarray(samples["phase_basis"], dtype=float)
    if (products.shape != (len(time), 3) or bases.shape != (len(time), 2)
            or not np.allclose(np.diff(time), np.diff(time)[0], rtol=0, atol=1e-9)
            or np.any(products[:, :2] <= 0) or np.any(bases[:, 0] >= 0)
            or np.any(bases[:, 1] >= 0)):
        raise ValueError(f"{item['name']} NQC samples violate inward-orbit basis")
    radius = products[:, 0] / products[:, 1]
    radial_momentum = -np.sqrt(bases[:, 1] / bases[:, 0])
    orbital_frequency = radial_momentum / (radius * bases[:, 0])
    if (not np.isfinite([radius, radial_momentum, orbital_frequency]).all()
            or np.any(orbital_frequency <= 0)):
        raise ValueError(f"{item['name']} reconstructed dynamics invalid")
    native_peak = samples["match_time"] + boundary["source_nqc_peak_offset_M"]
    output, _ = public_call(prior["row"], plan["public_output_step_M"], 0)
    high = np.asarray(output[2].data, dtype=float).reshape(5, -1)
    if q.canonical_hash(high.tolist()) != boundary["high_dynamics_sha256"]:
        raise ValueError(f"{item['name']} public high dynamics do not reproduce")
    public_time = high[0] - high[0, 0]
    public_peak = boundary["orbital_peak_time_M"]
    step = plan["public_output_step_M"]
    if (not np.allclose(np.diff(public_time), step, rtol=0, atol=1e-9)
            or abs(np.diff(time)[0] - step) > 1e-9):
        raise ValueError(f"{item['name']} native/public internal grids differ")
    evaluations = []
    for offset in plan["peak_relative_offsets_M"]:
        native_query = native_peak + offset
        native_values = np.array([
            gsl.derivatives(time, values, native_query)[0]
            for values in (radius, radial_momentum, orbital_frequency)
        ])
        public_values = []
        for shift in (-step, 0.0, step):
            query = public_peak + offset + shift
            public_values.append([
                gsl.derivatives(public_time, high[1], query)[0],
                gsl.derivatives(public_time, high[3], query)[0],
                gsl.derivatives(public_time, high[2], query)[1],
            ])
        public_values = np.asarray(public_values)
        relative = native_values / public_values - 1
        evaluations.append(dict(
            peak_relative_time_M=offset, native=native_values.tolist(),
            public_center=public_values[1].tolist(),
            relative_center=relative[1].tolist(),
            relative_one_sample_envelope=np.stack(
                (relative.min(axis=0), relative.max(axis=0)), axis=1).tolist(),
        ))
    equal_frequency = []
    public_phase = high[2]
    for frequency in plan["fixed_orbital_frequencies"]:
        crossing = np.where((orbital_frequency[:-1] <= frequency)
                            & (frequency <= orbital_frequency[1:])
                            & (time[1:] < native_peak))[0]
        if len(crossing) != 1:
            raise ValueError(f"{item['name']} frequency {frequency} has {len(crossing)} native crossings")
        i = crossing[0]
        native_query = float(np.interp(
            frequency, orbital_frequency[i:i + 2], time[i:i + 2]))
        public_query = brentq(
            lambda query: gsl.derivatives(public_time, public_phase, query)[1] - frequency,
            public_time[0], public_peak, xtol=1e-12)
        native_values = [gsl.derivatives(time, values, native_query)[0]
                         for values in (radius, radial_momentum)]
        public_values = [gsl.derivatives(public_time, values, public_query)[0]
                         for values in (high[1], high[3])]
        equal_frequency.append(dict(
            orbital_frequency=frequency,
            native_peak_relative_time_M=native_query - native_peak,
            public_peak_relative_time_M=public_query - public_peak,
            peak_relative_time_difference_M=(native_query - native_peak)
                                            - (public_query - public_peak),
            native_r_pr_star=native_values, public_r_pr_star=public_values,
            relative_r_pr_star=(np.asarray(native_values) / public_values - 1).tolist(),
            public_frequency_root_residual=float(
                gsl.derivatives(public_time, public_phase, public_query)[1] - frequency),
        ))
    targets = np.asarray(prior["target_counterfactual"]["source_targets"], dtype=float)
    public_coefficients = np.asarray(prior["public_coefficients"], dtype=float)
    baseline = nqc_coefficients_at(samples, samples["match_time"], targets, gsl)
    if not np.allclose(baseline, same_input["gsl_coefficients"], rtol=0, atol=1e-8):
        raise ValueError(f"{item['name']} same-input GSL baseline differs")
    timing_sensitivity = []
    for offset in plan["nqc_match_time_offsets_M"]:
        coefficients = nqc_coefficients_at(samples, samples["match_time"] + offset, targets, gsl)
        timing_sensitivity.append(dict(
            match_time_offset_M=offset,
            coefficients=coefficients.tolist(),
            distance_to_public_l2=float(np.linalg.norm(coefficients - public_coefficients)),
        ))
    return dict(
        name=item["name"], row=prior["row"], native_waveform_sha256=replay,
        native_nqc_sample_count=len(time), public_high_dynamics_sha256=boundary["high_dynamics_sha256"],
        native_nqc_match_time_M=float(samples["match_time"]),
        native_inferred_peak_time_M=float(native_peak), public_inferred_peak_time_M=public_peak,
        public_peak_uncertainty_M=step, variables=["r", "pr_star", "orbital_frequency"],
        comparisons=evaluations, equal_frequency_comparisons=equal_frequency,
        timing_sensitivity=timing_sensitivity,
        timing_sensitivity_holds_source_targets_fixed=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.public-nqc-dynamics-plan.v2"
            or plan["rows"] != ["strong_aligned_spin", "strong_antialigned_spin"]
            or plan["peak_relative_offsets_M"] != [-50, -25, -12, -5, 0]
            or plan["fixed_orbital_frequencies"] != [0.07, 0.08, 0.10, 0.12, 0.14, 0.16]
            or plan["nqc_match_time_offsets_M"] != [
                -0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0,
                0.005, 0.01, 0.015, 0.02, 0.025, 0.03]):
        parser.error("frozen dynamics plan differs")
    source_files = ("coefficient_evidence", "boundary_evidence", "waveform_plan", "same_input_evidence")
    paths = {key: q.ROOT / plan[key] for key in source_files}
    for key, path in paths.items():
        if q.sha256(path) != plan[key + "_sha256"]:
            parser.error(f"{key} differs")
    coefficients = json.loads(paths["coefficient_evidence"].read_text())
    boundary = json.loads(paths["boundary_evidence"].read_text())
    waveform_plan = json.loads(paths["waveform_plan"].read_text())
    same_input = json.loads(paths["same_input_evidence"].read_text())
    binary = coefficients["lal_binary"]
    if (q.lal_binary_identity(binary["lal_version"], binary["lalsimulation_version"]) != binary
            or boundary["lal_binary"] != binary
            or [r["name"] for r in coefficients["rows"]] != plan["rows"]
            or [r["name"] for r in boundary["rows"]] != plan["rows"]
            or [r["name"] for r in waveform_plan["rows"]] != plan["rows"]
            or [r["name"] for r in same_input["rows"]] != plan["rows"]):
        parser.error("pinned rows or public binary differ")
    gsl_paths = [Path(path) for path in binary["binaries"] if Path(path).name.startswith("libgsl.")]
    if len(gsl_paths) != 1:
        parser.error("pinned GSL library is ambiguous")
    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.public-nqc-dynamics.v2", scope=q.SCOPE,
        status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), lal_binary=binary, rows=[], errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        gsl = GSLCubic(gsl_paths[0])
        for item, prior, bound, gsl_prior in zip(waveform_plan["rows"], coefficients["rows"],
                                                boundary["rows"], same_input["rows"]):
            report["rows"].append(row_result(item, prior, bound, gsl_prior, plan, gsl))
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

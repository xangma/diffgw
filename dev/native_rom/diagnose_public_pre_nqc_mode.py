"""Recover the pinned public inspiral 22 mode before NQC at fixed orbital frequency."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

import lal
import lalsimulation as lalsim
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import torch

from . import qualify_generator as q
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_public_nqc_dynamics import replay_native
from .diagnose_public_nqc_mode_boundary import public_call


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/public_pre_nqc_mode_plan.json"


def public_raw_mode_at(time, high, high_mode, coefficients, query, gsl):
    radius, radius_dot = gsl.derivatives(time, high[1], query)[:2]
    momentum, momentum_dot = gsl.derivatives(time, high[3], query)[:2]
    _, frequency, frequency_dot = gsl.derivatives(time, high[2], query)
    post_amplitude = np.abs(high_mode)
    post_phase = np.unwrap(np.angle(high_mode))
    amplitude, amplitude_dot = gsl.derivatives(time, post_amplitude, query)[:2]
    phase_dot = gsl.derivatives(time, post_phase, query)[1]
    ratio = momentum / (radius * frequency)
    ratio_dot = ratio * (momentum_dot / momentum - radius_dot / radius
                         - frequency_dot / frequency)
    a1, a2, a3, b1, b2 = coefficients
    correction = 1 + ratio**2 * (a1 + a2 / radius + a3 / radius**1.5)
    correction_dot = (
        2 * ratio * ratio_dot * (a1 + a2 / radius + a3 / radius**1.5)
        - ratio**2 * radius_dot * (a2 / radius**2 + 1.5 * a3 / radius**2.5)
    )
    phase_correction_dot = (b1 * ratio_dot
                            + b2 * (ratio_dot * momentum**2
                                    + 2 * ratio * momentum * momentum_dot))
    if correction <= 0:
        raise ValueError("public NQC amplitude factor is not positive")
    return dict(
        post_nqc_amplitude=amplitude, amplitude_factor=correction,
        raw_amplitude=amplitude / correction,
        raw_amplitude_derivative=(amplitude_dot * correction
                                  - amplitude * correction_dot) / correction**2,
        raw_gravitational_wave_frequency=-phase_dot + phase_correction_dot,
        orbital_frequency=frequency,
    )


def row_result(item, prior, waveform_item, boundary, dynamics, plan, gsl):
    waveform_path = q.ROOT / waveform_item["waveform"]
    if q.sha256(waveform_path) != waveform_item["waveform_sha256"]:
        raise ValueError(f"{item['name']} frozen waveform differs")
    waveform = json.loads(waveform_path.read_text())
    level = waveform["candidate"][plan["waveform_level"]]["level"]
    run = replay_native(waveform_item, waveform, level, q.Settings(**waveform["settings"]))
    replay = q.summarize_run(run)["array_sha256"]
    if replay != prior["native_waveform_sha256"] or replay != dynamics["native_waveform_sha256"]:
        raise ValueError(f"{item['name']} native waveform does not reproduce")
    samples = run["nqc_samples"]
    native_time = np.asarray(samples["time"], dtype=float)
    products = np.asarray(samples["products"], dtype=float)
    bases = np.asarray(samples["phase_basis"], dtype=float)
    native_radius = products[:, 0] / products[:, 1]
    native_momentum = -np.sqrt(bases[:, 1] / bases[:, 0])
    native_frequency = native_momentum / (native_radius * bases[:, 0])
    native_peak = dynamics["native_inferred_peak_time_M"]
    output, _ = public_call(prior["row"], plan["public_output_step_M"], 0)
    modes, _, high_dynamics = output
    high = np.asarray(high_dynamics.data, dtype=float).reshape(5, -1)
    if q.canonical_hash(high.tolist()) != boundary["high_dynamics_sha256"]:
        raise ValueError(f"{item['name']} public high dynamics do not reproduce")
    mode = lalsim.SphHarmTimeSeriesGetMode(modes, 2, 2)
    strain = np.asarray(mode.data.data, dtype=np.complex128)
    if q.canonical_hash(strain.view(np.float64).tolist()) != boundary["mode_sha256"]:
        raise ValueError(f"{item['name']} public mode does not reproduce")
    start = boundary["high_mode_start_index"]
    high_mode = strain[start:start + high.shape[1]]
    time = high[0] - high[0, 0]
    mass_distance_scale = ((prior["row"]["mass1"] + prior["row"]["mass2"])
                           * lal.MRSUN_SI / (100e6 * lal.PC_SI))
    high_mode = high_mode / mass_distance_scale
    phase_spline = CubicSpline(time, high[2], bc_type="natural")
    omega_spline = phase_spline.derivative()
    coefficients = boundary["public_coefficients"]
    comparisons = []
    for frequency in item["orbital_frequencies"]:
        crossing = np.where((native_frequency[:-1] <= frequency)
                            & (frequency <= native_frequency[1:])
                            & (native_time[1:] < native_peak))[0]
        if len(crossing) != 1:
            raise ValueError(f"{item['name']} frequency {frequency} has {len(crossing)} native crossings")
        i = crossing[0]
        native_query = float(np.interp(
            frequency, native_frequency[i:i + 2], native_time[i:i + 2]))
        public_query = brentq(lambda t: omega_spline(t) - frequency,
                              time[0], boundary["orbital_peak_time_M"], xtol=1e-12)
        if public_query + plan["pre_attachment_margin_M"] >= min(
                boundary["inferred_match_time_M"],
                boundary["public_mode_amplitude_peak_time_M"]):
            raise ValueError(f"{item['name']} frequency {frequency} is too near attachment")
        public = public_raw_mode_at(time, high, high_mode, coefficients, public_query, gsl)
        native_amplitude = gsl.derivatives(native_time, samples["amplitude"], native_query)
        native_gw_frequency = gsl.derivatives(
            native_time, np.asarray(samples["waveform_phase"]), native_query)[1]
        omega_gsl = gsl.derivatives(time, high[2], public_query)[1]
        comparisons.append(dict(
            orbital_frequency=frequency,
            public_peak_relative_time_M=public_query - boundary["orbital_peak_time_M"],
            public_match_time_margin_M=boundary["inferred_match_time_M"] - public_query,
            public_mode_peak_margin_M=boundary["public_mode_amplitude_peak_time_M"] - public_query,
            public_orbital_frequency_crosscheck=omega_gsl - frequency,
            native_peak_relative_time_M=native_query - native_peak,
            public=public,
            native_raw_amplitude=native_amplitude[0],
            native_raw_amplitude_derivative=native_amplitude[1],
            native_raw_gravitational_wave_frequency=native_gw_frequency,
            raw_amplitude_relative=native_amplitude[0] / public["raw_amplitude"] - 1,
            raw_amplitude_derivative_difference=(native_amplitude[1]
                                                 - public["raw_amplitude_derivative"]),
            raw_gravitational_wave_frequency_relative=(native_gw_frequency
                                                        / public["raw_gravitational_wave_frequency"] - 1),
        ))
    return dict(name=item["name"], row=prior["row"], native_waveform_sha256=replay,
                public_high_dynamics_sha256=boundary["high_dynamics_sha256"],
                public_mode_sha256=boundary["mode_sha256"],
                public_coefficients=coefficients, comparisons=comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.public-pre-nqc-mode-plan.v1"
            or [row["name"] for row in plan["rows"]] != [
                "strong_aligned_spin", "strong_antialigned_spin"]
            or plan["public_output_step_M"] != 0.015625
            or plan["pre_attachment_margin_M"] != 1.0):
        parser.error("frozen public pre-NQC plan differs")
    source_files = ("coefficient_evidence", "waveform_plan", "boundary_evidence",
                    "dynamics_evidence")
    paths = {key: q.ROOT / plan[key] for key in source_files}
    for key, path in paths.items():
        if q.sha256(path) != plan[key + "_sha256"]:
            parser.error(f"{key} differs")
    coefficients = json.loads(paths["coefficient_evidence"].read_text())
    waveform_plan = json.loads(paths["waveform_plan"].read_text())
    boundary = json.loads(paths["boundary_evidence"].read_text())
    dynamics = json.loads(paths["dynamics_evidence"].read_text())
    binary = coefficients["lal_binary"]
    if (q.lal_binary_identity(binary["lal_version"], binary["lalsimulation_version"]) != binary
            or boundary["lal_binary"] != binary or dynamics["lal_binary"] != binary
            or any([row["name"] for row in source["rows"]]
                   != [item["name"] for item in plan["rows"]]
                   for source in (coefficients, waveform_plan, boundary, dynamics))):
        parser.error("pinned rows or binary differ")
    gsl_paths = [Path(path) for path in binary["binaries"] if Path(path).name.startswith("libgsl.")]
    if len(gsl_paths) != 1:
        parser.error("pinned GSL library is ambiguous")
    torch.set_num_threads(1)
    report = dict(
        schema="diffgw.seobnrv4.public-pre-nqc-mode.v1", scope=q.SCOPE,
        status="running", release_qualified=False, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), lal_binary=binary, rows=[], errors=[],
        process=dict(host=socket.gethostname(), cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv),
    )
    q.write_evidence(args.json_out, report)
    try:
        gsl = GSLCubic(gsl_paths[0])
        for item, prior, wave, bound, dyn in zip(
                plan["rows"], coefficients["rows"], waveform_plan["rows"],
                boundary["rows"], dynamics["rows"]):
            report["rows"].append(row_result(item, prior, wave, bound, dyn, plan, gsl))
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

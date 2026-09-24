"""Frozen frequency-aligned source-flux trajectory and pre-NQC diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_boundary_strong_spin_ode import array, array_hash
from .diagnose_lal_dynamics import COORDINATES, _five_point, _reference
from .diagnose_nqc_gsl_same_input import GSLCubic
from .eob.torch.adaptive_driver_clean import reconstruct_seobnrv4_22_adaptive


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_frequency_trace_plan.json"


def _digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _crossing(time_values, state_values, frequencies, target):
    indices = np.flatnonzero((frequencies[:-1] <= target)
                             & (target <= frequencies[1:]))
    if len(indices) != 1:
        raise ValueError(f"M Omega={target} has {len(indices)} pre-peak crossings")
    index = int(indices[0])
    fraction = (target-frequencies[index]) / (frequencies[index+1]-frequencies[index])
    state = state_values[index] + fraction*(state_values[index+1]-state_values[index])
    return dict(time_M=float(time_values[index] + fraction*(time_values[index+1]
                                                              -time_values[index])),
                state=dict(zip(COORDINATES, state.tolist(), strict=True)),
                bracket_indices=[index, index+1],
                bracket_frequencies_Momega=frequencies[index:index+2].tolist())


def _native(prior):
    row = prior["row"]
    numerical = prior["choice"]["numerical"]
    physical = prior["choice"]["physical"]
    placements = prior["choice"]["flux_placements"]
    level = prior["candidate"]["ode_2"]["level"]
    settings = q.Settings(**prior["settings"])
    started = time.monotonic()
    with q.bounded_run(settings.max_run_seconds), torch.no_grad():
        result = reconstruct_seobnrv4_22_adaptive(
            torch.tensor(row["omega_start"], dtype=torch.float64),
            **{key: row[key] for key in ("mass1", "mass2", "chi1", "chi2")},
            **q.CHOICES, **{key: level[key] for key in ("atol", "rtol", "dt_geom")},
            diagnostic_hermite_resampling=numerical["hermite_resampling"],
            diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
            diagnostic_matching_second_derivatives=True,
            diagnostic_restart_time_M=numerical["fixed_restart_time_M"],
            diagnostic_coarse_initial_step_M=0.125,
            diagnostic_warm_start_Momega=0.03162277660168379,
            diagnostic_nqc_phase_window_M=2.0,
            diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
            diagnostic_hamiltonian_log_derivative=physical[
                "hamiltonian_log_derivative"],
            diagnostic_mode_21_higher_spin=placements["mode_21_higher_spin"],
            diagnostic_mode_31_higher_spin=placements["mode_31_higher_spin"],
            diagnostic_mode_32_spin_square=placements["mode_32_spin_square"],
            ringdown_duration=settings.ringdown_duration_M,
            num_steps=settings.max_accepted_steps,
            max_accepted_steps=settings.max_accepted_steps,
            max_evaluations=settings.max_evaluations,
            max_rejected_steps=settings.max_rejected_steps)
    waveform_time, waveform_mode = array(result.time), array(result.mode)
    expected = prior["candidate"]["ode_2"]["native"]["array_sha256"]
    if (len(waveform_time) != prior["candidate"]["ode_2"]["native"]["samples"]
            or _digest(waveform_time) != expected["time"]
            or _digest(waveform_mode) != expected["mode"]):
        raise ValueError("native source-flux waveform hashes did not replay")
    trajectory = result.trajectory
    valid = array(trajectory.valid).astype(bool)
    if not np.all(valid[:int(valid.sum())]) or np.any(valid[int(valid.sum()):]):
        raise ValueError("native trajectory validity is not a prefix")
    peak = float(array(trajectory.peak.time))
    n = int(valid.sum())
    times = array(trajectory.time)[:n]
    states = array(trajectory.state)[:n]
    frequencies = array(trajectory.frequency)[:n]
    pre_peak = int(np.searchsorted(times, peak, side="left"))
    nqc = result.attachment.fine.nqc
    mode_samples = dict(
        time=array(nqc.amplitude_spline.knots),
        amplitude=array(nqc.sampled_amplitude),
        products=array(nqc.sampled_products),
        phase_basis=array(nqc.sampled_phase_basis),
        waveform_phase=array(nqc.sampled_waveform_phase))
    return (times[:pre_peak], states[:pre_peak], frequencies[:pre_peak],
            mode_samples, dict(elapsed_seconds=time.monotonic()-started,
                               waveform_array_sha256=expected,
                               valid_samples=n, pre_peak_samples=pre_peak,
                               orbital_peak_time_M=peak,
                               match_time_M=float(array(result.attachment.match_time))))


def _native_raw_mode(samples, target, gsl):
    time_values = samples["time"]
    products, bases = samples["products"], samples["phase_basis"]
    radius = products[:, 0]/products[:, 1]
    momentum = -np.sqrt(bases[:, 1]/bases[:, 0])
    frequency = momentum/(radius*bases[:, 0])
    indices = np.flatnonzero((frequency[:-1] <= target)
                             & (target <= frequency[1:]))
    if len(indices) != 1:
        raise ValueError(f"native raw mode M Omega={target} has {len(indices)} crossings")
    i = int(indices[0])
    query = float(np.interp(target, frequency[i:i+2], time_values[i:i+2]))
    amp = gsl.derivatives(time_values, samples["amplitude"], query)[0]
    gravitational_frequency = gsl.derivatives(
        time_values, samples["waveform_phase"], query)[1]
    return dict(local_time_M=query, raw_amplitude=amp,
                raw_gravitational_wave_frequency=gravitational_frequency)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    paths = {key: q.ROOT / plan[key] for key in (
        "source_flux_waveform", "public_field", "pre_nqc_mode")}
    if (plan["schema"] != "diffgw.seobnrv4.source-flux-frequency-trace-plan.v1"
            or plan["row_name"] != "strong_aligned_spin"
            or plan["native_level"] != "ode_2"
            or plan["public_output_steps_M"] != [0.125, 0.0625]
            or plan["orbital_frequencies_Momega"] != [
                0.0425, 0.045, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.13]
            or plan["phase_anchor_Momega"] != 0.0425
            or plan["mode_comparison_frequencies_Momega"] != [
                0.07, 0.08, 0.1, 0.12, 0.13]
            or plan["release_qualified"] is not False
            or any(q.sha256(path) != plan[key+"_sha256"]
                   for key, path in paths.items())):
        parser.error("frozen plan or prerequisite hashes differ")
    prior, public_field, pre_nqc = (json.loads(paths[key].read_text()) for key in
                                    ("source_flux_waveform", "public_field", "pre_nqc_mode"))
    expected_choice = dict(hamiltonian_log_derivative="lal_v4",
                           mode_33_higher_spin="lal_v4_rho", name="combined")
    if (any(item["status"] != "diagnostic_complete"
            for item in (prior, public_field, pre_nqc))
            or prior["row"] != public_field["row"]
            or prior["row"] != pre_nqc["rows"][0]["row"]
            or prior["choice"]["physical"] != expected_choice
            or prior["choice"]["flux_placements"] != dict(
                mode_21_higher_spin="rho_hybrid", mode_31_higher_spin="lal_v4_rho",
                mode_32_spin_square="lal_v4_zero")
            or prior["candidate"]["ode_2"]["level"]["dt_geom"] != 0.125
            or pre_nqc["rows"][0]["name"] != "strong_aligned_spin"):
        parser.error("frozen row, source choice, or numerical level differs")
    pin = prior["lal_binary"]
    if (q.lal_binary_identity("7.7.1", "6.2.1") != pin
            or public_field["lal_binary"] != pin or pre_nqc["lal_binary"] != pin):
        parser.error("pinned public binary identity changed")
    torch.set_num_threads(1)
    sources = ("dev/native_rom/eob/torch/adaptive_driver_clean.py",
               "dev/native_rom/eob/torch/historical_hamiltonian.py",
               "src/torchwave/approximants/eob/seobnrv4_flux.py",
               "dev/native_rom/diagnose_lal_dynamics.py",
               "dev/native_rom/diagnose_nqc_gsl_same_input.py")
    report = dict(
        schema="diffgw.seobnrv4.source-flux-frequency-trace.v1", scope=q.SCOPE,
        status="running", release_qualified=False,
        direct_public_flux_parity_established=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        prerequisite_sha256={key: q.sha256(path) for key, path in paths.items()},
        source_sha256={name: q.sha256(q.ROOT/name) for name in sources},
        lal_binary=pin, row=prior["row"],
        frozen_fitted_complex_relative_l2=prior["fitted_peak"]["metrics"][
            "complex_relative_l2"],
        frequency_grid_Momega=plan["orbital_frequencies_Momega"],
        phase_anchor_Momega=plan["phase_anchor_Momega"],
        native={}, public_grids={}, comparisons=[], pre_nqc_mode=[],
        summary={}, errors=[], process=dict(host=socket.gethostname(), cwd=os.getcwd(),
                                             pid=os.getpid(), argv=sys.argv))
    q.write_evidence(args.json_out, report)
    try:
        print("replaying_source_flux_native_waveform", flush=True)
        nt, ns, nw, mode_samples, native_meta = _native(prior)
        report["native"] = native_meta
        q.write_evidence(args.json_out, report)
        native_points = {target: _crossing(nt, ns, nw, target)
                         for target in plan["orbital_frequencies_Momega"]}
        public_points = {}
        for step in plan["public_output_steps_M"]:
            print(f"public_output_step_M={step}", flush=True)
            values = _reference(prior["row"], step)
            digest = array_hash(values)
            if digest != public_field["grids"][str(step)]["reference_array_sha256"]:
                raise ValueError(f"public dynamics {step} M hash did not replay")
            usable = np.arange(2, values.shape[1]-2)
            frequency = _five_point(values[2], usable, step)
            peak_index = int(np.argmax(frequency))
            public_points[str(step)] = {
                target: _crossing(values[0, usable[:peak_index+1]],
                                  values[1:, usable[:peak_index+1]].T,
                                  frequency[:peak_index+1], target)
                for target in plan["orbital_frequencies_Momega"]}
            report["public_grids"][str(step)] = dict(
                shape=list(values.shape), reference_array_sha256=digest,
                pre_peak_samples=peak_index+1)
            q.write_evidence(args.json_out, report)
        anchor = plan["phase_anchor_Momega"]
        for target in plan["orbital_frequencies_Momega"]:
            native = native_points[target]
            native_span = native["state"]["phi"]-native_points[anchor]["state"]["phi"]
            public = {}
            for step in plan["public_output_steps_M"]:
                item = public_points[str(step)][target]
                reference = public_points[str(step)][anchor]
                phase_span = item["state"]["phi"]-reference["state"]["phi"]
                public[str(step)] = dict(
                    **item, orbital_phase_span_rad=phase_span,
                    time_since_anchor_M=item["time_M"]-reference["time_M"],
                    native_minus_public_phase_span_rad=native_span-phase_span,
                    native_minus_public_time_span_M=(native["time_M"]
                         -native_points[anchor]["time_M"]
                         -item["time_M"]+reference["time_M"]),
                    native_minus_public_state={key: native["state"][key]
                                               -item["state"][key]
                                               for key in ("r", "pr_star", "pphi")})
            coarse, fine = public["0.125"], public["0.0625"]
            report["comparisons"].append(dict(
                frequency_Momega=target, native=dict(**native,
                    orbital_phase_span_rad=native_span,
                    time_since_anchor_M=native["time_M"]
                    -native_points[anchor]["time_M"]),
                public=public,
                public_grid_sensitivity=dict(
                    phase_span_coarse_minus_fine_rad=(coarse["orbital_phase_span_rad"]
                                                       -fine["orbital_phase_span_rad"]),
                    time_span_coarse_minus_fine_M=(coarse["time_since_anchor_M"]
                                                    -fine["time_since_anchor_M"]),
                    state_coarse_minus_fine={key: coarse["state"][key]
                                              -fine["state"][key]
                                              for key in ("r", "pr_star", "pphi")})))
            q.write_evidence(args.json_out, report)
            print(json.dumps(dict(frequency_Momega=target,
                phase_gap_rad=fine["native_minus_public_phase_span_rad"],
                public_grid_phase_span_delta_rad=(coarse["orbital_phase_span_rad"]
                                                   -fine["orbital_phase_span_rad"]))),
                flush=True)
        gsl_paths = [Path(path) for path in pin["binaries"]
                     if Path(path).name.startswith("libgsl.")]
        if len(gsl_paths) != 1:
            raise ValueError("pinned GSL library is ambiguous")
        gsl = GSLCubic(gsl_paths[0])
        prior_mode = {item["orbital_frequency"]: item for item in
                      pre_nqc["rows"][0]["comparisons"]}
        for target in plan["mode_comparison_frequencies_Momega"]:
            native = _native_raw_mode(mode_samples, target, gsl)
            public = prior_mode[target]["public"]
            report["pre_nqc_mode"].append(dict(
                frequency_Momega=target, native=native,
                public_raw_amplitude=public["raw_amplitude"],
                public_raw_gravitational_wave_frequency=public[
                    "raw_gravitational_wave_frequency"],
                raw_amplitude_relative=native["raw_amplitude"]
                    /public["raw_amplitude"]-1,
                raw_gravitational_wave_frequency_relative=(
                    native["raw_gravitational_wave_frequency"]
                    /public["raw_gravitational_wave_frequency"]-1),
                native_minus_public_raw_gravitational_wave_frequency=(
                    native["raw_gravitational_wave_frequency"]
                    -public["raw_gravitational_wave_frequency"]),
                public_reconstruction_match_margin_M=prior_mode[target][
                    "public_match_time_margin_M"]))
        points = report["comparisons"]
        report["summary"] = dict(
            max_abs_fine_native_minus_public_phase_span_rad=max(abs(
                item["public"]["0.0625"]["native_minus_public_phase_span_rad"])
                for item in points),
            max_abs_public_grid_phase_span_delta_rad=max(abs(
                item["public_grid_sensitivity"]["phase_span_coarse_minus_fine_rad"])
                for item in points),
            max_abs_fine_native_minus_public_time_span_M=max(abs(
                item["public"]["0.0625"]["native_minus_public_time_span_M"])
                for item in points),
            max_abs_pre_nqc_raw_amplitude_relative=max(abs(
                item["raw_amplitude_relative"]) for item in report["pre_nqc_mode"]),
            max_abs_pre_nqc_raw_gravitational_wave_frequency_relative=max(abs(
                item["raw_gravitational_wave_frequency_relative"])
                for item in report["pre_nqc_mode"]))
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"],
                          summary=report["summary"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

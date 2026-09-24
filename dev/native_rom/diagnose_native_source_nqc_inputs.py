"""Locate the strong-spin NQC gap using captured pinned-source pre-fit arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_nqc_gsl_same_input import GSLCubic
from .diagnose_source_flux_initial import _libraries


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/native_source_nqc_inputs_plan.json"


def _fit(gsl: GSLCubic, time: np.ndarray, amplitude: np.ndarray,
         phase: np.ndarray, radius: np.ndarray, momentum: np.ndarray,
         frequency: np.ndarray, match: float, targets: np.ndarray) -> dict:
    q1 = (momentum / (radius * frequency)) ** 2
    q2 = q1 / radius
    products = amplitude[:, None] * np.stack((q1, q2, q2 / np.sqrt(radius)), axis=-1)
    p1 = momentum / (radius * frequency)
    phase_basis = np.stack((p1, p1 * momentum**2), axis=-1)
    amp_matrix = np.stack([gsl.derivatives(time, products[:, j], match)
                           for j in range(3)], axis=-1)
    phase_matrix = -np.stack([gsl.derivatives(time, phase_basis[:, j], match)[1:]
                              for j in range(2)], axis=-1)
    amp_derivatives = np.asarray(gsl.derivatives(time, amplitude, match))
    phase_derivatives = np.asarray(gsl.derivatives(time, phase, match))
    raw_frequency, raw_derivative = phase_derivatives[1:]
    signed_derivative = (abs(raw_derivative) if raw_frequency * raw_derivative > 0
                         else -abs(raw_derivative))
    frequency_derivatives = np.array((abs(raw_frequency), signed_derivative))
    amp_rhs = targets[:3] - amp_derivatives
    phase_rhs = targets[3:] - frequency_derivatives
    coefficients = np.concatenate((np.linalg.solve(amp_matrix, amp_rhs),
                                   np.linalg.solve(phase_matrix, phase_rhs)))
    return dict(coefficients=coefficients.tolist(), amplitude_derivatives=amp_derivatives.tolist(),
                frequency_derivatives=frequency_derivatives.tolist(),
                amplitude_matrix=amp_matrix.tolist(), phase_matrix=phase_matrix.tolist(),
                amplitude_rhs=amp_rhs.tolist(), phase_rhs=phase_rhs.tolist(),
                amplitude_condition=float(np.linalg.cond(amp_matrix)),
                phase_condition=float(np.linalg.cond(phase_matrix)))


def _source_samples(asset: Path):
    with np.load(asset) as data:
        values = data["values"].copy()
        meta = data["meta"].copy()
        coefficients = data["coefficients"].copy()
    time = np.arange(len(values), dtype=np.float64) * meta[1]
    return time, values.T, meta, coefficients


def _native_samples(samples: dict):
    time = np.asarray(samples["time"], dtype=np.float64)
    amplitude = np.asarray(samples["amplitude"], dtype=np.float64)
    products = np.asarray(samples["products"], dtype=np.float64)
    basis = np.asarray(samples["phase_basis"], dtype=np.float64)
    radius = products[:, 0] / products[:, 1]
    momentum = -np.sqrt(basis[:, 1] / basis[:, 0])
    frequency = momentum / (radius * basis[:, 0])
    phase = -np.asarray(samples["waveform_phase"], dtype=np.float64)
    return time, np.stack((amplitude, phase, radius, momentum, frequency)), float(samples["match_time"])


def _replay(row, level, settings, native, prior):
    numerical = native["choice"]["numerical"]
    physical = native["choice"]["physical"]
    placements = native["choice"]["baseline_flux_placements"]
    return q.native_run(
        row, level, settings,
        diagnostic_hermite_resampling=numerical["hermite_resampling"],
        diagnostic_local_nqc_derivatives=numerical["local_nqc_phase_derivatives"],
        diagnostic_matching_second_derivatives=True,
        diagnostic_restart_time_M=numerical["fixed_restart_time_M"],
        diagnostic_coarse_initial_step_M=0.125,
        diagnostic_warm_start_Momega=0.03162277660168379,
        diagnostic_nqc_phase_window_M=prior["candidate_window_M"],
        diagnostic_mode_33_higher_spin=physical["mode_33_higher_spin"],
        diagnostic_hamiltonian_log_derivative=physical["hamiltonian_log_derivative"],
        diagnostic_mode_21_higher_spin=placements["mode_21_higher_spin"],
        diagnostic_mode_31_higher_spin=placements["mode_31_higher_spin"],
        diagnostic_mode_32_spin_square=placements["mode_32_spin_square"],
        diagnostic_spin_truncation="tabulated",
        diagnostic_mode_43_higher_spin="lal_v4_rho",
        return_nqc_coefficients=True, return_nqc_samples=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(PLAN.read_text())
    names = ("source_inputs", "source_array", "native_waveform", "prior_waveform",
             "source_flux_plan")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.native-source-nqc-inputs-plan.v1"
            or plan["row_name"] != "strong_aligned_spin" or plan["native_level"] != "ode_2"
            or plan["relative_match_offsets_M"] != [-12.5, -2.5, -1, 0, 1, 2.5, 5, 10, 12.5]
            or plan["native_array_asset"] != "dev/native_rom/evidence/2026-09-24/native_source_nqc_inputs.npz"
            or any(q.sha256(paths[name]) != plan[name+"_sha256"] for name in names)):
        parser.error("Frozen plan or prerequisites changed")
    source = json.loads(paths["source_inputs"].read_text())
    native = json.loads(paths["native_waveform"].read_text())
    prior = json.loads(paths["prior_waveform"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    if (source["status"] != "diagnostic_complete" or native["status"] != "diagnostic_complete"
            or source["array_asset_sha256"] != q.sha256(paths["source_array"])
            or native["row"] != source["rows"][0]["row"] or prior["row"] != native["row"]):
        parser.error("Pinned input row or array changed")
    gsl_path = _libraries(source_plan["pinned_wheel"])["libgsl.28.dylib"]
    gsl = GSLCubic(gsl_path)
    torch.set_num_threads(1)
    report = dict(schema="diffgw.seobnrv4.native-source-nqc-inputs.v1", scope=q.SCOPE,
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  input_sha256={name: q.sha256(path) for name, path in paths.items()},
                  gsl_library=str(gsl_path), process=dict(host=socket.gethostname(),
                  cwd=os.getcwd(), pid=os.getpid(), argv=sys.argv), comparison=None, errors=[])
    q.write_evidence(args.json_out, report)
    try:
        start = time.monotonic()
        source_time, source_values, source_meta, source_coefficients = _source_samples(
            paths["source_array"])
        source_match = float(source_meta[2])
        source_targets = source_meta[3:]
        source_fit = _fit(gsl, source_time, *source_values, source_match, source_targets)
        source_active = source_coefficients[[0, 1, 2, 6, 7]]
        source_rebuild_error = float(np.max(abs(np.asarray(source_fit["coefficients"])
                                                - source_active)))
        if source_rebuild_error > 1e-3:
            raise ValueError("Captured source inputs fail independent GSL fit replay")
        level = native["candidate"]["ode_2"]["level"]
        run = _replay(native["row"], level, q.Settings(**native["settings"]), native, prior)
        replay_hash = q.summarize_run(run)["array_sha256"]
        if replay_hash != native["candidate"]["ode_2"]["native"]["array_sha256"]:
            raise ValueError("Native waveform failed frozen replay")
        native_time, native_values, native_match = _native_samples(run["nqc_samples"])
        native_active = np.asarray(run["nqc_coefficients"]["amplitude"]
                                   + run["nqc_coefficients"]["phase"])
        array_path = q.ROOT / plan["native_array_asset"]
        np.savez_compressed(array_path, time=native_time, values=native_values,
                            match=np.asarray(native_match), coefficients=native_active)
        report["native_array_asset"] = plan["native_array_asset"]
        report["native_array_asset_sha256"] = q.sha256(array_path)
        native_fit = _fit(gsl, native_time, *native_values, native_match, source_targets)
        aligned = []
        phase_shift = (gsl.derivatives(source_time, source_values[1], source_match)[0]
                       - gsl.derivatives(native_time, native_values[1], native_match)[0])
        for offset in plan["relative_match_offsets_M"]:
            source_query = source_match + offset
            native_query = native_match + offset
            if not (source_time[0] <= source_query <= source_time[-1]
                    and native_time[0] <= native_query <= native_time[-1]):
                aligned.append(dict(relative_match_time_M=offset, available=False))
                continue
            source_vector = np.asarray([gsl.derivatives(source_time, values, source_query)[0]
                                        for values in source_values])
            native_vector = np.asarray([gsl.derivatives(native_time, values, native_query)[0]
                                        for values in native_values])
            source_phase_frequency = gsl.derivatives(source_time, source_values[1], source_query)[1]
            native_phase_frequency = gsl.derivatives(native_time, native_values[1], native_query)[1]
            aligned.append(dict(relative_match_time_M=offset, available=True,
                                source=source_vector.tolist(), native=native_vector.tolist(),
                                native_minus_source=(native_vector-source_vector).tolist(),
                                relative_nonphase=(native_vector[[0, 2, 3, 4]]
                                                   / source_vector[[0, 2, 3, 4]]-1).tolist(),
                                phase_difference_after_match_alignment=float(
                                    native_vector[1]+phase_shift-source_vector[1]),
                                source_phase_derivative=float(source_phase_frequency),
                                native_phase_derivative=float(native_phase_frequency)))
        if sum(item["available"] for item in aligned) < 5:
            raise ValueError("Insufficient common NQC support for comparison")
        report["comparison"] = dict(
            elapsed_seconds=time.monotonic()-start, source_samples=len(source_time),
            native_samples=len(native_time), source_match_time_M=source_match,
            native_match_time_M=native_match, source_targets=source_targets.tolist(),
            source_fit=source_fit, native_fit_with_source_targets=native_fit,
            source_active=source_active.tolist(), native_active=native_active.tolist(),
            source_fit_rebuild_max_abs_error=source_rebuild_error,
            native_gsl_minus_native=(np.asarray(native_fit["coefficients"])
                                     -native_active).tolist(),
            native_gsl_minus_source=(np.asarray(native_fit["coefficients"])
                                     -source_active).tolist(),
            coefficient_gap_l2=float(np.linalg.norm(native_active-source_active)),
            source_input_gap_l2=float(np.linalg.norm(
                np.asarray(native_fit["coefficients"])-np.asarray(source_fit["coefficients"]))),
            aligned_input_rows=aligned, native_waveform_sha256=replay_hash)
        report["status"] = "diagnostic_complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    q.write_evidence(args.json_out, report)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    return 0 if report["status"] == "diagnostic_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

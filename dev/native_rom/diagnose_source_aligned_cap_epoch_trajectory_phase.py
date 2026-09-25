"""Dev-only cap-epoch trajectory and raw-mode localization for one V4 row."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import lal
import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import diagnose_source_aligned_early_raw_crossover_v3 as early
from dev.native_rom import diagnose_source_aligned_early_raw_restart_v4 as restart
from dev.native_rom import diagnose_source_aligned_early_raw_warm_start_v5 as warm
from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_wheel_dynamics import _source as source_dynamics


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_cap_epoch_trajectory_phase_plan.json"
OUTPUT = EVIDENCE / "source_aligned_cap_epoch_trajectory_phase.json"
ROW = (30.0, 20.0, 0.8, 0.6, 0.04)


def relative_l2(left, right):
    return float(np.linalg.norm(left - right) / np.linalg.norm(right))


def pairs(values):
    return np.column_stack((values.real, values.imag)).tolist()


def paired(values):
    array = np.asarray(values, dtype=np.float64)
    return array[:, 0] + 1j * array[:, 1]


def at_times(time_M, values, target_M):
    """Interpolate only for the separately labeled conditional peak projection."""
    return np.column_stack([
        np.interp(target_M, time_M, values[:, j])
        for j in range(values.shape[1])
    ])


def main():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.cap-epoch-trajectory-phase-plan.v1"
            or plan["release_qualified"] is not False
            or tuple(plan["row_tuple"]) != ROW
            or plan["step_M"] != 0.5
            or plan["script_sha256"] != q.sha256(Path(__file__))):
        raise ValueError("frozen trajectory plan changed")
    for name, item in plan["inputs"].items():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen input changed: {name}")
    previous = json.loads((ROOT / plan["inputs"]["warm_result"]["path"]).read_text())
    full = json.loads((ROOT / plan["inputs"]["full_result"]["path"]).read_text())
    phase = json.loads((ROOT / plan["inputs"]["phase_result"]["path"]).read_text())
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    level = next(item for item in ladder["levels"] if item["name"] == "refined")
    if (previous["status"] != "diagnostic_complete"
            or full["status"] != "diagnostic_complete"
            or phase["status"] != "diagnostic_complete"):
        raise ValueError("frozen diagnostic controls are not complete")
    started = time.monotonic()
    result = dict(
        schema="diffgw.seobnrv4.cap-epoch-trajectory-phase.v1",
        status="running", release_qualified=False, host=platform.node(),
        cwd=os.getcwd(), pid=os.getpid(), plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(Path(__file__)),
    )
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.set_num_threads(1)
    m1, m2, chi1, chi2, _ = ROW

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    ham = driver.TorchSourceHam(tensor(m1*m2/(m1+m2)**2), tensor(chi1), tensor(chi2),
                                delta=tensor((m1-m2)/(m1+m2)))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    field = driver.TorchIntegrator(ham, flux).vector_field
    r0, pr0, pphi0 = driver.find_quasicircular_initial_conditions(
        ham, flux, tensor(plan["cap_Momega"]))
    native_initial = torch.stack((r0, torch.zeros_like(r0), pr0, pphi0)).detach()
    native_time, native_state = restart.run_coarse(
        field, native_initial, level, ladder, plan["step_M"])
    with tempfile.TemporaryDirectory(prefix="diffgw-cap-epoch-trajectory-") as tmp:
        private = Path(tmp) / "source"
        _snapshot(early.SOURCE_ROOT, source_plan["pinned_source"], private,
                  early.HEADERS)
        libraries = _libraries(source_plan["pinned_wheel"])

        def build(name, input_key):
            path = Path(tmp) / name
            _compile(private, early.HEADERS, early.GSL_INCLUDE, libraries,
                     ROOT / plan["inputs"][input_key]["path"], path,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            return ctypes.CDLL(str(path))

        dyn = build("dynamics.dylib", "dynamics_adapter").diffgw_v4_full_source_dynamics
        dyn.argtypes = ([ctypes.c_double]*6 + [ctypes.c_size_t,
                        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double)])
        dyn.restype = ctypes.c_int
        row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
                   omega_start=ROW[4])
        source_row = source_dynamics(dyn, row, plan["step_M"], 50000)
        source_time, source_state = source_row[0], source_row[1:].T
        exact = build("exact.dylib", "exact_adapter").diffgw_v4_source_nqc_state
        exact.argtypes = [ctypes.c_double]*8 + [ctypes.POINTER(ctypes.c_double)]
        exact.restype = ctypes.c_int
        source_omega, _ = early.source_raw(exact, ROW, source_state)
        native_omega, _ = early.source_raw(exact, ROW, native_state)

        chart_time = np.arange(plan["chart_start_M"],
                               plan["chart_end_M"] + plan["step_M"]/2,
                               plan["step_M"])
        start = int(round(plan["chart_start_M"]/plan["step_M"]))
        stop = start + len(chart_time)
        source_chart = source_state[start:stop]
        native_chart = native_state[start:stop]
        source_chart_time = source_time[start:stop]
        native_chart_time = native_time[start:stop]
        source_chart_omega = source_omega[start:stop]
        native_chart_omega = native_omega[start:stop]
        _, source_on_source = early.source_raw(exact, ROW, source_chart)
        native_on_source = early.native_raw(ham, flux, source_chart)
        native_on_native = early.native_raw(ham, flux, native_chart)

        target = np.asarray(previous["chart"]["Momega"], dtype=np.float64)
        source_frequency_chart, _, _ = early.frequency_states(
            source_omega, source_time, source_state, target)
        native_frequency_chart, _, _ = early.frequency_states(
            native_omega, native_time, native_state, target)
        source_crop_t, source_crop_state, _ = warm.crossing_state(
            source_omega, source_time, source_state, ROW[4])
        native_crop_t, native_crop_state, _ = warm.crossing_state(
            native_omega, native_time, native_state, ROW[4])
        source_frequency_chart[:, 1] -= source_crop_state[1]
        native_frequency_chart[:, 1] -= native_crop_state[1]
        _, source_frequency_mode = early.source_raw(
            exact, ROW, source_frequency_chart)
        native_frequency_mode = early.native_raw(
            ham, flux, native_frequency_chart)

    omega_difference = native_chart_omega - source_chart_omega
    orbital_phase_difference = native_chart[:, 1] - source_chart[:, 1]
    raw_phase_difference = np.unwrap(np.angle(native_on_native/source_on_source))
    raw_phase_residual_after_orbit = raw_phase_difference + 2*orbital_phase_difference
    same_state_phase_difference = np.unwrap(
        np.angle(native_on_source/source_on_source))
    v5_replay = {
        "source_initial_state_max_abs": float(np.max(abs(
            source_state[0]-np.asarray(previous["source_initial_state"])))),
        "native_initial_state_max_abs": float(np.max(abs(
            early.as_numpy(native_initial)
            -np.asarray(previous["native_warm_initial_state"])))),
        "source_frequency_state_max_abs": float(np.max(abs(
            source_frequency_chart
            -np.asarray(previous["chart"]["source_state_phase_zero"])))),
        "native_frequency_state_max_abs": float(np.max(abs(
            native_frequency_chart
            -np.asarray(previous["chart"]["warm_state_phase_zero"])))),
        "source_frequency_mode_max_abs": float(np.max(abs(
            source_frequency_mode-paired(previous["chart"]["source_mode"])))),
        "native_frequency_mode_max_abs": float(np.max(abs(
            native_frequency_mode-paired(previous["chart"]["warm_mode"])))),
        "source_crop_time_abs_M": abs(source_crop_t-previous["source_crop_time_M"]),
        "native_crop_time_abs_M": abs(native_crop_t-previous["warm_crop_time_M"]),
    }

    # This source low-dynamics/full-mode epoch mapping is a conditional control:
    # the prior full-mode reconstruction error is 0.00449, not an exact replay.
    source_epoch_s = full["source_full_control"]["already_cropped_mode_epoch_s"]
    mass_seconds = (m1+m2)*lal.MTSUN_SI
    source_peak_low_M = (float(source_time[-1]) + source_epoch_s/mass_seconds
                         + full["fixed_chart"]["source_anchor"]["index"]
                         * plan["step_M"])
    native_peak_M = full["levels"]["refined"]["native_peak_time_M"]
    relative_time = np.asarray(phase["measurements"]["early_pre_match_minus_50M"]
                               ["time_M"] if "time_M" in phase["measurements"]
                               ["early_pre_match_minus_50M"] else [], dtype=np.float64)
    if not len(relative_time):
        full_chart_start = full["fixed_chart"]["support_M"][0]
        relative_time = full_chart_start + np.arange(
            plan["chart_samples"], dtype=np.float64)*plan["step_M"]
    source_projected_t = source_peak_low_M + relative_time
    native_projected_t = native_peak_M + relative_time
    source_projected_state = at_times(source_time, source_state, source_projected_t)
    native_projected_state = at_times(native_time, native_state, native_projected_t)
    source_projected_omega = np.interp(source_projected_t, source_time, source_omega)
    native_projected_omega = np.interp(native_projected_t, native_time, native_omega)
    projected_delta_phi = native_projected_state[:, 1] - source_projected_state[:, 1]
    projected_delta_omega = native_projected_omega - source_projected_omega

    checks = {
        "source_finite": bool(np.isfinite(source_row).all()),
        "native_finite": bool(np.isfinite(native_state).all()),
        "frequency_finite_positive": bool(np.isfinite(source_omega).all()
            and np.isfinite(native_omega).all()
            and np.min(source_omega) > 0 and np.min(native_omega) > 0),
        "raw_modes_finite_nonzero": bool(all(np.isfinite(x).all() and np.min(abs(x)) > 0
            for x in (source_on_source, native_on_source, native_on_native))),
        "source_cadence": bool(np.max(abs(np.diff(source_time)-plan["step_M"]))
            <= plan["gates"]["cadence_max_abs_M"]),
        "native_cadence": bool(np.max(abs(np.diff(native_time)-plan["step_M"]))
            <= plan["gates"]["cadence_max_abs_M"]),
        "source_chart_exact": bool(len(source_chart) == plan["chart_samples"]
            and np.max(abs(source_chart_time-chart_time))
            <= plan["gates"]["time_max_abs_M"]),
        "native_chart_exact": bool(len(native_chart) == plan["chart_samples"]
            and np.max(abs(native_chart_time-chart_time))
            <= plan["gates"]["time_max_abs_M"]),
        "cap_phase_zero": bool(source_state[0, 1] == 0.
            and float(native_initial[1]) == 0.),
        "v5_initial_replay": bool(max(v5_replay["source_initial_state_max_abs"],
            v5_replay["native_initial_state_max_abs"])
            <= plan["gates"]["initial_replay_max_abs"]),
        "v5_frequency_chart_replay": bool(max(v for k, v in v5_replay.items()
            if "initial" not in k) <= plan["gates"]["v5_replay_max_abs"]),
        "same_source_state_raw_control": bool(relative_l2(
            native_on_source, source_on_source)
            <= plan["gates"]["same_state_raw_l2_max"]),
        "peak_anchor_replay": bool(full["fixed_chart"]["source_anchor"]
            == plan["source_peak_anchor"] and
            full["levels"]["refined"]["native_peak_anchor"]
            == plan["native_peak_anchor"] and
            full["fixed_chart"]["time_sha256"] == plan["full_chart_sha256"]),
        "projection_inside_low_dynamics": bool(
            source_projected_t[0] >= source_time[0]
            and source_projected_t[-1] <= source_time[-1]
            and native_projected_t[0] >= native_time[0]
            and native_projected_t[-1] <= native_time[-1]),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    result.update(
        status="diagnostic_complete" if all(checks.values()) else "failed_validity",
        elapsed_s=time.monotonic()-started, validity_checks=checks,
        validity_passed=all(checks.values()), v5_replay=v5_replay,
        primary_chart={
            "definition": "Same physical time from independently initialized cap states; no interpolation, shift, or fit.",
            "time_M": chart_time.tolist(), "samples": len(chart_time),
            "source_state": source_chart.tolist(), "native_state": native_chart.tolist(),
            "source_omega_M": source_chart_omega.tolist(),
            "native_omega_M": native_chart_omega.tolist(),
            "delta_orbital_phase_rad": orbital_phase_difference.tolist(),
            "delta_orbital_omega_M": omega_difference.tolist(),
            "source_raw_on_source": pairs(source_on_source),
            "native_raw_on_source": pairs(native_on_source),
            "native_raw_on_native": pairs(native_on_native),
            "raw_phase_difference_rad": raw_phase_difference.tolist(),
            "raw_phase_residual_after_minus2_orbit_rad":
                raw_phase_residual_after_orbit.tolist(),
        },
        primary_metrics={
            "delta_phi_first_last_rad": [float(orbital_phase_difference[0]),
                                         float(orbital_phase_difference[-1])],
            "delta_phi_min_max_rad": [float(np.min(orbital_phase_difference)),
                                       float(np.max(orbital_phase_difference))],
            "delta_omega_first_last_M": [float(omega_difference[0]),
                                          float(omega_difference[-1])],
            "delta_omega_max_abs_M": float(np.max(abs(omega_difference))),
            "same_state_raw_complex_l2": relative_l2(native_on_source, source_on_source),
            "trajectory_raw_complex_l2": relative_l2(native_on_native, native_on_source),
            "source_native_raw_complex_l2": relative_l2(native_on_native, source_on_source),
            "raw_phase_first_last_rad": [float(raw_phase_difference[0]),
                                         float(raw_phase_difference[-1])],
            "same_state_raw_phase_max_abs_rad": float(np.max(abs(same_state_phase_difference))),
            "raw_phase_residual_after_minus2_orbit_max_abs_rad":
                float(np.max(abs(raw_phase_residual_after_orbit))),
            "source_native_raw_amplitude_l2": relative_l2(
                abs(native_on_native), abs(source_on_source)),
        },
        conditional_peak_projection={
            "source_low_to_full_epoch_assumption":
                "Source low endpoint plus full already-cropped mode epoch; prior source raw+NQC/full reconstruction was not exact.",
            "prior_full_reconstruction_l2": plan["prior_full_reconstruction_l2"],
            "source_low_peak_time_M": source_peak_low_M,
            "native_peak_time_M": native_peak_M,
            "relative_time_M": relative_time.tolist(),
            "source_low_time_M": source_projected_t.tolist(),
            "native_low_time_M": native_projected_t.tolist(),
            "delta_orbital_phase_rad": projected_delta_phi.tolist(),
            "delta_orbital_omega_M": projected_delta_omega.tolist(),
            "source_full_phase_anchor_rad": full["fixed_chart"]["source_anchor"]["phase_rad"],
            "native_full_phase_anchor_rad": full["levels"]["refined"]
                ["native_peak_anchor"]["phase_rad"],
        },
        interpretation=(
            "Development-only source-exposed cap-time trajectory and raw-mode "
            "comparison. Same-state raw control distinguishes mode evaluation from "
            "trajectory input on this early chart. The independent full-mode peak "
            "projection is conditional on a source low/full epoch mapping that did "
            "not exactly reconstruct the source full mode; no claim of a direct "
            "whole-waveform attribution or release qualification."
        ),
    )
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "validity": checks,
                      "primary_metrics": result["primary_metrics"]}), flush=True)


if __name__ == "__main__":
    main()

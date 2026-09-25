"""Dev-only native cap warm-start intervention for one strong-aligned row."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import diagnose_source_aligned_early_raw_crossover_v3 as prior_runner
from dev.native_rom import diagnose_source_aligned_early_raw_restart_v4 as restart_runner
from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_wheel_dynamics import _source as source_dynamics


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_early_raw_warm_start_v5_plan.json"
OUTPUT = EVIDENCE / "source_aligned_early_raw_warm_start_v5.json"
ROW = (30.0, 20.0, 0.8, 0.6, 0.04)


def complex_from_pairs(values):
    pairs = np.asarray(values, dtype=np.float64)
    return pairs[:, 0] + 1j * pairs[:, 1]


def crossing_state(omega, time, state, target):
    indices = np.flatnonzero((omega[:-1] <= target) & (omega[1:] >= target))
    if len(indices) != 1:
        raise ValueError("low-dynamics crop has no unique upward orbital-frequency crossing")
    index = int(indices[0])
    crossing_time = float(np.interp(target, omega[index:index + 2],
                                    time[index:index + 2]))
    crossing_state_value = np.array([
        np.interp(crossing_time, time, state[:, component])
        for component in range(4)], dtype=np.float64)
    return crossing_time, crossing_state_value, index


def main():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-early-raw-warm-start-plan.v5"
            or plan["release_qualified"] is not False
            or tuple(plan["row_tuple"]) != ROW
            or plan["step_M"] != 0.5
            or plan["cap_Momega"] != 10**(-1.5)
            or plan["script_sha256"] != q.sha256(Path(__file__))):
        raise ValueError("frozen warm-start plan changed")
    for name, item in plan["inputs"].items():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen input changed: {name}")
    prior = json.loads((ROOT / plan["inputs"]["restart_result"]["path"]).read_text())
    if (prior["status"] != "diagnostic_complete"
            or not prior["validity_passed"] or not prior["hypothesis_passed"]
            or prior["plan_sha256"] != plan["restart_plan_sha256"]
            or prior["script_sha256"] != plan["restart_script_sha256"]):
        raise ValueError("frozen restart control identity changed")
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    level = next(item for item in ladder["levels"] if item["name"] == "refined")
    target = np.asarray(plan["fixed_orbital_frequency_Momega"], dtype=np.float64)
    libraries = _libraries(source_plan["pinned_wheel"])
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.source-aligned-early-raw-warm-start.v5",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.set_num_threads(1)
    m1, m2, chi1, chi2, requested_omega = ROW

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    ham = driver.TorchSourceHam(tensor(m1*m2/(m1+m2)**2), tensor(chi1), tensor(chi2),
                                delta=tensor((m1-m2)/(m1+m2)))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    field = driver.TorchIntegrator(ham, flux).vector_field
    cap = plan["cap_Momega"]
    r0, pr0, pphi0 = driver.find_quasicircular_initial_conditions(ham, flux, tensor(cap))
    warm_initial = torch.stack((r0, torch.zeros_like(r0), pr0, pphi0)).detach()
    warm_time, warm_state = restart_runner.run_coarse(
        field, warm_initial, level, ladder, plan["step_M"])
    with tempfile.TemporaryDirectory(prefix="diffgw-v5-native-warm-start-") as tmp:
        private = Path(tmp) / "source"
        _snapshot(prior_runner.SOURCE_ROOT, source_plan["pinned_source"], private,
                  prior_runner.HEADERS)

        def build(name, key):
            path = Path(tmp) / name
            _compile(private, prior_runner.HEADERS, prior_runner.GSL_INCLUDE,
                     libraries, ROOT / plan["inputs"][key]["path"], path,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            return ctypes.CDLL(str(path))

        dyn_lib = build("dynamics.dylib", "dynamics_adapter")
        dyn = dyn_lib.diffgw_v4_full_source_dynamics
        dyn.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
                        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double)]
        dyn.restype = ctypes.c_int
        row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
                   omega_start=requested_omega)
        source_row = source_dynamics(dyn, row, plan["step_M"], 50000)
        source_time, source_state = source_row[0], source_row[1:].T
        exact_lib = build("exact.dylib", "exact_adapter")
        exact = exact_lib.diffgw_v4_source_nqc_state
        exact.argtypes = [ctypes.c_double]*8 + [ctypes.POINTER(ctypes.c_double)]
        exact.restype = ctypes.c_int
        source_omega, _ = prior_runner.source_raw(exact, ROW, source_state)
        warm_omega, _ = prior_runner.source_raw(exact, ROW, warm_state)
        source_crop_time, source_crop_state, source_crop_index = crossing_state(
            source_omega, source_time, source_state, requested_omega)
        warm_crop_time, warm_crop_state, warm_crop_index = crossing_state(
            warm_omega, warm_time, warm_state, requested_omega)
        source_chart, source_chart_time, source_early_count = (
            prior_runner.frequency_states(source_omega, source_time, source_state, target))
        warm_chart, warm_chart_time, warm_early_count = (
            prior_runner.frequency_states(warm_omega, warm_time, warm_state, target))
        source_chart[:, 1] -= source_crop_state[1]
        warm_chart[:, 1] -= warm_crop_state[1]
        source_verified_omega, source_mode = prior_runner.source_raw(exact, ROW,
                                                                      source_chart)
        warm_verified_omega, _ = prior_runner.source_raw(exact, ROW, warm_chart)
        warm_mode = prior_runner.native_raw(ham, flux, warm_chart)
        native_on_source = prior_runner.native_raw(ham, flux, source_chart)
        cap_omega, _ = prior_runner.source_raw(exact, ROW, np.array([
            prior_runner.as_numpy(warm_initial)]))
        source_crop_omega, _ = prior_runner.source_raw(
            exact, ROW, source_crop_state[None, :])
        warm_crop_omega, _ = prior_runner.source_raw(
            exact, ROW, warm_crop_state[None, :])

    prior_chart = prior["chart"]
    prior_target = np.asarray(prior_chart["Momega"], dtype=np.float64)
    prior_source_chart = np.asarray(prior_chart["source_state_phase_zero"], dtype=np.float64)
    prior_source_mode = complex_from_pairs(prior_chart["source_mode"])
    direct_mode = complex_from_pairs(prior_chart["direct_mode"])
    restart_mode = complex_from_pairs(prior_chart["restart_mode"])
    rel_l2 = restart_runner.relative_l2
    anchor = prior_runner.normalized
    anchored_l2 = {
        "direct_vs_source_control": rel_l2(anchor(direct_mode), anchor(source_mode)),
        "restart_vs_source_control": rel_l2(anchor(restart_mode), anchor(source_mode)),
        "warm_vs_source": rel_l2(anchor(warm_mode), anchor(source_mode)),
        "same_source_state_native_vs_source": rel_l2(
            anchor(native_on_source), anchor(source_mode)),
    }
    amplitude_l2 = {
        "direct_vs_source_control": rel_l2(abs(direct_mode), abs(source_mode)),
        "restart_vs_source_control": rel_l2(abs(restart_mode), abs(source_mode)),
        "warm_vs_source": rel_l2(abs(warm_mode), abs(source_mode)),
        "same_source_state_native_vs_source": rel_l2(
            abs(native_on_source), abs(source_mode)),
    }
    replay = {
        "target_max_abs": float(np.max(abs(target-prior_target))),
        "source_chart_state_max_abs": float(np.max(abs(source_chart-prior_source_chart))),
        "source_mode_max_abs": float(np.max(abs(source_mode-prior_source_mode))),
        "source_crop_time_max_abs": abs(source_crop_time-prior["source_start_time_M"]),
        "source_crop_state_max_abs": float(np.max(abs(
            np.array([source_crop_state[0], 0., *source_crop_state[2:]])
            -np.asarray(prior["source_start_state_phase_zero"])))),
    }
    chart_omega_errors = {
        "source": float(np.max(abs(source_verified_omega-target))),
        "warm": float(np.max(abs(warm_verified_omega-target))),
    }
    validity = {
        "finite": all(np.isfinite(array).all() for array in (
            source_row, warm_state, source_chart, warm_chart, source_mode,
            warm_mode, native_on_source)),
        "source_cadence": np.max(abs(np.diff(source_time)-plan["step_M"])) < 1e-10,
        "warm_cadence": np.max(abs(np.diff(warm_time)-plan["step_M"])) < 1e-10,
        "source_start_near_cap": abs(source_omega[0]-cap)
            <= plan["gates"]["source_initial_cap_max_abs"],
        "warm_initial_cap": abs(cap_omega[0]-cap)
            <= plan["gates"]["warm_initial_cap_max_abs"],
        "source_crop_omega": abs(source_crop_omega[0]-requested_omega)
            <= plan["gates"]["crop_omega_max_abs"],
        "warm_crop_omega": abs(warm_crop_omega[0]-requested_omega)
            <= plan["gates"]["crop_omega_max_abs"],
        "chart_omega": max(chart_omega_errors.values())
            <= plan["gates"]["chart_omega_max_abs"],
        "source_control_replay": max(replay.values())
            <= plan["gates"]["source_replay_max_abs"],
        "control_complex_metrics_replay": max(
            abs(anchored_l2["direct_vs_source_control"]
                -prior["anchored_complex_l2"]["direct_vs_source"]),
            abs(anchored_l2["restart_vs_source_control"]
                -prior["anchored_complex_l2"]["restart_vs_source"]),
        ) <= plan["gates"]["control_metric_replay_max_abs"],
        "early_margin": (source_chart_time[-1]
            < source_time[-1]-plan["gates"]["terminal_margin_M"]
            and warm_chart_time[-1]
            < warm_time[-1]-plan["gates"]["terminal_margin_M"]),
    }
    validity = {name: bool(passed) for name, passed in validity.items()}
    hypothesis = {
        "warm_anchored_raw_l2_at_most": anchored_l2["warm_vs_source"]
            <= plan["gates"]["warm_anchored_raw_l2_at_most"],
    }
    hypothesis = {name: bool(passed) for name, passed in hypothesis.items()}
    result.update(
        status=("diagnostic_complete" if all(validity.values()) and all(hypothesis.values())
                else "failed_hypothesis" if all(validity.values()) else "failed_validity"),
        elapsed_s=time.monotonic()-started,
        validity_checks=validity, validity_passed=all(validity.values()),
        hypothesis_checks=hypothesis, hypothesis_passed=all(hypothesis.values()),
        cap_Momega=cap, source_initial_omega_M=float(source_omega[0]),
        native_warm_initial_omega_M=float(cap_omega[0]),
        source_initial_state=source_state[0].tolist(),
        native_warm_initial_state=prior_runner.as_numpy(warm_initial).tolist(),
        source_crop_time_M=source_crop_time, warm_crop_time_M=warm_crop_time,
        source_crop_index=source_crop_index, warm_crop_index=warm_crop_index,
        source_crop_state=source_crop_state.tolist(),
        warm_crop_state=warm_crop_state.tolist(),
        crop_state_difference=(warm_crop_state-source_crop_state).tolist(),
        source_crop_omega_M=float(source_crop_omega[0]),
        warm_crop_omega_M=float(warm_crop_omega[0]),
        source_samples=len(source_time), warm_samples=len(warm_time),
        source_early_count=source_early_count, warm_early_count=warm_early_count,
        source_control_replay=replay, chart_omega_max_abs=chart_omega_errors,
        anchored_complex_l2=anchored_l2, amplitude_l2=amplitude_l2,
        chart={
            "Momega": target.tolist(),
            "source_time_M": source_chart_time.tolist(),
            "warm_time_M": warm_chart_time.tolist(),
            "source_state_phase_zero": source_chart.tolist(),
            "warm_state_phase_zero": warm_chart.tolist(),
            "source_mode": np.column_stack((source_mode.real, source_mode.imag)).tolist(),
            "warm_mode": np.column_stack((warm_mode.real, warm_mode.imag)).tolist(),
        },
        interpretation=(
            "Native warm-start trajectory and low-dynamics orbital-frequency crop "
            "only. Full source h22 uses a separate instantaneous-frequency crop; "
            "this result does not attribute the whole-waveform residual or qualify release."
        ),
    )
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "validity": validity,
                      "hypothesis": hypothesis, "anchored_complex_l2": anchored_l2,
                      "amplitude_l2": amplitude_l2}), flush=True)


if __name__ == "__main__":
    main()

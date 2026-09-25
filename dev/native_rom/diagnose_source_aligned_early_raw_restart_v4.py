"""Dev-only source-state restart intervention for the exact strong-aligned row."""

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
from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_wheel_dynamics import _source as source_dynamics


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_early_raw_restart_v4_plan.json"
OUTPUT = EVIDENCE / "source_aligned_early_raw_restart_v4.json"
ROW = (30.0, 20.0, 0.8, 0.6, 0.04)


def relative_l2(candidate, reference):
    return float(np.linalg.norm(candidate-reference)/np.linalg.norm(reference))


def run_coarse(field, initial, level, ladder, step):
    coarse = driver.integrate_adaptive(
        field, initial, step, t_end=1000.0,
        event=driver.AlignedSpinV4Event("coarse"), initial_step=step,
        max_step=ladder["max_step_M"], controller="gsl_state_scaled",
        atol=level["atol"], rtol=level["rtol"])
    return (prior_runner.as_numpy(coarse.output_time),
            prior_runner.as_numpy(coarse.output_state))


def main():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-early-raw-restart-plan.v4"
            or plan["release_qualified"] is not False
            or tuple(plan["row_tuple"]) != ROW
            or plan["step_M"] != 0.5
            or plan["script_sha256"] != q.sha256(Path(__file__))):
        raise ValueError("frozen restart plan changed")
    for name, item in plan["inputs"].items():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen input changed: {name}")
    prior = json.loads((ROOT / plan["inputs"]["prior_result"]["path"]).read_text())
    if (prior["status"] != "diagnostic_complete"
            or prior["plan_sha256"] != plan["prior_plan_sha256"]
            or prior["script_sha256"] != plan["prior_script_sha256"]):
        raise ValueError("prior raw crossover result does not replay frozen identity")
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    level = next(item for item in ladder["levels"] if item["name"] == "refined")
    target = np.asarray(plan["fixed_orbital_frequency_Momega"], dtype=np.float64)
    libraries = _libraries(source_plan["pinned_wheel"])
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.source-aligned-early-raw-restart.v4",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    torch.set_num_threads(1)
    m1, m2, chi1, chi2, omega_start = ROW

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    ham = driver.TorchSourceHam(tensor(m1*m2/(m1+m2)**2), tensor(chi1), tensor(chi2),
                                delta=tensor((m1-m2)/(m1+m2)))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    field = driver.TorchIntegrator(ham, flux).vector_field
    r0, pr0, pphi0 = driver.find_quasicircular_initial_conditions(
        ham, flux, tensor(omega_start))
    direct_initial = torch.stack((r0, torch.zeros_like(r0), pr0, pphi0)).detach()
    direct_time, direct_state = run_coarse(
        field, direct_initial, level, ladder, plan["step_M"])
    with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-restart-") as tmp:
        private = Path(tmp)/"source"
        _snapshot(prior_runner.SOURCE_ROOT, source_plan["pinned_source"], private,
                  prior_runner.HEADERS)

        def build(name, key):
            path = Path(tmp)/name
            _compile(private, prior_runner.HEADERS, prior_runner.GSL_INCLUDE,
                     libraries, ROOT/plan["inputs"][key]["path"], path,
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
                   omega_start=omega_start)
        source_row = source_dynamics(dyn, row, plan["step_M"], 50000)
        source_time, source_state = source_row[0], source_row[1:].T
        exact_lib = build("exact.dylib", "exact_adapter")
        exact = exact_lib.diffgw_v4_source_nqc_state
        exact.argtypes = [ctypes.c_double]*8 + [ctypes.POINTER(ctypes.c_double)]
        exact.restype = ctypes.c_int
        source_omega, _ = prior_runner.source_raw(exact, ROW, source_state)
        crossings = np.flatnonzero((source_omega[:-1] <= omega_start)
                                   & (source_omega[1:] >= omega_start))
        if len(crossings) != 1:
            raise ValueError("source omega_start crossing is not unique")
        crossing = int(crossings[0])
        source_start_time = float(np.interp(
            omega_start, source_omega[crossing:crossing+2],
            source_time[crossing:crossing+2]))
        source_initial = np.array([
            np.interp(source_start_time, source_time, source_state[:, i])
            for i in range(4)], dtype=np.float64)
        source_phase_at_start = float(source_initial[1])
        source_initial[1] = 0.0
        source_start_omega = float(prior_runner.source_raw(
            exact, ROW, source_initial[None, :])[0][0])
        restart_time, restart_state = run_coarse(
            field, tensor(source_initial), level, ladder, plan["step_M"])
        field_lib = build("field.dylib", "field_adapter")
        source_field = field_lib.diffgw_v4_source_field
        source_field.argtypes = [ctypes.c_double]*8 + [ctypes.POINTER(ctypes.c_double)]
        source_field.restype = ctypes.c_int
        source_field_out = (ctypes.c_double*6)()
        if source_field(*ROW[:4], *source_initial, source_field_out):
            raise RuntimeError("pinned source field failed at restart state")
        pinned_field = np.asarray(source_field_out[:4], dtype=np.float64)
        native_field = prior_runner.as_numpy(field(tensor(source_initial)))
        source_chart, source_chart_time, source_early_count = (
            prior_runner.frequency_states(source_omega, source_time,
                                          source_state, target))
        source_chart[:, 1] -= source_phase_at_start
        direct_omega, _ = prior_runner.source_raw(exact, ROW, direct_state)
        direct_chart, direct_chart_time, direct_early_count = (
            prior_runner.frequency_states(direct_omega, direct_time,
                                          direct_state, target))
        restart_omega, _ = prior_runner.source_raw(exact, ROW, restart_state)
        restart_chart, restart_chart_time, restart_early_count = (
            prior_runner.frequency_states(restart_omega, restart_time,
                                          restart_state, target))
        verified_source_omega, source_mode = prior_runner.source_raw(
            exact, ROW, source_chart)
        verified_direct_omega, _ = prior_runner.source_raw(exact, ROW, direct_chart)
        verified_restart_omega, _ = prior_runner.source_raw(
            exact, ROW, restart_chart)
        direct_mode = prior_runner.native_raw(ham, flux, direct_chart)
        restart_mode = prior_runner.native_raw(ham, flux, restart_chart)
        native_on_source = prior_runner.native_raw(ham, flux, source_chart)
    prior_direct_state = np.asarray(prior["chart"]["native_state"])
    prior_direct_mode = np.asarray(prior["modes"]["native_on_native"])
    prior_direct_mode = prior_direct_mode[:, 0]+1j*prior_direct_mode[:, 1]
    chart_omega_errors = {
        "source": float(np.max(abs(verified_source_omega-target))),
        "direct": float(np.max(abs(verified_direct_omega-target))),
        "restart": float(np.max(abs(verified_restart_omega-target))),
    }
    fixed_anchor = prior_runner.normalized
    anchored_l2 = {
        "direct_vs_source": relative_l2(fixed_anchor(direct_mode),
                                        fixed_anchor(source_mode)),
        "restart_vs_source": relative_l2(fixed_anchor(restart_mode),
                                         fixed_anchor(source_mode)),
        "same_source_state_native_vs_source": relative_l2(
            fixed_anchor(native_on_source), fixed_anchor(source_mode)),
    }
    amplitude_l2 = {
        "direct_vs_source": relative_l2(abs(direct_mode), abs(source_mode)),
        "restart_vs_source": relative_l2(abs(restart_mode), abs(source_mode)),
        "same_source_state_native_vs_source": relative_l2(
            abs(native_on_source), abs(source_mode)),
    }
    replay_state_max_abs = float(np.max(abs(direct_chart-prior_direct_state)))
    replay_mode_max_abs = float(np.max(abs(direct_mode-prior_direct_mode)))
    validity = {
        "finite": all(np.isfinite(array).all() for array in (
            source_row, direct_state, restart_state, source_chart, direct_chart,
            restart_chart, source_mode, direct_mode, restart_mode,
            pinned_field, native_field)),
        "source_initial_phase_zero": source_initial[1] == 0.,
        "source_start_omega": abs(source_start_omega-omega_start)
            <= plan["gates"]["source_start_omega_max_abs"],
        "source_cadence": np.max(abs(np.diff(source_time)-plan["step_M"])) < 1e-10,
        "direct_cadence": np.max(abs(np.diff(direct_time)-plan["step_M"])) < 1e-10,
        "restart_cadence": np.max(abs(np.diff(restart_time)-plan["step_M"])) < 1e-10,
        "chart_omega": max(chart_omega_errors.values())
            <= plan["gates"]["chart_omega_max_abs"],
        "direct_chart_replay_state": replay_state_max_abs
            <= plan["gates"]["direct_replay_max_abs"],
        "direct_chart_replay_mode": replay_mode_max_abs
            <= plan["gates"]["direct_replay_max_abs"],
        "early_margin": (source_chart_time[-1]
            < source_time[-1]-plan["gates"]["terminal_margin_M"]
            and direct_chart_time[-1]
            < direct_time[-1]-plan["gates"]["terminal_margin_M"]
            and restart_chart_time[-1]
            < restart_time[-1]-plan["gates"]["terminal_margin_M"]),
    }
    validity = {name: bool(passed) for name, passed in validity.items()}
    hypothesis = {
        "restart_anchored_raw_l2_at_most": anchored_l2["restart_vs_source"]
            <= plan["gates"]["restart_anchored_raw_l2_at_most"],
    }
    hypothesis = {name: bool(passed) for name, passed in hypothesis.items()}
    result.update(
        status=("diagnostic_complete" if all(validity.values()) and all(hypothesis.values())
                else "failed_hypothesis" if all(validity.values()) else "failed_validity"),
        validity_checks=validity, validity_passed=all(validity.values()),
        hypothesis_checks=hypothesis, hypothesis_passed=all(hypothesis.values()),
        elapsed_s=time.monotonic()-started,
        source_start_time_M=source_start_time,
        source_start_omega_M=source_start_omega,
        source_phase_at_start_rad=source_phase_at_start,
        source_start_state_phase_zero=source_initial.tolist(),
        direct_initial_state=prior_runner.as_numpy(direct_initial).tolist(),
        initial_state_difference=(source_initial-prior_runner.as_numpy(direct_initial)).tolist(),
        source_field_at_restart=pinned_field.tolist(),
        native_field_at_restart=native_field.tolist(),
        field_difference=(native_field-pinned_field).tolist(),
        field_relative_l2=relative_l2(native_field, pinned_field),
        source_samples=int(len(source_time)), direct_samples=int(len(direct_time)),
        restart_samples=int(len(restart_time)),
        source_early_count=source_early_count, direct_early_count=direct_early_count,
        restart_early_count=restart_early_count,
        chart_omega_max_abs=chart_omega_errors,
        direct_replay_state_max_abs=replay_state_max_abs,
        direct_replay_mode_max_abs=replay_mode_max_abs,
        anchored_complex_l2=anchored_l2, amplitude_l2=amplitude_l2,
        chart={
            "Momega": target.tolist(),
            "source_time_M": source_chart_time.tolist(),
            "direct_time_M": direct_chart_time.tolist(),
            "restart_time_M": restart_chart_time.tolist(),
            "source_state_phase_zero": source_chart.tolist(),
            "direct_state": direct_chart.tolist(),
            "restart_state": restart_chart.tolist(),
            "source_mode": np.column_stack((source_mode.real, source_mode.imag)).tolist(),
            "direct_mode": np.column_stack((direct_mode.real, direct_mode.imag)).tolist(),
            "restart_mode": np.column_stack((restart_mode.real, restart_mode.imag)).tolist(),
        },
        interpretation="Early raw-stage restart intervention only. Source low dynamics are distinct from the cropped full source mode; no whole-waveform attribution or release qualification.",
    )
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "validity": validity,
                      "hypothesis": hypothesis, "anchored_complex_l2": anchored_l2,
                      "amplitude_l2": amplitude_l2}), flush=True)


if __name__ == "__main__":
    main()

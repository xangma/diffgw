"""Dev-only fixed-warm-state Torch/JAX trajectory localization."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom.eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from dev.native_rom.eob.jax.adaptive_driver_clean import fixed_fehlberg45_step as jax_fixed
from torchwave.approximants.eob.adaptive_driver_clean import (
    AlignedSpinV4Event, fixed_fehlberg45_step as torch_fixed, integrate_adaptive,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "warm_jax_trajectory_divergence_plan.json"
OUTPUT = EVIDENCE / "warm_jax_trajectory_divergence.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def relative_l2(left, right):
    return float(np.linalg.norm(left - right) / np.linalg.norm(right))


def component_comparison(left, right):
    return {
        "max_abs_by_component": np.max(np.abs(left - right), axis=0).tolist(),
        "relative_l2_by_component": [relative_l2(left[:, i], right[:, i])
                                      for i in range(right.shape[1])],
    }


def first_threshold(left, right, threshold):
    delta = np.max(np.abs(left - right), axis=1)
    crossed = np.flatnonzero(delta > threshold)
    return int(crossed[0]) if len(crossed) else None


def main():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.warm-jax-trajectory-divergence-plan.v1"
            or plan["release_qualified"] is not False
            or sha(__file__) != plan["script_sha256"]):
        raise ValueError("frozen trajectory diagnostic changed")
    for item in plan["inputs"].values():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("frozen trajectory input changed")
    prior = json.loads((ROOT / plan["inputs"]["warm_whole_v3"]["path"]).read_text())
    if prior["status"] != "diagnostic_overlap_only":
        raise ValueError("warm whole-waveform control missing")
    m1, m2, chi1, chi2, omega = prior["row_tuple"]
    nu = m1*m2/(m1+m2)**2
    delta = (m1-m2)/(m1+m2)
    state = np.asarray(prior["field_parity"]["warm"]["state"], dtype=np.float64)
    torch.set_num_threads(1)
    th = driver.TorchSourceHam(torch.tensor(nu, dtype=torch.float64),
                               torch.tensor(chi1, dtype=torch.float64),
                               torch.tensor(chi2, dtype=torch.float64),
                               delta=torch.tensor(delta, dtype=torch.float64))
    tf = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    jh = driver.JaxSourceHam(nu, chi1, chi2, delta=delta)
    jf = driver.JaxFlux(**driver.SOURCE_FLUX_OPTIONS)
    torch_field = driver.TorchIntegrator(th, tf).vector_field
    jax_field = jax.jit(driver.JaxIntegrator(jh, jf).vector_field)
    tstate = torch.as_tensor(state)
    jstate = jnp.asarray(state)
    result = {
        "schema": "diffgw.seobnrv4.warm-jax-trajectory-divergence.v1",
        "release_qualified": False, "status": "running", "host": platform.node(),
        "cwd": os.getcwd(), "pid": os.getpid(), "plan_sha256": sha(PLAN),
        "script_sha256": sha(__file__), "row_tuple": prior["row_tuple"],
        "warm_state": state.tolist(), "stages": {}, "checks": {},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    td = arr(torch_field(tstate))
    jd = arr(jax_field(jstate))
    step = plan["fixed_step_M"]
    tt = torch_fixed(torch_field, 0., tstate, step, atol=plan["atol"],
                     rtol=plan["rtol"], current_derivative=torch.as_tensor(td))
    jt = jax_fixed(lambda _time, value: jax_field(value), 0., jstate, step,
                   atol=plan["atol"], rtol=plan["rtol"],
                   current_derivative=jnp.asarray(jd))
    result["stages"]["fixed_first_trial"] = {
        "field_max_abs": float(np.max(abs(td-jd))),
        "fourth_max_abs": float(np.max(abs(arr(tt.fourth)-arr(jt.fourth)))),
        "fifth_max_abs": float(np.max(abs(arr(tt.fifth)-arr(jt.fifth)))),
        "error_max_abs": float(np.max(abs(arr(tt.error)-arr(jt.error)))),
        "error_ratio_abs": abs(float(tt.error_ratio)-float(jt.error_ratio)),
        "torch_error_ratio": float(tt.error_ratio), "jax_error_ratio": float(jt.error_ratio),
    }
    settings = plan["settings"]
    dt = settings["coarse_output_step_M"]
    max_step = settings["max_step_M"]
    atol, rtol = settings["atol"], settings["rtol"]
    coarse = integrate_adaptive(
        torch_field, tstate, dt, t_end=settings["max_duration_M"],
        event=AlignedSpinV4Event("coarse"), initial_step=dt, max_step=max_step,
        controller="gsl_state_scaled", atol=atol, rtol=rtol)
    print("Torch coarse done", flush=True)
    jrun = integrate_two_stage_trajectory(
        jax_field, jstate, output_step=dt,
        max_duration=settings["max_duration_M"], coarse_initial_step=dt,
        coarse_max_step=max_step, fine_max_step=max_step,
        fine_refinement_factor=settings["fine_refinement_factor"],
        atol=atol, rtol=rtol)
    print("JAX coarse/fine done", flush=True)
    tc_time, tc_state = arr(coarse.output_time), arr(coarse.output_state)
    jc_time, jc_state = arr(jrun.coarse_time), arr(jrun.coarse_state)
    n = min(len(tc_time), len(jc_time))
    if n < 10 or float(np.max(abs(tc_time[:n]-jc_time[:n]))) > 1e-12:
        raise ValueError("coarse output charts do not match")
    coarse_records = {}
    for window in plan["coarse_windows_M"]:
        count = min(n, int(math.floor(window/dt))+1)
        coarse_records[str(window)] = component_comparison(jc_state[:count], tc_state[:count])
    threshold_crossings = {}
    for threshold in plan["state_thresholds"]:
        index = first_threshold(jc_state[:n], tc_state[:n], threshold)
        threshold_crossings[str(threshold)] = None if index is None else {
            "index": index, "time_M": float(tc_time[index]),
            "state_max_abs": float(np.max(abs(jc_state[index]-tc_state[index]))),
        }
    ti, ji = np.asarray(coarse.time), arr(jrun.coarse.time)
    result["stages"]["coarse"] = {
        "torch_accepted_steps": int(coarse.statistics.accepted_steps),
        "jax_accepted_steps": jrun.coarse.accepted_steps,
        "torch_rejected_steps": int(coarse.statistics.rejected_steps),
        "jax_rejected_steps": jrun.coarse.rejected_steps,
        "torch_event_time_M": float(ti[-1]), "jax_event_time_M": float(ji[-1]),
        "torch_output_samples": len(tc_time), "jax_output_samples": len(jc_time),
        "common_output_samples": n,
        "common_output": component_comparison(jc_state[:n], tc_state[:n]),
        "windows": coarse_records, "first_state_threshold_crossing": threshold_crossings,
        "torch_restart_index": len(tc_time)-math.ceil(150/dt),
        "jax_restart_index": jrun.restart_index,
    }
    # On each backend's own fine continuation, separate inherited restart-state
    # drift from fine-stage growth. Both fine grids use the same local clock.
    restart = result["stages"]["coarse"]["torch_restart_index"]
    fine_step = dt/settings["fine_refinement_factor"]
    fine = integrate_adaptive(
        torch_field, coarse.output_state[restart], fine_step,
        t_end=settings["max_duration_M"], event=AlignedSpinV4Event("fine"),
        initial_step=fine_step, max_step=max_step,
        controller="gsl_state_scaled", atol=atol, rtol=rtol)
    ft, fs = arr(fine.output_time), arr(fine.output_state)
    gjt, gjs = arr(jrun.fine_time), arr(jrun.fine_state)
    fn = min(len(ft), len(gjt))
    if fn < 10 or float(np.max(abs(ft[:fn]-gjt[:fn]))) > 1e-12:
        raise ValueError("fine output charts do not match")
    torch_options = driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True}
    tm, _ = driver.torch_mode(torch.as_tensor(fs[:fn]), th, tf, **torch_options)
    jm_same, _ = driver.jax_mode(jnp.asarray(fs[:fn]), jh, jf, **torch_options)
    jm_own, _ = driver.jax_mode(jnp.asarray(gjs[:fn]), jh, jf, **torch_options)
    result["stages"]["fine"] = {
        "torch_output_samples": len(ft), "jax_output_samples": len(gjt),
        "common_output_samples": fn,
        "restart_state_max_abs": float(np.max(abs(gjs[0]-fs[0]))),
        "common_output": component_comparison(gjs[:fn], fs[:fn]),
        "raw_mode_same_state_complex_l2": relative_l2(arr(jm_same), arr(tm)),
        "raw_mode_own_state_complex_l2": relative_l2(arr(jm_own), arr(tm)),
        "first_64_raw_mode_own_state_complex_l2": relative_l2(arr(jm_own)[:64], arr(tm)[:64]),
    }
    result["checks"] = {
        "initial_state_same": float(np.max(abs(state-arr(jstate)))) <= 1e-15,
        "initial_field_same": result["stages"]["fixed_first_trial"]["field_max_abs"] <= 1e-12,
        "first_fixed_step_same": result["stages"]["fixed_first_trial"]["fifth_max_abs"] <= 1e-10,
        "fine_mode_formula_same_on_torch_states":
            result["stages"]["fine"]["raw_mode_same_state_complex_l2"] <= 1e-10,
        "coarse_100M_hypothesis":
            coarse_records["100.0"]["max_abs_by_component"][1] <= 1e-8,
    }
    result["status"] = "diagnostic_complete"
    result["elapsed_s"] = time.monotonic()-started
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

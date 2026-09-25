"""One-row independent-JAX-trajectory parity for the ODE NQC intervention."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_source_ode_boundary_frequency_parity import _jax_ode_rows
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _series_summary
from dev.native_rom.diagnose_source_aligned_ode_chain_boundary_frequency import _capture_level
from dev.native_rom.diagnose_source_ode_chain_boundary_frequency_domain import _source_components
from dev.native_rom.eob.jax.numerical_attachment_clean import _solve_checked, unwrap_mode_phase
from jaxwave.approximants.eob.merger_ringdown import (
    merger_ringdown_22, seobnrv4_merger_shape, seobnrv4_peak_preserving_shape,
)
from jaxwave.approximants.eob.nqc import NQCCoefficients, seobnrv4_nqc_targets
from torchwave.approximants.eob import numerical_attachment as torch_attachment


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "jax_source_ode_boundary_frequency_independent_plan.json"
OUTPUT = EVIDENCE / "jax_source_ode_boundary_frequency_independent.json"


def _checked_plan():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-ode-boundary-frequency-independent-plan.v1"
            or plan["case"] != "strong_aligned" or plan["level"] != "baseline"
            or plan["release_qualified"] is not False
            or plan["trajectory"] != "independent JAX initial solve and two-stage adaptive integration"):
        raise ValueError("independent-JAX plan changed")
    for name, item in plan["inputs"].items():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"pinned input changed: {name}")
    if q.sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("independent-JAX runner changed")
    return plan


def _relative_l2(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def _shared_relative_l2(actual, expected):
    count = min(len(actual), len(expected))
    return _relative_l2(actual[:count], expected[:count])


def _sha_array(values):
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def _hermite_center(time_grid, states, match_time, field):
    time_grid = np.asarray(time_grid)
    index = int(np.searchsorted(time_grid, match_time, side="left")-1)
    index = max(0, min(index, len(time_grid)-2))
    t0, t1 = map(float, time_grid[index:index+2])
    if not t0 <= match_time <= t1:
        raise ValueError("independent JAX match outside Hermite bracket")
    h = t1-t0
    u = (match_time-t0)/h
    y0, y1 = states[index], states[index+1]
    f0, f1 = field(y0), field(y1)
    center = ((2*u**3-3*u**2+1)*y0 + (u**3-2*u**2+u)*h*f0
              +(-2*u**3+3*u**2)*y1 + (u**3-u**2)*h*f1)
    return center, {"index": index, "times_M": [t0, t1], "fraction": u}


def main():
    plan = _checked_plan()
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    boundary_plan = json.loads((ROOT / plan["inputs"]["boundary_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    prior = json.loads((ROOT / plan["inputs"]["boundary_result"]["path"]).read_text())
    fixed_prior = json.loads((ROOT / plan["inputs"]["fixed_input_result"]["path"]).read_text())
    level = next(row for row in ladder["levels"] if row["name"] == plan["level"])
    settings = {"coarse_output_step_M": ladder["coarse_output_step_M"],
                "max_step_M": ladder["max_step_M"], "atol": level["atol"],
                "rtol": level["rtol"]}
    started = time.monotonic()
    report = {
        "schema": "diffgw.seobnrv4.jax-source-ode-boundary-frequency-independent.v1",
        "status": "running", "release_qualified": False,
        "scope": "One-row independent JAX trajectory and dev-only NQC/attachment parity; no physics or release qualification",
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
        "input_sha256": {name: item["sha256"] for name, item in plan["inputs"].items()},
        "case": plan["case"], "level": plan["level"], "settings": settings,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")

    field_t, ham_t, flux_t = _source_components(plan["case"])
    identity = driver._prepare_public(source_plan)
    original_integrate = driver.integrate_two_stage_trajectory
    original_run = driver.run
    jax_trajectories, jax_series = [], []

    def capture_integrate(*args, **kwargs):
        result = original_integrate(*args, **kwargs)
        jax_trajectories.append(result)
        return result

    def capture_run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        jax_series.append(result[2])
        return result

    with (patch.object(driver, "integrate_two_stage_trajectory", capture_integrate),
          patch.object(driver, "run", capture_run)):
        diagnostic, torch_series, stage, inputs, controls = _capture_level(
            plan["level"], settings, source_plan, identity,
            {**boundary_plan, "case": plan["case"]}, field_t, ham_t, flux_t,
            torch_attachment._full_series_nqc, torch_attachment._eval_batch_scalar)
    if len(jax_trajectories) != 1 or len(jax_series) != 1:
        raise ValueError("expected exactly one independent JAX integration and series")
    jrun = jax_trajectories[0]
    jax_baseline_series = jax_series[0]
    frozen = prior["levels"][plan["level"]]
    torch_summary = _series_summary(torch_series)
    if (torch_summary != frozen["candidate_torch"]
            or controls["continuous_frequency"] != frozen["controls"]["continuous_frequency"]
            or stage["boundary_time_M"] != frozen["boundary_time_M"]):
        raise ValueError("Torch candidate does not replay frozen boundary evidence")
    if fixed_prior["result"]["torch_replay_mode_sha256"] != torch_summary["mode_sha256"]:
        raise ValueError("fixed-input backend reference has different Torch candidate")
    print("Torch candidate and independent JAX trajectory replayed", flush=True)

    m1, m2, chi1, chi2, _ = driver.CASES[plan["case"]]
    nu = m1*m2/(m1+m2)**2
    delta = (m1-m2)/(m1+m2)
    ham_j = driver.JaxSourceHam(nu, chi1, chi2, delta=delta)
    flux_j = driver.JaxFlux(**driver.SOURCE_FLUX_OPTIONS)
    field_j = driver.JaxIntegrator(ham_j, flux_j).vector_field
    jnu, jchi = driver.jax_parameters(m1, m2, chi1, chi2)
    jfine_time = np.asarray(jrun.fine_time)
    jfine_state = jrun.fine_state
    restart_time = float(jrun.coarse_time[jrun.restart_index])
    jmatch = float(stage["jax_match_time_M"]-restart_time)
    center, bracket = _hermite_center(jfine_time, jfine_state, jmatch, field_j)
    rows = _jax_ode_rows(center, field_j, ham_j, flux_j)
    targets = np.asarray(jnp.stack(tuple(seobnrv4_nqc_targets(jnu, jchi))))
    amp_matrix = rows[:, 1:4]
    amp_rhs = targets[:3]-rows[:, 0]
    phase_matrix = rows[1:, 4:6]
    phase_rhs = np.asarray([rows[0, 6]-targets[3], rows[1, 6]-targets[4]])
    a, acond, ares = _solve_checked(jnp.asarray(amp_matrix), jnp.asarray(amp_rhs),
                                   1e10, "independent JAX ODE amplitude system")
    b, pcond, pres = _solve_checked(jnp.asarray(phase_matrix), jnp.asarray(phase_rhs),
                                   1e10, "independent JAX ODE phase system")
    a, b = np.asarray(a), np.asarray(b)
    torch_a, torch_b = (np.asarray(x) for x in stage["nqc_coefficients"])
    print("Independent JAX ODE matching solved", flush=True)

    fine_state = np.asarray(jfine_state)
    fine_omega = np.asarray(jax.jit(jax.vmap(field_j))(jfine_state))[:, 1]
    jbase = np.asarray(stage["jax_fine_base_mode"])
    jcorrected = np.asarray(jnp.asarray(jbase)*driver.jax_factor(
        jfine_state[:, 0], jfine_state[:, 2], jnp.asarray(fine_omega),
        NQCCoefficients(jnp.asarray(a), jnp.asarray(b))))
    trajectory_only = np.asarray(jnp.asarray(jbase)*driver.jax_factor(
        jfine_state[:, 0], jfine_state[:, 2], jnp.asarray(fine_omega),
        NQCCoefficients(jnp.asarray(torch_a), jnp.asarray(torch_b))))
    boundary_index = int(np.searchsorted(jfine_time, jmatch, side="right")-1)
    if not 1 <= boundary_index < len(jfine_time)-1:
        raise ValueError("independent JAX boundary lacks neighbors")
    boundary_time = float(jfine_time[boundary_index])
    boundary_rows = _jax_ode_rows(jfine_state[boundary_index], field_j, ham_j, flux_j)
    boundary_frequency = float(boundary_rows[0, 6]-boundary_rows[1, 4:6]@b)
    boundary_amplitude = float(abs(jcorrected[boundary_index]))
    boundary_phase = float(np.asarray(unwrap_mode_phase(jnp.asarray(jcorrected)))[boundary_index])

    coarse_omega = jax.jit(jax.vmap(field_j))(jrun.coarse_state)[:, 1]
    coarse_mode, _ = jax.jit(lambda states: driver.jax_mode(
        states, ham_j, flux_j,
        **(driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True})))(jrun.coarse_state)
    jcoarse = np.asarray(coarse_mode*driver.jax_factor(
        jrun.coarse_state[:, 0], jrun.coarse_state[:, 2], coarse_omega,
        NQCCoefficients(jnp.asarray(a), jnp.asarray(b))))
    torch_time, torch_mode = (np.asarray(x) for x in torch_series)
    jtime, _ = (np.asarray(x) for x in jax_baseline_series)
    jrestart = int(jrun.restart_index)
    post_query = jtime[jrestart:]-restart_time
    fine_step = float(jfine_time[1]-jfine_time[0])
    pre_boundary = post_query < boundary_time
    fine_indices = np.rint(post_query[pre_boundary]/fine_step).astype(int)
    if (np.any(fine_indices >= len(jfine_time))
            or np.max(abs(jfine_time[fine_indices]-post_query[pre_boundary])) > 1e-10):
        raise ValueError("independent JAX output is not on the fine knot chart")
    post_mode = np.empty(len(post_query), dtype=np.complex128)
    post_mode[pre_boundary] = jcorrected[fine_indices]

    jmass = driver.jax_final_mass(m1, m2, chi1, chi2)
    jspin = driver.jax_final_spin(m1, m2, chi1, chi2, fit_order=(3, 4))
    jw, jdamp = driver.jax_qnm(jspin)
    sigma = (jw-1j*jdamp)/jmass
    dt_output = float(jtime[jrestart+1]-jtime[jrestart])
    internal_step = dt_output/2**max(0, np.ceil(np.log2(50*dt_output)))
    support_count = int(np.floor(10/(-float(jnp.imag(sigma))*internal_step)))
    support_end = support_count*internal_step
    elapsed = jnp.asarray(np.maximum(post_query-boundary_time, 0))
    shape = seobnrv4_peak_preserving_shape(sigma, seobnrv4_merger_shape(jnu, jchi))
    ring = merger_ringdown_22(
        elapsed, sigma, jnp.asarray(boundary_amplitude), jnp.asarray(0.0),
        jnp.asarray(boundary_frequency), jnp.asarray(boundary_phase), shape)
    ring = np.asarray(jnp.where(elapsed < support_end, ring, 0))
    post_mode[~pre_boundary] = ring[~pre_boundary]
    jmode = np.concatenate((jcoarse[:jrestart], post_mode))
    if len(jmode) != len(jtime):
        raise ValueError("independent JAX whole-mode length changed")

    common = min(len(torch_time), len(jtime))
    if common <= max(stage["restart_index"], jrestart)+1:
        raise ValueError("insufficient common whole-mode support")
    chart_error = float(np.max(abs(torch_time[:common]-jtime[:common])))
    torch_restart = int(stage["restart_index"])
    torch_boundary = float(stage["boundary_time_M"])
    torch_regions = {
        "coarse_prefix": np.arange(common) < torch_restart,
        "post_restart_pre_boundary": (np.arange(common) >= torch_restart)
            & (torch_time[:common] < torch_boundary),
        "ringdown": torch_time[:common] >= torch_boundary,
    }
    region_metrics = {}
    for name, mask in torch_regions.items():
        region_metrics[name] = {
            "samples": int(mask.sum()),
            "complex_relative_l2": _relative_l2(jmode[:common][mask], torch_mode[:common][mask]),
        }
    torch_ring = torch_regions["ringdown"]
    jax_ring = jtime[:common] >= restart_time+boundary_time
    fine_common = min(len(stage["fine_time"]), len(jfine_time))
    fine_chart_error = float(np.max(abs(np.asarray(stage["fine_time"])[:fine_common]
                                        -np.asarray(stage["jax_fine_time"])[:fine_common])))
    fine_state_common = min(len(fine_state), len(stage["fine_state"]))
    fine_state_error = float(np.max(abs(fine_state[:fine_state_common]
                                        -np.asarray(stage["fine_state"])[:fine_state_common])))
    result = {
        "torch_candidate": torch_summary,
        "jax_baseline": _series_summary(jax_baseline_series),
        "jax_candidate": _series_summary((jtime, jmode)),
        "initial_state_max_abs": diagnostic["initial_state_max_abs"],
        "restart_indices": {"torch": torch_restart, "jax": jrestart},
        "whole_chart": {"torch_samples": len(torch_time), "jax_samples": len(jtime),
                        "shared_samples": common, "max_abs_M": chart_error,
                        "exact_on_shared_support": bool(np.array_equal(torch_time[:common], jtime[:common]))},
        "fine_chart": {"torch_samples": len(stage["fine_time"]), "jax_samples": len(jfine_time),
                       "max_abs_M": fine_chart_error},
        "fine_state_max_abs": fine_state_error,
        "match": {"torch_time_M": stage["match_time_M"], "jax_time_M": stage["jax_match_time_M"],
                  "time_abs_M": abs(stage["jax_match_time_M"]-stage["match_time_M"]),
                  "jax_hermite_bracket": bracket,
                  "jax_hermite_state": np.asarray(center).tolist(),
                  "torch_hermite_state": fixed_prior["result"]["fixed_match_center"],
                  "state_max_abs": float(np.max(abs(np.asarray(center)-np.asarray(fixed_prior["result"]["fixed_match_center"]))))},
        "nqc": {"jax_amplitude_coefficients": a.tolist(), "jax_phase_coefficients": b.tolist(),
                "torch_amplitude_coefficients": torch_a.tolist(), "torch_phase_coefficients": torch_b.tolist(),
                "amplitude_coefficient_max_abs": float(np.max(abs(a-torch_a))),
                "phase_coefficient_max_abs": float(np.max(abs(b-torch_b))),
                "jax_amplitude_condition": float(acond), "jax_phase_condition": float(pcond),
                "jax_linear_residual_max_abs": max(float(np.max(abs(np.asarray(ares)))),
                                                    float(np.max(abs(np.asarray(pres))))),
                "fixed_input_parity_reference": {
                    "corrected_fine_complex_relative_l2": fixed_prior["result"]["corrected_fine_complex_relative_l2"],
                    "ringdown_complex_relative_l2": fixed_prior["result"]["ringdown_complex_relative_l2"]}},
        "fine_mode": {"raw_complex_relative_l2": _shared_relative_l2(jbase, stage["fine_base_mode"]),
                      "trajectory_only_torch_coefficients_complex_relative_l2": _shared_relative_l2(
                          trajectory_only, stage["fine_corrected_inspiral"]),
                      "own_minus_torch_coefficients_complex_relative_l2": _shared_relative_l2(
                          jcorrected, trajectory_only),
                      "own_corrected_complex_relative_l2": _shared_relative_l2(
                          jcorrected, stage["fine_corrected_inspiral"]),
                      "jax_corrected_sha256": _sha_array(jcorrected)},
        "boundary": {"torch_index": int(stage["boundary_index"]), "jax_index": boundary_index,
                     "torch_time_M": torch_boundary, "jax_time_M": restart_time+boundary_time,
                     "amplitude_abs": abs(boundary_amplitude-inputs["amplitude"]),
                     "phase_modulo_abs": float(abs(np.angle(np.exp(1j*(boundary_phase-inputs["phase"]))))),
                     "frequency_abs": abs(boundary_frequency-controls["continuous_frequency"]),
                     "jax_frequency": boundary_frequency,
                     "torch_frequency": controls["continuous_frequency"]},
        "support": {"torch_count": int(inputs["support_count"]), "jax_count": support_count,
                    "torch_internal_step_M": inputs["dt_high"], "jax_internal_step_M": internal_step,
                    "torch_end_M": inputs["support_end"], "jax_end_M": support_end,
                    "ringdown_region_equal": bool(np.array_equal(torch_ring, jax_ring)),
                    "zero_pattern_equal": bool(np.array_equal(
                        jmode[:common][torch_ring] == 0, torch_mode[:common][torch_ring] == 0))},
        "regions": region_metrics,
        "whole_complex_relative_l2": _relative_l2(jmode[:common], torch_mode[:common]),
        "jax_mode_finite": bool(np.all(np.isfinite(jmode))),
    }
    report["result"] = result
    limits = plan["checks"]
    report["checks"] = {
        "initial_state_max_abs": result["initial_state_max_abs"] <= limits["initial_state_max_abs"],
        "restart_index_equal": torch_restart == jrestart,
        "whole_chart_exact": result["whole_chart"]["exact_on_shared_support"]
            and len(torch_time) == len(jtime),
        "fine_chart_max_abs": result["fine_chart"]["max_abs_M"] <= limits["fine_chart_max_abs_M"],
        "fine_state_max_abs": result["fine_state_max_abs"] <= limits["fine_state_max_abs"],
        "match_time_abs": result["match"]["time_abs_M"] <= limits["match_time_abs_M"],
        "match_state_max_abs": result["match"]["state_max_abs"] <= limits["match_state_max_abs"],
        "coefficient_max_abs": max(result["nqc"]["amplitude_coefficient_max_abs"],
                                   result["nqc"]["phase_coefficient_max_abs"])
            <= limits["coefficient_max_abs"],
        "linear_residual_max_abs": result["nqc"]["jax_linear_residual_max_abs"]
            <= limits["linear_residual_max_abs"],
        "boundary_index_equal": boundary_index == int(stage["boundary_index"]),
        "boundary_amplitude_abs": result["boundary"]["amplitude_abs"]
            <= limits["boundary_amplitude_abs"],
        "boundary_phase_modulo_abs": result["boundary"]["phase_modulo_abs"]
            <= limits["boundary_phase_modulo_abs"],
        "boundary_frequency_abs": result["boundary"]["frequency_abs"]
            <= limits["boundary_frequency_abs"],
        "support_equal": (result["support"]["torch_count"] == support_count
                          and inputs["dt_high"] == internal_step
                          and result["support"]["ringdown_region_equal"]
                          and result["support"]["zero_pattern_equal"]),
        "trajectory_only_fine_l2": result["fine_mode"]["trajectory_only_torch_coefficients_complex_relative_l2"]
            <= limits["trajectory_only_fine_l2"],
        "matching_delta_fine_l2": result["fine_mode"]["own_minus_torch_coefficients_complex_relative_l2"]
            <= limits["matching_delta_fine_l2"],
        "whole_mode_l2": result["whole_complex_relative_l2"] <= limits["whole_mode_l2"],
        "each_region_l2": all(item["complex_relative_l2"] <= limits["each_region_l2"]
                              for item in region_metrics.values()),
        "finite": result["jax_mode_finite"],
    }
    report["checks_passed"] = all(report["checks"].values())
    report["status"] = "diagnostic_complete"
    report["elapsed_s"] = time.monotonic()-started
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"checks": report["checks"], "whole_l2": result["whole_complex_relative_l2"],
                      "regions": region_metrics}), flush=True)


if __name__ == "__main__":
    main()

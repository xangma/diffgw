"""Fixed-input JAX/Torch parity for corrected ODE NQC and boundary frequency."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_source_aligned_ode_chain_boundary_frequency import (
    _capture_level,
)
from dev.native_rom.diagnose_source_ode_chain_boundary_frequency_domain import _source_components
from dev.native_rom.eob.jax.numerical_attachment_clean import (
    _solve_checked, unwrap_mode_phase,
)
from jaxwave.approximants.eob.merger_ringdown import (
    merger_ringdown_22, seobnrv4_merger_shape, seobnrv4_peak_preserving_shape,
)
from jaxwave.approximants.eob.nqc import NQCCoefficients, seobnrv4_nqc_targets
from torchwave.approximants.eob import numerical_attachment as torch_attachment


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "jax_source_ode_boundary_frequency_parity_plan_v2.json"
OUTPUT = EVIDENCE / "jax_source_ode_boundary_frequency_parity.json"


def _checked_plan():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-source-ode-boundary-frequency-parity-plan.v2"
            or plan["case"] != "strong_aligned" or plan["level"] != "baseline"
            or plan["fixed_input"] != "Torch fine trajectory and match chart"
            or plan["release_qualified"] is not False):
        raise ValueError("frozen parity plan changed")
    for name, pinned in plan["inputs"].items():
        if q.sha256(ROOT / pinned["path"]) != pinned["sha256"]:
            raise ValueError(f"pinned input changed: {name}")
    if q.sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen parity runner changed")
    return plan


def _complex_pairs(values):
    return [[float(z.real), float(z.imag)] for z in np.asarray(values).reshape(-1)]


def _relative_l2(actual, expected):
    return float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))


def _jax_ode_rows(center, field, hamiltonian, flux):
    options = driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True}

    def quantities(state):
        vector = field(state)
        mode, frequency = driver.jax_mode(state, hamiltonian, flux, **options)
        amplitude = jnp.abs(mode)
        q_basis, p_basis = driver.jax_basis(state[0], state[2], vector[1])
        return jnp.concatenate((amplitude[None], amplitude*q_basis,
                                p_basis, frequency[None]))

    def first_lie(state):
        return jax.jvp(quantities, (state,), (field(state),))[1]

    value = quantities(center)
    first = first_lie(center)
    second = jax.jvp(first_lie, (center,), (field(center),))[1]
    return np.asarray(jnp.stack((value, first, second)))


def main():
    plan = _checked_plan()
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    boundary_plan = json.loads((ROOT / plan["inputs"]["boundary_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    prior = json.loads((ROOT / plan["inputs"]["boundary_result"]["path"]).read_text())
    amplitude_prior = json.loads((ROOT / plan["inputs"]["amplitude_result"]["path"]).read_text())
    level = next(row for row in ladder["levels"] if row["name"] == plan["level"])
    settings = {"coarse_output_step_M": ladder["coarse_output_step_M"],
                "max_step_M": ladder["max_step_M"], "atol": level["atol"],
                "rtol": level["rtol"]}
    started = time.monotonic()
    report = {"schema": "diffgw.seobnrv4.jax-source-ode-boundary-frequency-parity.v1",
              "status": "running", "release_qualified": False,
              "scope": "One-row fixed-input backend parity; no independent trajectory or physical qualification",
              "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
              "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
              "input_sha256": {name: item["sha256"] for name, item in plan["inputs"].items()},
              "case": plan["case"], "level": plan["level"], "settings": settings}
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")

    field_t, ham_t, flux_t = _source_components(plan["case"])
    identity = driver._prepare_public(source_plan)
    _, series, stage, inputs, controls = _capture_level(
        plan["level"], settings, source_plan, identity,
        {**boundary_plan, "case": plan["case"]}, field_t, ham_t, flux_t,
        torch_attachment._full_series_nqc, torch_attachment._eval_batch_scalar)
    frozen = prior["levels"][plan["level"]]
    from dev.native_rom.diagnose_jax_source_tolerance_ladder import _series_summary
    if (_series_summary(series) != frozen["candidate_torch"]
            or controls["continuous_frequency"] != frozen["controls"]["continuous_frequency"]
            or stage["boundary_time_M"] != frozen["boundary_time_M"]):
        raise ValueError("Torch baseline does not replay frozen boundary-frequency evidence")
    print("Torch candidate replayed", flush=True)

    m1, m2, chi1, chi2, _ = driver.CASES[plan["case"]]
    nu = m1*m2/(m1+m2)**2
    delta = (m1-m2)/(m1+m2)
    ham_j = driver.JaxSourceHam(nu, chi1, chi2, delta=delta)
    flux_j = driver.JaxFlux(**driver.SOURCE_FLUX_OPTIONS)
    field_j = driver.JaxIntegrator(ham_j, flux_j).vector_field
    state = jnp.asarray(stage["fine_state"])
    fine_time = np.asarray(stage["fine_time"])-float(stage["fine_time"][0])
    center = jnp.asarray(amplitude_prior["levels"][plan["level"]]
                         ["chain_matrix"]["interpolated_state"])
    rows = _jax_ode_rows(center, field_j, ham_j, flux_j)
    amp_matrix = rows[:, 1:4]
    phase_matrix = rows[1:, 4:6]
    jnu, jchi = driver.jax_parameters(m1, m2, chi1, chi2)
    targets = seobnrv4_nqc_targets(jnu, jchi)
    target_values = np.asarray(jnp.stack(tuple(targets)))
    amp_rhs = target_values[:3]-rows[:, 0]
    phase_rhs = np.asarray([rows[0, 6]-target_values[3],
                            rows[1, 6]-target_values[4]])
    a, acond, ares = _solve_checked(jnp.asarray(amp_matrix), jnp.asarray(amp_rhs),
                                   1e10, "JAX ODE amplitude system")
    b, pcond, pres = _solve_checked(jnp.asarray(phase_matrix), jnp.asarray(phase_rhs),
                                   1e10, "JAX ODE phase system")
    a, b = np.asarray(a), np.asarray(b)
    amp_reference = amplitude_prior["levels"][plan["level"]]["amplitude_solve"]
    phase_reference = amplitude_prior["levels"][plan["level"]]
    matrix_errors = {
        "amplitude_matrix_max_abs": float(np.max(abs(amp_matrix-np.asarray(amp_reference["amplitude_matrix"])))),
        "amplitude_rhs_max_abs": float(np.max(abs(amp_rhs-np.asarray(amp_reference["amplitude_rhs"])))),
        "phase_matrix_max_abs": float(np.max(abs(phase_matrix-np.asarray(phase_reference["phase_matrix"])))),
        "phase_rhs_max_abs": float(np.max(abs(phase_rhs-np.asarray(phase_reference["phase_rhs"])))),
        "amplitude_coefficients_max_abs": float(np.max(abs(a-np.asarray(amp_reference["amplitude_coefficients"])))),
        "phase_coefficients_max_abs": float(np.max(abs(b-np.asarray(phase_reference["phase_coefficients"])))),
    }
    print("JAX ODE matching solved", flush=True)

    mode_fn = jax.jit(lambda states: driver.jax_mode(
        states, ham_j, flux_j, **(driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True})))
    raw_mode, _ = mode_fn(state)
    vectors = jax.jit(jax.vmap(field_j))(state)
    corrected = raw_mode*driver.jax_factor(
        state[:, 0], state[:, 2], vectors[:, 1],
        NQCCoefficients(jnp.asarray(a), jnp.asarray(b)))
    corrected_np = np.asarray(corrected)
    corrected_reference = np.asarray(stage["fine_corrected_inspiral"])
    corrected_l2 = _relative_l2(corrected_np, corrected_reference)
    boundary_index = int(stage["boundary_index"])
    if (boundary_index != controls["boundary_index"]
            or abs(fine_time[boundary_index]-controls["boundary_time_local_M"]) > 1e-12):
        raise ValueError("fixed boundary index or time changed")
    boundary_state = state[boundary_index]
    boundary_rows = _jax_ode_rows(boundary_state, field_j, ham_j, flux_j)
    continuous_frequency = float(boundary_rows[0, 6]-boundary_rows[1, 4:6]@b)
    boundary_amplitude = float(jnp.abs(corrected[boundary_index]))
    corrected_phase = np.asarray(unwrap_mode_phase(corrected))
    boundary_phase = float(corrected_phase[boundary_index])
    phase_error_modulo = float(abs(np.angle(np.exp(1j*(boundary_phase-inputs["phase"])))))

    whole_time, torch_whole = series
    whole_index = np.flatnonzero(np.asarray(whole_time) >= stage["boundary_time_M"])
    chart_index = boundary_plan["fine_refinement_factor"]*(whole_index-stage["restart_index"])
    elapsed = jnp.asarray(np.asarray(inputs["elapsed"]).reshape(-1)[chart_index])
    sigma = jnp.asarray(inputs["sigma"])
    shape = seobnrv4_peak_preserving_shape(sigma, seobnrv4_merger_shape(jnu, jchi))
    jring = merger_ringdown_22(elapsed, sigma, jnp.asarray(boundary_amplitude),
                               jnp.asarray(0.0), jnp.asarray(continuous_frequency),
                               jnp.asarray(boundary_phase), shape)
    jring = jnp.where(elapsed < inputs["support_end"], jring, 0)
    jring = np.asarray(jring).reshape(-1)
    tring = np.asarray(torch_whole)[whole_index]
    ring_l2 = _relative_l2(jring, tring)
    result = {
        "torch_replay_mode_sha256": frozen["candidate_torch"]["mode_sha256"],
        "fixed_match_center": np.asarray(center).tolist(),
        "jax_ode_rows": rows.tolist(),
        "jax_amplitude_matrix": amp_matrix.tolist(), "jax_amplitude_rhs": amp_rhs.tolist(),
        "jax_phase_matrix": phase_matrix.tolist(), "jax_phase_rhs": phase_rhs.tolist(),
        "jax_coefficients": {"amplitude": a.tolist(), "phase": b.tolist()},
        "jax_conditions": {"amplitude": float(acond), "phase": float(pcond)},
        "jax_linear_residual_max_abs": max(float(np.max(abs(np.asarray(ares)))),
                                            float(np.max(abs(np.asarray(pres))))),
        "matrix_errors": matrix_errors,
        "corrected_fine_complex_relative_l2": corrected_l2,
        "boundary_index": boundary_index,
        "boundary_amplitude_abs_difference": abs(boundary_amplitude-inputs["amplitude"]),
        "boundary_phase_modulo_abs_difference": phase_error_modulo,
        "boundary_frequency_abs_difference": abs(continuous_frequency-controls["continuous_frequency"]),
        "jax_boundary_frequency": continuous_frequency,
        "torch_boundary_frequency": controls["continuous_frequency"],
        "ringdown_complex_relative_l2": ring_l2,
        "ringdown_max_absolute": float(np.max(abs(jring-tring))),
        "ringdown_samples": len(jring),
        "ringdown_zero_pattern_equal": bool(np.array_equal(jring == 0, tring == 0)),
        "jax_ringdown": _complex_pairs(jring),
    }
    report["result"] = result
    limits = plan["checks"]
    report["checks"] = {
        "matching_rows_max_abs": max(matrix_errors[k] for k in
            ("amplitude_matrix_max_abs", "amplitude_rhs_max_abs",
             "phase_matrix_max_abs", "phase_rhs_max_abs")) <= limits["matching_rows_max_abs"],
        "coefficients_max_abs": max(matrix_errors[k] for k in
            ("amplitude_coefficients_max_abs", "phase_coefficients_max_abs"))
            <= limits["coefficients_max_abs"],
        "linear_residual_max_abs": result["jax_linear_residual_max_abs"]
            <= limits["linear_residual_max_abs"],
        "corrected_fine_complex_relative_l2": corrected_l2
            <= limits["corrected_fine_complex_relative_l2"],
        "boundary_amplitude_abs": result["boundary_amplitude_abs_difference"]
            <= limits["boundary_amplitude_abs"],
        "boundary_phase_modulo_abs": phase_error_modulo
            <= limits["boundary_phase_modulo_abs"],
        "boundary_frequency_abs": result["boundary_frequency_abs_difference"]
            <= limits["boundary_frequency_abs"],
        "ringdown_complex_relative_l2": ring_l2
            <= limits["ringdown_complex_relative_l2"],
        "ringdown_zero_pattern_equal": result["ringdown_zero_pattern_equal"],
    }
    report["checks_passed"] = all(report["checks"].values())
    report["status"] = "diagnostic_complete"
    report["elapsed_s"] = time.monotonic()-started
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"checks": report["checks"], "metrics": {
        k: v for k, v in result.items() if k.endswith("difference") or k.endswith("relative_l2")}}),
        flush=True)


if __name__ == "__main__":
    main()

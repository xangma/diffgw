"""Dev-only localization of eager/JIT changes through the physical NQC fit."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import qualify_generator as q
from .diagnose_jax_adaptive_physical_event import CASES
from .diagnose_jax_full_series_nqc_physical import MODE_OPTIONS
from .diagnose_jax_two_stage_trajectory import construct_case
from .eob.jax.adaptive_driver_clean import replay_accepted_steps
from .eob.jax.numerical_attachment_clean import (
    _unwrap_mode_phase_unchecked,
    full_orbital_peak,
    traceable_nqc_22_full_series_coefficients,
)
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import (
    _seobnrv4_nqc_targets_unchecked,
    nqc_22_basis,
    seobnrv4_nqc_parameters,
    seobnrv4_nqc_time_offset_numerical_compatibility,
)
from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_jit_stage_plan.json"
OUTPUT = Path(os.environ.get(
    "DIFFGW_DIAGNOSTIC_OUTPUT", PLAN.with_name("jax_nqc_jit_stages.json")))


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-jit-stage-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "high_start_mixed_spin"):
        raise ValueError("frozen NQC stage plan changed")
    case = plan["case"]
    m1, m2, chi1, chi2, _ = CASES[case]
    _, initial, _, field = construct_case(case)
    dt = plan["output_step_M"]
    run = integrate_two_stage_trajectory(
        field, initial, output_step=dt, max_duration=1000.,
        coarse_initial_step=dt, coarse_max_step=2., fine_max_step=2.,
        fine_refinement_factor=plan["fine_refinement_factor"],
        atol=plan["adaptive_atol"], rtol=plan["adaptive_rtol"])
    print("baseline trajectories", run.coarse.accepted_steps,
          run.fine.accepted_steps, flush=True)
    nu, delta = m1 * m2 / (m1 + m2) ** 2, (m1 - m2) / (m1 + m2)
    ham = SEOBNRv4Hamiltonian(nu, chi1, chi2, delta=delta)
    flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated",
                                 retain_pan_cubic_spin=False)
    nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    baseline_omega = jax.vmap(field)(run.fine_state)[:, 1]
    peak = full_orbital_peak(run.fine_time, baseline_omega).time
    match = peak + seobnrv4_nqc_time_offset_numerical_compatibility(nqc_nu, nqc_chi)
    targets = _seobnrv4_nqc_targets_unchecked(nqc_nu, nqc_chi)

    def stages(radius_offset):
        first = initial.at[0].add(radius_offset)
        coarse_knots = replay_accepted_steps(
            lambda _time, state: field(state), first, run.coarse.time)
        coarse = natural_cubic_spline(run.coarse.time, coarse_knots,
                                      scalar_output=False).evaluate(run.coarse_time)
        fine_knots = replay_accepted_steps(
            lambda _time, state: field(state), coarse[run.restart_index], run.fine.time)
        fine = natural_cubic_spline(run.fine.time, fine_knots,
                                    scalar_output=False).evaluate(run.fine_time)
        omega = jax.vmap(field)(fine)[:, 1]
        mode, _ = published_mode22_and_frequency(fine, ham, flux, **MODE_OPTIONS)
        amplitude = jnp.abs(mode)
        phase = _unwrap_mode_phase_unchecked(mode)
        q_basis, p_basis = nqc_22_basis(fine[:, 0], fine[:, 2], omega)
        products = amplitude[:, None] * q_basis
        combined = jnp.concatenate((amplitude[:, None], products, p_basis), axis=-1)
        spline = natural_cubic_spline(run.fine_time, combined, scalar_output=False)
        value = spline(match)
        first_derivative = spline.derivative(match)
        second_derivative = spline.derivative(match, 2)
        amplitude_matrix = jnp.stack((value[1:4], first_derivative[1:4],
                                      second_derivative[1:4]))
        amplitude_rhs = jnp.stack(targets[:3]) - jnp.stack((
            value[0], first_derivative[0], second_derivative[0]))
        phase_matrix = jnp.stack((first_derivative[4:6], second_derivative[4:6]))
        phase_spline = natural_cubic_spline(run.fine_time, phase,
                                            scalar_output=True)
        phase_derivatives = jnp.stack((phase_spline.derivative(match),
                                       phase_spline.derivative(match, 2)))
        phase_rhs = phase_derivatives - jnp.stack((targets.frequency,
                                                   targets.frequency_derivative))
        fit = traceable_nqc_22_full_series_coefficients(
            run.fine_time, fine[:, 0], fine[:, 2], omega, mode, match, targets)
        return (fine, omega, mode, phase, combined, value, first_derivative,
                second_derivative, phase_derivatives, amplitude_matrix,
                amplitude_rhs, phase_matrix, phase_rhs, fit[0], fit[1])

    eager = stages(jnp.asarray(0., dtype=jnp.float64))
    print("eager stages", time.monotonic() - started, flush=True)
    compiled = jax.jit(stages)(jnp.asarray(0., dtype=jnp.float64))
    print("compiled stages", time.monotonic() - started, flush=True)
    names = ("state", "orbital_frequency", "base_mode", "unwrapped_phase",
             "combined_spline_input", "matching_value", "matching_first",
             "matching_second", "phase_derivatives", "amplitude_matrix",
             "amplitude_rhs", "phase_matrix", "phase_rhs", "amplitude_coefficients",
             "phase_coefficients")
    differences = {name: {"max_abs": max_abs(compiled[i], eager[i]),
                          "relative_l2": relative_l2(compiled[i], eager[i])}
                   for i, name in enumerate(names)}
    def fit_only(fine, omega, mode):
        fit = traceable_nqc_22_full_series_coefficients(
            run.fine_time, fine[:, 0], fine[:, 2], omega, mode, match, targets)
        return fit[0], fit[1], fit[2], fit[3]

    fixed_input_eager = fit_only(eager[0], eager[1], eager[2])
    fixed_input_compiled = jax.jit(fit_only)(eager[0], eager[1], eager[2])
    compiled_input_separate = jax.jit(fit_only)(compiled[0], compiled[1], compiled[2])
    controlled = {}
    for name, index, component in (
        ("amplitude_coefficients", 13, 0), ("phase_coefficients", 14, 1),
        ("amplitude_matrix", 9, 2), ("phase_matrix", 11, 3)):
        controlled[name] = {
            "fixed_input_eager_to_whole_eager_max_abs": max_abs(
                fixed_input_eager[component], eager[index]),
            "fixed_input_jit_to_whole_eager_max_abs": max_abs(
                fixed_input_compiled[component], eager[index]),
            "fixed_input_jit_to_whole_jit_max_abs": max_abs(
                fixed_input_compiled[component], compiled[index]),
            "compiled_input_separate_jit_to_whole_jit_max_abs": max_abs(
                compiled_input_separate[component], compiled[index]),
            "fixed_input_jit_to_compiled_input_jit_max_abs": max_abs(
                fixed_input_compiled[component], compiled_input_separate[component]),
        }
    solves = {}
    for label, matrix_index, rhs_index, coefficient_index in (
        ("amplitude", 9, 10, 13), ("phase", 11, 12, 14)):
        eager_matrix, compiled_matrix = np.asarray(eager[matrix_index]), np.asarray(compiled[matrix_index])
        eager_rhs, compiled_rhs = np.asarray(eager[rhs_index]), np.asarray(compiled[rhs_index])
        eager_coefficients, compiled_coefficients = (
            np.asarray(eager[coefficient_index]), np.asarray(compiled[coefficient_index]))
        solves[label] = {
            "eager_condition": float(np.linalg.cond(eager_matrix)),
            "compiled_condition": float(np.linalg.cond(compiled_matrix)),
            "eager_residual_relative_l2": relative_l2(eager_matrix @ eager_coefficients, eager_rhs),
            "compiled_residual_relative_l2": relative_l2(compiled_matrix @ compiled_coefficients, compiled_rhs),
            "rhs_only_coefficient_change_max_abs": max_abs(
                np.linalg.solve(eager_matrix, compiled_rhs),
                np.linalg.solve(eager_matrix, eager_rhs)),
            "matrix_only_coefficient_change_max_abs": max_abs(
                np.linalg.solve(compiled_matrix, eager_rhs),
                np.linalg.solve(eager_matrix, eager_rhs)),
            "numpy_solver_eager_vs_compiled_coefficient_max_abs": max_abs(
                np.linalg.solve(compiled_matrix, compiled_rhs),
                np.linalg.solve(eager_matrix, eager_rhs)),
        }
    replay_error = max_abs(eager[0], run.fine_state)
    checks = {
        "replay": replay_error <= plan["checks"]["max_replay_state_absolute_error"],
        "base_mode": differences["base_mode"]["relative_l2"] <= plan["checks"]["max_eager_jit_base_mode_relative_l2"],
        "matching_residual": max(solves[k][f"{execution}_residual_relative_l2"]
                                 for k in solves for execution in ("eager", "compiled"))
        <= plan["checks"]["max_matching_residual_relative_l2"],
    }
    result = {
        "schema": "diffgw.seobnrv4.jax-nqc-jit-stages.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "platform": platform.platform(), "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "case": case, "fine_samples": len(run.fine_time),
        "match_time_M": float(match), "replay_state_max_abs": replay_error,
        "differences": differences, "controlled": controlled, "solves": solves,
        "checks": checks, "diagnostic_checks_passed": all(checks.values()),
        "elapsed_s": time.monotonic() - started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"differences": differences, "controlled": controlled,
                      "solves": solves}, sort_keys=True), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

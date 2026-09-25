"""Dev-only scaling and residual-centering controls for the NQC amplitude solve."""

from __future__ import annotations

import json
import os
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import qualify_generator as q
from .diagnose_jax_adaptive_physical_event import CASES
from .diagnose_jax_full_series_nqc_physical import MODE_OPTIONS
from .diagnose_jax_two_stage_trajectory import construct_case
from .eob.jax.numerical_attachment_clean import (
    _solve_traceable,
    full_orbital_peak,
    traceable_nqc_22_full_series_coefficients,
)
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import (
    _seobnrv4_nqc_targets_unchecked,
    NQCCoefficients,
    nqc_22_basis,
    nqc_22_factor,
    seobnrv4_nqc_parameters,
    seobnrv4_nqc_time_offset_numerical_compatibility,
)
from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_amplitude_solve_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_amplitude_solve.json")
VARIANTS = (
    "baseline_equilibrated", "local_time_derivative_row_scale",
    "residual_centered_refinement", "row_scale_then_residual_refinement",
)


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-amplitude-solve-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "high_start_mixed_spin"
            or tuple(plan["variants"]) != VARIANTS):
        raise ValueError("frozen amplitude-solve plan changed")
    baseline_path = q.ROOT / plan["baseline_result"]
    if q.sha256(baseline_path) != plan["baseline_sha256"]:
        raise ValueError("pinned fixed-input NQC evidence changed")
    baseline_evidence = json.loads(baseline_path.read_text())
    if (baseline_evidence["schema"] != "diffgw.seobnrv4.jax-nqc-spline-barrier.v1"
            or baseline_evidence["case"] != plan["case"]):
        raise ValueError("pinned fixed-input NQC scope changed")

    m1, m2, chi1, chi2, _ = CASES[plan["case"]]
    _, initial, _, field = construct_case(plan["case"])
    dt = plan["output_step_M"]
    run = integrate_two_stage_trajectory(
        field, initial, output_step=dt, max_duration=1000.,
        coarse_initial_step=dt, coarse_max_step=2., fine_max_step=2.,
        fine_refinement_factor=plan["fine_refinement_factor"],
        atol=plan["adaptive_atol"], rtol=plan["adaptive_rtol"])
    nu, delta = m1 * m2 / (m1 + m2) ** 2, (m1 - m2) / (m1 + m2)
    ham = SEOBNRv4Hamiltonian(nu, chi1, chi2, delta=delta)
    flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated",
                                 retain_pan_cubic_spin=False)
    omega = jax.vmap(field)(run.fine_state)[:, 1]
    mode, _ = published_mode22_and_frequency(run.fine_state, ham, flux,
                                              **MODE_OPTIONS)
    compiled_mode = jax.jit(lambda state: published_mode22_and_frequency(
        state, ham, flux, **MODE_OPTIONS)[0])(run.fine_state)
    nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    peak = full_orbital_peak(run.fine_time, omega).time
    match = peak + seobnrv4_nqc_time_offset_numerical_compatibility(
        nqc_nu, nqc_chi)
    targets = _seobnrv4_nqc_targets_unchecked(nqc_nu, nqc_chi)
    match_index = int(np.searchsorted(np.asarray(run.fine_time),
                                      float(match), side="right") - 1)
    match_cell_width = float(run.fine_time[match_index + 1]
                             - run.fine_time[match_index])
    row_factors = jnp.asarray((1., match_cell_width,
                               match_cell_width ** 2))
    print("fixed chart", len(run.fine_time), float(match),
          match_cell_width, flush=True)

    def stages(variant):
        def evaluate(fine, orbital_frequency, base_mode):
            amplitude = jnp.abs(base_mode)
            q_basis, p_basis = nqc_22_basis(
                fine[:, 0], fine[:, 2], orbital_frequency)
            combined = jnp.concatenate((amplitude[:, None],
                                        amplitude[:, None] * q_basis,
                                        p_basis), axis=-1)
            spline = natural_cubic_spline(
                run.fine_time, combined, scalar_output=False)
            value = spline(match)
            first = spline.derivative(match)
            second = spline.derivative(match, 2)
            matrix = jnp.stack((value[1:4], first[1:4], second[1:4]))
            rhs = jnp.stack(targets[:3]) - jnp.stack((
                value[0], first[0], second[0]))
            if variant in ("local_time_derivative_row_scale",
                           "row_scale_then_residual_refinement"):
                solve_matrix = matrix * row_factors[:, None]
                solve_rhs = rhs * row_factors
            else:
                solve_matrix, solve_rhs = matrix, rhs
            coefficients = _solve_traceable(solve_matrix, solve_rhs)[0]
            if variant in ("residual_centered_refinement",
                           "row_scale_then_residual_refinement"):
                residual = solve_rhs - solve_matrix @ coefficients
                coefficients = coefficients + _solve_traceable(
                    solve_matrix, residual)[0]
            reference_fit = traceable_nqc_22_full_series_coefficients(
                run.fine_time, fine[:, 0], fine[:, 2], orbital_frequency,
                base_mode, match, targets)
            corrected = base_mode * nqc_22_factor(
                fine[:, 0], fine[:, 2], orbital_frequency,
                NQCCoefficients(coefficients, reference_fit[1]))
            solve_residual = jnp.linalg.norm(matrix @ coefficients - rhs) / jnp.linalg.norm(rhs)
            return coefficients, corrected, solve_residual, reference_fit[0]
        return evaluate

    fixed_inputs = (run.fine_state, omega, mode)
    results = {}
    for variant in VARIANTS:
        eager = stages(variant)(*fixed_inputs)
        compiled = jax.jit(stages(variant))(*fixed_inputs)
        results[variant] = (eager, compiled)
        print(variant, time.monotonic() - started, flush=True)

    baseline_eager = results["baseline_equilibrated"][0]
    measures = {}
    checks = {}
    limits = plan["checks"]
    for variant, (eager, compiled) in results.items():
        observed = {
            "eager_jit_amplitude_coefficients_max_abs": max_abs(compiled[0], eager[0]),
            "eager_jit_corrected_mode_relative_l2": relative_l2(compiled[1], eager[1]),
            "eager_vs_baseline_amplitude_coefficients_max_abs": max_abs(
                eager[0], baseline_eager[0]),
            "eager_vs_baseline_corrected_mode_relative_l2": relative_l2(
                eager[1], baseline_eager[1]),
            "eager_solve_residual_relative_l2": float(eager[2]),
            "jit_solve_residual_relative_l2": float(compiled[2]),
            "eager_vs_reference_fit_coefficients_max_abs": max_abs(eager[0], eager[3]),
        }
        measures[variant] = observed
        checks[variant] = {
            "preserves_eager_corrected_mode": observed[
                "eager_vs_baseline_corrected_mode_relative_l2"]
            <= limits["max_eager_vs_baseline_corrected_mode_relative_l2"],
            "coefficients_eager_jit": observed[
                "eager_jit_amplitude_coefficients_max_abs"]
            <= limits["max_eager_jit_amplitude_coefficients_absolute_error"],
            "corrected_mode_eager_jit": observed[
                "eager_jit_corrected_mode_relative_l2"]
            <= limits["max_eager_jit_corrected_mode_relative_l2"],
            "solve_residual": max(observed["eager_solve_residual_relative_l2"],
                                  observed["jit_solve_residual_relative_l2"])
            <= limits["max_solve_residual_relative_l2"],
        }
    base_mode_relative_l2 = relative_l2(compiled_mode, mode)
    baseline_fit_coefficients_max_abs = measures[
        "baseline_equilibrated"]["eager_vs_reference_fit_coefficients_max_abs"]
    controls = {
        "base_mode": base_mode_relative_l2
        <= limits["max_base_mode_eager_jit_relative_l2"],
        "baseline_fit": baseline_fit_coefficients_max_abs
        <= limits["max_baseline_fit_coefficients_absolute_error"],
    }
    candidate_passed = any(all(checks[variant].values())
                           for variant in VARIANTS[1:]) and all(controls.values())
    result = {
        "schema": "diffgw.seobnrv4.jax-nqc-amplitude-solve.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "baseline_sha256": q.sha256(baseline_path),
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "platform": platform.platform(), "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(), "case": plan["case"],
        "fine_samples": len(run.fine_time), "match_time_M": float(match),
        "match_cell_width_M": match_cell_width,
        "base_mode_relative_l2": base_mode_relative_l2,
        "baseline_fit_coefficients_max_abs": baseline_fit_coefficients_max_abs,
        "measures": measures, "checks": checks, "controls": controls,
        "candidate_passed": candidate_passed,
        "elapsed_s": time.monotonic() - started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"measures": measures, "checks": checks,
                      "controls": controls}, sort_keys=True), flush=True)
    return 0 if candidate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

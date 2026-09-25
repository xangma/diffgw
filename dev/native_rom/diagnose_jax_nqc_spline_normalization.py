"""Dev-only time and ordinate normalization controls for the NQC spline."""

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
from .diagnose_jax_nqc_amplitude_solve import max_abs, relative_l2
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

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_spline_normalization_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_spline_normalization.json")
VARIANTS = (
    "baseline_original", "match_knot_unit_step", "match_knot_ordinate_center",
    "match_knot_unit_step_and_ordinate_center",
)
MEASURES = (
    "eager_jit_matching_value_max_abs", "eager_jit_matching_first_max_abs",
    "eager_jit_matching_second_max_abs", "eager_jit_amplitude_matrix_max_abs",
    "eager_jit_amplitude_rhs_max_abs",
    "eager_jit_amplitude_coefficients_max_abs",
    "eager_jit_corrected_mode_relative_l2",
    "eager_vs_baseline_corrected_mode_relative_l2",
    "eager_solve_residual_relative_l2", "jit_solve_residual_relative_l2",
)


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-spline-normalization-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "high_start_mixed_spin"
            or tuple(plan["variants"]) != VARIANTS
            or tuple(plan["measures"]) != MEASURES):
        raise ValueError("frozen NQC spline normalization plan changed")
    baseline_path = q.ROOT / plan["baseline_result"]
    if q.sha256(baseline_path) != plan["baseline_sha256"]:
        raise ValueError("pinned NQC amplitude-solve evidence changed")
    baseline_evidence = json.loads(baseline_path.read_text())
    if (baseline_evidence["schema"] != "diffgw.seobnrv4.jax-nqc-amplitude-solve.v1"
            or baseline_evidence["case"] != plan["case"]):
        raise ValueError("pinned NQC amplitude-solve scope changed")

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
    match_left = run.fine_time[match_index]
    match_cell_width = run.fine_time[match_index + 1] - match_left
    print("fixed chart", len(run.fine_time), float(match),
          float(match_left), float(match_cell_width), flush=True)

    def stages(variant):
        normalize_time = variant in (
            "match_knot_unit_step", "match_knot_unit_step_and_ordinate_center")
        center_ordinate = variant in (
            "match_knot_ordinate_center",
            "match_knot_unit_step_and_ordinate_center")

        def evaluate(fine, orbital_frequency, base_mode):
            amplitude = jnp.abs(base_mode)
            q_basis, p_basis = nqc_22_basis(
                fine[:, 0], fine[:, 2], orbital_frequency)
            combined = jnp.concatenate((amplitude[:, None],
                                        amplitude[:, None] * q_basis,
                                        p_basis), axis=-1)
            offset = combined[match_index] if center_ordinate else jnp.zeros_like(combined[0])
            ordinates = combined - offset if center_ordinate else combined
            if normalize_time:
                knots = (run.fine_time - match_left) / match_cell_width
                query = (match - match_left) / match_cell_width
                derivative_unit = match_cell_width
            else:
                knots, query = run.fine_time, match
                derivative_unit = jnp.asarray(1., dtype=run.fine_time.dtype)
            spline = natural_cubic_spline(knots, ordinates, scalar_output=False)
            value = spline(query) + offset
            first = spline.derivative(query) / derivative_unit
            second = spline.derivative(query, 2) / derivative_unit ** 2
            matrix = jnp.stack((value[1:4], first[1:4], second[1:4]))
            rhs = jnp.stack(targets[:3]) - jnp.stack((
                value[0], first[0], second[0]))
            coefficients = _solve_traceable(matrix, rhs)[0]
            reference_fit = traceable_nqc_22_full_series_coefficients(
                run.fine_time, fine[:, 0], fine[:, 2], orbital_frequency,
                base_mode, match, targets)
            corrected = base_mode * nqc_22_factor(
                fine[:, 0], fine[:, 2], orbital_frequency,
                NQCCoefficients(coefficients, reference_fit[1]))
            residual = jnp.linalg.norm(matrix @ coefficients - rhs) / jnp.linalg.norm(rhs)
            return (value, first, second, matrix, rhs, coefficients,
                    corrected, residual, reference_fit[0])
        return evaluate

    fixed_inputs = (run.fine_state, omega, mode)
    results = {}
    for variant in VARIANTS:
        eager = stages(variant)(*fixed_inputs)
        compiled = jax.jit(stages(variant))(*fixed_inputs)
        results[variant] = (eager, compiled)
        print(variant, time.monotonic() - started, flush=True)

    baseline_eager = results["baseline_original"][0]
    limits = plan["checks"]
    measures = {}
    checks = {}
    for variant, (eager, compiled) in results.items():
        observed = dict(zip(MEASURES, (
            max_abs(compiled[0], eager[0]),
            max_abs(compiled[1], eager[1]),
            max_abs(compiled[2], eager[2]),
            max_abs(compiled[3], eager[3]),
            max_abs(compiled[4], eager[4]),
            max_abs(compiled[5], eager[5]),
            relative_l2(compiled[6], eager[6]),
            relative_l2(eager[6], baseline_eager[6]),
            float(eager[7]), float(compiled[7]),
        )))
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
    baseline_fit_coefficients_max_abs = max_abs(
        baseline_eager[5], baseline_eager[8])
    controls = {
        "base_mode": base_mode_relative_l2
        <= limits["max_base_mode_eager_jit_relative_l2"],
        "baseline_fit": baseline_fit_coefficients_max_abs
        <= limits["max_baseline_fit_coefficients_absolute_error"],
    }
    candidate_passed = all(controls.values()) and any(
        all(checks[variant].values()) for variant in VARIANTS[1:])
    result = {
        "schema": "diffgw.seobnrv4.jax-nqc-spline-normalization.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "baseline_sha256": q.sha256(baseline_path),
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "platform": platform.platform(), "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(), "case": plan["case"],
        "fine_samples": len(run.fine_time), "match_time_M": float(match),
        "match_left_time_M": float(match_left),
        "match_cell_width_M": float(match_cell_width),
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

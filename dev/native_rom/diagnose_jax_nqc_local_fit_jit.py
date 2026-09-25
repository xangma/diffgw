"""Dev-only fixed-input eager/JIT control for local-polynomial v4 NQC matching."""

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
from .diagnose_jax_spliced_whole_waveform import _jax_polynomial_nqc_controls
from .diagnose_jax_two_stage_trajectory import construct_case
from .eob.jax.numerical_attachment_clean import (
    _solve_traceable,
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

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_local_fit_jit_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_local_fit_jit.json")


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-local-fit-jit-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "high_start_mixed_spin"
            or plan["local_polynomial"] != {"degree": 4, "half_width_M": 2.0}):
        raise ValueError("frozen local-fit JIT plan changed")
    case = plan["case"]
    m1, m2, chi1, chi2, _ = CASES[case]
    _, initial, _, field = construct_case(case)
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
    mode, _ = published_mode22_and_frequency(run.fine_state, ham, flux, **MODE_OPTIONS)
    nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    peak = full_orbital_peak(run.fine_time, omega).time
    match = peak + seobnrv4_nqc_time_offset_numerical_compatibility(nqc_nu, nqc_chi)
    targets = _seobnrv4_nqc_targets_unchecked(nqc_nu, nqc_chi)

    def local_fit(state, frequency, base_mode):
        time_grid = run.fine_time
        amplitude = jnp.abs(base_mode)
        q_basis, p_basis = nqc_22_basis(state[:, 0], state[:, 2], frequency)
        values = jnp.column_stack((amplitude, amplitude[:, None] * q_basis, p_basis))
        matched_value = natural_cubic_spline(
            time_grid, values, scalar_output=False)(match)
        amplitude_derivatives, phase_matrix, rhs_derivatives = (
            _jax_polynomial_nqc_controls(
                time_grid, state, frequency, base_mode, match,
                **plan["local_polynomial"]))
        amplitude_matrix = jnp.concatenate((
            matched_value[None, 1:4], amplitude_derivatives), axis=0)
        amplitude_rhs = jnp.stack(targets[:3]) - jnp.concatenate((
            matched_value[:1], rhs_derivatives[0]))
        phase_rhs = rhs_derivatives[1] - jnp.stack((
            targets.frequency, targets.frequency_derivative))
        a, amplitude_condition, amplitude_residual = _solve_traceable(
            amplitude_matrix, amplitude_rhs)
        b, phase_condition, phase_residual = _solve_traceable(
            phase_matrix, phase_rhs)
        corrected = base_mode * ((1 + q_basis @ a) * jnp.exp(1j * (p_basis @ b)))
        return (a, b, corrected, amplitude_matrix, amplitude_rhs,
                phase_matrix, phase_rhs, amplitude_condition, phase_condition,
                amplitude_residual, phase_residual)

    baseline = traceable_nqc_22_full_series_coefficients(
        run.fine_time, run.fine_state[:, 0], run.fine_state[:, 2],
        omega, mode, match, targets)
    eager = local_fit(run.fine_state, omega, mode)
    compiled = jax.jit(local_fit)(run.fine_state, omega, mode)
    names = ("amplitude_coefficients", "phase_coefficients", "corrected_mode",
             "amplitude_matrix", "amplitude_rhs", "phase_matrix", "phase_rhs")
    differences = {name: {"max_abs": max_abs(compiled[i], eager[i]),
                          "relative_l2": relative_l2(compiled[i], eager[i])}
                   for i, name in enumerate(names)}
    residuals = {label: {
        execution: relative_l2(fit[matrix_index] @ fit[coefficient_index],
                               fit[rhs_index])
        for execution, fit in (("eager", eager), ("jit", compiled))}
        for label, matrix_index, rhs_index, coefficient_index in (
            ("amplitude", 3, 4, 0), ("phase", 5, 6, 1))}
    limits = plan["checks"]
    checks = {
        "amplitude_coefficients": differences["amplitude_coefficients"]["max_abs"]
        <= limits["max_eager_jit_amplitude_coefficients_absolute_error"],
        "phase_coefficients": differences["phase_coefficients"]["max_abs"]
        <= limits["max_eager_jit_phase_coefficients_absolute_error"],
        "corrected_mode": differences["corrected_mode"]["relative_l2"]
        <= limits["max_eager_jit_corrected_mode_relative_l2"],
        "matching_residual": max(value for item in residuals.values()
                                 for value in item.values())
        <= limits["max_matching_residual_relative_l2"],
    }
    result = {
        "schema": "diffgw.seobnrv4.jax-nqc-local-fit-jit.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "case": case, "fine_samples": len(run.fine_time),
        "match_time_M": float(match), "differences": differences,
        "residuals": residuals,
        "local_vs_full_spline": {
            "amplitude_coefficients_max_abs": max_abs(eager[0], baseline[0]),
            "phase_coefficients_max_abs": max_abs(eager[1], baseline[1]),
        },
        "checks": checks, "diagnostic_checks_passed": all(checks.values()),
        "elapsed_s": time.monotonic() - started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"differences": differences, "checks": checks,
                      "local_vs_full_spline": result["local_vs_full_spline"]},
                     sort_keys=True), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

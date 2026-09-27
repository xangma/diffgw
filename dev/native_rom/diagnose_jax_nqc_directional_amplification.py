"""Freeze the FP64 pre-NQC-mode amplification through JAX NQC attachment."""

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
    _unwrap_mode_phase_unchecked,
    full_orbital_peak,
    traceable_fixed_chart_attached_mode22,
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
from jaxwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-27/jax_nqc_directional_amplification_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_directional_amplification.json")


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def relative_residual(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def write(report):
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    expected_gates = {
        "max_pre_nqc_mode_relative_l2": 1e-12,
        "unchanged_whole_eager_jit_gate": 1e-10,
        "min_endpoint_whole_relative_l2": 1e-10,
        "min_whole_to_mode_amplification": 100000.0,
        "min_fraction_of_frozen_whole_gap": 0.5,
        "max_fraction_of_frozen_whole_gap": 1.5,
        "max_half_step_linearity_residual_fraction": 0.05,
    }
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-directional-amplification-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "high_start_mixed_spin"
            or plan["alphas"] != [-1.0, -0.5, 0.0, 0.5, 1.0]
            or plan["gates"] != expected_gates):
        raise ValueError("frozen directional-amplification plan changed")
    for name, digest in {**plan["pinned_evidence"], **plan["source_sha256"]}.items():
        if q.sha256(q.ROOT / name) != digest:
            raise ValueError(f"pinned input changed: {name}")
    pathwise = json.loads((q.ROOT / next(iter(plan["pinned_evidence"]))).read_text())
    frozen_whole_gap = pathwise["metrics"]["eager_jit_relative_l2"]
    if frozen_whole_gap != 1.5471380096030089e-9:
        raise ValueError("frozen whole-waveform discrepancy changed")

    report = dict(
        schema="diffgw.seobnrv4.jax-nqc-directional-amplification.v1",
        status="running", release_qualified=False, source_change=False,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        pinned_evidence_sha256=plan["pinned_evidence"],
        host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
    )
    write(report)

    case = plan["case"]
    m1, m2, chi1, chi2, _ = CASES[case]
    _, initial, _, field = construct_case(case)
    dt = plan["output_step_M"]
    run = integrate_two_stage_trajectory(
        field, initial, output_step=dt, max_duration=1000.,
        coarse_initial_step=dt, coarse_max_step=2., fine_max_step=2.,
        fine_refinement_factor=plan["fine_refinement_factor"],
        atol=plan["adaptive_atol"], rtol=plan["adaptive_rtol"])
    print("trajectory", run.coarse.accepted_steps, run.fine.accepted_steps,
          time.monotonic() - started, flush=True)

    nu, delta = m1 * m2 / (m1 + m2)**2, (m1 - m2) / (m1 + m2)
    ham = SEOBNRv4Hamiltonian(nu, chi1, chi2, delta=delta)
    flux = SEOBNRv4PublishedFlux(
        spin_truncation="tabulated", retain_pan_cubic_spin=False)
    fine, coarse = run.fine_state, run.coarse_state
    omega = jax.vmap(field)(fine)[:, 1]
    coarse_omega = jax.vmap(field)(coarse)[:, 1]
    eager_mode, _ = published_mode22_and_frequency(fine, ham, flux, **MODE_OPTIONS)
    jit_mode = jax.jit(
        lambda state: published_mode22_and_frequency(
            state, ham, flux, **MODE_OPTIONS)[0])(fine)
    coarse_mode, _ = published_mode22_and_frequency(
        coarse, ham, flux, **MODE_OPTIONS)
    nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    peak = full_orbital_peak(run.fine_time, omega).time
    match = peak + seobnrv4_nqc_time_offset_numerical_compatibility(
        nqc_nu, nqc_chi)
    targets = _seobnrv4_nqc_targets_unchecked(nqc_nu, nqc_chi)
    final_mass = eob_final_mass_fraction(m1, m2, chi1, chi2)
    final_spin = hofmann_final_spin(m1, m2, chi1, chi2, fit_order=(3, 4))
    qnm_omega, qnm_damping = evaluate_kerr_qnm_220(final_spin)
    sigma = (qnm_omega - 1j * qnm_damping) / final_mass
    output_time = jnp.arange(int(np.ceil((float(match) + 150.) / dt)) + 1) * dt

    def stages(mode):
        amplitude = jnp.abs(mode)
        phase = _unwrap_mode_phase_unchecked(mode)
        q_basis, p_basis = nqc_22_basis(fine[:, 0], fine[:, 2], omega)
        combined = jnp.concatenate((amplitude[:, None],
                                    amplitude[:, None] * q_basis,
                                    p_basis, phase[:, None]), axis=-1)
        spline = natural_cubic_spline(run.fine_time, combined, scalar_output=False)
        rows = (spline(match), spline.derivative(match),
                spline.derivative(match, 2))
        attached = traceable_fixed_chart_attached_mode22(
            run.fine_time, fine[:, 0], fine[:, 2], omega, mode, match,
            targets, nqc_nu, nqc_chi, sigma, output_time)
        corrected_coarse = coarse_mode * nqc_22_factor(
            coarse[:, 0], coarse[:, 2], coarse_omega,
            NQCCoefficients(attached[2], attached[3]))
        whole = jnp.concatenate((corrected_coarse[:run.restart_index], attached[0]))
        return whole, attached[2], attached[3], combined, *rows

    delta_mode = jit_mode - eager_mode
    outputs = {str(alpha): stages(eager_mode + alpha * delta_mode)
               for alpha in plan["alphas"]}
    reference = outputs["0.0"]
    ladder = {}
    for alpha in plan["alphas"]:
        current = outputs[str(alpha)]
        varied_mode = eager_mode + alpha * delta_mode
        ladder[str(alpha)] = dict(
            pre_nqc_mode_relative_l2=relative_l2(varied_mode, eager_mode),
            combined_input_max_abs=max_abs(current[3], reference[3]),
            matching_value_max_abs=max_abs(current[4], reference[4]),
            matching_first_max_abs=max_abs(current[5], reference[5]),
            matching_second_max_abs=max_abs(current[6], reference[6]),
            amplitude_coefficients_max_abs=max_abs(current[1], reference[1]),
            phase_coefficients_max_abs=max_abs(current[2], reference[2]),
            whole_relative_l2=relative_l2(current[0], reference[0]),
        )

    endpoint = outputs["1.0"][0] - reference[0]
    half = outputs["0.5"][0] - reference[0]
    opposite = outputs["-1.0"][0] - reference[0]
    endpoint_mode = ladder["1.0"]["pre_nqc_mode_relative_l2"]
    endpoint_whole = ladder["1.0"]["whole_relative_l2"]
    metrics = dict(
        frozen_pathwise_whole_eager_jit_relative_l2=frozen_whole_gap,
        endpoint_fraction_of_frozen_whole_gap=endpoint_whole / frozen_whole_gap,
        whole_to_mode_amplification=endpoint_whole / endpoint_mode,
        half_step_linearity_residual_fraction=relative_residual(2 * half, endpoint),
        odd_symmetry_residual_fraction=float(
            np.linalg.norm(np.asarray(endpoint + opposite)) /
            np.linalg.norm(np.asarray(endpoint))),
        frozen_match_time_M=float(match),
        coarse_accepted_steps=run.coarse.accepted_steps,
        fine_accepted_steps=run.fine.accepted_steps,
    )
    limits = plan["gates"]
    checks = dict(
        pre_nqc_precision=endpoint_mode <= limits["max_pre_nqc_mode_relative_l2"],
        reproduces_gate_failure=endpoint_whole >= limits["min_endpoint_whole_relative_l2"],
        amplification=metrics["whole_to_mode_amplification"] >=
        limits["min_whole_to_mode_amplification"],
        frozen_gap_fraction=(limits["min_fraction_of_frozen_whole_gap"] <=
                             metrics["endpoint_fraction_of_frozen_whole_gap"] <=
                             limits["max_fraction_of_frozen_whole_gap"]),
        directional_linearity=metrics["half_step_linearity_residual_fraction"] <=
        limits["max_half_step_linearity_residual_fraction"],
    )
    report.update(
        status="diagnostic_complete", case=case, ladder=ladder, metrics=metrics,
        checks=checks, diagnostic_checks_passed=all(checks.values()),
        exact_blocker=("No gate-safe repair is identified: roundoff-scale pre-NQC "
                       "mode perturbations are amplified above the unchanged 1e-10 "
                       "whole-waveform gate on the fixed chart, while that isolated "
                       "path explains only part of the frozen pathwise discrepancy."),
        next_action=("Preserve the failed 1e-10 release gate. The unresolved pathwise "
                     "remainder and exact historical numerical semantics must be "
                     "qualified before any repair; a changed NQC formulation requires "
                     "separate source qualification."),
        elapsed_s=time.monotonic() - started,
    )
    write(report)
    print(json.dumps({"checks": checks, "metrics": metrics}, sort_keys=True),
          flush=True)
    return 0 if report["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

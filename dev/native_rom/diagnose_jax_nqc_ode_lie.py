"""Dev-only continuous Lie-derivative NQC amplitude matching experiment."""

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
from .eob.jax.adaptive_driver_clean import replay_accepted_steps
from .eob.jax.numerical_attachment_clean import (
    full_orbital_peak,
    traceable_fixed_chart_attached_mode22,
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
from jaxwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_ode_lie_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_ode_lie.json")


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual)-np.asarray(reference))))


def as_list(value):
    return np.asarray(value).tolist()


def save(report):
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-ode-lie-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "high_start_mixed_spin"
            or plan["varied_initial_component"] != "radius"):
        raise ValueError("frozen ODE Lie-derivative plan changed")
    expected_gates = {
        "max_replay_state_absolute_error": 1e-7,
        "max_fixed_amplitude_coefficients_eager_jit_absolute_error": 1e-8,
        "max_fixed_corrected_fine_eager_jit_relative_l2": 1e-10,
        "max_fixed_amplitude_solve_residual_relative_l2": 1e-8,
        "max_whole_eager_jit_relative_l2": 1e-10,
        "max_ad_fd_relative_l2": 0.002,
        "min_gradient_l2": 1e-9,
    }
    if plan["gates"] != expected_gates:
        raise ValueError("frozen ODE Lie-derivative gates changed")
    report = dict(schema="diffgw.seobnrv4.jax-nqc-ode-lie.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  case=plan["case"])
    save(report)
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
    nu, delta = m1*m2/(m1+m2)**2, (m1-m2)/(m1+m2)
    ham = SEOBNRv4Hamiltonian(nu, chi1, chi2, delta=delta)
    flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated",
                                 retain_pan_cubic_spin=False)
    nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    baseline_omega = jax.vmap(field)(run.fine_state)[:, 1]
    peak = full_orbital_peak(run.fine_time, baseline_omega).time
    match = peak + seobnrv4_nqc_time_offset_numerical_compatibility(nqc_nu, nqc_chi)
    final_mass = eob_final_mass_fraction(m1, m2, chi1, chi2)
    final_spin = hofmann_final_spin(m1, m2, chi1, chi2, fit_order=(3, 4))
    qnm_omega, qnm_damping = evaluate_kerr_qnm_220(final_spin)
    sigma = (qnm_omega-1j*qnm_damping)/final_mass
    output_count = int(np.ceil((float(match)+150.)/dt))+1
    output_time = jnp.arange(output_count)*dt
    targets = _seobnrv4_nqc_targets_unchecked(nqc_nu, nqc_chi)

    def ode_amplitude_rows(center):
        def quantities(state):
            vector = field(state)
            local_mode, _ = published_mode22_and_frequency(
                state, ham, flux, **MODE_OPTIONS)
            amplitude = jnp.abs(local_mode)
            q_basis, _ = nqc_22_basis(state[0], state[2], vector[1])
            return jnp.concatenate((amplitude[None], amplitude*q_basis))

        def first_lie(state):
            return jax.jvp(quantities, (state,), (field(state),))[1]

        return jnp.stack((quantities(center), first_lie(center),
                          jax.jvp(first_lie, (center,), (field(center),))[1]))

    target_amplitude = jnp.stack(targets[:3])

    def fixed(fine, omega, mode, center):
        rows = ode_amplitude_rows(center)
        coefficients = traceable_nqc_22_full_series_coefficients(
            run.fine_time, fine[:, 0], fine[:, 2], omega, mode, match,
            targets, amplitude_rows=rows)
        a, b, matrix, _, _, _, residual, _ = coefficients
        corrected = mode * nqc_22_factor(
            fine[:, 0], fine[:, 2], omega, NQCCoefficients(a, b))
        return rows, a, b, corrected, matrix, target_amplitude-rows[:, 0], residual

    fixed_center = natural_cubic_spline(
        run.fine_time, run.fine_state, scalar_output=False).evaluate(match)
    fixed_mode, _ = published_mode22_and_frequency(
        run.fine_state, ham, flux, **MODE_OPTIONS)
    fixed_eager = fixed(run.fine_state, baseline_omega, fixed_mode, fixed_center)
    fixed_jit = jax.jit(fixed)(run.fine_state, baseline_omega, fixed_mode, fixed_center)
    spline_fixed = traceable_nqc_22_full_series_coefficients(
        run.fine_time, run.fine_state[:, 0], run.fine_state[:, 2],
        baseline_omega, fixed_mode, match, targets)
    spline_corrected = fixed_mode * nqc_22_factor(
        run.fine_state[:, 0], run.fine_state[:, 2], baseline_omega,
        NQCCoefficients(spline_fixed[0], spline_fixed[1]))
    fixed_metrics = dict(
        eager_raw_amplitude_match_rows=as_list(fixed_eager[0]),
        jit_raw_amplitude_match_rows=as_list(fixed_jit[0]),
        eager_jit_raw_amplitude_match_rows_max_abs=max_abs(fixed_eager[0], fixed_jit[0]),
        eager_amplitude_coefficients=as_list(fixed_eager[1]),
        jit_amplitude_coefficients=as_list(fixed_jit[1]),
        eager_jit_amplitude_coefficients_max_abs=max_abs(fixed_eager[1], fixed_jit[1]),
        eager_jit_corrected_fine_relative_l2=relative_l2(fixed_jit[3], fixed_eager[3]),
        eager_amplitude_solve_residual_relative_l2=relative_l2(
            fixed_eager[4] @ fixed_eager[1], fixed_eager[5]),
        jit_amplitude_solve_residual_relative_l2=relative_l2(
            fixed_jit[4] @ fixed_jit[1], fixed_jit[5]),
        eager_vs_spline_amplitude_coefficients_max_abs=max_abs(
            fixed_eager[1], spline_fixed[0]),
        eager_vs_spline_corrected_fine_relative_l2=relative_l2(
            fixed_eager[3], spline_corrected),
        spline_amplitude_coefficients=as_list(spline_fixed[0]),
        fixed_center=as_list(fixed_center))
    report["fixed_input"] = fixed_metrics
    report["status"] = "fixed_input_complete"
    save(report)
    print("fixed input", fixed_metrics["eager_jit_amplitude_coefficients_max_abs"],
          time.monotonic()-started, flush=True)

    def replay(radius_offset, *, use_ode=True):
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
        center = natural_cubic_spline(
            run.fine_time, fine, scalar_output=False).evaluate(match)
        rows = ode_amplitude_rows(center) if use_ode else None
        attached = traceable_fixed_chart_attached_mode22(
            run.fine_time, fine[:, 0], fine[:, 2], omega, mode, match,
            targets, nqc_nu, nqc_chi, sigma, output_time,
            amplitude_rows=rows)
        coarse_mode, _ = published_mode22_and_frequency(coarse, ham, flux, **MODE_OPTIONS)
        coarse_omega = jax.vmap(field)(coarse)[:, 1]
        corrected_coarse = coarse_mode * nqc_22_factor(
            coarse[:, 0], coarse[:, 2], coarse_omega,
            NQCCoefficients(attached[2], attached[3]))
        full_mode = jnp.concatenate((corrected_coarse[:run.restart_index],
                                     attached[0]))
        return (full_mode, coarse, fine, mode, omega,
                jnp.concatenate((attached[2], attached[3])),
                corrected_coarse, attached[1], center, rows if use_ode else jnp.zeros((3, 4)))

    origin = jnp.asarray(0., dtype=jnp.float64)
    (eager_mode, eager_coarse, eager_fine, eager_base, eager_omega,
     eager_coefficients, eager_corrected_coarse, eager_corrected_fine,
     eager_center, eager_rows) = replay(origin)
    replay_error = max(float(jnp.max(abs(eager_coarse-run.coarse_state))),
                       float(jnp.max(abs(eager_fine-run.fine_state))))
    print("eager replay", replay_error, time.monotonic()-started, flush=True)
    compiled = jax.jit(replay)
    (compiled_mode, compiled_coarse, compiled_fine, compiled_base, compiled_omega,
     compiled_coefficients, compiled_corrected_coarse, compiled_corrected_fine,
     compiled_center, compiled_rows) = compiled(origin)
    print("compiled replay", time.monotonic()-started, flush=True)
    spline_replay = replay(origin, use_ode=False)
    report["whole_pre_gradient"] = dict(
        eager_jit_relative_l2=relative_l2(compiled_mode, eager_mode),
        eager_jit_amplitude_coefficients_max_abs=max_abs(
            compiled_coefficients[:3], eager_coefficients[:3]),
        eager_jit_raw_amplitude_match_rows_max_abs=max_abs(compiled_rows, eager_rows),
        eager_jit_center_max_abs=max_abs(compiled_center, eager_center),
        eager_vs_spline_full_mode_relative_l2=relative_l2(eager_mode, spline_replay[0]),
        eager_vs_spline_corrected_fine_relative_l2=relative_l2(
            eager_corrected_fine, spline_replay[7]),
        eager_vs_spline_amplitude_coefficients_max_abs=max_abs(
            eager_coefficients[:3], spline_replay[5][:3]),
        eager_raw_amplitude_match_rows=as_list(eager_rows),
        jit_raw_amplitude_match_rows=as_list(compiled_rows),
        eager_center=as_list(eager_center), jit_center=as_list(compiled_center))
    report["status"] = "whole_comparison_complete"
    save(report)
    gradient = jax.jit(jax.jacfwd(lambda offset: replay(offset)[0]))(origin)
    print("compiled derivative", time.monotonic()-started, flush=True)
    fd_errors = {}
    for step in plan["finite_difference_steps"]:
        finite_difference = (compiled(step)[0]-compiled(-step)[0])/(2*step)
        fd_errors[str(step)] = relative_l2(gradient, finite_difference)
    limits = plan["gates"]
    metrics = dict(
        replay_state_absolute_error=replay_error,
        whole_eager_jit_relative_l2=relative_l2(compiled_mode, eager_mode),
        eager_jit_coarse_state_max_abs=float(jnp.max(abs(compiled_coarse-eager_coarse))),
        eager_jit_fine_state_max_abs=float(jnp.max(abs(compiled_fine-eager_fine))),
        eager_jit_base_mode_relative_l2=relative_l2(compiled_base, eager_base),
        eager_jit_orbital_frequency_max_abs=float(jnp.max(abs(compiled_omega-eager_omega))),
        eager_jit_nqc_coefficients_max_abs=float(jnp.max(abs(
            compiled_coefficients-eager_coefficients))),
        eager_jit_corrected_coarse_relative_l2=relative_l2(
            compiled_corrected_coarse, eager_corrected_coarse),
        eager_jit_mode_max_abs=float(jnp.max(abs(compiled_mode-eager_mode))),
        eager_jit_prefix_relative_l2=relative_l2(
            compiled_mode[:run.restart_index], eager_mode[:run.restart_index]),
        eager_jit_attachment_relative_l2=relative_l2(
            compiled_mode[run.restart_index:], eager_mode[run.restart_index:]),
        gradient_l2=float(jnp.linalg.norm(gradient)),
        gradient_finite=bool(jnp.all(jnp.isfinite(gradient))),
        ad_fd_relative_l2=fd_errors,
        coarse_accepted_steps=run.coarse.accepted_steps,
        fine_accepted_steps=run.fine.accepted_steps,
        coarse_samples=len(run.coarse_time), fine_samples=len(run.fine_time),
        output_samples=output_count+run.restart_index,
        frozen_peak_time_M=float(peak),
        frozen_match_time_M=float(match))
    checks = dict(
        replay=metrics["replay_state_absolute_error"] <= limits["max_replay_state_absolute_error"],
        fixed_coefficients=(fixed_metrics["eager_jit_amplitude_coefficients_max_abs"] <=
                            limits["max_fixed_amplitude_coefficients_eager_jit_absolute_error"]),
        fixed_corrected_fine=(fixed_metrics["eager_jit_corrected_fine_relative_l2"] <=
                              limits["max_fixed_corrected_fine_eager_jit_relative_l2"]),
        fixed_solve_residual=(max(fixed_metrics["eager_amplitude_solve_residual_relative_l2"],
                                  fixed_metrics["jit_amplitude_solve_residual_relative_l2"]) <=
                              limits["max_fixed_amplitude_solve_residual_relative_l2"]),
        whole_eager_jit=(metrics["whole_eager_jit_relative_l2"] <=
                         limits["max_whole_eager_jit_relative_l2"]),
        derivative=metrics["gradient_finite"] and metrics["gradient_l2"] >= limits["min_gradient_l2"],
        finite_difference=max(fd_errors.values()) <= limits["max_ad_fd_relative_l2"])
    report.update(status="diagnostic_complete", metrics=metrics, checks=checks,
                  diagnostic_checks_passed=all(checks.values()),
                  elapsed_s=time.monotonic()-started)
    save(report)
    print(json.dumps(report["metrics"], sort_keys=True), flush=True)
    return 0 if report["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

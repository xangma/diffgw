"""Dev-only initial-state derivative through two-stage JAX whole waveform."""

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
)
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import (
    _seobnrv4_nqc_targets_unchecked,
    NQCCoefficients,
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

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_pathwise_whole_gradient_plan.json"
OUTPUT = PLAN.with_name("jax_pathwise_whole_gradient.json")


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-pathwise-whole-gradient-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "high_start_mixed_spin"
            or plan["varied_initial_component"] != "radius"):
        raise ValueError("frozen pathwise gradient plan changed")
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

    def replay(radius_offset):
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
        attached = traceable_fixed_chart_attached_mode22(
            run.fine_time, fine[:, 0], fine[:, 2], omega, mode, match,
            targets, nqc_nu, nqc_chi, sigma, output_time)
        coarse_mode, _ = published_mode22_and_frequency(coarse, ham, flux, **MODE_OPTIONS)
        coarse_omega = jax.vmap(field)(coarse)[:, 1]
        corrected_coarse = coarse_mode * nqc_22_factor(
            coarse[:, 0], coarse[:, 2], coarse_omega,
            NQCCoefficients(attached[2], attached[3]))
        full_mode = jnp.concatenate((corrected_coarse[:run.restart_index],
                                     attached[0]))
        return full_mode, coarse, fine, mode, omega, jnp.concatenate((attached[2],
                                                                       attached[3])), corrected_coarse

    origin = jnp.asarray(0., dtype=jnp.float64)
    (eager_mode, eager_coarse, eager_fine, eager_base, eager_omega,
     eager_coefficients, eager_corrected_coarse) = replay(origin)
    replay_error = max(float(jnp.max(abs(eager_coarse-run.coarse_state))),
                       float(jnp.max(abs(eager_fine-run.fine_state))))
    print("eager replay", replay_error, time.monotonic()-started, flush=True)
    compiled = jax.jit(replay)
    (compiled_mode, compiled_coarse, compiled_fine, compiled_base, compiled_omega,
     compiled_coefficients, compiled_corrected_coarse) = compiled(origin)
    print("compiled replay", time.monotonic()-started, flush=True)
    gradient = jax.jit(jax.jacfwd(lambda offset: replay(offset)[0]))(origin)
    print("compiled derivative", time.monotonic()-started, flush=True)
    fd_errors = {}
    for step in plan["finite_difference_steps"]:
        finite_difference = (compiled(step)[0]-compiled(-step)[0])/(2*step)
        fd_errors[str(step)] = relative_l2(gradient, finite_difference)
    limits = plan["gates"]
    metrics = dict(
        replay_state_absolute_error=replay_error,
        eager_jit_relative_l2=relative_l2(compiled_mode, eager_mode),
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
        eager_jit=metrics["eager_jit_relative_l2"] <= limits["max_eager_jit_relative_l2"],
        derivative=metrics["gradient_finite"] and metrics["gradient_l2"] >= limits["min_gradient_l2"],
        finite_difference=max(fd_errors.values()) <= limits["max_ad_fd_relative_l2"])
    result = dict(schema="diffgw.seobnrv4.jax-pathwise-whole-gradient.v1",
                  status="diagnostic_complete", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  case=case, metrics=metrics, checks=checks,
                  diagnostic_checks_passed=all(checks.values()),
                  elapsed_s=time.monotonic()-started)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result["metrics"], sort_keys=True), flush=True)
    return 0 if result["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded no-AD attribution of JAX NQC replay and attachment differences."""

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
    full_orbital_peak, traceable_fixed_chart_attached_mode22,
)
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import (
    _seobnrv4_nqc_targets_unchecked, NQCCoefficients, nqc_22_basis,
    nqc_22_factor, seobnrv4_nqc_parameters,
    seobnrv4_nqc_time_offset_numerical_compatibility,
)
from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency
from jaxwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_whole_attribution_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_whole_attribution.json")


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual)-np.asarray(reference))))


def write(report):
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-whole-attribution-plan.v1"
            or plan["case"] != "high_start_mixed_spin"
            or plan["release_qualified"] is not False
            or plan["gates"] != {
                "max_replay_state_absolute_error": 1e-7,
                "max_whole_eager_jit_relative_l2": 1e-10}):
        raise ValueError("frozen whole attribution plan changed")
    report = dict(schema="diffgw.seobnrv4.jax-nqc-whole-attribution.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid())
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
          time.monotonic()-started, flush=True)
    nu, delta = m1*m2/(m1+m2)**2, (m1-m2)/(m1+m2)
    ham = SEOBNRv4Hamiltonian(nu, chi1, chi2, delta=delta)
    flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated",
                                 retain_pan_cubic_spin=False)
    nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    baseline_omega = jax.vmap(field)(run.fine_state)[:, 1]
    peak = full_orbital_peak(run.fine_time, baseline_omega).time
    match = peak + seobnrv4_nqc_time_offset_numerical_compatibility(
        nqc_nu, nqc_chi)
    final_mass = eob_final_mass_fraction(m1, m2, chi1, chi2)
    final_spin = hofmann_final_spin(m1, m2, chi1, chi2, fit_order=(3, 4))
    qnm_omega, qnm_damping = evaluate_kerr_qnm_220(final_spin)
    sigma = (qnm_omega-1j*qnm_damping)/final_mass
    output_count = int(np.ceil((float(match)+150.)/dt))+1
    output_time = jnp.arange(output_count)*dt
    targets = _seobnrv4_nqc_targets_unchecked(nqc_nu, nqc_chi)
    inspiral_mask = np.asarray(output_time) < float(match)

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

    def trajectory(radius_offset):
        first = initial.at[0].add(radius_offset)
        coarse_knots = replay_accepted_steps(
            lambda _time, state: field(state), first, run.coarse.time)
        coarse = natural_cubic_spline(run.coarse.time, coarse_knots,
                                      scalar_output=False).evaluate(run.coarse_time)
        fine_knots = replay_accepted_steps(
            lambda _time, state: field(state), coarse[run.restart_index],
            run.fine.time)
        fine = natural_cubic_spline(run.fine.time, fine_knots,
                                    scalar_output=False).evaluate(run.fine_time)
        omega = jax.vmap(field)(fine)[:, 1]
        mode, _ = published_mode22_and_frequency(fine, ham, flux,
                                                  **MODE_OPTIONS)
        coarse_mode, _ = published_mode22_and_frequency(
            coarse, ham, flux, **MODE_OPTIONS)
        coarse_omega = jax.vmap(field)(coarse)[:, 1]
        center = natural_cubic_spline(run.fine_time, fine,
                                      scalar_output=False).evaluate(match)
        rows = ode_amplitude_rows(center)
        return (coarse, fine, omega, mode, coarse_mode, coarse_omega,
                center, rows)

    def assemble(coarse, fine, omega, mode, coarse_mode, coarse_omega,
                 center, rows, *, use_ode=True):
        del center
        attached = traceable_fixed_chart_attached_mode22(
            run.fine_time, fine[:, 0], fine[:, 2], omega, mode, match,
            targets, nqc_nu, nqc_chi, sigma, output_time,
            amplitude_rows=rows if use_ode else None)
        corrected_coarse = coarse_mode * nqc_22_factor(
            coarse[:, 0], coarse[:, 2], coarse_omega,
            NQCCoefficients(attached[2], attached[3]))
        whole = jnp.concatenate((corrected_coarse[:run.restart_index],
                                 attached[0]))
        return (whole, attached[0], corrected_coarse, attached[1],
                attached[2], attached[3])

    def stage_difference(actual, reference):
        attachment_actual = np.asarray(actual[1])
        attachment_reference = np.asarray(reference[1])
        return dict(
            whole_relative_l2=relative_l2(actual[0], reference[0]),
            whole_max_abs=max_abs(actual[0], reference[0]),
            corrected_coarse_relative_l2=relative_l2(actual[2], reference[2]),
            corrected_fine_relative_l2=relative_l2(actual[3], reference[3]),
            prefix_relative_l2=relative_l2(
                np.asarray(actual[0])[:run.restart_index],
                np.asarray(reference[0])[:run.restart_index]),
            attachment_inspiral_relative_l2=relative_l2(
                attachment_actual[inspiral_mask],
                attachment_reference[inspiral_mask]),
            ringdown_relative_l2=relative_l2(
                attachment_actual[~inspiral_mask],
                attachment_reference[~inspiral_mask]),
            amplitude_coefficients_max_abs=max_abs(actual[4], reference[4]),
            phase_coefficients_max_abs=max_abs(actual[5], reference[5]))

    origin = jnp.asarray(0., dtype=jnp.float64)
    eager_traj = trajectory(origin)
    jit_traj = jax.jit(trajectory)(origin)
    replay_error = max(max_abs(eager_traj[0], run.coarse_state),
                       max_abs(eager_traj[1], run.fine_state))
    report["trajectory"] = dict(
        replay_state_absolute_error=replay_error,
        eager_jit_coarse_state_max_abs=max_abs(eager_traj[0], jit_traj[0]),
        eager_jit_fine_state_max_abs=max_abs(eager_traj[1], jit_traj[1]),
        eager_jit_orbital_frequency_max_abs=max_abs(eager_traj[2], jit_traj[2]),
        eager_jit_base_mode_relative_l2=relative_l2(eager_traj[3], jit_traj[3]),
        eager_jit_match_center_max_abs=max_abs(eager_traj[6], jit_traj[6]),
        eager_jit_lie_rows_max_abs=max_abs(eager_traj[7], jit_traj[7]),
        frozen_match_time_M=float(match),
        coarse_accepted_steps=run.coarse.accepted_steps,
        fine_accepted_steps=run.fine.accepted_steps)
    report["status"] = "trajectory_complete"
    write(report)
    print("replay arrays", time.monotonic()-started, flush=True)

    eager_on_eager = assemble(*eager_traj)
    compiled_assemble = jax.jit(assemble)
    jit_on_eager = compiled_assemble(*eager_traj)
    eager_on_jit = assemble(*jit_traj)
    jit_on_jit = compiled_assemble(*jit_traj)
    report["two_by_two"] = dict(
        compiler_on_identical_eager_arrays=stage_difference(
            jit_on_eager, eager_on_eager),
        compiler_on_identical_jit_arrays=stage_difference(
            jit_on_jit, eager_on_jit),
        trajectory_input_effect_eager_assembly=stage_difference(
            eager_on_jit, eager_on_eager),
        trajectory_input_effect_jit_assembly=stage_difference(
            jit_on_jit, jit_on_eager))
    report["status"] = "two_by_two_complete"
    write(report)
    print("two by two", time.monotonic()-started, flush=True)

    fused = jax.jit(lambda offset: assemble(*trajectory(offset)))(origin)
    report["fused"] = dict(
        fused_vs_eager=stage_difference(fused, eager_on_eager),
        fused_vs_separate_jit=stage_difference(fused, jit_on_jit))
    report["status"] = "fused_complete"
    write(report)
    print("fused", time.monotonic()-started, flush=True)

    spline = assemble(*eager_traj, use_ode=False)
    report["source_drift"] = stage_difference(eager_on_eager, spline)
    limits = plan["gates"]
    report["checks"] = dict(
        replay=replay_error <= limits["max_replay_state_absolute_error"],
        whole_eager_jit=(report["fused"]["fused_vs_eager"]["whole_relative_l2"] <=
                         limits["max_whole_eager_jit_relative_l2"]))
    report.update(status="diagnostic_complete",
                  diagnostic_checks_passed=all(report["checks"].values()),
                  elapsed_s=time.monotonic()-started)
    write(report)
    print(json.dumps(report["fused"], sort_keys=True), flush=True)
    return 0 if report["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

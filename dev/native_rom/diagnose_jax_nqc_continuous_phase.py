"""Frozen no-AD continuous-phase NQC control on one fixed JAX chart."""

from __future__ import annotations

import json
import os
import platform
import subprocess
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
    _seobnrv4_nqc_targets_unchecked, NQCCoefficients, nqc_22_factor,
    seobnrv4_nqc_parameters, seobnrv4_nqc_time_offset_numerical_compatibility,
)
from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency
from jaxwave.approximants.eob.remnant import eob_final_mass_fraction, hofmann_final_spin
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_continuous_phase_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_continuous_phase.json")


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def save(report):
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=q.ROOT,
        text=True).strip()
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-continuous-phase-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_head"] != head
            or plan["case"] != "high_start_mixed_spin"
            or plan["gates"] != {
                "max_replay_state_absolute_error": 1e-7,
                "max_candidate_baseline_whole_relative_l2": 1e-10,
                "max_candidate_whole_eager_jit_relative_l2": 1e-10}):
        raise ValueError("frozen continuous-phase plan changed")
    report = dict(schema="diffgw.seobnrv4.jax-nqc-continuous-phase.v1",
                  status="running", release_qualified=False,
                  source_head=head, plan_sha256=q.sha256(PLAN),
                  script_sha256=q.sha256(__file__), host=platform.node(),
                  cwd=os.getcwd(), pid=os.getpid())
    save(report)

    m1, m2, chi1, chi2, _ = CASES[plan["case"]]
    _, initial, _, field = construct_case(plan["case"])
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
    flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated",
                                 retain_pan_cubic_spin=False)
    nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    omega = jax.vmap(field)(run.fine_state)[:, 1]
    peak = full_orbital_peak(run.fine_time, omega).time
    match = peak + seobnrv4_nqc_time_offset_numerical_compatibility(
        nqc_nu, nqc_chi)
    final_mass = eob_final_mass_fraction(m1, m2, chi1, chi2)
    final_spin = hofmann_final_spin(m1, m2, chi1, chi2, fit_order=(3, 4))
    qnm_omega, qnm_damping = evaluate_kerr_qnm_220(final_spin)
    sigma = (qnm_omega - 1j * qnm_damping) / final_mass
    output_time = jnp.arange(int(np.ceil((float(match) + 150.) / dt)) + 1) * dt
    targets = _seobnrv4_nqc_targets_unchecked(nqc_nu, nqc_chi)

    def phase_rows(center):
        def frequency(state):
            return published_mode22_and_frequency(
                state, ham, flux, **MODE_OPTIONS)[1]

        return jnp.stack((frequency(center),
                          jax.jvp(frequency, (center,), (field(center),))[1]))

    def trajectory(offset):
        first = initial.at[0].add(offset)
        coarse_knots = replay_accepted_steps(
            lambda _time, state: field(state), first, run.coarse.time)
        coarse = natural_cubic_spline(run.coarse.time, coarse_knots,
                                      scalar_output=False).evaluate(run.coarse_time)
        fine_knots = replay_accepted_steps(
            lambda _time, state: field(state), coarse[run.restart_index],
            run.fine.time)
        fine = natural_cubic_spline(run.fine.time, fine_knots,
                                    scalar_output=False).evaluate(run.fine_time)
        fine_omega = jax.vmap(field)(fine)[:, 1]
        mode, _ = published_mode22_and_frequency(fine, ham, flux,
                                                  **MODE_OPTIONS)
        coarse_mode, _ = published_mode22_and_frequency(
            coarse, ham, flux, **MODE_OPTIONS)
        coarse_omega = jax.vmap(field)(coarse)[:, 1]
        center = natural_cubic_spline(run.fine_time, fine,
                                      scalar_output=False).evaluate(match)
        return coarse, fine, fine_omega, mode, coarse_mode, coarse_omega, center

    def assemble(arrays, *, continuous):
        coarse, fine, fine_omega, mode, coarse_mode, coarse_omega, center = arrays
        rows = phase_rows(center) if continuous else None
        attached = traceable_fixed_chart_attached_mode22(
            run.fine_time, fine[:, 0], fine[:, 2], fine_omega, mode, match,
            targets, nqc_nu, nqc_chi, sigma, output_time,
            phase_derivative_rows=rows)
        corrected_coarse = coarse_mode * nqc_22_factor(
            coarse[:, 0], coarse[:, 2], coarse_omega,
            NQCCoefficients(attached[2], attached[3]))
        whole = jnp.concatenate((corrected_coarse[:run.restart_index],
                                 attached[0]))
        return whole, attached[0], attached[1], attached[2], attached[3], rows

    zero = jnp.asarray(0., dtype=jnp.float64)
    eager_arrays = trajectory(zero)
    jit_arrays = jax.jit(trajectory)(zero)
    replay_error = max(max_abs(eager_arrays[0], run.coarse_state),
                       max_abs(eager_arrays[1], run.fine_state))
    report["replay"] = dict(state_max_abs=replay_error,
                            eager_jit_state_max_abs=max(
                                max_abs(eager_arrays[0], jit_arrays[0]),
                                max_abs(eager_arrays[1], jit_arrays[1])),
                            match_center_max_abs=max_abs(
                                eager_arrays[-1], jit_arrays[-1]),
                            match_time_M=float(match))
    report["status"] = "replay_complete"
    save(report)
    print("replay", time.monotonic() - started, flush=True)

    baseline = assemble(eager_arrays, continuous=False)
    candidate = assemble(eager_arrays, continuous=True)
    compiled = jax.jit(lambda arrays: assemble(arrays, continuous=True))
    jit_same = compiled(eager_arrays)
    jit_replay = compiled(jit_arrays)
    report["fixed_input"] = dict(
        baseline_whole_relative_l2=relative_l2(candidate[0], baseline[0]),
        baseline_attachment_relative_l2=relative_l2(candidate[1], baseline[1]),
        baseline_corrected_fine_relative_l2=relative_l2(candidate[2], baseline[2]),
        baseline_phase_coefficients_max_abs=max_abs(candidate[4], baseline[4]),
        compiler_same_input_whole_relative_l2=relative_l2(jit_same[0], candidate[0]),
        compiler_same_input_phase_coefficients_max_abs=max_abs(jit_same[4], candidate[4]),
        continuous_phase_rows=np.asarray(candidate[5]).tolist(),
        candidate_phase_coefficients=np.asarray(candidate[4]).tolist(),
        baseline_phase_coefficients=np.asarray(baseline[4]).tolist())
    report["whole"] = dict(
        eager_jit_relative_l2=relative_l2(jit_replay[0], candidate[0]),
        eager_jit_attachment_relative_l2=relative_l2(jit_replay[1], candidate[1]),
        eager_jit_corrected_fine_relative_l2=relative_l2(jit_replay[2], candidate[2]),
        eager_jit_amplitude_coefficients_max_abs=max_abs(jit_replay[3], candidate[3]),
        eager_jit_phase_coefficients_max_abs=max_abs(jit_replay[4], candidate[4]),
        eager_jit_phase_rows_max_abs=max_abs(jit_replay[5], candidate[5]))
    gates = plan["gates"]
    report["checks"] = dict(
        replay=replay_error <= gates["max_replay_state_absolute_error"],
        source_preservation=(report["fixed_input"]["baseline_whole_relative_l2"] <=
                             gates["max_candidate_baseline_whole_relative_l2"]),
        whole_eager_jit=(report["whole"]["eager_jit_relative_l2"] <=
                         gates["max_candidate_whole_eager_jit_relative_l2"]))
    report.update(status="diagnostic_complete",
                  diagnostic_checks_passed=all(report["checks"].values()),
                  elapsed_s=time.monotonic() - started)
    save(report)
    print(json.dumps({"fixed_input": report["fixed_input"],
                      "whole": report["whole"], "checks": report["checks"]},
                     sort_keys=True), flush=True)
    return 0 if report["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

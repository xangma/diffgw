"""Dev-only direct published phase control for JAX NQC and attachment."""

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
    _unwrap_mode_phase_unchecked,
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
from jaxwave.approximants.eob.seobnrv4_mode22 import (
    seobnrv4_published_delta22,
    seobnrv4_published_phase22,
)
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_direct_phase_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_direct_phase.json")


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual)-np.asarray(reference))))


def save(report):
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-direct-phase-plan.v1"
            or plan["case"] != "high_start_mixed_spin"
            or plan["source_head"] != "1943fff"
            or plan["release_qualified"] is not False):
        raise ValueError("frozen direct-phase plan changed")
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=q.ROOT,
        text=True).strip()
    if head != plan["source_head"]:
        raise ValueError(f"frozen source HEAD {plan['source_head']} != {head}")
    report = dict(schema="diffgw.seobnrv4.jax-nqc-direct-phase.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  source_head=head,
                  source_sha256={
                      name: q.sha256(q.ROOT / name) for name in (
                          "src/jaxwave/approximants/eob/published_waveform.py",
                          "src/jaxwave/approximants/eob/seobnrv4_mode22.py",
                          "dev/native_rom/eob/jax/numerical_attachment_clean.py")},
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid())
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

    def direct_phase(state, mode):
        quantities = flux.from_hamiltonian(state, ham)
        residual = seobnrv4_published_delta22(
            **{key: quantities[key] for key in
               ("omega", "hreal_over_M", "x1", "x2", "chi1", "chi2")},
            spin_v6=MODE_OPTIONS["spin_v6"],
            include_pan_v8=MODE_OPTIONS["include_pan_v8"],
            include_positive_chi_540=MODE_OPTIONS["include_positive_chi_540"])
        phase = seobnrv4_published_phase22(
            quantities["omega"], state[:, 1],
            hreal_over_M=quantities["hreal_over_M"], residual_phase=residual)
        initial_branch = -jnp.angle(mode[:1])
        gauge = 2*jnp.pi*jnp.round((initial_branch-phase[:1])/(2*jnp.pi))
        return phase+gauge

    def phase_rhs(phase):
        spline = natural_cubic_spline(run.fine_time, phase,
                                      scalar_output=True)
        return jnp.stack((spline.derivative(match)-targets.frequency,
                          spline.derivative(match, 2)-targets.frequency_derivative))

    def fixed_candidate(fine, omega, mode):
        phase = direct_phase(fine, mode)
        attached = traceable_fixed_chart_attached_mode22(
            run.fine_time, fine[:, 0], fine[:, 2], omega, mode, match,
            targets, nqc_nu, nqc_chi, sigma, output_time,
            phase_series=phase)
        return attached, phase_rhs(phase), phase

    fixed_mode, _ = published_mode22_and_frequency(
        run.fine_state, ham, flux, **MODE_OPTIONS)
    fixed_eager = fixed_candidate(run.fine_state, baseline_omega, fixed_mode)
    fixed_jit = jax.jit(fixed_candidate)(
        run.fine_state, baseline_omega, fixed_mode)
    baseline_phase = _unwrap_mode_phase_unchecked(fixed_mode)
    fixed_source = traceable_fixed_chart_attached_mode22(
        run.fine_time, run.fine_state[:, 0], run.fine_state[:, 2],
        baseline_omega, fixed_mode, match, targets, nqc_nu, nqc_chi,
        sigma, output_time)
    baseline_rhs = phase_rhs(baseline_phase)
    limits = plan["gates"]
    fixed_metrics = dict(
        direct_phase_mode_max_abs=float(jnp.max(jnp.abs(
            jnp.abs(fixed_mode)*jnp.exp(-1j*fixed_eager[2])-fixed_mode))),
        direct_minus_unwrapped_phase_max_abs=max_abs(
            fixed_eager[2], baseline_phase),
        source_phase_rhs_max_abs=max_abs(fixed_eager[1], baseline_rhs),
        source_phase_coefficients_max_abs=max_abs(
            fixed_eager[0][3], fixed_source[3]),
        source_corrected_fine_relative_l2=relative_l2(
            fixed_eager[0][1], fixed_source[1]),
        source_attachment_relative_l2=relative_l2(
            fixed_eager[0][0], fixed_source[0]),
        eager_jit_phase_rhs_max_abs=max_abs(fixed_jit[1], fixed_eager[1]),
        eager_jit_phase_coefficients_max_abs=max_abs(
            fixed_jit[0][3], fixed_eager[0][3]),
        eager_jit_corrected_fine_relative_l2=relative_l2(
            fixed_jit[0][1], fixed_eager[0][1]),
        eager_phase_rhs=np.asarray(fixed_eager[1]).tolist(),
        jit_phase_rhs=np.asarray(fixed_jit[1]).tolist(),
        source_phase_rhs=np.asarray(baseline_rhs).tolist(),
        eager_phase_coefficients=np.asarray(fixed_eager[0][3]).tolist(),
        jit_phase_coefficients=np.asarray(fixed_jit[0][3]).tolist(),
        source_phase_coefficients=np.asarray(fixed_source[3]).tolist())
    fixed_checks = dict(
        direct_phase_mode=(fixed_metrics["direct_phase_mode_max_abs"] <=
                           limits["max_fixed_direct_phase_mode_absolute_error"]),
        source_phase_rhs=(fixed_metrics["source_phase_rhs_max_abs"] <=
                          limits["max_fixed_source_phase_rhs_absolute_error"]),
        source_phase_coefficients=(fixed_metrics["source_phase_coefficients_max_abs"] <=
                                   limits["max_fixed_source_phase_coefficients_absolute_error"]),
        source_corrected_fine=(fixed_metrics["source_corrected_fine_relative_l2"] <=
                               limits["max_fixed_source_corrected_fine_relative_l2"]),
        source_attachment=(fixed_metrics["source_attachment_relative_l2"] <=
                           limits["max_fixed_source_attachment_relative_l2"]),
        eager_jit_phase_rhs=(fixed_metrics["eager_jit_phase_rhs_max_abs"] <=
                             limits["max_fixed_eager_jit_phase_rhs_absolute_error"]),
        eager_jit_phase_coefficients=(fixed_metrics["eager_jit_phase_coefficients_max_abs"] <=
                                      limits["max_fixed_eager_jit_phase_coefficients_absolute_error"]),
        eager_jit_corrected_fine=(fixed_metrics["eager_jit_corrected_fine_relative_l2"] <=
                                  limits["max_fixed_eager_jit_corrected_fine_relative_l2"]))
    report.update(status="fixed_input_complete", fixed_input=fixed_metrics,
                  fixed_checks=fixed_checks)
    save(report)
    print("fixed input", json.dumps(fixed_metrics, sort_keys=True),
          time.monotonic()-started, flush=True)
    if not all(value for key, value in fixed_checks.items()
               if key.startswith("source_") or key == "direct_phase_mode"):
        report.update(status="stopped_fixed_source_drift",
                      diagnostic_checks_passed=False,
                      ad_fd_status="not_run_source_drift_stop_rule",
                      elapsed_s=time.monotonic()-started)
        save(report)
        return 1

    def replay(radius_offset, *, use_direct=True):
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
        phase = (direct_phase(fine, mode) if use_direct else
                 _unwrap_mode_phase_unchecked(mode))
        attached = traceable_fixed_chart_attached_mode22(
            run.fine_time, fine[:, 0], fine[:, 2], omega, mode, match,
            targets, nqc_nu, nqc_chi, sigma, output_time,
            phase_series=phase if use_direct else None)
        coarse_mode, _ = published_mode22_and_frequency(
            coarse, ham, flux, **MODE_OPTIONS)
        coarse_omega = jax.vmap(field)(coarse)[:, 1]
        corrected_coarse = coarse_mode*nqc_22_factor(
            coarse[:, 0], coarse[:, 2], coarse_omega,
            NQCCoefficients(attached[2], attached[3]))
        whole = jnp.concatenate((corrected_coarse[:run.restart_index],
                                 attached[0]))
        return (whole, coarse, fine, mode, attached[1], attached[2],
                attached[3], phase_rhs(phase))

    origin = jnp.asarray(0., dtype=jnp.float64)
    eager = replay(origin)
    compiled = jax.jit(replay)(origin)
    source = replay(origin, use_direct=False)
    replay_error = max(max_abs(eager[1], run.coarse_state),
                       max_abs(eager[2], run.fine_state))
    whole_metrics = dict(
        replay_state_absolute_error=replay_error,
        eager_jit_coarse_state_max_abs=max_abs(eager[1], compiled[1]),
        eager_jit_fine_state_max_abs=max_abs(eager[2], compiled[2]),
        eager_jit_base_mode_relative_l2=relative_l2(eager[3], compiled[3]),
        source_phase_rhs_max_abs=max_abs(eager[7], source[7]),
        source_phase_coefficients_max_abs=max_abs(eager[6], source[6]),
        source_relative_l2=relative_l2(eager[0], source[0]),
        source_corrected_fine_relative_l2=relative_l2(eager[4], source[4]),
        eager_jit_phase_rhs_max_abs=max_abs(eager[7], compiled[7]),
        eager_jit_phase_coefficients_max_abs=max_abs(eager[6], compiled[6]),
        eager_jit_amplitude_coefficients_max_abs=max_abs(eager[5], compiled[5]),
        eager_jit_relative_l2=relative_l2(eager[0], compiled[0]),
        eager_jit_corrected_fine_relative_l2=relative_l2(eager[4], compiled[4]),
        eager_phase_rhs=np.asarray(eager[7]).tolist(),
        jit_phase_rhs=np.asarray(compiled[7]).tolist(),
        source_phase_rhs=np.asarray(source[7]).tolist(),
        eager_phase_coefficients=np.asarray(eager[6]).tolist(),
        jit_phase_coefficients=np.asarray(compiled[6]).tolist(),
        source_phase_coefficients=np.asarray(source[6]).tolist(),
        frozen_match_time_M=float(match))
    whole_checks = dict(
        replay=replay_error <= limits["max_replay_state_absolute_error"],
        source_phase_rhs=whole_metrics["source_phase_rhs_max_abs"] <=
        limits["max_whole_source_phase_rhs_absolute_error"],
        source_phase_coefficients=whole_metrics["source_phase_coefficients_max_abs"] <=
        limits["max_whole_source_phase_coefficients_absolute_error"],
        source_mode=whole_metrics["source_relative_l2"] <=
        limits["max_whole_source_relative_l2"],
        eager_jit_phase_rhs=whole_metrics["eager_jit_phase_rhs_max_abs"] <=
        limits["max_whole_eager_jit_phase_rhs_absolute_error"],
        eager_jit_phase_coefficients=whole_metrics["eager_jit_phase_coefficients_max_abs"] <=
        limits["max_whole_eager_jit_phase_coefficients_absolute_error"],
        eager_jit_mode=whole_metrics["eager_jit_relative_l2"] <=
        limits["max_whole_eager_jit_relative_l2"])
    report.update(status="diagnostic_complete", whole=whole_metrics,
                  whole_checks=whole_checks,
                  diagnostic_checks_passed=(all(fixed_checks.values()) and
                                            all(whole_checks.values())),
                  ad_fd_status="not_run_prior_ad_compilation_exceeded_32_GiB_operational_cap",
                  elapsed_s=time.monotonic()-started)
    save(report)
    print("whole", json.dumps(whole_metrics, sort_keys=True), flush=True)
    return 0 if report["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

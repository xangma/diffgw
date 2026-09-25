"""Frozen dev-only fixed-state published-mode and NQC attribution."""

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
from .diagnose_jax_full_spline_arithmetic import stable_resample
from .diagnose_jax_two_stage_trajectory import construct_case
from .eob.jax.adaptive_driver_clean import replay_accepted_steps
from .eob.jax.numerical_attachment_clean import (
    full_orbital_peak, traceable_nqc_22_full_series_coefficients,
)
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import (
    _seobnrv4_nqc_targets_unchecked, seobnrv4_nqc_parameters,
    seobnrv4_nqc_time_offset_numerical_compatibility,
)
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_fixed_state_mode_plan.json"
OUTPUT = PLAN.with_name("jax_fixed_state_mode.json")


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def comparison(actual, reference):
    return {"max_abs": max_abs(actual, reference),
            "relative_l2": relative_l2(actual, reference)}


def save(report):
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=q.ROOT,
        text=True).strip()
    if (plan["schema"] != "diffgw.seobnrv4.jax-fixed-state-mode-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_head"] != head
            or plan["case"] != "high_start_mixed_spin"
            or plan["gates"] != {
                "max_replay_state_absolute_error": 1e-7,
                "max_fixed_input_mode_relative_l2": 1e-12,
                "max_fixed_input_frequency_relative_l2": 1e-12,
                "max_fixed_input_nqc_coefficient_absolute_error": 1e-9}):
        raise ValueError("frozen fixed-state mode plan changed")
    report = dict(schema="diffgw.seobnrv4.jax-fixed-state-mode.v1",
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

    def replay(offset):
        first = initial.at[0].add(offset)
        coarse_knots = replay_accepted_steps(
            lambda _time, state: field(state), first, run.coarse.time)
        coarse = stable_resample(run.coarse.time, coarse_knots,
                                 run.coarse_time)
        fine_knots = replay_accepted_steps(
            lambda _time, state: field(state), coarse[run.restart_index],
            run.fine.time)
        fine = stable_resample(run.fine.time, fine_knots, run.fine_time)
        return coarse, fine

    def mode_and_frequency(state):
        mode, continuous_frequency = published_mode22_and_frequency(
            state, ham, flux, **MODE_OPTIONS)
        orbital_frequency = jax.vmap(field)(state)[:, 1]
        return mode, continuous_frequency, orbital_frequency

    zero = jnp.asarray(0., dtype=jnp.float64)
    eager_states = replay(zero)
    compiled_states = jax.jit(replay)(zero)
    eager_coarse = mode_and_frequency(eager_states[0])
    eager_fine = mode_and_frequency(eager_states[1])
    compiled_coarse = jax.jit(mode_and_frequency)(eager_states[0])
    compiled_fine = jax.jit(mode_and_frequency)(eager_states[1])
    changed_input_fine = jax.jit(mode_and_frequency)(compiled_states[1])
    print("mode evaluations", time.monotonic() - started, flush=True)

    nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    reference_omega = jax.vmap(field)(run.fine_state)[:, 1]
    peak = full_orbital_peak(run.fine_time, reference_omega).time
    match = peak + seobnrv4_nqc_time_offset_numerical_compatibility(
        nqc_nu, nqc_chi)
    targets = _seobnrv4_nqc_targets_unchecked(nqc_nu, nqc_chi)

    def fit(state, omega, mode):
        return traceable_nqc_22_full_series_coefficients(
            run.fine_time, state[:, 0], state[:, 2], omega,
            mode, match, targets)[:2]

    eager_fit = fit(eager_states[1], eager_fine[2], eager_fine[0])
    compiled_fit = jax.jit(fit)(eager_states[1], eager_fine[2], eager_fine[0])
    changed_mode_fit = jax.jit(fit)(
        eager_states[1], compiled_fine[2], compiled_fine[0])
    changed_state_fit = jax.jit(fit)(
        compiled_states[1], changed_input_fine[2], changed_input_fine[0])
    print("NQC fits", time.monotonic() - started, flush=True)

    metrics = {
        "coarse_state_eager_jit": comparison(compiled_states[0], eager_states[0]),
        "fine_state_eager_jit": comparison(compiled_states[1], eager_states[1]),
        "fine_state_vs_reference_max_abs": max_abs(eager_states[1], run.fine_state),
        "coarse_mode_fixed_state": comparison(compiled_coarse[0], eager_coarse[0]),
        "fine_mode_fixed_state": comparison(compiled_fine[0], eager_fine[0]),
        "fine_continuous_frequency_fixed_state": comparison(
            compiled_fine[1], eager_fine[1]),
        "fine_orbital_frequency_fixed_state": comparison(
            compiled_fine[2], eager_fine[2]),
        "fine_mode_changed_state": comparison(
            changed_input_fine[0], compiled_fine[0]),
        "amplitude_fit_fixed_input": comparison(compiled_fit[0], eager_fit[0]),
        "phase_fit_fixed_input": comparison(compiled_fit[1], eager_fit[1]),
        "amplitude_fit_changed_mode": comparison(
            changed_mode_fit[0], compiled_fit[0]),
        "phase_fit_changed_mode": comparison(
            changed_mode_fit[1], compiled_fit[1]),
        "amplitude_fit_changed_state": comparison(
            changed_state_fit[0], changed_mode_fit[0]),
        "phase_fit_changed_state": comparison(
            changed_state_fit[1], changed_mode_fit[1]),
    }
    limits = plan["gates"]
    checks = {
        "replay": metrics["fine_state_vs_reference_max_abs"] <=
        limits["max_replay_state_absolute_error"],
        "mode": max(metrics[name]["relative_l2"] for name in (
            "coarse_mode_fixed_state", "fine_mode_fixed_state")) <=
        limits["max_fixed_input_mode_relative_l2"],
        "frequency": max(metrics[name]["relative_l2"] for name in (
            "fine_continuous_frequency_fixed_state",
            "fine_orbital_frequency_fixed_state")) <=
        limits["max_fixed_input_frequency_relative_l2"],
        "nqc": max(metrics[name]["max_abs"] for name in (
            "amplitude_fit_fixed_input", "phase_fit_fixed_input")) <=
        limits["max_fixed_input_nqc_coefficient_absolute_error"],
    }
    report.update(status="diagnostic_complete", metrics=metrics,
                  checks=checks, diagnostic_checks_passed=all(checks.values()),
                  elapsed_s=time.monotonic() - started)
    save(report)
    print(json.dumps({"checks": checks, "metrics": metrics}, sort_keys=True),
          flush=True)
    return 0 if report["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Frozen dev-only NQC combined-array versus spline/solve attribution."""

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
from .diagnose_jax_nqc_stable_spline import spline_rows
from .diagnose_jax_two_stage_trajectory import construct_case
from .eob.jax.numerical_attachment_clean import _solve_traceable, full_orbital_peak
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import (
    _seobnrv4_nqc_targets_unchecked, nqc_22_basis,
    seobnrv4_nqc_parameters,
    seobnrv4_nqc_time_offset_numerical_compatibility,
)
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_fixed_combined_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_fixed_combined.json")


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=q.ROOT,
        text=True).strip()
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-fixed-combined-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_head"] != head
            or plan["case"] != "high_start_mixed_spin"
            or plan["gates"] != {
                "max_fixed_combined_amplitude_coefficient_absolute_error": 1e-9,
                "max_fixed_rows_amplitude_coefficient_absolute_error": 1e-9}):
        raise ValueError("frozen fixed-combined plan changed")
    report = dict(schema="diffgw.seobnrv4.jax-nqc-fixed-combined.v1",
                  status="running", release_qualified=False,
                  source_head=head, plan_sha256=q.sha256(PLAN),
                  script_sha256=q.sha256(__file__), host=platform.node(),
                  cwd=os.getcwd(), pid=os.getpid())
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

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
    state = run.fine_state
    omega = jax.vmap(field)(state)[:, 1]
    mode, _ = published_mode22_and_frequency(state, ham, flux, **MODE_OPTIONS)
    nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    peak = full_orbital_peak(run.fine_time, omega).time
    match = peak + seobnrv4_nqc_time_offset_numerical_compatibility(
        nqc_nu, nqc_chi)
    targets = _seobnrv4_nqc_targets_unchecked(nqc_nu, nqc_chi)

    def build_combined(state_input, omega_input, mode_input):
        amplitude = jnp.abs(mode_input)
        q_basis, p_basis = nqc_22_basis(
            state_input[:, 0], state_input[:, 2], omega_input)
        return jnp.concatenate((amplitude[:, None],
                                amplitude[:, None] * q_basis, p_basis),
                               axis=-1)

    def rows(combined):
        return spline_rows(run.fine_time, combined, match)

    def amplitude_fit(matching_rows):
        value, first, second = matching_rows
        matrix = jnp.stack((value[1:4], first[1:4], second[1:4]))
        rhs = jnp.stack(targets[:3]) - jnp.stack((
            value[0], first[0], second[0]))
        return _solve_traceable(matrix, rhs)[0]

    eager_combined = build_combined(state, omega, mode)
    compiled_combined = jax.jit(build_combined)(state, omega, mode)
    eager_rows = rows(eager_combined)
    fixed_combined_rows = jax.jit(rows)(eager_combined)
    changed_combined_rows = jax.jit(rows)(compiled_combined)
    eager_fit = amplitude_fit(eager_rows)
    fixed_rows_fit = jax.jit(amplitude_fit)(eager_rows)
    fixed_combined_fit = jax.jit(amplitude_fit)(fixed_combined_rows)
    changed_combined_fit = jax.jit(amplitude_fit)(changed_combined_rows)

    metrics = {
        "combined_eager_jit_max_abs": max_abs(compiled_combined, eager_combined),
        "combined_eager_jit_relative_l2": relative_l2(
            compiled_combined, eager_combined),
        "value_fixed_combined_max_abs": max_abs(
            fixed_combined_rows[0], eager_rows[0]),
        "first_fixed_combined_max_abs": max_abs(
            fixed_combined_rows[1], eager_rows[1]),
        "second_fixed_combined_max_abs": max_abs(
            fixed_combined_rows[2], eager_rows[2]),
        "fixed_rows_fit_max_abs": max_abs(fixed_rows_fit, eager_fit),
        "fixed_combined_fit_max_abs": max_abs(fixed_combined_fit, eager_fit),
        "changed_combined_fit_max_abs": max_abs(
            changed_combined_fit, fixed_combined_fit),
    }
    limits = plan["gates"]
    checks = {
        "fixed_rows": metrics["fixed_rows_fit_max_abs"] <=
        limits["max_fixed_rows_amplitude_coefficient_absolute_error"],
        "fixed_combined": metrics["fixed_combined_fit_max_abs"] <=
        limits["max_fixed_combined_amplitude_coefficient_absolute_error"],
    }
    report.update(status="diagnostic_complete", metrics=metrics,
                  checks=checks, diagnostic_checks_passed=all(checks.values()),
                  elapsed_s=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checks": checks, "metrics": metrics}, sort_keys=True),
          flush=True)
    return 0 if report["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

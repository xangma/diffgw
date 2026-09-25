"""Frozen dev-only arithmetic-barrier NQC basis/product control."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import jax
from jax import lax
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

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_basis_barrier_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_basis_barrier.json")


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def original_combined(state, omega, mode):
    amplitude = jnp.abs(mode)
    q_basis, p_basis = nqc_22_basis(state[:, 0], state[:, 2], omega)
    return jnp.concatenate((amplitude[:, None],
                            amplitude[:, None] * q_basis, p_basis), axis=-1)


def candidate_combined(state, omega, mode):
    barrier = lax.optimization_barrier
    r, pr = state[:, 0], state[:, 2]
    ratio = barrier(pr / barrier(r * omega))
    ratio2 = barrier(ratio**2)
    rpow = barrier(r**1.5)
    q_basis = jnp.stack((ratio2, barrier(ratio2 / r),
                         barrier(ratio2 / rpow)), axis=-1)
    p_basis = jnp.stack((ratio, barrier(ratio * barrier(pr**2))), axis=-1)
    amplitude = barrier(jnp.abs(mode))
    products = barrier(amplitude[:, None] * q_basis)
    return jnp.concatenate((amplitude[:, None], products, p_basis), axis=-1)


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=q.ROOT,
        text=True).strip()
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-basis-barrier-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_head"] != head
            or plan["case"] != "high_start_mixed_spin"
            or plan["gates"] != {
                "max_candidate_source_combined_absolute_error": 1e-15,
                "max_candidate_source_coefficient_absolute_error": 1e-8,
                "max_candidate_eager_jit_combined_absolute_error": 0.0,
                "max_candidate_eager_jit_coefficient_absolute_error": 1e-9}):
        raise ValueError("frozen NQC basis barrier plan changed")
    report = dict(schema="diffgw.seobnrv4.jax-nqc-basis-barrier.v1",
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

    def amplitude_fit(combined):
        value, first, second = spline_rows(run.fine_time, combined, match)
        matrix = jnp.stack((value[1:4], first[1:4], second[1:4]))
        rhs = jnp.stack(targets[:3]) - jnp.stack((
            value[0], first[0], second[0]))
        return _solve_traceable(matrix, rhs)[0]

    source = original_combined(state, omega, mode)
    eager = candidate_combined(state, omega, mode)
    compiled = jax.jit(candidate_combined)(state, omega, mode)
    source_fit = amplitude_fit(source)
    eager_fit = amplitude_fit(eager)
    compiled_fit = jax.jit(amplitude_fit)(compiled)
    metrics = {
        "candidate_source_combined_max_abs": max_abs(eager, source),
        "candidate_source_coefficient_max_abs": max_abs(eager_fit, source_fit),
        "candidate_eager_jit_combined_max_abs": max_abs(compiled, eager),
        "candidate_eager_jit_coefficient_max_abs": max_abs(
            compiled_fit, eager_fit),
        "original_eager_jit_combined_max_abs": max_abs(
            jax.jit(original_combined)(state, omega, mode), source),
    }
    limits = plan["gates"]
    checks = {
        "source_combined": metrics["candidate_source_combined_max_abs"] <=
        limits["max_candidate_source_combined_absolute_error"],
        "source_coefficient": metrics["candidate_source_coefficient_max_abs"] <=
        limits["max_candidate_source_coefficient_absolute_error"],
        "eager_jit_combined": metrics["candidate_eager_jit_combined_max_abs"] <=
        limits["max_candidate_eager_jit_combined_absolute_error"],
        "eager_jit_coefficient": metrics["candidate_eager_jit_coefficient_max_abs"] <=
        limits["max_candidate_eager_jit_coefficient_absolute_error"],
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

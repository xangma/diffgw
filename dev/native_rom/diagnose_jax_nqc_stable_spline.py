"""Frozen dev-only arithmetic-barrier control for NQC match derivatives."""

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
from .diagnose_jax_two_stage_trajectory import construct_case
from .eob.jax.numerical_attachment_clean import (
    _solve_traceable, _unwrap_mode_phase_unchecked, full_orbital_peak,
    traceable_nqc_22_full_series_coefficients,
)
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import (
    _seobnrv4_nqc_targets_unchecked, NQCCoefficients, nqc_22_basis,
    nqc_22_factor, seobnrv4_nqc_parameters,
    seobnrv4_nqc_time_offset_numerical_compatibility,
)
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_stable_spline_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_stable_spline.json")


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def spline_rows(x, values, query):
    """Unbatched natural-cubic local rows with explicit rounding boundaries."""
    barrier = lax.optimization_barrier
    n = x.shape[0]
    h = barrier(x[1:] - x[:-1])
    slopes = barrier(barrier(values[1:] - values[:-1]) / h[:, None])
    lower = h[1:-1]
    diagonal = barrier(2.0 * barrier(h[:-1] + h[1:]))
    upper = h[1:-1]
    rhs = barrier(6.0 * barrier(slopes[1:] - slopes[:-1]))
    first_c = barrier(upper[0] / diagonal[0])
    first_d = barrier(rhs[0] / diagonal[0])
    upper_rest = jnp.concatenate((upper[1:], jnp.zeros_like(upper[:1])))

    def forward(carry, row):
        c_prev, d_prev = carry
        lo, diag, up, right = row
        pivot = barrier(diag - barrier(lo * c_prev))
        c_next = barrier(up / pivot)
        d_next = barrier(barrier(right - barrier(lo * d_prev)) / pivot)
        return (c_next, d_next), (c_next, d_next)

    _, (c_tail, d_tail) = lax.scan(
        forward, (first_c, first_d),
        (lower, diagonal[1:], upper_rest, rhs[1:]))
    cprime = jnp.concatenate((first_c[None], c_tail))
    dprime = jnp.concatenate((first_d[None], d_tail))

    def backward(next_solution, row):
        c_value, d_value = row
        solution = barrier(d_value - barrier(c_value * next_solution))
        return solution, solution

    _, reversed_solution = lax.scan(
        backward, dprime[-1], (cprime[:-1][::-1], dprime[:-1][::-1]))
    interior = jnp.concatenate((reversed_solution[::-1], dprime[-1:]))
    second = jnp.zeros_like(values).at[1:-1].set(interior)
    h_expanded = h[:, None]
    a = values[:-1]
    b = barrier(barrier(barrier(values[1:] - values[:-1]) / h_expanded)
                - barrier(barrier(h_expanded * barrier(
                    barrier(2.0 * second[:-1]) + second[1:])) / 6.0))
    c = barrier(second[:-1] / 2.0)
    d = barrier(barrier(second[1:] - second[:-1]) /
                barrier(6.0 * h_expanded))

    interval = jnp.clip(jnp.searchsorted(x, query, side="right") - 1,
                        0, n - 2)
    dx = barrier(query - x[interval])
    aa, bb, cc, dd = (array[interval] for array in (a, b, c, d))
    value = barrier(aa + barrier(dx * barrier(
        bb + barrier(dx * barrier(cc + barrier(dx * dd))))))
    first = barrier(bb + barrier(dx * barrier(
        barrier(2.0 * cc) + barrier(dx * barrier(3.0 * dd)))))
    second_row = barrier(barrier(2.0 * cc) +
                         barrier(dx * barrier(6.0 * dd)))
    return value, first, second_row


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=q.ROOT,
        text=True).strip()
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-stable-spline-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_head"] != head
            or plan["case"] != "high_start_mixed_spin"
            or plan["gates"] != {
                "max_fixed_input_coefficient_absolute_error": 1e-9,
                "max_candidate_source_coefficient_absolute_error": 1e-8,
                "max_candidate_source_corrected_mode_relative_l2": 1e-10}):
        raise ValueError("frozen NQC stable-spline plan changed")
    report = dict(schema="diffgw.seobnrv4.jax-nqc-stable-spline.v1",
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

    def stable_fit(state, orbital_frequency, base_mode):
        amplitude = jnp.abs(base_mode)
        phase = _unwrap_mode_phase_unchecked(base_mode)
        q_basis, p_basis = nqc_22_basis(
            state[:, 0], state[:, 2], orbital_frequency)
        combined = jnp.concatenate((amplitude[:, None],
                                    amplitude[:, None] * q_basis, p_basis),
                                   axis=-1)
        value, first, second = spline_rows(run.fine_time, combined, match)
        amplitude_matrix = jnp.stack((value[1:4], first[1:4], second[1:4]))
        amplitude_rhs = jnp.stack(targets[:3]) - jnp.stack((
            value[0], first[0], second[0]))
        phase_matrix = jnp.stack((first[4:6], second[4:6]))
        _, phase_first, phase_second = spline_rows(
            run.fine_time, phase[:, None], match)
        phase_rhs = jnp.stack((phase_first[0] - targets.frequency,
                               phase_second[0] - targets.frequency_derivative))
        a = _solve_traceable(amplitude_matrix, amplitude_rhs)[0]
        b = _solve_traceable(phase_matrix, phase_rhs)[0]
        return a, b

    reference = traceable_nqc_22_full_series_coefficients(
        run.fine_time, state[:, 0], state[:, 2], omega, mode,
        match, targets)[:2]
    eager = stable_fit(state, omega, mode)
    compiled = jax.jit(stable_fit)(state, omega, mode)
    original_corrected = mode * nqc_22_factor(
        state[:, 0], state[:, 2], omega, NQCCoefficients(*reference))
    candidate_corrected = mode * nqc_22_factor(
        state[:, 0], state[:, 2], omega, NQCCoefficients(*eager))
    metrics = {
        "amplitude_coefficient_eager_jit_max_abs": max_abs(compiled[0], eager[0]),
        "phase_coefficient_eager_jit_max_abs": max_abs(compiled[1], eager[1]),
        "amplitude_coefficient_source_max_abs": max_abs(eager[0], reference[0]),
        "phase_coefficient_source_max_abs": max_abs(eager[1], reference[1]),
        "corrected_mode_source_relative_l2": relative_l2(
            candidate_corrected, original_corrected),
    }
    limits = plan["gates"]
    checks = {
        "fixed_input": max(metrics[name] for name in (
            "amplitude_coefficient_eager_jit_max_abs",
            "phase_coefficient_eager_jit_max_abs")) <=
        limits["max_fixed_input_coefficient_absolute_error"],
        "source_coefficients": max(metrics[name] for name in (
            "amplitude_coefficient_source_max_abs",
            "phase_coefficient_source_max_abs")) <=
        limits["max_candidate_source_coefficient_absolute_error"],
        "source_corrected_mode": metrics["corrected_mode_source_relative_l2"] <=
        limits["max_candidate_source_corrected_mode_relative_l2"],
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

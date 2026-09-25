"""Dev-only fixed-input test of an NQC spline compiler barrier."""

from __future__ import annotations

import json
import os
import platform
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
    full_orbital_peak,
    traceable_fixed_chart_attached_mode22,
)
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.nqc import (
    _seobnrv4_nqc_targets_unchecked,
    nqc_22_basis,
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

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_nqc_spline_barrier_plan.json"
OUTPUT = PLAN.with_name("jax_nqc_spline_barrier.json")
STAGE_METRICS = (
    "combined_spline_input_max_abs", "matching_second_max_abs",
    "amplitude_matrix_max_abs", "amplitude_rhs_max_abs",
    "amplitude_coefficients_max_abs", "attachment_relative_l2",
)


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def relative_l2(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def compare(actual, reference):
    return {name: (relative_l2(actual[i], reference[i]) if i == 5
                   else max_abs(actual[i], reference[i]))
            for i, name in enumerate(STAGE_METRICS)}


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.jax-nqc-spline-barrier-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "high_start_mixed_spin"
            or plan["stage_metrics"] != list(STAGE_METRICS)):
        raise ValueError("frozen spline-barrier plan changed")
    baseline_path = q.ROOT / plan["baseline_result"]
    if q.sha256(baseline_path) != plan["baseline_sha256"]:
        raise ValueError("pinned baseline stage evidence changed")
    baseline = json.loads(baseline_path.read_text())
    if (baseline["schema"] != "diffgw.seobnrv4.jax-nqc-jit-stages.v1"
            or baseline["case"] != plan["case"]):
        raise ValueError("pinned baseline stage scope changed")

    m1, m2, chi1, chi2, _ = CASES[plan["case"]]
    _, initial, _, field = construct_case(plan["case"])
    dt = plan["output_step_M"]
    run = integrate_two_stage_trajectory(
        field, initial, output_step=dt, max_duration=1000.,
        coarse_initial_step=dt, coarse_max_step=2., fine_max_step=2.,
        fine_refinement_factor=plan["fine_refinement_factor"],
        atol=plan["adaptive_atol"], rtol=plan["adaptive_rtol"])
    nu, delta = m1 * m2 / (m1 + m2) ** 2, (m1 - m2) / (m1 + m2)
    ham = SEOBNRv4Hamiltonian(nu, chi1, chi2, delta=delta)
    flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated",
                                 retain_pan_cubic_spin=False)
    omega = jax.vmap(field)(run.fine_state)[:, 1]
    mode, _ = published_mode22_and_frequency(run.fine_state, ham, flux,
                                              **MODE_OPTIONS)
    compiled_mode = jax.jit(lambda state: published_mode22_and_frequency(
        state, ham, flux, **MODE_OPTIONS)[0])(run.fine_state)
    nqc_nu, nqc_chi = seobnrv4_nqc_parameters(m1, m2, chi1, chi2)
    peak = full_orbital_peak(run.fine_time, omega).time
    match = peak + seobnrv4_nqc_time_offset_numerical_compatibility(
        nqc_nu, nqc_chi)
    targets = _seobnrv4_nqc_targets_unchecked(nqc_nu, nqc_chi)
    final_mass = eob_final_mass_fraction(m1, m2, chi1, chi2)
    final_spin = hofmann_final_spin(m1, m2, chi1, chi2, fit_order=(3, 4))
    qnm_omega, qnm_damping = evaluate_kerr_qnm_220(final_spin)
    sigma = (qnm_omega - 1j * qnm_damping) / final_mass
    output_count = int(np.ceil((float(match) + 150.) / dt)) + 1
    output_time = jnp.arange(output_count) * dt
    print("fixed chart", len(run.fine_time), float(match), flush=True)

    def stages(barrier_combined):
        def evaluate(fine, orbital_frequency, base_mode):
            amplitude = jnp.abs(base_mode)
            q_basis, p_basis = nqc_22_basis(
                fine[:, 0], fine[:, 2], orbital_frequency)
            combined = jnp.concatenate((amplitude[:, None],
                                        amplitude[:, None] * q_basis,
                                        p_basis), axis=-1)
            if barrier_combined:
                combined = lax.optimization_barrier(combined)
            spline = natural_cubic_spline(
                run.fine_time, combined, scalar_output=False)
            value = spline(match)
            first = spline.derivative(match)
            second = spline.derivative(match, 2)
            matrix = jnp.stack((value[1:4], first[1:4], second[1:4]))
            rhs = jnp.stack(targets[:3]) - jnp.stack((
                value[0], first[0], second[0]))
            attached = traceable_fixed_chart_attached_mode22(
                run.fine_time, fine[:, 0], fine[:, 2], orbital_frequency,
                base_mode, match, targets, nqc_nu, nqc_chi, sigma,
                output_time, barrier_combined=barrier_combined)
            return combined, second[:4], matrix, rhs, attached[2], attached[0]
        return evaluate

    frozen_inputs = (run.fine_state, omega, mode)
    baseline_eager = stages(False)(*frozen_inputs)
    baseline_jit = jax.jit(stages(False))(*frozen_inputs)
    print("baseline fixed-input JIT", time.monotonic() - started, flush=True)
    barrier_eager = stages(True)(*frozen_inputs)
    barrier_jit = jax.jit(stages(True))(*frozen_inputs)
    print("barrier fixed-input JIT", time.monotonic() - started, flush=True)

    comparisons = {
        "baseline_eager_jit": compare(baseline_jit, baseline_eager),
        "barrier_eager_jit": compare(barrier_jit, barrier_eager),
        "barrier_vs_baseline_eager": compare(barrier_eager, baseline_eager),
        "barrier_vs_baseline_jit": compare(barrier_jit, baseline_jit),
    }
    limits = plan["checks"]
    checks = {
        "base_mode": relative_l2(compiled_mode, mode)
        <= limits["max_baseline_base_mode_relative_l2"],
        "eager_preserved_mode": comparisons["barrier_vs_baseline_eager"][
            "attachment_relative_l2"]
        <= limits["max_intervention_eager_attachment_relative_l2"],
        "eager_preserved_coefficients": comparisons["barrier_vs_baseline_eager"][
            "amplitude_coefficients_max_abs"]
        <= limits["max_intervention_eager_coefficients_absolute_error"],
        "barrier_coefficients": comparisons["barrier_eager_jit"][
            "amplitude_coefficients_max_abs"]
        <= limits["max_intervention_eager_jit_coefficients_absolute_error"],
        "barrier_attachment": comparisons["barrier_eager_jit"][
            "attachment_relative_l2"]
        <= limits["max_intervention_eager_jit_attachment_relative_l2"],
    }
    result = {
        "schema": "diffgw.seobnrv4.jax-nqc-spline-barrier.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "intervention_source_sha256": q.sha256(
            q.ROOT / "dev/native_rom/eob/jax/numerical_attachment_clean.py"),
        "baseline_sha256": q.sha256(baseline_path),
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "platform": platform.platform(), "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(), "case": plan["case"],
        "fine_samples": len(run.fine_time), "match_time_M": float(match),
        "output_samples": output_count,
        "base_mode_relative_l2": relative_l2(compiled_mode, mode),
        "comparisons": comparisons, "checks": checks,
        "diagnostic_checks_passed": all(checks.values()),
        "elapsed_s": time.monotonic() - started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"comparisons": comparisons, "checks": checks},
                     sort_keys=True), flush=True)
    return 0 if result["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

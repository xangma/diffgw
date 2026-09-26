"""Frozen dev-only JAX optimization-barrier control for the pre-NQC mode."""

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
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.published_waveform import published_mode22_and_frequency
from jaxwave.approximants.eob.seobnrv4_amplitude22 import seobnrv4_published_amplitude22
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux
from jaxwave.approximants.eob.seobnrv4_hamiltonian import SEOBNRv4Hamiltonian
from jaxwave.approximants.eob.seobnrv4_mode22 import (
    seobnrv4_published_delta22, seobnrv4_published_phase22,
)

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_mode_barrier_plan.json"
OUTPUT = PLAN.with_name("jax_mode_barrier.json")


def comparison(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    difference = actual - reference
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) /
                             np.linalg.norm(reference)),
    }


def save(report):
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=q.ROOT,
        text=True).strip()
    expected_paths = {
        str(path.relative_to(q.ROOT)): q.sha256(path)
        for path in (
            q.ROOT / "src/jaxwave/approximants/eob/published_waveform.py",
            q.ROOT / "src/jaxwave/approximants/eob/seobnrv4_mode22.py",
            q.ROOT / "src/jaxwave/approximants/eob/seobnrv4_amplitude22.py",
            q.ROOT / "src/jaxwave/approximants/eob/seobnrv4_flux.py",
        )
    }
    expected_gates = {
        "max_candidate_source_mode_relative_l2": 1e-13,
        "max_candidate_eager_jit_mode_relative_l2": 1e-16,
        "max_candidate_eager_jit_mode_absolute_error": 1e-14,
    }
    variants = ("control", "quantities", "quantities_residual", "all_stages")
    if (plan["schema"] != "diffgw.seobnrv4.jax-mode-barrier-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_head"] != head
            or plan["source_sha256"] != expected_paths
            or plan["case"] != "high_start_mixed_spin"
            or tuple(plan["variants"]) != variants
            or plan["gates"] != expected_gates):
        raise ValueError("frozen mode-barrier plan changed")
    report = dict(
        schema="diffgw.seobnrv4.jax-mode-barrier.v1",
        status="running", release_qualified=False,
        source_head=head, source_sha256=expected_paths,
        plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
        host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
    )
    save(report)

    m1, m2, chi1, chi2, _ = CASES[plan["case"]]
    _, initial, _, field = construct_case(plan["case"])
    dt = plan["output_step_M"]
    run = integrate_two_stage_trajectory(
        field, initial, output_step=dt, max_duration=1000.,
        coarse_initial_step=dt, coarse_max_step=2., fine_max_step=2.,
        fine_refinement_factor=plan["fine_refinement_factor"],
        atol=plan["adaptive_atol"], rtol=plan["adaptive_rtol"])
    state = run.fine_state
    print("trajectory", run.coarse.accepted_steps, run.fine.accepted_steps,
          time.monotonic() - started, flush=True)

    nu, delta = m1 * m2 / (m1 + m2)**2, (m1 - m2) / (m1 + m2)
    hamiltonian = SEOBNRv4Hamiltonian(nu, chi1, chi2, delta=delta)
    radiation = SEOBNRv4PublishedFlux(
        spin_truncation="tabulated", retain_pan_cubic_spin=False)

    def candidate(value, variant):
        quantities = radiation.from_hamiltonian(value, hamiltonian)
        if variant != "control":
            quantities = {
                key: jax.lax.optimization_barrier(item)
                for key, item in quantities.items()
            }
        residual = seobnrv4_published_delta22(
            **{key: quantities[key] for key in
               ("omega", "hreal_over_M", "x1", "x2", "chi1", "chi2")},
            spin_v6=MODE_OPTIONS["spin_v6"],
            include_pan_v8=MODE_OPTIONS["include_pan_v8"],
            include_positive_chi_540=MODE_OPTIONS["include_positive_chi_540"])
        if variant in ("quantities_residual", "all_stages"):
            residual = jax.lax.optimization_barrier(residual)
        amplitude = seobnrv4_published_amplitude22(
            **quantities, retain_pan_cubic_spin=False)
        phase = seobnrv4_published_phase22(
            quantities["omega"], value[..., 1],
            hreal_over_M=quantities["hreal_over_M"],
            residual_phase=residual)
        if variant == "all_stages":
            amplitude = jax.lax.optimization_barrier(amplitude)
            phase = jax.lax.optimization_barrier(phase)
        return amplitude * jnp.exp(-1j * phase)

    source, _ = published_mode22_and_frequency(
        state, hamiltonian, radiation, **MODE_OPTIONS)
    rows = {}
    checks = {}
    gates = plan["gates"]
    for variant in variants:
        eager = candidate(state, variant)
        compiled = jax.jit(lambda value: candidate(value, variant))(state)
        source_metrics = comparison(eager, source)
        jit_metrics = comparison(compiled, eager)
        rows[variant] = {
            "candidate_source_mode": source_metrics,
            "candidate_eager_jit_mode": jit_metrics,
        }
        checks[variant] = {
            "source": source_metrics["relative_l2"] <=
            gates["max_candidate_source_mode_relative_l2"],
            "eager_jit_relative": jit_metrics["relative_l2"] <=
            gates["max_candidate_eager_jit_mode_relative_l2"],
            "eager_jit_absolute": jit_metrics["max_abs"] <=
            gates["max_candidate_eager_jit_mode_absolute_error"],
        }
        print(variant, rows[variant], time.monotonic() - started, flush=True)

    passing_candidates = [
        variant for variant in variants[1:] if all(checks[variant].values())
    ]
    report.update(
        status="diagnostic_complete", variants=rows, checks=checks,
        passing_barrier_variants=passing_candidates,
        diagnostic_checks_passed=bool(passing_candidates),
        elapsed_s=time.monotonic() - started,
    )
    save(report)
    print(json.dumps({
        "passing_barrier_variants": passing_candidates,
        "checks": checks,
    }, sort_keys=True), flush=True)
    return 0 if passing_candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())

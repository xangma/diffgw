"""Frozen dev-only polar-assembly control for pre-NQC JAX mode roundoff."""

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

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-26/jax_polar_control_plan.json"
OUTPUT = PLAN.with_name("jax_polar_control.json")


def comparison(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    difference = actual - reference
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
    }


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    expected_files = {
        str(path.relative_to(q.ROOT)): q.sha256(path)
        for path in (
            q.ROOT / "src/jaxwave/approximants/eob/published_waveform.py",
            q.ROOT / "src/jaxwave/approximants/eob/seobnrv4_mode22.py",
            q.ROOT / "src/jaxwave/approximants/eob/seobnrv4_amplitude22.py",
            q.ROOT / "src/jaxwave/approximants/eob/seobnrv4_flux.py",
        )
    }
    variants = ("control", "split_trig", "split_trig_barrier", "complex_exp_barrier")
    gates = {
        "max_candidate_source_mode_relative_l2": 1e-13,
        "max_candidate_eager_jit_mode_relative_l2": 1e-16,
        "max_candidate_eager_jit_mode_absolute_error": 1e-14,
    }
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", plan["source_revision"], "HEAD"],
        cwd=q.ROOT, check=False,
    ).returncode == 0
    if (plan["schema"] != "diffgw.seobnrv4.jax-polar-control-plan.v1"
            or plan["release_qualified"] is not False
            or not ancestor or plan["source_sha256"] != expected_files
            or plan["case"] != "high_start_mixed_spin"
            or tuple(plan["variants"]) != variants or plan["gates"] != gates):
        raise ValueError("frozen polar control changed")
    report = dict(
        schema="diffgw.seobnrv4.jax-polar-control.v1", status="running",
        release_qualified=False, source_revision=plan["source_revision"],
        source_sha256=expected_files, plan_sha256=q.sha256(PLAN),
        script_sha256=q.sha256(__file__), host=platform.node(), cwd=os.getcwd(),
        pid=os.getpid(),
    )
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    m1, m2, chi1, chi2, _ = CASES[plan["case"]]
    _, initial, _, field = construct_case(plan["case"])
    dt = plan["output_step_M"]
    run = integrate_two_stage_trajectory(
        field, initial, output_step=dt, max_duration=1000.,
        coarse_initial_step=dt, coarse_max_step=2., fine_max_step=2.,
        fine_refinement_factor=plan["fine_refinement_factor"],
        atol=plan["adaptive_atol"], rtol=plan["adaptive_rtol"],
    )
    state = run.fine_state
    print("trajectory", run.coarse.accepted_steps, run.fine.accepted_steps,
          time.monotonic() - started, flush=True)

    nu, delta = m1 * m2 / (m1 + m2)**2, (m1 - m2) / (m1 + m2)
    ham = SEOBNRv4Hamiltonian(nu, chi1, chi2, delta=delta)
    flux = SEOBNRv4PublishedFlux(
        spin_truncation="tabulated", retain_pan_cubic_spin=False,
    )

    def candidate(value, variant):
        quantities = flux.from_hamiltonian(value, ham)
        residual = seobnrv4_published_delta22(
            **{key: quantities[key] for key in
               ("omega", "hreal_over_M", "x1", "x2", "chi1", "chi2")},
            spin_v6=MODE_OPTIONS["spin_v6"],
            include_pan_v8=MODE_OPTIONS["include_pan_v8"],
            include_positive_chi_540=MODE_OPTIONS["include_positive_chi_540"],
        )
        amplitude = seobnrv4_published_amplitude22(
            **quantities, retain_pan_cubic_spin=False,
        )
        phase = seobnrv4_published_phase22(
            quantities["omega"], value[..., 1],
            hreal_over_M=quantities["hreal_over_M"], residual_phase=residual,
        )
        if variant == "control":
            return amplitude * jnp.exp(-1j * phase)
        if variant == "split_trig":
            return jax.lax.complex(amplitude * jnp.cos(phase),
                                   -amplitude * jnp.sin(phase))
        if variant == "split_trig_barrier":
            amplitude = jax.lax.optimization_barrier(amplitude)
            phase = jax.lax.optimization_barrier(phase)
            real = jax.lax.optimization_barrier(
                amplitude * jax.lax.optimization_barrier(jnp.cos(phase)))
            imag = jax.lax.optimization_barrier(
                -amplitude * jax.lax.optimization_barrier(jnp.sin(phase)))
            return jax.lax.complex(real, imag)
        exponential = jax.lax.optimization_barrier(
            jnp.exp(jax.lax.optimization_barrier(-1j * phase)))
        return jax.lax.optimization_barrier(amplitude * exponential)

    source, _ = published_mode22_and_frequency(state, ham, flux, **MODE_OPTIONS)
    rows = {}
    checks = {}
    for variant in variants:
        eager = candidate(state, variant)
        compiled = jax.jit(lambda value: candidate(value, variant))(state)
        source_metrics = comparison(eager, source)
        jit_metrics = comparison(compiled, eager)
        rows[variant] = {"candidate_source_mode": source_metrics,
                         "candidate_eager_jit_mode": jit_metrics}
        checks[variant] = {
            "source": source_metrics["relative_l2"] <=
            gates["max_candidate_source_mode_relative_l2"],
            "eager_jit_relative": jit_metrics["relative_l2"] <=
            gates["max_candidate_eager_jit_mode_relative_l2"],
            "eager_jit_absolute": jit_metrics["max_abs"] <=
            gates["max_candidate_eager_jit_mode_absolute_error"],
        }
        print(variant, rows[variant], time.monotonic() - started, flush=True)
    passing = [variant for variant in variants[1:] if all(checks[variant].values())]
    report.update(status="diagnostic_complete", variants=rows, checks=checks,
                  passing_candidates=passing, diagnostic_checks_passed=bool(passing),
                  elapsed_s=time.monotonic() - started)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passing_candidates": passing, "checks": checks},
                     sort_keys=True), flush=True)
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())

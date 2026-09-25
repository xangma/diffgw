"""Dev-only fixed-state attribution of published pre-NQC mode arithmetic."""

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

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_mode_stage_plan.json"
OUTPUT = PLAN.with_name("jax_mode_stages.json")


def comparison(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    difference = actual - reference
    return {"max_abs": float(np.max(np.abs(difference))),
            "relative_l2": float(np.linalg.norm(difference) /
                                 np.linalg.norm(reference))}


def save(report):
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=q.ROOT,
        text=True).strip()
    if (plan["schema"] != "diffgw.seobnrv4.jax-mode-stage-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_head"] != head
            or plan["case"] != "high_start_mixed_spin"
            or plan["gates"] != {
                "max_instrumented_eager_source_mode_relative_l2": 1e-12,
                "min_published_eager_jit_mode_relative_l2": 1e-16}):
        raise ValueError("frozen mode-stage plan changed")
    report = dict(schema="diffgw.seobnrv4.jax-mode-stages.v1",
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
    state = run.fine_state
    print("trajectory", run.coarse.accepted_steps, run.fine.accepted_steps,
          time.monotonic() - started, flush=True)

    nu, delta = m1 * m2 / (m1 + m2)**2, (m1 - m2) / (m1 + m2)
    ham = SEOBNRv4Hamiltonian(nu, chi1, chi2, delta=delta)
    flux = SEOBNRv4PublishedFlux(spin_truncation="tabulated",
                                 retain_pan_cubic_spin=False)

    def mode_from_fixed_quantities(quantities, residual, phi):
        amplitude = seobnrv4_published_amplitude22(
            **quantities, retain_pan_cubic_spin=False)
        phase = seobnrv4_published_phase22(
            quantities["omega"], phi,
            hreal_over_M=quantities["hreal_over_M"],
            residual_phase=residual)
        mode = amplitude * jnp.exp(-1j * phase)
        return amplitude, phase, mode

    def stages(value):
        quantities = flux.from_hamiltonian(value, ham)
        residual = seobnrv4_published_delta22(
            **{key: quantities[key] for key in
               ("omega", "hreal_over_M", "x1", "x2", "chi1", "chi2")},
            spin_v6=MODE_OPTIONS["spin_v6"],
            include_pan_v8=MODE_OPTIONS["include_pan_v8"],
            include_positive_chi_540=MODE_OPTIONS["include_positive_chi_540"])
        amplitude, phase, mode = mode_from_fixed_quantities(
            quantities, residual, value[..., 1])
        published, _ = published_mode22_and_frequency(
            value, ham, flux, **MODE_OPTIONS)
        return quantities, residual, amplitude, phase, mode, published

    eager = stages(state)
    compiled = jax.jit(stages)(state)
    fixed = jax.jit(mode_from_fixed_quantities)(
        eager[0], eager[1], state[..., 1])
    names = tuple(eager[0])
    metrics = {
        "quantity_eager_jit": {
            name: comparison(compiled[0][name], eager[0][name])
            for name in names},
        "residual_eager_jit": comparison(compiled[1], eager[1]),
        "amplitude_eager_jit": comparison(compiled[2], eager[2]),
        "phase_eager_jit": comparison(compiled[3], eager[3]),
        "instrumented_mode_eager_jit": comparison(compiled[4], eager[4]),
        "published_mode_eager_jit": comparison(compiled[5], eager[5]),
        "instrumented_eager_source_mode": comparison(eager[4], eager[5]),
        "instrumented_jit_source_mode": comparison(compiled[4], compiled[5]),
        "fixed_quantity_amplitude_eager_jit": comparison(fixed[0], eager[2]),
        "fixed_quantity_phase_eager_jit": comparison(fixed[1], eager[3]),
        "fixed_quantity_mode_eager_jit": comparison(fixed[2], eager[4]),
    }
    checks = {
        "source_reproduced": metrics["instrumented_eager_source_mode"]["relative_l2"] <=
        plan["gates"]["max_instrumented_eager_source_mode_relative_l2"],
        "drift_reproduced": metrics["published_mode_eager_jit"]["relative_l2"] >=
        plan["gates"]["min_published_eager_jit_mode_relative_l2"],
    }
    report.update(status="diagnostic_complete", metrics=metrics, checks=checks,
                  diagnostic_checks_passed=all(checks.values()),
                  elapsed_s=time.monotonic() - started)
    save(report)
    print(json.dumps({"checks": checks, "metrics": metrics}, sort_keys=True),
          flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

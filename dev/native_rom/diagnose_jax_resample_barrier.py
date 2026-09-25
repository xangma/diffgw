"""Frozen development control for eager/JIT spline-resampling arithmetic."""

from __future__ import annotations

from dataclasses import replace
import json
import os
import platform
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import qualify_generator as q
from .diagnose_jax_two_stage_trajectory import construct_case
from .eob.jax.adaptive_driver_clean import replay_accepted_steps
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_resample_barrier_plan.json"
OUTPUT = PLAN.with_name("jax_resample_barrier.json")


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
    if (plan["schema"] != "diffgw.seobnrv4.jax-resample-barrier-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_head"] != head
            or plan["case"] != "high_start_mixed_spin"
            or plan["gates"] != {
                "max_candidate_eager_jit_sample_absolute_error": 0.0,
                "max_candidate_baseline_sample_absolute_error": 1e-14,
                "max_replay_reference_state_absolute_error": 1e-7}):
        raise ValueError("frozen spline-barrier plan changed")
    report = dict(schema="diffgw.seobnrv4.jax-resample-barrier.v1",
                  status="running", release_qualified=False,
                  source_head=head, plan_sha256=q.sha256(PLAN),
                  script_sha256=q.sha256(__file__), host=platform.node(),
                  cwd=os.getcwd(), pid=os.getpid())
    save(report)

    _, initial, _, field = construct_case(plan["case"])
    dt = plan["output_step_M"]
    run = integrate_two_stage_trajectory(
        field, initial, output_step=dt, max_duration=1000.,
        coarse_initial_step=dt, coarse_max_step=2., fine_max_step=2.,
        fine_refinement_factor=plan["fine_refinement_factor"],
        atol=plan["adaptive_atol"], rtol=plan["adaptive_rtol"])
    print("trajectory", run.coarse.accepted_steps, run.fine.accepted_steps,
          time.monotonic() - started, flush=True)

    def resample(time, knots, query, variant):
        if variant == "input":
            knots = jax.lax.optimization_barrier(knots)
        spline = natural_cubic_spline(time, knots, scalar_output=False)
        if variant == "coefficients":
            spline = replace(spline, **{
                name: jax.lax.optimization_barrier(getattr(spline, name))
                for name in ("a", "b", "c", "d")})
        return spline.evaluate(query)

    def replay(offset, variant):
        first = initial.at[0].add(offset)
        coarse_knots = replay_accepted_steps(
            lambda _time, state: field(state), first, run.coarse.time)
        coarse = resample(run.coarse.time, coarse_knots, run.coarse_time,
                          variant)
        fine_knots = replay_accepted_steps(
            lambda _time, state: field(state), coarse[run.restart_index],
            run.fine.time)
        fine = resample(run.fine.time, fine_knots, run.fine_time, variant)
        return coarse, fine

    zero = jnp.asarray(0., dtype=jnp.float64)
    baseline = replay(zero, "none")
    variants = {}
    for variant in ("none", "input", "coefficients"):
        eager = baseline if variant == "none" else replay(zero, variant)
        compiled = jax.jit(lambda offset: replay(offset, variant))(zero)
        variants[variant] = dict(
            coarse_eager_jit_max_abs=max_abs(eager[0], compiled[0]),
            fine_eager_jit_max_abs=max_abs(eager[1], compiled[1]),
            coarse_baseline_max_abs=max_abs(eager[0], baseline[0]),
            fine_baseline_max_abs=max_abs(eager[1], baseline[1]))
        print(variant, variants[variant], time.monotonic() - started,
              flush=True)

    reference_error = max(max_abs(baseline[0], run.coarse_state),
                          max_abs(baseline[1], run.fine_state))
    limits = plan["gates"]
    checks = {"replay": reference_error <=
              limits["max_replay_reference_state_absolute_error"]}
    for variant in ("input", "coefficients"):
        metrics = variants[variant]
        checks[f"{variant}_eager_jit"] = max(
            metrics["coarse_eager_jit_max_abs"],
            metrics["fine_eager_jit_max_abs"]) <= limits[
                "max_candidate_eager_jit_sample_absolute_error"]
        checks[f"{variant}_source"] = max(
            metrics["coarse_baseline_max_abs"],
            metrics["fine_baseline_max_abs"]) <= limits[
                "max_candidate_baseline_sample_absolute_error"]
    report.update(status="diagnostic_complete", variants=variants,
                  replay_reference_state_max_abs=reference_error,
                  checks=checks, diagnostic_checks_passed=all(checks.values()),
                  elapsed_s=time.monotonic() - started)
    save(report)
    return 0 if report["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

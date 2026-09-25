"""Frozen fixed-coefficient Horner arithmetic control for JAX resampling."""

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
from .diagnose_jax_two_stage_trajectory import construct_case
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_fixed_spline_horner_plan.json"
OUTPUT = PLAN.with_name("jax_fixed_spline_horner.json")


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def stable_evaluate(knots, query, a, b, c, d):
    interval = jnp.clip(jnp.searchsorted(knots, query, side="right") - 1,
                        0, knots.shape[0] - 2)
    dx = (query - jnp.take(knots, interval))[:, None]
    aa, bb, cc, dd = (jnp.take(value, interval, axis=0)
                      for value in (a, b, c, d))
    barrier = jax.lax.optimization_barrier
    value = barrier(dx * dd)
    value = barrier(cc + value)
    value = barrier(dx * value)
    value = barrier(bb + value)
    value = barrier(dx * value)
    return aa + value


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=q.ROOT,
        text=True).strip()
    if (plan["schema"] != "diffgw.seobnrv4.jax-fixed-spline-horner-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_head"] != head
            or plan["case"] != "high_start_mixed_spin"
            or plan["gates"] != {
                "max_barrier_eager_jit_absolute_error": 0.0,
                "max_barrier_source_absolute_error": 1e-14}):
        raise ValueError("frozen Horner plan changed")
    report = dict(schema="diffgw.seobnrv4.jax-fixed-spline-horner.v1",
                  status="running", release_qualified=False,
                  source_head=head, plan_sha256=q.sha256(PLAN),
                  script_sha256=q.sha256(__file__), host=platform.node(),
                  cwd=os.getcwd(), pid=os.getpid())
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    _, initial, _, field = construct_case(plan["case"])
    dt = plan["output_step_M"]
    run = integrate_two_stage_trajectory(
        field, initial, output_step=dt, max_duration=1000.,
        coarse_initial_step=dt, coarse_max_step=2., fine_max_step=2.,
        fine_refinement_factor=plan["fine_refinement_factor"],
        atol=plan["adaptive_atol"], rtol=plan["adaptive_rtol"])
    stages = (("coarse", run.coarse.time, run.coarse.state, run.coarse_time),
              ("fine", run.fine.time, run.fine.state, run.fine_time))
    rows = {}
    for name, knots, states, query in stages:
        spline = natural_cubic_spline(knots, states, scalar_output=False)
        args = (knots, query, spline.a, spline.b, spline.c, spline.d)
        eager = stable_evaluate(*args)
        compiled = jax.jit(stable_evaluate)(*args)
        source = spline.evaluate(query)
        rows[name] = dict(eager_jit_max_abs=max_abs(eager, compiled),
                          eager_source_max_abs=max_abs(eager, source))
    limits = plan["gates"]
    report.update(status="diagnostic_complete", stages=rows,
                  checks={
                      "eager_jit": max(row["eager_jit_max_abs"] for row in rows.values()) <=
                      limits["max_barrier_eager_jit_absolute_error"],
                      "source": max(row["eager_source_max_abs"] for row in rows.values()) <=
                      limits["max_barrier_source_absolute_error"]},
                  elapsed_s=time.monotonic() - started)
    report["diagnostic_checks_passed"] = all(report["checks"].values())
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows, sort_keys=True), flush=True)
    return 0 if report["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

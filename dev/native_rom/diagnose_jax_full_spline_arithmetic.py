"""Development-only fixed-chart natural-spline arithmetic control."""

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
from .diagnose_jax_two_stage_trajectory import construct_case
from .eob.jax.adaptive_driver_clean import replay_accepted_steps
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_full_spline_arithmetic_plan.json"
OUTPUT = PLAN.with_name("jax_full_spline_arithmetic.json")


def max_abs(actual, reference):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(reference))))


def stable_resample(x, values, query):
    """Unbatched fixed-chart spline; barriers expose each rounding boundary."""
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

    q_eval = jnp.clip(query, x[0], x[-1])
    interval = jnp.clip(jnp.searchsorted(x, q_eval, side="right") - 1,
                        0, n - 2)
    dx = barrier(q_eval - jnp.take(x, interval))[:, None]
    aa, bb, cc, dd = (jnp.take(array, interval, axis=0)
                      for array in (a, b, c, d))
    result = barrier(dx * dd)
    result = barrier(cc + result)
    result = barrier(dx * result)
    result = barrier(bb + result)
    result = barrier(dx * result)
    return aa + result


def main() -> int:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=q.ROOT,
        text=True).strip()
    if (plan["schema"] != "diffgw.seobnrv4.jax-full-spline-arithmetic-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_head"] != head
            or plan["case"] != "high_start_mixed_spin"
            or plan["gates"] != {
                "max_candidate_eager_jit_state_absolute_error": 0.0,
                "max_candidate_baseline_state_absolute_error": 1e-14,
                "max_replay_reference_state_absolute_error": 1e-7}):
        raise ValueError("frozen full-spline arithmetic plan changed")
    report = dict(schema="diffgw.seobnrv4.jax-full-spline-arithmetic.v1",
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

    def replay(offset, stable):
        first = initial.at[0].add(offset)
        coarse_knots = replay_accepted_steps(
            lambda _time, state: field(state), first, run.coarse.time)
        if stable:
            coarse = stable_resample(run.coarse.time, coarse_knots,
                                     run.coarse_time)
        else:
            coarse = natural_cubic_spline(
                run.coarse.time, coarse_knots,
                scalar_output=False).evaluate(run.coarse_time)
        fine_knots = replay_accepted_steps(
            lambda _time, state: field(state), coarse[run.restart_index],
            run.fine.time)
        if stable:
            fine = stable_resample(run.fine.time, fine_knots, run.fine_time)
        else:
            fine = natural_cubic_spline(
                run.fine.time, fine_knots,
                scalar_output=False).evaluate(run.fine_time)
        return coarse, fine

    zero = jnp.asarray(0., dtype=jnp.float64)
    baseline = replay(zero, False)
    candidate = replay(zero, True)
    compiled = jax.jit(lambda offset: replay(offset, True))(zero)
    rows = {
        name: dict(eager_jit_max_abs=max_abs(candidate[i], compiled[i]),
                   baseline_max_abs=max_abs(candidate[i], baseline[i]))
        for i, name in enumerate(("coarse", "fine"))}
    reference_error = max(max_abs(baseline[0], run.coarse_state),
                          max_abs(baseline[1], run.fine_state))
    limits = plan["gates"]
    report.update(status="diagnostic_complete", stages=rows,
                  replay_reference_state_max_abs=reference_error,
                  checks={
                      "eager_jit": max(row["eager_jit_max_abs"] for row in rows.values()) <=
                      limits["max_candidate_eager_jit_state_absolute_error"],
                      "source": max(row["baseline_max_abs"] for row in rows.values()) <=
                      limits["max_candidate_baseline_state_absolute_error"],
                      "replay": reference_error <=
                      limits["max_replay_reference_state_absolute_error"]},
                  elapsed_s=time.monotonic() - started)
    report["diagnostic_checks_passed"] = all(report["checks"].values())
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows, sort_keys=True), flush=True)
    return 0 if report["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

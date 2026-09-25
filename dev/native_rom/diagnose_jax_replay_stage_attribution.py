"""Locate eager/JIT drift in accepted-knot replay versus grid resampling."""

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
from .eob.jax.adaptive_driver_clean import replay_accepted_steps
from .eob.jax.two_stage_trajectory import integrate_two_stage_trajectory
from jaxwave.approximants.eob.numerical_attachment import natural_cubic_spline

jax.config.update("jax_enable_x64", True)

PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/jax_replay_stage_attribution_plan.json"
OUTPUT = PLAN.with_name("jax_replay_stage_attribution.json")


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
    if (plan["schema"] != "diffgw.seobnrv4.jax-replay-stage-attribution-plan.v1"
            or plan["release_qualified"] is not False
            or plan["source_head"] != head
            or plan["case"] != "high_start_mixed_spin"
            or plan["gates"] != {"max_replay_state_absolute_error": 1e-7}):
        raise ValueError("frozen replay attribution plan changed")
    report = dict(schema="diffgw.seobnrv4.jax-replay-stage-attribution.v1",
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

    def resample(time, knots, query):
        return natural_cubic_spline(time, knots,
                                    scalar_output=False).evaluate(query)

    def stages(offset):
        first = initial.at[0].add(offset)
        coarse_knots = replay_accepted_steps(
            lambda _time, state: field(state), first, run.coarse.time)
        coarse = resample(run.coarse.time, coarse_knots, run.coarse_time)
        fine_knots = replay_accepted_steps(
            lambda _time, state: field(state), coarse[run.restart_index],
            run.fine.time)
        fine = resample(run.fine.time, fine_knots, run.fine_time)
        return coarse_knots, coarse, fine_knots, fine

    zero = jnp.asarray(0., dtype=jnp.float64)
    eager = stages(zero)
    compiled = jax.jit(stages)(zero)
    names = ("coarse_accepted_knots", "coarse_uniform_samples",
             "fine_accepted_knots", "fine_uniform_samples")
    differences = {name: max_abs(compiled[i], eager[i])
                   for i, name in enumerate(names)}
    same_input_resampling = {
        "coarse_max_abs": max_abs(
            jax.jit(lambda knots: resample(
                run.coarse.time, knots, run.coarse_time))(eager[0]), eager[1]),
        "fine_max_abs": max_abs(
            jax.jit(lambda knots: resample(
                run.fine.time, knots, run.fine_time))(eager[2]), eager[3]),
    }
    reference_error = max(max_abs(eager[1], run.coarse_state),
                          max_abs(eager[3], run.fine_state))
    report.update(status="diagnostic_complete",
                  accepted_steps={"coarse": run.coarse.accepted_steps,
                                  "fine": run.fine.accepted_steps},
                  differences_max_abs=differences,
                  identical_input_resampling_max_abs=same_input_resampling,
                  replay_reference_state_max_abs=reference_error,
                  checks={"replay": reference_error <=
                          plan["gates"]["max_replay_state_absolute_error"]},
                  elapsed_s=time.monotonic() - started)
    report["diagnostic_checks_passed"] = all(report["checks"].values())
    save(report)
    print(json.dumps({"differences": differences,
                      "identical_input_resampling": same_input_resampling},
                     sort_keys=True), flush=True)
    return 0 if report["diagnostic_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

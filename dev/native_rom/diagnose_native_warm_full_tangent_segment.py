"""Dev-only checkpointed fixed-mesh spin tangent before the 0.04 crossing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom.diagnose_jax_spliced_whole_waveform import SOURCE_FLUX_OPTIONS
from dev.native_rom.eob.jax.adaptive_driver_clean import _B, _B5
from dev.native_rom.eob.jax.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as JaxHamiltonian,
)
from dev.native_rom.eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as TorchHamiltonian,
)
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator as JaxIntegrator
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from torchwave.approximants.eob.adaptive_driver_clean import fixed_fehlberg45_step
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator as TorchIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "native_warm_full_tangent_segments_plan.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def scaled_error(actual, expected):
    return float(np.max(np.abs(actual - expected) / np.maximum(1.0, np.abs(expected))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen segment script changed")
    for path, expected in plan["inputs_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"frozen source changed: {path}")
    mesh_source = plan["mesh_source"]
    if sha256(ROOT / mesh_source["path"]) != mesh_source["sha256"]:
        raise ValueError("frozen mesh evidence changed")
    mesh_evidence = json.loads((ROOT / mesh_source["path"]).read_text())
    if mesh_evidence["status"] != "diagnostic_complete" or not mesh_evidence["diagnostic_checks_passed"]:
        raise ValueError("mesh discovery did not pass")
    case = mesh_evidence["case"]
    mesh = mesh_evidence["mesh"]
    pre_crossing_intervals = plan["mesh"]["pre_crossing_intervals"]
    if (mesh["pre_crossing_intervals"] != pre_crossing_intervals
            or mesh["phase_b_segment_count"] != plan["mesh"]["segment_count"]
            or mesh["last_bracket_time_M"] != plan["mesh"]["last_bracket_time_M"]):
        raise ValueError("mesh does not match frozen segment bounds")
    segment = args.segment
    count = plan["mesh"]["segment_count"]
    if not 1 <= segment <= count:
        raise ValueError("segment index outside frozen range")
    width = plan["mesh"]["intervals_per_segment_at_most"]
    start = (segment - 1) * width
    end = min(segment * width, pre_crossing_intervals)
    times = np.asarray(mesh["accepted_time_M"], dtype=np.float64)
    baseline_states = np.asarray(mesh["accepted_state"], dtype=np.float64)
    if len(times) != mesh["accepted_steps"] + 1 or end >= len(times) - 1:
        raise ValueError("segment must end at pre-crossing accepted knot")

    prior = None
    if segment == 1:
        state0 = np.asarray(case["frozen_state"], dtype=np.float64)
        tangent0 = np.asarray(case["state_tangent_per_chi1"], dtype=np.float64)
        torch_state = jax_state = state0.copy()
        torch_tangent = jax_tangent = tangent0.copy()
    else:
        previous_path = EVIDENCE / f"native_warm_full_tangent_segment_{segment-1:02d}.json"
        previous_sha = sha256(previous_path)
        previous = json.loads(previous_path.read_text())
        if (previous["status"] != "diagnostic_complete"
                or not previous["diagnostic_checks_passed"]
                or previous["segment"] != segment - 1
                or previous["end_knot"] != start
                or previous["plan_sha256"] != sha256(PLAN)
                or previous["mesh_source"] != mesh_source):
            raise ValueError("previous checkpoint failed or changed chain")
        prior = {"path": str(previous_path.relative_to(ROOT)), "sha256": previous_sha}
        torch_state = np.asarray(previous["torch"]["state"], dtype=np.float64)
        torch_tangent = np.asarray(previous["torch"]["spin_tangent"], dtype=np.float64)
        jax_state = np.asarray(previous["jax"]["state"], dtype=np.float64)
        jax_tangent = np.asarray(previous["jax"]["spin_tangent"], dtype=np.float64)

    output = EVIDENCE / f"native_warm_full_tangent_segment_{segment:02d}.json"
    result = {
        "schema": "diffgw.seobnrv4.native-warm-full-tangent-segment.v1",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": sha256(PLAN), "script_sha256": sha256(Path(__file__)),
        "input_sha256": plan["inputs_sha256"], "mesh_source": mesh_source,
        "prior_checkpoint": prior, "segment": segment,
        "start_knot": start, "end_knot": end,
        "interval_count": end - start,
        "initial": {
            "torch_state": torch_state.tolist(), "torch_spin_tangent": torch_tangent.tolist(),
            "jax_state": jax_state.tolist(), "jax_spin_tangent": jax_tangent.tolist(),
        },
    }
    write(output, result)
    started = time.monotonic()
    try:
        jax.config.update("jax_enable_x64", True)
        torch.set_num_threads(1)
        mass1, mass2 = case["mass1"], case["mass2"]
        nu = mass1 * mass2 / (mass1 + mass2) ** 2
        delta = (mass1 - mass2) / (mass1 + mass2)
        chi0 = case["chi1"]
        chi2 = case["chi2"]
        torch_flux, jax_flux = TorchFlux(**SOURCE_FLUX_OPTIONS), JaxFlux(**SOURCE_FLUX_OPTIONS)

        def tensor(value, *, grad=False):
            return torch.tensor(value, dtype=torch.float64, requires_grad=grad)

        def torch_field(spin):
            ham = TorchHamiltonian(tensor(nu), spin, tensor(chi2), delta=tensor(delta))
            return TorchIntegrator(ham, torch_flux).vector_field

        def jax_field(spin, state):
            ham = JaxHamiltonian(jnp.asarray(nu), spin, jnp.asarray(chi2),
                                 delta=jnp.asarray(delta))
            return JaxIntegrator(ham, jax_flux).vector_field(state)

        def directional(spin, state, spin_tangent, state_tangent):
            return jax.jvp(jax_field, (spin, state), (spin_tangent, state_tangent))

        compiled_directional = jax.jit(directional)

        def jax_step(state, tangent, h):
            stages, tangent_stages = [], []
            for index in range(6):
                argument = state.copy()
                argument_tangent = tangent.copy()
                for earlier, weight in enumerate(_B[index]):
                    argument += h * weight * stages[earlier]
                    argument_tangent += h * weight * tangent_stages[earlier]
                stage, tangent_stage = compiled_directional(
                    jnp.asarray(chi0), jnp.asarray(argument),
                    jnp.asarray(1.0), jnp.asarray(argument_tangent),
                )
                stages.append(np.asarray(stage))
                tangent_stages.append(np.asarray(tangent_stage))
            next_state, next_tangent = state.copy(), tangent.copy()
            for index, weight in enumerate(_B5):
                next_state += h * weight * stages[index]
                next_tangent += h * weight * tangent_stages[index]
            return next_state, next_tangent

        cross_state_max = 0.0
        torch_mesh_max = 0.0
        jax_mesh_max = 0.0
        cross_tangent_max = 0.0
        for knot in range(start, end):
            h = float(times[knot + 1] - times[knot])
            spin = tensor(chi0, grad=True)
            state = tensor(torch_state) + (spin - chi0) * tensor(torch_tangent)
            next_torch = fixed_fehlberg45_step(torch_field(spin), 0.0, state, h).fifth
            next_torch_state = np.asarray(next_torch.detach()).copy()
            next_torch_tangent = np.asarray(torch.stack([
                torch.autograd.grad(component, spin, retain_graph=True)[0]
                for component in next_torch
            ]).detach()).copy()
            next_jax_state, next_jax_tangent = jax_step(jax_state, jax_tangent, h)
            torch_state, torch_tangent = next_torch_state, next_torch_tangent
            jax_state, jax_tangent = next_jax_state, next_jax_tangent
            baseline = baseline_states[knot + 1]
            cross_state_max = max(cross_state_max, float(np.max(np.abs(torch_state - jax_state))))
            torch_mesh_max = max(torch_mesh_max, float(np.max(np.abs(torch_state - baseline))))
            jax_mesh_max = max(jax_mesh_max, float(np.max(np.abs(jax_state - baseline))))
            cross_tangent_max = max(cross_tangent_max, scaled_error(torch_tangent, jax_tangent))
            result["completed_intervals"] = knot + 1 - start
            write(output, result)

        metrics = {
            "max_torch_jax_state_abs": cross_state_max,
            "max_torch_mesh_state_abs": torch_mesh_max,
            "max_jax_mesh_state_abs": jax_mesh_max,
            "max_torch_jax_spin_tangent_scaled": cross_tangent_max,
        }
        result["torch"] = {"state": torch_state.tolist(), "spin_tangent": torch_tangent.tolist()}
        result["jax"] = {"state": jax_state.tolist(), "spin_tangent": jax_tangent.tolist()}
        result["metrics"] = metrics
        gates = {
            "finite_final_states_and_tangents": bool(all(np.isfinite(array).all() for array in (
                torch_state, torch_tangent, jax_state, jax_tangent,
            ))),
            "strictly_increasing_segment_times": bool(np.all(np.diff(times[start:end+1]) > 0)),
            "segment_interval_count_within_bound": 0 < end - start <= width,
        }
        metric_keys = {
            "torch_jax_state_abs_at_most": "max_torch_jax_state_abs",
            "torch_mesh_state_abs_at_most": "max_torch_mesh_state_abs",
            "jax_mesh_state_abs_at_most": "max_jax_mesh_state_abs",
            "torch_jax_spin_tangent_scaled_at_most": "max_torch_jax_spin_tangent_scaled",
        }
        gates.update({name: metrics[metric] <= threshold
                      for name, metric in metric_keys.items()
                      for threshold in (plan["checks"][name],)})
        result["gates"] = gates
        result["diagnostic_checks_passed"] = all(gates.values())
        result["status"] = "diagnostic_complete"
    except Exception:
        result["status"] = "diagnostic_failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - started
        write(output, result)


if __name__ == "__main__":
    main()

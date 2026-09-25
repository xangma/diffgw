"""Dev-only eight-knot warm-path tangent from the exact JAX cap state."""

from __future__ import annotations

import gc
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
from dev.native_rom.eob.torch.warm_start import _FrequencyCrossing
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator as JaxIntegrator
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from torchwave.approximants.eob.adaptive_driver_clean import (
    GSL_STATE_SCALED_CONTROLLER,
    fixed_fehlberg45_step as torch_fixed,
    integrate_adaptive,
)
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator as TorchIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/native_warm_eight_step_tangent_v2_plan.json"
OUTPUT = PLAN.with_name("native_warm_eight_step_tangent_v2.json")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(data):
    OUTPUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def scaled_error(actual, expected):
    return float(np.max(np.abs(actual - expected) / np.maximum(1.0, np.abs(expected))))


def main():
    plan = json.loads(PLAN.read_text())
    if sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen eight-step script changed")
    for path, expected in plan["inputs_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"frozen input changed: {path}")
    for key in ("prior_one_step_evidence", "prior_mesh_evidence", "prior_failed_evidence"):
        prior = plan[key]
        if sha256(ROOT / prior["path"]) != prior["sha256"]:
            raise ValueError(f"frozen prior evidence changed: {key}")
    case = plan["case"]
    result = {
        "schema": "diffgw.seobnrv4.native-warm-eight-step-tangent.v2",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": sha256(PLAN), "script_sha256": sha256(Path(__file__)),
        "input_sha256": plan["inputs_sha256"],
        "prior_one_step_evidence": plan["prior_one_step_evidence"],
        "prior_mesh_evidence": plan["prior_mesh_evidence"],
        "prior_failed_evidence": plan["prior_failed_evidence"],
        "case": case,
    }
    write(result)
    started = time.monotonic()
    try:
        jax.config.update("jax_enable_x64", True)
        torch.set_num_threads(1)
        m1, m2 = case["mass1"], case["mass2"]
        nu = m1 * m2 / (m1 + m2) ** 2
        delta = (m1 - m2) / (m1 + m2)
        chi0 = case["chi1"]
        state0 = np.asarray(case["frozen_state"], dtype=np.float64)
        tangent0 = np.asarray(case["state_tangent_per_chi1"], dtype=np.float64)
        tf, jf = TorchFlux(**SOURCE_FLUX_OPTIONS), JaxFlux(**SOURCE_FLUX_OPTIONS)

        def ttensor(value, *, grad=False):
            return torch.tensor(value, dtype=torch.float64, requires_grad=grad)

        def torch_field_at(chi):
            ham = TorchHamiltonian(ttensor(nu), chi, ttensor(case["chi2"]),
                                   delta=ttensor(delta))
            return TorchIntegrator(ham, tf).vector_field

        mesh_plan = plan["mesh_discovery"]
        discovery = integrate_adaptive(
            torch_field_at(ttensor(chi0)), ttensor(state0),
            mesh_plan["initial_step_M"],
            event=_FrequencyCrossing(ttensor(case["near_cap_target_Momega"])),
            atol=mesh_plan["atol"], rtol=mesh_plan["rtol"],
            initial_step=mesh_plan["initial_step_M"],
            controller=GSL_STATE_SCALED_CONTROLLER,
            max_duration=mesh_plan["max_duration_M"],
        )
        times = np.asarray(discovery.time.detach())
        if (discovery.statistics.termination != ("event",)
                or discovery.statistics.total_accepted != mesh_plan["expected_accepted_steps"]
                or abs(times[-2] - mesh_plan["expected_left_time_M"])
                > mesh_plan["time_check_abs_M_at_most"]
                or abs(times[-1] - mesh_plan["expected_right_time_M"])
                > mesh_plan["time_check_abs_M_at_most"]):
            raise RuntimeError("detached mesh differs from frozen near-cap bracket")
        result["mesh"] = {
            "accepted_time_M": times.tolist(),
            "accepted_steps": int(discovery.statistics.total_accepted),
            "evaluations": int(discovery.statistics.total_evaluations),
            "last_left_frequency": float(discovery.derivative[-2, 1]),
            "last_right_frequency": float(discovery.derivative[-1, 1]),
        }
        write(result)
        del discovery
        gc.collect()

        def torch_step(spin, state, step):
            return torch_fixed(torch_field_at(spin), 0.0, state, step).fifth

        tvalue = state0.copy()
        ttangent = tangent0.copy()
        for index, step in enumerate(np.diff(times)):
            tchi = ttensor(chi0, grad=True)
            tstate = ttensor(tvalue) + (tchi - chi0) * ttensor(ttangent)
            tout = torch_step(tchi, tstate, float(step))
            next_value = np.asarray(tout.detach()).copy()
            next_tangent = np.asarray(torch.stack([
                torch.autograd.grad(component, tchi, retain_graph=True)[0]
                for component in tout
            ]).detach()).copy()
            tvalue, ttangent = next_value, next_tangent
            result["torch_progress"] = {"completed_steps": index + 1}
            write(result)
            del tchi, tstate, tout
            gc.collect()
        result["torch"] = {
            "value": tvalue.tolist(), "directional_derivative": ttangent.tolist(),
        }
        write(result)

        def jax_field_at(chi, state):
            ham = JaxHamiltonian(jnp.asarray(nu), chi, jnp.asarray(case["chi2"]),
                                 delta=jnp.asarray(delta))
            return JaxIntegrator(ham, jf).vector_field(state)

        compiled_field = jax.jit(jax_field_at)

        def field_directional(chi, state, dchi, dstate):
            return jax.jvp(jax_field_at, (chi, state), (dchi, dstate))

        compiled_directional = jax.jit(field_directional)

        def stagewise(spin, initial, step, initial_tangent=None):
            stages, tangent_stages = [], []
            for index in range(6):
                argument = initial.copy()
                argument_tangent = None if initial_tangent is None else initial_tangent.copy()
                for earlier, weight in enumerate(_B[index]):
                    argument += step * weight * stages[earlier]
                    if argument_tangent is not None:
                        argument_tangent += step * weight * tangent_stages[earlier]
                if argument_tangent is None:
                    stages.append(np.asarray(compiled_field(jnp.asarray(spin),
                                                            jnp.asarray(argument))))
                else:
                    field, direction = compiled_directional(
                        jnp.asarray(spin), jnp.asarray(argument),
                        jnp.asarray(case["chi1_tangent"]), jnp.asarray(argument_tangent),
                    )
                    stages.append(np.asarray(field))
                    tangent_stages.append(np.asarray(direction))
            value = initial.copy()
            derivative = None if initial_tangent is None else initial_tangent.copy()
            for index, weight in enumerate(_B5):
                value += step * weight * stages[index]
                if derivative is not None:
                    derivative += step * weight * tangent_stages[index]
            return value, derivative

        jvalue = state0.copy()
        jtangent = tangent0.copy()
        for index, step in enumerate(np.diff(times)):
            jvalue, jtangent = stagewise(chi0, jvalue, float(step), jtangent)
            result["jax_progress"] = {"completed_steps": index + 1}
            write(result)
        result["jax"] = {
            "value": jvalue.tolist(), "directional_derivative": jtangent.tolist(),
        }
        write(result)

        def torch_replay(spin, initial):
            value = ttensor(initial)
            for step in np.diff(times):
                value = torch_step(ttensor(spin), value, float(step))
            return np.asarray(value.detach())

        def jax_replay(spin, initial):
            value = initial.copy()
            for step in np.diff(times):
                value, _ = stagewise(spin, value, float(step))
            return value

        finite_differences = {}
        for fd_step in plan["finite_difference_chi1_steps"]:
            tfd = (torch_replay(chi0 + fd_step, state0 + fd_step * tangent0)
                   - torch_replay(chi0 - fd_step, state0 - fd_step * tangent0)) / (2 * fd_step)
            jfd = (jax_replay(chi0 + fd_step, state0 + fd_step * tangent0)
                   - jax_replay(chi0 - fd_step, state0 - fd_step * tangent0)) / (2 * fd_step)
            finite_differences[str(fd_step)] = {
                "torch": tfd.tolist(), "jax": jfd.tolist(),
                "torch_autodiff_scaled_error": scaled_error(ttangent, tfd),
                "jax_tangent_scaled_error": scaled_error(jtangent, jfd),
            }
            result["finite_differences"] = finite_differences
            write(result)
        metrics = {
            "final_state_torch_jax_max_abs": float(np.max(np.abs(tvalue - jvalue))),
            "final_tangent_torch_jax_scaled_error": scaled_error(ttangent, jtangent),
            "final_tangent_jax_fd_scaled_error_max": max(
                x["jax_tangent_scaled_error"] for x in finite_differences.values()),
            "final_tangent_torch_fd_scaled_error_max": max(
                x["torch_autodiff_scaled_error"] for x in finite_differences.values()),
        }
        result["metrics"] = metrics
        metric_for_gate = {
            "final_state_torch_jax_max_abs_at_most": "final_state_torch_jax_max_abs",
            "final_tangent_torch_jax_scaled_error_at_most": "final_tangent_torch_jax_scaled_error",
            "final_tangent_jax_fd_scaled_error_at_most_each": "final_tangent_jax_fd_scaled_error_max",
            "final_tangent_torch_fd_scaled_error_at_most_each": "final_tangent_torch_fd_scaled_error_max",
        }
        gates = {"finite": bool(all(np.isfinite(x).all()
                                    for x in (tvalue, ttangent, jvalue, jtangent)))}
        gates.update({name: metrics[metric_for_gate[name]] <= threshold
                      for name, threshold in plan["checks"].items()})
        result["gates"] = gates
        result["diagnostic_checks_passed"] = all(gates.values())
        result["status"] = "diagnostic_complete"
    except Exception:
        result["status"] = "diagnostic_failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - started
        write(result)


if __name__ == "__main__":
    main()

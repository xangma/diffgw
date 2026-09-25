"""Dev-only implicit frequency-crossing tangent on a fixed eight-step mesh."""

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
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator as JaxIntegrator
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from torchwave.approximants.eob.adaptive_driver_clean import fixed_fehlberg45_step as torch_fixed
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator as TorchIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/native_warm_crossing_tangent_v2_plan.json"
OUTPUT = PLAN.with_name("native_warm_crossing_tangent_v2.json")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(data):
    OUTPUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def scaled_error(actual, expected):
    return float(np.max(np.abs(actual - expected) / np.maximum(1.0, np.abs(expected))))


def main():
    plan = json.loads(PLAN.read_text())
    if sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen crossing script changed")
    for path, expected in plan["inputs_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"frozen input changed: {path}")
    for key in ("mesh_source", "prior_failed_evidence", "prior_crossing_failure"):
        prior = plan[key]
        if sha256(ROOT / prior["path"]) != prior["sha256"]:
            raise ValueError(f"frozen prior evidence changed: {key}")
    source = json.loads((ROOT / plan["mesh_source"]["path"]).read_text())
    case = plan["case"]
    if source["status"] != "diagnostic_complete" or not source["diagnostic_checks_passed"]:
        raise ValueError("frozen eight-step source did not pass")
    if case["frozen_state"] != source["case"]["frozen_state"]:
        raise ValueError("crossing cap state differs from fixed-mesh source")
    if case["state_tangent_per_chi1"] != source["case"]["state_tangent_per_chi1"]:
        raise ValueError("crossing cap tangent differs from fixed-mesh source")
    result = {
        "schema": "diffgw.seobnrv4.native-warm-crossing-tangent.v2",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": sha256(PLAN), "script_sha256": sha256(Path(__file__)),
        "input_sha256": plan["inputs_sha256"],
        "mesh_source": plan["mesh_source"],
        "prior_failed_evidence": plan["prior_failed_evidence"],
        "prior_crossing_failure": plan["prior_crossing_failure"],
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
        chi0, target = case["chi1"], case["target_Momega"]
        state0 = np.asarray(case["frozen_state"], dtype=np.float64)
        tangent0 = np.asarray(case["state_tangent_per_chi1"], dtype=np.float64)
        times = np.asarray(source["mesh"]["accepted_time_M"], dtype=np.float64)
        if (source["mesh"]["accepted_steps"] != plan["mesh_gate"]["accepted_steps"]
                or len(times) != 9
                or abs(times[-2] - plan["mesh_gate"]["expected_left_time_M"])
                > plan["mesh_gate"]["time_abs_at_most_M"]
                or abs(times[-1] - plan["mesh_gate"]["expected_right_time_M"])
                > plan["mesh_gate"]["time_abs_at_most_M"]):
            raise RuntimeError("fixed accepted mesh does not meet frozen gate")
        left_time, right_time = times[-2:]
        width = float(right_time - left_time)
        tf, jf = TorchFlux(**SOURCE_FLUX_OPTIONS), JaxFlux(**SOURCE_FLUX_OPTIONS)

        def ttensor(value, *, grad=False):
            return torch.tensor(value, dtype=torch.float64, requires_grad=grad)

        def torch_field_at(chi):
            ham = TorchHamiltonian(ttensor(nu), chi, ttensor(case["chi2"]),
                                   delta=ttensor(delta))
            return TorchIntegrator(ham, tf).vector_field

        def torch_step(spin, state, step):
            return torch_fixed(torch_field_at(spin), 0.0, state, step).fifth

        tleft, tleft_tangent = state0.copy(), tangent0.copy()
        for index, step in enumerate(np.diff(times[:-1])):
            tchi = ttensor(chi0, grad=True)
            tstate = ttensor(tleft) + (tchi - chi0) * ttensor(tleft_tangent)
            tout = torch_step(tchi, tstate, float(step))
            tleft_next = np.asarray(tout.detach()).copy()
            tleft_tangent_next = np.asarray(torch.stack([
                torch.autograd.grad(component, tchi, retain_graph=True)[0]
                for component in tout
            ]).detach()).copy()
            tleft, tleft_tangent = tleft_next, tleft_tangent_next
            result["torch_progress"] = {"completed_pre_crossing_steps": index + 1}
            write(result)
            del tchi, tstate, tout
            gc.collect()
        result["left_torch"] = {
            "state": tleft.tolist(), "spin_tangent": tleft_tangent.tolist(),
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

        def stagewise(spin, initial, step, initial_tangent=None,
                      spin_tangent=0.0, step_tangent=0.0):
            stages, tangent_stages = [], []
            for index in range(6):
                argument = initial.copy()
                argument_tangent = None if initial_tangent is None else initial_tangent.copy()
                for earlier, weight in enumerate(_B[index]):
                    argument += step * weight * stages[earlier]
                    if argument_tangent is not None:
                        argument_tangent += (step_tangent * weight * stages[earlier]
                                             + step * weight * tangent_stages[earlier])
                if argument_tangent is None:
                    stages.append(np.asarray(compiled_field(jnp.asarray(spin),
                                                            jnp.asarray(argument))))
                else:
                    field, direction = compiled_directional(
                        jnp.asarray(spin), jnp.asarray(argument),
                        jnp.asarray(spin_tangent), jnp.asarray(argument_tangent),
                    )
                    stages.append(np.asarray(field))
                    tangent_stages.append(np.asarray(direction))
            value = initial.copy()
            derivative = None if initial_tangent is None else initial_tangent.copy()
            for index, weight in enumerate(_B5):
                value += step * weight * stages[index]
                if derivative is not None:
                    derivative += (step_tangent * weight * stages[index]
                                   + step * weight * tangent_stages[index])
            return value, derivative

        jleft, jleft_tangent = state0.copy(), tangent0.copy()
        for index, step in enumerate(np.diff(times[:-1])):
            jleft, jleft_tangent = stagewise(chi0, jleft, float(step),
                                            jleft_tangent, spin_tangent=1.0)
            result["jax_progress"] = {"completed_pre_crossing_steps": index + 1}
            write(result)
        result["left_jax"] = {
            "state": jleft.tolist(), "spin_tangent": jleft_tangent.tolist(),
        }
        write(result)

        def torch_frequency_at(offset, spin=chi0, left=tleft):
            field = torch_field_at(ttensor(spin))
            if offset == 0.0:
                return float(field(ttensor(left))[1].detach())
            trial = torch_fixed(field, ttensor(left_time), ttensor(left),
                                ttensor(offset)).fifth
            return float(field(trial)[1].detach())

        left_frequency = torch_frequency_at(0.0)
        right_frequency = torch_frequency_at(width)
        if not left_frequency < target <= right_frequency:
            raise RuntimeError("frozen last interval does not bracket frequency")
        low, high = 0.0, width
        for _ in range(plan["root"]["bisection_iterations"]):
            offset = 0.5 * (low + high)
            if torch_frequency_at(offset) < target:
                low = offset
            else:
                high = offset
        root_offset = 0.5 * (low + high)
        result["root"] = {
            "offset_M": root_offset, "left_frequency": left_frequency,
            "right_frequency": right_frequency,
            "torch_bisection_residual": torch_frequency_at(root_offset) - target,
        }
        write(result)

        tchi = ttensor(chi0, grad=True)
        th = ttensor(root_offset, grad=True)
        tleft_graph = ttensor(tleft) + (tchi - chi0) * ttensor(tleft_tangent)
        tfield = torch_field_at(tchi)
        tcross = torch_fixed(tfield, ttensor(left_time), tleft_graph, th).fifth
        tomega = tfield(tcross)[1]
        tomega_chi, tomega_h = torch.autograd.grad(tomega, (tchi, th), retain_graph=True)
        tstate_chi, tstate_h = [], []
        for component in tcross:
            spin_grad, time_grad = torch.autograd.grad(
                component, (tchi, th), retain_graph=True,
            )
            tstate_chi.append(float(spin_grad))
            tstate_h.append(float(time_grad))
        tstate_chi, tstate_h = np.asarray(tstate_chi), np.asarray(tstate_h)
        tslope = float(tomega_h)
        tdt_dchi, tdt_dtarget = -float(tomega_chi) / tslope, 1.0 / tslope
        selected = [0, 2, 3]
        tvalue = np.asarray([*np.asarray(tcross.detach())[selected], left_time + root_offset])
        tjac = np.stack((
            np.asarray([*(tstate_chi[selected] + tstate_h[selected] * tdt_dchi),
                        tdt_dchi]),
            np.asarray([*(tstate_h[selected] * tdt_dtarget), tdt_dtarget]),
        ), axis=1)
        result["torch"] = {
            "value": tvalue.tolist(), "jacobian": tjac.tolist(),
            "frequency_residual": float(tomega.detach()) - target,
            "frequency_time_slope": tslope,
            "event_time_derivative": [tdt_dchi, tdt_dtarget],
        }
        write(result)
        del tchi, th, tleft_graph, tfield, tcross, tomega
        gc.collect()

        jcross, jstate_chi = stagewise(chi0, jleft, root_offset,
                                      jleft_tangent, spin_tangent=1.0)
        jcross_time, jstate_h = stagewise(chi0, jleft, root_offset,
                                         np.zeros_like(jleft), step_tangent=1.0)
        _, jfrequency_chi = compiled_directional(
            jnp.asarray(chi0), jnp.asarray(jcross), jnp.asarray(1.0),
            jnp.asarray(jstate_chi),
        )
        jfield, jfrequency_h = compiled_directional(
            jnp.asarray(chi0), jnp.asarray(jcross), jnp.asarray(0.0),
            jnp.asarray(jstate_h),
        )
        jslope = float(jfrequency_h[1])
        jdt_dchi, jdt_dtarget = -float(jfrequency_chi[1]) / jslope, 1.0 / jslope
        jvalue = np.asarray([*jcross[selected], left_time + root_offset])
        jjac = np.stack((
            np.asarray([*(jstate_chi[selected] + jstate_h[selected] * jdt_dchi),
                        jdt_dchi]),
            np.asarray([*(jstate_h[selected] * jdt_dtarget), jdt_dtarget]),
        ), axis=1)
        result["jax"] = {
            "value": jvalue.tolist(), "jacobian": jjac.tolist(),
            "frequency_residual": float(jfield[1]) - target,
            "frequency_time_slope": jslope,
            "event_time_derivative": [jdt_dchi, jdt_dtarget],
            "two_direction_primal_max_abs": float(np.max(np.abs(jcross - jcross_time))),
        }
        write(result)

        def torch_value(spin, requested):
            left = ttensor(state0 + (spin - chi0) * tangent0)
            for step in np.diff(times[:-1]):
                left = torch_step(ttensor(spin), left, float(step))
            field = torch_field_at(ttensor(spin))
            offset = root_offset
            for _ in range(plan["root"]["finite_difference_newton_iterations"]):
                crossed = torch_fixed(field, ttensor(left_time), left,
                                      ttensor(offset)).fifth
                offset -= (float(field(crossed)[1].detach()) - requested) / tslope
            crossed = torch_fixed(field, ttensor(left_time), left,
                                  ttensor(offset)).fifth
            residual = float(field(crossed)[1].detach()) - requested
            return np.asarray([*np.asarray(crossed.detach())[selected],
                               left_time + offset]), residual, offset

        def jax_value(spin, requested):
            left = state0 + (spin - chi0) * tangent0
            for step in np.diff(times[:-1]):
                left, _ = stagewise(spin, left, float(step))
            offset = root_offset
            for _ in range(plan["root"]["finite_difference_newton_iterations"]):
                crossed, _ = stagewise(spin, left, offset)
                offset -= (float(compiled_field(jnp.asarray(spin),
                                                jnp.asarray(crossed))[1])
                           - requested) / jslope
            crossed, _ = stagewise(spin, left, offset)
            residual = (float(compiled_field(jnp.asarray(spin),
                                             jnp.asarray(crossed))[1]) - requested)
            return np.asarray([*crossed[selected], left_time + offset]), residual, offset

        finite_differences = {}
        for column, (parameter, steps) in enumerate(plan["finite_difference_steps"].items()):
            for step in steps:
                args_plus = [chi0, target]
                args_minus = [chi0, target]
                args_plus[column] += step
                args_minus[column] -= step
                tp, trp, top = torch_value(*args_plus)
                tm, trm, tom = torch_value(*args_minus)
                jp, jrp, jop = jax_value(*args_plus)
                jm, jrm, jom = jax_value(*args_minus)
                tfd, jfd = (tp - tm) / (2 * step), (jp - jm) / (2 * step)
                finite_differences[f"{parameter}:{step}"] = {
                    "parameter": parameter, "step": step, "column": column,
                    "torch": tfd.tolist(), "jax": jfd.tolist(),
                    "torch_scaled_error": scaled_error(tjac[:, column], tfd),
                    "jax_scaled_error": scaled_error(jjac[:, column], jfd),
                    "max_abs_frequency_residual": max(map(abs, (trp, trm, jrp, jrm))),
                    "root_offsets_M": [top, tom, jop, jom],
                }
                result["finite_differences"] = finite_differences
                write(result)

        metrics = {
            "left_state_torch_jax_max_abs": float(np.max(np.abs(tleft - jleft))),
            "left_tangent_torch_jax_scaled_error": scaled_error(tleft_tangent, jleft_tangent),
            "event_time_torch_jax_abs_M": float(abs(tvalue[-1] - jvalue[-1])),
            "phase_zero_state_torch_jax_max_abs": float(np.max(np.abs(tvalue[:3] - jvalue[:3]))),
            "jacobian_torch_jax_scaled_error": scaled_error(tjac, jjac),
            "jacobian_jax_fd_scaled_error_max": max(x["jax_scaled_error"]
                                                    for x in finite_differences.values()),
            "jacobian_torch_fd_scaled_error_max": max(x["torch_scaled_error"]
                                                      for x in finite_differences.values()),
            "frequency_residual_abs_max": max(
                abs(result["torch"]["frequency_residual"]),
                abs(result["jax"]["frequency_residual"]),
                *(x["max_abs_frequency_residual"] for x in finite_differences.values()),
            ),
        }
        result["metrics"] = metrics
        metric_for_gate = {
            "left_state_torch_jax_max_abs_at_most": "left_state_torch_jax_max_abs",
            "left_tangent_torch_jax_scaled_error_at_most": "left_tangent_torch_jax_scaled_error",
            "event_time_torch_jax_abs_at_most_M": "event_time_torch_jax_abs_M",
            "phase_zero_state_torch_jax_max_abs_at_most": "phase_zero_state_torch_jax_max_abs",
            "jacobian_torch_jax_scaled_error_at_most": "jacobian_torch_jax_scaled_error",
            "jacobian_jax_fd_scaled_error_at_most_each": "jacobian_jax_fd_scaled_error_max",
            "jacobian_torch_fd_scaled_error_at_most_each": "jacobian_torch_fd_scaled_error_max",
        }
        gates = {
            "finite": bool(all(np.isfinite(x).all() for x in (tleft, tleft_tangent,
                         jleft, jleft_tangent, tvalue, jvalue, tjac, jjac))),
            "root_bracket": bool(0 < root_offset < width),
            "positive_crossing_slope": bool(tslope > 0 and jslope > 0),
            "frequency_residual_abs_at_most": (
                metrics["frequency_residual_abs_max"]
                <= plan["root"]["frequency_residual_abs_at_most"]),
            "finite_difference_roots_in_bracket": all(
                0 < offset < width
                for item in finite_differences.values()
                for offset in item["root_offsets_M"]),
        }
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

"""Dev-only final-interval 0.04 crossing and local event sensitivities."""

from __future__ import annotations

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
PLAN = EVIDENCE / "native_warm_full_crossing_plan.json"
OUTPUT = EVIDENCE / "native_warm_full_crossing.json"


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
            raise ValueError(f"frozen source changed: {path}")
    for key in ("mesh_source", "phase_b_plan", "last_checkpoint"):
        reference = plan[key]
        if sha256(ROOT / reference["path"]) != reference["sha256"]:
            raise ValueError(f"frozen evidence changed: {key}")
    mesh_data = json.loads((ROOT / plan["mesh_source"]["path"]).read_text())
    checkpoint = json.loads((ROOT / plan["last_checkpoint"]["path"]).read_text())
    if (mesh_data["status"] != "diagnostic_complete"
            or not mesh_data["diagnostic_checks_passed"]
            or checkpoint["status"] != "diagnostic_complete"
            or not checkpoint["diagnostic_checks_passed"]
            or checkpoint["end_knot"] != mesh_data["mesh"]["pre_crossing_intervals"]
            or checkpoint["plan_sha256"] != plan["phase_b_plan"]["sha256"]
            or checkpoint["mesh_source"] != plan["mesh_source"]):
        raise ValueError("full-mesh checkpoint did not pass")
    case = mesh_data["case"]
    times = np.asarray(mesh_data["mesh"]["accepted_time_M"], dtype=np.float64)
    if (case["target_Momega"] != plan["target_Momega"]
            or times[-2:].tolist() != plan["bracket"]["accepted_time_M"]
            or mesh_data["mesh"]["last_bracket_frequency"]
            != plan["bracket"]["accepted_frequency"]):
        raise ValueError("target or last accepted bracket changed")
    left_time, right_time = times[-2:]
    width = float(right_time - left_time)
    result = {
        "schema": "diffgw.seobnrv4.native-warm-full-crossing.v1",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": sha256(PLAN), "script_sha256": sha256(Path(__file__)),
        "input_sha256": plan["inputs_sha256"],
        "mesh_source": plan["mesh_source"], "phase_b_plan": plan["phase_b_plan"],
        "last_checkpoint": plan["last_checkpoint"],
        "case": case, "left_time_M": float(left_time), "right_time_M": float(right_time),
    }
    write(result)
    started = time.monotonic()
    try:
        jax.config.update("jax_enable_x64", True)
        torch.set_num_threads(1)
        mass1, mass2 = case["mass1"], case["mass2"]
        nu = mass1 * mass2 / (mass1 + mass2) ** 2
        delta = (mass1 - mass2) / (mass1 + mass2)
        chi0, chi2, target = case["chi1"], case["chi2"], case["target_Momega"]
        tf, jf = TorchFlux(**SOURCE_FLUX_OPTIONS), JaxFlux(**SOURCE_FLUX_OPTIONS)
        tleft = np.asarray(checkpoint["torch"]["state"], dtype=np.float64)
        tleft_tangent = np.asarray(checkpoint["torch"]["spin_tangent"], dtype=np.float64)
        jleft = np.asarray(checkpoint["jax"]["state"], dtype=np.float64)
        jleft_tangent = np.asarray(checkpoint["jax"]["spin_tangent"], dtype=np.float64)
        cap_state = np.asarray(case["frozen_state"], dtype=np.float64)
        cap_tangent = np.asarray(case["state_tangent_per_chi1"], dtype=np.float64)

        def tensor(value, *, grad=False):
            return torch.tensor(value, dtype=torch.float64, requires_grad=grad)

        def torch_field(spin):
            ham = TorchHamiltonian(tensor(nu), spin, tensor(chi2), delta=tensor(delta))
            return TorchIntegrator(ham, tf).vector_field

        def torch_frequency(offset):
            field = torch_field(tensor(chi0))
            if offset == 0.0:
                return float(field(tensor(tleft))[1].detach())
            trial = fixed_fehlberg45_step(field, tensor(left_time), tensor(tleft),
                                          tensor(offset)).fifth
            return float(field(trial)[1].detach())

        left_frequency = torch_frequency(0.0)
        right_frequency = torch_frequency(width)
        if not left_frequency < target <= right_frequency:
            raise RuntimeError("final fixed interval does not bracket 0.04")
        low, high = 0.0, width
        for _ in range(plan["root"]["bisection_iterations"]):
            middle = 0.5 * (low + high)
            if torch_frequency(middle) < target:
                low = middle
            else:
                high = middle
        root_offset = 0.5 * (low + high)
        result["root"] = {
            "offset_M": root_offset,
            "frequency_left": left_frequency,
            "frequency_right": right_frequency,
            "bisection_residual": torch_frequency(root_offset) - target,
        }
        write(result)

        spin = tensor(chi0, grad=True)
        offset = tensor(root_offset, grad=True)
        left_graph = tensor(tleft) + (spin - chi0) * tensor(tleft_tangent)
        field = torch_field(spin)
        crossing = fixed_fehlberg45_step(field, tensor(left_time), left_graph, offset).fifth
        omega = field(crossing)[1]
        omega_spin, omega_offset = torch.autograd.grad(omega, (spin, offset), retain_graph=True)
        state_spin, state_offset = [], []
        for component in crossing:
            dspin, doffset = torch.autograd.grad(component, (spin, offset), retain_graph=True)
            state_spin.append(float(dspin))
            state_offset.append(float(doffset))
        state_spin, state_offset = np.asarray(state_spin), np.asarray(state_offset)
        tslope = float(omega_offset)
        tdt_spin, tdt_target = -float(omega_spin) / tslope, 1.0 / tslope
        selected = [0, 2, 3]
        tvalue = np.asarray([*np.asarray(crossing.detach())[selected], left_time + root_offset])
        tjac = np.stack((
            np.asarray([*(state_spin[selected] + state_offset[selected] * tdt_spin),
                        tdt_spin]),
            np.asarray([*(state_offset[selected] * tdt_target), tdt_target]),
        ), axis=1)
        result["torch"] = {
            "value": tvalue.tolist(), "jacobian": tjac.tolist(),
            "frequency_residual": float(omega.detach()) - target,
            "frequency_time_slope": tslope,
            "event_time_derivative": [tdt_spin, tdt_target],
        }
        write(result)

        def jax_field(spin, state):
            ham = JaxHamiltonian(jnp.asarray(nu), spin, jnp.asarray(chi2),
                                 delta=jnp.asarray(delta))
            return JaxIntegrator(ham, jf).vector_field(state)

        compiled_field = jax.jit(jax_field)

        def directional(spin, state, dspin, dstate):
            return jax.jvp(jax_field, (spin, state), (dspin, dstate))

        compiled_directional = jax.jit(directional)

        def jax_step(spin, initial, step, initial_tangent=None,
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
                    value, derivative = compiled_directional(
                        jnp.asarray(spin), jnp.asarray(argument),
                        jnp.asarray(spin_tangent), jnp.asarray(argument_tangent),
                    )
                    stages.append(np.asarray(value))
                    tangent_stages.append(np.asarray(derivative))
            final = initial.copy()
            final_tangent = None if initial_tangent is None else initial_tangent.copy()
            for index, weight in enumerate(_B5):
                final += step * weight * stages[index]
                if final_tangent is not None:
                    final_tangent += (step_tangent * weight * stages[index]
                                      + step * weight * tangent_stages[index])
            return final, final_tangent

        jcross, jstate_spin = jax_step(chi0, jleft, root_offset, jleft_tangent,
                                      spin_tangent=1.0)
        jcross_time, jstate_offset = jax_step(chi0, jleft, root_offset,
                                             np.zeros_like(jleft), step_tangent=1.0)
        jfield, jomega_spin = compiled_directional(
            jnp.asarray(chi0), jnp.asarray(jcross), jnp.asarray(1.0),
            jnp.asarray(jstate_spin),
        )
        _, jomega_offset = compiled_directional(
            jnp.asarray(chi0), jnp.asarray(jcross), jnp.asarray(0.0),
            jnp.asarray(jstate_offset),
        )
        jslope = float(jomega_offset[1])
        jdt_spin, jdt_target = -float(jomega_spin[1]) / jslope, 1.0 / jslope
        jvalue = np.asarray([*jcross[selected], left_time + root_offset])
        jjac = np.stack((
            np.asarray([*(jstate_spin[selected] + jstate_offset[selected] * jdt_spin),
                        jdt_spin]),
            np.asarray([*(jstate_offset[selected] * jdt_target), jdt_target]),
        ), axis=1)
        result["jax"] = {
            "value": jvalue.tolist(), "jacobian": jjac.tolist(),
            "frequency_residual": float(jfield[1]) - target,
            "frequency_time_slope": jslope,
            "event_time_derivative": [jdt_spin, jdt_target],
            "two_direction_primal_max_abs": float(np.max(np.abs(jcross - jcross_time))),
        }
        write(result)

        def torch_value(spin_value, requested):
            spin_const = tensor(spin_value)
            field_const = torch_field(spin_const)
            if spin_value == chi0:
                left = tensor(tleft)
            else:
                left = tensor(cap_state + (spin_value - chi0) * cap_tangent)
                for step in np.diff(times[:-1]):
                    left = fixed_fehlberg45_step(field_const, tensor(0.0), left,
                                                 tensor(float(step))).fifth
            root = root_offset
            for _ in range(plan["root"]["finite_difference_newton_iterations"]):
                crossed = fixed_fehlberg45_step(field_const, tensor(left_time),
                                                left, tensor(root)).fifth
                root -= (float(field_const(crossed)[1].detach()) - requested) / tslope
            crossed = fixed_fehlberg45_step(field_const, tensor(left_time),
                                            left, tensor(root)).fifth
            residual = float(field_const(crossed)[1].detach()) - requested
            return np.asarray([*np.asarray(crossed.detach())[selected],
                               left_time + root]), residual, root

        def jax_value(spin_value, requested):
            if spin_value == chi0:
                left = jleft
            else:
                left = cap_state + (spin_value - chi0) * cap_tangent
                for step in np.diff(times[:-1]):
                    left, _ = jax_step(spin_value, left, float(step))
            root = root_offset
            for _ in range(plan["root"]["finite_difference_newton_iterations"]):
                crossed, _ = jax_step(spin_value, left, root)
                root -= (float(compiled_field(jnp.asarray(spin_value),
                                              jnp.asarray(crossed))[1]) - requested) / jslope
            crossed, _ = jax_step(spin_value, left, root)
            residual = float(compiled_field(jnp.asarray(spin_value),
                                            jnp.asarray(crossed))[1]) - requested
            return np.asarray([*crossed[selected], left_time + root]), residual, root

        finite_differences = {}
        for column, (parameter, steps) in enumerate(plan["finite_difference_steps"].items()):
            for step in steps:
                plus, minus = [chi0, target], [chi0, target]
                plus[column] += step
                minus[column] -= step
                tp, trp, top = torch_value(*plus)
                tm, trm, tom = torch_value(*minus)
                jp, jrp, jop = jax_value(*plus)
                jm, jrm, jom = jax_value(*minus)
                tfd, jfd = (tp - tm) / (2 * step), (jp - jm) / (2 * step)
                finite_differences[f"{parameter}:{step}"] = {
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
            "left_tangent_torch_jax_scaled": scaled_error(tleft_tangent, jleft_tangent),
            "event_time_torch_jax_abs_M": float(abs(tvalue[-1] - jvalue[-1])),
            "crossing_state_torch_jax_max_abs": float(np.max(np.abs(tvalue[:3] - jvalue[:3]))),
            "jacobian_torch_jax_scaled": scaled_error(tjac, jjac),
            "jacobian_torch_fd_scaled_max": max(x["torch_scaled_error"]
                                                for x in finite_differences.values()),
            "jacobian_jax_fd_scaled_max": max(x["jax_scaled_error"]
                                              for x in finite_differences.values()),
            "frequency_residual_abs_max": max(
                abs(result["torch"]["frequency_residual"]),
                abs(result["jax"]["frequency_residual"]),
                *(x["max_abs_frequency_residual"] for x in finite_differences.values()),
            ),
        }
        result["metrics"] = metrics
        metric_keys = {
            "left_state_torch_jax_max_abs_at_most": "left_state_torch_jax_max_abs",
            "left_tangent_torch_jax_scaled_at_most": "left_tangent_torch_jax_scaled",
            "event_time_torch_jax_abs_M_at_most": "event_time_torch_jax_abs_M",
            "crossing_state_torch_jax_max_abs_at_most": "crossing_state_torch_jax_max_abs",
            "jacobian_torch_jax_scaled_at_most": "jacobian_torch_jax_scaled",
            "jacobian_torch_fd_scaled_at_most_each": "jacobian_torch_fd_scaled_max",
            "jacobian_jax_fd_scaled_at_most_each": "jacobian_jax_fd_scaled_max",
        }
        gates = {
            "finite": bool(all(np.isfinite(x).all() for x in (tvalue, jvalue, tjac, jjac))),
            "target_bracket": bool(left_frequency < target <= right_frequency),
            "root_inside_last_interval": bool(0 < root_offset < width),
            "positive_frequency_time_slope": bool(tslope > 0 and jslope > 0),
            "frequency_residual_abs_at_most": (
                metrics["frequency_residual_abs_max"]
                <= plan["root"]["frequency_residual_abs_at_most"]),
            "finite_difference_roots_inside_last_interval": all(
                0 < root < width for item in finite_differences.values()
                for root in item["root_offsets_M"]),
        }
        gates.update({name: metrics[key] <= plan["checks"][name]
                      for name, key in metric_keys.items()})
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

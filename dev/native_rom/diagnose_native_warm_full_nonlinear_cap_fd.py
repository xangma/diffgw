"""Dev-only nonlinear circular-cap finite differences on one frozen warm mesh."""

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
from dev.native_rom.eob.jax.initial_conditions_clean import (
    traceable_quasicircular_initial_conditions,
)
from dev.native_rom.eob.jax.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as JaxHamiltonian,
)
from dev.native_rom.eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as TorchHamiltonian,
)
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator as JaxIntegrator
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from torchwave.approximants.eob.adaptive_driver_clean import fixed_fehlberg45_step
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator as TorchIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "native_warm_full_nonlinear_cap_fd_plan.json"
OUTPUT = EVIDENCE / "native_warm_full_nonlinear_cap_fd.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(data):
    OUTPUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def scaled_error(actual, expected):
    return float(np.max(np.abs(actual - expected) / np.maximum(1.0, np.abs(expected))))


def main():
    plan = json.loads(PLAN.read_text())
    if sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen nonlinear-cap script changed")
    for path, expected in plan["inputs_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"frozen source changed: {path}")
    for key in ("mesh_source", "crossing_source"):
        reference = plan[key]
        if sha256(ROOT / reference["path"]) != reference["sha256"]:
            raise ValueError(f"frozen evidence changed: {key}")
    mesh_result = json.loads((ROOT / plan["mesh_source"]["path"]).read_text())
    crossing_result = json.loads((ROOT / plan["crossing_source"]["path"]).read_text())
    if (
        mesh_result["status"] != "diagnostic_complete"
        or not mesh_result["diagnostic_checks_passed"]
        or crossing_result["status"] != "diagnostic_complete"
        or not crossing_result["diagnostic_checks_passed"]
        or crossing_result["mesh_source"] != plan["mesh_source"]
    ):
        raise ValueError("prior fixed-mesh crossing evidence did not pass")
    case = mesh_result["case"]
    mesh = mesh_result["mesh"]
    if (
        case["target_Momega"] != plan["target_Momega"]
        or mesh["pre_crossing_intervals"] != plan["pre_crossing_intervals"]
        or mesh["last_bracket_time_M"] != plan["last_bracket_time_M"]
    ):
        raise ValueError("frozen target or mesh changed")
    times = np.asarray(mesh["accepted_time_M"], dtype=np.float64)
    if (
        len(times) != plan["pre_crossing_intervals"] + 2
        or times[0] != 0.0
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError("frozen accepted mesh is not finite and monotone")
    left_time, right_time = times[-2:]
    width = float(right_time - left_time)
    result = {
        "schema": "diffgw.seobnrv4.native-warm-full-nonlinear-cap-fd.v1",
        "status": "running",
        "release_qualified": False,
        "host": platform.node(),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "plan_sha256": sha256(PLAN),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": plan["inputs_sha256"],
        "mesh_source": plan["mesh_source"],
        "crossing_source": plan["crossing_source"],
        "case": case,
    }
    write(result)
    started = time.monotonic()
    try:
        jax.config.update("jax_enable_x64", True)
        torch.set_num_threads(1)
        mass1, mass2 = case["mass1"], case["mass2"]
        nu = mass1 * mass2 / (mass1 + mass2) ** 2
        delta = (mass1 - mass2) / (mass1 + mass2)
        chi0, chi2 = case["chi1"], case["chi2"]
        cap_frequency, target = case["cap_Momega"], case["target_Momega"]
        cap_baseline = np.asarray(case["frozen_state"], dtype=np.float64)
        cap_tangent = np.asarray(case["state_tangent_per_chi1"], dtype=np.float64)
        torch_event_tangent = np.asarray(
            crossing_result["torch"]["jacobian"],
            dtype=np.float64,
        )[:, 0]
        jax_event_tangent = np.asarray(
            crossing_result["jax"]["jacobian"],
            dtype=np.float64,
        )[:, 0]
        root_initial = crossing_result["root"]["offset_M"]
        torch_flux, jax_flux = TorchFlux(**SOURCE_FLUX_OPTIONS), JaxFlux(**SOURCE_FLUX_OPTIONS)

        def tensor(value):
            return torch.tensor(value, dtype=torch.float64)

        def torch_hamiltonian(spin):
            return TorchHamiltonian(tensor(nu), tensor(spin), tensor(chi2), delta=tensor(delta))

        def torch_cap(spin):
            values = find_quasicircular_initial_conditions(
                torch_hamiltonian(spin),
                torch_flux,
                tensor(cap_frequency),
            )
            return np.asarray([float(values[0]), 0.0, float(values[1]), float(values[2])])

        def jax_cap(spin):
            ham = JaxHamiltonian(jnp.asarray(nu), spin, jnp.asarray(chi2), delta=jnp.asarray(delta))
            r, pr, pphi = traceable_quasicircular_initial_conditions(
                ham,
                jax_flux,
                jnp.asarray(cap_frequency),
                max_iter=8,
            )
            return jnp.stack((r, jnp.asarray(0.0), pr, pphi))

        compiled_cap = jax.jit(jax_cap)

        def jax_field(spin, state):
            ham = JaxHamiltonian(jnp.asarray(nu), spin, jnp.asarray(chi2), delta=jnp.asarray(delta))
            return JaxIntegrator(ham, jax_flux).vector_field(state)

        compiled_field = jax.jit(jax_field)

        def jax_step(spin, state, h):
            stages = []
            for index in range(6):
                argument = state.copy()
                for earlier, weight in enumerate(_B[index]):
                    argument += h * weight * stages[earlier]
                stages.append(np.asarray(compiled_field(jnp.asarray(spin), jnp.asarray(argument))))
            final = state.copy()
            for index, weight in enumerate(_B5):
                final += h * weight * stages[index]
            return final

        def torch_event(spin, initial):
            field = TorchIntegrator(torch_hamiltonian(spin), torch_flux).vector_field
            state = tensor(initial)
            for h in np.diff(times[:-1]):
                state = fixed_fehlberg45_step(field, tensor(0.0), state, tensor(float(h))).fifth
            offset = root_initial
            for _ in range(plan["root"]["newton_iterations"]):
                crossed = fixed_fehlberg45_step(field, tensor(left_time), state, tensor(offset)).fifth
                offset -= (float(field(crossed)[1].detach()) - target) / plan["root"]["torch_baseline_slope"]
            crossed = fixed_fehlberg45_step(field, tensor(left_time), state, tensor(offset)).fifth
            residual = float(field(crossed)[1].detach()) - target
            return (
                np.asarray([float(crossed[0]), float(crossed[2]), float(crossed[3]), left_time + offset]),
                residual,
                offset,
            )

        def jax_event(spin, initial):
            state = initial.copy()
            for h in np.diff(times[:-1]):
                state = jax_step(spin, state, float(h))
            offset = root_initial
            for _ in range(plan["root"]["newton_iterations"]):
                crossed = jax_step(spin, state, offset)
                offset -= (float(compiled_field(jnp.asarray(spin), jnp.asarray(crossed))[1]) - target) / plan["root"][
                    "jax_baseline_slope"
                ]
            crossed = jax_step(spin, state, offset)
            residual = float(compiled_field(jnp.asarray(spin), jnp.asarray(crossed))[1]) - target
            return np.asarray([crossed[0], crossed[2], crossed[3], left_time + offset]), residual, offset

        baseline_torch = torch_cap(chi0)
        baseline_jax = np.asarray(compiled_cap(jnp.asarray(chi0)))
        result["baseline_cap"] = {"torch": baseline_torch.tolist(), "jax": baseline_jax.tolist()}
        write(result)
        print("baseline_cap_complete", flush=True)
        rows = {}
        for step in plan["finite_difference_chi1_steps"]:
            pair = {}
            for side, spin in (("plus", chi0 + step), ("minus", chi0 - step)):
                tcap = torch_cap(spin)
                jcap = np.asarray(compiled_cap(jnp.asarray(spin)))
                tevent, tresidual, toffset = torch_event(spin, tcap)
                jevent, jresidual, joffset = jax_event(spin, jcap)
                pair[side] = {
                    "spin": spin,
                    "torch_cap": tcap.tolist(),
                    "jax_cap": jcap.tolist(),
                    "torch_event": tevent.tolist(),
                    "jax_event": jevent.tolist(),
                    "torch_frequency_residual": tresidual,
                    "jax_frequency_residual": jresidual,
                    "torch_root_offset_M": toffset,
                    "jax_root_offset_M": joffset,
                }
                rows[str(step)] = pair
                result["perturbations"] = rows
                write(result)
                print(f"perturbation_complete step={step} side={side}", flush=True)
            plus, minus = pair["plus"], pair["minus"]
            tcap_fd = (np.asarray(plus["torch_cap"]) - np.asarray(minus["torch_cap"])) / (2 * step)
            jcap_fd = (np.asarray(plus["jax_cap"]) - np.asarray(minus["jax_cap"])) / (2 * step)
            tevent_fd = (np.asarray(plus["torch_event"]) - np.asarray(minus["torch_event"])) / (2 * step)
            jevent_fd = (np.asarray(plus["jax_event"]) - np.asarray(minus["jax_event"])) / (2 * step)
            rows[str(step)]["finite_difference"] = {
                "torch_cap": tcap_fd.tolist(),
                "jax_cap": jcap_fd.tolist(),
                "torch_event": tevent_fd.tolist(),
                "jax_event": jevent_fd.tolist(),
                "torch_cap_tangent_scaled_error": scaled_error(tcap_fd, cap_tangent),
                "jax_cap_tangent_scaled_error": scaled_error(jcap_fd, cap_tangent),
                "torch_event_tangent_scaled_error": scaled_error(
                    tevent_fd,
                    torch_event_tangent,
                ),
                "jax_event_tangent_scaled_error": scaled_error(
                    jevent_fd,
                    jax_event_tangent,
                ),
                "cap_fd_torch_jax_scaled_error": scaled_error(tcap_fd, jcap_fd),
                "event_fd_torch_jax_scaled_error": scaled_error(tevent_fd, jevent_fd),
            }
            result["perturbations"] = rows
            write(result)

        all_rows = [row for pair in rows.values() for name, row in pair.items() if name in ("plus", "minus")]
        all_fd = [pair["finite_difference"] for pair in rows.values()]
        first_fd, second_fd = all_fd
        metrics = {
            "baseline_cap_torch_jax_abs": float(np.max(np.abs(baseline_torch - baseline_jax))),
            "baseline_cap_to_frozen_abs": float(
                max(np.max(np.abs(baseline_torch - cap_baseline)), np.max(np.abs(baseline_jax - cap_baseline)))
            ),
            "perturbed_cap_torch_jax_abs_max": max(
                float(np.max(np.abs(np.asarray(row["torch_cap"]) - np.asarray(row["jax_cap"])))) for row in all_rows
            ),
            "perturbed_event_torch_jax_abs_max": max(
                float(np.max(np.abs(np.asarray(row["torch_event"]) - np.asarray(row["jax_event"])))) for row in all_rows
            ),
            "cap_tangent_fd_scaled_error_max": max(
                max(fd["torch_cap_tangent_scaled_error"], fd["jax_cap_tangent_scaled_error"]) for fd in all_fd
            ),
            "event_tangent_fd_scaled_error_max": max(
                max(fd["torch_event_tangent_scaled_error"], fd["jax_event_tangent_scaled_error"]) for fd in all_fd
            ),
            "cap_fd_torch_jax_scaled_error_max": max(fd["cap_fd_torch_jax_scaled_error"] for fd in all_fd),
            "event_fd_torch_jax_scaled_error_max": max(fd["event_fd_torch_jax_scaled_error"] for fd in all_fd),
            "cap_fd_step_consistency_scaled_error_max": max(
                scaled_error(np.asarray(first_fd[key]), np.asarray(second_fd[key])) for key in ("torch_cap", "jax_cap")
            ),
            "event_fd_step_consistency_scaled_error_max": max(
                scaled_error(np.asarray(first_fd[key]), np.asarray(second_fd[key]))
                for key in ("torch_event", "jax_event")
            ),
            "frequency_residual_abs_max": max(
                abs(row[key]) for row in all_rows for key in ("torch_frequency_residual", "jax_frequency_residual")
            ),
        }
        result["metrics"] = metrics
        metric_keys = {
            "baseline_cap_torch_jax_abs_at_most": "baseline_cap_torch_jax_abs",
            "baseline_cap_to_frozen_abs_at_most": "baseline_cap_to_frozen_abs",
            "perturbed_cap_torch_jax_abs_at_most": "perturbed_cap_torch_jax_abs_max",
            "perturbed_event_torch_jax_abs_at_most": "perturbed_event_torch_jax_abs_max",
            "cap_tangent_fd_scaled_error_at_most_each": "cap_tangent_fd_scaled_error_max",
            "event_tangent_fd_scaled_error_at_most_each": "event_tangent_fd_scaled_error_max",
            "cap_fd_torch_jax_scaled_error_at_most_each": "cap_fd_torch_jax_scaled_error_max",
            "event_fd_torch_jax_scaled_error_at_most_each": "event_fd_torch_jax_scaled_error_max",
            "cap_fd_step_consistency_scaled_error_at_most": "cap_fd_step_consistency_scaled_error_max",
            "event_fd_step_consistency_scaled_error_at_most": "event_fd_step_consistency_scaled_error_max",
        }
        gates = {
            "finite": bool(
                all(
                    np.isfinite(np.asarray(row[key])).all()
                    for row in all_rows
                    for key in ("torch_cap", "jax_cap", "torch_event", "jax_event")
                )
            ),
            "positive_cap_radius": bool(all(row[key][0] > 0 for row in all_rows for key in ("torch_cap", "jax_cap"))),
            "all_roots_inside_frozen_last_interval": all(
                0 < row[key] < width for row in all_rows for key in ("torch_root_offset_M", "jax_root_offset_M")
            ),
            "frequency_residual_abs_at_most": (
                metrics["frequency_residual_abs_max"] <= plan["root"]["frequency_residual_abs_at_most"]
            ),
        }
        gates.update({name: metrics[key] <= plan["checks"][name] for name, key in metric_keys.items()})
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

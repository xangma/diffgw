"""Dev-only JAX/Torch pathwise gradient check on a short v4 warm-start crossing."""

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
from dev.native_rom.eob.jax.adaptive_driver_clean import (
    fixed_fehlberg45_step,
    replay_accepted_steps,
)
from dev.native_rom.eob.jax.initial_conditions_clean import (
    traceable_quasicircular_initial_conditions,
)
from dev.native_rom.eob.jax.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as JaxHamiltonian,
)
from dev.native_rom.eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as TorchHamiltonian,
)
from dev.native_rom.eob.torch.warm_start import (
    _FrequencyCrossing,
    native_frequency_warm_start,
)
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator as JaxIntegrator
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from torchwave.approximants.eob.adaptive_driver_clean import (
    GSL_STATE_SCALED_CONTROLLER,
    integrate_adaptive,
)
from torchwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions as torch_initial_conditions,
)
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator as TorchIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/native_warm_start_jax_pathwise_plan.json"
OUTPUT = ROOT / "dev/native_rom/evidence/2026-09-25/native_warm_start_jax_pathwise.json"
INPUTS = (
    "dev/native_rom/eob/jax/adaptive_driver_clean.py",
    "dev/native_rom/eob/jax/initial_conditions_clean.py",
    "dev/native_rom/eob/jax/source_calibration_hamiltonian.py",
    "dev/native_rom/eob/torch/source_calibration_hamiltonian.py",
    "dev/native_rom/eob/torch/warm_start.py",
    "src/jaxwave/approximants/eob/integrator.py",
    "src/jaxwave/approximants/eob/seobnrv4_flux.py",
    "src/torchwave/approximants/eob/adaptive_driver_clean.py",
    "src/torchwave/approximants/eob/initial_conditions.py",
    "src/torchwave/approximants/eob/integrator.py",
    "src/torchwave/approximants/eob/seobnrv4_flux.py",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(data):
    OUTPUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def scaled_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return float(np.max(np.abs(actual-expected)/np.maximum(1.0, np.abs(expected))))


def main():
    plan = json.loads(PLAN.read_text())
    case, adaptive, limits = plan["case"], plan["adaptive"], plan["checks"]
    result = {
        "schema": "diffgw.seobnrv4.native-warm-start-jax-pathwise.v1",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "command": "env PYTHONPATH=src:. JAX_ENABLE_X64=1 /Users/xangma/repos/diffgw/.venv/bin/python dev/native_rom/diagnose_native_warm_start_jax_pathwise.py",
        "plan_sha256": sha256(PLAN), "script_sha256": sha256(Path(__file__)),
        "input_sha256": {p: sha256(ROOT/p) for p in INPUTS},
        "case": case,
    }
    write(result)
    started = time.monotonic()
    try:
        jax.config.update("jax_enable_x64", True)
        torch.set_num_threads(1)
        nu = case["mass1"]*case["mass2"]/(case["mass1"]+case["mass2"])**2
        delta = (case["mass1"]-case["mass2"])/(case["mass1"]+case["mass2"])
        cap = case["cap_Momega"]
        chi1 = case["chi1"]
        target = case["target_Momega"]
        steps = plan["finite_difference_steps"]
        tf = TorchFlux(**SOURCE_FLUX_OPTIONS)
        jf = JaxFlux(**SOURCE_FLUX_OPTIONS)

        def tensor(x, *, grad=False):
            return torch.tensor(x, dtype=torch.float64, requires_grad=grad)

        def torch_model(spin):
            return TorchHamiltonian(tensor(nu), spin, tensor(case["chi2"]),
                                    delta=tensor(delta))

        # The baseline discovery pass is detached. The mesh, including its event
        # knot, is then an explicit fixed input to both pathwise derivatives.
        th = torch_model(tensor(chi1))
        r, pr, pphi = torch_initial_conditions(th, tf, tensor(cap))
        initial = torch.stack((r, torch.zeros_like(r), pr, pphi)).detach()
        tfield = TorchIntegrator(th, tf).vector_field
        run = integrate_adaptive(
            tfield, initial, adaptive["initial_step_M"],
            event=_FrequencyCrossing(tensor(target)),
            atol=adaptive["atol"], rtol=adaptive["rtol"],
            initial_step=adaptive["initial_step_M"],
            controller=GSL_STATE_SCALED_CONTROLLER,
            max_duration=adaptive["max_duration_M"],
            max_accepted_steps=adaptive["max_accepted_steps"],
        )
        if run.statistics.termination != ("event",) or run.time.numel() < 2:
            raise RuntimeError("Torch discovery did not bracket near-cap target")
        accepted_time = np.asarray(run.time.detach())
        left_torch = np.asarray(run.state[-2].detach())
        result["mesh"] = {
            "accepted_steps": run.statistics.total_accepted,
            "evaluations": run.statistics.total_evaluations,
            "last_left_time_M": float(accepted_time[-2]),
            "last_right_time_M": float(accepted_time[-1]),
            "last_left_frequency": float(run.derivative[-2, 1].detach()),
            "last_right_frequency": float(run.derivative[-1, 1].detach()),
        }
        write(result)

        # Torch's native helper provides the actual crossing and its implicit
        # local derivative, while the frozen mesh defines JAX's pathwise route.
        tch = tensor(chi1, grad=True)
        ttgt = tensor(target, grad=True)
        twarm = native_frequency_warm_start(
            torch_model(tch), tf, tensor(cap), ttgt,
            atol=adaptive["atol"], rtol=adaptive["rtol"],
            first_step_M=adaptive["initial_step_M"],
            max_duration_M=adaptive["max_duration_M"],
        )
        if twarm.accepted_steps != run.statistics.total_accepted:
            raise RuntimeError("Torch warm helper did not replay discovered event mesh")
        tout = torch.stack((twarm.state[0], twarm.state[2],
                            twarm.state[3], twarm.crossing_time))
        tjac = np.asarray([
            [float(g) for g in torch.autograd.grad(
                value, (tch, ttgt), retain_graph=True,
            )] for value in tout
        ])
        tvalue = np.asarray(tout.detach())
        root0 = float(twarm.crossing_time.detach())-float(accepted_time[-2])
        result["torch"] = {
            "value": tvalue.tolist(), "jacobian": tjac.tolist(),
            "frequency_residual": float(twarm.frequency_residual.detach()),
            "root_offset_M": root0,
        }
        write(result)

        frozen_time = jnp.asarray(accepted_time, dtype=jnp.float64)
        root0_jax = jnp.asarray(root0, dtype=jnp.float64)

        def jax_map(x, *, return_left=False):
            jh = JaxHamiltonian(jnp.asarray(nu), x[0],
                                jnp.asarray(case["chi2"]), delta=jnp.asarray(delta))
            field = JaxIntegrator(jh, jf).vector_field
            rj, prj, ppj = traceable_quasicircular_initial_conditions(
                jh, jf, jnp.asarray(cap), max_iter=8,
            )
            initial_j = jnp.stack((rj, jnp.asarray(0.0), prj, ppj))
            def f(_time, state):
                return field(state)
            replay = replay_accepted_steps(f, initial_j, frozen_time)
            left = replay[-2]
            left_time = frozen_time[-2]
            left_derivative = field(left)

            def trial(offset):
                return fixed_fehlberg45_step(
                    f, left_time, left, offset,
                    atol=adaptive["atol"], rtol=adaptive["rtol"],
                    current_derivative=left_derivative,
                ).fifth

            def frequency(offset):
                return field(trial(offset))[1]

            slope = jax.lax.stop_gradient(jax.grad(frequency)(root0_jax))
            corrected = root0_jax-(frequency(root0_jax)-x[1])/slope
            crossed = trial(corrected)
            output = jnp.stack((crossed[0], crossed[2], crossed[3],
                                left_time+corrected))
            return (output, left, frequency(corrected), slope) if return_left else output

        x = jnp.asarray((chi1, target), dtype=jnp.float64)
        jvalue, jleft, jfrequency, jslope = jax_map(x, return_left=True)
        jvalue = np.asarray(jvalue)
        jjac = np.asarray(jax.jacfwd(jax_map)(x))
        jfd = np.stack([
            (np.asarray(jax_map(x.at[i].add(step)))
             - np.asarray(jax_map(x.at[i].add(-step))))/(2*step)
            for i, step in enumerate((steps["chi1"], steps["target_Momega"]))
        ], axis=1)
        result["jax"] = {
            "value": jvalue.tolist(), "jacobian": jjac.tolist(),
            "finite_difference_jacobian": jfd.tolist(),
            "replayed_left_state": np.asarray(jleft).tolist(),
            "crossing_frequency_residual": float(jfrequency-target),
            "crossing_frequency_time_slope": float(jslope),
        }
        write(result)

        def torch_value(spin, frequency):
            warm = native_frequency_warm_start(
                torch_model(tensor(spin)), tf, tensor(cap), tensor(frequency),
                atol=adaptive["atol"], rtol=adaptive["rtol"],
                first_step_M=adaptive["initial_step_M"],
                max_duration_M=adaptive["max_duration_M"],
            )
            return np.asarray((float(warm.state[0]), float(warm.state[2]),
                               float(warm.state[3]), float(warm.crossing_time)))

        tfd = np.stack([
            (torch_value(chi1+steps["chi1"], target)
             - torch_value(chi1-steps["chi1"], target))/(2*steps["chi1"]),
            (torch_value(chi1, target+steps["target_Momega"])
             - torch_value(chi1, target-steps["target_Momega"]))
            /(2*steps["target_Momega"]),
        ], axis=1)
        result["torch"]["finite_difference_jacobian"] = tfd.tolist()
        metrics = {
            "mesh_left_state_max_abs": float(np.max(np.abs(left_torch-np.asarray(jleft)))),
            "crossing_frequency_residual_abs": max(abs(result["torch"]["frequency_residual"]),
                                                   abs(result["jax"]["crossing_frequency_residual"])),
            "event_time_abs_M": float(abs(tvalue[-1]-jvalue[-1])),
            "phase_zero_state_max_abs": float(np.max(np.abs(tvalue[:3]-jvalue[:3]))),
            "jacobian_torch_jax_scaled_error": scaled_error(tjac,jjac),
            "jacobian_jax_fd_scaled_error": scaled_error(jjac,jfd),
            "jacobian_torch_fd_scaled_error": scaled_error(tjac,tfd),
        }
        result["metrics"] = metrics
        gates = {
            "finite": bool(all(np.isfinite(a).all() for a in (
                accepted_time, left_torch, tvalue, tjac, jvalue, jjac, jfd, tfd
            ))),
            "mesh_bracket": bool(result["mesh"]["last_left_frequency"] < target <=
                                 result["mesh"]["last_right_frequency"]),
            "positive_crossing_slope": bool(float(jslope)>0),
        }
        metric_for_gate = {
            "mesh_left_state_torch_jax_max_abs_at_most": "mesh_left_state_max_abs",
            "crossing_frequency_residual_abs_at_most": "crossing_frequency_residual_abs",
            "event_time_torch_jax_abs_at_most_M": "event_time_abs_M",
            "phase_zero_state_torch_jax_max_abs_at_most": "phase_zero_state_max_abs",
            "jacobian_torch_jax_scaled_error_at_most": "jacobian_torch_jax_scaled_error",
            "jacobian_jax_fd_scaled_error_at_most": "jacobian_jax_fd_scaled_error",
            "jacobian_torch_fd_scaled_error_at_most": "jacobian_torch_fd_scaled_error",
        }
        for name, threshold in limits.items():
            gates[name] = metrics[metric_for_gate[name]] <= threshold
        result["gates"] = gates
        result["diagnostic_checks_passed"] = all(gates.values())
        result["status"] = "diagnostic_complete"
    except Exception:
        result["status"] = "diagnostic_failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["elapsed_s"] = time.monotonic()-started
        write(result)


if __name__ == "__main__":
    main()

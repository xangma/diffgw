"""Dev-only local high-spin warm-start RKF45 value and spin-tangent check."""

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
from dev.native_rom.eob.jax.adaptive_driver_clean import fixed_fehlberg45_step as jax_fixed
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
from torchwave.approximants.eob.adaptive_driver_clean import fixed_fehlberg45_step as torch_fixed
from torchwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions as torch_initial_conditions,
)
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator as TorchIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/native_warm_start_one_step_gradient_plan.json"
OUTPUT = PLAN.with_name("native_warm_start_one_step_gradient.json")
INPUTS = (
    "dev/native_rom/eob/jax/adaptive_driver_clean.py",
    "dev/native_rom/eob/jax/initial_conditions_clean.py",
    "dev/native_rom/eob/jax/source_calibration_hamiltonian.py",
    "dev/native_rom/eob/torch/source_calibration_hamiltonian.py",
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
    return float(np.max(np.abs(actual - expected) / np.maximum(1.0, np.abs(expected))))


def main():
    plan = json.loads(PLAN.read_text())
    case = plan["case"]
    result = {
        "schema": "diffgw.seobnrv4.native-warm-start-one-step-gradient.v1",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": sha256(PLAN), "script_sha256": sha256(Path(__file__)),
        "input_sha256": {p: sha256(ROOT / p) for p in INPUTS},
        "case": case,
    }
    write(result)
    started = time.monotonic()
    try:
        jax.config.update("jax_enable_x64", True)
        torch.set_num_threads(1)
        nu = case["mass1"] * case["mass2"] / (case["mass1"] + case["mass2"]) ** 2
        delta = (case["mass1"] - case["mass2"]) / (case["mass1"] + case["mass2"])
        cap, step = case["cap_Momega"], case["step_M"]
        spin, fd_step = case["chi1"], plan["finite_difference_chi1_step"]
        tf, jf = TorchFlux(**SOURCE_FLUX_OPTIONS), JaxFlux(**SOURCE_FLUX_OPTIONS)

        def tensor(x, grad=False):
            return torch.tensor(x, dtype=torch.float64, requires_grad=grad)

        def torch_step(chi):
            ham = TorchHamiltonian(tensor(nu), chi, tensor(case["chi2"]),
                                   delta=tensor(delta))
            r, pr, pphi = torch_initial_conditions(ham, tf, tensor(cap))
            initial = torch.stack((r, torch.zeros_like(r), pr, pphi))
            field = TorchIntegrator(ham, tf).vector_field
            return torch_fixed(field, 0.0, initial, step).fifth

        torch_spin = tensor(spin, grad=True)
        torch_value = torch_step(torch_spin)
        torch_gradient = torch.stack([
            torch.autograd.grad(value, torch_spin, retain_graph=True)[0]
            for value in torch_value
        ])
        tv = np.asarray(torch_value.detach())
        tg = np.asarray(torch_gradient.detach())
        result["torch"] = {"value": tv.tolist(), "spin_gradient": tg.tolist()}
        write(result)

        def jax_step(chi):
            ham = JaxHamiltonian(jnp.asarray(nu), chi, jnp.asarray(case["chi2"]),
                                 delta=jnp.asarray(delta))
            r, pr, pphi = traceable_quasicircular_initial_conditions(
                ham, jf, jnp.asarray(cap), max_iter=8,
            )
            initial = jnp.stack((r, jnp.asarray(0.0), pr, pphi))
            field = JaxIntegrator(ham, jf).vector_field
            return jax_fixed(lambda _time, state: field(state), 0.0,
                             initial, step).fifth

        compiled_jax_step = jax.jit(jax_step)
        jv = np.asarray(compiled_jax_step(jnp.asarray(spin)))
        result["jax"] = {"value": jv.tolist()}
        write(result)
        _, jvp = jax.jvp(compiled_jax_step, (jnp.asarray(spin),),
                         (jnp.asarray(1.0),))
        jg = np.asarray(jvp)
        result["jax"]["spin_gradient"] = jg.tolist()
        write(result)

        tfd = ((np.asarray(torch_step(tensor(spin + fd_step)).detach())
                - np.asarray(torch_step(tensor(spin - fd_step)).detach()))
               / (2 * fd_step))
        jfd = ((np.asarray(compiled_jax_step(jnp.asarray(spin + fd_step)))
                - np.asarray(compiled_jax_step(jnp.asarray(spin - fd_step))))
               / (2 * fd_step))
        result["torch"]["finite_difference_gradient"] = tfd.tolist()
        result["jax"]["finite_difference_gradient"] = jfd.tolist()
        metrics = {
            "state_max_abs": float(np.max(np.abs(tv - jv))),
            "gradient_torch_jax_scaled_error": scaled_error(tg, jg),
            "gradient_jax_fd_scaled_error": scaled_error(jg, jfd),
            "gradient_torch_fd_scaled_error": scaled_error(tg, tfd),
        }
        result["metrics"] = metrics
        metric_for_gate = {
            "state_torch_jax_max_abs_at_most": "state_max_abs",
            "gradient_torch_jax_scaled_error_at_most": "gradient_torch_jax_scaled_error",
            "gradient_jax_fd_scaled_error_at_most": "gradient_jax_fd_scaled_error",
            "gradient_torch_fd_scaled_error_at_most": "gradient_torch_fd_scaled_error",
        }
        gates = {"finite": bool(all(np.isfinite(x).all() for x in (tv, tg, jv, jg, tfd, jfd)))}
        gates.update({key: metrics[metric_for_gate[key]] <= threshold
                      for key, threshold in plan["checks"].items()})
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

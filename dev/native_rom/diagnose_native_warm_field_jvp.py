"""Dev-only bounded source-field JVP probe at the high-spin cap state."""

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
from dev.native_rom.eob.jax.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as JaxHamiltonian,
)
from dev.native_rom.eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as TorchHamiltonian,
)
from jaxwave.approximants.eob.integrator import EOBTrajectoryIntegrator as JaxIntegrator
from jaxwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as JaxFlux
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator as TorchIntegrator
from torchwave.approximants.eob.seobnrv4_flux import SEOBNRv4PublishedFlux as TorchFlux


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/native_warm_field_jvp_plan.json"
OUTPUT = PLAN.with_name("native_warm_field_jvp.json")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(data):
    OUTPUT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def scaled_error(actual, expected):
    return float(np.max(np.abs(actual - expected) / np.maximum(1.0, np.abs(expected))))


def main():
    plan = json.loads(PLAN.read_text())
    if sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen probe script changed")
    for path, expected in plan["inputs_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"frozen input changed: {path}")
    prior = plan["prior_cap_evidence"]
    if sha256(ROOT / prior["path"]) != prior["sha256"]:
        raise ValueError("frozen cap evidence changed")
    case = plan["case"]
    result = {
        "schema": "diffgw.seobnrv4.native-warm-field-jvp.v1",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": sha256(PLAN), "script_sha256": sha256(Path(__file__)),
        "input_sha256": plan["inputs_sha256"], "prior_cap_evidence": prior,
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
        chi0, state0 = case["chi1"], np.asarray(case["frozen_state"])
        tangent = np.asarray(case["state_tangent_per_chi1"])
        tf, jf = TorchFlux(**SOURCE_FLUX_OPTIONS), JaxFlux(**SOURCE_FLUX_OPTIONS)

        def ttensor(value, *, grad=False):
            return torch.tensor(value, dtype=torch.float64, requires_grad=grad)

        def torch_field_at(chi, state):
            ham = TorchHamiltonian(ttensor(nu), chi, ttensor(case["chi2"]),
                                   delta=ttensor(delta))
            return TorchIntegrator(ham, tf).vector_field(state)

        tchi = ttensor(chi0, grad=True)
        tstate = ttensor(state0) + (tchi - chi0) * ttensor(tangent)
        tout = torch_field_at(tchi, tstate)
        tv = np.asarray(tout.detach())
        tg = np.asarray(torch.stack([
            torch.autograd.grad(value, tchi, retain_graph=True)[0]
            for value in tout
        ]).detach())
        result["torch"] = {"value": tv.tolist(), "directional_derivative": tg.tolist()}
        write(result)

        def jax_field_at(chi, state):
            ham = JaxHamiltonian(jnp.asarray(nu), chi, jnp.asarray(case["chi2"]),
                                 delta=jnp.asarray(delta))
            return JaxIntegrator(ham, jf).vector_field(state)

        compiled_field = jax.jit(jax_field_at)
        jchi = jnp.asarray(chi0)
        jstate = jnp.asarray(state0)
        jv = np.asarray(compiled_field(jchi, jstate))
        result["jax"] = {"value": jv.tolist()}
        write(result)

        def directional(chi, state, dchi, dstate):
            return jax.jvp(jax_field_at, (chi, state), (dchi, dstate))

        compiled_directional = jax.jit(directional)
        primal, derivative = compiled_directional(
            jchi, jstate, jnp.asarray(case["chi1_tangent"]), jnp.asarray(tangent),
        )
        jg = np.asarray(derivative)
        result["jax"].update(jvp_primal=np.asarray(primal).tolist(),
                             directional_derivative=jg.tolist())
        write(result)

        finite_differences = {}
        for h in plan["finite_difference_chi1_steps"]:
            tfd = (np.asarray(torch_field_at(ttensor(chi0 + h), ttensor(state0 + h * tangent)).detach())
                   - np.asarray(torch_field_at(ttensor(chi0 - h), ttensor(state0 - h * tangent)).detach())) / (2 * h)
            jfd = (np.asarray(compiled_field(jnp.asarray(chi0 + h), jnp.asarray(state0 + h * tangent)))
                   - np.asarray(compiled_field(jnp.asarray(chi0 - h), jnp.asarray(state0 - h * tangent)))) / (2 * h)
            finite_differences[str(h)] = {
                "torch": tfd.tolist(), "jax": jfd.tolist(),
                "torch_autodiff_scaled_error": scaled_error(tg, tfd),
                "jax_jvp_scaled_error": scaled_error(jg, jfd),
            }
        result["finite_differences"] = finite_differences
        metrics = {
            "primal_torch_jax_max_abs": float(np.max(np.abs(tv - jv))),
            "tangent_torch_jax_scaled_error": scaled_error(tg, jg),
            "tangent_jax_fd_scaled_error_max": max(
                x["jax_jvp_scaled_error"] for x in finite_differences.values()),
            "tangent_torch_fd_scaled_error_max": max(
                x["torch_autodiff_scaled_error"] for x in finite_differences.values()),
        }
        result["metrics"] = metrics
        metric_for_gate = {
            "primal_torch_jax_max_abs_at_most": "primal_torch_jax_max_abs",
            "tangent_torch_jax_scaled_error_at_most": "tangent_torch_jax_scaled_error",
            "tangent_jax_fd_scaled_error_at_most_each": "tangent_jax_fd_scaled_error_max",
            "tangent_torch_fd_scaled_error_at_most_each": "tangent_torch_fd_scaled_error_max",
        }
        gates = {"finite": bool(all(np.isfinite(x).all() for x in (tv, tg, jv, jg)))}
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

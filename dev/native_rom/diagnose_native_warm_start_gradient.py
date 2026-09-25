"""Frozen exact-row component gradient diagnostic for the native v4 warm start."""

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
from dev.native_rom.eob.jax.initial_conditions_clean import (
    traceable_quasicircular_initial_conditions,
)
from dev.native_rom.eob.jax.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as JaxHamiltonian,
)
from dev.native_rom.eob.torch.source_calibration_hamiltonian import (
    SourceCalibratedHistoricalHamiltonian as TorchHamiltonian,
)
from dev.native_rom.eob.torch.warm_start import native_frequency_warm_start
from jaxwave.approximants.eob.seobnrv4_flux import (
    SEOBNRv4PublishedFlux as JaxFlux,
)
from torchwave.approximants.eob.initial_conditions import (
    find_quasicircular_initial_conditions as torch_initial_conditions,
)
from torchwave.approximants.eob.seobnrv4_flux import (
    SEOBNRv4PublishedFlux as TorchFlux,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "dev/native_rom/evidence/2026-09-25/native_warm_start_gradient_plan.json"
OUTPUT = ROOT / "dev/native_rom/evidence/2026-09-25/native_warm_start_gradient.json"
INPUTS = (
    "dev/native_rom/eob/jax/initial_conditions_clean.py",
    "dev/native_rom/eob/jax/source_calibration_hamiltonian.py",
    "dev/native_rom/eob/torch/source_calibration_hamiltonian.py",
    "dev/native_rom/eob/torch/warm_start.py",
    "src/jaxwave/approximants/eob/seobnrv4_flux.py",
    "src/torchwave/approximants/eob/seobnrv4_flux.py",
    "src/torchwave/approximants/eob/initial_conditions.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(result: dict) -> None:
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def scaled_error(actual, expected) -> float:
    actual, expected = np.asarray(actual), np.asarray(expected)
    return float(np.max(np.abs(actual - expected) / np.maximum(1.0, np.abs(expected))))


def main() -> None:
    plan = json.loads(PLAN.read_text())
    case = plan["case"]
    steps = plan["finite_difference_steps"]
    checks = plan["checks"]
    result = {
        "schema": "diffgw.seobnrv4.native-warm-start-gradient.v1",
        "status": "running",
        "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "command": "PYTHONPATH=src:. JAX_ENABLE_X64=1 /Users/xangma/repos/diffgw/.venv/bin/python dev/native_rom/diagnose_native_warm_start_gradient.py",
        "plan_sha256": sha256(PLAN),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {path: sha256(ROOT / path) for path in INPUTS},
        "case": case,
    }
    write(result)
    started = time.monotonic()
    try:
        jax.config.update("jax_enable_x64", True)
        torch.set_num_threads(1)
        nu = case["mass1"] * case["mass2"] / (case["mass1"] + case["mass2"]) ** 2
        delta = (case["mass1"] - case["mass2"]) / (case["mass1"] + case["mass2"])
        chi2 = case["chi2"]
        jflux = JaxFlux(**SOURCE_FLUX_OPTIONS)
        tflux = TorchFlux(**SOURCE_FLUX_OPTIONS)

        def cap_jax(x):
            h = JaxHamiltonian(jnp.asarray(nu), x[0], jnp.asarray(chi2),
                               delta=jnp.asarray(delta))
            return jnp.stack(traceable_quasicircular_initial_conditions(
                h, jflux, x[1], max_iter=8,
            ))

        def tensor(value, *, grad=False):
            return torch.tensor(value, dtype=torch.float64, requires_grad=grad)

        def cap_torch(chi1, cap):
            h = TorchHamiltonian(tensor(nu), chi1, tensor(chi2),
                                 delta=tensor(delta))
            return torch.stack(torch_initial_conditions(h, tflux, cap))

        x = jnp.asarray((case["chi1"], case["cap_Momega"]), dtype=jnp.float64)
        jcap = np.asarray(jax.jit(cap_jax)(x))
        jcap_jac = np.asarray(jax.jit(jax.jacfwd(cap_jax))(x))
        tchi = tensor(case["chi1"], grad=True)
        tcap = tensor(case["cap_Momega"], grad=True)
        tcap_state = cap_torch(tchi, tcap)
        tcap_jac = np.asarray([
            [float(g) for g in torch.autograd.grad(
                value, (tchi, tcap), retain_graph=True,
            )]
            for value in tcap_state
        ])
        tcap_value = tcap_state.detach().numpy()
        cap_fd = np.stack([
            (np.asarray(jax.jit(cap_jax)(x.at[i].add(step)))
             - np.asarray(jax.jit(cap_jax)(x.at[i].add(-step)))) / (2 * step)
            for i, step in enumerate((steps["cap_chi1"], steps["cap_Momega"]))
        ], axis=1)
        result["cap"] = {
            "jax_state": jcap.tolist(), "torch_state": tcap_value.tolist(),
            "jax_jacobian": jcap_jac.tolist(), "torch_jacobian": tcap_jac.tolist(),
            "jax_finite_difference_jacobian": cap_fd.tolist(),
            "state_max_abs": float(np.max(np.abs(jcap - tcap_value))),
            "jacobian_max_abs": float(np.max(np.abs(jcap_jac - tcap_jac))),
            "jax_fd_scaled_error": scaled_error(jcap_jac, cap_fd),
        }
        write(result)

        target = tensor(case["target_Momega"], grad=True)
        crossing = native_frequency_warm_start(
            TorchHamiltonian(tensor(nu), tchi, tensor(chi2), delta=tensor(delta)),
            tflux, tensor(case["cap_Momega"]), target,
        )
        warm = torch.stack((crossing.state[0], crossing.state[2],
                            crossing.state[3], crossing.crossing_time))
        warm_jac = np.asarray([
            [float(g) for g in torch.autograd.grad(value, (tchi, target),
                                                   retain_graph=True)]
            for value in warm
        ])
        result["warm"] = {
            "state_and_crossing_time": warm.detach().numpy().tolist(),
            "jacobian_chi1_target": warm_jac.tolist(),
            "frequency_residual": float(crossing.frequency_residual.detach()),
            "accepted_steps": crossing.accepted_steps,
            "evaluations": crossing.evaluations,
        }
        write(result)

        def warm_value(chi1, target_frequency):
            h = TorchHamiltonian(tensor(nu), tensor(chi1), tensor(chi2),
                                 delta=tensor(delta))
            run = native_frequency_warm_start(
                h, tflux, tensor(case["cap_Momega"]), tensor(target_frequency),
            )
            return np.asarray((float(run.state[0]), float(run.state[2]),
                               float(run.state[3]), float(run.crossing_time)))

        warm_fd = np.stack([
            (warm_value(case["chi1"] + steps["warm_chi1"], case["target_Momega"])
             - warm_value(case["chi1"] - steps["warm_chi1"], case["target_Momega"]))
            / (2 * steps["warm_chi1"]),
            (warm_value(case["chi1"], case["target_Momega"] + steps["target_Momega"])
             - warm_value(case["chi1"], case["target_Momega"] - steps["target_Momega"]))
            / (2 * steps["target_Momega"]),
        ], axis=1)
        result["warm"]["finite_difference_jacobian_chi1_target"] = warm_fd.tolist()
        result["warm"]["chi1_fd_scaled_error"] = scaled_error(warm_jac[:, 0], warm_fd[:, 0])
        result["warm"]["target_fd_scaled_error"] = scaled_error(warm_jac[:, 1], warm_fd[:, 1])
        gates = {
            "finite_cap_state_and_jacobians": bool(np.isfinite(jcap).all() and
                np.isfinite(jcap_jac).all() and np.isfinite(tcap_value).all() and
                np.isfinite(tcap_jac).all() and np.isfinite(cap_fd).all()),
            "finite_warm_state_and_jacobians": bool(np.isfinite(warm_jac).all() and
                np.isfinite(warm_fd).all() and np.isfinite(warm.detach().numpy()).all()),
            "cap_initial_state_torch_jax_max_abs_at_most": result["cap"]["state_max_abs"] <= checks["cap_initial_state_torch_jax_max_abs_at_most"],
            "cap_initial_jacobian_torch_jax_max_abs_at_most": result["cap"]["jacobian_max_abs"] <= checks["cap_initial_jacobian_torch_jax_max_abs_at_most"],
            "cap_initial_jacobian_jax_fd_scaled_error_at_most": result["cap"]["jax_fd_scaled_error"] <= checks["cap_initial_jacobian_jax_fd_scaled_error_at_most"],
            "warm_frequency_residual_abs_at_most": abs(result["warm"]["frequency_residual"]) <= checks["warm_frequency_residual_abs_at_most"],
            "warm_target_jacobian_torch_fd_scaled_error_at_most": result["warm"]["target_fd_scaled_error"] <= checks["warm_target_jacobian_torch_fd_scaled_error_at_most"],
            "warm_chi1_jacobian_torch_fd_scaled_error_at_most": result["warm"]["chi1_fd_scaled_error"] <= checks["warm_chi1_jacobian_torch_fd_scaled_error_at_most"],
        }
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

"""Predeclared dense-spin comparison of original Kerr QNM solves to a pinned wheel.

The wheel is a historical interpolation comparator, never a source of native
coefficients. Existing sparse author-data checks remain the independent test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import lal
import lalsimulation as lalsim
import numpy as np
import torch

from . import qualify_generator as q
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220
from torchwave.qnm.solver import solve


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-24/qnm_dense_spin_plan.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    grid = plan["spin_grid"]
    if (plan["schema"] != "diffgw.seobnrv4.qnm-dense-spin-plan.v2"
            or plan["release_qualified"] is not False
            or plan["lal_version"] != "7.7.1"
            or plan["lalsimulation_version"] != "6.2.1"
            or grid != {
                "linear_start": -0.9, "linear_stop": 0.9, "linear_count": 91,
                "high_tail_log10_gap_start": -1.0,
                "high_tail_log10_gap_stop": -3.3979400086720375,
                "high_tail_count": 41, "round_decimal_places": 12,
            }
            or plan["dimensionless_absolute_tolerance"] != 6e-7
            or plan["posthoc_solver_control_spins"] != [
                0.999, 0.9994, 0.999540793009, 0.9996]
            or plan["solver_control_absolute_tolerance"] != 1e-10):
        parser.error("Frozen dense-spin QNM plan changed")
    identity = q.lal_binary_identity(plan["lal_version"], plan["lalsimulation_version"])
    linear = np.linspace(grid["linear_start"], grid["linear_stop"], grid["linear_count"])
    tail = 1 - 10**np.linspace(
        grid["high_tail_log10_gap_start"], grid["high_tail_log10_gap_stop"],
        grid["high_tail_count"],
    )
    spins = np.unique(np.round(np.concatenate((linear, tail)), grid["round_decimal_places"]))
    if len(spins) != 131 or spins[0] != -0.9 or spins[-1] != 0.9996:
        raise ValueError("Dense-spin grid changed")
    jax.config.update("jax_enable_x64", True)
    torch_spin = torch.tensor(spins, dtype=torch.float64)
    torch_values = np.stack([
        np.asarray(value.detach().cpu(), dtype=np.float64)
        for value in signed_qnm_220(torch_spin)
    ], axis=-1)
    jax_values = np.stack([
        np.asarray(value, dtype=np.float64)
        for value in evaluate_kerr_qnm_220(jax.numpy.asarray(spins))
    ], axis=-1)
    binary_values = np.empty((len(spins), 2), dtype=np.float64)
    for index, spin in enumerate(spins):
        vector = lal.CreateCOMPLEX16Vector(1)
        status = lalsim.SimIMREOBGenerateQNMFreqV2fromFinal(
            vector, 1., float(spin), 2, 2, 1,
        )
        if status != 0:
            raise RuntimeError(f"Pinned wheel QNM failed at spin {spin}: {status}")
        value = vector.data[0] * lal.MTSUN_SI
        binary_values[index] = value.real, value.imag
    errors = {"torch": abs(torch_values - binary_values),
              "jax": abs(jax_values - binary_values)}
    worst = {}
    for backend, error in errors.items():
        component = np.unravel_index(np.argmax(error), error.shape)
        worst[backend] = {
            "spin": float(spins[component[0]]),
            "component": ["frequency", "damping"][component[1]],
            "absolute_error": float(error[component]),
            "native": float((torch_values if backend == "torch" else jax_values)[component]),
            "binary": float(binary_values[component]),
        }
    cross_backend = float(np.max(abs(torch_values - jax_values)))
    solver_controls = []
    for spin in plan["posthoc_solver_control_spins"]:
        frequency, damping = (
            float(value) for value in signed_qnm_220(torch.tensor(spin, dtype=torch.float64))
        )
        root = solve(spin, 2, 2, 0, omega_seed=complex(frequency, -damping), strict=True)
        if not root.converged:
            raise RuntimeError(f"Independent Kerr equation solver failed at {spin}")
        solver_controls.append({
            "spin": spin, "native_frequency": frequency, "native_damping": damping,
            "solver_frequency": root.omega.real, "solver_damping": -root.omega.imag,
            "native_minus_solver_abs": abs(root.omega - complex(frequency, -damping)),
            "radial_residual_abs": abs(root.radial_residual),
            "angular_error": root.angular_error, "continued_fraction_error": root.cf_error,
            "continued_fraction_depth": root.cf_depth,
        })
    report = {
        "schema": "diffgw.seobnrv4.qnm-dense-spin.v2",
        "status": "diagnostic_complete", "release_qualified": False,
        "scope": dict(q.SCOPE, description="Dense-spin component comparison only; no waveform qualification.",
                      known_limitations=[
                          "The wheel interpolation is a historical comparator, not independent physical validation.",
                          "The grid ends at the pinned binary's supported spin 0.9996; native endpoint clamps differ.",
                          "Post-hoc original-solver checks test our interpolation, not independence of our equation solver.",
                          "This component check does not qualify the full generator or independent ROM.",
                      ]),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "binary": identity, "spin_grid": grid,
        "spin_count": len(spins), "spins_sha256": q.canonical_hash(spins.tolist()),
        "worst_native_minus_binary": worst,
        "posthoc_solver_controls": solver_controls,
        "torch_jax_max_abs": cross_backend,
        "tolerance": plan["dimensionless_absolute_tolerance"],
        "gates": {
            "torch_near_pinned_binary": worst["torch"]["absolute_error"] < plan["dimensionless_absolute_tolerance"],
            "jax_near_pinned_binary": worst["jax"]["absolute_error"] < plan["dimensionless_absolute_tolerance"],
            "torch_jax_agree": cross_backend < 1e-12,
            "native_interpolation_matches_original_solver": max(
                row["native_minus_solver_abs"] for row in solver_controls
            ) < plan["solver_control_absolute_tolerance"],
        },
    }
    q.write_evidence(args.json_out, report)
    if not all(report["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

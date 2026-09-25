"""Dev-only strong-aligned NQC phase matrix from continuous ODE derivatives.

The match state is interpolated with endpoint ODE derivatives. Both temporal
derivative rows then come from differentiating the phase basis along the native
source-backed vector field, rather than differentiating sampled basis values.
No production code or registration is changed.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _compare, _series_summary


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_ode_chain_phase_matrix_plan.json"
OUTPUT = EVIDENCE / "source_aligned_ode_chain_phase_matrix.json"


def _checked_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-ode-chain-phase-matrix-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "strong_aligned"
            or plan["levels"] != ["baseline", "refined"]
            or plan["method"] != "endpoint-Hermite state; autograd Lie derivatives of phase basis"
            or plan["runner_sentinel"] != {"half_width_M": 0.0, "degree": 0}):
        raise ValueError("frozen ODE-chain plan changed")
    for key, frozen in plan["inputs"].items():
        path = ROOT / frozen["path"]
        if q.sha256(path) != frozen["sha256"]:
            raise ValueError(f"frozen {key} SHA256 changed: {path}")
    if q.sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen diagnostic script changed")
    return plan


def _source_field():
    m1, m2, chi1, chi2, _ = driver.CASES["strong_aligned"]

    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)

    hamiltonian = driver.TorchSourceHam(
        tensor(m1*m2/(m1+m2)**2), tensor(chi1), tensor(chi2),
        delta=tensor((m1-m2)/(m1+m2)))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    return driver.TorchIntegrator(hamiltonian, flux).vector_field


def _phase_matrix(time_grid, state, orbital_frequency, match_time, *,
                  half_width_M, degree, field, capture):
    if (half_width_M, degree) != (0.0, 0):
        raise ValueError("unexpected driver sentinel")
    time = float(match_time)
    index = int(torch.searchsorted(time_grid, match_time).item()) - 1
    index = max(0, min(index, len(time_grid)-2))
    t0, t1 = (float(time_grid[index]), float(time_grid[index+1]))
    if not t0 <= time <= t1:
        raise ValueError("match event outside Hermite bracket")
    h = t1-t0
    u = (time-t0)/h
    y0, y1 = state[index].detach(), state[index+1].detach()
    with torch.no_grad():
        f0, f1 = field(y0), field(y1)
    center = ((2*u**3-3*u**2+1)*y0 + (u**3-2*u**2+u)*h*f0
              +(-2*u**3+3*u**2)*y1 + (u**3-u**2)*h*f1)
    center = center.detach().requires_grad_(True)
    vector = field(center)
    phase = driver.torch_basis(center[0], center[2], vector[1])[1]
    first = torch.stack(tuple(
        (torch.autograd.grad(phase[i], center, create_graph=True,
                             retain_graph=True)[0]*vector).sum()
        for i in range(2)))
    second = torch.stack(tuple(
        (torch.autograd.grad(first[i], center, retain_graph=True)[0]*vector).sum()
        for i in range(2)))
    matrix = torch.stack((first, second)).detach()
    capture.append({
        "match_time_local_M": time, "bracket_index": index,
        "bracket_time_M": [t0, t1], "bracket_fraction": u,
        "interpolated_state": center.detach().tolist(),
        "field_at_match": vector.detach().tolist(),
        "phase_basis": phase.detach().tolist(),
        "orbital_frequency_minus_bracket_linear": float(
            orbital_frequency[index]*(1-u)+orbital_frequency[index+1]*u-vector[1]),
        "first": first.detach().tolist(), "second": second.detach().tolist(),
    })
    return matrix


def main() -> None:
    plan = _checked_plan()
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    natural = json.loads((ROOT / plan["inputs"]["natural_result"]["path"]).read_text())
    polynomial = json.loads((ROOT / plan["inputs"]["polynomial_result"]["path"]).read_text())
    identity = driver._prepare_public(source_plan)
    field = _source_field()
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.source-aligned-ode-chain-phase-matrix.v1",
        "status": "running", "release_qualified": False,
        "scope": "Torch-only dev counterfactual; JAX and public wheel retain their baseline NQC route",
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
        "source_flux_options": driver.SOURCE_FLUX_OPTIONS,
        "levels": {}, "comparisons": {}, "checks": {},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    series = {}
    for name in plan["levels"]:
        level = next(item for item in ladder["levels"] if item["name"] == name)
        settings = {
            "coarse_output_step_M": ladder["coarse_output_step_M"],
            "max_step_M": ladder["max_step_M"],
            "atol": level["atol"], "rtol": level["rtol"],
        }
        capture = []

        def phase_hook(time_grid, state, orbital_frequency, match_time, *,
                       half_width_M, degree):
            return _phase_matrix(
                time_grid, state, orbital_frequency, match_time,
                half_width_M=half_width_M, degree=degree,
                field=field, capture=capture)

        with patch.object(driver, "_polynomial_phase_matrix", phase_hook):
            diagnostic, torch_series, jax_series, stage = driver.run(
                plan["case"], source_plan, identity, variant="source",
                independent_init=True, numerical_settings=settings,
                diagnostic_nqc_phase_matrix=plan["runner_sentinel"],
                write_result=False, return_series=True, return_stages=True)
        if len(capture) != 1:
            raise ValueError("expected one continuous derivative evaluation")
        series[name] = torch_series
        matrix = np.asarray(stage["nqc_phase_matrix"])
        rhs = np.asarray(stage["nqc_phase_rhs"])
        coefficients = np.asarray(stage["nqc_coefficients"][1])
        result["levels"][name] = {
            "settings": settings, "match_time_M": stage["match_time_M"],
            "chain": capture[0], "phase_matrix": matrix.tolist(),
            "phase_rhs": rhs.tolist(), "phase_coefficients": coefficients.tolist(),
            "phase_condition": stage["nqc_phase_condition"],
            "phase_linear_residual_linf": float(np.max(abs(matrix@coefficients-rhs))),
            "torch": _series_summary(torch_series),
            "jax_natural_cubic": _series_summary(jax_series),
            "torch_public_complex_relative_l2": diagnostic["torch_public"]["amplitude_peak_anchored"]["metrics"]["complex_relative_l2"],
            "torch_to_jax_natural_cubic": _compare(torch_series, jax_series),
        }
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(name, result["levels"][name]["phase_coefficients"], flush=True)
    candidate = _compare(series["baseline"], series["refined"])
    baseline_natural = natural["comparisons"]["torch_baseline_to_refined"]
    prior_polynomial = polynomial["comparisons"]["candidate_torch_baseline_to_refined"]
    result["comparisons"] = {
        "candidate_torch_baseline_to_refined": candidate,
        "natural_cubic_torch_baseline_to_refined": baseline_natural,
        "prior_polynomial_torch_baseline_to_refined": prior_polynomial,
        "candidate_to_natural_drift_ratio": (
            candidate["complex_relative_l2"]/baseline_natural["complex_relative_l2"]),
        "candidate_to_prior_polynomial_drift_ratio": (
            candidate["complex_relative_l2"]/prior_polynomial["complex_relative_l2"]),
        "phase_matrix_adjacent_frobenius": float(np.linalg.norm(
            np.asarray(result["levels"]["baseline"]["phase_matrix"])
            -np.asarray(result["levels"]["refined"]["phase_matrix"]))),
        "phase_coefficient_adjacent_l2": float(np.linalg.norm(
            np.asarray(result["levels"]["baseline"]["phase_coefficients"])
            -np.asarray(result["levels"]["refined"]["phase_coefficients"]))),
    }
    limits = plan["checks"]
    result["checks"] = {
        "whole_mode_drift_at_most": candidate["complex_relative_l2"]
            <= limits["whole_mode_drift_at_most"],
        "drift_at_most_one_tenth_natural": result["comparisons"]["candidate_to_natural_drift_ratio"] <= 0.1,
        "phase_linear_residual_at_most": max(
            result["levels"][name]["phase_linear_residual_linf"] for name in plan["levels"])
            <= limits["phase_linear_residual_at_most"],
        "finite_derivatives": all(np.isfinite(np.asarray(
            result["levels"][name]["phase_matrix"])).all() for name in plan["levels"]),
    }
    result["prospective_gate_passed"] = all(result["checks"].values())
    result["status"] = "diagnostic_complete"
    result["elapsed_s"] = time.monotonic()-started
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"comparisons": result["comparisons"], "checks": result["checks"]}), flush=True)


if __name__ == "__main__":
    main()

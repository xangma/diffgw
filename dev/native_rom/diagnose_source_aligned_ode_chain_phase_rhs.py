"""Dev-only follow-up: continuous raw-mode phase RHS at a fixed NQC match.

This retains the prior ODE-chain phase matrix and natural-cubic amplitude
inputs. Only the phase RHS derivatives change. The fixed output chart, event,
targets, and public/JAX comparison routes are unchanged.
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
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import (
    _phase_matrix, _source_field,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_ode_chain_phase_rhs_plan.json"
OUTPUT = EVIDENCE / "source_aligned_ode_chain_phase_rhs.json"


def _checked_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-ode-chain-phase-rhs-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "strong_aligned"
            or plan["levels"] != ["baseline", "refined"]
            or plan["runner_sentinel"] != {"half_width_M": 0.0, "degree": 0}):
        raise ValueError("frozen ODE-chain RHS plan changed")
    for key, frozen in plan["inputs"].items():
        path = ROOT / frozen["path"]
        if q.sha256(path) != frozen["sha256"]:
            raise ValueError(f"frozen {key} SHA256 changed: {path}")
    if q.sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen RHS diagnostic script changed")
    return plan


def _rhs_derivatives(time_grid, base_mode, match_time, *, center_state,
                     hamiltonian, flux, field, capture):
    """Keep amplitude spline rows and replace only raw-mode phase rows."""
    amplitude_spline = driver.torch_natural_cubic_spline(time_grid, base_mode.abs())
    amplitude_pair = torch.stack((
        driver._eval_batch_scalar(amplitude_spline, match_time, derivative=1),
        driver._eval_batch_scalar(amplitude_spline, match_time, derivative=2)))
    center = torch.tensor(center_state, dtype=torch.float64, requires_grad=True)
    mode, frequency = driver.torch_mode(
        center, hamiltonian, flux,
        **(driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True}))
    vector = field(center)
    direct_first = (torch.autograd.grad(-torch.angle(mode), center,
                                        retain_graph=True)[0]*vector).sum()
    frequency_dot = (torch.autograd.grad(frequency, center)[0]*vector).sum()
    phase_pair = torch.stack((frequency, frequency_dot)).detach()
    capture.append({
        "raw_mode": [float(mode.real), float(mode.imag)],
        "raw_mode_phase_first": float(frequency.detach()),
        "raw_mode_phase_second": float(frequency_dot.detach()),
        "direct_mode_phase_first_minus_frequency": float(
            (direct_first-frequency).detach()),
        "amplitude_natural_cubic_first_second": amplitude_pair.tolist(),
    })
    return torch.stack((amplitude_pair, phase_pair))


def main() -> None:
    plan = _checked_plan()
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    matrix_only = json.loads((ROOT / plan["inputs"]["matrix_only_result"]["path"]).read_text())
    identity = driver._prepare_public(source_plan)
    field = _source_field()
    m1, m2, chi1, chi2, _ = driver.CASES[plan["case"]]

    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)

    hamiltonian = driver.TorchSourceHam(
        tensor(m1*m2/(m1+m2)**2), tensor(chi1), tensor(chi2),
        delta=tensor((m1-m2)/(m1+m2)))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.source-aligned-ode-chain-phase-rhs.v1",
        "status": "running", "release_qualified": False,
        "scope": "Torch-only dev counterfactual: continuous phase matrix and raw-mode phase RHS; natural-cubic amplitude inputs",
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
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
        phase_capture = []
        rhs_capture = []

        def phase_hook(time_grid, state, orbital_frequency, match_time, *,
                       half_width_M, degree):
            return _phase_matrix(
                time_grid, state, orbital_frequency, match_time,
                half_width_M=half_width_M, degree=degree,
                field=field, capture=phase_capture)

        def rhs_hook(time_grid, base_mode, match_time, *, half_width_M, degree):
            if (half_width_M, degree) != (0.0, 0) or len(phase_capture) != 1:
                raise ValueError("phase match state must precede raw-mode RHS")
            return _rhs_derivatives(
                time_grid, base_mode, match_time,
                center_state=phase_capture[0]["interpolated_state"],
                hamiltonian=hamiltonian, flux=flux, field=field,
                capture=rhs_capture)

        with (patch.object(driver, "_polynomial_phase_matrix", phase_hook),
              patch.object(driver, "_polynomial_rhs_derivatives", rhs_hook)):
            diagnostic, torch_series, jax_series, stage = driver.run(
                plan["case"], source_plan, identity, variant="source",
                independent_init=True, numerical_settings=settings,
                diagnostic_nqc_phase_matrix=plan["runner_sentinel"],
                diagnostic_nqc_rhs_derivatives=plan["runner_sentinel"],
                write_result=False, return_series=True, return_stages=True)
        if len(phase_capture) != 1 or len(rhs_capture) != 1:
            raise ValueError("expected one phase matrix and RHS evaluation")
        series[name] = torch_series
        matrix = np.asarray(stage["nqc_phase_matrix"])
        rhs = np.asarray(stage["nqc_phase_rhs"])
        coefficients = np.asarray(stage["nqc_coefficients"][1])
        prior_level = matrix_only["levels"][name]
        result["levels"][name] = {
            "settings": settings, "match_time_M": stage["match_time_M"],
            "chain_matrix": phase_capture[0], "raw_mode_rhs": rhs_capture[0],
            "phase_matrix": matrix.tolist(), "phase_rhs": rhs.tolist(),
            "phase_coefficients": coefficients.tolist(),
            "amplitude_coefficients": np.asarray(stage["nqc_coefficients"][0]).tolist(),
            "phase_linear_residual_linf": float(np.max(abs(matrix@coefficients-rhs))),
            "phase_matrix_minus_prior_linf": float(np.max(abs(
                matrix-np.asarray(prior_level["phase_matrix"])))),
            "phase_rhs_minus_prior": (
                rhs-np.asarray(prior_level["phase_rhs"])).tolist(),
            "torch": _series_summary(torch_series),
            "jax_natural_cubic": _series_summary(jax_series),
            "torch_public_complex_relative_l2": diagnostic["torch_public"]["amplitude_peak_anchored"]["metrics"]["complex_relative_l2"],
            "torch_to_jax_natural_cubic": _compare(torch_series, jax_series),
        }
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(name, result["levels"][name]["phase_coefficients"], flush=True)
    candidate = _compare(series["baseline"], series["refined"])
    mcomp = matrix_only["comparisons"]
    natural = mcomp["natural_cubic_torch_baseline_to_refined"]
    matrix_only_drift = mcomp["candidate_torch_baseline_to_refined"]
    baseline, refined = (result["levels"][name] for name in plan["levels"])
    mb, mr = (np.asarray(item["phase_matrix"]) for item in (baseline, refined))
    rb, rr = (np.asarray(item["phase_rhs"]) for item in (baseline, refined))
    cb, cr = (np.asarray(item["phase_coefficients"]) for item in (baseline, refined))
    result["comparisons"] = {
        "candidate_torch_baseline_to_refined": candidate,
        "matrix_only_torch_baseline_to_refined": matrix_only_drift,
        "natural_cubic_torch_baseline_to_refined": natural,
        "candidate_to_natural_drift_ratio": candidate["complex_relative_l2"]/natural["complex_relative_l2"],
        "candidate_to_matrix_only_drift_ratio": candidate["complex_relative_l2"]/matrix_only_drift["complex_relative_l2"],
        "phase_matrix_adjacent_frobenius": float(np.linalg.norm(mb-mr)),
        "phase_rhs_adjacent_l2": float(np.linalg.norm(rb-rr)),
        "phase_coefficient_adjacent_l2": float(np.linalg.norm(cb-cr)),
        "matrix_first_coefficient_shift_l2": float(np.linalg.norm(np.linalg.solve(mr,rb)-cb)),
        "rhs_first_coefficient_shift_l2": float(np.linalg.norm(np.linalg.solve(mb,rr)-cb)),
    }
    limits = plan["checks"]
    result["checks"] = {
        "whole_mode_drift_at_most": candidate["complex_relative_l2"] <= limits["whole_mode_drift_at_most"],
        "drift_below_matrix_only": candidate["complex_relative_l2"] < matrix_only_drift["complex_relative_l2"],
        "phase_matrix_replay_linf_at_most": max(
            result["levels"][name]["phase_matrix_minus_prior_linf"] for name in plan["levels"])
            <= limits["phase_matrix_replay_linf_at_most"],
        "phase_linear_residual_at_most": max(
            result["levels"][name]["phase_linear_residual_linf"] for name in plan["levels"])
            <= limits["phase_linear_residual_at_most"],
        "direct_phase_first_agreement_at_most": max(abs(
            result["levels"][name]["raw_mode_rhs"]["direct_mode_phase_first_minus_frequency"])
            for name in plan["levels"]) <= limits["direct_phase_first_agreement_at_most"],
    }
    result["prospective_gate_passed"] = all(result["checks"].values())
    result["status"] = "diagnostic_complete"
    result["elapsed_s"] = time.monotonic()-started
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"comparisons": result["comparisons"], "checks": result["checks"]}), flush=True)


if __name__ == "__main__":
    main()

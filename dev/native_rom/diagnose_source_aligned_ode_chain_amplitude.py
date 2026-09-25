"""Dev-only continuous-ODE NQC amplitude matching at a fixed strong-spin match.

Replays the previously frozen phase matrix/RHS intervention. Only the four
amplitude matching columns (raw A and A*q_j) at the match are replaced by
their ODE values and first/second Lie derivatives. The attachment's sampled
inspiral and ringdown construction remain the original numerical route.
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
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_rhs import _rhs_derivatives
from torchwave.approximants.eob import numerical_attachment as attachment


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_ode_chain_amplitude_plan.json"
OUTPUT = EVIDENCE / "source_aligned_ode_chain_amplitude.json"


def _checked_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-ode-chain-amplitude-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "strong_aligned"
            or plan["levels"] != ["baseline", "refined"]
            or plan["runner_sentinel"] != {"half_width_M": 0.0, "degree": 0}):
        raise ValueError("frozen amplitude plan changed")
    for key, frozen in plan["inputs"].items():
        path = ROOT / frozen["path"]
        if q.sha256(path) != frozen["sha256"]:
            raise ValueError(f"frozen {key} SHA256 changed: {path}")
    if q.sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen amplitude diagnostic script changed")
    return plan


def _continuous_amplitude(center_state, hamiltonian, flux, field):
    center = torch.tensor(center_state, dtype=torch.float64, requires_grad=True)
    vector = field(center)
    mode, _ = driver.torch_mode(
        center, hamiltonian, flux,
        **(driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True}))
    basis, _ = driver.torch_basis(center[0], center[2], vector[1])
    amplitude = mode.abs()
    values = torch.cat((amplitude[None], amplitude*basis))
    first = torch.stack(tuple(
        (torch.autograd.grad(values[i], center, create_graph=True,
                             retain_graph=True)[0]*vector).sum()
        for i in range(4)))
    mode_dot = (torch.autograd.grad(mode.real, center, retain_graph=True)[0]*vector).sum() + 1j*(
        torch.autograd.grad(mode.imag, center, retain_graph=True)[0]*vector).sum()
    direct_amplitude_dot = (mode.conj()*mode_dot).real/amplitude
    second = torch.stack(tuple(
        (torch.autograd.grad(first[i], center, retain_graph=True)[0]*vector).sum()
        for i in range(4)))
    rows = torch.stack((values, first, second)).detach()
    return rows, {
        "raw_mode": [float(mode.real.detach()), float(mode.imag.detach())],
        "amplitude_columns_value_first_second": rows.tolist(),
        "direct_amplitude_first_difference": float((first[0]-direct_amplitude_dot).detach()),
    }


def _amplitude_hook(original_full_series, original_eval, rows, capture):
    """Replace only the first four values of the combined spline's first three queries."""
    def full_series(*args, **kwargs):
        if capture:
            raise ValueError("expected one NQC full-series solve")
        queried = []

        def controlled_eval(spline, coordinate, derivative=0):
            value = original_eval(spline, coordinate, derivative=derivative)
            if len(queried) >= 3:
                return value
            expected = len(queried)
            if derivative != expected or value.shape[-1] != 6:
                raise ValueError("unexpected full-series matching query order")
            queried.append({
                "derivative": derivative,
                "spline_amplitude_columns": value[..., :4].detach().tolist(),
            })
            return torch.cat((rows[expected].to(value), value[..., 4:]), dim=-1)

        with patch.object(attachment, "_eval_batch_scalar", controlled_eval):
            nqc = original_full_series(*args, **kwargs)
        if len(queried) != 3:
            raise ValueError("missing full-series matching queries")
        targets = torch.stack(args[8][:3]).detach().numpy()
        matrix = np.stack((rows[0, 1:].numpy(), rows[1, 1:].numpy(),
                           rows[2, 1:].numpy()))
        rhs = targets-rows[:, 0].numpy()
        coefficients = nqc.calibration.coefficients.amplitude.detach().numpy()
        capture.append({
            "spline_queries": queried,
            "targets": targets.tolist(),
            "amplitude_matrix": matrix.tolist(),
            "amplitude_rhs": rhs.tolist(),
            "amplitude_coefficients": coefficients.tolist(),
            "amplitude_linear_residual_linf": float(np.max(abs(matrix@coefficients-rhs))),
        })
        return nqc

    return full_series


def main() -> None:
    plan = _checked_plan()
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    phase_rhs = json.loads((ROOT / plan["inputs"]["phase_rhs_result"]["path"]).read_text())
    identity = driver._prepare_public(source_plan)
    field = _source_field()
    m1, m2, chi1, chi2, _ = driver.CASES[plan["case"]]

    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)

    hamiltonian = driver.TorchSourceHam(
        tensor(m1*m2/(m1+m2)**2), tensor(chi1), tensor(chi2),
        delta=tensor((m1-m2)/(m1+m2)))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    original_full_series = attachment._full_series_nqc
    original_eval = attachment._eval_batch_scalar
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.source-aligned-ode-chain-amplitude.v1",
        "status": "running", "release_qualified": False,
        "scope": "Torch-only continuous amplitude matching; fixed ODE phase matrix/RHS, JAX/public controls",
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
        phase_capture, rhs_capture, amplitude_capture, raw_capture = [], [], [], []

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

        def full_series_hook(*args, **kwargs):
            if len(phase_capture) != 1 or len(rhs_capture) != 1:
                raise ValueError("phase controls must precede amplitude solve")
            rows, raw = _continuous_amplitude(
                phase_capture[0]["interpolated_state"], hamiltonian, flux, field)
            raw_capture.append(raw)
            return _amplitude_hook(
                original_full_series, original_eval, rows, amplitude_capture)(*args, **kwargs)

        with (patch.object(driver, "_polynomial_phase_matrix", phase_hook),
              patch.object(driver, "_polynomial_rhs_derivatives", rhs_hook),
              patch.object(attachment, "_full_series_nqc", full_series_hook)):
            diagnostic, torch_series, jax_series, stage = driver.run(
                plan["case"], source_plan, identity, variant="source",
                independent_init=True, numerical_settings=settings,
                diagnostic_nqc_phase_matrix=plan["runner_sentinel"],
                diagnostic_nqc_rhs_derivatives=plan["runner_sentinel"],
                write_result=False, return_series=True, return_stages=True)
        if any(len(item) != 1 for item in (
                phase_capture, rhs_capture, amplitude_capture, raw_capture)):
            raise ValueError("expected one matching event and one amplitude solve")
        series[name] = torch_series
        prior = phase_rhs["levels"][name]
        amp = amplitude_capture[0]
        result["levels"][name] = {
            "settings": settings, "match_time_M": stage["match_time_M"],
            "chain_matrix": phase_capture[0], "raw_mode_rhs": rhs_capture[0],
            "raw_mode_amplitude": raw_capture[0], "amplitude_solve": amp,
            "phase_matrix": np.asarray(stage["nqc_phase_matrix"]).tolist(),
            "phase_rhs": np.asarray(stage["nqc_phase_rhs"]).tolist(),
            "phase_coefficients": np.asarray(stage["nqc_coefficients"][1]).tolist(),
            "phase_matrix_minus_prior_linf": float(np.max(abs(
                np.asarray(stage["nqc_phase_matrix"])-np.asarray(prior["phase_matrix"])))),
            "phase_rhs_minus_prior_linf": float(np.max(abs(
                np.asarray(stage["nqc_phase_rhs"])-np.asarray(prior["phase_rhs"])))),
            "phase_coefficients_minus_prior_linf": float(np.max(abs(
                np.asarray(stage["nqc_coefficients"][1])-np.asarray(prior["phase_coefficients"])))),
            "torch": _series_summary(torch_series),
            "jax_natural_cubic": _series_summary(jax_series),
            "torch_public_complex_relative_l2": diagnostic["torch_public"]["amplitude_peak_anchored"]["metrics"]["complex_relative_l2"],
            "torch_to_jax_natural_cubic": _compare(torch_series, jax_series),
        }
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(name, amp["amplitude_coefficients"], flush=True)
    candidate = _compare(series["baseline"], series["refined"])
    prior_comparison = phase_rhs["comparisons"]["candidate_torch_baseline_to_refined"]
    baseline, refined = (result["levels"][name] for name in plan["levels"])
    ab, ar = (np.asarray(item["amplitude_solve"]["amplitude_coefficients"])
              for item in (baseline, refined))
    mb, mr = (np.asarray(item["amplitude_solve"]["amplitude_matrix"])
              for item in (baseline, refined))
    rb, rr = (np.asarray(item["amplitude_solve"]["amplitude_rhs"])
              for item in (baseline, refined))
    result["comparisons"] = {
        "candidate_torch_baseline_to_refined": candidate,
        "phase_rhs_only_torch_baseline_to_refined": prior_comparison,
        "whole_drift_ratio_to_phase_rhs_only": candidate["complex_relative_l2"]/prior_comparison["complex_relative_l2"],
        "amplitude_coefficient_adjacent_l2": float(np.linalg.norm(ab-ar)),
        "phase_rhs_only_amplitude_coefficient_adjacent_l2": float(np.linalg.norm(
            np.asarray(phase_rhs["levels"]["baseline"]["amplitude_coefficients"])
            -np.asarray(phase_rhs["levels"]["refined"]["amplitude_coefficients"]))),
        "amplitude_matrix_adjacent_frobenius": float(np.linalg.norm(mb-mr)),
        "amplitude_rhs_adjacent_l2": float(np.linalg.norm(rb-rr)),
        "matrix_first_coefficient_shift_l2": float(np.linalg.norm(np.linalg.solve(mr,rb)-ab)),
        "rhs_first_coefficient_shift_l2": float(np.linalg.norm(np.linalg.solve(mb,rr)-ab)),
    }
    limits = plan["checks"]
    result["checks"] = {
        "whole_mode_drift_at_most": candidate["complex_relative_l2"] <= limits["whole_mode_drift_at_most"],
        "drift_below_phase_rhs_only": candidate["complex_relative_l2"] < prior_comparison["complex_relative_l2"],
        "amplitude_coefficient_drift_at_most": result["comparisons"]["amplitude_coefficient_adjacent_l2"] <= limits["amplitude_coefficient_drift_at_most"],
        "amplitude_linear_residual_at_most": max(
            item["amplitude_solve"]["amplitude_linear_residual_linf"] for item in (baseline, refined)) <= limits["amplitude_linear_residual_at_most"],
        "phase_replay_linf_at_most": max(
            max(item[key] for key in ("phase_matrix_minus_prior_linf", "phase_rhs_minus_prior_linf",
                                       "phase_coefficients_minus_prior_linf")) for item in (baseline, refined)) <= limits["phase_replay_linf_at_most"],
        "direct_amplitude_first_agreement_at_most": max(abs(
            item["raw_mode_amplitude"]["direct_amplitude_first_difference"])
            for item in (baseline, refined)) <= limits["direct_amplitude_first_agreement_at_most"],
    }
    result["prospective_gate_passed"] = all(result["checks"].values())
    result["status"] = "diagnostic_complete"
    result["elapsed_s"] = time.monotonic()-started
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"comparisons": result["comparisons"], "checks": result["checks"]}), flush=True)


if __name__ == "__main__":
    main()

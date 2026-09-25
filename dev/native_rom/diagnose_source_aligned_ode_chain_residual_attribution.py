"""Dev-only time-window attribution of the frozen continuous-NQC residual."""

from __future__ import annotations

import hashlib
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
from dev.native_rom.diagnose_source_aligned_ode_chain_amplitude_v2 import (
    _amplitude_hook, _continuous_amplitude,
)
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import (
    _phase_matrix, _source_field,
)
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_rhs import _rhs_derivatives
from torchwave.approximants.eob import numerical_attachment as attachment


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_ode_chain_residual_attribution_plan.json"
OUTPUT = EVIDENCE / "source_aligned_ode_chain_residual_attribution.json"


def _checked_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-ode-chain-residual-attribution-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "strong_aligned"
            or plan["levels"] != ["baseline", "refined"]
            or plan["runner_sentinel"] != {"half_width_M": 0.0, "degree": 0}
            or plan["nqc_window_M"] != 50.0
            or plan["edge_window_M"] != 50.0):
        raise ValueError("frozen residual-attribution plan changed")
    for key, frozen in plan["inputs"].items():
        path = ROOT / frozen["path"]
        if q.sha256(path) != frozen["sha256"]:
            raise ValueError(f"frozen {key} SHA256 changed: {path}")
    if q.sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen residual-attribution script changed")
    return plan


def _window_metrics(time_grid, error_squared, reference_squared, mask):
    count = int(mask.sum())
    numerator = float(error_squared[mask].sum())
    denominator = float(reference_squared[mask].sum())
    return {
        "samples": count,
        "first_M": float(time_grid[mask][0]) if count else None,
        "last_M": float(time_grid[mask][-1]) if count else None,
        "error_l2": float(np.sqrt(numerator)),
        "reference_l2": float(np.sqrt(denominator)),
        "error_squared": numerator,
        "reference_squared": denominator,
        "local_complex_relative_l2": float(np.sqrt(numerator/denominator)) if denominator else None,
        "share_of_total_error_squared": float(numerator/error_squared.sum()) if error_squared.sum() else None,
        "share_of_total_reference_squared": float(denominator/reference_squared.sum()),
        "max_absolute_error": float(np.sqrt(error_squared[mask].max())) if count else None,
    }


def _run_level(name, settings, source_plan, identity, case, sentinel, field,
               hamiltonian, flux, original_full_series, original_eval):
    phase_capture, rhs_capture, amplitude_capture, raw_capture, amplitude_rows = [], [], [], [], []

    def phase_hook(time_grid, state, orbital_frequency, match_time, *, half_width_M, degree):
        return _phase_matrix(
            time_grid, state, orbital_frequency, match_time,
            half_width_M=half_width_M, degree=degree, field=field,
            capture=phase_capture)

    def rhs_hook(time_grid, base_mode, match_time, *, half_width_M, degree):
        if (half_width_M, degree) != (0.0, 0) or len(phase_capture) != 1:
            raise ValueError("phase match state must precede raw-mode RHS")
        rows, raw = _continuous_amplitude(
            phase_capture[0]["interpolated_state"], hamiltonian, flux, field)
        amplitude_rows.append(rows)
        raw_capture.append(raw)
        rhs = _rhs_derivatives(
            time_grid, base_mode, match_time,
            center_state=phase_capture[0]["interpolated_state"],
            hamiltonian=hamiltonian, flux=flux, field=field,
            capture=rhs_capture)
        rhs[0, :] = rows[1:, 0]
        return rhs

    def full_series_hook(*args, **kwargs):
        if any(len(item) != 1 for item in (
                phase_capture, rhs_capture, amplitude_rows, raw_capture)):
            raise ValueError("phase and amplitude controls must precede matching solve")
        return _amplitude_hook(
            original_full_series, original_eval, amplitude_rows[0], amplitude_capture)(
                *args, **kwargs)

    with (patch.object(driver, "_polynomial_phase_matrix", phase_hook),
          patch.object(driver, "_polynomial_rhs_derivatives", rhs_hook),
          patch.object(attachment, "_full_series_nqc", full_series_hook)):
        diagnostic, torch_series, _, stage = driver.run(
            case, source_plan, identity, variant="source", independent_init=True,
            numerical_settings=settings,
            diagnostic_nqc_phase_matrix=sentinel,
            diagnostic_nqc_rhs_derivatives=sentinel,
            write_result=False, return_series=True, return_stages=True)
    if any(len(item) != 1 for item in (
            phase_capture, rhs_capture, amplitude_capture, raw_capture, amplitude_rows)):
        raise ValueError("expected one matching event and amplitude solve")
    return diagnostic, torch_series, stage


def main() -> None:
    plan = _checked_plan()
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    prior = json.loads((ROOT / plan["inputs"]["amplitude_v2_result"]["path"]).read_text())
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
        "schema": "diffgw.seobnrv4.source-aligned-ode-chain-residual-attribution.v1",
        "status": "running", "release_qualified": False,
        "scope": "Torch-only decomposition of the frozen v2 baseline/refined complex waveform residual",
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
        "levels": {}, "comparison": {}, "checks": {},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    runs = {}
    for name in plan["levels"]:
        level = next(item for item in ladder["levels"] if item["name"] == name)
        settings = {
            "coarse_output_step_M": ladder["coarse_output_step_M"],
            "max_step_M": ladder["max_step_M"],
            "atol": level["atol"], "rtol": level["rtol"],
        }
        diagnostic, series, stage = _run_level(
            name, settings, source_plan, identity, plan["case"],
            plan["runner_sentinel"], field, hamiltonian, flux,
            original_full_series, original_eval)
        summary = _series_summary(series)
        if summary != prior["levels"][name]["torch"]:
            raise ValueError(f"{name} v2 whole-series replay changed")
        runs[name] = (series, stage)
        result["levels"][name] = {
            "settings": settings, "torch": summary,
            "match_time_M": stage["match_time_M"],
            "boundary_time_M": stage["boundary_time_M"],
            "restart_index": stage["restart_index"],
            "fine_time_sha256": hashlib.sha256(
                np.asarray(stage["fine_time"]).tobytes()).hexdigest(),
            "fine_base_mode_sha256": hashlib.sha256(
                np.asarray(stage["fine_base_mode"]).tobytes()).hexdigest(),
            "public_complex_relative_l2": diagnostic["torch_public"]["amplitude_peak_anchored"]["metrics"]["complex_relative_l2"],
        }
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(name, summary["mode_sha256"], flush=True)

    (baseline, bs), (refined, rs) = runs["baseline"], runs["refined"]
    comparison = _compare(baseline, refined)
    prior_comparison = prior["comparisons"]["candidate_torch_baseline_to_refined"]
    if comparison != prior_comparison:
        raise ValueError("v2 whole-waveform comparison did not replay")
    bt, bm = baseline
    rt, rm = refined
    if len(bt) != len(rt) or not np.array_equal(bt, rt):
        raise ValueError("whole-waveform charts differ")
    match = float(bs["match_time_M"])
    boundary = float(bs["boundary_time_M"])
    if (match != float(rs["match_time_M"])
            or boundary != float(rs["boundary_time_M"])):
        raise ValueError("matching or attachment boundary differs between levels")
    error_squared = np.abs(bm-rm)**2
    reference_squared = np.abs(rm)**2
    nqc_start = match-plan["nqc_window_M"]
    windows = {
        "early_pre_match": bt < nqc_start,
        "nqc_window_inspiral": (bt >= nqc_start) & (bt < boundary),
        "post_match_inspiral_plunge": (bt >= match) & (bt < boundary),
        "ringdown": bt >= boundary,
    }
    partition = (windows["early_pre_match"].astype(int)
                 + windows["nqc_window_inspiral"].astype(int)
                 + windows["ringdown"].astype(int))
    if not np.all(partition == 1):
        raise ValueError("attribution windows do not partition shared support")
    edges = {
        "first_50_M": bt < bt[0]+plan["edge_window_M"],
        "last_50_M": bt > bt[-1]-plan["edge_window_M"],
    }
    if not (np.array_equal(bs["fine_time"], rs["fine_time"])
            and np.asarray(bs["fine_base_mode"]).shape
            == np.asarray(rs["fine_base_mode"]).shape):
        raise ValueError("raw fine pre-NQC modes lack identical support")
    fine_time = np.asarray(bs["fine_time"])
    raw_baseline, raw_refined = (np.asarray(stage["fine_base_mode"])
                                 for stage in (bs, rs))
    raw_error_squared = np.abs(raw_baseline-raw_refined)**2
    raw_reference_squared = np.abs(raw_refined)**2
    raw_masks = {
        "fine_all": np.ones_like(fine_time, dtype=bool),
        "fine_before_boundary": fine_time < boundary,
        "fine_last_50_M_before_match": (fine_time >= nqc_start) & (fine_time < boundary),
    }
    result["comparison"] = {
        "whole": comparison,
        "match_time_M": match, "boundary_time_M": boundary,
        "nqc_window_start_M": nqc_start,
        "window_policy": "Early: t<match-50; NQC inspiral: match-50<=t<boundary; post-match inspiral: match<=t<boundary; ringdown: t>=boundary. Ringdown owns the boundary sample.",
        "time_M": bt.tolist(),
        "error_squared_by_sample": error_squared.tolist(),
        "reference_squared_by_sample": reference_squared.tolist(),
        "windows": {name: _window_metrics(bt, error_squared, reference_squared, mask)
                    for name, mask in windows.items()},
        "edges": {name: _window_metrics(bt, error_squared, reference_squared, mask)
                  for name, mask in edges.items()},
        "raw_fine_pre_nqc": {
            "fine_time_max_abs_M": float(np.max(abs(fine_time-np.asarray(rs["fine_time"])))),
            "windows": {name: _window_metrics(fine_time, raw_error_squared,
                                             raw_reference_squared, mask)
                        for name, mask in raw_masks.items()},
        },
    }
    limits = plan["checks"]
    result["checks"] = {
        "whole_replay_l2_at_most": abs(comparison["complex_relative_l2"]
                                         - prior_comparison["complex_relative_l2"])
            <= limits["whole_replay_l2_at_most"],
        "same_match_and_boundary": match == float(rs["match_time_M"])
            and boundary == float(rs["boundary_time_M"]),
        "window_partition_exact": bool(np.all(partition == 1)),
        "window_error_square_reconstruction_at_most": abs(sum(
            result["comparison"]["windows"][name]["error_squared"]
            for name in ("early_pre_match", "nqc_window_inspiral", "ringdown"))
            - float(error_squared.sum())) <= limits["window_error_square_reconstruction_at_most"],
    }
    result["prospective_checks_passed"] = all(result["checks"].values())
    result["status"] = "diagnostic_complete"
    result["elapsed_s"] = time.monotonic()-started
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"windows": result["comparison"]["windows"],
                      "edges": result["comparison"]["edges"],
                      "raw": result["comparison"]["raw_fine_pre_nqc"],
                      "checks": result["checks"]}), flush=True)


if __name__ == "__main__":
    main()

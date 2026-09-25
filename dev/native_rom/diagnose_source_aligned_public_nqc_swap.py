"""Dev-only fixed-trajectory NQC-coefficient sensitivity to pinned public v4.

The historical-source coefficients are applied to the native trajectory only as
an intervention. They are not a refit or a valid replacement calibration.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _series_summary
from dev.native_rom.diagnose_peak_anchor import public_mode
from dev.native_rom.diagnose_source_aligned_ode_chain_boundary_frequency import _capture_level
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _source_field
from dev.native_rom.diagnose_source_aligned_public_gap import _common_chart, _sha_array
from torchwave.approximants.eob import numerical_attachment as attachment
from torchwave.approximants.eob.nqc import nqc_22_basis


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_public_nqc_swap_plan.json"
OUTPUT = EVIDENCE / "source_aligned_public_nqc_swap.json"


def _checked_plan():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-public-nqc-swap-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "strong_aligned"
            or plan["level"] != "refined"
            or plan["pre_match_width_M"] != 50.0
            or plan["boundary_margin_M"] != 1.0
            or plan["script_sha256"] != q.sha256(Path(__file__))):
        raise ValueError("frozen NQC-swap plan changed")
    for key, item in plan["inputs"].items():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen NQC-swap input changed: {key}")
    return plan


def _complex_l2(candidate, reference):
    return float(np.linalg.norm(candidate-reference)/np.linalg.norm(reference))


def _window(candidate, reference, mask, baseline_error_squared):
    difference = candidate[mask]-reference[mask]
    error_squared = float(np.vdot(difference, difference).real)
    reference_squared = float(np.vdot(reference[mask], reference[mask]).real)
    return {
        "samples": int(mask.sum()),
        "local_complex_relative_l2": float(np.sqrt(error_squared/reference_squared)),
        "error_squared": error_squared,
        "error_squared_delta_from_baseline": error_squared-baseline_error_squared,
        "error_squared_ratio_to_baseline": error_squared/baseline_error_squared,
    }


def _factors(q_basis, p_basis, coefficients):
    a, b = coefficients
    amplitude = 1 + q_basis @ a
    phase = p_basis @ b
    return amplitude, phase


def main():
    plan = _checked_plan()
    previous = json.loads((ROOT / plan["inputs"]["previous_result"]["path"]).read_text())
    boundary_plan = json.loads((ROOT / plan["inputs"]["boundary_plan"]["path"]).read_text())
    boundary_result = json.loads((ROOT / plan["inputs"]["boundary_result"]["path"]).read_text())
    captured = json.loads((ROOT / plan["inputs"]["historical_coefficients"]["path"]).read_text())
    source_plan = json.loads((ROOT / boundary_plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / boundary_plan["inputs"]["ladder_plan"]["path"]).read_text())
    if (previous["status"] != "diagnostic_complete"
            or not previous["diagnostic_checks_passed"]
            or captured["status"] != "diagnostic_complete"):
        raise ValueError("pinned prior result is incomplete")
    source_coefficients = np.asarray(
        captured["baselines"]["source"]["captured_active_coefficients"], dtype=float)
    if source_coefficients.shape != (5,) or not np.isfinite(source_coefficients).all():
        raise ValueError("historical source active coefficient vector is invalid")
    identity = driver._prepare_public(source_plan)
    level = next(x for x in ladder["levels"] if x["name"] == plan["level"])
    settings = {"coarse_output_step_M": ladder["coarse_output_step_M"],
                "max_step_M": ladder["max_step_M"], "atol": level["atol"],
                "rtol": level["rtol"]}
    m1, m2, chi1, chi2, omega_start = driver.CASES[plan["case"]]
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)
    hamiltonian = driver.TorchSourceHam(
        tensor(m1*m2/(m1+m2)**2), tensor(chi1), tensor(chi2),
        delta=tensor((m1-m2)/(m1+m2)))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    field = _source_field()
    started = time.monotonic()
    result = {"schema": "diffgw.seobnrv4.source-aligned-public-nqc-swap.v1",
              "status": "running", "release_qualified": False,
              "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
              "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
              "checks": {}}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    diagnostic, series, stage, _, _ = _capture_level(
        plan["level"], settings, source_plan, identity, boundary_plan,
        field, hamiltonian, flux,
        attachment._full_series_nqc, attachment._eval_batch_scalar)
    time_grid, baseline_mode = map(np.asarray, series)
    frozen = boundary_result["levels"][plan["level"]]
    restart = int(stage["restart_index"])
    boundary = float(stage["boundary_time_M"])
    pre_boundary_count = int(np.searchsorted(time_grid, boundary))
    coarse_state = tensor(np.asarray(stage["coarse_state"])[:restart])
    with torch.no_grad():
        coarse_omega = field(coarse_state)[:, 1]
        coarse_raw = driver.torch_mode(
            coarse_state, hamiltonian, flux,
            **(driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True}))[0]
        coarse_q, coarse_p = nqc_22_basis(
            coarse_state[:, 0], coarse_state[:, 2], coarse_omega)
    fine_index = np.arange(pre_boundary_count-restart)*boundary_plan["fine_refinement_factor"]
    fine_time = np.asarray(stage["fine_time"])[fine_index]
    if not np.allclose(fine_time, time_grid[restart:pre_boundary_count], rtol=0, atol=1e-12):
        raise ValueError("fine/output charts do not share sample indices")
    fine_state = tensor(np.asarray(stage["fine_state"])[fine_index])
    fine_omega = tensor(np.asarray(stage["fine_orbital_frequency"])[fine_index])
    with torch.no_grad():
        fine_q, fine_p = nqc_22_basis(
            fine_state[:, 0], fine_state[:, 2], fine_omega)
    q_basis = np.concatenate((coarse_q.numpy(), fine_q.numpy()))
    p_basis = np.concatenate((coarse_p.numpy(), fine_p.numpy()))
    raw = np.concatenate((coarse_raw.detach().numpy(),
                          np.asarray(stage["fine_base_mode"])[fine_index]))
    native_a, native_b = (np.asarray(x, dtype=float)
                          for x in stage["nqc_coefficients"])
    source_a, source_b = source_coefficients[:3], source_coefficients[3:]
    native_amp, native_phase = _factors(q_basis, p_basis, (native_a, native_b))
    source_amp, source_phase = _factors(q_basis, p_basis, (source_a, source_b))
    reconstructed = raw*native_amp*np.exp(1j*native_phase)
    pre_boundary = baseline_mode[:pre_boundary_count]
    reconstruction_linf = float(np.max(abs(reconstructed-pre_boundary)))
    reconstruction_relative_l2 = _complex_l2(reconstructed, pre_boundary)
    variants = {
        "baseline_native": reconstructed,
        "source_amplitude_only": raw*source_amp*np.exp(1j*native_phase),
        "source_phase_only": raw*native_amp*np.exp(1j*source_phase),
        "source_both": raw*source_amp*np.exp(1j*source_phase),
    }
    row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, omega_start=omega_start)
    run = {"time": time_grid, "mode": baseline_mode,
           "peak_time_M": float(stage["orbital_peak_time_M"]),
           "dt_geom": settings["coarse_output_step_M"]}
    nt, nm, rt, rm, mass_seconds = public_mode(row, run)
    ni, ri = np.argmax(abs(nm)), np.argmax(abs(rm))
    native_anchor = (nt[ni], np.angle(nm[ni]))
    public_anchor = (rt[ri], np.angle(rm[ri]))
    chart_time, baseline_chart, reference, common = _common_chart(
        nt, nm, rt, rm, mass_seconds, native_anchor, public_anchor)
    native_peak_time_M = float(time_grid[ni])
    match = float(stage["match_time_M"])
    windows = {
        "early_pre_match_minus_50M": chart_time < match-50-native_peak_time_M,
        "nqc_local_50M": ((chart_time >= match-50-native_peak_time_M)
                            & (chart_time < boundary-plan["boundary_margin_M"]
                               -native_peak_time_M)),
        "full_pre_boundary_with_margin": chart_time < boundary-plan["boundary_margin_M"]-native_peak_time_M,
    }
    chart_variants = {}
    for name, pre_mode in variants.items():
        whole_mode = baseline_mode.copy()
        whole_mode[:pre_boundary_count] = pre_mode
        variant_nm = nm.copy()
        variant_nm[:pre_boundary_count] = (
            whole_mode[:pre_boundary_count]*(nm[0]/baseline_mode[0]))
        t, candidate, compare_reference, support = _common_chart(
            nt, variant_nm, rt, rm, mass_seconds, native_anchor, public_anchor)
        if (not np.array_equal(t, chart_time)
                or not np.array_equal(compare_reference, reference)
                or not np.array_equal(support, common)):
            raise ValueError("variant changed public comparison chart or support")
        chart_variants[name] = candidate
    baseline_errors = {name: float(np.vdot(
        baseline_chart[mask]-reference[mask],
        baseline_chart[mask]-reference[mask]).real)
        for name, mask in windows.items()}
    metrics = {name: {window: _window(candidate, reference, mask, baseline_errors[window])
                      for window, mask in windows.items()}
               for name, candidate in chart_variants.items()}
    checks = {
        "candidate_mode_hash_replays": _series_summary(series) == frozen["candidate_torch"],
        "public_binary_identity_replays": identity == previous["identity"],
        "public_mode_hash_replays": _sha_array(rm) == previous["public_mode_sha256"],
        "baseline_chart_samples_replay": int(common.sum()) == previous["common_peak_chart_samples"],
        "baseline_whole_metric_replays": abs(_complex_l2(baseline_chart, reference)
                                             -previous["whole_complex_relative_l2"]) <= 1e-12,
        "native_factor_reconstruction_relative_l2_at_most": reconstruction_relative_l2 <= 1e-12,
        "native_factor_reconstruction_linf_at_most": reconstruction_linf <= 1e-12,
        "variant_noop_chart_replays": bool(np.allclose(
            chart_variants["baseline_native"], baseline_chart, rtol=1e-12, atol=0)),
        "early_window_samples_replay": (int(windows["early_pre_match_minus_50M"].sum())
                                        == previous["peak_chart_windows"]["early_pre_match"]["samples"]),
    }
    result.update({
        "status": "diagnostic_complete", "elapsed_s": time.monotonic()-started,
        "identity": identity, "settings": settings,
        "historical_source_coefficient_provenance": {
            "evidence_path": plan["inputs"]["historical_coefficients"]["path"],
            "evidence_sha256": plan["inputs"]["historical_coefficients"]["sha256"],
            "basis": "N22=(1+sum(a_j*q_j))*exp(i*sum(b_j*p_j)); same 22 basis order in native and historical capture",
            "amplitude_coefficients": source_a.tolist(),
            "phase_coefficients": source_b.tolist(),
            "native_amplitude_coefficients": native_a.tolist(),
            "native_phase_coefficients": native_b.tolist(),
            "historical_coefficients_are_cross_trajectory": True,
        },
        "pre_boundary_samples": pre_boundary_count,
        "baseline_native_pre_boundary_mode_sha256": _sha_array(pre_boundary),
        "reconstructed_native_pre_boundary_mode_sha256": _sha_array(reconstructed),
        "native_factor_reconstruction_linf": reconstruction_linf,
        "native_factor_reconstruction_relative_l2": reconstruction_relative_l2,
        "fixed_chart": {"native_peak_sample_index": int(ni),
                        "public_peak_sample_index": int(ri),
                        "samples": int(common.sum()),
                        "support_M": [float(chart_time[0]), float(chart_time[-1])],
                        "boundary_margin_M": plan["boundary_margin_M"]},
        "metrics": metrics,
        "checks": {name: bool(value) for name, value in checks.items()},
    })
    result["diagnostic_checks_passed"] = all(result["checks"].values())
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"checks": result["checks"], "metrics": metrics}), flush=True)
    if not result["diagnostic_checks_passed"]:
        raise ValueError("NQC-swap replay check failed")


if __name__ == "__main__":
    main()

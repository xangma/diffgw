"""Development-only ODE boundary-frequency counterfactual for frozen v2 NQC."""

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
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _source_field
from dev.native_rom.diagnose_source_aligned_ode_chain_residual_attribution_v2 import _run_level
from dev.native_rom.diagnose_source_aligned_ode_chain_ringdown_inputs import (
    _metrics, _mode, _pairs, _scalar,
)
from torchwave.approximants.eob import numerical_attachment as attachment


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_ode_chain_boundary_frequency_plan.json"
OUTPUT = EVIDENCE / "source_aligned_ode_chain_boundary_frequency.json"


def _checked_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-ode-chain-boundary-frequency-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "strong_aligned"
            or plan["levels"] != ["baseline", "refined"]
            or plan["runner_sentinel"] != {"half_width_M": 0.0, "degree": 0}
            or plan["fine_refinement_factor"] != 32):
        raise ValueError("frozen boundary-frequency plan changed")
    for key, frozen in plan["inputs"].items():
        if q.sha256(ROOT / frozen["path"]) != frozen["sha256"]:
            raise ValueError(f"frozen {key} SHA256 changed")
    if q.sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen boundary-frequency runner SHA256 changed")
    return plan


def _array(value):
    return np.asarray(value.detach().cpu().numpy()).copy()


def _capture_level(name, settings, source_plan, identity, plan, field,
                   hamiltonian, flux, original_full_series, original_eval):
    original_attach = driver.torch_attach
    original_ringdown = attachment.merger_ringdown_22
    original_support = attachment.seobnrv4_ringdown_support_samples
    records, support = [], []

    def attach_hook(time_grid, state, orbital_frequency, base_mode, base_frequency,
                    nu, chi, sigma, query, **kwargs):
        nqc_records = []
        inner_full_series = attachment._full_series_nqc

        def nqc_hook(*args, **nqc_kwargs):
            nqc = inner_full_series(*args, **nqc_kwargs)
            match = args[7]
            index = (torch.searchsorted(time_grid.contiguous(),
                                        match.unsqueeze(-1).contiguous(), right=True)
                     .squeeze(-1)-1)
            if index.numel() != 1:
                raise ValueError("expected one boundary index")
            nqc_records.append((nqc, int(index.item()), match))
            return nqc

        def ringdown_hook(elapsed, ring_sigma, amplitude, amplitude_derivative,
                          frequency, phase, shape):
            if len(nqc_records) != 1:
                raise ValueError("NQC solve must precede ringdown")
            nqc, index, match = nqc_records[0]
            if index < 1 or index+1 >= len(time_grid):
                raise ValueError("boundary lacks central-secant neighbors")
            with torch.enable_grad():
                center = state[index].detach().clone().requires_grad_(True)
                vector = field(center)
                raw_mode, raw_frequency = driver.torch_mode(
                    center, hamiltonian, flux,
                    **(driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True}))
                direct_phase_first = (
                    torch.autograd.grad(-torch.angle(raw_mode), center,
                                        retain_graph=True)[0]*vector).sum()
                basis = driver.torch_basis(center[0], center[2], vector[1])[1]
                basis_first = torch.stack(tuple(
                    (torch.autograd.grad(basis[i], center,
                                         retain_graph=True)[0]*vector).sum()
                    for i in range(2)))
                b = nqc.calibration.coefficients.phase.detach()
                continuous = (raw_frequency-basis_first.dot(b)).detach()
            boundary = time_grid[index]
            sampled_phase = nqc.sampled_corrected_phase
            central_secant = ((sampled_phase[index+1]-sampled_phase[index-1])
                              /(time_grid[index+1]-time_grid[index-1]))
            spline_first = (
                attachment._eval_batch_scalar(nqc.waveform_phase_spline, boundary,
                                              derivative=1)
                -(attachment._eval_batch_scalar(nqc.phase_basis_spline, boundary,
                                                derivative=1)*b).sum())
            actual_step = time_grid[index]-time_grid[index-1]
            backward_secant = (sampled_phase[index]-sampled_phase[index-1])/actual_step
            inputs = {
                "elapsed": _array(elapsed),
                "sigma": _scalar(ring_sigma),
                "amplitude": _scalar(amplitude),
                "amplitude_derivative": _scalar(amplitude_derivative),
                "frequency": _scalar(frequency),
                "phase": _scalar(phase),
                "shape_amplitude_rate": _scalar(shape.amplitude_rate),
                "shape_amplitude_offset": _scalar(shape.amplitude_offset),
                "shape_phase_rate": _scalar(shape.phase_rate),
                "shape_phase_offset": _scalar(shape.phase_offset),
            }
            controls = {
                "boundary_index": index,
                "boundary_time_local_M": float(boundary),
                "match_time_local_M": float(match),
                "high_step_M": float(actual_step),
                "phase_coefficients": b.tolist(),
                "sampled_phase_basis": _array(nqc.sampled_phase_basis[index]).tolist(),
                "ode_phase_basis": _array(basis.detach()).tolist(),
                "ode_phase_basis_first": _array(basis_first.detach()).tolist(),
                "raw_frequency": float(raw_frequency.detach()),
                "sampled_raw_frequency": float(base_frequency[index]),
                "direct_raw_phase_first": float(direct_phase_first.detach()),
                "original_frequency": float(frequency),
                "backward_secant": float(backward_secant),
                "central_secant_control": float(central_secant),
                "spline_derivative_control": float(spline_first),
                "continuous_frequency": float(continuous),
            }
            records.append((inputs, controls))
            return original_ringdown(
                elapsed, ring_sigma, amplitude, amplitude_derivative,
                continuous, phase, shape)

        def support_hook(damping, dt_high):
            count = original_support(damping, dt_high)
            support.append({"count": int(_scalar(count)),
                            "dt_high": float(_scalar(dt_high))})
            return count

        with (patch.object(attachment, "_full_series_nqc", nqc_hook),
              patch.object(attachment, "merger_ringdown_22", ringdown_hook),
              patch.object(attachment, "seobnrv4_ringdown_support_samples", support_hook)):
            return original_attach(time_grid, state, orbital_frequency, base_mode,
                                   base_frequency, nu, chi, sigma, query, **kwargs)

    with patch.object(driver, "torch_attach", attach_hook):
        diagnostic, series, stage = _run_level(
            name, settings, source_plan, identity, plan["case"],
            plan["runner_sentinel"], field, hamiltonian, flux,
            original_full_series, original_eval)
    if len(records) != 1 or len(support) != 1:
        raise ValueError("expected one boundary and support capture")
    inputs, controls = records[0]
    inputs["support_count"] = support[0]["count"]
    inputs["dt_high"] = support[0]["dt_high"]
    inputs["support_end"] = support[0]["count"]*support[0]["dt_high"]
    return diagnostic, series, stage, inputs, controls


def main() -> None:
    plan = _checked_plan()
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    prior = json.loads((ROOT / plan["inputs"]["amplitude_v2_result"]["path"]).read_text())
    capture_prior = json.loads((ROOT / plan["inputs"]["ringdown_inputs_result"]["path"]).read_text())
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
    original_ringdown = attachment.merger_ringdown_22
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.source-aligned-ode-chain-boundary-frequency.v1",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
        "levels": {}, "comparisons": {}, "checks": {},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    runs = {}
    for name in plan["levels"]:
        level = next(item for item in ladder["levels"] if item["name"] == name)
        settings = {"coarse_output_step_M": ladder["coarse_output_step_M"],
                    "max_step_M": ladder["max_step_M"],
                    "atol": level["atol"], "rtol": level["rtol"]}
        diagnostic, series, stage, inputs, controls = _capture_level(
            name, settings, source_plan, identity, plan, field, hamiltonian, flux,
            original_full_series, original_eval)
        whole_time, candidate_mode = series
        prior_level = capture_prior["levels"][name]
        ring_mask = np.asarray(whole_time) >= float(stage["boundary_time_M"])
        whole_index = np.flatnonzero(ring_mask)
        chart_index = plan["fine_refinement_factor"]*(whole_index-int(stage["restart_index"]))
        original_ring = _mode(inputs, original_ringdown=original_ringdown,
                              chart_index=chart_index)
        candidate_inputs = dict(inputs)
        candidate_inputs["frequency"] = controls["continuous_frequency"]
        candidate_ring = _mode(candidate_inputs, original_ringdown=original_ringdown,
                               chart_index=chart_index)
        noop_mode = np.asarray(candidate_mode).copy()
        noop_mode[whole_index] = original_ring
        no_op_summary = _series_summary((whole_time, noop_mode))
        if no_op_summary != prior["levels"][name]["torch"]:
            raise ValueError(f"{name} no-op whole mode does not replay frozen v2")
        original_inputs = prior_level["inputs"]
        captured_inputs = {
            key: [float(value.real), float(value.imag)] if key == "sigma"
            else float(value) for key, value in inputs.items() if key != "elapsed"}
        if captured_inputs != original_inputs:
            raise ValueError(f"{name} captured ringdown inputs differ from frozen v2")
        if (controls["boundary_index"] != int(stage["boundary_index"])
                or float(stage["boundary_time_M"]) != float(prior_level["boundary_time_M"])):
            raise ValueError(f"{name} boundary changed")
        record = {
            "settings": settings,
            "candidate_torch": _series_summary(series),
            "noop_torch": no_op_summary,
            "match_time_M": float(stage["match_time_M"]),
            "boundary_time_M": float(stage["boundary_time_M"]),
            "restart_index": int(stage["restart_index"]),
            "ringdown_indices": whole_index.tolist(),
            "ringdown_time_M": np.asarray(whole_time)[whole_index].tolist(),
            "fine_query_indices": chart_index.tolist(),
            "ringdown_elapsed_M": np.asarray(inputs["elapsed"]).reshape(-1)[chart_index].tolist(),
            "captured_inputs": captured_inputs,
            "candidate_inputs": {**captured_inputs, "frequency": controls["continuous_frequency"]},
            "controls": controls,
            "candidate_ringdown": _pairs(np.asarray(candidate_mode)[whole_index]),
            "reconstructed_candidate_ringdown": _pairs(candidate_ring),
            "reconstructed_noop_ringdown": _pairs(original_ring),
            "candidate_ringdown_reconstruction": _metrics(
                np.asarray(candidate_mode)[whole_index], candidate_ring),
            "torch_public_complex_relative_l2": diagnostic["torch_public"]["amplitude_peak_anchored"]["metrics"]["complex_relative_l2"],
        }
        result["levels"][name] = record
        runs[name] = {"candidate": series, "noop": (whole_time, noop_mode),
                      "controls": controls, "record": record}
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
        print(name, no_op_summary["mode_sha256"], record["candidate_torch"]["mode_sha256"],
              controls["continuous_frequency"], flush=True)

    baseline, refined = (runs[name] for name in plan["levels"])
    candidate = _compare(baseline["candidate"], refined["candidate"])
    noop = _compare(baseline["noop"], refined["noop"])
    if noop != prior["comparisons"]["candidate_torch_baseline_to_refined"]:
        raise ValueError("no-op whole-mode comparison changed")
    bt, bm = baseline["candidate"]
    rt, rm = refined["candidate"]
    if not np.array_equal(bt, rt):
        raise ValueError("candidate output charts differ")
    boundary = baseline["record"]["boundary_time_M"]
    ring_mask = np.asarray(bt) >= boundary
    ringdown = _compare((bt[ring_mask], bm[ring_mask]),
                        (rt[ring_mask], rm[ring_mask]))
    original_ring = _compare((bt[ring_mask], baseline["noop"][1][ring_mask]),
                             (rt[ring_mask], refined["noop"][1][ring_mask]))
    result["comparisons"] = {
        "candidate_baseline_to_refined_whole": candidate,
        "noop_baseline_to_refined_whole": noop,
        "candidate_baseline_to_refined_ringdown": ringdown,
        "noop_baseline_to_refined_ringdown": original_ring,
        "frequency_drift_original": (refined["controls"]["original_frequency"]
                                      -baseline["controls"]["original_frequency"]),
        "frequency_drift_continuous": (refined["controls"]["continuous_frequency"]
                                        -baseline["controls"]["continuous_frequency"]),
        "public_complex_relative_l2": {
            name: runs[name]["record"]["torch_public_complex_relative_l2"]
            for name in plan["levels"]},
    }
    limits = plan["checks"]
    result["checks"] = {
        "whole_mode_drift_at_most": candidate["complex_relative_l2"]
            <= limits["whole_mode_drift_at_most"],
        "noop_whole_replay_exact": all(
            runs[name]["record"]["noop_torch"] == prior["levels"][name]["torch"]
            for name in plan["levels"]),
        "candidate_ringdown_reconstruction_max_absolute_at_most": all(
            runs[name]["record"]["candidate_ringdown_reconstruction"]["max_absolute"]
            <= limits["candidate_ringdown_reconstruction_max_absolute_at_most"]
            for name in plan["levels"]),
        "direct_raw_phase_derivative_at_most": all(
            abs(runs[name]["controls"]["direct_raw_phase_first"]
                -runs[name]["controls"]["raw_frequency"])
            <= limits["direct_raw_phase_derivative_at_most"]
            for name in plan["levels"]),
        "raw_frequency_sample_replay_at_most": all(
            abs(runs[name]["controls"]["sampled_raw_frequency"]
                -runs[name]["controls"]["raw_frequency"])
            <= limits["raw_frequency_sample_replay_at_most"]
            for name in plan["levels"]),
        "phase_basis_sample_replay_at_most": all(
            np.max(np.abs(np.asarray(runs[name]["controls"]["sampled_phase_basis"])
                          -np.asarray(runs[name]["controls"]["ode_phase_basis"])))
            <= limits["phase_basis_sample_replay_at_most"]
            for name in plan["levels"]),
        "backward_secant_replay_at_most": all(
            abs(runs[name]["controls"]["backward_secant"]
                -runs[name]["controls"]["original_frequency"])
            <= limits["backward_secant_replay_at_most"]
            for name in plan["levels"]),
        "continuous_frequency_positive": all(
            runs[name]["controls"]["continuous_frequency"] > 0
            for name in plan["levels"]),
        "boundary_and_support_unchanged": all(
            runs[name]["record"]["captured_inputs"]
            == capture_prior["levels"][name]["inputs"]
            and runs[name]["record"]["boundary_time_M"]
            == capture_prior["levels"][name]["boundary_time_M"]
            for name in plan["levels"]),
        "public_comparison_finite": all(
            np.isfinite(runs[name]["record"]["torch_public_complex_relative_l2"])
            for name in plan["levels"]),
    }
    result["prospective_checks_passed"] = all(result["checks"].values())
    result["status"] = "diagnostic_complete"
    result["elapsed_s"] = time.monotonic()-started
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"comparisons": result["comparisons"],
                      "checks": result["checks"]}), flush=True)


if __name__ == "__main__":
    main()

"""Development-only attribution of the frozen v2 ringdown residual by input swaps."""

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
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _series_summary
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _source_field
from dev.native_rom.diagnose_source_aligned_ode_chain_residual_attribution_v2 import _run_level
from torchwave.approximants.eob import numerical_attachment as attachment
from torchwave.approximants.eob.merger_ringdown import MergerRingdownShape


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_ode_chain_ringdown_inputs_plan.json"
OUTPUT = EVIDENCE / "source_aligned_ode_chain_ringdown_inputs.json"
PARAMETERS = (
    "amplitude", "amplitude_derivative", "frequency", "phase", "sigma",
    "shape_amplitude_rate", "shape_amplitude_offset", "shape_phase_rate",
    "shape_phase_offset", "elapsed", "support_end",
)


def _checked_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-ode-chain-ringdown-inputs-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "strong_aligned"
            or plan["levels"] != ["baseline", "refined"]
            or plan["parameters"] != list(PARAMETERS)
            or plan["runner_sentinel"] != {"half_width_M": 0.0, "degree": 0}
            or plan["fine_refinement_factor"] != 32):
        raise ValueError("frozen ringdown-input plan changed")
    for key, frozen in plan["inputs"].items():
        if q.sha256(ROOT / frozen["path"]) != frozen["sha256"]:
            raise ValueError(f"frozen {key} SHA256 changed")
    if q.sha256(Path(__file__)) != plan["script_sha256"]:
        raise ValueError("frozen ringdown-input runner SHA256 changed")
    return plan


def _array(value):
    return np.asarray(value.detach().cpu().numpy()).copy()


def _scalar(value):
    data = _array(value)
    if data.size != 1:
        raise ValueError(f"expected scalar ringdown input, got {data.shape}")
    return data.reshape(()).item()


def _capture_level(name, settings, source_plan, identity, plan, field,
                   hamiltonian, flux, original_full_series, original_eval):
    captured, support = [], []
    original_ringdown = attachment.merger_ringdown_22
    original_support = attachment.seobnrv4_ringdown_support_samples

    def ringdown_hook(elapsed, sigma, amplitude, amplitude_derivative,
                      frequency, phase, shape):
        captured.append({
            "elapsed": _array(elapsed),
            "sigma": _scalar(sigma),
            "amplitude": _scalar(amplitude),
            "amplitude_derivative": _scalar(amplitude_derivative),
            "frequency": _scalar(frequency),
            "phase": _scalar(phase),
            "shape_amplitude_rate": _scalar(shape.amplitude_rate),
            "shape_amplitude_offset": _scalar(shape.amplitude_offset),
            "shape_phase_rate": _scalar(shape.phase_rate),
            "shape_phase_offset": _scalar(shape.phase_offset),
        })
        return original_ringdown(elapsed, sigma, amplitude,
                                 amplitude_derivative, frequency, phase, shape)

    def support_hook(damping, dt_high):
        count = original_support(damping, dt_high)
        support.append({"count": int(_scalar(count)),
                        "dt_high": float(_scalar(dt_high))})
        return count

    with (patch.object(attachment, "merger_ringdown_22", ringdown_hook),
          patch.object(attachment, "seobnrv4_ringdown_support_samples", support_hook)):
        diagnostic, series, stage = _run_level(
            name, settings, source_plan, identity, plan["case"],
            plan["runner_sentinel"], field, hamiltonian, flux,
            original_full_series, original_eval)
    if len(captured) != 1 or len(support) != 1:
        raise ValueError("expected one ringdown and support call")
    inputs = captured[0]
    inputs["support_end"] = support[0]["count"] * support[0]["dt_high"]
    inputs["support_count"] = support[0]["count"]
    inputs["dt_high"] = support[0]["dt_high"]
    return diagnostic, series, stage, inputs


def _mode(inputs, *, original_ringdown, chart_index):
    def real(key):
        return torch.as_tensor(inputs[key], dtype=torch.float64)

    shape = MergerRingdownShape(*(
        real(key) for key in PARAMETERS[5:9]))
    elapsed = torch.as_tensor(np.asarray(inputs["elapsed"]).reshape(-1)[chart_index],
                              dtype=torch.float64)
    sigma = torch.as_tensor(inputs["sigma"], dtype=torch.complex128)
    mode = original_ringdown(
        elapsed, sigma, real("amplitude"), real("amplitude_derivative"),
        real("frequency"), real("phase"), shape)
    mode = torch.where(elapsed < real("support_end"), mode,
                       torch.zeros_like(mode))
    return _array(mode).reshape(-1)


def _pairs(values):
    array = np.asarray(values).reshape(-1)
    return [[float(z.real), float(z.imag)] for z in array]


def _norm(values):
    return float(np.linalg.norm(values))


def _metrics(reference, candidate):
    difference = candidate-reference
    return {"l2": _norm(difference),
            "relative_l2": _norm(difference)/_norm(reference),
            "max_absolute": float(np.max(np.abs(difference)))}


def main() -> None:
    plan = _checked_plan()
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    prior = json.loads((ROOT / plan["inputs"]["amplitude_v2_result"]["path"]).read_text())
    attribution = json.loads((ROOT / plan["inputs"]["attribution_v2_result"]["path"]).read_text())
    identity = driver._prepare_public(source_plan)
    field = _source_field()
    m1, m2, chi1, chi2, _ = driver.CASES[plan["case"]]
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64)

    hamiltonian = driver.TorchSourceHam(
        tensor(m1*m2/(m1+m2)**2), tensor(chi1), tensor(chi2),
        delta=tensor((m1-m2)/(m1+m2)))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    original_ringdown = attachment.merger_ringdown_22
    original_full_series = attachment._full_series_nqc
    original_eval = attachment._eval_batch_scalar
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.source-aligned-ode-chain-ringdown-inputs.v1",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
        "levels": {}, "comparison": {}, "checks": {},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    runs = {}
    for name in plan["levels"]:
        level = next(item for item in ladder["levels"] if item["name"] == name)
        settings = {"coarse_output_step_M": ladder["coarse_output_step_M"],
                    "max_step_M": ladder["max_step_M"],
                    "atol": level["atol"], "rtol": level["rtol"]}
        _, series, stage, inputs = _capture_level(
            name, settings, source_plan, identity, plan, field, hamiltonian, flux,
            original_full_series, original_eval)
        summary = _series_summary(series)
        if summary != prior["levels"][name]["torch"]:
            raise ValueError(f"{name} v2 whole-series replay changed")
        restart = int(stage["restart_index"])
        whole_time, whole_mode = series
        ring_mask = np.asarray(whole_time) >= float(stage["boundary_time_M"])
        whole_index = np.flatnonzero(ring_mask)
        if not np.all(whole_index >= restart):
            raise ValueError("ringdown enters coarse prefix")
        chart_index = plan["fine_refinement_factor"] * (whole_index-restart)
        elapsed = np.asarray(inputs["elapsed"]).reshape(-1)
        if chart_index[-1] >= len(elapsed):
            raise ValueError("ringdown chart shorter than whole series")
        reconstruction = _mode(inputs, original_ringdown=original_ringdown,
                               chart_index=chart_index)
        no_op = _mode(dict(inputs), original_ringdown=original_ringdown,
                      chart_index=chart_index)
        emitted = np.asarray(whole_mode)[whole_index]
        record = {
            "settings": settings, "torch": summary,
            "match_time_M": float(stage["match_time_M"]),
            "boundary_time_M": float(stage["boundary_time_M"]),
            "boundary_index": int(stage["boundary_index"]),
            "restart_index": restart,
            "whole_ringdown_indices": whole_index.tolist(),
            "whole_ringdown_time_M": np.asarray(whole_time)[whole_index].tolist(),
            "fine_query_index": chart_index.tolist(),
            "inputs": {key: (float(value.real), float(value.imag)) if key == "sigma"
                       else float(value) for key, value in inputs.items()
                       if key != "elapsed"},
            "elapsed_M": elapsed[chart_index].tolist(),
            "emitted_mode": _pairs(emitted),
            "reconstructed_mode": _pairs(reconstruction),
            "reconstruction": _metrics(emitted, reconstruction),
            "no_op_max_absolute": float(np.max(np.abs(no_op-reconstruction))),
        }
        result["levels"][name] = record
        runs[name] = (inputs, chart_index, reconstruction, emitted)
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(name, summary["mode_sha256"], record["reconstruction"], flush=True)

    bi, bx, bmode, bemitted = runs["baseline"]
    ri, rx, rmode, remitted = runs["refined"]
    if not np.array_equal(
            result["levels"]["baseline"]["whole_ringdown_time_M"],
            result["levels"]["refined"]["whole_ringdown_time_M"]):
        raise ValueError("ringdown whole charts differ")
    if len(bmode) != len(rmode):
        raise ValueError("ringdown segments differ in length")
    total = rmode-bmode
    total_l2 = _norm(total)
    if not total_l2:
        raise ValueError("zero ringdown residual")
    swaps = {}
    deltas = []
    for key in PARAMETERS:
        forward_inputs = dict(bi)
        forward_inputs[key] = ri[key]
        # Elapsed is a sample array, so it must be indexed on the refined chart.
        forward_chart = rx if key == "elapsed" else bx
        forward = _mode(forward_inputs, original_ringdown=original_ringdown,
                        chart_index=forward_chart)
        reverse_inputs = dict(ri)
        reverse_inputs[key] = bi[key]
        reverse_chart = bx if key == "elapsed" else rx
        reverse = _mode(reverse_inputs, original_ringdown=original_ringdown,
                        chart_index=reverse_chart)
        delta = forward-bmode
        deltas.append(delta)
        swaps[key] = {
            "baseline_to_refined_input": _pairs(forward),
            "refined_to_baseline_input": _pairs(reverse),
            "forward_effect_over_total_l2": _norm(delta)/total_l2,
            "forward_error_remaining_over_total_l2": _norm(forward-rmode)/total_l2,
            "forward_error_square_removed_fraction": 1-(_norm(forward-rmode)/total_l2)**2,
            "reverse_effect_over_total_l2": _norm(rmode-reverse)/total_l2,
            "reverse_error_remaining_over_total_l2": _norm(reverse-bmode)/total_l2,
            "input_differs": bool(not np.array_equal(np.asarray(bi[key]), np.asarray(ri[key]))),
        }
    sum_single = np.sum(np.stack(deltas), axis=0)
    interaction = total-sum_single
    time_grid = np.asarray(result["levels"]["baseline"]["whole_ringdown_time_M"])
    original_squared = np.abs(remitted-bemitted)**2
    result["comparison"] = {
        "ringdown_time_M": time_grid.tolist(),
        "emitted_ringdown_error_squared": float(original_squared.sum()),
        "reconstructed_ringdown_error_squared": float(np.vdot(total, total).real),
        "reconstructed_ringdown_relative_l2": _norm(total)/_norm(rmode),
        "prior_ringdown_error_squared": attribution["comparison"]["windows"]["ringdown"]["error_squared"],
        "baseline_refined_l2": total_l2,
        "input_deltas": {key: (float(ri[key].real-bi[key].real),
                               float(ri[key].imag-bi[key].imag)) if key == "sigma"
                          else (float(np.max(np.abs(np.asarray(ri[key])-np.asarray(bi[key]))))
                                if key == "elapsed" else float(ri[key]-bi[key]))
                          for key in PARAMETERS},
        "swaps": swaps,
        "sum_single_effect_over_total_l2": _norm(sum_single)/total_l2,
        "interaction_residual_over_total_l2": _norm(interaction)/total_l2,
        "interaction_residual": _pairs(interaction),
        "total_difference": _pairs(total),
    }
    limits = plan["checks"]
    result["checks"] = {
        "level_reconstruction_max_absolute_at_most": all(
            result["levels"][name]["reconstruction"]["max_absolute"]
            <= limits["level_reconstruction_max_absolute_at_most"] for name in plan["levels"]),
        "no_op_max_absolute_at_most": all(
            result["levels"][name]["no_op_max_absolute"]
            <= limits["no_op_max_absolute_at_most"] for name in plan["levels"]),
        "prior_ringdown_error_square_replay_at_most": abs(
            float(original_squared.sum())
            - result["comparison"]["prior_ringdown_error_squared"])
            <= limits["prior_ringdown_error_square_replay_at_most"],
        "reconstructed_ringdown_error_square_replay_at_most": abs(
            float(original_squared.sum())
            - result["comparison"]["reconstructed_ringdown_error_squared"])
            <= limits["reconstructed_ringdown_error_square_replay_at_most"],
    }
    result["prospective_checks_passed"] = all(result["checks"].values())
    result["status"] = "diagnostic_complete"
    result["elapsed_s"] = time.monotonic()-started
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"inputs": result["comparison"]["input_deltas"],
                      "swaps": {key: {field: value for field, value in row.items()
                                       if not field.endswith("_input")}
                                for key, row in swaps.items()},
                      "checks": result["checks"]}), flush=True)


if __name__ == "__main__":
    main()

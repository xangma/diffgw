"""Dev-only corrected h22 comparison on the exact source cap-time chart."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch

import jax.numpy as jnp
import lal
import numpy as np
from scipy.interpolate import CubicSpline
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _series_summary
from dev.native_rom.diagnose_source_aligned_cap_full_epoch_suffix_v2 import array_sha
from dev.native_rom.diagnose_source_aligned_ode_chain_boundary_frequency import (
    _capture_level,
)
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _source_field
from dev.native_rom.diagnose_source_aligned_public_gap import _sha_array
from dev.native_rom.eob.torch.warm_start import native_frequency_warm_start
from torchwave.approximants.eob import numerical_attachment as attachment
from torchwave.approximants.eob.nqc import NQCCoefficients, nqc_22_factor
from torchwave.constants import C_SI, MPC_SI


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_exact_cap_corrected_plan.json"
OUTPUT = EVIDENCE / "source_aligned_exact_cap_corrected.json"
ROW = (30.0, 20.0, 0.8, 0.6, 0.04)


def _pair(values: np.ndarray) -> list[list[float]]:
    return np.column_stack((values.real, values.imag)).tolist()


def _complex(values: list[list[float]]) -> np.ndarray:
    pairs = np.asarray(values, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("expected [real, imaginary] samples")
    return pairs[:, 0] + 1j*pairs[:, 1]


def _metrics(candidate: np.ndarray, reference: np.ndarray) -> dict:
    if len(candidate) == 0 or len(candidate) != len(reference):
        raise ValueError("invalid comparison lengths")
    if np.min(abs(reference)) <= 0:
        raise ValueError("zero reference amplitude")
    phase = np.unwrap(np.angle(candidate/reference))
    amplitude = abs(candidate)-abs(reference)
    return {
        "samples": len(candidate),
        "complex_relative_l2": float(np.linalg.norm(candidate-reference)
                                     / np.linalg.norm(reference)),
        "amplitude_relative_l2": float(np.linalg.norm(amplitude)
                                       / np.linalg.norm(abs(reference))),
        "phase_first_last_rad": [float(phase[0]), float(phase[-1])],
        "phase_min_max_rad": [float(np.min(phase)), float(np.max(phase))],
        "phase_rms_rad": float(np.sqrt(np.mean(phase**2))),
        "phase_max_abs_rad": float(np.max(abs(phase))),
    }


def _checked_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-exact-cap-corrected-plan.v1"
            or plan["release_qualified"] is not False
            or tuple(plan["row_tuple"]) != ROW
            or plan["step_M"] != 0.5
            or plan["level"] != "refined"
            or plan["script_sha256"] != q.sha256(Path(__file__))):
        raise ValueError("frozen exact-cap corrected plan changed")
    for key, item in plan["inputs"].items():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen input changed: {key}")
    return plan


def _run(plan: dict, result: dict) -> None:
    source = json.loads((ROOT / plan["inputs"]["source_epoch_result"]["path"]).read_text())
    prior_raw = json.loads((ROOT / plan["inputs"]["cap_raw_result"]["path"]).read_text())
    v5 = json.loads((ROOT / plan["inputs"]["cap_full_v5_result"]["path"]).read_text())
    v5_plan = json.loads((ROOT / plan["inputs"]["cap_full_v5_plan"]["path"]).read_text())
    boundary_plan = json.loads((ROOT / v5_plan["inputs"]["boundary_plan"]["path"]).read_text())
    source_plan = json.loads((ROOT / boundary_plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / v5_plan["inputs"]["ladder_plan"]["path"]).read_text())
    for nested_plan in (v5_plan, boundary_plan):
        for key, item in nested_plan["inputs"].items():
            if q.sha256(ROOT / item["path"]) != item["sha256"]:
                raise ValueError(f"nested frozen input changed: {key}")
    if (source["status"] != "diagnostic_complete" or not source["validity_passed"]
            or prior_raw["status"] != "diagnostic_complete"
            or not prior_raw["validity_passed"]
            or v5["status"] != "diagnostic_complete" or not v5["validity_passed"]
            or tuple(source["row_tuple"]) != ROW or tuple(v5["row_tuple"]) != ROW):
        raise ValueError("frozen source/native controls failed")
    source_control = source["source_epoch_control"]
    source_mode = _complex(source_control["cap_mode"])
    source_time = np.arange(len(source_mode), dtype=np.float64)*plan["step_M"]
    source_raw = _complex(prior_raw["primary_chart"]["source_raw_on_source"])
    native_raw = _complex(prior_raw["primary_chart"]["native_raw_on_native"])
    raw_time = np.asarray(prior_raw["primary_chart"]["time_M"], dtype=np.float64)
    source_state = np.asarray(prior_raw["primary_chart"]["source_state"],
                              dtype=np.float64)
    source_omega = np.asarray(prior_raw["primary_chart"]["source_omega_M"],
                              dtype=np.float64)
    coeff = np.asarray(source["nqc_control"]["cap_coefficients"], dtype=np.float64)
    mass_seconds = (ROW[0]+ROW[1])*float(lal.MTSUN_SI)
    physical_scale = mass_seconds*float(C_SI)/(100*float(MPC_SI))

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    torch.set_num_threads(1)
    hamiltonian = driver.TorchSourceHam(
        tensor(ROW[0]*ROW[1]/(ROW[0]+ROW[1])**2), tensor(ROW[2]),
        tensor(ROW[3]), delta=tensor((ROW[0]-ROW[1])/(ROW[0]+ROW[1])))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    field = _source_field()
    warm = native_frequency_warm_start(
        hamiltonian, flux, tensor(v5_plan["cap_Momega"]), tensor(ROW[4]),
        **v5_plan["warm_start_settings"])
    cap_state = warm.initial_state.detach()
    cap_state_np = cap_state.cpu().numpy()

    def torch_initial(ham, flux_instance, frequency):
        if float(frequency) != ROW[4]:
            raise ValueError("unexpected Torch initialization frequency")
        return cap_state[0], cap_state[2], cap_state[3]

    def jax_initial(ham, flux_instance, frequency):
        if float(frequency) != ROW[4]:
            raise ValueError("unexpected JAX initialization frequency")
        return tuple(jnp.asarray(cap_state_np[i]) for i in (0, 2, 3))

    level = next(item for item in ladder["levels"] if item["name"] == plan["level"])
    settings = {
        "coarse_output_step_M": ladder["coarse_output_step_M"],
        "max_step_M": ladder["max_step_M"], "atol": level["atol"],
        "rtol": level["rtol"],
    }
    identity = driver._prepare_public(source_plan)
    with (patch.object(driver, "find_quasicircular_initial_conditions", torch_initial),
          patch.object(driver, "jax_initial_conditions", jax_initial)):
        diagnostic, series, stage, inputs, controls = _capture_level(
            plan["level"], settings, source_plan, identity, boundary_plan,
            field, hamiltonian, flux,
            attachment._full_series_nqc, attachment._eval_batch_scalar)
    native_time = np.asarray(series[0], dtype=np.float64)
    native_mode = np.asarray(series[1], dtype=np.complex128)*physical_scale
    native_summary = _series_summary(series)
    expected = v5["levels"][plan["level"]]
    common_count = min(len(source_mode), len(native_mode))
    common_time = native_time[:common_count]
    source_exact = source_mode[:common_count]
    native_common = native_mode[:common_count]

    shift = float(source_control["exact_minus_legacy_first_time_M"])
    spline_re = CubicSpline(source_time, source_mode.real, bc_type="natural")
    spline_im = CubicSpline(source_time, source_mode.imag, bc_type="natural")
    source_old_map = spline_re(common_time+shift)+1j*spline_im(common_time+shift)
    raw_indices = np.rint(raw_time/plan["step_M"]).astype(int)
    source_early = source_exact[raw_indices]
    native_early = native_common[raw_indices]
    old_early = source_old_map[raw_indices]
    source_nqc = NQCCoefficients(tensor(coeff[:3]), tensor(coeff[6:8]))
    factor = nqc_22_factor(
        tensor(source_state[:, 0]), tensor(source_state[:, 2]),
        tensor(source_omega), source_nqc, sample_dims=1)
    source_reconstructed = (source_raw*factor.detach().cpu().numpy()
                            * physical_scale)

    match = float(stage["match_time_M"])
    boundary = float(stage["boundary_time_M"])
    windows = {
        "whole_common_cap_support": np.ones(common_count, dtype=bool),
        "pre_native_match_minus_50M": common_time < match-50,
        "native_nqc_local_50M": ((common_time >= match-50)
                                 & (common_time < boundary)),
        "post_native_boundary": common_time >= boundary,
    }
    metrics = {
        name: {
            "native_vs_source_exact": _metrics(native_common[mask], source_exact[mask]),
            "native_vs_source_old_map": _metrics(native_common[mask],
                                                  source_old_map[mask]),
            "source_old_map_vs_exact": _metrics(source_old_map[mask],
                                                source_exact[mask]),
        }
        for name, mask in windows.items() if mask.any()
    }
    metrics["primary_869_exact_cap_time"] = {
        "native_vs_source_exact": _metrics(native_early, source_early),
        "native_vs_source_old_map": _metrics(native_early, old_early),
        "source_old_map_vs_exact": _metrics(old_early, source_early),
        "source_raw_nqc_reconstruction_vs_full": _metrics(
            source_reconstructed, source_early),
        "native_raw_vs_source_raw": _metrics(native_raw, source_raw),
    }
    checks = {
        "source_cap_mode_hash": array_sha(source_mode)
            == source_control["cap_mode_sha256"],
        "source_exact_suffix": (source_control["exact_suffix_offsets"]
            == [source_control["tested_offset"]] == [805]),
        "source_raw_chart_count": len(raw_time) == plan["primary_samples"]
            == len(source_state) == len(source_omega) == len(source_raw)
            == len(native_raw),
        "source_raw_chart_time": bool(len(raw_time) == plan["primary_samples"]
            and np.array_equal(raw_time, np.arange(
                plan["primary_start_M"], plan["primary_end_M"]
                +plan["step_M"]/2, plan["step_M"]))),
        "native_cap_state_replay": bool(np.max(abs(cap_state_np
            -np.asarray(v5["cap_state"]))) <= plan["gates"]["cap_state_max_abs"]),
        "native_full_series_replay": native_summary == expected["full_series"],
        "native_stage_replay": abs(match-expected["match_time_M"])
            <= plan["gates"]["stage_time_max_abs_M"]
            and abs(boundary-expected["boundary_time_M"])
            <= plan["gates"]["stage_time_max_abs_M"],
        "native_time_exact_grid": bool(np.array_equal(native_time,
            np.arange(len(native_time), dtype=np.float64)*plan["step_M"])),
        "common_chart_count": common_count == plan["common_samples"],
        "primary_indices_valid": bool(np.all(raw_indices >= 0)
            and np.all(raw_indices < common_count)
            and np.array_equal(common_time[raw_indices], raw_time)),
        "source_shift_inside_support": bool(common_time[0]+shift >= source_time[0]
            and common_time[-1]+shift <= source_time[-1]),
        "legacy_shift_replay": abs(shift-plan["legacy_shift_M"])
            <= plan["gates"]["legacy_shift_replay_max_abs_M"],
        "mode_and_factor_finite": bool(all(np.isfinite(value).all() for value in (
            source_mode, native_mode, source_raw, native_raw, source_reconstructed,
            source_old_map))),
        "all_windows_nonempty": all(mask.any() for mask in windows.values()),
        "window_counts_replay": {name: int(mask.sum()) for name, mask
            in windows.items()} == plan["expected_window_samples"],
        "physical_scale_positive": physical_scale > 0,
    }
    result.update({
        "status": "diagnostic_complete" if all(checks.values()) else "failed_validity",
        "validity_passed": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "input_replay": {
            "source_cap_mode_sha256": array_sha(source_mode),
            "source_raw_mode_sha256": _sha_array(source_raw),
            "native_raw_mode_sha256": _sha_array(native_raw),
            "native_full_series": native_summary,
            "native_initial_state": cap_state_np.tolist(),
            "native_torch_jax_checks": diagnostic["checks"],
            "native_torch_jax_whole_l2": diagnostic["whole_mode_relative_l2"],
        },
        "chart": {
            "definition": "Cap-init time t=i*0.5 M; fixed phase-zero cap states, fixed physical scale, no time/phase fit.",
            "common_samples": common_count,
            "common_end_M": float(common_time[-1]),
            "primary_samples": len(raw_time),
            "primary_start_end_M": [float(raw_time[0]), float(raw_time[-1])],
            "old_map_delta_M": shift,
            "old_map_rule": "The legacy low-endpoint formula labeled source index i at exact time minus delta; querying it at native t samples the exact source spline at t+delta.",
            "old_map_interpolation": "Independent natural cubic on source corrected Re/Im; exact comparison uses output samples only.",
            "source_amplitude_peak_index": int(np.argmax(abs(source_mode))),
            "native_amplitude_peak_index": int(np.argmax(abs(native_mode))),
            "native_match_time_M": match,
            "native_boundary_time_M": boundary,
            "source_orbital_peak_time_M": source_control[
                "cap_peak_time_from_uncropped_epoch_M"],
            "native_orbital_peak_time_M": float(stage["orbital_peak_time_M"]),
            "window_samples": {name: int(mask.sum()) for name, mask in windows.items()},
        },
        "primary_series": {
            "time_M": raw_time.tolist(),
            "source_exact_corrected": _pair(source_early),
            "native_exact_corrected": _pair(native_early),
            "source_old_map_corrected": _pair(old_early),
            "source_raw_nqc_reconstructed": _pair(source_reconstructed),
        },
        "metrics": metrics,
        "boundary_controls": {
            "frequency_original": controls["original_frequency"],
            "frequency_continuous": controls["continuous_frequency"],
            "phase_coefficients": controls["phase_coefficients"],
            "ringdown_input_frequency": inputs["frequency"],
        },
        "limitations": [
            "The old-map shift is a deliberate chart error; the established peak-aligned waveform used independent sampled-mode anchors and did not rely on the old low-endpoint formula.",
            "Raw same-state and own-state controls were captured by an earlier refined cap trajectory, not by this full-mode run; exact full-series replay is the bridge.",
            "Native-defined NQC-local and post-boundary windows do not identify source NQC or attachment times; their residuals are descriptive, not causal component isolation.",
            "Source raw*NQC reconstruction is a separate source-formula control. A failure remains a finding and does not alter corrected-mode chart metrics.",
        ],
    })


def main() -> None:
    plan = _checked_plan()
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.source-aligned-exact-cap-corrected.v1",
        "status": "running", "release_qualified": False,
        "row_tuple": list(ROW), "host": platform.node(),
        "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    try:
        _run(plan, result)
    except Exception as exc:
        result.update({"status": "runner_error", "error": repr(exc)})
        raise
    finally:
        result["elapsed_s"] = time.monotonic()-started
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"], result.get("checks"), flush=True)


if __name__ == "__main__":
    main()

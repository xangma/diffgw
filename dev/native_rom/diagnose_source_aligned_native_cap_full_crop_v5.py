"""Dev-only v4 cap full-crop replay with audited initialization gates."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import torch
from scipy.interpolate import CubicSpline

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import diagnose_source_aligned_three_way_waveform as three_way
from dev.native_rom import qualify_generator as q
from dev.native_rom.eob.torch.warm_start import native_frequency_warm_start
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _series_summary
from dev.native_rom.diagnose_peak_anchor import public_mode
from dev.native_rom.diagnose_source_aligned_ode_chain_boundary_frequency import _capture_level
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _source_field
from dev.native_rom.diagnose_source_aligned_public_gap import _sha_array
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_wheel_mode import _public, _source
from torchwave.approximants.eob import numerical_attachment as attachment
from torchwave.constants import C_SI, MPC_SI


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_native_cap_full_crop_v5_plan.json"
OUTPUT = EVIDENCE / "source_aligned_native_cap_full_crop_v5.json"
ROW = (30.0, 20.0, 0.8, 0.6, 0.04)


def checked_plan():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-native-cap-full-crop-plan.v5"
            or plan["release_qualified"] is not False
            or tuple(plan["row_tuple"]) != ROW
            or plan["levels"] != ["baseline", "refined", "finest"]
            or plan["cap_Momega"] != 10**(-1.5)
            or plan["step_M"] != 0.5
            or plan["fine_refinement_factor"] != 32
            or plan["script_sha256"] != q.sha256(Path(__file__))):
        raise ValueError("frozen whole-waveform warm-start plan changed")
    for name, item in plan["inputs"].items():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen input changed: {name}")
    return plan


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def final_h22_crop(time_M, mode, requested_Momega):
    """First interior output index passing the pinned full-h22 frequency rule."""
    times = np.asarray(time_M, dtype=np.float64)
    values = np.asarray(mode, dtype=np.complex128)
    if (len(times) < 4 or np.any(np.diff(times) <= 0)
            or not np.isfinite(values).all()):
        raise ValueError("invalid final h22 crop inputs")
    dre = CubicSpline(times, values.real, bc_type="natural")(times, 1)
    dim = CubicSpline(times, values.imag, bc_type="natural")(times, 1)
    magnitude_squared = values.real**2 + values.imag**2
    if np.any(magnitude_squared <= 0):
        raise ValueError("zero final h22 amplitude at crop scan")
    orbital_equivalent = (dre*values.imag-dim*values.real)/(2*magnitude_squared)
    interior = np.flatnonzero(orbital_equivalent[1:-1] > requested_Momega)+1
    if len(interior) == 0:
        raise ValueError("final h22 did not pass requested frequency")
    index = int(interior[0])
    return {
        "index": index, "time_M": float(times[index]),
        "first_Momega": float(orbital_equivalent[index]),
        "previous_Momega": float(orbital_equivalent[index-1]),
        "initial_Momega": float(orbital_equivalent[0]),
        "next_Momega": float(orbital_equivalent[index+1]),
        "frequency_finite": bool(np.isfinite(orbital_equivalent).all()),
        "strict_first_interior_crossing": bool(
            np.all(orbital_equivalent[1:index] <= requested_Momega)
            and orbital_equivalent[index] > requested_Momega),
    }


def main():
    plan = checked_plan()
    v4_result = json.loads((ROOT / plan["inputs"]["v4_result"]["path"]).read_text())
    audit = json.loads((ROOT / plan["inputs"]["cap_audit_result"]["path"]).read_text())
    prior = json.loads((ROOT / plan["inputs"]["three_way_result"]["path"]).read_text())
    v5 = json.loads((ROOT / plan["inputs"]["early_warm_result"]["path"]).read_text())
    public_gap = json.loads((ROOT / plan["inputs"]["public_gap_result"]["path"]).read_text())
    overlap_v3 = json.loads((ROOT / plan["inputs"]["overlap_v3_result"]["path"]).read_text())
    three_plan = json.loads((ROOT / plan["inputs"]["three_way_plan"]["path"]).read_text())
    boundary_plan = json.loads((ROOT / plan["inputs"]["boundary_plan"]["path"]).read_text())
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    native_source_plan = json.loads((ROOT / boundary_plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / plan["inputs"]["ladder_plan"]["path"]).read_text())
    if (v4_result["status"] != "failed_validity"
            or sorted(name for name, passed in v4_result["validity_checks"].items()
                      if not passed) != ["cap_frequency"]
            or audit["status"] != "audit_complete"
            or prior["status"] != "diagnostic_complete"
            or not prior["diagnostic_checks_passed"]
            or v5["status"] != "diagnostic_complete"
            or not v5["validity_passed"]
            or prior["fixed_chart"]["samples"] != 1169
            or prior["fixed_chart"]["time_sha256"] != plan["chart_time_sha256"]
            or overlap_v3["status"] != "diagnostic_overlap_only"
            or overlap_v3["hypothesis_checks"]["full_fixed_chart_covered"]
            or three_plan["source_identity"] != prior["source_identity"]
            or three_plan["wheel_library_sha256"] != prior["wheel_library_sha256"]):
        raise ValueError("frozen three-way or early warm controls changed")
    libraries = _libraries(source_plan["pinned_wheel"])
    identity = driver._prepare_public(native_source_plan)
    m1, m2, chi1, chi2, requested = ROW

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    hamiltonian = driver.TorchSourceHam(
        tensor(m1*m2/(m1+m2)**2), tensor(chi1), tensor(chi2),
        delta=tensor((m1-m2)/(m1+m2)))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    field = _source_field()
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.source-aligned-native-cap-full-crop.v5",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
        "row_tuple": list(ROW), "levels": {},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    torch.set_num_threads(1)
    warm = native_frequency_warm_start(
        hamiltonian, flux, tensor(plan["cap_Momega"]), tensor(requested),
        **plan["warm_start_settings"])
    warm_state = warm.state.detach()
    cap_state = warm.initial_state.detach()
    cap_state_np = array(cap_state)
    v5_crop = np.asarray(v5["warm_crop_state"], dtype=np.float64)
    v5_crop[1] = 0.
    result["cap_state"] = cap_state_np.tolist()
    result["cap_orbital_Momega"] = float(field(cap_state)[1])
    circular_state = cap_state_np.copy()
    circular_state[2] = 0.
    circular_omega = float(field(torch.as_tensor(circular_state))[1])
    result["cap_frequency_controls"] = {
        "nominal_circular_target_Momega": plan["cap_Momega"],
        "zero_radial_momentum_circular_Momega": circular_omega,
        "dissipative_cap_Momega": result["cap_orbital_Momega"],
        "frozen_v5_native_dissipative_cap_Momega":
            v5["native_warm_initial_omega_M"],
        "frozen_v5_source_dissipative_cap_Momega": v5["source_initial_omega_M"],
    }

    def torch_initial(ham, flux_instance, frequency):
        if float(frequency) != requested:
            raise ValueError("unexpected native warm-start initial-condition call")
        return cap_state[0], cap_state[2], cap_state[3]

    def jax_initial(ham, flux_instance, frequency):
        if float(frequency) != requested:
            raise ValueError("unexpected JAX warm-start initial-condition call")
        return tuple(jnp.asarray(cap_state_np[i]) for i in (0, 2, 3))

    levels = {}
    for name in plan["levels"]:
        level = next(item for item in ladder["levels"] if item["name"] == name)
        settings = {
            "coarse_output_step_M": ladder["coarse_output_step_M"],
            "max_step_M": ladder["max_step_M"], "atol": level["atol"],
            "rtol": level["rtol"],
        }
        with (patch.object(driver, "find_quasicircular_initial_conditions", torch_initial),
              patch.object(driver, "jax_initial_conditions", jax_initial)):
            diagnostic, full_series, stage, inputs, controls = _capture_level(
                name, settings, native_source_plan, identity, boundary_plan,
                field, hamiltonian, flux,
                attachment._full_series_nqc, attachment._eval_batch_scalar)
        crop = final_h22_crop(full_series[0], full_series[1], requested)
        series = (np.asarray(full_series[0])[crop["index"]:],
                  np.asarray(full_series[1])[crop["index"]:])
        levels[name] = (series, stage)
        result["levels"][name] = {
            "settings": settings, "full_series": _series_summary(full_series),
            "cropped_series": _series_summary(series), "crop": crop,
            "initial_state_max_abs": diagnostic["initial_state_max_abs"],
            "torch_initial_state": diagnostic["torch_initial_state"],
            "jax_initial_state": diagnostic["jax_initial_state"],
            "torch_jax_whole_l2": diagnostic["whole_mode_relative_l2"],
            "torch_jax_coarse_prefix_l2": diagnostic["coarse_prefix_relative_l2"],
            "torch_jax_post_restart_l2": diagnostic["post_restart_relative_l2"],
            "torch_jax_checks": diagnostic["checks"],
            "match_time_M": stage["match_time_M"],
            "boundary_time_M": stage["boundary_time_M"],
            "ringdown_boundary_inputs": {
                "frequency": inputs["frequency"],
                "amplitude": inputs["amplitude"],
                "amplitude_derivative": inputs["amplitude_derivative"],
                "phase": inputs["phase"],
                "sigma": [inputs["sigma"].real, inputs["sigma"].imag],
                "support_count": inputs["support_count"],
                "dt_high": inputs["dt_high"],
                "elapsed_sha256": _sha_array(inputs["elapsed"]),
            },
            "ringdown_boundary_controls": controls,
        }
        OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(name, "cap full h22 candidate complete", flush=True)

    row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, omega_start=requested)
    first_series = levels["refined"][0]
    run = {"time": np.asarray(first_series[0]), "mode": np.asarray(first_series[1]),
           "peak_time_M": float(levels["refined"][1]["orbital_peak_time_M"]),
           "dt_geom": plan["step_M"]}
    _, _, public_time, public_mode_values, mass_seconds = public_mode(row, run)
    direct_epoch, direct_dt, direct_public, _ = _public(row, plan["step_M"])
    direct_time = direct_epoch + np.arange(len(direct_public))*direct_dt
    with tempfile.TemporaryDirectory(prefix="diffgw-v5-native-warm-whole-") as tmp:
        private = Path(tmp) / "source"
        _snapshot(three_way.SOURCE_ROOT, source_plan["pinned_source"], private,
                  three_way.HEADERS)
        library_path = Path(tmp) / "full_source_mode.dylib"
        _compile(private, three_way.HEADERS, three_way.GSL_INCLUDE, libraries,
                 ROOT / plan["inputs"]["source_adapter"]["path"], library_path,
                 source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"],
                 require_wheel_flux=False)
        library = ctypes.CDLL(str(library_path))
        source_function = library.diffgw_v4_full_source_mode
        source_function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        source_function.restype = ctypes.c_int
        source = _source(source_function, row, plan["step_M"], plan["capacity"])
        source_replay = _source(source_function, row, plan["step_M"], plan["capacity"])
    source_epoch, source_dt, source_values = source
    source_time = source_epoch + np.arange(len(source_values))*source_dt
    ptime, pmode, panchor = three_way._anchored(
        public_time, public_mode_values, mass_seconds)
    stime, smode, sanchor = three_way._anchored(
        source_time, source_values, mass_seconds)
    lo, hi = prior["fixed_chart"]["support_M"]
    chart = ptime[(ptime >= lo) & (ptime <= hi)]
    source_aligned = np.interp(chart, stime, smode.real) + 1j*np.interp(
        chart, stime, smode.imag)
    public_aligned = pmode[(ptime >= lo) & (ptime <= hi)]
    fixed_match = prior["fixed_chart"]["match_relative_to_native_sampled_peak_M"]
    fixed_boundary = prior["fixed_chart"]["boundary_relative_to_native_sampled_peak_M"]
    windows = {
        "whole_common_support": np.ones(len(chart), dtype=bool),
        "early_pre_match_minus_50M": chart < fixed_match-50,
        "nqc_local_50M": ((chart >= fixed_match-50) & (chart < fixed_boundary-1)),
        "full_pre_boundary_with_margin": chart < fixed_boundary-1,
        "boundary_margin_1M": ((chart >= fixed_boundary-1) & (chart < fixed_boundary)),
        "post_boundary": chart >= fixed_boundary,
    }
    source_public_metrics = {
        name: three_way._metrics(source_aligned, public_aligned, mask)
        for name, mask in windows.items() if mask.sum() > 0
    }
    chart_checks = {
        "public_identity_replays": identity == public_gap["identity"],
        "public_raw_hash_replays": _sha_array(public_mode_values)
            == prior["public_mode_raw_sha256"],
        "public_direct_mode_replays": _sha_array(direct_public)
            == prior["source_public_direct_mode_sha256"],
        "public_direct_grid_replays": (len(direct_time) == len(public_time)
            and np.max(abs(direct_time-public_time)) <= 1e-12),
        "source_repeat_bitwise": (source[:2] == source_replay[:2]
                                  and np.array_equal(source[2], source_replay[2])),
        "source_raw_hash_replays": _sha_array(source_values)
            == prior["source_mode_raw_sha256"],
        "source_epoch_replays": source_epoch == prior["source_mode_epoch_s"],
        "source_sample_count_replays": len(source_values) == prior["raw_samples"]["source"],
        "source_cadence": abs(source_dt/mass_seconds-plan["step_M"]) <= 1e-10,
        "chart_samples": len(chart) == prior["fixed_chart"]["samples"],
        "chart_time_hash": _sha_array(chart) == prior["fixed_chart"]["time_sha256"],
        "source_aligned_hash": _sha_array(source_aligned)
            == prior["fixed_chart"]["aligned_mode_sha256"]["source"],
        "public_aligned_hash": _sha_array(public_aligned)
            == prior["fixed_chart"]["aligned_mode_sha256"]["public"],
        "source_public_metrics_replay": all(
            abs(source_public_metrics[name]["complex_relative_l2"]
                -prior["metrics"]["source_vs_public"][name]["complex_relative_l2"])
            <= plan["gates"]["control_metric_max_abs"]
            for name in source_public_metrics),
        "window_samples_replay": all(
            int(mask.sum()) == prior["fixed_chart"]["window_samples"][name]
            for name, mask in windows.items()),
        "warm_frequency_residual": abs(float(warm.frequency_residual))
            <= plan["gates"]["warm_frequency_residual_max_abs"],
        "cap_initial_phase_zero": float(cap_state[1]) == 0.,
        "cap_state_replays_v5": np.max(abs(cap_state_np-np.asarray(
            v5["native_warm_initial_state"]))) <= plan["gates"]["cap_state_max_abs"],
        "cap_circular_target": abs(circular_omega-plan["cap_Momega"])
            <= plan["gates"]["circular_target_max_abs"],
        "cap_dissipative_v5_native": abs(result["cap_orbital_Momega"]
            -v5["native_warm_initial_omega_M"])
                <= plan["gates"]["dissipative_frequency_parity_max_abs"],
        "cap_dissipative_v5_source": abs(result["cap_orbital_Momega"]
            -v5["source_initial_omega_M"])
                <= plan["gates"]["dissipative_frequency_parity_max_abs"],
        "cap_dissipative_v4_replay": abs(result["cap_orbital_Momega"]
            -v4_result["cap_orbital_Momega"])
                <= plan["gates"]["cap_field_replay_max_abs"],
    }
    aligned_modes = {}
    native_anchors = {}
    for name in plan["levels"]:
        series, stage = levels[name]
        native_time = np.asarray(series[0])
        native_mode = np.asarray(series[1])
        mass_time = native_time*mass_seconds
        physical_mode = native_mode*mass_seconds*float(C_SI)/(100*float(MPC_SI))
        ntime, nmode, nanchor = three_way._anchored(mass_time, physical_mode,
                                                    mass_seconds)
        native_anchors[name] = (ntime, nmode, nanchor, native_time, stage)
    common_mask = np.logical_and.reduce([
        (chart >= item[0][0]) & (chart <= item[0][-1])
        for item in native_anchors.values()])
    if int(common_mask.sum()) < plan["gates"]["minimum_common_overlap_samples"]:
        raise ValueError("insufficient shared fixed-chart overlap")
    overlap_chart = chart[common_mask]
    overlap_windows = {name: mask[common_mask] for name, mask in windows.items()}
    source_overlap = source_aligned[common_mask]
    public_overlap = public_aligned[common_mask]
    for name in plan["levels"]:
        ntime, nmode, nanchor, native_time, stage = native_anchors[name]
        candidate = np.interp(overlap_chart, ntime, nmode.real) + 1j*np.interp(
            overlap_chart, ntime, nmode.imag)
        aligned_modes[name] = candidate
        candidate_metrics = {
            pair: {window: three_way._metrics(left, right, mask)
                   for window, mask in overlap_windows.items() if mask.sum() > 0}
            for pair, left, right in (
                ("source_vs_warm", source_overlap, candidate),
                ("warm_vs_source", candidate, source_overlap),
                ("warm_vs_public", candidate, public_overlap))
        }
        result["levels"][name].update(
            raw_samples=len(levels[name][0][1]),
            native_peak_anchor=nanchor,
            native_support_M=[float(ntime[0]), float(ntime[-1])],
            fixed_chart_covered=bool(ntime[0] <= chart[0] and ntime[-1] >= chart[-1]),
            aligned_mode_sha256=_sha_array(candidate),
            native_peak_time_M=float(native_time[nanchor["index"]]),
            match_relative_to_native_peak_M=float(stage["match_time_M"]
                -native_time[nanchor["index"]]),
            boundary_relative_to_native_peak_M=float(stage["boundary_time_M"]
                -native_time[nanchor["index"]]),
            metrics=candidate_metrics,
            common_overlap_mode=np.column_stack((candidate.real, candidate.imag)).tolist(),
        )
        chart_checks[f"{name}_finite"] = bool(np.isfinite(candidate).all())
        chart_checks[f"{name}_initial_replay"] = (
            np.max(abs(np.asarray(result["levels"][name]["torch_initial_state"])
                       -cap_state_np)) <= plan["gates"]["initial_state_max_abs"]
            and result["levels"][name]["initial_state_max_abs"]
                <= plan["gates"]["initial_state_max_abs"])
        chart_checks[f"{name}_crop_rule"] = (
            result["levels"][name]["crop"]["frequency_finite"]
            and result["levels"][name]["crop"]["strict_first_interior_crossing"])
    convergence = {}
    for left, right in (("baseline", "refined"), ("refined", "finest")):
        numerator = np.linalg.norm(aligned_modes[left]-aligned_modes[right])
        denominator = np.linalg.norm(aligned_modes[right])
        convergence[f"{left}_to_{right}"] = float(numerator/denominator)
    hypothesis = {
        "full_fixed_chart_covered": bool(common_mask.all()),
        "refined_to_finest_at_most": convergence["refined_to_finest"]
            <= plan["gates"]["refined_to_finest_at_most"],
        "convergence_monotone": convergence["refined_to_finest"]
            <= convergence["baseline_to_refined"],
    }
    for name in plan["levels"]:
        observed = result["levels"][name]
        frozen = v4_result["levels"][name]
        chart_checks[f"{name}_v4_series_replay"] = (
            observed["full_series"] == frozen["full_series"]
            and observed["cropped_series"] == frozen["cropped_series"]
            and observed["crop"] == frozen["crop"]
            and observed["aligned_mode_sha256"] == frozen["aligned_mode_sha256"]
            and observed["metrics"] == frozen["metrics"])
    chart_checks["v4_existing_validity_replay"] = all(
        chart_checks[name] == value for name, value in
        v4_result["validity_checks"].items() if name != "cap_frequency")
    chart_checks["v4_hypothesis_replay"] = hypothesis == v4_result["hypothesis_checks"]
    chart_checks["v4_convergence_replay"] = convergence == v4_result["convergence"]
    chart_checks = {key: bool(value) for key, value in chart_checks.items()}
    hypothesis = {key: bool(value) for key, value in hypothesis.items()}
    result.update(
        status=("diagnostic_complete" if all(chart_checks.values())
                and all(hypothesis.values()) else "diagnostic_overlap_only"
                if all(chart_checks.values()) else "failed_validity"),
        elapsed_s=time.monotonic()-started,
        warm_start={
            "cap_Momega": plan["cap_Momega"],
            "crossing_time_M": float(warm.crossing_time),
            "frequency_residual_M": float(warm.frequency_residual),
            "initial_state": cap_state_np.tolist(),
            "phase_zero_orbital_crossing_state": array(warm_state).tolist(),
            "accepted_steps": warm.accepted_steps,
            "evaluations": warm.evaluations,
            "v5_coarse_crop_state_max_abs": float(np.max(abs(array(warm_state)-v5_crop))),
        },
        fixed_chart={
            "rule": plan["chart_rule"],
            "time_sha256": _sha_array(chart),
            "samples": len(chart), "support_M": [float(chart[0]), float(chart[-1])],
            "window_samples": {name: int(mask.sum()) for name, mask in windows.items()},
            "source_anchor": sanchor, "public_anchor": panchor,
            "source_aligned_hash": _sha_array(source_aligned),
            "public_aligned_hash": _sha_array(public_aligned),
        },
        common_overlap={
            "samples": len(overlap_chart),
            "support_M": [float(overlap_chart[0]), float(overlap_chart[-1])],
            "time_sha256": _sha_array(overlap_chart),
            "fixed_chart_coverage_fraction": float(common_mask.mean()),
            "window_samples": {name: int(mask.sum()) for name, mask in overlap_windows.items()},
            "source_mode_sha256": _sha_array(source_overlap),
            "public_mode_sha256": _sha_array(public_overlap),
            "source_vs_public_metrics": {
                name: three_way._metrics(source_overlap, public_overlap, mask)
                for name, mask in overlap_windows.items() if mask.sum() > 0},
        },
        source_public_metrics=source_public_metrics,
        source_full_control={
            "already_cropped_mode_samples": len(source_values),
            "already_cropped_mode_epoch_s": source_epoch,
            "already_cropped_mode_delta_t_s": source_dt,
            "raw_mode_sha256": _sha_array(source_values),
            "crop_index_accessible_from_adapter": False,
        },
        direct_control={
            "source_vs_native_whole_l2": prior["metrics"]["source_vs_native"]
                ["whole_common_support"]["complex_relative_l2"],
            "native_vs_public_whole_l2": prior["metrics"]["native_vs_public"]
                ["whole_common_support"]["complex_relative_l2"],
        },
        convergence=convergence,
        validity_checks=chart_checks, validity_passed=all(chart_checks.values()),
        hypothesis_checks=hypothesis, hypothesis_passed=all(hypothesis.values()),
        interpretation=(
            "Development-only v4 cap-start replay with audited initialization "
            "gates; the original v4 failed_validity result remains unchanged. "
            "The final "
            "assembled h22 is cropped at the first interior output index where "
            "the natural-cubic Re/Im derivative frequency exceeds the requested "
            "value, as in the pinned source convention. The pinned adapter only "
            "returns the already-cropped source full h22, so its internal crop "
            "index cannot be independently read here; its output hash and epoch "
            "are replayed. This result does not qualify a release model."
        ),
    )
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "status": result["status"], "validity": chart_checks,
        "hypothesis": hypothesis, "convergence": convergence,
            "source_vs_warm_overlap": {name: result["levels"][name]["metrics"]
            ["source_vs_warm"]["whole_common_support"]["complex_relative_l2"]
            for name in plan["levels"]},
    }), flush=True)


if __name__ == "__main__":
    main()

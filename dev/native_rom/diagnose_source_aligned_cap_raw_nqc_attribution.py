"""Dev-only same-chart attribution of cap-start raw and NQC-corrected h22."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import diagnose_source_aligned_native_cap_full_crop_v4 as cap_v4
from dev.native_rom import diagnose_source_aligned_three_way_waveform as three_way
from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _series_summary
from dev.native_rom.diagnose_peak_anchor import public_mode
from dev.native_rom.diagnose_source_aligned_ode_chain_boundary_frequency import _capture_level
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _source_field
from dev.native_rom.diagnose_source_aligned_public_gap import _sha_array
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_wheel_mode import _public, _source
from dev.native_rom.eob.torch.warm_start import native_frequency_warm_start
from torchwave.approximants.eob import numerical_attachment as attachment
from torchwave.constants import C_SI, MPC_SI


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_cap_raw_nqc_attribution_plan.json"
OUTPUT = EVIDENCE / "source_aligned_cap_raw_nqc_attribution.json"


def checked_plan():
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.cap-raw-nqc-attribution-plan.v1"
            or plan["release_qualified"] is not False
            or plan["row_tuple"] != list(cap_v4.ROW)
            or plan["level"] != "finest"
            or plan["script_sha256"] != q.sha256(Path(__file__))):
        raise ValueError("frozen raw/NQC attribution plan changed")
    for name, item in plan["inputs"].items():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen attribution input changed: {name}")
    return plan


def as_complex(pairs):
    values = np.asarray(pairs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("expected stored complex pairs")
    return values[:, 0]+1j*values[:, 1]


def metrics(left, right, masks):
    return {name: three_way._metrics(left, right, mask)
            for name, mask in masks.items() if mask.sum() > 0}


def main() -> None:
    plan = checked_plan()
    v4 = json.loads((ROOT / plan["inputs"]["v4_result"]["path"]).read_text())
    audit = json.loads((ROOT / plan["inputs"]["cap_audit_result"]["path"]).read_text())
    v4_plan = json.loads((ROOT / plan["inputs"]["v4_plan"]["path"]).read_text())
    source_plan = json.loads((ROOT / v4_plan["inputs"]["source_plan"]["path"]).read_text())
    boundary_plan = json.loads((ROOT / v4_plan["inputs"]["boundary_plan"]["path"]).read_text())
    native_source_plan = json.loads((ROOT / boundary_plan["inputs"]["source_plan"]["path"]).read_text())
    prior = json.loads((ROOT / v4_plan["inputs"]["three_way_result"]["path"]).read_text())
    ladder = json.loads((ROOT / v4_plan["inputs"]["ladder_plan"]["path"]).read_text())
    if (v4["status"] != "failed_validity"
            or audit["status"] != "audit_complete"
            or v4["hypothesis_passed"] is not True
            or v4["fixed_chart"]["samples"] != 1169):
        raise ValueError("frozen v4/audit status changed")
    m1, m2, chi1, chi2, requested = cap_v4.ROW

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)

    hamiltonian = driver.TorchSourceHam(
        tensor(m1*m2/(m1+m2)**2), tensor(chi1), tensor(chi2),
        delta=tensor((m1-m2)/(m1+m2)))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    field = _source_field()
    torch.set_num_threads(1)
    warm = native_frequency_warm_start(
        hamiltonian, flux, tensor(v4_plan["cap_Momega"]), tensor(requested),
        **v4_plan["warm_start_settings"])
    cap_state = warm.initial_state.detach()
    cap_state_np = cap_v4.array(cap_state)

    def torch_initial(ham, flux_instance, frequency):
        if float(frequency) != requested:
            raise ValueError("unexpected Torch initial frequency")
        return cap_state[0], cap_state[2], cap_state[3]

    def jax_initial(ham, flux_instance, frequency):
        if float(frequency) != requested:
            raise ValueError("unexpected JAX initial frequency")
        return tuple(jnp.asarray(cap_state_np[i]) for i in (0, 2, 3))

    level = next(item for item in ladder["levels"] if item["name"] == plan["level"])
    settings = {"coarse_output_step_M": ladder["coarse_output_step_M"],
                "max_step_M": ladder["max_step_M"], "atol": level["atol"],
                "rtol": level["rtol"]}
    with (patch.object(driver, "find_quasicircular_initial_conditions", torch_initial),
          patch.object(driver, "jax_initial_conditions", jax_initial)):
        diagnostic, full_series, stage, _, _ = _capture_level(
            plan["level"], settings, native_source_plan,
            driver._prepare_public(native_source_plan), boundary_plan,
            field, hamiltonian, flux,
            attachment._full_series_nqc, attachment._eval_batch_scalar)
    crop = cap_v4.final_h22_crop(full_series[0], full_series[1], requested)
    cropped = (np.asarray(full_series[0])[crop["index"]:],
               np.asarray(full_series[1])[crop["index"]:])
    restart = int(stage["restart_index"])
    refinement = v4_plan["fine_refinement_factor"]
    coarse_state = torch.as_tensor(stage["coarse_state"], dtype=torch.float64)
    with torch.no_grad():
        coarse_base, _ = driver.torch_mode(
            coarse_state, hamiltonian, flux,
            **(driver.MODE_OPTIONS | {"retain_pan_cubic_spin": True}))
    raw_time = np.concatenate((np.asarray(stage["coarse_time"])[:restart],
                               np.asarray(stage["fine_time"])[::refinement]))
    raw_mode = np.concatenate((cap_v4.array(coarse_base)[:restart],
                               np.asarray(stage["fine_base_mode"])[::refinement]))
    coeffs = tuple(tensor(value) for value in stage["nqc_coefficients"])
    fine_state = torch.as_tensor(stage["fine_state"], dtype=torch.float64)
    with torch.no_grad():
        coarse_omega = field(coarse_state)[:, 1]
        coarse_factor = driver.torch_factor(
            coarse_state[:, 0], coarse_state[:, 2], coarse_omega, coeffs)
        fine_factor = driver.torch_factor(
            fine_state[:, 0], fine_state[:, 2],
            torch.as_tensor(stage["fine_orbital_frequency"]), coeffs)
    reconstructed = np.concatenate((
        cap_v4.array(coarse_base*coarse_factor)[:restart],
        np.asarray(stage["fine_base_mode"])[::refinement]
            *cap_v4.array(fine_factor)[::refinement]))
    pre_boundary = raw_time < float(stage["boundary_time_M"])
    raw_time, raw_mode = raw_time[pre_boundary], raw_mode[pre_boundary]
    reconstructed = reconstructed[pre_boundary]
    full_pre_boundary = np.asarray(full_series[0]) < float(stage["boundary_time_M"])
    corrected_time = np.asarray(full_series[0])[full_pre_boundary]
    corrected_mode = np.asarray(full_series[1])[full_pre_boundary]
    row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, omega_start=requested)
    _, _, public_time, public_values, mass_seconds = public_mode(
        row, {"time": cropped[0], "mode": cropped[1],
              "peak_time_M": float(stage["orbital_peak_time_M"]),
              "dt_geom": v4_plan["step_M"]})
    direct_epoch, direct_dt, direct_public, _ = _public(row, v4_plan["step_M"])
    direct_time = direct_epoch+np.arange(len(direct_public))*direct_dt
    libraries = _libraries(source_plan["pinned_wheel"])
    with tempfile.TemporaryDirectory(prefix="diffgw-cap-raw-nqc-") as tmp:
        private = Path(tmp)/"source"
        _snapshot(three_way.SOURCE_ROOT, source_plan["pinned_source"], private,
                  three_way.HEADERS)
        library_path = Path(tmp)/"full_source_mode.dylib"
        _compile(private, three_way.HEADERS, three_way.GSL_INCLUDE, libraries,
                 ROOT/v4_plan["inputs"]["source_adapter"]["path"], library_path,
                 source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"],
                 require_wheel_flux=False)
        library = ctypes.CDLL(str(library_path))
        source_function = library.diffgw_v4_full_source_mode
        source_function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        source_function.restype = ctypes.c_int
        source_epoch, source_dt, source_values = _source(
            source_function, row, v4_plan["step_M"], v4_plan["capacity"])
    ptime, pmode, _ = three_way._anchored(public_time, public_values, mass_seconds)
    stime, smode, _ = three_way._anchored(
        source_epoch+np.arange(len(source_values))*source_dt,
        source_values, mass_seconds)
    lo, hi = v4["fixed_chart"]["support_M"]
    chart = ptime[(ptime >= lo) & (ptime <= hi)]
    source_aligned = np.interp(chart, stime, smode.real)+1j*np.interp(
        chart, stime, smode.imag)
    public_aligned = pmode[(ptime >= lo) & (ptime <= hi)]
    physical_scale = mass_seconds*float(C_SI)/(100*float(MPC_SI))
    native_time, native_mode, native_anchor = three_way._anchored(
        cropped[0]*mass_seconds, cropped[1]*physical_scale, mass_seconds)
    corrected_aligned = np.interp(chart, native_time, native_mode.real)+1j*np.interp(
        chart, native_time, native_mode.imag)
    # Apply the corrected full-mode anchor to raw h22; no new raw peak or phase fit.
    raw_chart_time = raw_time-native_anchor["time_s"]/mass_seconds
    raw_physical = raw_mode*physical_scale*np.exp(-1j*native_anchor["phase_rad"])
    raw_support = (chart >= raw_chart_time[0]) & (chart <= raw_chart_time[-1])
    raw_aligned = (np.interp(chart[raw_support], raw_chart_time, raw_physical.real)
                   +1j*np.interp(chart[raw_support], raw_chart_time, raw_physical.imag))
    fixed_match = float(prior["fixed_chart"]["match_relative_to_native_sampled_peak_M"])
    fixed_boundary = float(prior["fixed_chart"]["boundary_relative_to_native_sampled_peak_M"])
    masks = {
        "early_pre_match_minus_50M": chart < fixed_match-50,
        "nqc_local_50M": (chart >= fixed_match-50) & (chart < fixed_boundary-1),
        "full_pre_boundary_with_margin": chart < fixed_boundary-1,
        "boundary_margin_1M": (chart >= fixed_boundary-1) & (chart < fixed_boundary),
    }
    masks = {name: mask[raw_support] for name, mask in masks.items()}
    source_pre, corrected_pre = source_aligned[raw_support], corrected_aligned[raw_support]
    change = corrected_pre-raw_aligned
    raw_residual = raw_aligned-source_pre
    nqc_effect = {}
    for name, mask in masks.items():
        if mask.sum() == 0:
            continue
        before = float(np.vdot(raw_residual[mask], raw_residual[mask]).real)
        after = float(np.vdot((raw_residual+change)[mask],
                              (raw_residual+change)[mask]).real)
        cross = float(2*np.vdot(raw_residual[mask], change[mask]).real)
        increment = float(np.vdot(change[mask], change[mask]).real)
        nqc_effect[name] = {
            "raw_error_squared": before, "corrected_error_squared": after,
            "error_squared_change": after-before,
            "cross_term": cross, "increment_squared": increment,
            "identity_residual": after-before-cross-increment,
            "identity_relative_residual": abs(after-before-cross-increment)
                /max(before, after),
        }
    checks = {
        "cap_state_v4_replay": np.max(abs(cap_state_np-np.asarray(v4["cap_state"])))
            <= plan["gates"]["cap_state_max_abs"],
        "warm_crossing_residual": abs(float(warm.frequency_residual))
            <= plan["gates"]["warm_residual_max_abs"],
        "finest_settings_replay": settings == v4["levels"]["finest"]["settings"],
        "whole_series_replay": _series_summary(full_series)
            == v4["levels"]["finest"]["full_series"],
        "crop_replay": crop == v4["levels"]["finest"]["crop"],
        "cropped_series_replay": _series_summary(cropped)
            == v4["levels"]["finest"]["cropped_series"],
        "native_peak_anchor_replay": native_anchor
            == v4["levels"]["finest"]["native_peak_anchor"],
        "native_aligned_replay": _sha_array(corrected_aligned)
            == v4["levels"]["finest"]["aligned_mode_sha256"],
        "raw_corrected_sample_chart": (len(raw_time) == len(corrected_time)
            and np.max(abs(raw_time-corrected_time))
                <= plan["gates"]["sample_time_max_abs_M"]),
        "nqc_knot_reconstruction": (len(reconstructed) == len(corrected_mode)
            and np.linalg.norm(reconstructed-corrected_mode)
                /np.linalg.norm(corrected_mode)
                    <= plan["gates"]["nqc_knot_reconstruction_relative_l2"]),
        "public_raw_replay": _sha_array(public_values)
            == prior["public_mode_raw_sha256"],
        "public_direct_replay": _sha_array(direct_public)
            == prior["source_public_direct_mode_sha256"],
        "public_direct_grid_replay": (len(direct_time) == len(public_time)
            and np.max(abs(direct_time-public_time))
                <= plan["gates"]["public_grid_max_abs_s"]),
        "source_raw_replay": (_sha_array(source_values)
            == v4["source_full_control"]["raw_mode_sha256"]
            and source_epoch == v4["source_full_control"]["already_cropped_mode_epoch_s"]
            and len(source_values)
                == v4["source_full_control"]["already_cropped_mode_samples"]),
        "fixed_chart_replay": (len(chart) == 1169 and _sha_array(chart)
            == v4["fixed_chart"]["time_sha256"]),
        "source_aligned_replay": _sha_array(source_aligned)
            == v4["fixed_chart"]["source_aligned_hash"],
        "public_aligned_replay": _sha_array(public_aligned)
            == v4["fixed_chart"]["public_aligned_hash"],
        "raw_covers_pre_boundary_windows": all(
            np.all(raw_support[chart_mask]) for chart_mask in (
                chart < fixed_match-50,
                (chart >= fixed_match-50) & (chart < fixed_boundary-1))),
        "raw_and_corrected_finite": bool(np.isfinite(raw_aligned).all()
            and np.isfinite(corrected_pre).all()),
        "nqc_error_identity": all(item["identity_relative_residual"]
            <= plan["gates"]["error_identity_relative_max"]
            for item in nqc_effect.values()),
    }
    result = {
        "schema": "diffgw.seobnrv4.cap-raw-nqc-attribution.v1",
        "status": "diagnostic_complete" if all(checks.values()) else "failed_validity",
        "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
        "row_tuple": list(cap_v4.ROW), "level": plan["level"],
        "settings": settings, "checks": {k: bool(v) for k, v in checks.items()},
        "diagnostic_checks_passed": all(checks.values()),
        "chart_sha256": _sha_array(chart), "chart_samples": len(chart),
        "raw_sample_count": len(raw_time),
        "nqc_knot_reconstruction_relative_l2": float(
            np.linalg.norm(reconstructed-corrected_mode)
            /np.linalg.norm(corrected_mode))
            if len(reconstructed) == len(corrected_mode) else None,
        "raw_sample_time_max_abs_M": float(np.max(abs(raw_time-corrected_time)))
            if len(raw_time) == len(corrected_time) else None,
        "raw_support_samples": int(raw_support.sum()),
        "window_samples": {k: int(v.sum()) for k, v in masks.items()},
        "native_peak_anchor": native_anchor,
        "raw_vs_source": metrics(raw_aligned, source_pre, masks),
        "corrected_vs_source": metrics(corrected_pre, source_pre, masks),
        "raw_vs_corrected": metrics(raw_aligned, corrected_pre, masks),
        "nqc_error_decomposition": nqc_effect,
        "v4_full_source_vs_native_l2": v4["levels"]["finest"]["metrics"]
            ["source_vs_warm"]["whole_common_support"]["complex_relative_l2"],
        "raw_aligned_mode": np.column_stack((raw_aligned.real, raw_aligned.imag)).tolist(),
        "corrected_aligned_mode": np.column_stack(
            (corrected_pre.real, corrected_pre.imag)).tolist(),
        "source_aligned_mode": np.column_stack((source_pre.real, source_pre.imag)).tolist(),
        "chart_time_M": chart[raw_support].tolist(),
        "interpretation": (
            "All raw and corrected inspiral samples use the same native trajectory, "
            "the v4 corrected-mode peak anchor, and the original fixed source/public "
            "chart. Raw h22 is only defined before attachment. Comparing each to "
            "the fully corrected pinned source isolates the native NQC increment "
            "on that trajectory; it does not separately identify source NQC, "
            "trajectory, or raw-mode equation error. No pairwise fit."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "raw_vs_source": {k: v["complex_relative_l2"] for k, v in
                                        result["raw_vs_source"].items()},
                      "corrected_vs_source": {k: v["complex_relative_l2"] for k, v in
                                              result["corrected_vs_source"].items()}},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

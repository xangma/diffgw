"""Dev-only pinned source/native/public v4 whole-mode comparison on one chart."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import torch

from dev.native_rom import diagnose_jax_spliced_whole_waveform as driver
from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_jax_source_tolerance_ladder import _series_summary
from dev.native_rom.diagnose_peak_anchor import public_mode
from dev.native_rom.diagnose_source_aligned_ode_chain_boundary_frequency import _capture_level
from dev.native_rom.diagnose_source_aligned_ode_chain_phase_matrix import _source_field
from dev.native_rom.diagnose_source_aligned_public_gap import _sha_array
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_wheel_mode import _public, _source, _source_nqc
from torchwave.approximants.eob import numerical_attachment as attachment


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_three_way_waveform_plan.json"
OUTPUT = EVIDENCE / "source_aligned_three_way_waveform.json"
ROW = (30.0, 20.0, 0.8, 0.6, 0.04)
SOURCE_ROOT = Path("/Users/xangma/repos/lalsuite-git")
HEADERS = Path("/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include")
GSL_INCLUDE = Path("/usr/local/include")


def _checked_plan() -> dict:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-aligned-three-way-waveform-plan.v1"
            or plan["release_qualified"] is not False
            or plan["case"] != "strong_aligned"
            or tuple(plan["row_tuple"]) != ROW
            or tuple(driver.CASES["strong_aligned"]) != ROW
            or plan["level"] != "refined"
            or plan["step_M"] != 0.5
            or plan["capacity"] != 50000
            or plan["pre_match_width_M"] != 50.0
            or plan["boundary_margin_M"] != 1.0
            or plan["script_sha256"] != q.sha256(Path(__file__))):
        raise ValueError("Frozen three-way waveform plan changed")
    for name, item in plan["inputs"].items():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen three-way waveform input changed: {name}")
    return plan


def _anchored(time_s, mode, mass_seconds):
    index = int(np.argmax(abs(mode)))
    anchor_time_s = float(time_s[index])
    anchor_phase = float(np.angle(mode[index]))
    return ((time_s-anchor_time_s)/mass_seconds,
            mode*np.exp(-1j*anchor_phase),
            {"index": index, "time_s": anchor_time_s, "phase_rad": anchor_phase})


def _metrics(candidate, reference, mask):
    error = candidate[mask]-reference[mask]
    error_squared = float(np.vdot(error, error).real)
    reference_squared = float(np.vdot(reference[mask], reference[mask]).real)
    return {"samples": int(mask.sum()),
            "complex_relative_l2": float(np.sqrt(error_squared/reference_squared)),
            "error_squared": error_squared,
            "reference_squared": reference_squared}


def main() -> None:
    plan = _checked_plan()
    prior = json.loads((ROOT / plan["inputs"]["prior_gap"]["path"]).read_text())
    boundary_plan = json.loads((ROOT / plan["inputs"]["boundary_plan"]["path"]).read_text())
    boundary_result = json.loads((ROOT / plan["inputs"]["boundary_result"]["path"]).read_text())
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    native_source_plan = json.loads((ROOT / boundary_plan["inputs"]["source_plan"]["path"]).read_text())
    ladder = json.loads((ROOT / boundary_plan["inputs"]["ladder_plan"]["path"]).read_text())
    capture = json.loads((ROOT / plan["inputs"]["same_row_nqc_capture"]["path"]).read_text())
    if (prior["status"] != "diagnostic_complete"
            or not prior["diagnostic_checks_passed"]
            or tuple(capture["row_tuple"]) != ROW
            or not capture["diagnostic_checks_passed"]
            or plan["source_identity"] != {
                "tag": source_plan["pinned_source"]["tag"],
                "tag_object": source_plan["pinned_source"]["tag_object"],
                "peeled_commit": source_plan["pinned_source"]["peeled_commit"],
                "aligned_source_sha256": source_plan["pinned_source"]["files_sha256"][
                    "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"],
            }
            or plan["wheel_library_sha256"]
            != source_plan["pinned_wheel"]["libraries_sha256"]
            or capture["source_peeled_commit"] != plan["source_identity"]["peeled_commit"]
            or capture["wheel_library_sha256"] != plan["wheel_library_sha256"]):
        raise ValueError("Pinned three-way prerequisites changed")
    libraries = _libraries(source_plan["pinned_wheel"])
    identity = driver._prepare_public(native_source_plan)
    level = next(item for item in ladder["levels"] if item["name"] == plan["level"])
    settings = {"coarse_output_step_M": ladder["coarse_output_step_M"],
                "max_step_M": ladder["max_step_M"], "atol": level["atol"],
                "rtol": level["rtol"]}
    if settings["coarse_output_step_M"] != plan["step_M"]:
        raise ValueError("Native and source sampling cadence differ")
    m1, m2, chi1, chi2, omega_start = ROW
    row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, omega_start=omega_start)
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)
    hamiltonian = driver.TorchSourceHam(
        tensor(m1*m2/(m1+m2)**2), tensor(chi1), tensor(chi2),
        delta=tensor((m1-m2)/(m1+m2)))
    flux = driver.TorchFlux(**driver.SOURCE_FLUX_OPTIONS)
    started = time.monotonic()
    result = {"schema": "diffgw.seobnrv4.source-aligned-three-way-waveform.v1",
              "status": "running", "release_qualified": False,
              "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
              "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
              "row": row, "row_tuple": list(ROW),
              "source_identity": plan["source_identity"],
              "wheel_library_sha256": {name: q.sha256(path) for name, path in libraries.items()},
              "checks": {}}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _, native_series, stage, _, _ = _capture_level(
        plan["level"], settings, native_source_plan, identity, boundary_plan,
        _source_field(), hamiltonian, flux,
        attachment._full_series_nqc, attachment._eval_batch_scalar)
    native_time_M, native_mode = map(np.asarray, native_series)
    run = {"time": native_time_M, "mode": native_mode,
           "peak_time_M": float(stage["orbital_peak_time_M"]),
           "dt_geom": settings["coarse_output_step_M"]}
    nt, nm, rt, rm, mass_seconds = public_mode(row, run)
    direct_epoch, direct_dt, direct_public, _ = _public(row, plan["step_M"])
    direct_time = direct_epoch + np.arange(len(direct_public))*direct_dt
    with tempfile.TemporaryDirectory(prefix="diffgw-v4-three-way-") as tmp:
        private = Path(tmp) / "source"
        _snapshot(SOURCE_ROOT, source_plan["pinned_source"], private, HEADERS)
        library_path = Path(tmp) / "full_source_mode.dylib"
        _compile(private, HEADERS, GSL_INCLUDE, libraries,
                 ROOT / plan["inputs"]["source_adapter"]["path"], library_path,
                 source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"],
                 require_wheel_flux=False)
        library = ctypes.CDLL(str(library_path))
        mode_function = library.diffgw_v4_full_source_mode
        mode_function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        mode_function.restype = ctypes.c_int
        source = _source(mode_function, row, plan["step_M"], plan["capacity"])
        source_replay = _source(mode_function, row, plan["step_M"], plan["capacity"])
        coefficient_function = library.diffgw_v4_full_source_nqc
        coefficient_function.argtypes = [ctypes.c_double]*6 + [
            ctypes.POINTER(ctypes.c_double)]
        coefficient_function.restype = ctypes.c_int
        source_coefficients = _source_nqc(coefficient_function, row, plan["step_M"])
    source_epoch, source_dt, source_mode = source
    source_time = source_epoch + np.arange(len(source_mode))*source_dt
    ntime, nmode, nanchor = _anchored(nt, nm, mass_seconds)
    stime, smode, sanchor = _anchored(source_time, source_mode, mass_seconds)
    ptime, pmode, panchor = _anchored(rt, rm, mass_seconds)
    common = ((ptime >= max(ntime[0], stime[0]))
              & (ptime <= min(ntime[-1], stime[-1])))
    chart = ptime[common]
    modes = {
        "native": np.interp(chart, ntime, nmode.real)
                  + 1j*np.interp(chart, ntime, nmode.imag),
        "source": np.interp(chart, stime, smode.real)
                  + 1j*np.interp(chart, stime, smode.imag),
        "public": pmode[common],
    }
    native_peak_time_M = float(native_time_M[nanchor["index"]])
    match = float(stage["match_time_M"])-native_peak_time_M
    boundary = float(stage["boundary_time_M"])-native_peak_time_M
    windows = {
        "whole_common_support": np.ones_like(chart, dtype=bool),
        "early_pre_match_minus_50M": chart < match-plan["pre_match_width_M"],
        "nqc_local_50M": ((chart >= match-plan["pre_match_width_M"])
                           & (chart < boundary-plan["boundary_margin_M"])),
        "full_pre_boundary_with_margin": chart < boundary-plan["boundary_margin_M"],
        "boundary_margin_1M": ((chart >= boundary-plan["boundary_margin_M"])
                               & (chart < boundary)),
        "post_boundary": chart >= boundary,
    }
    pairs = {"source_vs_native": ("source", "native"),
             "source_vs_public": ("source", "public"),
             "native_vs_public": ("native", "public")}
    metrics = {name: {window: _metrics(modes[left], modes[right], mask)
                      for window, mask in windows.items() if mask.sum() > 0}
               for name, (left, right) in pairs.items()}
    direct_public_l2 = float(np.linalg.norm(direct_public-rm)/np.linalg.norm(rm))
    checks = {
        "exact_requested_tuple": tuple(row.values()) == ROW,
        "native_candidate_replays": (_series_summary(native_series)
                                     == boundary_result["levels"][plan["level"]]["candidate_torch"]),
        "public_identity_replays": identity == prior["identity"],
        "public_mode_hash_replays": _sha_array(rm) == prior["public_mode_sha256"],
        "source_mode_repeat_bitwise": (source[:2] == source_replay[:2]
                                       and np.array_equal(source[2], source_replay[2])),
        "source_public_same_units": direct_public_l2 <= 1e-12,
        "source_public_same_grid": (len(direct_public) == len(rm)
                                    and float(np.max(abs(direct_time-rt))) <= 1e-12),
        "source_cadence": abs(source_dt/mass_seconds-plan["step_M"]) <= 1e-10,
        "all_mode_samples_finite": all(np.isfinite(mode).all() for mode in modes.values()),
        "common_chart_samples_replay": int(common.sum()) == prior["common_peak_chart_samples"],
        "native_public_whole_metric_replays": abs(
            metrics["native_vs_public"]["whole_common_support"]["complex_relative_l2"]
            -prior["whole_complex_relative_l2"]) <= 1e-12,
        "early_window_samples_replay": int(windows["early_pre_match_minus_50M"].sum()) == 869,
        "nqc_local_samples_replay": int(windows["nqc_local_50M"].sum()) == 98,
    }
    result.update({
        "status": "diagnostic_complete" if all(checks.values()) else "failed",
        "elapsed_s": time.monotonic()-started,
        "native_settings": settings,
        "source_step_M": plan["step_M"], "source_mode_epoch_s": source_epoch,
        "source_mode_delta_t_s": source_dt,
        "source_nqc_coefficients_at_step_M": source_coefficients.tolist(),
        "source_mode_raw_sha256": _sha_array(source_mode),
        "native_mode_raw_sha256": _sha_array(native_mode),
        "public_mode_raw_sha256": _sha_array(rm),
        "source_public_direct_mode_l2": direct_public_l2,
        "source_public_direct_mode_sha256": _sha_array(direct_public),
        "raw_samples": {"source": len(source_mode), "native": len(native_mode),
                        "public": len(rm)},
        "fixed_chart": {
            "rule": "Independently subtract each waveform's sampled maximum-amplitude time and phase exactly once; interpolate source and native onto the public samples within the intersection of all three supports. No pairwise fitting or amplitude rescaling.",
            "anchors": {"native": nanchor, "source": sanchor, "public": panchor},
            "samples": int(common.sum()),
            "support_M": [float(chart[0]), float(chart[-1])],
            "time_sha256": _sha_array(chart),
            "aligned_mode_sha256": {name: _sha_array(mode) for name, mode in modes.items()},
            "match_relative_to_native_sampled_peak_M": match,
            "boundary_relative_to_native_sampled_peak_M": boundary,
            "window_samples": {name: int(mask.sum()) for name, mask in windows.items()},
        },
        "pair_definition": {name: {"candidate": left, "reference": right}
                            for name, (left, right) in pairs.items()},
        "metrics": metrics,
        "checks": {name: bool(value) for name, value in checks.items()},
    })
    result["diagnostic_checks_passed"] = all(checks.values())
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "whole": {name: value["whole_common_support"]["complex_relative_l2"]
                                for name, value in metrics.items()}}), flush=True)
    if not result["diagnostic_checks_passed"]:
        raise ValueError("Three-way waveform replay gate failed")


if __name__ == "__main__":
    main()

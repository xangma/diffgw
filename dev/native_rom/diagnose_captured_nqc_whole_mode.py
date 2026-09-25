"""Dev-only fixed-source-trajectory whole-mode replay of captured NQC fits.

The four coefficient vectors were solved prospectively on captured pre-NQC
inputs by diagnose_captured_nqc_event_derivative. This runner substitutes each
vector at the pinned source generator's NQC call, then compares the complete
mode on one alignment determined from untouched baseline waveforms.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import lal
import numpy as np
import torch

from . import qualify_generator as q
from .diagnose_native_source_nqc_inputs import _replay
from .diagnose_peak_anchor import fitted_peak
from .diagnose_q8_frequency_anchor import frequency_anchor
from .diagnose_source_flux_initial import _compile, _libraries, _snapshot
from .diagnose_source_nqc_inputs import INCLUDE
from .diagnose_source_nqc_whole_mode_override import OVERRIDE, override_source
from .diagnose_source_wheel_mode import _public, _source


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/captured_nqc_whole_mode_plan.json"


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _crop_metadata(library: ctypes.CDLL, mode: tuple) -> dict:
    k_min = ctypes.c_int.in_dll(library, "diffgw_last_kmin").value
    pre_crop_length = ctypes.c_int.in_dll(library, "diffgw_last_precrop_length").value
    if k_min < 0 or pre_crop_length - k_min != len(mode[2]):
        raise ValueError("Instrumented source crop does not match returned mode length")
    return {"k_min": k_min, "pre_crop_length": pre_crop_length,
            "returned_samples": len(mode[2])}


def _chart(mode: tuple) -> tuple[np.ndarray, np.ndarray]:
    epoch, step, values = mode[:3]
    return epoch + np.arange(len(values)) * step, values


def _anchor(mode: tuple, mass_seconds: float, target: float) -> dict:
    time_axis, values = _chart(mode)
    peak = fitted_peak(time_axis, values, "quadratic_power")
    return frequency_anchor(time_axis, values, mass_seconds, target, peak["time_s"])


def _aligned(mode: tuple, anchor: dict, mass_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    time_axis, values = _chart(mode)
    return ((time_axis - anchor["time_s"]) / mass_seconds,
            values * np.exp(-1j * anchor["phase_rad"]))


def _sample(chart: tuple[np.ndarray, np.ndarray], grid: np.ndarray) -> np.ndarray:
    time_axis, values = chart
    return np.interp(grid, time_axis, values.real) + 1j * np.interp(
        grid, time_axis, values.imag)


def _metric(candidate: np.ndarray, reference: np.ndarray,
            grid: np.ndarray, peak_M: float, match_M: float) -> dict:
    windows = {
        "whole": np.ones(len(grid), dtype=bool),
        "pre_match_minus_2M": grid < match_M - 2,
        "late_inspiral": (grid >= peak_M - 20) & (grid < peak_M),
        "post_peak": grid >= peak_M,
    }
    return {name: {"samples": int(mask.sum()),
                   "support_M": [float(grid[mask][0]), float(grid[mask][-1])],
                   "metrics": q.mode_metrics(candidate[mask], reference[mask])}
            for name, mask in windows.items() if mask.sum() >= 3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    names = ("capture", "capture_array", "source_wheel", "source_flux_plan",
             "native_waveform", "native_prior", "prior_stability", "adapter",
             "override_impl", "native_replay_impl", "compiler_impl",
             "anchor_impl", "metric_impl", "native_driver_impl", "first_attempt",
             "first_plan", "second_attempt", "second_plan",
             "third_attempt", "third_plan", "fourth_attempt", "fourth_plan")
    paths = {name: q.ROOT / plan[name] for name in names}
    if (plan["schema"] != "diffgw.seobnrv4.captured-nqc-whole-mode-plan.v5"
            or plan["release_qualified"] is not False
            or plan["row_name"] != "strong_aligned_spin"
            or plan["step_M"] != 0.125
            or plan["mode_capacity"] != 50000
            or plan["anchor_angular_frequency_M"] != 0.1
            or plan["script_sha256"] != q.sha256(__file__)
            or any(q.sha256(path) != plan[name + "_sha256"]
                   for name, path in paths.items())):
        parser.error("Frozen plan, runner, or inputs changed")
    captured = json.loads(paths["capture"].read_text())
    wheel_evidence = json.loads(paths["source_wheel"].read_text())
    source_plan = json.loads(paths["source_flux_plan"].read_text())
    native_evidence = json.loads(paths["native_waveform"].read_text())
    native_prior = json.loads(paths["native_prior"].read_text())
    stability = json.loads(paths["prior_stability"].read_text())
    first_attempt = json.loads(paths["first_attempt"].read_text())
    first_plan = json.loads(paths["first_plan"].read_text())
    second_attempt = json.loads(paths["second_attempt"].read_text())
    second_plan = json.loads(paths["second_plan"].read_text())
    third_attempt = json.loads(paths["third_attempt"].read_text())
    third_plan = json.loads(paths["third_plan"].read_text())
    fourth_attempt = json.loads(paths["fourth_attempt"].read_text())
    fourth_plan = json.loads(paths["fourth_plan"].read_text())
    wheel_row = next(item for item in wheel_evidence["rows"]
                     if item["name"] == plan["row_name"])
    row = plan["row"]
    if (captured["status"] != "diagnostic_complete"
            or captured["row"] != row or wheel_row["row"] != row
            or native_evidence["row"] != row
            or stability["gates"]["coefficient_stencil"] is not False
            or stability["stencil_coefficient_l2"] <= 0.01
            or first_attempt["status"] != "failed"
            or first_attempt["plan_sha256"] != q.sha256(paths["first_plan"])
            or first_plan["schema"] != "diffgw.seobnrv4.captured-nqc-whole-mode-plan.v1"
            or first_attempt["errors"] != [{
                "type": "ValueError", "message": "Intervention changed source epoch or cadence: "
                "stationary_frequency_peak_minus_reported_offset__historical_gsl_cubic"}]
            or first_attempt["comparisons"]
            or second_attempt["status"] != "failed"
            or second_attempt["plan_sha256"] != q.sha256(paths["second_plan"])
            or second_plan["schema"] != "diffgw.seobnrv4.captured-nqc-whole-mode-plan.v2"
            or second_attempt["errors"] != [{
                "type": "ValueError", "message": "Intervention changed pre-NQC prefix: "
                "reported_match__historical_gsl_cubic"}]
            or second_attempt["comparisons"]
            or third_attempt["status"] != "failed"
            or third_attempt["plan_sha256"] != q.sha256(paths["third_plan"])
            or third_plan["schema"] != "diffgw.seobnrv4.captured-nqc-whole-mode-plan.v3"
            or third_attempt["errors"] != [{
                "type": "ValueError", "message": "No substantial unchanged coarse prefix: "
                "reported_match__historical_gsl_cubic"}]
            or third_attempt["comparisons"]
            or fourth_attempt["status"] != "failed"
            or fourth_attempt["plan_sha256"] != q.sha256(paths["fourth_plan"])
            or fourth_plan["schema"] != "diffgw.seobnrv4.captured-nqc-whole-mode-plan.v4"
            or fourth_attempt["errors"] != [{
                "type": "ValueError", "message": "Intervention changed source count or cadence: "
                "stationary_frequency_peak_minus_reported_offset__historical_gsl_cubic"}]
            or fourth_attempt["comparisons"]):
        parser.error("Pinned capture, row, or prior stability failure changed")
    with np.load(paths["capture_array"]) as data:
        source_active = np.ascontiguousarray(data["coefficients"][[0, 1, 2, 6, 7]])
        source_meta = data["meta"].copy()
    if (np.max(abs(source_active - np.asarray(
            captured["rows"]["source"]["historical_active_coefficients"])))
            > plan["historical_coefficient_replay_linf_max"]):
        parser.error("Captured historical active coefficients changed")
    mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    report = {
        "schema": "diffgw.seobnrv4.captured-nqc-whole-mode.v5",
        "scope": dict(**q.SCOPE, development_only=True,
                      fixed_source_trajectory_inferred_from_code=True),
        "status": "running", "release_qualified": False,
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "input_sha256": {name: q.sha256(path) for name, path in paths.items()},
        "process": {"host": platform.node(), "cwd": os.getcwd(),
                    "pid": os.getpid(), "argv": sys.argv},
        "row": row, "step_M": plan["step_M"],
        "fixed": {"source_capture_array_sha256": q.sha256(paths["capture_array"]),
                  "source_match_M": float(source_meta[2]),
                  "source_reported_peak_M": float(source_meta[0]),
                  "source_stationary_match_M": captured["rows"]["source"][
                      "stationary_match_time_M"],
                  "prior_failed_0p01_stability_l2": stability["stencil_coefficient_l2"],
                  "trajectory": "same pinned source settings; fixed dynamics inferred from source call order, not independently certified by a prefix identity",
                  "alignment": "baseline-only 0.1/M frequency crossing and phase, reused for all source NQC interventions"},
        "baselines": {}, "interventions": {}, "excluded_interventions": {},
        "comparisons": {}, "errors": [],
    }
    q.write_evidence(args.json_out, report)
    started = time.monotonic()
    try:
        libraries = _libraries(source_plan["pinned_wheel"])
        with tempfile.TemporaryDirectory(prefix="diffgw-captured-nqc-whole-") as temporary:
            private = Path(temporary) / "source"
            _snapshot(args.source_root, source_plan["pinned_source"], private,
                      args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("Pinned source NQC include anchor changed")
            crop_anchor = "XLALGPSAdd (&tc, deltaT * (REAL8) kMin);"
            if original.count(crop_anchor) != 1:
                raise ValueError("Pinned source crop anchor changed")
            instrumented = original.replace(INCLUDE, INCLUDE + OVERRIDE).replace(
                crop_anchor,
                "diffgw_last_kmin = kMin;\n"
                "diffgw_last_precrop_length = (int) sigReVec->length;\n"
                + crop_anchor)
            source_file.write_text(
                "int diffgw_last_kmin = -1;\n"
                "int diffgw_last_precrop_length = -1;\n"
                + instrumented)
            dylib = Path(temporary) / "source.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths["adapter"], dylib,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            library = ctypes.CDLL(str(dylib))
            baseline_function = library.diffgw_v4_full_source_mode
            baseline_function.argtypes = [ctypes.c_double] * 6 + [
                ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double)]
            baseline_function.restype = ctypes.c_int
            override_function = library.diffgw_v4_full_source_mode_with_nqc
            override_function.argtypes = baseline_function.argtypes + [
                ctypes.POINTER(ctypes.c_double)]
            override_function.restype = ctypes.c_int
            baseline = _source(baseline_function, row, plan["step_M"],
                               plan["mode_capacity"])
            baseline_crop = _crop_metadata(library, baseline)
            noop = override_source(override_function, row, plan["step_M"],
                                   plan["mode_capacity"], source_active)
            noop_crop = _crop_metadata(library, noop)
            if (baseline[:2] != noop[:2]
                    or not np.array_equal(baseline[2], noop[2])
                    or baseline_crop != noop_crop
                    or _hash(baseline[2]) != wheel_row["source_mode_sha256"]):
                raise ValueError("Source baseline/no-op fails exact historical replay")
            report["baselines"]["source"] = {
                "mode_sha256": _hash(baseline[2]), "samples": len(baseline[2]),
                "epoch_s": baseline[0], "delta_t_s": baseline[1],
                "no_op_mode_sha256": _hash(noop[2]),
                "captured_active_coefficients": source_active.tolist(),
                "crop": baseline_crop, "no_op_crop": noop_crop,
            }
            print("source_baseline_and_noop_replayed", flush=True)
            q.write_evidence(args.json_out, report)
            baseline_values = baseline[2]
            interventions = {}
            for event in plan["event_rules"]:
                for derivative in plan["derivative_rules"]:
                    name = f"{event}__{derivative}"
                    fit = captured["rows"]["source"]["fits"][event][derivative]
                    active = np.ascontiguousarray(fit["coefficients"], dtype=np.float64)
                    mode = override_source(override_function, row, plan["step_M"],
                                           plan["mode_capacity"], active)
                    crop = _crop_metadata(library, mode)
                    if mode[1] != baseline[1]:
                        report["excluded_interventions"][name] = {
                            "reason": "different deltaT prevents shared pre-crop index comparison",
                            "coefficients": active.tolist(),
                            "mode_sha256": _hash(mode[2]),
                            "samples": len(mode[2]),
                            "crop": crop,
                            "returned_epoch_s": mode[0],
                            "returned_epoch_minus_baseline_M": float(
                                (mode[0] - baseline[0]) / mass_seconds),
                            "delta_t_s": mode[1],
                        }
                        print(f"excluded_intervention={name}", flush=True)
                        q.write_evidence(args.json_out, report)
                        raise ValueError(f"Intervention changed source cadence: {name}")
                    common_start = max(baseline_crop["k_min"], crop["k_min"])
                    common_end = min(baseline_crop["pre_crop_length"],
                                     crop["pre_crop_length"])
                    if common_end - common_start < 100:
                        raise ValueError(f"Insufficient common pre-crop indices: {name}")
                    baseline_common = baseline_values[
                        common_start - baseline_crop["k_min"]:
                        common_end - baseline_crop["k_min"]]
                    candidate_common = mode[2][
                        common_start - crop["k_min"]:common_end - crop["k_min"]]
                    changed = np.flatnonzero(candidate_common != baseline_common)
                    first_changed = (common_start + int(changed[0])
                                     if len(changed) else None)
                    # The source generator's returned epoch is peak-centered;
                    # index against the measured pre-crop kMin, then reuse
                    # the untouched baseline's pre-crop time origin.
                    comparison_epoch = (baseline[0]
                                        + (crop["k_min"] - baseline_crop["k_min"])
                                        * baseline[1])
                    interventions[name] = (comparison_epoch, mode[1], mode[2])
                    report["interventions"][name] = {
                        "coefficients": active.tolist(), "mode_sha256": _hash(mode[2]),
                        "samples": len(mode[2]), "returned_epoch_s": mode[0],
                        "crop": crop,
                        "common_precrop_indices": [common_start, common_end - 1],
                        "common_precrop_samples": common_end - common_start,
                        "returned_epoch_minus_baseline_M": float(
                            (mode[0] - baseline[0]) / mass_seconds),
                        "returned_epoch_minus_comparison_M": float(
                            (mode[0] - comparison_epoch) / mass_seconds),
                        "comparison_epoch_s": comparison_epoch,
                        "delta_t_s": mode[1],
                        "source_active_difference_l2": float(np.linalg.norm(
                            active - source_active)),
                        "unchanged_common_prefix_samples": (
                            first_changed - common_start
                            if first_changed is not None else common_end - common_start),
                        "unchanged_prefix_sha256": (
                            _hash(candidate_common[:first_changed - common_start])
                            if first_changed is not None and first_changed > common_start
                            else None),
                        "first_changed_precrop_index": first_changed,
                        "first_changed_baseline_time_s": (
                            float(baseline[0] + (first_changed - baseline_crop["k_min"])
                                  * baseline[1])
                            if first_changed is not None else None),
                    }
                    print(f"source_intervention={name}", flush=True)
                    q.write_evidence(args.json_out, report)
        public = _public(row, plan["step_M"])
        if _hash(public[2]) != wheel_row["wheel_mode_sha256"]:
            raise ValueError("Pinned public full mode fails replay")
        report["baselines"]["public"] = {
            "mode_sha256": _hash(public[2]), "samples": len(public[2]),
            "epoch_s": public[0], "delta_t_s": public[1],
        }
        torch.set_num_threads(1)
        level = native_evidence["candidate"]["ode_2"]["level"]
        native = _replay(row, level, q.Settings(**native_evidence["settings"]),
                         native_evidence, native_prior)
        native_summary = q.summarize_run(native)
        if native_summary["array_sha256"] != native_evidence["candidate"]["ode_2"][
                "native"]["array_sha256"]:
            raise ValueError("Pinned native full mode fails replay")
        from torchwave.constants import C_SI, MPC_SI

        scale = mass_seconds * float(C_SI) / (100 * float(MPC_SI))
        native_mode = (float(native["time"][0] * mass_seconds),
                       plan["step_M"] * mass_seconds,
                       native["mode"] * scale)
        report["baselines"]["native"] = {
            "native_array_sha256": native_summary["array_sha256"],
            "samples": len(native["mode"]), "epoch_s": native_mode[0],
            "delta_t_s": native_mode[1], "match_time_M": native["match_time_M"],
            "boundary_time_M": native["boundary_time_M"],
            "peak_time_M": native["peak_time_M"],
        }
        print("public_and_native_replayed", flush=True)
        q.write_evidence(args.json_out, report)
        modes = {"source": baseline, "native": native_mode, "public": public,
                 **interventions}
        anchors = {name: _anchor(mode, mass_seconds, plan["anchor_angular_frequency_M"])
                   for name, mode in modes.items() if name in {"source", "native", "public"}}
        chart = {name: _aligned(mode, anchors.get(name, anchors["source"]), mass_seconds)
                 for name, mode in modes.items()}
        lower = max(series[0][0] for series in chart.values())
        upper = min(series[0][-1] for series in chart.values())
        grid = chart["public"][0]
        grid = grid[(grid >= lower) & (grid <= upper)]
        if len(grid) < 100:
            raise ValueError("Insufficient common aligned whole-mode support")
        sampled = {name: _sample(series, grid) for name, series in chart.items()}
        public_peak_M = fitted_peak(*_chart(public), "quadratic_power")["time_s"]
        public_peak_M = (public_peak_M - anchors["public"]["time_s"]) / mass_seconds
        source_match_M = (source_meta[2] - source_meta[0]
                          + (fitted_peak(*_chart(baseline), "quadratic_power")["time_s"]
                             - anchors["source"]["time_s"]) / mass_seconds)
        # The source capture's NQC local time has a separate origin; its
        # peak-to-match offset transfers it to the full-mode peak chart.
        report["alignment"] = {
            "rule": "first pre-peak omega_22=0.1 crossing on each untouched baseline",
            "anchors": anchors, "candidate_anchor": "source baseline, unchanged",
            "source_precrop_time_origin_s": (baseline[0]
                                             - baseline_crop["k_min"] * baseline[1]),
            "common_support_M": [float(grid[0]), float(grid[-1])],
            "common_samples": len(grid), "public_peak_M": public_peak_M,
            "source_match_estimate_M": source_match_M,
            "source_match_chart_note": "captured peak-to-match offset transferred to fitted source full-mode peak",
        }
        pairs = [("source_vs_public", "source", "public"),
                 ("native_vs_public", "native", "public"),
                 ("source_vs_native", "source", "native")]
        pairs += [(f"{name}_vs_public", name, "public") for name in interventions]
        pairs += [(f"{name}_vs_source", name, "source") for name in interventions]
        pairs += [(f"{name}_vs_native", name, "native") for name in interventions]
        report["comparisons"] = {
            label: _metric(sampled[candidate], sampled[reference], grid,
                           public_peak_M, source_match_M)
            for label, candidate, reference in pairs
        }
        report["elapsed_seconds"] = time.monotonic() - started
        report["status"] = ("diagnostic_partial" if report["excluded_interventions"]
                            else "diagnostic_complete")
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    q.write_evidence(args.json_out, report)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}),
          flush=True)
    return 0 if report["status"] in {"diagnostic_complete", "diagnostic_partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Dev-only private source NQC input and same-grid fit attribution."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_native_source_nqc_inputs import _fit
from dev.native_rom.diagnose_nqc_gsl_same_input import GSLCubic
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_q4_ultrafine_nqc_inputs_plan.json"
OUTPUT = EVIDENCE / "source_q4_ultrafine_nqc_inputs.json"
NAMES = ("amplitude", "phase", "radius", "radial_momentum", "orbital_frequency")


def _fit_values(gsl, meta, values, match):
    axis = np.arange(len(values), dtype=np.float64)*meta[1]
    return _fit(gsl, axis, *values.T, match, meta[3:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    override_path = EVIDENCE / "source_q4_ultrafine_nqc_override.json"
    source_plan_path = ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_initial_plan.json"
    adapter = ROOT / "dev/native_rom/reference/full_v4_source_nqc_inputs_adapter.c"
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-ultrafine-nqc-inputs-plan.v1"
            or plan["release_qualified"] is not False
            or plan["output_steps_M"] != [0.03125, 0.015625]
            or plan["capacity"] != 50000 or plan["near_match_half_width_M"] != 2.0
            or q.sha256(override_path) != plan["override_result_sha256"]
            or q.sha256(source_plan_path) != plan["source_flux_plan_sha256"]
            or q.sha256(adapter) != plan["adapter_sha256"]):
        raise ValueError("frozen source NQC input plan changed")
    prior = json.loads(override_path.read_text())
    source_plan = json.loads(source_plan_path.read_text())
    if prior["status"] != "diagnostic_complete":
        raise ValueError("causal q4 NQC control is incomplete")
    row = prior["row"]
    result = dict(schema="diffgw.seobnrv4.source-q4-ultrafine-nqc-inputs.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  public_binary=prior["public_binary"],
                  instrumentation_sha256=hashlib.sha256(CAPTURE.encode()).hexdigest(),
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(),
                  row=row, grids={}, adjacent={})
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    captures = {}
    with tempfile.TemporaryDirectory(prefix="diffgw-q4-ultrafine-nqc-inputs-") as temp:
        private = Path(temp)/"source"
        _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
        source_file = private/"lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
        original = source_file.read_text()
        if original.count(INCLUDE) != 1:
            raise ValueError("pinned source NQC include anchor changed")
        source_file.write_text(original.replace(INCLUDE, INCLUDE+CAPTURE))
        libraries = _libraries(source_plan["pinned_wheel"])
        library = Path(temp)/"source_q4_inputs.dylib"
        _compile(private, args.headers, args.gsl_include, libraries, adapter, library,
                 source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"],
                 require_wheel_flux=False)
        function = ctypes.CDLL(str(library)).diffgw_v4_full_source_nqc_inputs
        function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        function.restype = ctypes.c_int
        gsl = GSLCubic(libraries["libgsl.28.dylib"])
        for step in reversed(plan["output_steps_M"]):
            meta, values, coefficients = _capture(function, row, step, plan["capacity"])
            source_active = np.asarray(prior["grids"][str(step)]["source_active"])
            active = coefficients[[0, 1, 2, 6, 7]]
            if np.max(abs(active-source_active)) > 1e-9:
                raise ValueError("source private NQC coefficient replay changed")
            fit = _fit_values(gsl, meta, values, meta[2])
            fit_error = float(np.linalg.norm(np.asarray(fit["coefficients"])-active))
            if fit_error > 1e-6:
                raise ValueError("GSL fit does not reproduce pinned source coefficients")
            result["grids"][str(step)] = dict(
                samples=len(values), input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                orbital_peak_M=float(meta[0]), internal_step_M=float(meta[1]),
                match_time_M=float(meta[2]), targets=meta[3:].tolist(),
                source_active=active.tolist(), python_gsl_fit=fit,
                python_gsl_fit_error_l2=fit_error)
            captures[step] = meta, values
            OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
            print(f"grid={step:g} samples={len(values)} internal_step={meta[1]:.9g} "
                  f"match={meta[2]:.9g} fit_error={fit_error:.9g}", flush=True)
        coarse_step, fine_step = plan["output_steps_M"]
        coarse_meta, coarse = captures[coarse_step]
        fine_meta, fine = captures[fine_step]
        if abs(fine_meta[1]-coarse_meta[1]) > 1e-12:
            raise ValueError("source private NQC grids do not share a time step")
        fine_index = np.arange(len(fine))
        near = abs(fine_index*fine_meta[1]-fine_meta[2]) <= plan["near_match_half_width_M"]
        if near.sum() < 10:
            raise ValueError("too few NQC samples near matching")
        lag_errors = {}
        for lag in range(-12, 13):
            coarse_index = fine_index[near]+lag
            valid = (coarse_index >= 0) & (coarse_index < len(coarse))
            lag_errors[lag] = float(np.sqrt(np.mean((
                fine[fine_index[near][valid], 2]-coarse[coarse_index[valid], 2])**2)))
        lag = min(lag_errors, key=lag_errors.get)
        fine_start = max(0, -lag)
        coarse_start = fine_start+lag
        shared_count = min(len(fine)-fine_start, len(coarse)-coarse_start)
        if shared_count < 100:
            raise ValueError("too few aligned source NQC samples")
        coarse_shared = coarse[coarse_start:coarse_start+shared_count]
        fine_shared = fine[fine_start:fine_start+shared_count]
        axis = (fine_start+np.arange(shared_count))*fine_meta[1]
        shared_near = abs(axis-fine_meta[2]) <= plan["near_match_half_width_M"]
        difference = fine_shared-coarse_shared
        match_relative = float(fine_meta[2]-fine_start*fine_meta[1])
        coarse_match_relative = float(coarse_meta[2]-coarse_start*coarse_meta[1])
        shared_fits = {
            "coarse_aligned_at_own_match": _fit_values(
                gsl, coarse_meta, coarse_shared, coarse_match_relative),
            "coarse_aligned_at_fine_match": _fit_values(
                gsl, coarse_meta, coarse_shared, match_relative),
            "fine_aligned_at_fine_match": _fit_values(
                gsl, fine_meta, fine_shared, match_relative),
        }
        fit_delta = np.asarray(shared_fits["fine_aligned_at_fine_match"]["coefficients"])-np.asarray(
            shared_fits["coarse_aligned_at_fine_match"]["coefficients"])
        match_delta = np.asarray(shared_fits["coarse_aligned_at_fine_match"]["coefficients"])-np.asarray(
            shared_fits["coarse_aligned_at_own_match"]["coefficients"])
        observed_delta = np.asarray(result["grids"][str(fine_step)]["source_active"])-np.asarray(
            result["grids"][str(coarse_step)]["source_active"])
        if (abs(match_relative-coarse_match_relative) > 1e-9
                or np.linalg.norm(fit_delta-observed_delta) > 1e-5):
            raise ValueError("aligned fit does not explain pinned coefficient drift")
        result["adjacent"] = dict(
            shared_samples=shared_count, near_match_samples=int(shared_near.sum()),
            common_step_M=float(coarse_meta[1]),
            local_coordinate_match_shift_M=float(fine_meta[2]-coarse_meta[2]),
            best_radius_alignment_lag_samples=lag,
            best_radius_alignment_rms=float(lag_errors[lag]),
            unshifted_radius_alignment_rms=float(lag_errors[0]),
            aligned_match_shift_M=float(match_relative-coarse_match_relative),
            target_max_abs_difference=float(np.max(abs(fine_meta[3:]-coarse_meta[3:]))),
            shared_input_difference={name: dict(
                max_abs_near_match=float(np.max(abs(difference[shared_near, j]))),
                rms_near_match=float(np.sqrt(np.mean(difference[shared_near, j]**2))))
                for j, name in enumerate(NAMES)},
            shared_fits=shared_fits,
            match_time_fit_coefficient_difference=match_delta.tolist(),
            match_time_fit_coefficient_difference_l2=float(np.linalg.norm(match_delta)),
            shared_input_fit_coefficient_difference=fit_delta.tolist(),
            shared_input_fit_coefficient_difference_l2=float(np.linalg.norm(fit_delta)),
            observed_source_coefficient_difference=observed_delta.tolist(),
            shared_fit_minus_observed_coefficient_difference_l2=float(
                np.linalg.norm(fit_delta-observed_delta)))
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()

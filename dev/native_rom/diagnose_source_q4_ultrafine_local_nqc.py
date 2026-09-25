"""Dev-only q4 ultrafine local-NQC fit and causal full-mode control."""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

import lal
import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_native_source_nqc_inputs import _fit
from dev.native_rom.diagnose_peak_anchor import fitted_metrics, fitted_peak
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_nqc_high_rate import LocalPolynomial
from dev.native_rom.diagnose_source_nqc_inputs import CAPTURE, INCLUDE, _capture
from dev.native_rom.diagnose_source_nqc_whole_mode_override import OVERRIDE, override_source
from dev.native_rom.diagnose_source_q4_ultrafine_nqc_override import _pre_peak
from dev.native_rom.diagnose_source_wheel_mode import _public, _source

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_q4_ultrafine_local_nqc_plan.json"
OUTPUT = EVIDENCE / "source_q4_ultrafine_local_nqc.json"


def _mode_function(library, name):
    function = getattr(library, name)
    function.argtypes = [ctypes.c_double] * 6 + [ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    function.restype = ctypes.c_int
    return function


def _chart(mode):
    epoch, step, values = mode
    return epoch + np.arange(len(values)) * step, values


def _fit_drift(coarse, fine, prefix):
    matrix_key = prefix + "_matrix"
    rhs_key = prefix + "_rhs"
    old_matrix, new_matrix = (np.asarray(item[matrix_key]) for item in (coarse, fine))
    old_rhs, new_rhs = (np.asarray(item[rhs_key]) for item in (coarse, fine))
    old_coefficients = np.linalg.solve(old_matrix, old_rhs)
    new_coefficients = np.linalg.solve(new_matrix, new_rhs)
    target_term = np.linalg.solve(new_matrix, new_rhs-old_rhs)
    matrix_term = -np.linalg.solve(new_matrix, (new_matrix-old_matrix) @ old_coefficients)
    if np.linalg.norm(new_coefficients-old_coefficients-target_term-matrix_term) > 1e-10:
        raise ValueError("NQC fit drift decomposition failed")
    return dict(total_l2=float(np.linalg.norm(new_coefficients-old_coefficients)),
                target_term_l2=float(np.linalg.norm(target_term)),
                matrix_term_l2=float(np.linalg.norm(matrix_term)),
                target_term=target_term.tolist(), matrix_term=matrix_term.tolist(),
                new_matrix_singular_values=np.linalg.svd(new_matrix).S.tolist())


def _frequency_anchor(chart, mass_seconds, target):
    time, mode = chart
    phase = np.unwrap(np.angle(mode))
    frequency = -np.gradient(phase, time / mass_seconds)
    peak = fitted_peak(time, mode, "quadratic_power")
    crossing = np.flatnonzero((frequency[:-1] < target)
        & (frequency[1:] >= target)
        & (time[:-1] < peak["time_s"] - 20 * mass_seconds))
    if len(crossing) != 1:
        raise ValueError("expected one early inspiral frequency crossing")
    index = int(crossing[0])
    fraction = (target-frequency[index]) / (frequency[index+1]-frequency[index])
    anchor_time = time[index] + fraction * (time[index+1]-time[index])
    anchor_phase = phase[index] + fraction * (phase[index+1]-phase[index])
    return dict(time_s=float(anchor_time), phase_rad=float(anchor_phase),
                time_before_peak_M=float((peak["time_s"]-anchor_time)/mass_seconds),
                bracket_M=[float(time[index]/mass_seconds),
                           float(time[index+1]/mass_seconds)])


def _anchored_metrics(candidate_chart, reference_chart, mass_seconds,
                      candidate_anchor, reference_anchor):
    candidate_time, candidate_mode = candidate_chart
    reference_time, reference_mode = reference_chart
    ct = candidate_time - candidate_anchor["time_s"]
    rt = reference_time - reference_anchor["time_s"]
    cm = candidate_mode * np.exp(-1j*candidate_anchor["phase_rad"])
    rm = reference_mode * np.exp(-1j*reference_anchor["phase_rad"])
    common = (rt >= ct[0]) & (rt <= ct[-1])
    time = rt[common]
    if len(time) < 100:
        raise ValueError("too few common frequency-anchored mode samples")
    candidate = np.interp(time, ct, cm.real) + 1j*np.interp(time, ct, cm.imag)
    reference = rm[common]
    peak = fitted_peak(*reference_chart, "quadratic_power")
    peak_time = peak["time_s"] - reference_anchor["time_s"]
    windows = {}
    for name, mask in (
        ("pre_attachment", time < peak_time-20*mass_seconds),
        ("peak_neighborhood", (time >= peak_time-20*mass_seconds)
                              & (time < peak_time+20*mass_seconds)),
        ("post_peak_20M", time >= peak_time+20*mass_seconds),
    ):
        if mask.sum() >= 3:
            windows[name] = q.mode_metrics(candidate[mask], reference[mask])
    return dict(metrics=q.mode_metrics(candidate, reference), windows=windows,
                reference_peak_from_anchor_M=float(peak_time/mass_seconds),
                common_support_from_anchor_M=[float(time[0]/mass_seconds),
                                              float(time[-1]/mass_seconds)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--headers", required=True, type=Path)
    parser.add_argument("--gsl-include", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    paths = {
        "source_plan": ROOT / "dev/native_rom/evidence/2026-09-24/source_flux_initial_plan.json",
        "input_evidence": EVIDENCE / "source_q4_ultrafine_nqc_inputs.json",
        "override_evidence": EVIDENCE / "source_q4_ultrafine_nqc_override.json",
        "capture_adapter": ROOT / "dev/native_rom/reference/full_v4_source_nqc_inputs_adapter.c",
        "override_adapter": ROOT / "dev/native_rom/reference/full_v4_source_large_nqc_override_adapter.c",
    }
    if (plan["schema"] != "diffgw.seobnrv4.source-q4-ultrafine-local-nqc-plan.v1"
            or plan["release_qualified"] is not False
            or plan["output_steps_M"] != [0.03125, 0.015625]
            or plan["polynomial"] != {"degree": 4, "half_width_M": 2.0}
            or plan["capacity"] != 50000 or plan["pre_peak_cut_M"] != 20.0
            or plan["inspiral_anchor_frequency_M_inv"] != 0.1
            or plan["peak_fit"] != "quadratic_power"
            or any(q.sha256(path) != plan[name + "_sha256"] for name, path in paths.items())):
        raise ValueError("frozen local NQC control changed")
    source_plan = json.loads(paths["source_plan"].read_text())
    inputs = json.loads(paths["input_evidence"].read_text())
    prior = json.loads(paths["override_evidence"].read_text())
    if (inputs["status"] != "diagnostic_complete"
            or prior["status"] != "diagnostic_complete"
            or inputs["row"] != prior["row"]):
        raise ValueError("pinned input or override result is unavailable")
    row = inputs["row"]
    mass_seconds = (row["mass1"] + row["mass2"]) * lal.MTSUN_SI
    result = dict(schema="diffgw.seobnrv4.source-q4-ultrafine-local-nqc.v1",
                  status="running", release_qualified=False,
                  plan_sha256=q.sha256(PLAN), script_sha256=q.sha256(__file__),
                  host=platform.node(), cwd=os.getcwd(), pid=os.getpid(), row=row,
                  instrumentation_sha256={"capture": hashlib.sha256(CAPTURE.encode()).hexdigest(),
                                           "override": hashlib.sha256(OVERRIDE.encode()).hexdigest()},
                  grids={}, adjacent={})
    shared = inputs["adjacent"]["shared_fits"]
    coarse_fit = shared["coarse_aligned_at_fine_match"]
    fine_fit = shared["fine_aligned_at_fine_match"]
    result["gsl_fit_drift"] = {
        name: _fit_drift(coarse_fit, fine_fit, name) for name in ("amplitude", "phase")}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with tempfile.TemporaryDirectory(prefix="diffgw-q4-local-nqc-") as temp:
        libraries = _libraries(source_plan["pinned_wheel"])
        compiled = {}
        for kind in ("capture", "override"):
            private = Path(temp) / kind
            _snapshot(args.source_root, source_plan["pinned_source"], private, args.headers)
            source_file = private / "lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"
            original = source_file.read_text()
            if original.count(INCLUDE) != 1:
                raise ValueError("pinned source NQC include anchor changed")
            source_file.write_text(original.replace(INCLUDE,
                INCLUDE + (CAPTURE if kind == "capture" else OVERRIDE)))
            library_path = Path(temp) / f"{kind}.dylib"
            _compile(private, args.headers, args.gsl_include, libraries,
                     paths[kind + "_adapter"], library_path,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            compiled[kind] = ctypes.CDLL(str(library_path))
        capture = compiled["capture"].diffgw_v4_full_source_nqc_inputs
        capture.argtypes = [ctypes.c_double] * 6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        capture.restype = ctypes.c_int
        baseline = _mode_function(compiled["override"], "diffgw_v4_full_source_mode")
        modified = _mode_function(compiled["override"], "diffgw_v4_full_source_mode_with_nqc")
        modified.argtypes += [ctypes.POINTER(ctypes.c_double)]
        polynomial = LocalPolynomial(plan["polynomial"]["half_width_M"],
                                     plan["polynomial"]["degree"])
        charts = {}
        fixed_charts = {}
        reference_active = None
        for step in reversed(plan["output_steps_M"]):
            meta, values, coefficients = _capture(capture, row, step, plan["capacity"])
            if (hashlib.sha256(values.tobytes()).hexdigest()
                    != inputs["grids"][str(step)]["input_sha256"]
                    or not np.array_equal(coefficients[[0, 1, 2, 6, 7]],
                                          np.asarray(inputs["grids"][str(step)]["source_active"]))):
                raise ValueError("pinned NQC input capture changed")
            time = np.arange(len(values), dtype=np.float64) * meta[1]
            fit = _fit(polynomial, time, *values.T, meta[2], meta[3:])
            local_active = np.asarray(fit["coefficients"], dtype=np.float64)
            if reference_active is None:
                reference_active = local_active.copy()
            source_active = coefficients[[0, 1, 2, 6, 7]]
            source = _source(baseline, row, step, plan["capacity"])
            noop = override_source(modified, row, step, plan["capacity"], source_active)
            local = override_source(modified, row, step, plan["capacity"], local_active)
            fixed = override_source(modified, row, step, plan["capacity"], reference_active)
            wheel = _public(row, step)
            if (hashlib.sha256(source[2].tobytes()).hexdigest()
                    != prior["grids"][str(step)]["source_mode_sha256"]
                    or not np.array_equal(noop[2], source[2])
                    or local[:2] != source[:2]
                    or fixed[:2] != source[:2]
                    or len(local[2]) != len(source[2])
                    or len(fixed[2]) != len(source[2])):
                raise ValueError("pinned source mode or NQC no-op changed")
            wheel_chart = _chart(wheel[:3])
            baseline_metrics = fitted_metrics(*_chart(source), *wheel_chart,
                                              mass_seconds, plan["peak_fit"])
            local_metrics = fitted_metrics(*_chart(local), *wheel_chart,
                                           mass_seconds, plan["peak_fit"])
            baseline_error = prior["grids"][str(step)]["fitted_baseline"]["metrics"]["complex_relative_l2"]
            if abs(baseline_metrics["metrics"]["complex_relative_l2"]-baseline_error) > 1e-12:
                raise ValueError("pinned source/wheel chart changed")
            local_pre = _pre_peak(_chart(local), wheel_chart, mass_seconds,
                                  plan["pre_peak_cut_M"], plan["peak_fit"])
            local_anchor = _frequency_anchor(_chart(local), mass_seconds,
                plan["inspiral_anchor_frequency_M_inv"])
            wheel_anchor = _frequency_anchor(wheel_chart, mass_seconds,
                plan["inspiral_anchor_frequency_M_inv"])
            anchored_wheel = _anchored_metrics(_chart(local), wheel_chart,
                                               mass_seconds, local_anchor, wheel_anchor)
            result["grids"][str(step)] = dict(
                input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                source_mode_sha256=hashlib.sha256(source[2].tobytes()).hexdigest(),
                local_mode_sha256=hashlib.sha256(local[2].tobytes()).hexdigest(),
                fixed_coefficient_mode_sha256=hashlib.sha256(fixed[2].tobytes()).hexdigest(),
                source_active=source_active.tolist(), local_fit=fit,
                local_minus_source_coefficient_l2=float(np.linalg.norm(local_active-source_active)),
                baseline_wheel_fitted_l2=baseline_error,
                local_wheel_fitted_l2=local_metrics["metrics"]["complex_relative_l2"],
                local_wheel_pre_attachment_l2=local_pre["metrics"]["complex_relative_l2"],
                local_wheel_inspiral_anchored=anchored_wheel,
                local_inspiral_frequency_anchor=local_anchor,
                wheel_inspiral_frequency_anchor=wheel_anchor,
                local_source_same_grid=q.mode_metrics(local[2], source[2]),
                fixed_minus_local_same_grid=q.mode_metrics(fixed[2], local[2]))
            charts[step] = _chart(local)
            fixed_charts[step] = _chart(fixed)
            OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(f"step={step:g} local/wheel={result['grids'][str(step)]['local_wheel_fitted_l2']:.9g}", flush=True)
        coarse, fine = plan["output_steps_M"]
        coarse_coeff = np.asarray(result["grids"][str(coarse)]["local_fit"]["coefficients"])
        fine_coeff = np.asarray(result["grids"][str(fine)]["local_fit"]["coefficients"])
        local_adjacent = fitted_metrics(*charts[coarse], *charts[fine],
                                        mass_seconds, plan["peak_fit"])
        fixed_adjacent = fitted_metrics(*fixed_charts[coarse], *fixed_charts[fine],
                                        mass_seconds, plan["peak_fit"])
        local_pre = _pre_peak(charts[coarse], charts[fine], mass_seconds,
                              plan["pre_peak_cut_M"], plan["peak_fit"])
        fixed_pre = _pre_peak(fixed_charts[coarse], fixed_charts[fine], mass_seconds,
                              plan["pre_peak_cut_M"], plan["peak_fit"])
        fixed_anchors = {
            str(step): _frequency_anchor(fixed_charts[step], mass_seconds,
                                         plan["inspiral_anchor_frequency_M_inv"])
            for step in (coarse, fine)}
        inspiral_anchored = _anchored_metrics(
            fixed_charts[coarse], fixed_charts[fine], mass_seconds,
            fixed_anchors[str(coarse)], fixed_anchors[str(fine)])
        result["adjacent"] = dict(
            source_coefficient_l2=inputs["adjacent"]["shared_input_fit_coefficient_difference_l2"],
            local_coefficient_l2=float(np.linalg.norm(fine_coeff-coarse_coeff)),
            local_fitted_complex_relative_l2=local_adjacent["metrics"]["complex_relative_l2"],
            local_fitted_windows=local_adjacent["windows"],
            local_pre_attachment_complex_relative_l2=local_pre["metrics"]["complex_relative_l2"],
            fixed_fine_coefficient_fitted_complex_relative_l2=
                fixed_adjacent["metrics"]["complex_relative_l2"],
            fixed_fine_coefficient_fitted_windows=fixed_adjacent["windows"],
            fixed_fine_coefficient_pre_attachment_complex_relative_l2=
                fixed_pre["metrics"]["complex_relative_l2"],
            fixed_fine_coefficient_inspiral_frequency_anchors=fixed_anchors,
            fixed_fine_coefficient_inspiral_anchored=inspiral_anchored)
    result["status"] = "diagnostic_complete"
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

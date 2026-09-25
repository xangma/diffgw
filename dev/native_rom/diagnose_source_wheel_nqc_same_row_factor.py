"""Dev-only same-row source/wheel NQC factor sensitivity on wheel dynamics."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import lal
import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_source_aligned_public_gap import _sha_array
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_wheel_mode import _public, _public_nqc, _source, _source_nqc


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_wheel_nqc_same_row_factor_plan.json"
OUTPUT = EVIDENCE / "source_wheel_nqc_same_row_factor.json"
ROW = (30.0, 20.0, 0.8, 0.6, 0.04)
SOURCE_ROOT = Path("/Users/xangma/repos/lalsuite-git")
HEADERS = Path("/tmp/diffgw-lalsuite-pinned-53acc0c/merged-include")
GSL_INCLUDE = Path("/usr/local/include")


def _plan() -> dict:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-wheel-nqc-same-row-factor-plan.v1"
            or plan["release_qualified"] is not False
            or tuple(plan["row_tuple"]) != ROW
            or plan["step_M"] != 0.5
            or plan["pre_peak_margin_M"] != 20.0
            or plan["capacity"] != 50000
            or plan["script_sha256"] != q.sha256(Path(__file__))):
        raise ValueError("Frozen same-row factor plan changed")
    for name, item in plan["inputs"].items():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    return plan


def _metric(candidate: np.ndarray, reference: np.ndarray) -> dict:
    residual = candidate-reference
    numerator = float(np.vdot(residual, residual).real)
    denominator = float(np.vdot(reference, reference).real)
    return {"samples": len(reference), "complex_relative_l2": float(np.sqrt(numerator/denominator)),
            "error_squared": numerator, "reference_squared": denominator}


def _factor(coeff: np.ndarray, r: np.ndarray, pr: np.ndarray,
            omega: np.ndarray) -> np.ndarray:
    ratio = pr/(r*omega)
    amplitude = 1.+ratio**2*(coeff[0]+coeff[1]/r+coeff[2]/r**1.5)
    phase = coeff[6]*ratio+coeff[7]*ratio*pr**2
    return amplitude*np.exp(1j*phase)


def main() -> None:
    plan = _plan()
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    three_way = json.loads((ROOT / plan["inputs"]["three_way"]["path"]).read_text())
    pinned = source_plan["pinned_source"]
    if (plan["source_identity"] != {
            "tag": pinned["tag"], "tag_object": pinned["tag_object"],
            "peeled_commit": pinned["peeled_commit"],
            "aligned_source_sha256": pinned["files_sha256"]["lalsimulation/lib/LALSimIMRSpinAlignedEOB.c"]}
            or plan["wheel_library_sha256"] != source_plan["pinned_wheel"]["libraries_sha256"]
            or three_way["row_tuple"] != list(ROW)
            or not three_way["diagnostic_checks_passed"]):
        raise ValueError("Source/wheel identities or prerequisite changed")
    libraries = _libraries(source_plan["pinned_wheel"])
    m1, m2, chi1, chi2, omega_start = ROW
    row = dict(mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, omega_start=omega_start)
    start = time.monotonic()
    result = {"schema": "diffgw.seobnrv4.source-wheel-nqc-same-row-factor.v1",
              "status": "running", "release_qualified": False,
              "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
              "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
              "row_tuple": list(ROW), "source_identity": plan["source_identity"],
              "wheel_library_sha256": {name: q.sha256(path) for name, path in libraries.items()}}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    wheel_epoch, wheel_dt, wheel_mode, low = _public(row, plan["step_M"])
    wheel_coeff = _public_nqc(row, plan["step_M"])
    with tempfile.TemporaryDirectory(prefix="diffgw-v4-same-row-factor-") as tmp:
        private = Path(tmp) / "source"
        _snapshot(SOURCE_ROOT, pinned, private, HEADERS)
        dylib = Path(tmp) / "full_source_mode.dylib"
        _compile(private, HEADERS, GSL_INCLUDE, libraries,
                 ROOT / plan["inputs"]["source_adapter"]["path"], dylib,
                 source_plan["full_generator_source_includes"],
                 source_plan["full_generator_public_includes"],
                 require_wheel_flux=False)
        library = ctypes.CDLL(str(dylib))
        function = library.diffgw_v4_full_source_mode
        function.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        function.restype = ctypes.c_int
        source_epoch, source_dt, source_mode = _source(function, row, plan["step_M"], plan["capacity"])
        coefficient_function = library.diffgw_v4_full_source_nqc
        coefficient_function.argtypes = [ctypes.c_double]*6 + [ctypes.POINTER(ctypes.c_double)]
        coefficient_function.restype = ctypes.c_int
        source_coeff = _source_nqc(coefficient_function, row, plan["step_M"])
    mass_seconds = (m1+m2)*lal.MTSUN_SI
    source_times_M = (source_epoch+np.arange(len(source_mode))*source_dt)/mass_seconds
    wheel_times_M = (wheel_epoch+np.arange(len(wheel_mode))*wheel_dt)/mass_seconds
    peak = int(np.argmax(abs(wheel_mode)))
    peak_time_M = float(wheel_times_M[peak])
    # The LAL mode epoch is relative to the orbital-frequency peak at the end
    # of the uniform low-resolution trajectory. Evaluate at continuous times;
    # no rounded mode-to-dynamics index is used in the correction.
    low_time_M = low[0]
    wheel_to_low_time_M = low_time_M[-1]+wheel_times_M
    mask = ((wheel_times_M >= source_times_M[0])
            & (wheel_times_M <= source_times_M[-1])
            & (wheel_times_M < peak_time_M-plan["pre_peak_margin_M"])
            & (wheel_to_low_time_M >= low_time_M[0])
            & (wheel_to_low_time_M <= low_time_M[-1]))
    if mask.sum() < 100:
        raise ValueError("Too few shared preattachment samples")
    query = wheel_to_low_time_M[mask]
    radius = np.interp(query, low_time_M, low[1])
    pr_star = np.interp(query, low_time_M, low[3])
    orbital_frequency = np.interp(query, low_time_M,
                                  np.gradient(low[2], low_time_M, edge_order=2))
    source_on_wheel = (np.interp(wheel_times_M[mask], source_times_M, source_mode.real)
                       +1j*np.interp(wheel_times_M[mask], source_times_M, source_mode.imag))
    wheel_before = wheel_mode[mask]
    source_factor = _factor(source_coeff, radius, pr_star, orbital_frequency)
    wheel_factor = _factor(wheel_coeff, radius, pr_star, orbital_frequency)
    no_op = wheel_before*(wheel_factor/wheel_factor)
    ratio_adjusted = wheel_before*(source_factor/wheel_factor)
    baseline = _metric(wheel_before, source_on_wheel)
    adjusted = _metric(ratio_adjusted, source_on_wheel)
    noop = _metric(no_op, wheel_before)
    low_dt_max_error = float(np.max(abs(np.diff(low_time_M)-plan["step_M"])))
    nearest_sample_error = float(np.max(abs((query-low_time_M[0])/plan["step_M"]
                                         -np.round((query-low_time_M[0])/plan["step_M"]))))
    checks = {
        "source_mode_hash_replays": _sha_array(source_mode) == three_way["source_mode_raw_sha256"],
        "wheel_mode_hash_replays": _sha_array(wheel_mode) == three_way["source_public_direct_mode_sha256"],
        "source_coefficients_replay": source_coeff.tolist() == three_way["source_nqc_coefficients_at_step_M"],
        "time_cadences": (abs(wheel_dt/mass_seconds-plan["step_M"]) < 1e-10
                          and abs(source_dt/mass_seconds-plan["step_M"]) < 1e-10
                          and low_dt_max_error < 1e-12),
        "source_wheel_epoch_agreement": abs(source_epoch-wheel_epoch)/mass_seconds < 1e-4,
        "continuous_low_grid_mapping": (nearest_sample_error < 0.25
                                        and np.all(query >= low_time_M[0])
                                        and np.all(query <= low_time_M[-1])),
        "physical_fields_finite": (np.isfinite(radius).all() and np.isfinite(pr_star).all()
                                   and np.isfinite(orbital_frequency).all()
                                   and np.all(radius > 0.) and np.all(orbital_frequency > 0.)),
        "factors_finite_nonzero": (np.isfinite(source_factor).all()
                                   and np.isfinite(wheel_factor).all()
                                   and np.all(abs(wheel_factor) > 0.)),
        "no_op_bitwise": np.array_equal(no_op, wheel_before),
        "no_op_zero_error": noop["complex_relative_l2"] == 0.,
        "strict_preattachment": float(query[-1]) < float(low_time_M[-1]),
    }
    result.update({
        "status": "diagnostic_complete" if all(checks.values()) else "failed",
        "elapsed_s": time.monotonic()-start,
        "step_M": plan["step_M"], "pre_peak_margin_M": plan["pre_peak_margin_M"],
        "source_epoch_s": source_epoch, "wheel_epoch_s": wheel_epoch,
        "source_delta_t_s": source_dt, "wheel_delta_t_s": wheel_dt,
        "source_mode_sha256": _sha_array(source_mode), "wheel_mode_sha256": _sha_array(wheel_mode),
        "source_coefficients": source_coeff.tolist(), "wheel_coefficients": wheel_coeff.tolist(),
        "source_minus_wheel_coefficients": (source_coeff-wheel_coeff).tolist(),
        "source_samples": len(source_mode), "wheel_samples": len(wheel_mode),
        "wheel_peak_index": peak, "wheel_peak_time_M": peak_time_M,
        "mapping": {"rule": "Place low-grid orbital-frequency peak at mode time 0, then interpolate the wheel low dynamics at continuous wheel mode times. No rounded index is used.",
                    "low_time_start_M": float(low_time_M[0]),
                    "low_time_end_M": float(low_time_M[-1]),
                    "query_start_M": float(query[0]), "query_end_M": float(query[-1]),
                    "low_dt_max_error_M": low_dt_max_error,
                    "max_nearest_sample_offset": nearest_sample_error,
                    "query_time_sha256": _sha_array(query),
                    "radius_sha256": _sha_array(radius),
                    "pr_star_sha256": _sha_array(pr_star),
                    "orbital_frequency_sha256": _sha_array(orbital_frequency)},
        "window": {"rule": "Shared source/wheel support, strictly earlier than sampled wheel amplitude peak minus 20M, and within low-dynamics support; no time, phase, or amplitude fitting.",
                   "samples": int(mask.sum()),
                   "first_wheel_time_M": float(wheel_times_M[mask][0]),
                   "last_wheel_time_M": float(wheel_times_M[mask][-1]),
                   "source_interpolated_sha256": _sha_array(source_on_wheel),
                   "wheel_mode_sha256": _sha_array(wheel_before),
                   "adjusted_mode_sha256": _sha_array(ratio_adjusted)},
        "metrics": {"baseline_wheel_vs_source": baseline,
                    "source_factor_over_wheel_factor_vs_source": adjusted,
                    "wheel_factor_over_itself_noop": noop},
        "squared_error_ratio_adjusted_over_baseline": adjusted["error_squared"]/baseline["error_squared"],
        "checks": {name: bool(value) for name, value in checks.items()},
        "diagnostic_checks_passed": all(checks.values()),
    })
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "baseline_l2": baseline["complex_relative_l2"],
                      "adjusted_l2": adjusted["complex_relative_l2"]}), flush=True)
    if not result["diagnostic_checks_passed"]:
        raise ValueError("Same-row source/wheel factor diagnostic failed gates")


if __name__ == "__main__":
    main()

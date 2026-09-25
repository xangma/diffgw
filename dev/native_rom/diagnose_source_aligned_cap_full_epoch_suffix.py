"""Development-only source cap/full-mode epoch and exact-crop suffix check."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time

import lal
import numpy as np

from dev.native_rom import qualify_generator as q
from dev.native_rom.diagnose_source_flux_initial import _compile, _libraries, _snapshot
from dev.native_rom.diagnose_source_wheel_dynamics import _source as source_dynamics
from dev.native_rom.diagnose_source_wheel_mode import _source as source_mode, _source_nqc


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "dev/native_rom/evidence/2026-09-25"
PLAN = EVIDENCE / "source_aligned_cap_full_epoch_suffix_plan.json"
OUTPUT = EVIDENCE / "source_aligned_cap_full_epoch_suffix.json"
ROW = (30.0, 20.0, 0.8, 0.6, 0.04)


def array_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def pair(values: np.ndarray) -> list[list[float]]:
    return np.column_stack((values.real, values.imag)).tolist()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.source-cap-full-epoch-suffix-plan.v1"
            or plan["release_qualified"] is not False
            or tuple(plan["row_tuple"]) != ROW
            or plan["step_M"] != 0.5
            or plan["cap_Momega"] != math.pow(10.0, -1.5)
            or plan["script_sha256"] != q.sha256(Path(__file__))):
        raise ValueError("frozen source epoch plan changed")
    for item in plan["inputs"].values():
        if q.sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"frozen input changed: {item['path']}")
    source_plan = json.loads((ROOT / plan["inputs"]["source_plan"]["path"]).read_text())
    v5 = json.loads((ROOT / plan["inputs"]["v5_result"]["path"]).read_text())
    if (tuple(v5["row_tuple"]) != ROW or v5["release_qualified"] is not False
            or not v5["validity_passed"]):
        raise ValueError("frozen requested-frequency source control changed")
    libraries = _libraries(source_plan["pinned_wheel"])
    started = time.monotonic()
    result = {
        "schema": "diffgw.seobnrv4.source-cap-full-epoch-suffix.v1",
        "status": "running", "release_qualified": False,
        "host": platform.node(), "cwd": os.getcwd(), "pid": os.getpid(),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(Path(__file__)),
        "row_tuple": list(ROW), "step_M": plan["step_M"],
        "source_root": plan["source_root"], "checks": {},
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    with tempfile.TemporaryDirectory(prefix="diffgw-v4-source-cap-epoch-") as tmp:
        private = Path(tmp) / "source"
        headers = Path(plan["headers"])
        _snapshot(Path(plan["source_root"]), source_plan["pinned_source"],
                  private, headers)

        def build(name: str, key: str) -> ctypes.CDLL:
            library = Path(tmp) / name
            _compile(private, headers, Path(plan["gsl_include"]), libraries,
                     ROOT / plan["inputs"][key]["path"], library,
                     source_plan["full_generator_source_includes"],
                     source_plan["full_generator_public_includes"],
                     require_wheel_flux=False)
            return ctypes.CDLL(str(library))

        mode_library = build("mode.dylib", "mode_adapter")
        mode_fn = mode_library.diffgw_v4_full_source_mode
        mode_fn.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        mode_fn.restype = ctypes.c_int
        nqc_fn = mode_library.diffgw_v4_full_source_nqc
        nqc_fn.argtypes = [ctypes.c_double]*6 + [ctypes.POINTER(ctypes.c_double)]
        nqc_fn.restype = ctypes.c_int
        dynamics_library = build("dynamics.dylib", "dynamics_adapter")
        dynamics_fn = dynamics_library.diffgw_v4_full_source_dynamics
        dynamics_fn.argtypes = [ctypes.c_double]*6 + [ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double)]
        dynamics_fn.restype = ctypes.c_int

        row = dict(zip(("mass1", "mass2", "chi1", "chi2", "omega_start"), ROW))
        cap_row = row | {"omega_start": plan["cap_Momega"]}
        cap_epoch, cap_dt, cap_mode = source_mode(
            mode_fn, cap_row, plan["step_M"], plan["capacity"])
        requested_epoch, requested_dt, requested_mode = source_mode(
            mode_fn, row, plan["step_M"], plan["capacity"])
        cap_low = source_dynamics(dynamics_fn, cap_row, plan["step_M"],
                                  plan["capacity"])
        requested_low = source_dynamics(dynamics_fn, row, plan["step_M"],
                                        plan["capacity"])
        cap_nqc = _source_nqc(nqc_fn, cap_row, plan["step_M"])
        requested_nqc = _source_nqc(nqc_fn, row, plan["step_M"])

    delta = len(cap_mode)-len(requested_mode)
    exact_offsets = [k for k in range(max(0, delta)+1)
                     if np.array_equal(cap_mode[k:k+len(requested_mode)],
                                       requested_mode)]
    epoch_offset = round((requested_epoch-cap_epoch)/cap_dt)
    matched_offset = exact_offsets[0] if len(exact_offsets) == 1 else None
    diagnostic_offset = matched_offset if matched_offset is not None else epoch_offset
    candidate_slice = cap_mode[diagnostic_offset:diagnostic_offset+len(requested_mode)]
    candidate_difference = (float(np.max(abs(candidate_slice-requested_mode)))
                            if len(candidate_slice) == len(requested_mode) else None)
    epoch_residual = (requested_epoch-cap_epoch-diagnostic_offset*cap_dt)
    mass_seconds = (ROW[0]+ROW[1])*lal.MTSUN_SI
    peak_from_uncropped_epoch = -cap_epoch/mass_seconds
    low_end = float(cap_low[0, -1])
    mapped_first = diagnostic_offset*plan["step_M"]
    legacy_first = low_end+requested_epoch/mass_seconds
    gates = plan["gates"]
    checks = {
        "finite_modes_dynamics_nqc_epochs": bool(all(np.isfinite(x).all() for x in (
            cap_mode, requested_mode, cap_low, requested_low, cap_nqc,
            requested_nqc)) and np.isfinite((cap_epoch, requested_epoch,
                                              cap_dt, requested_dt)).all()),
        "cap_mode_longer": delta > 0,
        "same_output_delta_t": abs(cap_dt-requested_dt)
            <= gates["delta_t_max_abs_s"],
        "expected_output_delta_t": abs(cap_dt/mass_seconds-plan["step_M"])
            <= gates["cadence_max_abs_M"],
        "unique_exact_mode_suffix": len(exact_offsets) == 1
            and matched_offset == delta,
        "epoch_matches_discrete_crop": matched_offset is not None
            and abs(epoch_residual) <= gates["epoch_residual_max_abs_s"],
        "identical_low_dynamics": cap_low.shape == requested_low.shape
            and np.array_equal(cap_low, requested_low),
        "low_time_starts_at_zero": abs(cap_low[0, 0])
            <= gates["time_origin_max_abs_M"],
        "low_output_cadence": np.max(abs(np.diff(cap_low[0])-plan["step_M"]))
            <= gates["cadence_max_abs_M"],
        "identical_nqc_coefficients": np.array_equal(cap_nqc, requested_nqc),
        "prior_cropped_mode_replay": (
            len(requested_mode) == v5["source_full_control"]["already_cropped_mode_samples"]
            and requested_epoch == v5["source_full_control"]["already_cropped_mode_epoch_s"]
            and requested_dt == v5["source_full_control"]["already_cropped_mode_delta_t_s"]
            and array_sha(requested_mode)
            == v5["source_full_control"]["raw_mode_sha256"]),
    }
    result.update(
        status="diagnostic_complete" if all(checks.values()) else "failed_validity",
        elapsed_s=time.monotonic()-started, checks=checks,
        validity_passed=all(checks.values()),
        source_epoch_control={
            "pinned_formula": "Uncropped tc=-Msec*(low[hiSRndx]+high_timePeak); cropped tc adds kMin*deltaT.",
            "cap_mode_epoch_s": cap_epoch,
            "requested_mode_epoch_s": requested_epoch,
            "cap_output_delta_t_s": cap_dt,
            "requested_output_delta_t_s": requested_dt,
            "cap_mode_samples": len(cap_mode),
            "requested_mode_samples": len(requested_mode),
            "cap_mode_sha256": array_sha(cap_mode),
            "requested_mode_sha256": array_sha(requested_mode),
            "cap_mode": pair(cap_mode),
            "exact_suffix_offsets": exact_offsets,
            "epoch_implied_offset": epoch_offset,
            "tested_offset": diagnostic_offset,
            "tested_suffix_max_abs": candidate_difference,
            "epoch_residual_s": epoch_residual,
            "cap_peak_time_from_uncropped_epoch_M": peak_from_uncropped_epoch,
            "low_dynamics_endpoint_M": low_end,
            "peak_minus_low_endpoint_M": peak_from_uncropped_epoch-low_end,
            "exact_first_cropped_low_time_M": mapped_first,
            "legacy_low_endpoint_plus_cropped_epoch_time_M": legacy_first,
            "exact_minus_legacy_first_time_M": mapped_first-legacy_first,
        },
        low_dynamics_control={
            "cap_shape": list(cap_low.shape),
            "requested_shape": list(requested_low.shape),
            "cap_sha256": array_sha(cap_low),
            "requested_sha256": array_sha(requested_low),
            "initial_state": cap_low[1:, 0].tolist(),
            "last_state": cap_low[1:, -1].tolist(),
        },
        nqc_control={
            "cap_coefficients": cap_nqc.tolist(),
            "requested_coefficients": requested_nqc.tolist(),
            "cap_sha256": array_sha(cap_nqc),
            "requested_sha256": array_sha(requested_nqc),
        },
        interpretation=(
            "Source-only numerical epoch/suffix check. A passing result fixes the "
            "source full-mode sample chart to the low-stage cap-time grid; it does "
            "not assess native corrected-mode parity or release qualification."),
    )
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(result["status"], result["source_epoch_control"]
          ["exact_minus_legacy_first_time_M"], flush=True)


if __name__ == "__main__":
    main()

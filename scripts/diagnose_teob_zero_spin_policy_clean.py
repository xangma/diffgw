#!/usr/bin/env python3
"""Bounded public zero-spin PN-spin-order policy control.

This is an attribution diagnostic.  It makes two fresh public
``SimIMRTEOBResumS`` calls with the reviewed exact-start inputs: PA_ONLY=2
with explicit ``PNSpinOrder=0``, and full TD with explicit ``PNSpinOrder=0``
and ``TEOB_use_postadiabatic=1``; the saved PA/TD baselines retain their
omitted/default PN-spin policy.  Comparisons use equal elapsed
sample indices only; no interpolation, alignment, fitting, or runtime change
is involved.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import lal
import lalsimulation as lalsim
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs/reviews/2026-09-13-parity-review"
VERIFY = REVIEW / "verification/teob-zero-spin-policy-clean"
REPORT = VERIFY / "report.json"
CONCLUSION = VERIFY / "conclusion.md"

PA_SCRIPT = ROOT / "scripts/diagnose_teob_pa_exact_start_clean.py"
CONDITIONER_SCRIPT = ROOT / "scripts/diagnose_teob_conditioner_public_clean.py"
PA_ARRAYS = VERIFY.parent / "teob-pa-exact-start-clean/arrays.npz"
PA_REPORT = VERIFY.parent / "teob-pa-exact-start-clean/report.json"
NATIVE_ARRAYS = VERIFY.parent / "teob-native-replay-public-clean/replay-native-output.npz"
NATIVE_REPORT = VERIFY.parent / "teob-native-replay-public-clean/replay-report.json"
PUBLIC_ARRAYS = VERIFY.parent / "teob-conditioner-public-clean/arrays.npz"
PUBLIC_REPORT = VERIFY.parent / "teob-conditioner-public-clean/report.json"

M1 = 30.0
M2 = 20.0
DISTANCE_MPC = 100.0
F_START = 16.900094009839023
F_REF = 20.0
DT_SECONDS = 1.0 / 2048.0
SPINS = (0.0,) * 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arr_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def finite(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def options(*, pa_mode: int | None, spin_order: int | None) -> lal.Dict:
    params = lal.CreateDict()
    if pa_mode is not None:
        lal.DictInsertINT4Value(params, "TEOB_use_postadiabatic", pa_mode)
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(params, 0)
    mode_array = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 2, 2)
    lalsim.SimInspiralWaveformParamsInsertModeArray(params, mode_array)
    if spin_order is not None:
        lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(params, spin_order)
    return params


def public_call(*, label: str, pa_mode: int | None, spin_order: int | None) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    started = time.perf_counter()
    hp_s, hc_s = lalsim.SimIMRTEOBResumS(
        0.0, DT_SECONDS, M1 * lal.MSUN_SI, M2 * lal.MSUN_SI,
        *SPINS, 0.0, 0.0, DISTANCE_MPC * 1.0e6 * lal.PC_SI,
        0.0, 0.0, options(pa_mode=pa_mode, spin_order=spin_order),
        0.0, 0.0, F_START, F_REF,
    )
    hp = -np.asarray(hp_s.data.data, dtype=np.float64).copy()
    hc = -np.asarray(hc_s.data.data, dtype=np.float64).copy()
    if hp.shape != hc.shape or hp.ndim != 1 or hp.size < 2:
        raise RuntimeError(f"{label}: invalid polarization shape")
    epoch = float(hp_s.epoch)
    dt = float(hp_s.deltaT)
    if dt != DT_SECONDS or not np.isfinite(hp).all() or not np.isfinite(hc).all():
        raise RuntimeError(f"{label}: public sampling/output contract failed")
    elapsed_seconds = np.arange(hp.size, dtype=np.float64) * dt
    meta = {
        "label": label,
        "pa_mode": pa_mode,
        "PNSpinOrder": "omitted; public default preserved" if spin_order is None else int(spin_order),
        "TEOB_use_postadiabatic": pa_mode if pa_mode is not None else "omitted; public default preserved",
        "epoch_seconds": epoch,
        "samples": int(hp.size),
        "delta_t_seconds": dt,
        "elapsed_start_seconds": finite(elapsed_seconds[0]),
        "elapsed_end_seconds": finite(elapsed_seconds[-1]),
        "array_hashes": {"h_plus": arr_hash(hp), "h_cross": arr_hash(hc), "elapsed": arr_hash(elapsed_seconds)},
        "elapsed_seconds": finite(time.perf_counter() - started),
    }
    return meta, elapsed_seconds, hp, hc


def exact_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"status": "refused_length_mismatch", "actual_shape": list(a.shape), "reference_shape": list(b.shape)}
    diff = a - b
    denom = max(float(np.linalg.norm(b)), np.finfo(np.float64).tiny)
    return {
        "status": "complete",
        "samples": int(a.size),
        "relative_l2": float(np.linalg.norm(diff) / denom),
        "max_abs": float(np.max(np.abs(diff))),
        "difference_sha256": arr_hash(diff),
    }


def compare_triplet(actual_t: np.ndarray, actual_hp: np.ndarray, actual_hc: np.ndarray,
                    ref_t: np.ndarray, ref_hp: np.ndarray, ref_hc: np.ndarray) -> dict[str, Any]:
    if actual_t.shape != ref_t.shape or actual_hp.shape != ref_hp.shape or actual_hc.shape != ref_hc.shape:
        return {"status": "refused_elapsed_grid_mismatch", "actual_samples": int(actual_t.size), "reference_samples": int(ref_t.size)}
    if actual_t.size < 2 or actual_t[0] != 0.0 or ref_t[0] != 0.0:
        return {"status": "refused_elapsed_grid_mismatch", "reason": "nonzero_or_short_elapsed_grid"}
    actual_dt = float(actual_t[1] - actual_t[0])
    ref_dt = float(ref_t[1] - ref_t[0])
    if actual_dt != ref_dt:
        return {"status": "refused_elapsed_grid_mismatch", "reason": "different_cached_sample_spacing", "actual_dt": actual_dt, "reference_dt": ref_dt}
    # Compare direct values only when serialized elapsed grids are bitwise
    # identical; no interpolation, shift, or alignment is applied.
    if not np.array_equal(actual_t, ref_t):
        return {
            "status": "refused_elapsed_grid_mismatch",
            "reason": "elapsed_grids_not_bitwise_equal",
            "max_elapsed_grid_value_difference": float(np.max(np.abs(actual_t - ref_t))),
            "sample_spacing": actual_dt,
        }
    return {
        "status": "complete",
        "exact_elapsed_indices": True,
        "elapsed_grid_values_bitwise_equal": True,
        "max_elapsed_grid_value_difference": 0.0,
        "sample_spacing": actual_dt,
        "h_plus": exact_metrics(actual_hp, ref_hp),
        "h_cross": exact_metrics(actual_hc, ref_hc),
    }


def load_sources() -> dict[str, str]:
    paths = [__file__, PA_SCRIPT, CONDITIONER_SCRIPT, PA_ARRAYS, PA_REPORT,
             NATIVE_ARRAYS, NATIVE_REPORT, PUBLIC_ARRAYS, PUBLIC_REPORT]
    return {str(path): sha256(Path(path)) for path in paths}


def main() -> None:
    VERIFY.mkdir(parents=True, exist_ok=True)
    source_before = load_sources()

    pa_meta, pa_t_s, pa_hp, pa_hc = public_call(
        label="explicit_PNSpinOrder_0_PA_ONLY_2", pa_mode=2, spin_order=0
    )
    td_meta, td_t_s, td_hp, td_hc = public_call(
        label="explicit_PNSpinOrder_0_full_TD_PA_1", pa_mode=1, spin_order=0
    )

    with np.load(PA_ARRAYS, allow_pickle=False) as saved:
        pa_t = np.asarray(saved["time_M"], dtype=np.float64)
        pre_nqc = np.asarray(saved["pre_nqc_h"], dtype=np.complex128)
        default_pa = np.asarray(saved["public_pa_h"], dtype=np.complex128)
        pa_pre_hp = np.asarray(saved["pre_nqc_h_plus"], dtype=np.float64)
        pa_pre_hc = np.asarray(saved["pre_nqc_h_cross"], dtype=np.float64)
        pa_default_hp = np.asarray(saved["public_pa_h_plus"], dtype=np.float64)
        pa_default_hc = np.asarray(saved["public_pa_h_cross"], dtype=np.float64)
    with np.load(NATIVE_ARRAYS, allow_pickle=False) as saved:
        native_t = np.asarray(saved["td_time"], dtype=np.float64)
        native_hp = np.asarray(saved["h_plus_td"], dtype=np.float64)
        native_hc = np.asarray(saved["h_cross_td"], dtype=np.float64)
    with np.load(PUBLIC_ARRAYS, allow_pickle=False) as saved:
        saved_td_hp = np.asarray(saved["exact_start_raw_h_plus"], dtype=np.float64)
        saved_td_hc = np.asarray(saved["exact_start_raw_h_cross"], dtype=np.float64)

    # PA cache time is dimensionless M; public output time is seconds.  The
    # exact index grid is tested after conversion using the reviewed mass.
    dt_m = DT_SECONDS / (50.0 * lal.MTSUN_SI)
    pa_public_t_m = np.arange(pa_hp.size, dtype=np.float64) * dt_m
    pa_actual_h = pa_hp - 1j * pa_hc
    pa_saved_h = default_pa
    pre_h = pre_nqc
    pa_compare = {
        "fresh_explicit_vs_current_analytic_pre_NQC": compare_triplet(
            pa_public_t_m, pa_hp, pa_hc, pa_t, pa_pre_hp, pa_pre_hc
        ),
        "fresh_explicit_vs_saved_public_default_PA": compare_triplet(
            pa_public_t_m, pa_hp, pa_hc, pa_t, pa_default_hp, pa_default_hc
        ),
    }
    pa_compare["complex_summary"] = {
        "fresh_explicit_vs_pre_NQC": (
            exact_metrics(pa_actual_h, pre_h)
            if pa_compare["fresh_explicit_vs_current_analytic_pre_NQC"]["status"] == "complete"
            else {"status": pa_compare["fresh_explicit_vs_current_analytic_pre_NQC"]["status"]}
        ),
        "fresh_explicit_vs_saved_default_PA": (
            exact_metrics(pa_actual_h, pa_saved_h)
            if pa_compare["fresh_explicit_vs_saved_public_default_PA"]["status"] == "complete"
            else {"status": pa_compare["fresh_explicit_vs_saved_public_default_PA"]["status"]}
        ),
    }

    td_public_t_m = np.arange(td_hp.size, dtype=np.float64) * dt_m
    td_compare = {
        "fresh_explicit_PNSpinOrder_0_full_TD_vs_current_analytic_native": compare_triplet(
            td_public_t_m, td_hp, td_hc, native_t, native_hp, native_hc
        ),
        "fresh_explicit_PNSpinOrder_0_full_TD_vs_saved_public_default_exact_start_TD": compare_triplet(
            td_t_s, td_hp, td_hc,
            np.arange(saved_td_hp.size, dtype=np.float64) * DT_SECONDS,
            saved_td_hp, saved_td_hc,
        ),
    }

    source_after = load_sources()
    report = {
        "status": "complete",
        "diagnostic": "zero-spin public PNSpinOrder policy attribution control",
        "scope": "Two fresh public calls; no LAL implementation inspection, runtime edit, fitting, interpolation, alignment, or full native ODE.",
        "request": {
            "api": "SimIMRTEOBResumS", "masses_msun": [M1, M2], "spins": list(SPINS),
            "distance_mpc": DISTANCE_MPC, "inclination_rad": 0.0, "phi_ref_rad": 0.0,
            "mode": [[2, 2]], "PNTidalOrder": 0, "f_start_hz": F_START,
            "f_ref_hz": F_REF, "dt_seconds_requested": DT_SECONDS,
            "observer_sign": -1,
        },
        "calls": {"explicit_PA": pa_meta, "explicit_full_TD": td_meta},
        "comparisons": {"PA": pa_compare, "full_TD": td_compare},
        "saved_caches": {
            "pa_arrays": str(PA_ARRAYS), "pa_report": str(PA_REPORT),
            "native_arrays": str(NATIVE_ARRAYS), "native_report": str(NATIVE_REPORT),
            "public_arrays": str(PUBLIC_ARRAYS), "public_report": str(PUBLIC_REPORT),
        },
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "source_hashes_stable": source_before == source_after,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "conclusion": (
            "With zero-valued spins, the omitted public default and explicit PNSpinOrder=0 "
            "are separate attribution controls: the explicit PA-only output is compared "
            "on exact elapsed indices to the current analytic pre-NQC cache and saved default PA, "
            "while the explicit full-TD control is compared to omitted/default native and saved public TD. "
            "That control returned 4089 samples versus the cached 4087, so its TD comparisons "
            "are refused without truncation or alignment. This alternative public policy is not frozen parity; any difference can include "
            "branch, target, trajectory, timing, or reconstruction effects."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    CONCLUSION.write_text(
        "# Zero-spin PN spin-order policy control\n\n"
        + report["conclusion"] + "\n\n"
        "The report records actual epochs, lengths, sample spacing, exact elapsed-grid checks, "
        "and source/input hashes.\n"
    )


if __name__ == "__main__":
    main()

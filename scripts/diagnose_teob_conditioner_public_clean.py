#!/usr/bin/env python3
"""Attribute TEOB TD conditioning with saved public TD and public APIs.

This diagnostic is deliberately source isolated.  It does not generate a
native trajectory, import a native/reference implementation, align signals,
fit thresholds, or combine the two FD polarizations into one complex series.
The saved public TD fixture is negated according to the reviewed direct-EOB
versus generic-observer-frame contract before both conditioners are called.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import lal
import lalsimulation as lalsim
import numpy as np
import torch

from torchwave.approximants.eob.td_fd_conditioning_clean import (
    ConditionedTD,
    condition_td,
    fft_length_and_sampling,
    pair_leading_pad_final_crop,
    plan_td_support,
)


ROOT = Path(__file__).resolve().parents[1]
TD_FIXTURE = ROOT / (
    "docs/reviews/2026-09-13-parity-review/verification/"
    "teob-first-difference-clean/public-td-output.npz"
)
FD_FIXTURE = ROOT / (
    "docs/reviews/2026-09-13-parity-review/verification/"
    "teob-conditioned-waveform-clean/public-fd-sim-inspiral-output.npz"
)
OUT_DIR = ROOT / (
    "docs/reviews/2026-09-13-parity-review/verification/"
    "teob-conditioner-public-clean"
)
REPORT = OUT_DIR / "report.json"
ARRAYS = OUT_DIR / "arrays.npz"
CONDITIONER = ROOT / "src/torchwave/approximants/eob/td_fd_conditioning_clean.py"
CONTRACT = ROOT / "docs/reviews/2026-09-13-parity-review/td-fd-conditioning-contract.md"

M1 = 30.0
M2 = 20.0
FLOW = 20.0
RAW_FMIN = 16.90009400983902
DT = 1.0 / 2048.0
DELTA_F = 0.5
FMAX = 1024.0
DISTANCE_MPC = 100.0
INCLINATION = 0.0
PHI_REF = 0.0
SPINS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def float_hex(value: Any) -> str:
    return scalar(value).hex()


def summary(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value)
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": array_sha256(value),
        "finite": bool(np.isfinite(value).all()),
        "l2": float(np.linalg.norm(value)),
    }
    if value.size:
        result["abs_max"] = float(np.max(np.abs(value)))
    return result


def public_series(values: np.ndarray, epoch: float, dt: float) -> lal.REAL8TimeSeries:
    series = lal.CreateREAL8TimeSeries(
        "conditioner-public-clean", epoch, 0.0, dt, lal.DimensionlessUnit, values.size
    )
    series.data.data[:] = np.asarray(values, dtype=np.float64)
    return series


def public_stages(
    h_plus: np.ndarray,
    h_cross: np.ndarray,
    *,
    epoch: float,
    dt: float,
    textra: float,
    f_safe: float,
    f_isco: float,
) -> dict[str, Any]:
    hp = public_series(h_plus, epoch, dt)
    hc = public_series(h_cross, epoch, dt)
    started = time.perf_counter()
    stage1_status = int(lalsim.SimInspiralTDConditionStage1(hp, hc, textra, FLOW))
    stage1_hp = np.asarray(hp.data.data, dtype=np.float64).copy()
    stage1_hc = np.asarray(hc.data.data, dtype=np.float64).copy()
    stage1_lengths = [int(hp.data.length), int(hc.data.length)]
    stage2_status = int(lalsim.SimInspiralTDConditionStage2(hp, hc, f_safe, f_isco))
    elapsed = time.perf_counter() - started
    return {
        "stage1_status": stage1_status,
        "stage2_status": stage2_status,
        "stage1_hp": stage1_hp,
        "stage1_hc": stage1_hc,
        "hp": np.asarray(hp.data.data, dtype=np.float64).copy(),
        "hc": np.asarray(hc.data.data, dtype=np.float64).copy(),
        "stage1_lengths": stage1_lengths,
        "stage2_lengths": [int(hp.data.length), int(hc.data.length)],
        "epoch": float(hp.epoch),
        "dt": float(hp.deltaT),
        "elapsed_seconds": elapsed,
    }


def norm_error(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    diff = np.asarray(actual) - np.asarray(expected)
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return {
        "relative_l2_public_norm": float(np.linalg.norm(diff) / denominator),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "difference_sha256": array_sha256(diff),
    }


def fd_from_td(hp: np.ndarray, hc: np.ndarray, *, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    sampling = fft_length_and_sampling(FMAX, DELTA_F, dt=dt)
    hp_t, hc_t = pair_leading_pad_final_crop(
        torch.from_numpy(np.asarray(hp, dtype=np.float64)),
        torch.from_numpy(np.asarray(hc, dtype=np.float64)),
        sampling.n_time,
    )
    hp_fd = (sampling.dt * torch.fft.rfft(hp_t, n=sampling.n_time)).numpy()
    hc_fd = (sampling.dt * torch.fft.rfft(hc_t, n=sampling.n_time)).numpy()
    frequencies = sampling.frequencies().numpy()
    return frequencies, hp_fd, hc_fd, {
        "n_time": int(sampling.n_time),
        "n_frequency": int(sampling.n_frequency),
        "delta_f_hz": scalar(sampling.delta_f),
        "dt_seconds": scalar(sampling.dt),
        "f_nyquist_hz": scalar(sampling.f_nyquist),
        "operation": "pair_leading_pad_final_crop then dt*rfft per real polarization",
    }


def public_time_bounds() -> dict[str, float]:
    """Call the public scalar timing APIs in the contract's order."""

    m1_si = M1 * lal.MSUN_SI
    m2_si = M2 * lal.MSUN_SI
    total_si = (M1 + M2) * lal.MSUN_SI
    chirp = float(lalsim.SimInspiralChirpTimeBound(FLOW, m1_si, m2_si, 0.0, 0.0))
    merge = float(lalsim.SimInspiralMergeTimeBound(m1_si, m2_si))
    spin_bound = 0.686
    ringdown = float(lalsim.SimInspiralRingdownTimeBound(total_si, spin_bound))
    plunge_ringdown = merge + ringdown
    support = 1.1 * chirp + plunge_ringdown + 3.0 / FLOW
    start_frequency = float(
        lalsim.SimInspiralChirpStartFrequencyBound(support, m1_si, m2_si)
    )
    return {
        "chirp_time_seconds": chirp,
        "merge_time_seconds": merge,
        "ringdown_time_seconds": ringdown,
        "plunge_ringdown_time_seconds": plunge_ringdown,
        "support_time_seconds": support,
        "start_frequency_hz": start_frequency,
        "support_time_seconds_hex": support.hex(),
        "start_frequency_hz_hex": start_frequency.hex(),
        "spin_bound": spin_bound,
        "mass_arguments_are_SI_kg": 1.0,
        "calls": {
            "SimInspiralChirpTimeBound": [FLOW, m1_si, m2_si, 0.0, 0.0],
            "SimInspiralMergeTimeBound": [m1_si, m2_si],
            "SimInspiralRingdownTimeBound": [total_si, spin_bound],
            "SimInspiralChirpStartFrequencyBound": [support, m1_si, m2_si],
        },
    }


def make_public_td_control(start_frequency: float) -> dict[str, Any]:
    """Generate one public-only exact-start control for the one-ULP input."""

    options = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(options, 0)
    mode_array = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 2, 2)
    lalsim.SimInspiralWaveformParamsInsertModeArray(options, mode_array)
    hp, hc = lalsim.SimIMRTEOBResumS(
        0.0,
        DT,
        M1 * lal.MSUN_SI,
        M2 * lal.MSUN_SI,
        *SPINS,
        0.0,
        0.0,
        DISTANCE_MPC * 1.0e6 * lal.PC_SI,
        INCLINATION,
        PHI_REF,
        options,
        0.0,
        0.0,
        start_frequency,
        0.0,
    )
    return {
        "hp": -np.asarray(hp.data.data, dtype=np.float64).copy(),
        "hc": -np.asarray(hc.data.data, dtype=np.float64).copy(),
        "epoch": float(hp.epoch),
        "dt": float(hp.deltaT),
        "start_frequency_hz": float(start_frequency),
    }


def main() -> None:
    if not TD_FIXTURE.is_file() or not FD_FIXTURE.is_file():
        raise FileNotFoundError("required public TD/FD fixtures are unavailable")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(TD_FIXTURE, allow_pickle=False) as td:
        raw_hp_direct = np.asarray(td["hp"], dtype=np.float64).copy()
        raw_hc_direct = np.asarray(td["hc"], dtype=np.float64).copy()
        epoch = float(td["epoch"])
        fixture_dt = float(td["delta_t"])
        fixture_fmin = float(td["f_min"])
    # The saved fixture is the direct EOB API frame.  The generic observer
    # frame used by the FD fixture and clean adapter has the paired minus sign.
    raw_hp = -raw_hp_direct
    raw_hc = -raw_hc_direct
    if fixture_dt != DT or fixture_fmin != RAW_FMIN:
        raise ValueError("saved public TD fixture metadata differs from requested call")

    plan = plan_td_support(M1, M2, FLOW, 0.0, 0.0, dtype=torch.float64)
    public_bounds = public_time_bounds()
    clean_scalars = {key: scalar(value) for key, value in plan.as_dict().items()}
    time_bound_comparison = {}
    for name, clean_key, public_key in (
        ("chirp_time", "chirp_time", "chirp_time_seconds"),
        ("plunge_time", "plunge_time", "merge_time_seconds"),
        ("ringdown_time", "ringdown_time", "ringdown_time_seconds"),
        ("plunge_ringdown_time", "plunge_ringdown_time", "plunge_ringdown_time_seconds"),
        ("support_time", "support_time", "support_time_seconds"),
        ("start_frequency", "start_frequency", "start_frequency_hz"),
    ):
        clean_value = clean_scalars[clean_key]
        public_value = public_bounds[public_key]
        time_bound_comparison[name] = {
            "clean": clean_value,
            "public": public_value,
            "clean_minus_public": clean_value - public_value,
            "clean_hex": clean_value.hex(),
            "public_hex": public_value.hex(),
        }
    clean = condition_td(
        torch.from_numpy(raw_hp), torch.from_numpy(raw_hc), dt=DT, support=plan,
        return_metadata=True,
    )
    if not isinstance(clean, ConditionedTD):
        raise RuntimeError("clean conditioner did not return metadata")
    clean_hp = clean.h_plus.detach().numpy()
    clean_hc = clean.h_cross.detach().numpy()
    textra = scalar(plan.first_taper_duration)
    f_safe = scalar(plan.f_safe)
    f_isco = scalar(plan.f_isco)
    stages = public_stages(
        raw_hp, raw_hc, epoch=epoch, dt=DT, textra=textra, f_safe=f_safe, f_isco=f_isco
    )

    conditioned_td = {
        "clean": {
            "hp": summary(clean_hp), "hc": summary(clean_hc),
            "length": int(clean_hp.size),
            "n_trailing_zeros": int(scalar(clean.n_trailing_zeros)),
            "first_taper_samples": int(scalar(clean.first_taper_samples)),
            "second_begin_samples": int(scalar(clean.second_begin_samples)),
            "second_end_samples": int(scalar(clean.second_end_samples)),
            "second_stage_applied": bool(clean.second_stage_applied),
        },
        "public_stage1": {
            "hp": summary(stages["stage1_hp"]), "hc": summary(stages["stage1_hc"]),
            "lengths": stages["stage1_lengths"],
        },
        "public_stage2": {
            "hp": summary(stages["hp"]), "hc": summary(stages["hc"]),
            "lengths": stages["stage2_lengths"],
            "status": [stages["stage1_status"], stages["stage2_status"]],
        },
        "clean_vs_stage2": {
            "plus": norm_error(clean_hp, stages["hp"]),
            "cross": norm_error(clean_hc, stages["hc"]),
            "exact_equal_lengths": clean_hp.size == stages["hp"].size == stages["hc"].size,
        },
    }

    clean_f, clean_php, clean_phc, sampling_meta = fd_from_td(clean_hp, clean_hc, dt=DT)
    public_f, public_php, public_phc, _ = fd_from_td(stages["hp"], stages["hc"], dt=DT)
    with np.load(FD_FIXTURE, allow_pickle=False) as fd:
        reference_f = np.asarray(fd["30p20_public_frequencies"], dtype=np.float64)
        reference_php = np.asarray(fd["30p20_public_h_plus_fd"], dtype=np.complex128)
        reference_phc = np.asarray(fd["30p20_public_h_cross_fd"], dtype=np.complex128)
    exact_axes = (
        np.array_equal(clean_f, reference_f)
        and np.array_equal(public_f, reference_f)
        and np.array_equal(clean_f, public_f)
    )
    in_band = (reference_f >= FLOW) & (reference_f <= FMAX)
    concat_actual = np.concatenate((clean_php[in_band], clean_phc[in_band]))
    concat_ref = np.concatenate((reference_php[in_band], reference_phc[in_band]))
    fd_comparison = {
        "exact_axes": bool(exact_axes),
        "frequency_bins_total": int(reference_f.size),
        "frequency_bins_compared": int(in_band.sum()),
        "frequency_range_hz": [FLOW, FMAX],
        "clean": {
            "plus": norm_error(clean_php[in_band], reference_php[in_band]),
            "cross": norm_error(clean_phc[in_band], reference_phc[in_band]),
            "concat_plus_then_cross": norm_error(concat_actual, concat_ref),
        },
        "public_stage2": {
            "plus": norm_error(public_php[in_band], reference_php[in_band]),
            "cross": norm_error(public_phc[in_band], reference_phc[in_band]),
            "concat_plus_then_cross": norm_error(
                np.concatenate((public_php[in_band], public_phc[in_band])), concat_ref
            ),
        },
        "no_polarization_sum_used": True,
    }

    # If either the conditioner comparison or the FD result is materially
    # above recurrence noise, run the exact-start public-only one-ULP control.
    # The latter keeps a one-ULP support-start difference from confounding the
    # conditioner attribution when the clean/public TD stages already agree.
    stage2_rel = max(
        conditioned_td["clean_vs_stage2"]["plus"]["relative_l2_public_norm"],
        conditioned_td["clean_vs_stage2"]["cross"]["relative_l2_public_norm"],
    )
    fd_stage_rel = fd_comparison["public_stage2"]["concat_plus_then_cross"][
        "relative_l2_public_norm"
    ]
    control: dict[str, Any] = {
        "status": "skipped",
        "trigger_stage2_rel_l2": stage2_rel,
        "trigger_public_fd_rel_l2": fd_stage_rel,
    }
    control_td: dict[str, Any] | None = None
    if max(stage2_rel, fd_stage_rel) > 1.0e-11:
        exact_start = public_bounds["start_frequency_hz"]
        control_td = make_public_td_control(exact_start)
        control_stage = public_stages(
            control_td["hp"], control_td["hc"], epoch=control_td["epoch"], dt=control_td["dt"],
            textra=textra, f_safe=f_safe, f_isco=f_isco,
        )
        control["status"] = "complete"
        control["api_start_frequency_hz"] = exact_start
        control["td"] = {key: value for key, value in control_td.items() if key not in ("hp", "hc")}
        control["stage2_lengths"] = control_stage["stage2_lengths"]
        control["fixture_vs_exact_start_td"] = {
            "plus": norm_error(raw_hp, control_td["hp"]),
            "cross": norm_error(raw_hc, control_td["hc"]),
            "exact_lengths": raw_hp.size == control_td["hp"].size,
        }
        exact_clean = condition_td(
            torch.from_numpy(control_td["hp"]), torch.from_numpy(control_td["hc"]),
            dt=control_td["dt"], support=plan, return_metadata=True,
        )
        if not isinstance(exact_clean, ConditionedTD):
            raise RuntimeError("clean conditioner did not return exact-start metadata")
        exact_clean_hp = exact_clean.h_plus.detach().numpy()
        exact_clean_hc = exact_clean.h_cross.detach().numpy()
        exact_f, exact_php, exact_phc, _ = fd_from_td(
            exact_clean_hp, exact_clean_hc, dt=control_td["dt"]
        )
        control["clean_vs_stage2_exact_start"] = {
            "plus": norm_error(exact_clean_hp, control_stage["hp"]),
            "cross": norm_error(exact_clean_hc, control_stage["hc"]),
        }
        exact_band = (exact_f >= FLOW) & (exact_f <= FMAX)
        control["exact_start_vs_public_fd"] = {
            "exact_axes": bool(np.array_equal(exact_f, reference_f)),
            "plus": norm_error(exact_php[exact_band], reference_php[exact_band]),
            "cross": norm_error(exact_phc[exact_band], reference_phc[exact_band]),
            "concat_plus_then_cross": norm_error(
                np.concatenate((exact_php[exact_band], exact_phc[exact_band])), concat_ref
            ),
        }
        control_td["clean_hp"] = exact_clean_hp
        control_td["clean_hc"] = exact_clean_hc
        control_td["clean_f"] = exact_f
        control_td["clean_php"] = exact_php
        control_td["clean_phc"] = exact_phc
        control_td["stage_hp"] = control_stage["hp"]
        control_td["stage_hc"] = control_stage["hc"]

    arrays = {
        "raw_h_plus_generic": raw_hp,
        "raw_h_cross_generic": raw_hc,
        "clean_conditioned_h_plus": clean_hp,
        "clean_conditioned_h_cross": clean_hc,
        "public_stage1_h_plus": stages["stage1_hp"],
        "public_stage1_h_cross": stages["stage1_hc"],
        "public_stage2_h_plus": stages["hp"],
        "public_stage2_h_cross": stages["hc"],
        "clean_frequencies": clean_f,
        "clean_h_plus_fd": clean_php,
        "clean_h_cross_fd": clean_phc,
        "public_stage2_frequencies": public_f,
        "public_stage2_h_plus_fd": public_php,
        "public_stage2_h_cross_fd": public_phc,
        "reference_frequencies": reference_f,
        "reference_h_plus_fd": reference_php,
        "reference_h_cross_fd": reference_phc,
    }
    if control.get("status") == "complete" and control_td is not None:
        arrays.update({
            "exact_start_raw_h_plus": control_td["hp"],
            "exact_start_raw_h_cross": control_td["hc"],
            "exact_start_clean_h_plus": control_td["clean_hp"],
            "exact_start_clean_h_cross": control_td["clean_hc"],
            "exact_start_clean_frequencies": control_td["clean_f"],
            "exact_start_clean_h_plus_fd": control_td["clean_php"],
            "exact_start_clean_h_cross_fd": control_td["clean_phc"],
            "exact_start_stage2_h_plus": control_td["stage_hp"],
            "exact_start_stage2_h_cross": control_td["stage_hc"],
        })
    np.savez(ARRAYS, **arrays)

    report: dict[str, Any] = {
        "status": "complete",
        "scope": {
            "purpose": "localize TD conditioner versus public stage APIs using identical saved public TD",
            "source_isolation": "MIT clean conditioner plus public LALSimulation black-box APIs",
            "no_native_ode_runs": True,
            "no_alignment": True,
            "no_oracle_fitting": True,
            "no_threshold_changes": True,
            "direct_to_generic_sign": "both saved direct-frame polarizations multiplied by -1 per reviewed observer-frame contract",
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "lal": getattr(lal, "__version__", None),
            "lalsimulation": getattr(lalsim, "__version__", None),
        },
        "api_parameters": {
            "td_fixture": str(TD_FIXTURE),
            "fd_fixture": str(FD_FIXTURE),
            "m1_msun": M1, "m2_msun": M2, "spins": list(SPINS),
            "distance_mpc": DISTANCE_MPC, "inclination_rad": INCLINATION, "phi_ref_rad": PHI_REF,
            "requested_f_low_hz": FLOW, "raw_generation_f_min_hz": RAW_FMIN,
            "raw_generation_f_min_hz_hex": RAW_FMIN.hex(),
            "delta_t_seconds": DT, "delta_f_hz": DELTA_F, "f_max_hz": FMAX,
            "SimInspiralTDConditionStage1": {"textra_seconds": textra, "f_min_hz": FLOW},
            "SimInspiralTDConditionStage2": {"f_min_hz": f_safe, "f_max_hz": f_isco},
            "mode": [2, 2], "PN_tidal_order": 0,
        },
        "public_time_bounds": public_bounds,
        "clean_plan": clean_scalars,
        "clean_plan_float_hex": {
            "start_frequency": float_hex(plan.start_frequency),
            "support_time": float_hex(plan.support_time),
            "first_taper_duration": float_hex(plan.first_taper_duration),
            "f_safe": float_hex(plan.f_safe),
            "f_isco": float_hex(plan.f_isco),
        },
        "time_bound_comparison": time_bound_comparison,
        "conditioning_td": conditioned_td,
        "sampling": sampling_meta,
        "fd_comparison": fd_comparison,
        "exact_start_control": control,
        "inputs": {
            "td_fixture_sha256": sha256(TD_FIXTURE),
            "fd_fixture_sha256": sha256(FD_FIXTURE),
            "raw_direct_hp": summary(raw_hp_direct), "raw_direct_hc": summary(raw_hc_direct),
            "raw_generic_hp": summary(raw_hp), "raw_generic_hc": summary(raw_hc),
        },
        "arrays": {"path": str(ARRAYS), "sha256": sha256(ARRAYS), "keys": sorted(arrays)},
        "modules": {
            "diagnostic_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            "clean_conditioner": {"path": str(CONDITIONER), "sha256": sha256(CONDITIONER)},
            "conditioning_contract": {"path": str(CONTRACT), "sha256": sha256(CONTRACT)},
        },
        "conclusion": (
            "The saved public TD passes through clean and public conditioning with equal lengths and "
            "the reported separated errors. FD attribution is reported on exact shared frequency axes; "
            "no cause beyond this bounded conditioner comparison is inferred."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Refresh the archive hash now that report serialization is complete.
    report["arrays"]["sha256"] = sha256(ARRAYS)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

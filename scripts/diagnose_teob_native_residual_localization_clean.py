#!/usr/bin/env python3
"""Localize raw-TD differences between the cached native replay and public TD.

This is a read-only diagnostic.  It compares samples at identical elapsed
indices, without time or phase alignment.  The public arrays are the
exact-start arrays saved by the conditioner diagnostic and already use its
reviewed direct-to-generic sign contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs/reviews/2026-09-13-parity-review"
NATIVE_PATH = REVIEW / "verification/teob-native-replay-public-clean/replay-native-output.npz"
NATIVE_REPORT_PATH = REVIEW / "verification/teob-native-replay-public-clean/replay-report.json"
PUBLIC_ARRAYS_PATH = REVIEW / "verification/teob-conditioner-public-clean/arrays.npz"
PUBLIC_REPORT_PATH = REVIEW / "verification/teob-conditioner-public-clean/report.json"
OUT_DIR = REVIEW / "verification/teob-native-residual-localization-clean"
OUT_PATH = OUT_DIR / "residual-localization-report.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scalar(z: np.ndarray) -> float:
    return float(np.asarray(z).item())


def stats(native: np.ndarray, public: np.ndarray) -> dict[str, object]:
    if native.shape != public.shape or native.ndim != 1 or native.size == 0:
        raise ValueError(f"incompatible arrays: {native.shape} and {public.shape}")
    if not (np.isfinite(native).all() and np.isfinite(public).all()):
        raise ValueError("nonfinite TD array")
    d = native - public
    abs_d = np.abs(d)
    max_i = int(np.argmax(abs_d))
    pnorm = float(np.linalg.norm(public))
    if pnorm == 0.0:
        raise ValueError("zero public denominator")
    return {
        "samples": int(native.size),
        "native_norm": float(np.linalg.norm(native)),
        "public_norm": pnorm,
        "residual_norm": float(np.linalg.norm(d)),
        "relative_l2_public_norm": float(np.linalg.norm(d) / pnorm),
        "max_abs": float(abs_d[max_i]),
        "max_abs_index": max_i,
        "first_nonzero_index": int(np.flatnonzero(d != 0.0)[0]) if np.any(d != 0.0) else None,
        "first_sample_difference": float(d[0]),
    }


def segment_stats(
    name: str,
    mask: np.ndarray,
    native_plus: np.ndarray,
    native_cross: np.ndarray,
    public_plus: np.ndarray,
    public_cross: np.ndarray,
    td_time: np.ndarray,
    total_mass_seconds: float,
) -> dict[str, object]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(f"empty segment: {name}")
    lo, hi = int(indices[0]), int(indices[-1])
    p = stats(native_plus[mask], public_plus[mask])
    c = stats(native_cross[mask], public_cross[mask])
    # Concatenating polarizations preserves separate denominators while
    # providing one joint TD diagnostic.  This is unrelated to FD hp+i hc.
    joint = stats(
        np.concatenate((native_plus[mask], native_cross[mask])),
        np.concatenate((public_plus[mask], public_cross[mask])),
    )
    return {
        "name": name,
        "indices": [lo, hi],
        "samples": int(indices.size),
        "td_time_dimensionless": [float(td_time[lo]), float(td_time[hi])],
        "elapsed_time_seconds": [
            float(td_time[lo] * total_mass_seconds),
            float(td_time[hi] * total_mass_seconds),
        ],
        "h_plus": p,
        "h_cross": c,
        "concat_plus_then_cross": joint,
        "global_max_abs_indices": {
            "h_plus": lo + int(p["max_abs_index"]),
            "h_cross": lo + int(c["max_abs_index"]),
            "concat_plus_then_cross": lo + int(joint["max_abs_index"] % indices.size),
        },
        "global_first_nonzero_indices": {
            "h_plus": None if p["first_nonzero_index"] is None else lo + int(p["first_nonzero_index"]),
            "h_cross": None if c["first_nonzero_index"] is None else lo + int(c["first_nonzero_index"]),
        },
    }


def threshold_markers(
    native_plus: np.ndarray,
    native_cross: np.ndarray,
    public_plus: np.ndarray,
    public_cross: np.ndarray,
    td_time: np.ndarray,
    total_mass_seconds: float,
) -> dict[str, object]:
    # These fixed markers describe growth of the observed residual; they do
    # not select a model, alter a threshold, or perform an optimization.
    diff = np.sqrt((native_plus - public_plus) ** 2 + (native_cross - public_cross) ** 2)
    denom = np.sqrt(public_plus**2 + public_cross**2)
    ratio = np.divide(diff, denom, out=np.full_like(diff, np.nan), where=denom != 0.0)
    result: dict[str, object] = {}
    finite = np.flatnonzero(np.isfinite(ratio))
    if finite.size == 0:
        raise ValueError("no finite pointwise TD envelope")
    first = int(finite[0])
    peak = int(finite[np.argmax(ratio[finite])])
    result["first_finite"] = {
        "index": first,
        "td_time_dimensionless": float(td_time[first]),
        "elapsed_time_seconds": float(td_time[first] * total_mass_seconds),
        "ratio": float(ratio[first]),
    }
    result["maximum"] = {
        "index": peak,
        "td_time_dimensionless": float(td_time[peak]),
        "elapsed_time_seconds": float(td_time[peak] * total_mass_seconds),
        "ratio": float(ratio[peak]),
    }
    for level in (1e-6, 1e-5, 1e-4, 1e-3):
        hits = np.flatnonzero(np.isfinite(ratio) & (ratio >= level))
        if hits.size:
            i = int(hits[0])
            result[f"relative_envelope_ge_{level:g}"] = {
                "index": i,
                "td_time_dimensionless": float(td_time[i]),
                "elapsed_time_seconds": float(td_time[i] * total_mass_seconds),
                "ratio": float(ratio[i]),
            }
        else:
            result[f"relative_envelope_ge_{level:g}"] = None
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-cache", type=Path, default=NATIVE_PATH,
                        help="native replay .npz (default: reviewed baseline cache)")
    parser.add_argument("--native-report", type=Path, default=NATIVE_REPORT_PATH,
                        help="native replay report.json (default: reviewed baseline report)")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR,
                        help="directory for residual-localization-report.json (default: reviewed directory)")
    args = parser.parse_args(argv)
    native_path = args.native_cache
    native_report_path = args.native_report
    out_dir = args.output_dir
    out_path = out_dir / "residual-localization-report.json"

    for path in (native_path, native_report_path, PUBLIC_ARRAYS_PATH, PUBLIC_REPORT_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    native = np.load(native_path, allow_pickle=False)
    public_arrays = np.load(PUBLIC_ARRAYS_PATH, allow_pickle=False)
    native_plus = np.asarray(native["h_plus_td"])
    native_cross = np.asarray(native["h_cross_td"])
    public_plus = np.asarray(public_arrays["exact_start_raw_h_plus"])
    public_cross = np.asarray(public_arrays["exact_start_raw_h_cross"])
    td_time = np.asarray(native["td_time"])
    if native_plus.shape != native_cross.shape or native_plus.shape != public_plus.shape or native_plus.shape != public_cross.shape:
        raise ValueError("native/public raw TD lengths do not match exactly")
    if td_time.shape != native_plus.shape:
        raise ValueError("native td_time length does not match raw TD")

    native_report = json.loads(native_report_path.read_text())
    public_report = json.loads(PUBLIC_REPORT_PATH.read_text())
    if native_report.get("source_hashes_unchanged") is not True:
        raise ValueError("native replay report does not certify unchanged source hashes")
    total_mass = scalar(native["total_mass_seconds"])
    dt = scalar(native["sampling_dt_seconds"])
    native_epoch = scalar(native["physical_epoch_seconds"])
    public_td_meta = public_report["exact_start_control"]["td"]
    public_dt = float(public_td_meta["dt"])
    if dt != public_dt:
        raise ValueError(f"native/public sampling dt mismatch: {dt!r} != {public_dt!r}")
    public_epoch = float(public_td_meta["epoch"])
    pa_end = scalar(native["pa_time"][-1])
    ode_end = scalar(native["nqc_auxiliary_time"][0])
    attachment = scalar(native["event_attachment_time"])

    public_start_frequency = float(public_report["exact_start_control"]["api_start_frequency_hz"])
    native_start_frequency = scalar(native["support_start_frequency"])
    if native_start_frequency != public_start_frequency:
        raise ValueError(
            f"native/public exact-start frequency mismatch: {native_start_frequency!r} != {public_start_frequency!r}"
        )
    if td_time[0] != 0.0:
        raise ValueError(f"native elapsed time does not start at zero: {td_time[0]!r}")
    expected_step = dt / total_mass
    if not np.allclose(np.diff(td_time), expected_step, rtol=0.0, atol=1e-12):
        raise ValueError("native elapsed-time spacing is inconsistent with dt / total mass")

    # Segment boundaries use recorded event/support arrays and elapsed native
    # samples.  Every sample belongs to one of these windows by construction.
    masks = {
        "pa": td_time <= pa_end,
        "ode_inspiral": (td_time > pa_end) & (td_time < ode_end),
        "nqc_auxiliary_to_attachment": (td_time >= ode_end) & (td_time <= attachment),
        "tail": td_time > attachment,
    }
    covered = np.logical_or.reduce(tuple(masks.values()))
    if not covered.all():
        raise ValueError("segment boundary rules left native samples unassigned")

    input_paths = [native_path, native_report_path, PUBLIC_ARRAYS_PATH, PUBLIC_REPORT_PATH]
    report: dict[str, object] = {
        "status": "complete",
        "scope": {
            "purpose": "bounded raw TD localization of native replay versus exact-start public TD",
            "comparison": "same elapsed array indices; no time or phase alignment",
            "fd_parity_warning": "raw TD index comparison is diagnostic and is not strict FD parity",
            "no_alignment": True,
            "no_oracle_fitting": True,
            "no_parameter_search": True,
            "no_runtime_edits": True,
            "source_isolation": "cached native output versus cached public black-box TD output",
        },
        "inputs": {
            "files": [
                {"path": str(p), "sha256": sha256(p)} for p in input_paths
            ],
            "native_arrays": ["h_plus_td", "h_cross_td", "td_time"],
            "public_arrays": ["exact_start_raw_h_plus", "exact_start_raw_h_cross"],
        },
        "array_contract": {
            "exact_lengths": True,
            "samples": int(native_plus.size),
            "dtype_native_plus": str(native_plus.dtype),
            "dtype_native_cross": str(native_cross.dtype),
            "dtype_public_plus": str(public_plus.dtype),
            "dtype_public_cross": str(public_cross.dtype),
            "finite": bool(np.isfinite(native_plus).all() and np.isfinite(native_cross).all() and np.isfinite(public_plus).all() and np.isfinite(public_cross).all()),
            "public_sign_contract": public_report["scope"]["direct_to_generic_sign"],
            "sampling_dt_seconds": dt,
            "public_sampling_dt_seconds": public_dt,
            "exact_sampling_dt": dt == public_dt,
            "native_td_time_start_end": [float(td_time[0]), float(td_time[-1])],
        },
        "epoch_comparison": {
            "native_epoch_seconds": native_epoch,
            "public_exact_start_epoch_seconds": public_epoch,
            "public_minus_native_seconds": public_epoch - native_epoch,
            "absolute_mismatch_seconds": abs(public_epoch - native_epoch),
            "comparison_correction_applied": False,
        },
        "support_and_events": {
            "total_mass_seconds": total_mass,
            "native_support_start_frequency_hz": native_start_frequency,
            "native_support_start_frequency_hz_hex": float(native_start_frequency).hex(),
            "public_exact_start_frequency_hz": public_start_frequency,
            "native_support_time_seconds": scalar(native["support_time"]),
            "pa_end_td_time": pa_end,
            "ode_inspiral_end_td_time": ode_end,
            "event_attachment_td_time": attachment,
            "event_attachment_index_recorded": int(scalar(native["event_attachment_index"])),
            "nqc_matching_index_recorded": int(scalar(native["nqc_matching_index"])),
            "segment_rule": "PA <= pa_time[-1]; ODE > PA and < nqc_auxiliary_time[0]; NQC auxiliary-to-attachment >= auxiliary start and <= event_attachment_time; tail > event_attachment_time",
            "segment_caveat": "All stored native TD samples already include the replay's NQC/ringdown processing; the auxiliary-to-attachment interval is a recorded diagnostic window, not a claim that NQC starts at its first sample.",
            "replay_support_start_arithmetic": native_report.get("support_start_arithmetic"),
            "replay_native_start": native_report.get("native_start"),
        },
        "overall": {
            "h_plus": stats(native_plus, public_plus),
            "h_cross": stats(native_cross, public_cross),
            "concat_plus_then_cross": stats(
                np.concatenate((native_plus, native_cross)),
                np.concatenate((public_plus, public_cross)),
            ),
        },
        "segments": [
            segment_stats(name, mask, native_plus, native_cross, public_plus, public_cross, td_time, total_mass)
            for name, mask in masks.items()
        ],
        "earliest_deviation": {
            "raw_nonzero": {
                "h_plus_index": stats(native_plus, public_plus)["first_nonzero_index"],
                "h_cross_index": stats(native_cross, public_cross)["first_nonzero_index"],
                "interpretation": "Both polarizations differ at the first stored sample; this is a small deterministic raw difference, not an identified conditioner boundary.",
            },
            "fixed_relative_envelope_markers": threshold_markers(native_plus, native_cross, public_plus, public_cross, td_time, total_mass),
            "interpretation": "A discrepancy is already present within PA and grows during ODE inspiral, with larger aggregate residuals in the NQC auxiliary-to-attachment and tail windows. Cached TD outputs alone cannot distinguish PA state error, NQC correction, projection, or sampling effects; the next contract should pin those trajectory, support, sampling, and event-handoff quantities before assigning causality.",
        },
        "reproducibility": {
            "diagnostic_script": str(Path(__file__).resolve()),
            "diagnostic_script_sha256": sha256(Path(__file__).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "native_replay_report_status": native_report.get("status"),
            "public_conditioner_report_status": public_report.get("status"),
            "replay_source_hashes_unchanged": native_report.get("source_hashes_unchanged"),
            "replay_source_sha256_before": native_report.get("source_sha256_before"),
            "replay_source_sha256_after": native_report.get("source_sha256_after"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(out_path), "status": report["status"], "samples": report["array_contract"]["samples"]}, sort_keys=True))


if __name__ == "__main__":
    main()

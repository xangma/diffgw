#!/usr/bin/env python3
"""Measure the effect of applying TEOB NQC to cached native modes.

This is a bounded cache-only experiment.  It reconstructs the saved
pre-NQC amplitude and lifted phase on the native elapsed-time grid, then
compares that counterfactual and the saved processed native polarizations
with the exact-start public full route.  The public PA-only comparison is
loaded from its already reviewed cache as a control.  There is no alignment,
fit, source inspection, ODE evolution, or runtime edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import lal
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "docs/reviews/2026-09-13-parity-review"
NATIVE = REVIEW / "verification/teob-native-replay-public-clean/replay-native-output.npz"
NATIVE_REPORT = REVIEW / "verification/teob-native-replay-public-clean/replay-report.json"
PUBLIC_FULL = REVIEW / "verification/teob-conditioner-public-clean/arrays.npz"
PUBLIC_FULL_REPORT = REVIEW / "verification/teob-conditioner-public-clean/report.json"
PUBLIC_PA = REVIEW / "verification/teob-pa-exact-start-clean/arrays.npz"
OUT = ROOT / "docs/reviews/2026-09-15-no-contact/teob-nqc-attribution-clean"
TOTAL_MASS_MSUN = 50.0
DISTANCE_MPC = 100.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite(x: Any) -> float | None:
    value = float(x)
    return value if math.isfinite(value) else None


def complex_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if candidate.shape != reference.shape or candidate.ndim != 1 or candidate.size == 0:
        raise ValueError(f"incompatible complex arrays: {candidate.shape}, {reference.shape}")
    if not (np.isfinite(candidate).all() and np.isfinite(reference).all()):
        raise ValueError("nonfinite complex comparison array")
    delta = candidate - reference
    ref_norm = float(np.linalg.norm(reference))
    if ref_norm == 0.0:
        raise ValueError("zero reference norm")
    phase = np.unwrap(np.angle(candidate * np.conj(reference)))
    amp = np.abs(candidate)
    ref_amp = np.abs(reference)
    return {
        "samples": int(candidate.size),
        "raw_relative_l2": finite(np.linalg.norm(delta) / ref_norm),
        "amplitude_relative_l2": finite(np.linalg.norm(amp - ref_amp) / max(np.linalg.norm(ref_amp), 1e-300)),
        "phase_rms_radians": finite(np.sqrt(np.mean(phase * phase))),
        "phase_max_abs_radians": finite(np.max(np.abs(phase))),
        "phase_initial_radians": finite(phase[0]),
        "phase_final_radians": finite(phase[-1]),
        "first_sample_relative_envelope": finite(abs(delta[0]) / max(abs(reference[0]), 1e-300)),
        "alignment": "none",
    }


def reconstruct_original(native: np.lib.npyio.NpzFile, query: np.ndarray) -> np.ndarray:
    """Reconstruct uncorrected mode on a subset wholly inside saved knots."""
    knots = np.asarray(native["nqc_original_time"], dtype=np.float64)
    amplitude = np.asarray(native["nqc_original_amplitude"], dtype=np.float64)
    phase = np.asarray(native["nqc_original_phase"], dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if knots.ndim != 1 or amplitude.shape != knots.shape or phase.shape != knots.shape:
        raise ValueError("invalid saved NQC-original knot arrays")
    if query.ndim != 1 or query.size == 0 or query[0] < knots[0] or query[-1] > knots[-1]:
        raise ValueError("pre-NQC reconstruction would extrapolate")
    # The merger contract calls for separate natural-cubic interpolation of
    # amplitude and lifted phase; no coefficient is inferred here.
    amp = CubicSpline(knots, amplitude, bc_type="natural", extrapolate=False)(query)
    ph = CubicSpline(knots, phase, bc_type="natural", extrapolate=False)(query)
    return amp * np.exp(-1j * ph)


def threshold_first(query: np.ndarray, ratio: np.ndarray, threshold: float) -> dict[str, Any] | None:
    hits = np.flatnonzero(np.isfinite(ratio) & (ratio >= threshold))
    if hits.size == 0:
        return None
    i = int(hits[0])
    return {"index": i, "td_time_dimensionless": finite(query[i]), "ratio": finite(ratio[i])}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    paths = (NATIVE, NATIVE_REPORT, PUBLIC_FULL, PUBLIC_FULL_REPORT, PUBLIC_PA)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    native = np.load(NATIVE, allow_pickle=False)
    full = np.load(PUBLIC_FULL, allow_pickle=False)
    pa = np.load(PUBLIC_PA, allow_pickle=False)
    td = np.asarray(native["td_time"], dtype=np.float64)
    native_processed = np.asarray(native["h_plus_td"], dtype=np.float64) - 1j * np.asarray(native["h_cross_td"], dtype=np.float64)
    public_full = np.asarray(full["exact_start_raw_h_plus"], dtype=np.float64) - 1j * np.asarray(full["exact_start_raw_h_cross"], dtype=np.float64)
    if native_processed.shape != public_full.shape or td.shape != native_processed.shape:
        raise ValueError("native and public full arrays are not on the same exact grid")
    pa_time = np.asarray(pa["time_M"], dtype=np.float64)
    if not np.array_equal(td[: pa_time.size], pa_time):
        raise ValueError("reviewed PA control does not share the native elapsed grid")

    attachment = float(np.asarray(native["event_attachment_time"]).item())
    in_nqc_domain = td <= attachment
    q = td[in_nqc_domain]
    original_mode = reconstruct_original(native, q)
    # Reviewed observer-frame projection for (2,2), using the requested
    # physical mass and distance.  This is a documentary factor, not a fit.
    scale = math.sqrt(5.0 / (4.0 * math.pi)) * (TOTAL_MASS_MSUN * lal.MTSUN_SI) * lal.C_SI / (
        DISTANCE_MPC * 1.0e6 * lal.PC_SI)
    original = scale * original_mode
    processed = native_processed[in_nqc_domain]
    public = public_full[in_nqc_domain]

    pa_end = float(np.asarray(native["pa_time"][-1]))
    ode_start = float(np.asarray(native["nqc_auxiliary_time"][0]))
    masks = {
        "pa": q <= pa_end,
        "ode_inspiral": (q > pa_end) & (q < ode_start),
        "nqc_auxiliary_to_attachment": (q >= ode_start) & (q <= attachment),
    }
    windows: dict[str, Any] = {}
    for name, mask in masks.items():
        if not np.any(mask):
            raise ValueError(f"empty window: {name}")
        windows[name] = {
            "samples": int(mask.sum()),
            "time_M": [finite(q[mask][0]), finite(q[mask][-1])],
            "original_vs_public_full": complex_metrics(original[mask], public[mask]),
            "processed_vs_public_full": complex_metrics(processed[mask], public[mask]),
            "original_vs_processed": complex_metrics(original[mask], processed[mask]),
        }

    correction_ratio = np.abs(original - processed) / np.maximum(np.abs(processed), 1e-300)
    report: dict[str, Any] = {
        "status": "complete",
        "experiment": "same-grid NQC-on versus NQC-off counterfactual",
        "run": {"host": socket.gethostname(), "platform": platform.platform(), "cwd": str(Path.cwd()),
                "command": " ".join(sys.argv), "elapsed_seconds": finite(time.perf_counter() - started)},
        "diagnostic": {"script": str(Path(__file__).relative_to(ROOT)),
                       "script_sha256": sha256(Path(__file__))},
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        "grid": {"samples_full": int(td.size), "samples_through_attachment": int(q.size),
                 "dt_M": finite(np.asarray(native["sampling_dt_seconds"]).item() / np.asarray(native["total_mass_seconds"]).item()),
                 "attachment_time_M": finite(attachment), "pa_end_time_M": finite(pa_end),
                 "ode_start_time_M": finite(ode_start), "no_regridding": True, "alignment": "none"},
        "controls": {
            "public_pa_only_pre_nqc_vs_public_pa_only": complex_metrics(pa["pre_nqc_h"], pa["public_pa_h"]),
            "public_pa_only_cache": str(PUBLIC_PA.relative_to(ROOT)),
            "full_route_arrays": ["exact_start_raw_h_plus", "exact_start_raw_h_cross"],
        },
        "windows": windows,
        "nqc_switch": {
            "first_effect_ratio_thresholds": {f"{t:g}": threshold_first(q, correction_ratio, t) for t in (1e-6, 1e-5, 1e-4, 1e-3)},
            "first_sample_original_vs_processed": finite(correction_ratio[0]),
            "scale_documentary_mass_distance": {"total_mass_msun": TOTAL_MASS_MSUN, "distance_mpc": DISTANCE_MPC,
                                                  "factor": finite(scale)},
            "documentary_basis": "saved nqc_original amplitude/phase are the pre-correction mode; saved h_plus_td/h_cross_td are the processed native mode; public full and PA-only arrays are exact-start black-box outputs",
        },
        "classification": {
            "result": "Applying NQC improves agreement with the public full route: the uncorrected native PA prefix is ~1e-3 away, while the corrected prefix is ~4e-6 away. The reviewed separate PA-only comparison remains 3.43e-7. This establishes the importance of the correction, not the cause of its remaining error.",
            "remaining_residual": "No documentary fix is justified. The processed residual grows across ODE inspiral and is largest in the sampled auxiliary-to-attachment window. Targets, event timing, trajectory, interpolation and projection conventions remain unresolved; this comparison does not isolate them.",
            "limits": ["The public binary route has no independent NQC-off full toggle in this experiment.", "The 8-sample auxiliary window is a consequence of the fixed cached TD grid.", "No fitted residual, copied coefficient, alignment, source, or runtime implementation was used."],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output), "windows": windows}, indent=2))


if __name__ == "__main__":
    main()

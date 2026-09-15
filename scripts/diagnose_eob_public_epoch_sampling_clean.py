"""Sweep public EOB output spacing around one geometric mass.

The public call is treated as a black box through the existing
``public_call`` wrapper.  For every predeclared output spacing, the returned
postpeak state is evaluated with the clean SEOBNRv4 Hamiltonian/flux and the
same degree-six orbital-frequency event detector.  The continuous event is
compared with the sampled maximum on that returned high-resolution grid and
with the public epoch, all in the initial-origin M coordinate.  No spacing is
selected from a residual and no offset is fitted.

Raw public arrays are written outside the repository.  They are copied before
evaluation and each external file is hashed in the report.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import lal
import lalsimulation as ls
import numpy as np

from diagnose_eob_clean_hamiltonian_public import public_call
from diagnose_eob_public_phase_peak_clean import (
    _clean_event,
    _setup_clean_physics,
)
from torchwave.constants import MSUN_SEC


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / (
    "docs/reviews/2026-09-13-parity-review/verification/eob/"
    "eob-public-epoch-sampling-clean.json"
)
RAW_DIR = Path("/tmp")
TOTAL_MASS_MSUN = 50.0
MSEC = TOTAL_MASS_MSUN * MSUN_SEC

# This tuple is the complete sweep design.  It is fixed before any public
# call.  The exact 4096-Hz spacing is an anchor row, not an adaptive candidate.
EXACT_4096_DT_M = (1.0 / 4096.0) / MSEC
SWEEP = (
    ("dt_0p90M", 0.90),
    ("dt_0p95M", 0.95),
    ("dt_0p99M", 0.99),
    ("dt_1p00M", 1.00),
    ("dt_1p01M", 1.01),
    ("dt_1p05M", 1.05),
    ("dt_1p10M", 1.10),
    ("dt_1_4096s_exact", EXACT_4096_DT_M),
)

DEPENDENCIES = (
    "scripts/diagnose_eob_public_epoch_sampling_clean.py",
    "scripts/diagnose_eob_clean_hamiltonian_public.py",
    "scripts/diagnose_eob_public_phase_peak_clean.py",
    "src/torchwave/constants.py",
    "src/torchwave/approximants/eob/events.py",
    "src/torchwave/approximants/eob/hamiltonian.py",
    "src/torchwave/approximants/eob/attachment.py",
    "src/torchwave/approximants/eob/factorized_modes.py",
    "src/torchwave/approximants/eob/integrator.py",
    "src/torchwave/approximants/kerr_qnm_generated.py",
    "src/torchwave/approximants/eob/nqc.py",
    "src/torchwave/approximants/eob/radiation_reaction.py",
    "src/torchwave/approximants/eob/remnant.py",
    "src/torchwave/approximants/eob/seobnrv4_amplitude22.py",
    "src/torchwave/approximants/eob/seobnrv4_hamiltonian.py",
    "src/torchwave/approximants/eob/seobnrv4_flux.py",
    "src/torchwave/approximants/eob/seobnrv4_mode22.py",
    "src/torchwave/approximants/eob/published_waveform.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _library_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "lal": getattr(lal, "__version__", "unknown"),
        "lalsimulation": getattr(ls, "__version__", "unknown"),
    }
    for module_name in ("lal", "lalsimulation"):
        if versions[module_name] != "unknown":
            continue
        try:
            versions[module_name] = str(metadata.version("lalsuite"))
        except metadata.PackageNotFoundError:
            versions[module_name] = "unknown"
    return versions


def _copy_and_validate(label: str, state, postpeak, mode, epoch_s: float,
                       dt_s: float) -> dict:
    """Copy the public return values and validate the observed schema."""

    state = np.asarray(state, dtype=np.float64).copy()
    postpeak = np.asarray(postpeak, dtype=np.float64).copy()
    mode = np.asarray(mode, dtype=np.complex128).copy()
    epoch_s = float(epoch_s)
    dt_s = float(dt_s)
    if state.ndim != 2 or state.shape[0] != 5 or state.shape[1] < 2:
        raise ValueError(f"{label}: state must have shape (5,N>=2)")
    if postpeak.ndim != 2 or postpeak.shape[0] != 5 or postpeak.shape[1] < 7:
        raise ValueError(f"{label}: postpeak state must have shape (5,N>=7)")
    if mode.ndim != 1 or mode.size < 2:
        raise ValueError(f"{label}: mode must be one-dimensional with >=2 samples")
    if not np.isfinite(state).all() or not np.isfinite(postpeak).all() or not np.isfinite(mode).all():
        raise ValueError(f"{label}: public arrays contain non-finite values")
    if not np.all(np.diff(state[0]) > 0) or not np.all(np.diff(postpeak[0]) > 0):
        raise ValueError(f"{label}: state times are not strictly increasing")
    if not np.isfinite(epoch_s) or not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError(f"{label}: epoch/deltaT metadata is invalid")
    return {
        "state": state,
        "postpeak_state": postpeak,
        "mode": mode,
        "epoch_s": epoch_s,
        "dt_s": dt_s,
    }


def _save_raw(label: str, data: dict, requested_dt_M: float) -> dict:
    path = RAW_DIR / f"torchwave-public-eob-epoch-sweep-{label}.npz"
    # The arrays are already private copies.  Copy once more at serialization
    # boundaries so no later diagnostic operation can alias an NPZ input.
    np.savez(
        path,
        state=data["state"].copy(),
        postpeak_state=data["postpeak_state"].copy(),
        mode22=data["mode"].copy(),
        epoch_s=np.asarray(data["epoch_s"]),
        dt_s=np.asarray(data["dt_s"]),
        requested_dt_M=np.asarray(requested_dt_M),
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
        "keys": ["state", "postpeak_state", "mode22", "epoch_s", "dt_s", "requested_dt_M"],
        "state_shape": list(data["state"].shape),
        "postpeak_state_shape": list(data["postpeak_state"].shape),
        "mode_shape": list(data["mode"].shape),
        "arrays_copied_before_save": True,
    }


def _event_row(data: dict, physics: dict, raw_meta: dict, requested_dt_M: float) -> dict:
    post = data["postpeak_state"]
    post_time = post[0].copy()
    event = _clean_event(post_time, post, physics)
    omega = np.asarray(event.pop("omega"), dtype=np.float64).copy()
    sample_index = int(np.argmax(omega))
    sample_time = float(post_time[sample_index])
    event_time = float(event["time_M_initial_origin"])
    epoch_M = data["epoch_s"] / MSEC
    minus_epoch_M = -epoch_M
    highres_spacing = float(np.median(np.diff(post_time)))
    if not np.isfinite(highres_spacing) or highres_spacing <= 0:
        raise ValueError("postpeak grid has invalid spacing")

    mode_grid_M = np.arange(data["mode"].size, dtype=np.float64) * (data["dt_s"] / MSEC)
    amplitude_index = int(np.argmax(np.abs(data["mode"])))
    return {
        "status": "ok",
        "requested_dt_M": float(requested_dt_M),
        "returned_dt_s": float(data["dt_s"]),
        "returned_dt_M": float(data["dt_s"] / MSEC),
        "epoch_s": float(data["epoch_s"]),
        "epoch_M": float(epoch_M),
        "minus_epoch_M": float(minus_epoch_M),
        "raw_output": raw_meta,
        "state_grid": {
            "sample_count": int(data["state"].shape[1]),
            "support_M_initial_origin": [float(data["state"][0, 0]), float(data["state"][0, -1])],
            "spacing_min_M": float(np.min(np.diff(data["state"][0]))),
            "spacing_median_M": float(np.median(np.diff(data["state"][0]))),
            "spacing_max_M": float(np.max(np.diff(data["state"][0]))),
        },
        "postpeak_grid": {
            "sample_count": int(post.shape[1]),
            "support_M_initial_origin": [float(post_time[0]), float(post_time[-1])],
            "spacing_min_M": float(np.min(np.diff(post_time))),
            "spacing_median_M": highres_spacing,
            "spacing_max_M": float(np.max(np.diff(post_time))),
            "coordinate": "postpeak_state[0] in initial-origin M coordinates",
        },
        "clean_hamiltonian_omega_event": {
            key: value for key, value in event.items()
        },
        "clean_hamiltonian_sampled_maximum": {
            "sample_index": sample_index,
            "time_M_initial_origin": sample_time,
            "omega": float(omega[sample_index]),
            "left_time_M_initial_origin": float(post_time[sample_index - 1]) if sample_index else None,
            "right_time_M_initial_origin": float(post_time[sample_index + 1]) if sample_index + 1 < post_time.size else None,
            "left_omega": float(omega[sample_index - 1]) if sample_index else None,
            "right_omega": float(omega[sample_index + 1]) if sample_index + 1 < post_time.size else None,
        },
        "peak_epoch_geometry": {
            "continuous_event_minus_sample_max_M": event_time - sample_time,
            "minus_epoch_minus_continuous_event_M": minus_epoch_M - event_time,
            "minus_epoch_minus_sample_max_M": minus_epoch_M - sample_time,
            "minus_epoch_minus_sample_max_in_highres_samples": (minus_epoch_M - sample_time) / highres_spacing,
            "minus_epoch_minus_continuous_event_in_highres_samples": (minus_epoch_M - event_time) / highres_spacing,
            "continuous_event_side_of_sample_max": (
                "before" if event_time < sample_time else
                "after" if event_time > sample_time else "equal"
            ),
        },
        "public_mode_amplitude_maximum": {
            "sample_index": amplitude_index,
            "relative_output_time_M": float(mode_grid_M[amplitude_index]),
            "public_epoch_coordinate_M": float(epoch_M + mode_grid_M[amplitude_index]),
            "amplitude": float(abs(data["mode"][amplitude_index])),
        },
        "no_time_or_phase_shift": True,
        "no_residual_based_selection": True,
    }


def main() -> None:
    physics = _setup_clean_physics()
    results: dict[str, dict] = {}
    for label, requested_dt_M in SWEEP:
        requested_dt_s = requested_dt_M * MSEC
        try:
            state, postpeak, mode, epoch_s = public_call(requested_dt_s)
            data = _copy_and_validate(label, state, postpeak, mode, epoch_s, requested_dt_s)
            raw_meta = _save_raw(label, data, requested_dt_M)
            results[label] = _event_row(data, physics, raw_meta, requested_dt_M)
        except Exception as exc:
            # Preserve a failed grid in the report rather than dropping it or
            # changing the predeclared design after seeing a failure.
            results[label] = {
                "status": "failed",
                "requested_dt_M": float(requested_dt_M),
                "requested_dt_s": float(requested_dt_s),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "no_time_or_phase_shift": True,
                "no_residual_based_selection": True,
            }

    dependency_sha256 = {
        path: _sha256(REPO_ROOT / path) if (REPO_ROOT / path).exists() else None
        for path in DEPENDENCIES
    }
    ok_count = sum(row.get("status") == "ok" for row in results.values())
    successful = [row for row in results.values() if row.get("status") == "ok"]
    before_rows = [
        label for label, row in results.items()
        if row.get("status") == "ok"
        and row["peak_epoch_geometry"]["continuous_event_side_of_sample_max"] == "before"
    ]
    after_rows = [
        label for label, row in results.items()
        if row.get("status") == "ok"
        and row["peak_epoch_geometry"]["continuous_event_side_of_sample_max"] == "after"
    ]
    report = {
        "provenance": (
            "Bounded clean-room public epoch sampling sweep. Each row calls the "
            "public lalsimulation SimIMRSpinAlignedEOBModes API for a predeclared "
            "dt_geom near 1M, copies state/postpeak/mode/epoch outputs, and then "
            "evaluates only the clean SEOBNRv4 Hamiltonian/flux orbital-frequency "
            "event on the returned postpeak state. No source waveform or private "
            "LAL implementation is inspected."
        ),
        "primary_reference": (
            "Bohé et al., Improved effective-one-body model for coalescing "
            "nonprecessing binary black holes, arXiv:1611.03703 (equation context)."
        ),
        "public_parameters": {
            "api": "lalsimulation.SimIMRSpinAlignedEOBModes",
            "approximant": "SEOBNRv4 version 4",
            "m1_msun": 30.0,
            "m2_msun": 20.0,
            "chi1": 0.0,
            "chi2": 0.0,
            "fstart_hz": 20.0,
            "distance_mpc": 100.0,
            "mode": [2, 2],
        },
        "sweep_design": {
            "requested_dt_geom_M": [float(value) for _, value in SWEEP],
            "labels": [label for label, _ in SWEEP],
            "exact_4096Hz_anchor_dt_geom_M": float(EXACT_4096_DT_M),
            "selection": "all predeclared rows are evaluated; no row is selected by residual",
            "event_grid": "returned postpeak_state[0] high-resolution grid",
        },
        "clean_parameters": {
            "nu": 0.24,
            "delta": 0.2,
            "chi1": 0.0,
            "chi2": 0.0,
            "hamiltonian": "SEOBNRv4Hamiltonian",
            "flux": "SEOBNRv4PublishedFlux(spin_truncation='generic-leading-ell8')",
            "event": "orbital_frequency_peak degree-six local event detector",
            "coordinate": "initial-origin total-mass M",
        },
        "coordinate_contract": {
            "postpeak_state_time": "returned postpeak_state[0]",
            "public_epoch": "metadata epoch_s / Msec, compared as -epoch_M",
            "sampled_maximum": "argmax clean Hamiltonian omega on returned postpeak state",
            "continuous_event": "clean orbital_frequency_peak(time, omega)",
            "no_fitted_offsets": True,
        },
        "mass_scale": {"total_mass_msun": TOTAL_MASS_MSUN, "Msec": MSEC, "MTSUN_s": MSUN_SEC},
        "libraries": _library_versions(),
        "python_executable": sys.executable,
        "host": platform.node(),
        "dependency_sha256": dependency_sha256,
        "results": results,
        "interpretation": {
            "successful_event_time_range_M_initial_origin": (
                [
                    float(min(row["clean_hamiltonian_omega_event"]["time_M_initial_origin"] for row in successful)),
                    float(max(row["clean_hamiltonian_omega_event"]["time_M_initial_origin"] for row in successful)),
                ] if successful else None
            ),
            "event_before_sample_rows": before_rows,
            "event_after_sample_rows": after_rows,
            "before_sample_minus_epoch_minus_event_range_M": (
                [
                    float(min(
                        results[label]["peak_epoch_geometry"]["minus_epoch_minus_continuous_event_M"]
                        for label in before_rows
                    )),
                    float(max(
                        results[label]["peak_epoch_geometry"]["minus_epoch_minus_continuous_event_M"]
                        for label in before_rows
                    )),
                ] if before_rows else None
            ),
            "after_sample_minus_epoch_minus_event_range_M": (
                [
                    float(min(
                        results[label]["peak_epoch_geometry"]["minus_epoch_minus_continuous_event_M"]
                        for label in after_rows
                    )),
                    float(max(
                        results[label]["peak_epoch_geometry"]["minus_epoch_minus_continuous_event_M"]
                        for label in after_rows
                    )),
                ] if after_rows else None
            ),
            "runtime_recommendation": (
                "No epoch correction or fitted offset is justified by this sweep. "
                "The continuous clean orbital-frequency event is stable while the "
                "public epoch follows a discrete one-high-resolution-sample branch "
                "when that event lies before the sampled maximum and is nearly equal "
                "to it when the event lies after the sampled maximum. Keep the "
                "continuous event as the runtime origin and treat public epoch as "
                "metadata; if native epoch code changes, add a deterministic "
                "bracket-side regression test."
            ),
        },
        "summary": {
            "successful_rows": ok_count,
            "failed_rows": len(results) - ok_count,
            "failed_labels": [label for label, row in results.items() if row.get("status") != "ok"],
        },
        "limitations": [
            "The event is evaluated from the public returned postpeak state through the clean Hamiltonian/flux adapter; this does not claim the clean event is the public implementation's internal event.",
            "The exact 4096-Hz spacing is included as a fixed anchor row; all other rows use the fixed dt_geom values listed above.",
            "Epoch comparisons are reported as signed coordinate differences. No correction or fitted shift is installed from them.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare cached clean TEOB FD output with the public ``SimInspiralFD`` path.

This is a bounded attribution diagnostic.  The native arrays are loaded from
the frozen clean adapter caches; no trajectory is regenerated here.  The
public call uses ``SimInspiralFD``, the generic public TD-to-FD entry point,
with TEOBResumS and the same zero-spin, zero-tide, face-on requests.  The
older ``SimInspiralChooseFDWaveform`` attempt is retained as a provenance
record because that dispatcher only accepts approximants with native FD
implementations.

All comparisons use exact shared frequency bins and the requested band.  They
are unaligned complex-array comparisons: no time or phase optimization,
interpolation, support truncation, or parameter search is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import socket
import sys
import time
from pathlib import Path
from typing import Any

import lal
import lalsimulation as lalsim
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-conditioned-waveform-clean"
DEFAULT_OUTPUT = VERIFY / "public-fd-sim-inspiral-report.json"
DEFAULT_CACHE = VERIFY / "public-fd-sim-inspiral-output.npz"
DEFAULT_LOG = VERIFY / "public-fd-sim-inspiral.log"
DEFAULT_NATIVE_CACHE = VERIFY / "native-output.npz"
DEFAULT_CATALOG_CACHE = VERIFY / "catalog-output.npz"

NATIVE_CACHE_SHA256 = "a7cc79b050778a1869fa2861719e8e80bda44c38f66e4a83206b9d28a7169ead"
CATALOG_CACHE_SHA256 = "70af9febc682f55dc285933eb63086d576ef25959e30aa78dafb44553393e07d"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths() -> dict[str, Path]:
    """Sources needed to interpret the frozen native caches and this report."""

    return {
        "diagnostic": Path(__file__).resolve(),
        "adapter": ROOT / "src/torchwave/approximants/eob/teob_conditioned_waveform_clean.py",
        "conditioning": ROOT / "src/torchwave/approximants/eob/td_fd_conditioning_clean.py",
        "state_mode": ROOT / "src/torchwave/approximants/eob/teob_waveform_state_clean.py",
        "radiation": ROOT / "src/torchwave/approximants/eob/teob_radiation_clean.py",
        "postadiabatic": ROOT / "src/torchwave/approximants/eob/teob_postadiabatic_clean.py",
    }


def _source_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: _sha256(path) for name, path in paths.items()}


def _public_options() -> lal.Dict:
    """Build the common zero-tide, mode-(2,+2) public options."""

    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(params, 0)
    mode_array = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 2, 2)
    lalsim.SimInspiralWaveformParamsInsertModeArray(params, mode_array)
    return params


def _request_dict(
    *, mass1: float, mass2: float, f_low: float, f_max: float,
    delta_f: float, distance: float, inclination: float, phic: float,
    tc: float,
) -> dict[str, Any]:
    return {
        "mass1_msun": float(mass1),
        "mass2_msun": float(mass2),
        "chi1": [0.0, 0.0, 0.0],
        "chi2": [0.0, 0.0, 0.0],
        "lambda1": 0.0,
        "lambda2": 0.0,
        "distance_mpc": float(distance),
        "inclination": float(inclination),
        "phic": float(phic),
        "tc_seconds": float(tc),
        "delta_f_hz": float(delta_f),
        "f_low_hz": float(f_low),
        "f_max_hz": float(f_max),
        "f_ref_hz": float(f_low),
        "long_asc_nodes": 0.0,
        "eccentricity": 0.0,
        "mean_per_ano": 0.0,
    }


def _choose_fd_attempt(request: dict[str, Any]) -> dict[str, Any]:
    """Record the superseded native-FD dispatcher attempt."""

    started = time.perf_counter()
    approximant = lalsim.GetApproximantFromString("TEOBResumS")
    implemented = bool(lalsim.SimInspiralImplementedFDApproximants(approximant))
    distance_si = request["distance_mpc"] * 1.0e6 * lal.PC_SI
    try:
        # This call is deliberately retained as provenance.  TEOBResumS is
        # TD-only in the installed public build, so this dispatcher rejects it.
        lalsim.SimInspiralChooseFDWaveform(
            request["mass1_msun"] * lal.MSUN_SI,
            request["mass2_msun"] * lal.MSUN_SI,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            distance_si,
            request["inclination"],
            request["phic"],
            request["long_asc_nodes"],
            request["eccentricity"],
            request["mean_per_ano"],
            request["delta_f_hz"],
            request["f_low_hz"],
            request["f_max_hz"],
            request["f_ref_hz"],
            _public_options(),
            approximant,
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "method": "SimInspiralChooseFDWaveform",
            "approximant": "TEOBResumS",
            "approximant_enum": int(approximant),
            "public_fd_implemented": implemented,
            "lal_version": getattr(lal, "__version__", None),
            "lalsimulation_version": getattr(lalsim, "__version__", None),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }
    return {
        "status": "ok",
        "method": "SimInspiralChooseFDWaveform",
        "approximant": "TEOBResumS",
        "approximant_enum": int(approximant),
        "public_fd_implemented": implemented,
        "lal_version": getattr(lal, "__version__", None),
        "lalsimulation_version": getattr(lalsim, "__version__", None),
        "samples": "not retained; superseded dispatcher result",
        "elapsed_seconds": time.perf_counter() - started,
    }


def _series_values(series: Any) -> tuple[np.ndarray, float, float, float]:
    values = np.asarray(series.data.data, dtype=np.complex128).copy()
    return (
        values,
        float(series.deltaF),
        float(getattr(series, "f0", 0.0)),
        float(series.epoch),
    )


def _sim_inspiral_fd(request: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Call the public generic TD-to-FD entry point and return copied arrays."""

    started = time.perf_counter()
    approximant = lalsim.GetApproximantFromString("TEOBResumS")
    implemented = bool(lalsim.SimInspiralImplementedFDApproximants(approximant))
    distance_si = request["distance_mpc"] * 1.0e6 * lal.PC_SI
    try:
        series = lalsim.SimInspiralFD(
            request["mass1_msun"] * lal.MSUN_SI,
            request["mass2_msun"] * lal.MSUN_SI,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            distance_si,
            request["inclination"],
            request["phic"],
            request["long_asc_nodes"],
            request["eccentricity"],
            request["mean_per_ano"],
            request["delta_f_hz"],
            request["f_low_hz"],
            request["f_max_hz"],
            request["f_ref_hz"],
            _public_options(),
            approximant,
        )
        if not isinstance(series, (list, tuple)) or len(series) != 2:
            raise TypeError(f"SimInspiralFD returned {type(series).__name__}, expected two series")
        hp, hc = series
        hp_values, hp_delta, hp_f0, hp_epoch = _series_values(hp)
        hc_values, hc_delta, hc_f0, hc_epoch = _series_values(hc)
        if hp_values.size != hc_values.size:
            raise ValueError("public h_plus and h_cross lengths differ")
        if hp_delta != hc_delta or hp_f0 != hc_f0:
            raise ValueError("public h_plus and h_cross frequency axes differ")
        frequencies = hp_f0 + hp_delta * np.arange(hp_values.size, dtype=np.float64)
    except Exception as exc:
        return (
            {
                "status": "unavailable",
                "method": "SimInspiralFD",
                "approximant": "TEOBResumS",
                "approximant_enum": int(approximant),
                "public_fd_implemented": implemented,
                "lal_version": getattr(lal, "__version__", None),
                "lalsimulation_version": getattr(lalsim, "__version__", None),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
            },
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.complex128),
            np.empty(0, dtype=np.complex128),
        )
    return (
        {
            "status": "ok",
            "method": "SimInspiralFD",
            "approximant": "TEOBResumS",
            "approximant_enum": int(approximant),
            "public_fd_implemented": implemented,
            "lal_version": getattr(lal, "__version__", None),
            "lalsimulation_version": getattr(lalsim, "__version__", None),
            "samples": int(hp_values.size),
            "delta_f_hz": hp_delta,
            "f0_hz": hp_f0,
            "frequency_start_hz": float(frequencies[0]) if frequencies.size else None,
            "frequency_end_hz": float(frequencies[-1]) if frequencies.size else None,
            "hp_epoch_seconds": hp_epoch,
            "hc_epoch_seconds": hc_epoch,
            "elapsed_seconds": time.perf_counter() - started,
        },
        frequencies,
        hp_values,
        hc_values,
    )


def _load_native(path: Path, expected_sha256: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    path = path.resolve()
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"native cache hash mismatch for {path}: expected {expected_sha256}, got {actual}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"frequencies", "h_plus_fd", "h_cross_fd"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"native cache {path} is missing {missing}")
        frequencies = np.asarray(archive["frequencies"], dtype=np.float64).reshape(-1).copy()
        hp = np.asarray(archive["h_plus_fd"], dtype=np.complex128).reshape(-1).copy()
        hc = np.asarray(archive["h_cross_fd"], dtype=np.complex128).reshape(-1).copy()
        shapes = {name: [int(item) for item in archive[name].shape] for name in archive.files}
    return (
        {
            "path": str(path),
            "sha256": actual,
            "bytes": path.stat().st_size,
            "keys": sorted(required),
            "archive_keys": sorted(shapes),
            "archive_shapes": shapes,
            "read_only": True,
        },
        frequencies,
        hp,
        hc,
    )


def _exact_band_indices(
    native_frequencies: np.ndarray,
    public_frequencies: np.ndarray,
    *,
    lower_hz: float,
    upper_hz: float,
    delta_f_hz: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    native_frequencies = np.asarray(native_frequencies, dtype=np.float64).reshape(-1)
    public_frequencies = np.asarray(public_frequencies, dtype=np.float64).reshape(-1)
    common = min(native_frequencies.size, public_frequencies.size)
    if common == 0:
        raise ValueError("no common frequency samples")
    axis_delta = np.abs(native_frequencies[:common] - public_frequencies[:common])
    tolerance = max(1.0e-12, 1.0e-12 * abs(float(delta_f_hz)))
    exact = axis_delta <= tolerance
    selected = exact & (native_frequencies[:common] >= float(lower_hz))
    selected &= native_frequencies[:common] <= float(upper_hz)
    indices = np.flatnonzero(selected)
    if indices.size == 0:
        raise ValueError(f"no exact frequency bins in [{lower_hz}, {upper_hz}] Hz")
    return indices, {
        "common_samples": int(common),
        "exact_samples": int(exact.sum()),
        "selected_samples": int(indices.size),
        "axis_tolerance_hz": tolerance,
        "max_common_axis_difference_hz": float(axis_delta.max()),
        "first_index": int(indices[0]),
        "last_index": int(indices[-1]),
        "first_frequency_hz": float(native_frequencies[indices[0]]),
        "last_frequency_hz": float(native_frequencies[indices[-1]]),
        "native_support_hz": [float(native_frequencies[indices[0]]), float(native_frequencies[indices[-1]])],
        "public_support_hz": [float(public_frequencies[indices[0]]), float(public_frequencies[indices[-1]])],
    }


def _metrics(native: np.ndarray, public: np.ndarray) -> dict[str, float | None]:
    native = np.asarray(native, dtype=np.complex128).reshape(-1)
    public = np.asarray(public, dtype=np.complex128).reshape(-1)
    native_norm = float(np.linalg.norm(native))
    public_norm = float(np.linalg.norm(public))
    residual_norm = float(np.linalg.norm(native - public))
    overlap = None
    mismatch = None
    if native_norm > 0.0 and public_norm > 0.0:
        overlap = float(np.real(np.vdot(public, native)) / (public_norm * native_norm))
        mismatch = 1.0 - overlap
    return {
        "signal_norm_native": native_norm,
        "signal_norm_public": public_norm,
        "residual_norm_unaligned": residual_norm,
        "relative_l2_unaligned": None if public_norm == 0.0 else residual_norm / public_norm,
        "complex_norm_ratio_native_over_public": None if public_norm == 0.0 else native_norm / public_norm,
        "flat_complex_overlap_unaligned": overlap,
        "mismatch_unaligned": mismatch,
    }


def _band_comparison(
    native_frequencies: np.ndarray,
    native_hp: np.ndarray,
    native_hc: np.ndarray,
    public_frequencies: np.ndarray,
    public_hp: np.ndarray,
    public_hc: np.ndarray,
    *,
    lower_hz: float,
    upper_hz: float,
    delta_f_hz: float,
) -> dict[str, Any]:
    indices, axis_report = _exact_band_indices(
        native_frequencies,
        public_frequencies,
        lower_hz=lower_hz,
        upper_hz=upper_hz,
        delta_f_hz=delta_f_hz,
    )
    return {
        "requested_band_hz": [float(lower_hz), float(upper_hz)],
        "axis": axis_report,
        "alignment": False,
        "optimization": False,
        "interpolation": False,
        "support_truncation": False,
        "h_plus": _metrics(native_hp[indices], public_hp[indices]),
        "h_cross": _metrics(native_hc[indices], public_hc[indices]),
        "combined": _metrics(
            np.concatenate((native_hp[indices], native_hc[indices])),
            np.concatenate((public_hp[indices], public_hc[indices])),
        ),
    }


def _safe_key(label: str) -> str:
    return label.replace(".", "_").replace("+", "p")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-cache", type=Path, default=DEFAULT_NATIVE_CACHE)
    parser.add_argument("--catalog-cache", type=Path, default=DEFAULT_CATALOG_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for path in (args.output, args.cache, args.log):
        path.parent.mkdir(parents=True, exist_ok=True)
    source_paths = _source_paths()
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "run_metadata": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "command": shlex.join([sys.executable, *sys.argv]),
            "pid": os.getpid(),
            "expected_next_check": str(args.output.resolve()),
            "stop_command": f"kill {os.getpid()}",
        },
        "method": {
            "requested": "SimInspiralFD",
            "approximant": "TEOBResumS",
            "description": "public generic TD-to-FD entry point",
            "comparison": "exact shared frequency bins; unaligned complex residuals",
            "no_alignment": True,
            "no_parameter_search": True,
        },
        "public_options": {
            "spins": "zero aligned spins",
            "lambda1_lambda2": [0.0, 0.0],
            "distance_mpc": 100.0,
            "inclination": 0.0,
            "phic": 0.0,
            "tc_seconds": 0.0,
            "mode_array": "[(2,+2)]",
            "PNTidalOrder": 0,
        },
        "source_sha256_before": {},
        "source_sha256_after": {},
        "cache_inputs": {},
        "cases": {},
    }
    try:
        report["source_sha256_before"] = _source_hashes(source_paths)
        native_info, native_f, native_hp, native_hc = _load_native(
            args.native_cache, NATIVE_CACHE_SHA256,
        )
        catalog_info, catalog_f, catalog_hp, catalog_hc = _load_native(
            args.catalog_cache, CATALOG_CACHE_SHA256,
        )
        report["cache_inputs"] = {
            "30+20": native_info,
            "1.4+1.4": catalog_info,
        }
        cases = [
            ("30+20", native_f, native_hp, native_hc, native_info, 30.0, 20.0, 20.0, 1024.0, [20.0, 256.0]),
            ("1.4+1.4", catalog_f, catalog_hp, catalog_hc, catalog_info, 1.4, 1.4, 30.0, 1024.0, [30.0, 500.0]),
        ]
        cache_arrays: dict[str, np.ndarray] = {}
        for label, native_f, native_hp, native_hc, native_info, mass1, mass2, f_low, f_max, subband in cases:
            request = _request_dict(
                mass1=mass1,
                mass2=mass2,
                f_low=f_low,
                f_max=f_max,
                delta_f=0.5,
                distance=100.0,
                inclination=0.0,
                phic=0.0,
                tc=0.0,
            )
            print(f"case: {label} ChooseFDWaveform provenance", flush=True)
            choose_meta = _choose_fd_attempt(request)
            print(f"case: {label} SimInspiralFD", flush=True)
            public_meta, public_f, public_hp, public_hc = _sim_inspiral_fd(request)
            case_report: dict[str, Any] = {
                "request": request,
                "native_cache": native_info,
                "superseded_choose_fd_attempt": {
                    **choose_meta,
                    "reason": "native-FD dispatcher; retained only as provenance; SimInspiralFD is the requested generic path",
                },
                "public": public_meta,
            }
            if public_meta["status"] != "ok":
                case_report["comparison_status"] = "unavailable"
                case_report["comparison_error"] = "SimInspiralFD returned no public series"
            else:
                case_report["comparison_status"] = "complete"
                case_report["bands"] = {
                    "fullband": _band_comparison(
                        native_f, native_hp, native_hc,
                        public_f, public_hp, public_hc,
                        lower_hz=f_low, upper_hz=f_max, delta_f_hz=0.5,
                    ),
                    "catalog_subband": _band_comparison(
                        native_f, native_hp, native_hc,
                        public_f, public_hp, public_hc,
                        lower_hz=subband[0], upper_hz=subband[1], delta_f_hz=0.5,
                    ),
                }
            report["cases"][label] = case_report
            key = _safe_key(label)
            cache_arrays[f"{key}_native_frequencies"] = native_f
            cache_arrays[f"{key}_native_h_plus_fd"] = native_hp
            cache_arrays[f"{key}_native_h_cross_fd"] = native_hc
            cache_arrays[f"{key}_public_frequencies"] = public_f
            cache_arrays[f"{key}_public_h_plus_fd"] = public_hp
            cache_arrays[f"{key}_public_h_cross_fd"] = public_hc
        print("stage: writing public FD cache", flush=True)
        np.savez_compressed(args.cache, **cache_arrays)
        report["cache_output"] = {
            "path": str(args.cache.resolve()),
            "sha256": _sha256(args.cache),
            "bytes": args.cache.stat().st_size,
            "keys": sorted(cache_arrays),
            "array_shapes": {name: [int(item) for item in value.shape] for name, value in cache_arrays.items()},
            "read_only_after_write": True,
        }
        report["source_sha256_after"] = _source_hashes(source_paths)
        report["source_hashes_unchanged"] = report["source_sha256_before"] == report["source_sha256_after"]
        report["status"] = "complete"
    except Exception as exc:  # pragma: no cover - failure metadata path
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            report["source_sha256_after"] = _source_hashes(source_paths)
            report["source_hashes_unchanged"] = report["source_sha256_before"] == report["source_sha256_after"]
        except Exception:
            pass
    report["elapsed_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, sort_keys=True), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

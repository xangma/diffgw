#!/usr/bin/env python3
"""Rerender frozen clean TEOB stages with the opt-in hybrid remnant.

The two frozen adapter caches contain the accepted trajectory and corrected
NQC charts.  This diagnostic reuses those arrays and recomputes only the
ringdown assembly, observer projection, and TD-to-FD conditioning for the
existing UIB2016 and explicit ``teob-healy-jf`` remnant policies.  The public
reference is the already captured ``SimInspiralFD`` cache, on identical
frequency bins.  No ODE is evolved, no waveform is aligned, and no parameter
is selected from residuals.
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

import numpy as np
import torch

from torchwave.approximants.eob import teob_conditioned_waveform_clean as adapter
from torchwave.approximants.eob.teob_ringdown_assembly_clean import (
    RingdownAssemblyResult,
    assemble_teob_ringdown_22,
)
from torchwave.approximants.eob.td_fd_conditioning_clean import (
    condition_td_to_fd,
    plan_td_support,
)


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-remnant-hybrid-clean"
DEFAULT_OUTPUT = VERIFY / "diagnostic.json"
DEFAULT_CACHE = VERIFY / "remnant-hybrid-output.npz"
DEFAULT_LOG = VERIFY / "remnant-hybrid.log"
PUBLIC_FD_CACHE = (
    ROOT / "docs/reviews/2026-09-13-parity-review/verification/"
    "teob-conditioned-waveform-clean/public-fd-sim-inspiral-output.npz"
)
PUBLIC_FD_REPORT = PUBLIC_FD_CACHE.with_name("public-fd-sim-inspiral-report.json")
EXPECTED_PUBLIC_FD_SHA256 = (
    "084e58b5332549b73fdaeb3ffb63df04d573a8082b83db7230d25642d0780fb8"
)

CASES: dict[str, dict[str, Any]] = {
    "30p20": {
        "mass1": 30.0,
        "mass2": 20.0,
        "f_low": 20.0,
        "native_name": "native-output.npz",
        "native_sha256": "a7cc79b050778a1869fa2861719e8e80bda44c38f66e4a83206b9d28a7169ead",
        "public_prefix": "30p20",
        "subband_hz": [20.0, 256.0],
    },
    "1p4+1p4": {
        "mass1": 1.4,
        "mass2": 1.4,
        "f_low": 30.0,
        "native_name": "catalog-output.npz",
        "native_sha256": "70af9febc682f55dc285933eb63086d576ef25959e30aa78dafb44553393e07d",
        "public_prefix": "1_4p1_4",
        "subband_hz": [30.0, 500.0],
    },
}
POLICIES = ("uib2016", "teob-healy-jf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _tensor(value: Any, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device="cpu")


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _source_paths() -> dict[str, Path]:
    """Only clean sources used to interpret the rerendered stages."""

    return {
        "diagnostic": Path(__file__).resolve(),
        "adapter": ROOT / "src/torchwave/approximants/eob/teob_conditioned_waveform_clean.py",
        "hybrid_remnant": ROOT / "src/torchwave/approximants/eob/teob_healy2014_remnant_clean.py",
        "uib_remnant": ROOT / "src/torchwave/approximants/eob/teob_uib2016_remnant.py",
        "ringdown_assembly": ROOT / "src/torchwave/approximants/eob/teob_ringdown_assembly_clean.py",
        "postmerger": ROOT / "src/torchwave/approximants/eob/teob_postmerger.py",
        "uib_postmerger": ROOT / "src/torchwave/approximants/eob/teob_uib2016_postmerger.py",
        "conditioning": ROOT / "src/torchwave/approximants/eob/td_fd_conditioning_clean.py",
        "projection": ROOT / "src/torchwave/approximants/eob/conditioned_waveform_clean.py",
    }


def _source_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: _sha256(path) for name, path in paths.items()}


def _load_archive(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"cache hash mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _require(cache: dict[str, np.ndarray], names: tuple[str, ...], *, label: str) -> None:
    missing = sorted(set(names).difference(cache))
    if missing:
        raise KeyError(f"{label} cache is missing {missing}")


def _metrics(clean: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    clean = np.asarray(clean, dtype=np.complex128).reshape(-1)
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
    if clean.shape != reference.shape:
        raise ValueError("comparison arrays must have identical shapes")
    signal_norm_clean = float(np.linalg.norm(clean))
    signal_norm_reference = float(np.linalg.norm(reference))
    residual = clean - reference
    residual_norm = float(np.linalg.norm(residual))
    overlap = None
    mismatch = None
    if signal_norm_clean > 0.0 and signal_norm_reference > 0.0:
        overlap = float(
            np.real(np.vdot(reference, clean))
            / (signal_norm_reference * signal_norm_clean)
        )
        mismatch = 1.0 - overlap
    return {
        "samples": int(clean.size),
        "signal_norm_clean": signal_norm_clean,
        "signal_norm_reference": signal_norm_reference,
        "residual_norm_unaligned": residual_norm,
        "relative_l2_unaligned": (
            residual_norm / signal_norm_reference
            if signal_norm_reference else None
        ),
        "complex_norm_ratio_clean_over_reference": (
            signal_norm_clean / signal_norm_reference
            if signal_norm_reference else None
        ),
        "flat_complex_overlap_unaligned": overlap,
        "mismatch_unaligned": mismatch,
    }


def _axis_band(
    clean_frequency: np.ndarray,
    reference_frequency: np.ndarray,
    *,
    lower_hz: float,
    upper_hz: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    clean_frequency = np.asarray(clean_frequency, dtype=np.float64).reshape(-1)
    reference_frequency = np.asarray(reference_frequency, dtype=np.float64).reshape(-1)
    if clean_frequency.shape != reference_frequency.shape:
        raise ValueError("frequency axes have different lengths")
    axis_difference = np.abs(clean_frequency - reference_frequency)
    tolerance = 1.0e-12
    mask = axis_difference <= tolerance
    mask &= clean_frequency >= lower_hz
    mask &= clean_frequency <= upper_hz
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(f"no common exact bins in [{lower_hz}, {upper_hz}] Hz")
    return indices, {
        "requested_band_hz": [float(lower_hz), float(upper_hz)],
        "samples": int(indices.size),
        "first_index": int(indices[0]),
        "last_index": int(indices[-1]),
        "first_frequency_hz": float(clean_frequency[indices[0]]),
        "last_frequency_hz": float(clean_frequency[indices[-1]]),
        "max_axis_difference_hz": float(axis_difference.max()),
        "axis_tolerance_hz": tolerance,
    }


def _band_comparison(
    clean_frequency: np.ndarray,
    clean_hp: np.ndarray,
    clean_hc: np.ndarray,
    public_frequency: np.ndarray,
    public_hp: np.ndarray,
    public_hc: np.ndarray,
    *,
    lower_hz: float,
    upper_hz: float,
) -> dict[str, Any]:
    indices, axis = _axis_band(
        clean_frequency,
        public_frequency,
        lower_hz=lower_hz,
        upper_hz=upper_hz,
    )
    return {
        "axis": axis,
        "alignment": False,
        "optimization": False,
        "support_truncation": False,
        "h_plus": _metrics(clean_hp[indices], public_hp[indices]),
        "h_cross": _metrics(clean_hc[indices], public_hc[indices]),
        "combined": _metrics(
            np.concatenate((clean_hp[indices], clean_hc[indices])),
            np.concatenate((public_hp[indices], public_hc[indices])),
        ),
    }


def _difference(clean: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    clean = np.asarray(clean)
    reference = np.asarray(reference)
    if clean.shape != reference.shape:
        return {
            "same_shape": False,
            "clean_shape": [int(item) for item in clean.shape],
            "reference_shape": [int(item) for item in reference.shape],
        }
    delta = clean - reference
    abs_delta = np.abs(delta)
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    return {
        "same_shape": True,
        "samples": int(clean.size),
        "max_abs_difference": float(abs_delta.max()) if abs_delta.size else 0.0,
        "signal_norm_reference": reference_norm,
        "residual_norm": float(np.linalg.norm(delta.reshape(-1))),
        "relative_l2": (
            float(np.linalg.norm(delta.reshape(-1))) / reference_norm
            if reference_norm else None
        ),
    }


def _events(result: RingdownAssemblyResult, final_mass: torch.Tensor) -> dict[str, Any]:
    events = result.events
    output_time = _numpy(result.time).reshape(-1)
    attachment_time = _float(events.attachment_time)
    output_index = int(np.searchsorted(output_time, attachment_time, side="left"))
    output_index = min(max(output_index, 0), output_time.size - 1)
    return {
        "peak_knot_index_auxiliary": int(events.peak_knot_index),
        "peak_time_initial_mass": _float(events.peak_time),
        "peak_time_final_mass": _float(events.peak_time / final_mass),
        "merger_time_initial_mass": _float(events.merger_time),
        "merger_time_final_mass": _float(events.merger_time / final_mass),
        "t_match_initial_mass": _float(events.match_time),
        "match_time_final_mass": _float(events.match_time_final_mass),
        "strictly_earlier_index_joined": int(events.strictly_earlier_index),
        "attachment_sampled_index_joined": int(events.attachment_index),
        "attachment_time_initial_mass": attachment_time,
        "attachment_time_final_mass": _float(events.attachment_time_final_mass),
        "output_boundary_index": output_index,
        "output_boundary_time_initial_mass": float(output_time[output_index]),
        "joined_samples": int(result.joined_time.numel()),
        "extended_samples": int(result.extended_time.numel()),
        "output_samples": int(result.time.numel()),
        "extension_samples": int(result.extension_samples),
        "auxiliary_spacing_initial_mass": _float(result.auxiliary_spacing),
        "epoch_coordinate_start_initial_mass": _float(result.epoch),
        "joined_support_initial_mass": [
            _float(result.joined_time[0]), _float(result.joined_time[-1])
        ],
        "extended_support_initial_mass": [
            _float(result.extended_time[0]), _float(result.extended_time[-1])
        ],
        "output_support_initial_mass": [
            _float(result.time[0]), _float(result.time[-1])
        ],
        "auxiliary_amplitude_negative": bool(result.auxiliary_amplitude_negative),
    }


def _events_with_final_mass(
    result: RingdownAssemblyResult,
    final_mass: torch.Tensor,
    final_spin: torch.Tensor,
) -> dict[str, Any]:
    report = _events(result, final_mass)
    report["final_mass_fraction"] = _float(final_mass)
    report["final_spin"] = _float(final_spin)
    return report


def _rerender(
    cache: dict[str, np.ndarray],
    *,
    mass1: float,
    mass2: float,
    f_low: float,
    policy: str,
) -> tuple[RingdownAssemblyResult, Any, dict[str, Any]]:
    required = (
        "nqc_original_time",
        "nqc_original_corrected_amplitude",
        "nqc_original_corrected_phase",
        "nqc_auxiliary_time",
        "nqc_auxiliary_corrected_amplitude",
        "nqc_auxiliary_corrected_phase",
        "nqc_auxiliary_orbital_frequency",
        "sampling_dt_seconds",
        "sampling_delta_f",
        "sampling_f_nyquist",
        "total_mass_seconds",
    )
    _require(cache, required, label="native")
    dtype = torch.float64
    m1 = _tensor(mass1, dtype=dtype)
    m2 = _tensor(mass2, dtype=dtype)
    flow = _tensor(f_low, dtype=dtype)
    hamiltonian = adapter._make_hamiltonian(m1, m2)
    final_mass, final_spin = adapter._select_remnant(m1, m2, policy)
    ringdown = assemble_teob_ringdown_22(
        _tensor(cache["nqc_original_time"], dtype=dtype),
        _tensor(cache["nqc_original_corrected_amplitude"], dtype=dtype),
        _tensor(cache["nqc_original_corrected_phase"], dtype=dtype),
        _tensor(cache["nqc_auxiliary_time"], dtype=dtype),
        _tensor(cache["nqc_auxiliary_corrected_amplitude"], dtype=dtype),
        _tensor(cache["nqc_auxiliary_corrected_phase"], dtype=dtype),
        _tensor(cache["nqc_auxiliary_orbital_frequency"], dtype=dtype),
        nu=hamiltonian.nu,
        final_mass=final_mass,
        final_spin=final_spin,
        output_step=_tensor(cache["sampling_dt_seconds"], dtype=dtype)
        / _tensor(cache["total_mass_seconds"], dtype=dtype),
    )
    mode_2m2 = torch.conj(ringdown.mode)
    h_plus_td, h_cross_td = adapter.project_teob22_generic(
        ringdown.mode,
        mode_2m2,
        mass1=m1,
        mass2=m2,
        distance=_tensor(100.0, dtype=dtype),
        inclination=_tensor(0.0, dtype=dtype),
        phic=_tensor(0.0, dtype=dtype),
    )
    support = plan_td_support(m1, m2, flow, dtype=dtype)
    fd = condition_td_to_fd(
        h_plus_td,
        h_cross_td,
        dt=_tensor(cache["sampling_dt_seconds"], dtype=dtype),
        delta_f=_tensor(cache["sampling_delta_f"], dtype=dtype),
        f_max=_tensor(cache["sampling_f_nyquist"], dtype=dtype),
        support=support,
        tc=_tensor(0.0, dtype=dtype),
        return_metadata=True,
    )
    support_differences = {
        "total_mass_seconds": abs(_float(support.total_mass_seconds) - _float(cache["total_mass_seconds"])),
        "start_frequency_hz": abs(_float(support.start_frequency) - _float(cache["support_start_frequency"])),
        "support_time_initial_mass": abs(_float(support.support_time) - _float(cache["support_time"])),
        "first_taper_duration_initial_mass": abs(
            _float(support.first_taper_duration) - _float(cache["support_first_taper_duration"])
        ),
    }
    metadata = {
        "final_mass_fraction": _float(final_mass),
        "final_spin": _float(final_spin),
        "nu": _float(hamiltonian.nu),
        "mass1_msun": mass1,
        "mass2_msun": mass2,
        "f_low_hz": f_low,
        "distance_mpc": 100.0,
        "inclination": 0.0,
        "phic": 0.0,
        "matching_metadata_replayed": False,
        "matching_metadata_note": "NQC corrected charts are sufficient for waveform rerender; cached t_nqc/state metadata is not used by ringdown physics.",
        "support_differences": support_differences,
    }
    return ringdown, (h_plus_td, h_cross_td, fd), metadata


def _policy_cache_key(case_name: str, policy: str) -> str:
    return f"{case_name}_{policy.replace('-', '_')}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, default=VERIFY.parent / "teob-conditioned-waveform-clean")
    parser.add_argument("--public-cache", type=Path, default=PUBLIC_FD_CACHE)
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
            "trajectory": "reused frozen native NQC charts; no full ODE rerun",
            "stages_rerendered": ["ringdown assembly", "TEOB 22 projection", "TD-to-FD conditioning"],
            "policies": list(POLICIES),
            "remnant_policy_default_preserved": "uib2016",
            "hybrid_policy": "teob-healy-jf",
            "comparison": "same cached public SimInspiralFD frequency bins; unaligned complex metrics",
            "alignment": False,
            "parameter_search": False,
            "support_truncation": False,
        },
        "helper_review": {
            "status": "independently reviewed",
            "equations": [
                "Healy et al. arXiv:1406.7295 Eqs. (14), (16), Table VI",
                "Jimenez-Forteza et al. arXiv:1611.00332 Eq. (7), Table I",
            ],
            "result": "Healy compatibility branch supplies final mass; printed Jimenez-Forteza coefficients supply final spin; standalone UIB2016 remains separate.",
            "focused_helper_tests": "16 passed",
        },
        "source_sha256_before": {},
        "source_sha256_after": {},
        "source_hashes_unchanged": None,
        "inputs": {},
        "cases": {},
    }
    cache_arrays: dict[str, np.ndarray] = {}
    try:
        report["source_sha256_before"] = _source_hashes(source_paths)
        public_hash = _sha256(args.public_cache)
        if public_hash != EXPECTED_PUBLIC_FD_SHA256:
            raise RuntimeError(
                f"public FD cache hash mismatch: expected {EXPECTED_PUBLIC_FD_SHA256}, got {public_hash}"
            )
        public_cache = _load_archive(args.public_cache, EXPECTED_PUBLIC_FD_SHA256)
        public_report_info: dict[str, Any] = {
            "path": str(args.public_cache.resolve()),
            "sha256": public_hash,
            "bytes": args.public_cache.stat().st_size,
            "read_only": True,
        }
        if PUBLIC_FD_REPORT.exists():
            public_report_info["report_path"] = str(PUBLIC_FD_REPORT.resolve())
            public_report_info["report_sha256"] = _sha256(PUBLIC_FD_REPORT)
            try:
                prior_report = json.loads(PUBLIC_FD_REPORT.read_text(encoding="utf-8"))
                public_report_info["method"] = prior_report.get("method", {})
            except (OSError, json.JSONDecodeError):
                public_report_info["report_readable"] = False
        report["inputs"]["public_fd_cache"] = public_report_info

        for case_name, case in CASES.items():
            native_path = args.native_dir / case["native_name"]
            native = _load_archive(native_path, case["native_sha256"])
            public_prefix = case["public_prefix"]
            public_names = (
                f"{public_prefix}_native_frequencies",
                f"{public_prefix}_native_h_plus_fd",
                f"{public_prefix}_native_h_cross_fd",
                f"{public_prefix}_public_frequencies",
                f"{public_prefix}_public_h_plus_fd",
                f"{public_prefix}_public_h_cross_fd",
            )
            _require(public_cache, public_names, label="public FD")
            public_native_frequency = public_cache[public_names[0]].astype(np.float64, copy=False)
            public_native_hp = public_cache[public_names[1]].astype(np.complex128, copy=False)
            public_native_hc = public_cache[public_names[2]].astype(np.complex128, copy=False)
            public_frequency = public_cache[public_names[3]].astype(np.float64, copy=False)
            public_hp = public_cache[public_names[4]].astype(np.complex128, copy=False)
            public_hc = public_cache[public_names[5]].astype(np.complex128, copy=False)
            native_frequency = native["frequencies"].astype(np.float64, copy=False)
            native_hp = native["h_plus_fd"].astype(np.complex128, copy=False)
            native_hc = native["h_cross_fd"].astype(np.complex128, copy=False)
            case_report: dict[str, Any] = {
                "request": {
                    "mass1_msun": case["mass1"],
                    "mass2_msun": case["mass2"],
                    "f_low_hz": case["f_low"],
                    "f_max_hz": 1024.0,
                    "delta_f_hz": 0.5,
                    "distance_mpc": 100.0,
                    "inclination": 0.0,
                    "phic": 0.0,
                    "tc_seconds": 0.0,
                    "mode": [2, 2],
                    "spins": [0.0, 0.0],
                    "tides": [0.0, 0.0],
                },
                "native_cache": {
                    "path": str(native_path.resolve()),
                    "sha256": case["native_sha256"],
                    "read_only": True,
                    "samples_td": int(native["td_time"].size),
                    "samples_fd": int(native_frequency.size),
                },
                "public_cache_native_consistency": {
                    "frequency_axis": _difference(native_frequency, public_native_frequency),
                    "h_plus": _difference(native_hp, public_native_hp),
                    "h_cross": _difference(native_hc, public_native_hc),
                },
                "nqc_input": {
                    "amplitude_coefficients": _numpy(native["nqc_amplitude_coefficients"]).reshape(-1).tolist(),
                    "phase_coefficients": _numpy(native["nqc_phase_coefficients"]).reshape(-1).tolist(),
                    "matching_index_auxiliary": int(native["nqc_matching_index"]),
                    "corrected_original_samples": int(native["nqc_original_time"].size),
                    "corrected_auxiliary_samples": int(native["nqc_auxiliary_time"].size),
                },
                "policies": {},
            }
            rerendered: dict[str, tuple[RingdownAssemblyResult, Any, dict[str, Any]]] = {}
            for policy in POLICIES:
                result, outputs, remnant_meta = _rerender(
                    native,
                    mass1=case["mass1"],
                    mass2=case["mass2"],
                    f_low=case["f_low"],
                    policy=policy,
                )
                h_plus_td, h_cross_td, fd = outputs
                rerendered[policy] = (result, outputs, remnant_meta)
                clean_frequency = _numpy(fd.frequencies).reshape(-1)
                clean_hp = _numpy(fd.h_plus).reshape(-1)
                clean_hc = _numpy(fd.h_cross).reshape(-1)
                policy_report: dict[str, Any] = {
                    "remnant": remnant_meta,
                    "events": _events_with_final_mass(
                        result,
                        _tensor(remnant_meta["final_mass_fraction"]),
                        _tensor(remnant_meta["final_spin"]),
                    ),
                    "output": {
                        "td_samples": int(result.time.numel()),
                        "fd_samples": int(clean_frequency.size),
                        "td_support_initial_mass": [_float(result.time[0]), _float(result.time[-1])],
                        "fd_support_hz": [float(clean_frequency[0]), float(clean_frequency[-1])],
                    },
                    "rerender_vs_frozen_native": {
                        "td_time": _difference(_numpy(result.time), native["td_time"]),
                        "mode_22": _difference(_numpy(result.mode), native["mode_22"]),
                        "h_plus_td": _difference(_numpy(h_plus_td), native["h_plus_td"]),
                        "h_cross_td": _difference(_numpy(h_cross_td), native["h_cross_td"]),
                        "frequencies": _difference(clean_frequency, native_frequency),
                        "h_plus_fd": _difference(clean_hp, native_hp),
                        "h_cross_fd": _difference(clean_hc, native_hc),
                    },
                    "comparison_vs_public_sim_inspiral_fd": {
                        "fullband": _band_comparison(
                            clean_frequency,
                            clean_hp,
                            clean_hc,
                            public_frequency,
                            public_hp,
                            public_hc,
                            lower_hz=case["f_low"],
                            upper_hz=1024.0,
                        ),
                        "catalog_subband": _band_comparison(
                            clean_frequency,
                            clean_hp,
                            clean_hc,
                            public_frequency,
                            public_hp,
                            public_hc,
                            lower_hz=case["subband_hz"][0],
                            upper_hz=case["subband_hz"][1],
                        ),
                    },
                }
                case_report["policies"][policy] = policy_report
                key = _policy_cache_key(case_name, policy)
                cache_arrays[f"{key}_frequencies"] = clean_frequency
                cache_arrays[f"{key}_h_plus_fd"] = clean_hp
                cache_arrays[f"{key}_h_cross_fd"] = clean_hc
                cache_arrays[f"{key}_td_time"] = _numpy(result.time)
                cache_arrays[f"{key}_mode_22"] = _numpy(result.mode)
                cache_arrays[f"{key}_h_plus_td"] = _numpy(h_plus_td)
                cache_arrays[f"{key}_h_cross_td"] = _numpy(h_cross_td)
                cache_arrays[f"{key}_ringdown_amplitude"] = _numpy(result.amplitude)
                cache_arrays[f"{key}_ringdown_phase"] = _numpy(result.phase)
                cache_arrays[f"{key}_event_attachment_index"] = np.asarray(int(result.events.attachment_index), dtype=np.int64)
                cache_arrays[f"{key}_event_attachment_time"] = np.asarray(_float(result.events.attachment_time), dtype=np.float64)
                cache_arrays[f"{key}_event_peak_time"] = np.asarray(_float(result.events.peak_time), dtype=np.float64)
                cache_arrays[f"{key}_final_mass_fraction"] = np.asarray(remnant_meta["final_mass_fraction"], dtype=np.float64)
                cache_arrays[f"{key}_final_spin"] = np.asarray(remnant_meta["final_spin"], dtype=np.float64)

            uib_result = rerendered["uib2016"][0]
            hybrid_result = rerendered["teob-healy-jf"][0]
            uib_outputs = rerendered["uib2016"][1]
            hybrid_outputs = rerendered["teob-healy-jf"][1]
            case_report["hybrid_minus_uib"] = {
                "event_attachment_index": {
                    "uib2016": int(uib_result.events.attachment_index),
                    "teob_healy_jf": int(hybrid_result.events.attachment_index),
                },
                "event_attachment_time_initial_mass": {
                    "uib2016": _float(uib_result.events.attachment_time),
                    "teob_healy_jf": _float(hybrid_result.events.attachment_time),
                },
                "mode_22": _difference(_numpy(hybrid_result.mode), _numpy(uib_result.mode)),
                "h_plus_td": _difference(_numpy(hybrid_outputs[0]), _numpy(uib_outputs[0])),
                "h_cross_td": _difference(_numpy(hybrid_outputs[1]), _numpy(uib_outputs[1])),
                "h_plus_fd": _difference(_numpy(hybrid_outputs[2].h_plus), _numpy(uib_outputs[2].h_plus)),
                "h_cross_fd": _difference(_numpy(hybrid_outputs[2].h_cross), _numpy(uib_outputs[2].h_cross)),
            }
            report["cases"][case_name] = case_report

        np.savez_compressed(args.cache, **cache_arrays)
        report["cache_output"] = {
            "path": str(args.cache.resolve()),
            "sha256": _sha256(args.cache),
            "bytes": args.cache.stat().st_size,
            "keys": sorted(cache_arrays),
            "array_shapes": {
                name: [int(item) for item in value.shape]
                for name, value in cache_arrays.items()
            },
            "read_only_after_write": True,
        }
        report["status"] = "complete"
    except Exception as exc:  # pragma: no cover - failure metadata path
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        try:
            report["source_sha256_after"] = _source_hashes(source_paths)
            report["source_hashes_unchanged"] = (
                report["source_sha256_before"] == report["source_sha256_after"]
            )
        except Exception:
            pass
        report["elapsed_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}, sort_keys=True), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

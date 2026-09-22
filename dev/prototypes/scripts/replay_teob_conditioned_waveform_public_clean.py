#!/usr/bin/env python3
"""Bounded native 30+20 TEOB replay with a public FD comparison.

This is a fresh-output replay of the reviewed clean adapter configuration.
It is deliberately limited to one 30+20 native call and one public
``SimInspiralFD`` call.  No alignment, interpolation, fitting,
threshold tuning, or native source is used.  The script is not run as part of
the clean-room diagnostics; launch it explicitly when the helper worker has
declared the support-plan arithmetic ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import lal
import lalsimulation as lalsim

from torchwave.approximants.eob.teob_adaptive_continuation_clean import (
    DEFAULT_ATOL,
    DEFAULT_CONTINUATION_DURATION,
    DEFAULT_FIXED_STEP_CAP,
    DEFAULT_MAX_ACCEPTED_STEPS,
    DEFAULT_MAX_DURATION,
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_MAX_REJECTED_STEPS,
    DEFAULT_RTOL,
)
from torchwave.approximants.eob.td_fd_conditioning_clean import plan_td_support
from torchwave.approximants.eob.teob_conditioned_waveform_clean import (
    conditioned_teob_fd,
)

# These helpers are read-only utilities from the existing clean diagnostic.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from diagnose_teob_conditioned_waveform_clean import (  # noqa: E402
    _cache_arrays,
    _sha256,
    _support_summary,
    _sampling_summary,
    _tensor_numpy,
    _statistics_summary,
)


ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-native-replay-public-clean"
DEFAULT_REPORT = VERIFY / "replay-report.json"
DEFAULT_NATIVE = VERIFY / "replay-native-output.npz"
DEFAULT_FD = VERIFY / "replay-public-fd-output.npz"
DEFAULT_LOG = VERIFY / "replay.log"
BASELINE_NATIVE = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-conditioned-waveform-clean/native-output.npz"
BASELINE_FD = ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-conditioned-waveform-clean/public-fd-sim-inspiral-output.npz"
PRODUCER = ROOT / "scripts/diagnose_teob_public_fd_clean.py"


def _f(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _hex(value: Any) -> str:
    return _f(value).hex()


def _metric(native: np.ndarray, public: np.ndarray) -> dict[str, Any]:
    native = np.asarray(native, dtype=np.complex128).reshape(-1)
    public = np.asarray(public, dtype=np.complex128).reshape(-1)
    if native.shape != public.shape:
        raise ValueError(f"metric shape mismatch: native={native.shape}, public={public.shape}")
    if native.size == 0 or public.size == 0:
        raise ValueError("metric arrays must be non-empty")
    if not np.isfinite(native).all() or not np.isfinite(public).all():
        raise ValueError("metric arrays must be finite")
    pn = float(np.linalg.norm(public))
    rn = float(np.linalg.norm(native - public))
    return {
        "status": "ok",
        "samples": int(native.size),
        "public_norm": pn,
        "residual_norm": rn,
        "relative_l2_public_norm": None if pn == 0.0 else rn / pn,
        "max_abs": float(np.max(np.abs(native - public))) if native.size else 0.0,
    }


def _axis_metric(native_f: np.ndarray, public_f: np.ndarray) -> dict[str, Any]:
    native_f = np.asarray(native_f, dtype=np.float64).reshape(-1)
    public_f = np.asarray(public_f, dtype=np.float64).reshape(-1)
    common = min(native_f.size, public_f.size)
    delta = np.abs(native_f[:common] - public_f[:common])
    return {
        "native_samples": int(native_f.size),
        "public_samples": int(public_f.size),
        "common_samples": int(common),
        "exact_common_axis": bool(common and np.all(delta == 0.0)),
        "maximum_common_difference_hz": float(delta.max()) if common else None,
    }


def _public_options() -> lal.Dict:
    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(params, 0)
    mode_array = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 2, 2)
    lalsim.SimInspiralWaveformParamsInsertModeArray(params, mode_array)
    return params


def _public_fd_call(args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Call the generic public SimInspiralFD path, which supports TEOBResumS."""
    if args.tc != 0.0:
        raise ValueError("the public SimInspiralFD comparison supports only tc=0")
    started = time.perf_counter()
    approximant = lalsim.GetApproximantFromString("TEOBResumS")
    series = lalsim.SimInspiralFD(
        args.mass1 * lal.MSUN_SI, args.mass2 * lal.MSUN_SI,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        args.distance * 1.0e6 * lal.PC_SI, args.inclination, args.phic,
        0.0, 0.0, 0.0, args.delta_f, args.f_low, args.f_max, args.f_low,
        _public_options(), approximant,
    )
    if not isinstance(series, (list, tuple)) or len(series) != 2:
        raise TypeError(f"SimInspiralFD returned {type(series).__name__}, expected two series")
    hp, hc = series
    public_hp = np.asarray(hp.data.data, dtype=np.complex128).copy()
    public_hc = np.asarray(hc.data.data, dtype=np.complex128).copy()
    if public_hp.size == 0 or public_hc.size == 0:
        raise ValueError("SimInspiralFD returned empty polarization arrays")
    if public_hp.shape != public_hc.shape or not np.isfinite(public_hp).all() or not np.isfinite(public_hc).all():
        raise ValueError("SimInspiralFD returned invalid polarization arrays")
    if float(hp.deltaF) != float(hc.deltaF) or float(getattr(hp, "f0", 0.0)) != float(getattr(hc, "f0", 0.0)):
        raise ValueError("SimInspiralFD h_plus and h_cross axes differ")
    delta = float(hp.deltaF)
    f0 = float(getattr(hp, "f0", 0.0))
    frequencies = f0 + delta * np.arange(public_hp.size, dtype=np.float64)
    if f0 != 0.0 or delta != args.delta_f or frequencies[-1] != args.f_max:
        raise ValueError("SimInspiralFD returned an unexpected full frequency axis")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0) or np.any(frequencies > args.f_max):
        raise ValueError("public frequency axis violates the requested [0,f_max] bounds")
    if not np.any((frequencies >= args.f_low) & (frequencies <= args.f_max)):
        raise ValueError("public frequency axis has no bin in the requested [f_low,f_max] band")
    return ({"status": "ok", "method": "SimInspiralFD", "approximant": "TEOBResumS",
             "approximant_enum": int(approximant), "public_fd_implemented": bool(lalsim.SimInspiralImplementedFDApproximants(approximant)),
             "lal_version": getattr(lal, "__version__", None), "lalsimulation_version": getattr(lalsim, "__version__", None),
             "samples": int(public_hp.size), "delta_f_hz": delta, "f0_hz": f0,
             "frequency_start_hz": float(frequencies[0]), "frequency_end_hz": float(frequencies[-1]),
             "hp_epoch_seconds": float(hp.epoch), "hc_epoch_seconds": float(hc.epoch),
             "elapsed_seconds": time.perf_counter() - started}, frequencies, public_hp, public_hc)


def _verify_public_cache(path: Path, frequencies: np.ndarray, hp: np.ndarray, hc: np.ndarray) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"public FD cache is unavailable: {path}")
    with np.load(path, allow_pickle=False) as old:
        expected = (old["30p20_public_frequencies"], old["30p20_public_h_plus_fd"], old["30p20_public_h_cross_fd"])
        exact = [bool(np.array_equal(a, b)) for a, b in zip((frequencies, hp, hc), expected)]
        if not all(exact):
            raise ValueError(f"fresh SimInspiralFD arrays differ from saved public reference: {exact}")
        return {"path": str(path.resolve()), "sha256": _sha256(path), "exact_arrays": exact,
                "keys": ["30p20_public_frequencies", "30p20_public_h_plus_fd", "30p20_public_h_cross_fd"]}


def _source_paths() -> dict[str, Path]:
    # Keep this list explicit so the provenance remains independent of the
    # existing diagnostic's source list and includes this replay wrapper.
    names = {
        "replay_script": Path(__file__).resolve(),
        "existing_diagnostic": ROOT / "scripts/diagnose_teob_conditioned_waveform_clean.py",
        "producer_diagnostic": PRODUCER,
        "adapter": ROOT / "src/torchwave/approximants/eob/teob_conditioned_waveform_clean.py",
        "conditioning": ROOT / "src/torchwave/approximants/eob/td_fd_conditioning_clean.py",
        "conditioning_contract": ROOT / "docs/reviews/2026-09-13-parity-review/td-fd-conditioning-contract.md",
        "postadiabatic": ROOT / "src/torchwave/approximants/eob/teob_postadiabatic_clean.py",
        "adaptive_continuation": ROOT / "src/torchwave/approximants/eob/teob_adaptive_continuation_clean.py",
        "adaptive_driver": ROOT / "src/torchwave/approximants/eob/adaptive_driver_clean.py",
        "vacuum_driver": ROOT / "src/torchwave/approximants/eob/teob_vacuum_driver_clean.py",
        "trajectory": ROOT / "src/torchwave/approximants/eob/teob_trajectory.py",
        "radial_acceleration": ROOT / "src/torchwave/approximants/eob/teob_radial_acceleration_clean.py",
        "radiation": ROOT / "src/torchwave/approximants/eob/teob_radiation_clean.py",
        "state_mode": ROOT / "src/torchwave/approximants/eob/teob_waveform_state_clean.py",
        "hamiltonian": ROOT / "src/torchwave/approximants/teobresums.py",
        "nqc": ROOT / "src/torchwave/approximants/eob/teob_merger_nqc_clean.py",
        "nqc_basis": ROOT / "src/torchwave/approximants/eob/teob_nqc_clean.py",
        "ringdown": ROOT / "src/torchwave/approximants/eob/teob_ringdown_assembly_clean.py",
        "postmerger": ROOT / "src/torchwave/approximants/eob/teob_postmerger.py",
        "uib_remnant": ROOT / "src/torchwave/approximants/eob/teob_uib2016_remnant.py",
        "projection": ROOT / "src/torchwave/approximants/eob/conditioned_waveform_clean.py",
    }
    return {name: path for name, path in names.items() if path.is_file()}


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--radial-acceleration-policy", default="frozen-mobility")
    p.add_argument("--horizon-policy", default="leading-order")
    p.add_argument("--pa-initializer-policy", default="analytic")
    p.add_argument("--mass1", type=float, default=30.0)
    p.add_argument("--mass2", type=float, default=20.0)
    p.add_argument("--f-low", type=float, default=20.0)
    p.add_argument("--f-max", type=float, default=1024.0)
    p.add_argument("--delta-f", type=float, default=0.5)
    p.add_argument("--distance", type=float, default=100.0)
    p.add_argument("--inclination", type=float, default=0.0)
    p.add_argument("--phic", type=float, default=0.0)
    p.add_argument("--tc", type=float, default=0.0)
    p.add_argument("--r-min", type=float, default=14.0)
    p.add_argument("--dr", type=float, default=0.1)
    p.add_argument("--pa-order", type=int, default=8)
    p.add_argument("--fixed-step-cap", type=float, default=DEFAULT_FIXED_STEP_CAP)
    p.add_argument("--continuation-duration", type=float, default=DEFAULT_CONTINUATION_DURATION)
    p.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    p.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    p.add_argument("--max-rejected-steps", type=int, default=DEFAULT_MAX_REJECTED_STEPS)
    p.add_argument("--max-evaluations", type=int, default=DEFAULT_MAX_EVALUATIONS)
    p.add_argument("--max-accepted-steps", type=int, default=DEFAULT_MAX_ACCEPTED_STEPS)
    p.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--native-output", type=Path, default=DEFAULT_NATIVE)
    p.add_argument("--fd-output", type=Path, default=DEFAULT_FD)
    p.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    p.add_argument("--baseline-native", type=Path, default=BASELINE_NATIVE)
    p.add_argument("--baseline-fd", type=Path, default=BASELINE_FD)
    p.add_argument("--public-only", action="store_true", help="run and verify public SimInspiralFD only; never call the native adapter")
    return p.parse_args()


def main() -> int:
    args = _args()
    for path in (args.report, args.native_output, args.fd_output, args.log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    started = time.perf_counter()
    source_paths = _source_paths()
    source_before = {name: _sha256(path) for name, path in source_paths.items()}
    command = shlex.join([sys.executable, *sys.argv])
    report: dict[str, Any] = {
        "status": "running",
        "scope": "one clean native 30+20 TEOB replay and public FD two-polarization comparison",
        "run_control": {
            "host": socket.gethostname(), "cwd": os.getcwd(), "command": command,
            "pid": os.getpid(), "python": sys.executable, "platform": platform.platform(),
            "device": "cpu", "started_unix": time.time(),
            "log_path": str(args.log_path.resolve()),
            "expected_next_check": "inspect replay-report.json and replay-native-output.npz after process exit",
            "stop_command": f"kill {os.getpid()}",
            "resource_bound": "single CPU process; continuation max_duration=6000 s and max_evaluations=30000",
        },
        "request": {
            "mass1_msun": args.mass1, "mass2_msun": args.mass2,
            "f_low_hz": args.f_low, "f_max_hz": args.f_max, "delta_f_hz": args.delta_f,
            "distance_mpc": args.distance, "inclination": args.inclination,
            "phic": args.phic, "tc_seconds": args.tc,
            "chi1": [0.0, 0.0, 0.0], "chi2": [0.0, 0.0, 0.0],
            "lambda1": 0.0, "lambda2": 0.0, "mode_array": "[(2,+2)]",
            "pn_tidal_order": 0, "radial_acceleration_policy": args.radial_acceleration_policy,
            "horizon_policy": args.horizon_policy, "pa_initializer_policy": args.pa_initializer_policy,
            "r_min": args.r_min, "dr": args.dr, "pa_order": args.pa_order,
            "fixed_step_cap": args.fixed_step_cap, "continuation_duration": args.continuation_duration,
            "atol": args.atol, "rtol": args.rtol, "max_rejected_steps": args.max_rejected_steps,
            "max_evaluations": args.max_evaluations, "max_accepted_steps": args.max_accepted_steps,
            "max_duration": args.max_duration, "alignment": "none", "fitting": "none",
        },
        "output_paths": {"report": str(args.report.resolve()), "native": str(args.native_output.resolve()), "fd": str(args.fd_output.resolve())},
        "source_sha256_before": source_before,
        "baseline_inputs": {name: {"path": str(path.resolve()), "sha256": _sha256(path)} for name, path in (("native", args.baseline_native), ("fd", args.baseline_fd)) if path.is_file()},
        "public_versions": {},
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        if args.public_only:
            print("stage: public-only SimInspiralFD preflight", flush=True)
            public_meta, public_f, public_hp, public_hc = _public_fd_call(args)
            report["public_versions"] = {"lal": public_meta["lal_version"], "lalsimulation": public_meta["lalsimulation_version"]}
            report["public_preflight"] = {"public": public_meta, "cache_verification": _verify_public_cache(args.baseline_fd, public_f, public_hp, public_hc),
                                           "axis": _axis_metric(public_f, public_f), "native_ode_called": False}
            report["status"] = "complete"
            return 0
        print("stage: clean support plan and native evolution", flush=True)
        support_plan = plan_td_support(args.mass1, args.mass2, args.f_low, 0.0, 0.0, dtype=torch.float64, device="cpu")
        result = conditioned_teob_fd(
            args.mass1, args.mass2, f_low=args.f_low, f_max=args.f_max, delta_f=args.delta_f,
            distance=args.distance, inclination=args.inclination, phic=args.phic, tc=args.tc,
            radial_acceleration_policy=args.radial_acceleration_policy,
            horizon_policy=args.horizon_policy, pa_initializer_policy=args.pa_initializer_policy,
            pa_kwargs={"r_min": args.r_min, "dr": args.dr, "pa_order": args.pa_order},
            continuation_kwargs={"atol": args.atol, "rtol": args.rtol, "fixed_step_cap": args.fixed_step_cap,
                                 "continuation_duration": args.continuation_duration,
                                 "max_rejected_steps": args.max_rejected_steps, "max_evaluations": args.max_evaluations,
                                 "max_accepted_steps": args.max_accepted_steps, "max_duration": args.max_duration},
            dtype=torch.float64, device="cpu",
        )
        total_mass = _f(result.total_mass_seconds)
        eta = _f(result.support.eta)
        eta_plan = _f(support_plan.eta)
        eta_alternative = args.mass1 * args.mass2 / (args.mass1 + args.mass2) ** 2
        t_support = _f(result.support.support_time)
        power_base = 5.0 * total_mass / (eta * t_support)
        direct_f_start = math.pow(power_base, 3.0 / 8.0) / (8.0 * math.pi * total_mass)
        nested_f_start = math.pow(math.pow(power_base, 1.0 / 8.0), 3.0) / (8.0 * math.pi * total_mass)
        report["support"] = _support_summary(result.support)
        report["sampling"] = _sampling_summary(result.sampling, result.total_mass_seconds)
        report["support_start_arithmetic"] = {
            "formula": "f_start=(5*M/(eta*t_support))**(3/8)/(8*pi*M)",
            "total_mass_seconds": total_mass, "total_mass_seconds_hex": total_mass.hex(),
            "eta": eta, "eta_hex": eta.hex(), "eta_alternative_python_msun_ratio": eta_alternative,
            "eta_alternative_python_msun_ratio_hex": eta_alternative.hex(),
            "eta_independent_plan": eta_plan, "eta_independent_plan_hex": eta_plan.hex(),
            "t_support": t_support, "t_support_hex": t_support.hex(),
            "power_base": power_base, "power_base_hex": power_base.hex(),
            "direct_fractional_power_hz": direct_f_start, "direct_fractional_power_hz_hex": direct_f_start.hex(),
            "nested_eighth_root_then_cube_hz": nested_f_start, "nested_eighth_root_then_cube_hz_hex": nested_f_start.hex(),
            "adapter_start_frequency_hz": _f(result.support.start_frequency),
            "adapter_start_frequency_hz_hex": _hex(result.support.start_frequency),
            "adapter_omega_start_hex": _hex(result.omega_start),
        }
        report["native_start"] = {
            "omega_start": _f(result.omega_start), "omega_start_hex": _hex(result.omega_start),
            "initial_radius": _f(result.initial_radius), "initial_radius_hex": _hex(result.initial_radius),
            "formula": "max(r_min, omega_start**(-2/3))",
            "coordinate_epoch": _f(result.coordinate_epoch),
            "physical_epoch_seconds": _f(result.physical_epoch_seconds),
            "continuation_peak_event_time": _f(result.continuation.peak_event_time),
        }
        report["continuation"] = {"status": result.continuation.status, "samples": int(result.continuation.time.numel()),
                                   "statistics": _statistics_summary(result.continuation.statistics),
                                   "peak_event_index": result.continuation.peak_event_index,
                                   "peak_event_frequency": _f(result.continuation.peak_event_frequency)}
        cache_arrays = _cache_arrays(result)
        np.savez_compressed(args.native_output, **cache_arrays)
        report["native_cache"] = {"path": str(args.native_output.resolve()), "sha256": _sha256(args.native_output),
                                   "bytes": args.native_output.stat().st_size,
                                   "keys": sorted(cache_arrays),
                                   "array_shapes": {k: list(v.shape) for k, v in cache_arrays.items()}}

        print("stage: public SimInspiralFD and exact-index metrics", flush=True)
        public_meta, public_f, public_hp, public_hc = _public_fd_call(args)
        public_cache_verification = _verify_public_cache(args.baseline_fd, public_f, public_hp, public_hc)
        report["public_reference_cache_verification"] = public_cache_verification
        report["public_versions"] = {"lal": public_meta.get("lal_version"), "lalsimulation": public_meta.get("lalsimulation_version")}
        native_f = _tensor_numpy(result.frequencies)
        native_hp = _tensor_numpy(result.h_plus)
        native_hc = _tensor_numpy(result.h_cross)
        common = min(native_f.size, np.asarray(public_f).size)
        indices = np.arange(common, dtype=np.int64)
        exact = np.abs(native_f[:common] - np.asarray(public_f)[:common]) == 0.0
        in_band = (native_f[:common] >= args.f_low) & (native_f[:common] <= args.f_max)
        indices = indices[exact & in_band]
        if common != native_f.size or common != np.asarray(public_f).size or not np.all(exact) or not indices.size:
            raise ValueError("native/public frequency axes are not full exact matches with in-band bins")
        report["public_fd_comparison"] = {
            "api": "SimInspiralFD", "public": public_meta,
            "axis": _axis_metric(native_f, public_f), "compared_bins": int(indices.size),
            "frequency_start_hz": float(native_f[indices[0]]) if indices.size else None,
            "frequency_end_hz": float(native_f[indices[-1]]) if indices.size else None,
            "h_plus": _metric(native_hp[indices], np.asarray(public_hp)[indices]),
            "h_cross": _metric(native_hc[indices], np.asarray(public_hc)[indices]),
            "combined_polarizations": _metric(np.concatenate((native_hp[indices], native_hc[indices])),
                                               np.concatenate((np.asarray(public_hp)[indices], np.asarray(public_hc)[indices]))),
            "denominator": "fresh public SimInspiralFD norm on exact bins with f_low <= f <= f_max",
            "alignment": "none", "optimization": "none", "index_rule": "exact common frequency bins with f >= f_low",
        }
        np.savez_compressed(args.fd_output, native_frequencies=native_f, native_h_plus_fd=native_hp,
                            native_h_cross_fd=native_hc, public_frequencies=np.asarray(public_f),
                            public_h_plus_fd=np.asarray(public_hp), public_h_cross_fd=np.asarray(public_hc))
        report["public_fd_cache"] = {"path": str(args.fd_output.resolve()), "sha256": _sha256(args.fd_output), "bytes": args.fd_output.stat().st_size}

        if args.baseline_native.is_file():
            with np.load(args.baseline_native, allow_pickle=False) as old:
                old_f = np.asarray(old["frequencies"], dtype=np.float64)
                old_hp = np.asarray(old["h_plus_fd"], dtype=np.complex128)
                old_hc = np.asarray(old["h_cross_fd"], dtype=np.complex128)
                old_common = min(old_f.size, np.asarray(public_f).size)
                old_exact = np.abs(old_f[:old_common] - np.asarray(public_f)[:old_common]) == 0.0
                old_idx = np.arange(old_common, dtype=np.int64)[old_exact & (old_f[:old_common] >= args.f_low) & (old_f[:old_common] <= args.f_max)]
                if old_f.size != np.asarray(public_f).size or old_common != old_f.size or not np.all(old_exact) or not old_idx.size:
                    raise ValueError("baseline native/public frequency axes are not full exact matches")
                report["baseline_native_comparison"] = {
                    "native_cache_path": str(args.baseline_native.resolve()),
                    "denominator": "saved public SimInspiralFD norm on exact bins with f_low <= f <= f_max",
                    "axis": _axis_metric(old_f, public_f), "compared_bins": int(old_idx.size),
                    "h_plus": _metric(old_hp[old_idx], np.asarray(public_hp)[old_idx]),
                    "h_cross": _metric(old_hc[old_idx], np.asarray(public_hc)[old_idx]),
                    "combined_polarizations": _metric(np.concatenate((old_hp[old_idx], old_hc[old_idx])),
                                                       np.concatenate((np.asarray(public_hp)[old_idx], np.asarray(public_hc)[old_idx]))),
                }
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["source_sha256_after"] = {name: _sha256(path) for name, path in source_paths.items()}
        report["source_hashes_unchanged"] = report["source_sha256_before"] == report["source_sha256_after"]
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

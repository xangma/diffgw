#!/usr/bin/env python3
"""Cross-check the clean TEOB vector field with the public GSL RKF45 API.

This is a bounded, diagnostic-only integration.  The right-hand side is the
same clean zero-spin, zero-tide field used by the paired TEOB driver, while
the adaptive stepping and error control are supplied by the public GSL
``gsl_odeiv2`` API through ``ctypes``.  The trajectory starts at the exact
post-adiabatic endpoint stored in the frozen leading-order-horizon cache and
stops at the first accepted decrease of ``d phase / dt``.  No waveform is
assembled and no production route is changed.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.diagnose_teob_paired_driver_clean import (
    _LeadingOrderHorizonReaction,
    _hamiltonian,
)
from torchwave.approximants.eob.teob_vacuum_driver_clean import (
    TEOBVacuumRadiationDriver,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "verification/teob-paired-driver-clean/paired-physical-output-lo-horizon.npz"
DEFAULT_REPORT_DIR = (
    ROOT / "docs/reviews/2026-09-13-parity-review/verification/teob-gsl-public-driver-clean"
)
DEFAULT_GSL = Path(
    "/Users/xangma/miniconda3/envs/pycbcorig313/lib/python3.13/site-packages/lal/.dylibs/libgsl.28.dylib"
)
PUBLIC_PEAK_TIME = 12316.763885040122
DIMENSION = 4
EPSABS = 1.0e-13
EPSREL = 1.0e-11
MAX_ACCEPTED = 1000
MAX_DURATION = 6000.0
GSL_EBADFUNC = 9


RHS = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
)


class GSLSystem(ctypes.Structure):
    """ABI layout documented for ``gsl_odeiv2_system``."""

    _fields_ = [
        ("function", RHS),
        ("jacobian", ctypes.c_void_p),
        ("dimension", ctypes.c_size_t),
        ("params", ctypes.c_void_p),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _source_paths(gsl_path: Path) -> dict[str, Path]:
    """Only clean sources and the public binary used by this diagnostic."""

    return {
        "script": Path(__file__).resolve(),
        "paired_helper": ROOT / "scripts/diagnose_teob_paired_driver_clean.py",
        "vacuum_driver": ROOT / "src/torchwave/approximants/eob/teob_vacuum_driver_clean.py",
        "integrator": ROOT / "src/torchwave/approximants/eob/integrator.py",
        "hamiltonian": ROOT / "src/torchwave/approximants/teobresums.py",
        "radiation": ROOT / "src/torchwave/approximants/eob/teob_radiation_clean.py",
        "gsl_library": gsl_path.resolve(),
    }


class _Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def line(self, message: str) -> None:
        print(message, flush=True)
        self._handle.write(message + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--gsl-library", type=Path, default=DEFAULT_GSL)
    return parser.parse_args()


def _load_endpoint(path: Path) -> tuple[np.ndarray, float, float, float, str]:
    digest = _sha256(path)
    required = {"pa_radius", "pa_time", "pa_phase", "pa_pr_star", "pa_p_phi", "pa_frequency"}
    with np.load(path, allow_pickle=False) as data:
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"frozen cache is missing {sorted(missing)}")
        arrays = {name: np.asarray(data[name], dtype=np.float64) for name in required}
    lengths = {name: array.shape for name, array in arrays.items()}
    if any(array.ndim != 1 or array.size < 2 for array in arrays.values()):
        raise ValueError(f"pa arrays must be one-dimensional with at least two samples: {lengths}")
    n = arrays["pa_time"].size
    if any(array.size != n for array in arrays.values()):
        raise ValueError(f"pa arrays have inconsistent shapes: {lengths}")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("pa arrays contain nonfinite values")
    if not np.all(np.diff(arrays["pa_time"]) > 0.0):
        raise ValueError("pa_time must be strictly increasing")
    # Match the clean continuation contract: the first adaptive proposal is
    # half of the final PA interval, rather than the full sparse-knot gap.
    h0 = float(0.5 * (arrays["pa_time"][-1] - arrays["pa_time"][-2]))
    t0 = float(arrays["pa_time"][-1])
    state = np.array(
        [
            arrays["pa_radius"][-1],
            arrays["pa_phase"][-1],
            arrays["pa_pr_star"][-1],
            arrays["pa_p_phi"][-1],
        ],
        dtype=np.float64,
    )
    if not math.isfinite(h0) or h0 <= 0.0:
        raise ValueError(f"initial h must be finite and positive, got {h0!r}")
    if state[0] <= 0.0:
        raise ValueError("endpoint radius must be positive")
    return state, t0, h0, float(arrays["pa_frequency"][-1]), digest


def _configure_library(path: Path) -> tuple[ctypes.CDLL, ctypes.c_void_p]:
    library = ctypes.CDLL(str(path))
    library.gsl_odeiv2_step_alloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.gsl_odeiv2_step_alloc.restype = ctypes.c_void_p
    library.gsl_odeiv2_control_y_new.argtypes = [ctypes.c_double, ctypes.c_double]
    library.gsl_odeiv2_control_y_new.restype = ctypes.c_void_p
    library.gsl_odeiv2_evolve_alloc.argtypes = [ctypes.c_size_t]
    library.gsl_odeiv2_evolve_alloc.restype = ctypes.c_void_p
    library.gsl_odeiv2_evolve_apply.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(GSLSystem),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    library.gsl_odeiv2_evolve_apply.restype = ctypes.c_int
    library.gsl_odeiv2_step_free.argtypes = [ctypes.c_void_p]
    library.gsl_odeiv2_step_free.restype = None
    library.gsl_odeiv2_control_free.argtypes = [ctypes.c_void_p]
    library.gsl_odeiv2_control_free.restype = None
    library.gsl_odeiv2_evolve_free.argtypes = [ctypes.c_void_p]
    library.gsl_odeiv2_evolve_free.restype = None
    if hasattr(library, "gsl_set_error_handler_off"):
        library.gsl_set_error_handler_off.argtypes = []
        library.gsl_set_error_handler_off.restype = ctypes.c_void_p
        library.gsl_set_error_handler_off()
    # The exported object is a pointer to the public rkf45 step type.
    rkf45 = ctypes.c_void_p.in_dll(library, "gsl_odeiv2_step_rkf45")
    if not rkf45.value:
        raise RuntimeError("gsl_odeiv2_step_rkf45 is null")
    return library, rkf45


def _status_name(library: ctypes.CDLL, status: int) -> str:
    if status == 0:
        return "GSL_SUCCESS"
    if hasattr(library, "gsl_strerror"):
        library.gsl_strerror.argtypes = [ctypes.c_int]
        library.gsl_strerror.restype = ctypes.c_char_p
        value = library.gsl_strerror(status)
        if value:
            return value.decode("utf-8", errors="replace")
    return f"gsl_status_{status}"


def _integrate(
    library: ctypes.CDLL,
    rkf45: ctypes.c_void_p,
    initial_state: np.ndarray,
    t0: float,
    h0: float,
    logger: _Logger,
) -> dict[str, Any]:
    hamiltonian = _hamiltonian()
    reaction = _LeadingOrderHorizonReaction()
    driver = TEOBVacuumRadiationDriver(hamiltonian, reaction)
    callback_state: dict[str, Any] = {"calls": 0, "exceptions": []}

    def evaluate(values: np.ndarray) -> np.ndarray:
        state = torch.as_tensor(values, dtype=torch.float64)
        with torch.inference_mode():
            derivative = driver.vector_field(state)
        result = derivative.detach().cpu().numpy().astype(np.float64, copy=False)
        if result.shape != (DIMENSION,) or not np.isfinite(result).all():
            raise FloatingPointError(f"clean vector field returned {result!r}")
        return result

    @RHS
    def callback(t: float, y: ctypes.POINTER(ctypes.c_double), dydt: ctypes.POINTER(ctypes.c_double), _params: ctypes.c_void_p) -> int:
        callback_state["calls"] += 1
        values = np.ctypeslib.as_array(y, shape=(DIMENSION,)).copy()
        try:
            derivative = evaluate(values)
            for index, value in enumerate(derivative):
                dydt[index] = float(value)
            return 0
        except BaseException as error:  # Never allow a Python exception across the C ABI.
            if len(callback_state["exceptions"]) < 8:
                callback_state["exceptions"].append(
                    {"time": float(t), "state": values.tolist(), "exception": repr(error)}
                )
            return GSL_EBADFUNC

    system = GSLSystem(callback, None, DIMENSION, None)
    step = library.gsl_odeiv2_step_alloc(rkf45, DIMENSION)
    control = library.gsl_odeiv2_control_y_new(EPSABS, EPSREL)
    evolve = library.gsl_odeiv2_evolve_alloc(DIMENSION)
    if not step or not control or not evolve:
        for pointer, free in (
            (evolve, library.gsl_odeiv2_evolve_free),
            (control, library.gsl_odeiv2_control_free),
            (step, library.gsl_odeiv2_step_free),
        ):
            if pointer:
                free(pointer)
        raise MemoryError("GSL ODE allocation failed")

    accepted_time: list[float] = []
    accepted_state: list[list[float]] = []
    accepted_frequency: list[float] = []
    accepted_h_proposed: list[float] = []
    accepted_h_next: list[float] = []
    status_counts: dict[str, int] = {}
    t = ctypes.c_double(t0)
    h = ctypes.c_double(h0)
    y = (ctypes.c_double * DIMENSION)(*initial_state.tolist())
    t_limit = t0 + MAX_DURATION
    initial_derivative = evaluate(initial_state)
    previous_frequency = float(initial_derivative[1])
    event: dict[str, Any] | None = None
    error_status: dict[str, Any] | None = None
    logger.line(
        f"GSL start pid={os.getpid()} t0={t0:.17g} h0={h0:.17g} t_limit={t_limit:.17g}"
    )
    started = time.perf_counter()
    try:
        for index in range(1, MAX_ACCEPTED + 1):
            proposed = float(h.value)
            status = int(
                library.gsl_odeiv2_evolve_apply(
                    evolve,
                    control,
                    step,
                    ctypes.byref(system),
                    ctypes.byref(t),
                    t_limit,
                    ctypes.byref(h),
                    y,
                )
            )
            status_name = _status_name(library, status)
            status_counts[status_name] = status_counts.get(status_name, 0) + 1
            if status != 0:
                error_status = {
                    "code": status,
                    "name": status_name,
                    "time": float(t.value),
                    "h": float(h.value),
                }
                break
            current_time = float(t.value)
            current_state = np.ctypeslib.as_array(y, shape=(DIMENSION,)).astype(np.float64, copy=True)
            try:
                derivative = evaluate(current_state)
            except BaseException as error:
                error_status = {
                    "code": GSL_EBADFUNC,
                    "name": "clean_vector_field_error",
                    "time": current_time,
                    "h": float(h.value),
                    "exception": repr(error),
                }
                break
            frequency = float(derivative[1])
            accepted_time.append(current_time)
            accepted_state.append(current_state.tolist())
            accepted_frequency.append(frequency)
            accepted_h_proposed.append(proposed)
            accepted_h_next.append(float(h.value))
            if frequency < previous_frequency:
                event = {
                    "accepted_index": index,
                    "time": current_time,
                    "frequency": frequency,
                    "previous_frequency": previous_frequency,
                    "state": current_state.tolist(),
                    "h_proposed": proposed,
                    "h_next": float(h.value),
                }
                logger.line(
                    f"first accepted dphase/dt decrease at index={index} "
                    f"t={current_time:.17g} omega={frequency:.17g}"
                )
                break
            previous_frequency = frequency
            if index == 1 or index % 100 == 0:
                logger.line(
                    f"accepted={index} t={current_time:.17g} omega={frequency:.17g} "
                    f"h_next={float(h.value):.6g} rhs_calls={callback_state['calls']}"
                )
            if current_time >= t_limit:
                break
    finally:
        elapsed = time.perf_counter() - started
        # The callback and system remain alive until every GSL object is freed.
        library.gsl_odeiv2_evolve_free(evolve)
        library.gsl_odeiv2_control_free(control)
        library.gsl_odeiv2_step_free(step)
    logger.line(
        f"GSL end accepted={len(accepted_time)} rhs_calls={callback_state['calls']} "
        f"elapsed={elapsed:.6f}s"
    )
    event_delta = None if event is None else float(event["time"] - PUBLIC_PEAK_TIME)
    return {
        "initial_derivative": initial_derivative.tolist(),
        "initial_frequency": float(initial_derivative[1]),
        "accepted_time": accepted_time,
        "accepted_state": accepted_state,
        "accepted_frequency": accepted_frequency,
        "accepted_h_proposed": accepted_h_proposed,
        "accepted_h_next": accepted_h_next,
        "accepted_count": len(accepted_time),
        "rhs_callback_calls": int(callback_state["calls"]),
        "callback_exceptions": callback_state["exceptions"],
        "status_counts": status_counts,
        "error_status": error_status,
        "event": event,
        "event_delta_to_public_peak_M": event_delta,
        "elapsed_seconds": elapsed,
        "t_limit": t_limit,
    }


def _write_report(
    report_dir: Path,
    report: dict[str, Any],
    trajectory: dict[str, Any],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = report_dir / "gsl-trajectory.npz"
    np.savez_compressed(
        trajectory_path,
        accepted_time=np.asarray(trajectory["accepted_time"], dtype=np.float64),
        accepted_state=np.asarray(trajectory["accepted_state"], dtype=np.float64),
        accepted_frequency=np.asarray(trajectory["accepted_frequency"], dtype=np.float64),
        accepted_h_proposed=np.asarray(trajectory["accepted_h_proposed"], dtype=np.float64),
        accepted_h_next=np.asarray(trajectory["accepted_h_next"], dtype=np.float64),
    )
    report["trajectory_artifact"] = {
        "path": str(trajectory_path),
        "sha256": _sha256(trajectory_path),
        "arrays": [
            "accepted_time",
            "accepted_state",
            "accepted_frequency",
            "accepted_h_proposed",
            "accepted_h_next",
        ],
    }
    report["trajectory"] = trajectory
    report_path = report_dir / "diagnostic.json"
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    report_dir = args.report_dir.resolve()
    log_path = report_dir / "gsl-driver.log"
    logger = _Logger(log_path)
    started_unix = time.time()
    command = " ".join(sys.argv)
    source_paths = _source_paths(args.gsl_library)
    source_before = {name: _sha256(path) for name, path in source_paths.items()}
    report: dict[str, Any] = {
        "status": "running",
        "diagnostic_scope": (
            "Public GSL gsl_odeiv2 RKF45 adaptive integration of the clean "
            "TEOB zero-spin, zero-tide LO-horizon vector field; no waveform assembly."
        ),
        "run_control": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "python": sys.executable,
            "pid": os.getpid(),
            "command": command,
            "log_path": str(log_path),
            "started_unix": started_unix,
            "expected_next_check": "After the bounded RKF45 trajectory completes, inspect diagnostic.json and gsl-trajectory.npz.",
            "stop_command": f"kill {os.getpid()}",
        },
        "gsl": {
            "library_path": str(args.gsl_library.resolve()),
            "step_type": "gsl_odeiv2_step_rkf45",
            "dimension": DIMENSION,
            "control": "gsl_odeiv2_control_y_new",
            "epsabs": EPSABS,
            "epsrel": EPSREL,
            "evolve": "gsl_odeiv2_evolve_alloc/evolve_apply",
        },
        "input": {
            "cache_path": str(args.input.resolve()),
            "cache_sha256": None,
        },
        "source_sha256_before": source_before,
    }
    # Cache is intentionally hashed separately from executable/source inputs.
    report["input"]["cache_sha256"] = _sha256(args.input)
    logger.line(f"started pid={os.getpid()} cwd={Path.cwd()} log={log_path}")
    trajectory: dict[str, Any] = {}
    try:
        initial_state, t0, h0, pa_frequency_endpoint, cache_sha256 = _load_endpoint(args.input)
        report["input"].update(
            {
                "cache_sha256": cache_sha256,
                "endpoint_state": initial_state.tolist(),
                "endpoint_time_M": t0,
                "initial_h_M": h0,
                "pa_endpoint_fields": ["pa_radius", "pa_phase", "pa_pr_star", "pa_p_phi"],
                "pa_frequency_endpoint": pa_frequency_endpoint,
            }
        )
        if not args.gsl_library.exists():
            raise FileNotFoundError(args.gsl_library)
        library, rkf45 = _configure_library(args.gsl_library)
        trajectory = _integrate(library, rkf45, initial_state, t0, h0, logger)
        report["status"] = "completed" if trajectory["event"] is not None else "completed_without_event"
        report["public_peak_time_M"] = PUBLIC_PEAK_TIME
        report["public_peak_comparison"] = {
            "event_time_M": None if trajectory["event"] is None else trajectory["event"]["time"],
            "delta_M": trajectory["event_delta_to_public_peak_M"],
            "interpretation": "Timing comparison only; no waveform or phase alignment is performed.",
        }
    except BaseException as error:
        report["status"] = "error"
        report["exception"] = repr(error)
        logger.line(f"ERROR {error!r}")
    finally:
        report["elapsed_seconds"] = time.time() - started_unix
        source_after = {name: _sha256(path) for name, path in source_paths.items()}
        report["source_sha256_after"] = source_after
        report["source_hashes_unchanged"] = source_before == source_after
        report["run_control"]["finished_unix"] = time.time()
        report["run_control"]["elapsed_seconds"] = report["elapsed_seconds"]
        _write_report(report_dir, report, trajectory)
        logger.line(f"report={report_dir / 'diagnostic.json'} status={report['status']}")
        logger.close()
    return 0 if report["status"].startswith("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

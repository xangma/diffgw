#!/usr/bin/env python3
"""Bounded clean-room reconstruction using an exact independent N=61 grid.

This is a separate follow-up to ``diagnose_london_qnm_fit_clean.py``.  It
solves every N=61 training root afresh with the independent Leaver solver;
there is no interpolation from the previous N=257 roots and no production
coefficient change.  The N=61 count is an explicitly declared metadata
hypothesis inferred from the public four-decimal coefficient rounding.  The
paper does not publish that historical count.

The fit basis and public comparison are delegated to the original clean-room
helper, which is imported read-only.  Numerical roots are still generated in
this script on the exact N=61 grid.  The 33 fixed check nodes are solved with
no training-grid seed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
HELPER_PATH = ROOT / "scripts" / "diagnose_london_qnm_fit_clean.py"
SOLVER_PATH = ROOT / "scripts" / "kerr_leaver_solver.py"
PRINTED_PATH = ROOT / "src" / "torchwave" / "approximants" / "kerr_qnm_london_published.py"

EXACT_GRID_COUNT = 61
SPIN_LIMIT = 0.995
MODES = ((2, 2, 0), (2, 1, 0), (3, 3, 0), (3, 2, 0), (4, 4, 0), (4, 3, 0))
ZERO_DAMPED = frozenset(((2, 2, 0), (3, 3, 0), (4, 4, 0), (4, 3, 0)))
CHECK_NODE_COUNT = 33
CHECK_NODE_OFFSET = 0.37


def _load_helper():
    """Import the original diagnostic as a read-only numerical helper."""
    if str(HELPER_PATH.parent) not in sys.path:
        sys.path.insert(0, str(HELPER_PATH.parent))
    import diagnose_london_qnm_fit_clean as helper  # type: ignore

    return helper


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert NumPy values and complex roots to strict JSON."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def exact_theta_grid(count: int = EXACT_GRID_COUNT) -> np.ndarray:
    """Return the declared exact N=61 theta grid."""
    if count != EXACT_GRID_COUNT:
        raise ValueError(f"count must be the declared exact count {EXACT_GRID_COUNT}")
    return np.linspace(-math.pi / 2, math.pi / 2, count, dtype=float)


def exact_spin_grid(count: int = EXACT_GRID_COUNT) -> np.ndarray:
    """Return ``a=0.995*sin(theta)`` at exact N=61 nodes."""
    return SPIN_LIMIT * np.sin(exact_theta_grid(count))


def check_theta_grid() -> np.ndarray:
    """Return the fixed off-grid nodes used by the original diagnostic."""
    return -math.pi / 2 + (np.arange(CHECK_NODE_COUNT, dtype=float) + CHECK_NODE_OFFSET) * (
        math.pi / CHECK_NODE_COUNT
    )


def _source_hashes() -> dict[str, str]:
    return {
        "scripts/diagnose_london_qnm_fit_exact61.py": _sha256(SCRIPT_PATH),
        "scripts/diagnose_london_qnm_fit_clean.py": _sha256(HELPER_PATH),
        "scripts/kerr_leaver_solver.py": _sha256(SOLVER_PATH),
        "src/torchwave/approximants/kerr_qnm_london_published.py": _sha256(PRINTED_PATH),
    }


def _versions() -> dict[str, Any]:
    result: dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__}
    for package in ("scipy", "gmpy2"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def generate_exact_training_grid(mode: tuple[int, int, int], solve: Any | None = None) -> list[dict[str, Any]]:
    """Generate all exact N=61 roots by fresh center and spin continuation.

    Continuation seeds are roots solved earlier on this same N=61 grid.  No
    root from the N=257 diagnostic, a QNM table, or an interpolant is read.
    """
    helper = _load_helper()
    solve = helper._load_solver() if solve is None else solve
    theta = exact_theta_grid()
    spins = exact_spin_grid()
    center = EXACT_GRID_COUNT // 2
    rows: list[dict[str, Any] | None] = [None] * EXACT_GRID_COUNT

    center_root = helper._solve_certified(solve, 0.0, mode)
    rows[center] = helper._root_row(theta[center], spins[center], center_root)

    for indices in (range(center - 1, -1, -1), range(center + 1, EXACT_GRID_COUNT)):
        previous_spin = 0.0
        previous_root = center_root
        for index in indices:
            root, refinements = helper._continued_point(
                solve, float(spins[index]), mode, previous_spin, previous_root
            )
            rows[index] = helper._root_row(theta[index], spins[index], root, refinements)
            previous_spin = float(spins[index])
            previous_root = root

    if any(row is None for row in rows):
        raise AssertionError(f"incomplete exact N=61 continuation for {mode}")
    return [row for row in rows if row is not None]


def _fit_exact_rows(
    rows: Sequence[Mapping[str, Any]],
    check_rows: Sequence[Mapping[str, Any]],
    mode: tuple[int, int, int],
) -> dict[str, Any]:
    """Fit one exact grid and attach training/check/rounding diagnostics."""
    helper = _load_helper()
    spins, values = helper._root_values(rows)
    coefficients, quality = helper.fit_coefficients(spins, values, mode, "main")
    result: dict[str, Any] = {
        "count": EXACT_GRID_COUNT,
        "grid_definition": "exact fresh N=61 theta grid; no interpolation",
        "theta_nodes": [float(row["theta"]) for row in rows],
        "spin_nodes": spins,
        "fit": quality,
        "coefficients_rectangular": coefficients,
        "coefficients_polar": [helper._polar(complex(value)) for value in coefficients],
        "printed_comparison": helper.compare_printed(coefficients, mode),
        "independent_check_main": helper._check_fit(check_rows, mode, coefficients, "main"),
    }
    if mode in ZERO_DAMPED:
        direct_coefficients, direct_quality = helper.fit_coefficients(spins, values, mode, "direct_weighted")
        direct = {
            **direct_quality,
            "coefficients_rectangular": direct_coefficients,
            "coefficients_polar": [helper._polar(complex(value)) for value in direct_coefficients],
            "printed_comparison": helper.compare_printed(direct_coefficients, mode),
            "independent_check": helper._check_fit(check_rows, mode, direct_coefficients, "direct_weighted"),
        }
        result["direct_constrained_weighted_alternative"] = direct
    return result


def _mode_result(
    mode: tuple[int, int, int],
    rows: Sequence[Mapping[str, Any]],
    check_rows: Sequence[Mapping[str, Any]],
    raw_dir: Path,
) -> dict[str, Any]:
    helper = _load_helper()
    label = ",".join(str(value) for value in mode)
    training_path = raw_dir / f"mode-{label}-grid61.json"
    check_path = raw_dir / f"mode-{label}-check.json"
    _write_json(training_path, rows)
    _write_json(check_path, check_rows)
    fit = _fit_exact_rows(rows, check_rows, mode)
    return {
        "mode": mode,
        "fit_result": fit,
        "root_summary": {
            "training_root_count": len(rows),
            "check_root_count": len(check_rows),
            "all_training_roots_converged": all(bool(row["converged"]) for row in rows),
            "all_check_roots_converged": all(bool(row["converged"]) for row in check_rows),
            "max_cf_depth_training": max(int(row["cf_depth"]) for row in rows),
            "max_cf_depth_check": max(int(row["cf_depth"]) for row in check_rows),
            "max_estimated_frequency_error_training": max(
                float(row["metadata"]["estimated_frequency_error"]) for row in rows
            ),
            "max_estimated_frequency_error_check": max(
                float(row["metadata"]["estimated_frequency_error"]) for row in check_rows
            ),
            "max_continuation_refinement_steps_training": max(
                int(row["continuation_refinement_steps"]) for row in rows
            ),
            "max_continuation_refinement_steps_check": max(
                int(row["continuation_refinement_steps"]) for row in check_rows
            ),
        },
        "raw_artifacts": {
            "training_path": str(training_path),
            "training_sha256": _sha256(training_path),
            "check_path": str(check_path),
            "check_sha256": _sha256(check_path),
        },
    }


def _parse_modes(parser: argparse.ArgumentParser, labels: Sequence[str] | None) -> tuple[tuple[int, int, int], ...]:
    if not labels:
        return MODES
    try:
        modes = tuple(tuple(int(value) for value in label.split(",")) for label in labels)
    except (TypeError, ValueError):
        parser.error("modes must use labels like 2,2,0")
    if any(len(mode) != 3 or mode not in MODES for mode in modes) or len(set(modes)) != len(modes):
        parser.error("modes must be a subset of the predeclared six fundamentals")
    return modes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "reviews" / "2026-09-13-parity-review" / "verification" / "london-qnm-fit-exact61",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("/tmp/torchwave-london-qnm-fit-exact61"))
    parser.add_argument("--log-path", type=Path, help="path recorded in the report for the redirected job log")
    parser.add_argument("--modes", nargs="+", help="optional subset, labels like 2,2,0")
    args = parser.parse_args(argv)

    modes = _parse_modes(parser, args.modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    solve = _load_helper()._load_solver()
    source_hashes_before = _source_hashes()
    started = time.monotonic()
    command = [sys.executable, str(SCRIPT_PATH), *(argv if argv is not None else sys.argv[1:])]
    report: dict[str, Any] = {
        "provenance": {
            "paper": "L. London and E. Fauchon-Jones, arXiv:1810.03550v2",
            "paper_url": "https://arxiv.org/html/1810.03550",
            "equations": "Eqs. (4)-(6), (21)-(32); Fig. 1 caption",
            "source": "published rendered paper plus the fresh independent Leaver continued-fraction solver",
            "solver_source": "scripts/kerr_leaver_solver.py: independent radial continued fraction and spheroidal angular eigenproblem",
            "solver_convention": "exp(-i omega t), omega=omega_r-i*omega_d",
            "paper_convention": "tilde{omega}=omega_r+i/tau; independent roots conjugated before fit",
            "printed_reference": "src/torchwave/approximants/kerr_qnm_london_published.py, public four-decimal transcription",
            "clean_room_scope": "No LAL, PyCBC, TEOB implementation, author implementation, oracle-fitting, calibration files, waveform outputs, or public target binary used",
            "historical_limit": "The paper does not specify the historical training count or all GMVP selections; this is a bounded independent reconstruction.",
            "count_hypothesis": "N=61 is a metadata hypothesis inferred from public coefficient rounding, not a published fact.",
            "count_uncertainty": "A different historical training count, node ordering, hidden GMVP subset, precision, or fit preprocessing could produce different coefficients; this run does not identify those unpublished choices.",
        },
        "choices": {
            "modes": modes,
            "overtone": 0,
            "spin_limit": SPIN_LIMIT,
            "spin_grid": "a=0.995*sin(theta), theta uniform on [-pi/2, pi/2]",
            "exact_training_grid_count": EXACT_GRID_COUNT,
            "grid_generation": "fresh exact N=61 grid; each continuation seed is an earlier root on this same grid",
            "interpolation": "none",
            "zero_damped_modes": sorted(ZERO_DAMPED),
            "zero_damped_powers": [0, 1, 2, 3, 4],
            "zero_damped_baseline": "m/2",
            "nonzero_damped_modes": [(2, 1, 0), (3, 2, 0)],
            "nonzero_damped_powers": list(range(7)),
            "least_squares": "numpy.linalg.lstsq, complex values, rcond=None",
            "main_zero_damped_target": "(omega-m/2)/kappa",
            "direct_alternative": "omega=m/2+sum(c_k*kappa**(k+1)); residual multiplier kappa**(-1/2), sample weight 1/kappa; reported sensitivity only",
            "check_nodes": f"{CHECK_NODE_COUNT} fixed nodes theta=-pi/2+(i+{CHECK_NODE_OFFSET})*pi/{CHECK_NODE_COUNT}; no training seeds",
            "solver_certification": "strict=True; reject converged=False, nonphysical roots, or estimated frequency error >=1e-9",
            "branch_recovery": "fixed midpoint continuation only after a solver failure or jump; no fit-count adaptation",
        },
        "execution": {
            "host": socket.gethostname(),
            "cwd": str(Path.cwd()),
            "command": command,
            "pid": os.getpid(),
            "log_path": str(args.log_path) if args.log_path else None,
            "next_check": "inspect progress.json after each completed mode; final report.json is written after all selected modes",
            "stop_command": f"kill {os.getpid()}",
        },
        "source_hashes_before": source_hashes_before,
        "versions": _versions(),
        "modes": {},
        "status": "running",
    }
    _write_json(args.output_dir / "progress.json", report)

    for mode_index, mode in enumerate(modes):
        label = ",".join(str(value) for value in mode)
        print(f"generating exact N=61 mode={label} ({mode_index + 1}/{len(modes)})", flush=True)
        mode_started = time.monotonic()
        rows = generate_exact_training_grid(mode, solve)
        helper = _load_helper()
        check_rows = helper._check_rows_for_mode(mode, solve)
        mode_result = _mode_result(mode, rows, check_rows, args.raw_dir)
        mode_result["elapsed_seconds"] = time.monotonic() - mode_started
        report["modes"][label] = mode_result
        _write_json(args.output_dir / "progress.json", report)
        print(f"completed exact N=61 mode={label} elapsed={mode_result['elapsed_seconds']:.2f}s", flush=True)

    source_hashes_after = _source_hashes()
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("clean-room inputs changed during exact N=61 reconstruction")
    report["source_hashes_after"] = source_hashes_after
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "completed"
    _write_json(args.output_dir / "report.json", report)
    _write_json(args.output_dir / "progress.json", report)
    print(f"wrote {args.output_dir / 'report.json'} in {report['elapsed_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

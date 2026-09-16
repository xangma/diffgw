#!/usr/bin/env python3
"""Bounded clean-room reconstruction using an exact independent N=61 grid.

This script delegates the clean-room London N=61 Kerr QNM fitting and evaluation
logic to ``torchwave.qnm.london`` while retaining diagnostics and compatibility
provenance checks.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
_SRC = str(ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from torchwave.qnm import london  # noqa: E402
from torchwave.qnm.london import (  # noqa: E402
    EXACT_GRID_COUNT,
    SPIN_LIMIT,
    MODES,
    ZERO_DAMPED,
    NONZERO_DAMPED,
    CHECK_NODE_COUNT,
    CHECK_NODE_OFFSET,
    exact_theta_grid,
    exact_spin_grid,
    check_theta_grid,
    kappa_value,
    design_matrix,
    evaluate_coefficients,
    fit_coefficients,
    compare_printed,
    generate_exact_training_grid,
    fit_exact_rows,
    _fit_exact_rows,
    _load_solver,
    _mode_result,
    _root_values,
    _check_fit,
    _check_rows_for_mode,
    _solve_certified,
    _continued_point,
    _root_row,
    _polar,
    _wrap_phase,
    _jsonable,
    _write_json,
    _versions,
    _parse_modes,
)

__all__ = [
    "EXACT_GRID_COUNT",
    "SPIN_LIMIT",
    "MODES",
    "ZERO_DAMPED",
    "NONZERO_DAMPED",
    "CHECK_NODE_COUNT",
    "CHECK_NODE_OFFSET",
    "exact_theta_grid",
    "exact_spin_grid",
    "check_theta_grid",
    "kappa_value",
    "design_matrix",
    "evaluate_coefficients",
    "fit_coefficients",
    "compare_printed",
    "generate_exact_training_grid",
    "fit_exact_rows",
    "_fit_exact_rows",
    "_load_solver",
    "_load_helper",
    "_mode_result",
    "_root_values",
    "_check_fit",
    "_check_rows_for_mode",
    "_solve_certified",
    "_continued_point",
    "_root_row",
    "_polar",
    "_wrap_phase",
    "_jsonable",
    "_write_json",
    "_versions",
    "_parse_modes",
    "_source_hashes",
    "main",
]

SCRIPT_PATH = Path(__file__).resolve()
HELPER_PATH = ROOT / "scripts" / "diagnose_london_qnm_fit_clean.py"
SOLVER_PATH = ROOT / "scripts" / "kerr_leaver_solver.py"
PRINTED_PATH = ROOT / "src" / "torchwave" / "approximants" / "kerr_qnm_london_published.py"


def _load_helper():
    """Return the clean-room numerical helper module."""
    return london


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        "scripts/diagnose_london_qnm_fit_exact61.py": _sha256(SCRIPT_PATH),
        "scripts/diagnose_london_qnm_fit_clean.py": _sha256(HELPER_PATH),
        "scripts/kerr_leaver_solver.py": _sha256(SOLVER_PATH),
        "src/torchwave/approximants/kerr_qnm_london_published.py": _sha256(PRINTED_PATH),
    }


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
    solve = _load_solver()
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
        check_rows = _check_rows_for_mode(mode, solve)
        mode_res = _mode_result(mode, rows, check_rows, args.raw_dir)
        mode_res["elapsed_seconds"] = time.monotonic() - mode_started
        report["modes"][label] = mode_res
        _write_json(args.output_dir / "progress.json", report)
        print(f"completed exact N=61 mode={label} elapsed={mode_res['elapsed_seconds']:.2f}s", flush=True)

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

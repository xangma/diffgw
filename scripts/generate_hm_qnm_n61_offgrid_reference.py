#!/usr/bin/env python3
"""Regenerate independent, off-grid Kerr roots for the HM N61 fit gate.

This script reads no HM fit coefficients or waveform implementation.  The
roots come from the clean-room Cook--Zalutskiy continued-fraction solver.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "src" / "torchwave" / "qnm" / "solver.py"
DEFAULT_OUTPUT = ROOT / "tests" / "data" / "hm_qnm_n61_offgrid_roots.json"
MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3))
SPINS = (-0.71, 0.37, 0.79)
N61_GRID = tuple(0.995 * math.sin(-math.pi / 2 + j * math.pi / 60) for j in range(61))
MIN_GRID_DISTANCE = 1e-3
MAX_SOLVER_FREQUENCY_ERROR = 1e-9


def _load_solver():
    """Load only the independent numerical module, avoiding package imports."""
    name = "diffgw_hm_offgrid_kerr_solver"
    spec = importlib.util.spec_from_file_location(name, SOLVER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load independent solver at {SOLVER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.solve


def _pair(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def generate() -> dict:
    """Solve all declared points without fit-derived seeds or tuning."""
    solve = _load_solver()
    rows = []
    for spin in SPINS:
        distance = min(abs(spin - node) for node in N61_GRID)
        if distance <= MIN_GRID_DISTANCE:
            raise ValueError(f"spin {spin} is too close to the N61 training-grid hypothesis")
        for ell, m in MODES:
            root = solve(spin, ell, m, 0, strict=True)
            error = float(root.metadata["estimated_frequency_error"])
            if (
                not root.converged
                or not math.isfinite(error)
                or error >= MAX_SOLVER_FREQUENCY_ERROR
                or root.omega.real <= 0
                or root.omega.imag >= 0
            ):
                raise RuntimeError(f"uncertified independent root for spin={spin}, mode={(ell, m, 0)}: {root}")
            rows.append(
                {
                    "spin": spin,
                    "mode": [ell, m, 0],
                    "omega_exp_minus_iwt": _pair(root.omega),
                    "angular_eigenvalue": _pair(root.separation_constant),
                    "radial_residual": _pair(root.radial_residual),
                    "cf_depth": int(root.cf_depth),
                    "cf_error": float(root.cf_error),
                    "angular_error": float(root.angular_error),
                    "estimated_frequency_error": error,
                    "branch_origin": str(root.metadata["branch_origin"]),
                    "grid_distance": distance,
                }
            )
    return {
        "specification_id": "hm.qnm.n61.offgrid-six-mode/v1",
        "reference": "Cook--Zalutskiy 2014, arXiv:1410.7698v2, Eqs. 23--56",
        "reference_solver_sha256": hashlib.sha256(SOLVER.read_bytes()).hexdigest(),
        "variant": "n61_main; hypothesized 61-node a=0.995 sin(theta) training grid",
        "spins": list(SPINS),
        "modes": [[ell, m, 0] for ell, m in MODES],
        "solver_frequency_error_ceiling": MAX_SOLVER_FREQUENCY_ERROR,
        "minimum_training_grid_distance": MIN_GRID_DISTANCE,
        "python": platform.python_version(),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {len(report['rows'])} independent roots to {args.output}")


if __name__ == "__main__":
    main()

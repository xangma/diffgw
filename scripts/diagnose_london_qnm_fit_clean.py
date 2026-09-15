#!/usr/bin/env python3
"""Reconstruct the London--Fauchon-Jones QNM fits from fresh Kerr solves.

This diagnostic uses only the independent continued-fraction eigenproblem in
``scripts/kerr_leaver_solver.py`` as numerical data.  It does not read a QNM
table, a waveform implementation, an author implementation, or an external
binary oracle.  The printed polar coefficients in
``kerr_qnm_london_published.py`` are used only as a public-paper rounding
reference after the independent regressions have been performed.

The historical training count and details of the authors' GMVP symbol
selection are not specified by the paper.  Consequently this script declares
the nested counts 65, 129, and 257 in advance, generates the finest grid once,
and reports every count.  It never chooses a count from a fit residual.

The paper's frequency convention is ``omega = omega_r + i/tau``.  The
independent solver uses ``omega = omega_r - i*omega_d`` for exp(-i omega t),
so roots are conjugated before regression.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOLVER_PATH = ROOT / "scripts" / "kerr_leaver_solver.py"
PRINTED_PATH = ROOT / "src" / "torchwave" / "approximants" / "kerr_qnm_london_published.py"

# These are an explicit, predeclared convergence plan.  The finest grid is
# generated once and the first two grids select nested theta nodes from it.
GRID_COUNTS = (65, 129, 257)
FINEST_COUNT = GRID_COUNTS[-1]
SPIN_LIMIT = 0.995
MODES = ((2, 2, 0), (2, 1, 0), (3, 3, 0), (3, 2, 0), (4, 4, 0), (4, 3, 0))
ZERO_DAMPED = frozenset(((2, 2, 0), (3, 3, 0), (4, 4, 0), (4, 3, 0)))
NONZERO_DAMPED = frozenset(((2, 1, 0), (3, 2, 0)))

# Fixed, non-grid verification nodes.  The denominator 61 is coprime to the
# 64, 128, and 256 subinterval denominators of the declared grids, and the
# fractional offset avoids both endpoints.  These roots are solved afresh,
# without using a training-grid seed.
CHECK_NODE_COUNT = 33
CHECK_NODE_OFFSET = 0.37


def _load_solver():
    """Import the permitted clean-room solver without making scripts a package."""
    if str(SOLVER_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SOLVER_PATH.parent))
    from kerr_leaver_solver import solve  # type: ignore

    return solve


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert roots, NumPy values, and complex numbers to strict JSON."""
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


def _polar(z: complex) -> list[float]:
    """Return radius and a phase in [0, 2*pi), matching the paper's display."""
    radius = abs(z)
    phase = math.atan2(z.imag, z.real) % (2 * math.pi)
    return [float(radius), float(phase)]


def _wrap_phase(delta: float) -> float:
    return float((delta + math.pi) % (2 * math.pi) - math.pi)


def kappa_value(spin: float, ell: int, m: int) -> float:
    """Equation (21), evaluated independently for the regression."""
    if not -1.0 <= float(spin) <= 1.0:
        raise ValueError("spin must lie in [-1, 1]")
    return math.log(2.0 - float(spin), 3.0) ** (1.0 / (2 + ell - abs(m)))


def theta_grid(count: int = FINEST_COUNT) -> np.ndarray:
    """Return the paper-described theta grid and its oriented spin values."""
    if count not in GRID_COUNTS:
        raise ValueError(f"count must be one of predeclared {GRID_COUNTS}")
    return np.linspace(-math.pi / 2, math.pi / 2, count, dtype=float)


def spin_grid(count: int = FINEST_COUNT) -> np.ndarray:
    return SPIN_LIMIT * np.sin(theta_grid(count))


def nested_indices(count: int, finest: int = FINEST_COUNT) -> np.ndarray:
    """Indices selecting a declared count from the finest nested grid."""
    if count not in GRID_COUNTS or finest != FINEST_COUNT:
        raise ValueError(f"count must be one of {GRID_COUNTS} and finest={FINEST_COUNT}")
    stride = (finest - 1) // (count - 1)
    indices = np.arange(0, finest, stride, dtype=int)
    if len(indices) != count or indices[-1] != finest - 1:
        raise AssertionError("declared grids are not nested")
    return indices


def check_theta_grid() -> np.ndarray:
    """Return fixed nodes absent from every declared training grid."""
    return -math.pi / 2 + (np.arange(CHECK_NODE_COUNT, dtype=float) + CHECK_NODE_OFFSET) * (
        math.pi / CHECK_NODE_COUNT
    )


def _root_is_usable(root: Any) -> bool:
    """Require numerical convergence and the physical solver branch."""
    if not bool(root.converged):
        return False
    omega = complex(root.omega)
    if not (np.isfinite(omega.real) and np.isfinite(omega.imag)):
        return False
    if omega.real <= 0.0 or omega.imag >= 0.0:
        return False
    estimate = float(root.metadata.get("estimated_frequency_error", math.inf))
    return np.isfinite(estimate) and estimate < 1e-9


def _solve_certified(
    solve: Any,
    spin: float,
    mode: tuple[int, int, int],
    omega_seed: complex | None = None,
    angular_seed: complex | None = None,
) -> Any:
    """Call the independent solver in strict mode and reject weak roots."""
    ell, m, n = mode
    root = solve(
        float(spin),
        ell,
        m,
        n,
        omega_seed=omega_seed,
        angular_seed=angular_seed,
        strict=True,
    )
    if not _root_is_usable(root):
        raise RuntimeError(f"unusable independent root at spin={spin}, mode={mode}: {root}")
    return root


def _root_row(theta: float, spin: float, root: Any, refinement_steps: int = 0) -> dict[str, Any]:
    """Persist all numerical diagnostics needed to audit a generated root."""
    row = {
        "theta": float(theta),
        "spin": float(spin),
        "omega_solver": complex(root.omega),
        "omega_paper": complex(root.omega).conjugate(),
        "separation_constant": complex(root.separation_constant),
        "radial_residual": complex(root.radial_residual),
        "cf_depth": int(root.cf_depth),
        "cf_error": float(root.cf_error),
        "angular_error": float(root.angular_error),
        "converged": bool(root.converged),
        "metadata": dict(root.metadata),
        "continuation_refinement_steps": int(refinement_steps),
    }
    return row


def _continued_point(
    solve: Any,
    target_spin: float,
    mode: tuple[int, int, int],
    previous_spin: float,
    previous_root: Any,
    max_refinement: int = 4,
) -> tuple[Any, int]:
    """Continue to a point, inserting fixed midpoint steps after a failure.

    The midpoint recovery is a branch-tracking safeguard.  It never changes
    the declared training nodes or their regression weights.
    """

    def attempt(spin: float, seed_root: Any) -> Any:
        candidate = _solve_certified(
            solve,
            spin,
            mode,
            omega_seed=complex(seed_root.omega),
            angular_seed=complex(seed_root.separation_constant),
        )
        jump = abs(complex(candidate.omega) - complex(seed_root.omega))
        # A step on the fixed 257-node theta grid is small.  A large jump is a
        # branch-tracking failure even when the continued fraction itself has
        # converged to a different root.
        if jump > 0.15:
            raise RuntimeError(f"continuation branch jump {jump:g} at spin={spin}")
        return candidate

    # First try the actual node.  This is the normal path for the finest grid.
    try:
        return attempt(target_spin, previous_root), 0
    except Exception as initial_error:
        last_error: Exception = initial_error

    for level in range(1, max_refinement + 1):
        pieces = 2**level
        seed = previous_root
        try:
            for fraction in range(1, pieces + 1):
                spin = previous_spin + (target_spin - previous_spin) * fraction / pieces
                seed = attempt(float(spin), seed)
            return seed, pieces - 1
        except Exception as error:
            last_error = error
    raise RuntimeError(
        f"continuation failed after fixed midpoint refinement, mode={mode}, "
        f"from spin={previous_spin} to {target_spin}: {last_error}"
    ) from last_error


def generate_training_grid(mode: tuple[int, int, int], solve: Any | None = None) -> list[dict[str, Any]]:
    """Generate all 257 roots from a=0 on each oriented-spin side."""
    solve = _load_solver() if solve is None else solve
    theta = theta_grid(FINEST_COUNT)
    spins = SPIN_LIMIT * np.sin(theta)
    center = FINEST_COUNT // 2
    rows: list[dict[str, Any] | None] = [None] * FINEST_COUNT

    center_root = _solve_certified(solve, 0.0, mode)
    rows[center] = _root_row(theta[center], spins[center], center_root)

    for side, indices in (
        ("negative", range(center - 1, -1, -1)),
        ("positive", range(center + 1, FINEST_COUNT)),
    ):
        previous_spin = 0.0
        previous_root = center_root
        for index in indices:
            root, refinements = _continued_point(
                solve, float(spins[index]), mode, previous_spin, previous_root
            )
            rows[index] = _root_row(theta[index], spins[index], root, refinements)
            previous_spin = float(spins[index])
            previous_root = root

    if any(row is None for row in rows):
        raise AssertionError(f"incomplete {side} continuation for {mode}")
    return [row for row in rows if row is not None]


def design_matrix(kappas: np.ndarray, mode: tuple[int, int, int], strategy: str = "main") -> np.ndarray:
    """Build the declared one-dimensional polynomial basis."""
    if strategy == "main":
        powers = range(5) if mode in ZERO_DAMPED else range(7)
    elif strategy == "direct_weighted":
        if mode not in ZERO_DAMPED:
            raise ValueError("direct constrained strategy applies only to zero-damped modes")
        powers = range(1, 6)
    else:
        raise ValueError(f"unknown fit strategy: {strategy}")
    return np.column_stack([kappas**power for power in powers])


def _mode_kappas(spins: np.ndarray, mode: tuple[int, int, int]) -> np.ndarray:
    ell, m, _ = mode
    return np.asarray([kappa_value(float(spin), ell, m) for spin in spins], dtype=float)


def fit_coefficients(
    spins: Sequence[float],
    frequencies: Sequence[complex],
    mode: tuple[int, int, int],
    strategy: str = "main",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit complex coefficients with generic NumPy least squares.

    ``main`` follows the paper's constrained ansatz by fitting
    ``(omega-m/2)/kappa`` for zero-damped modes, with powers 0..4.  For
    nonzero-damped modes it fits omega with powers 0..6.  The separate
    ``direct_weighted`` sensitivity check fits the constrained frequency
    directly with powers 1..5 and residual multiplier kappa**(-1/2), i.e. an
    inverse-kappa sample weight.  That choice is reported and is never used to
    select a reconstruction.
    """
    spins_array = np.asarray(spins, dtype=float)
    values = np.asarray(frequencies, dtype=complex)
    if spins_array.ndim != 1 or values.ndim != 1 or len(spins_array) != len(values):
        raise ValueError("spins and frequencies must be one-dimensional and equal length")
    kappas = _mode_kappas(spins_array, mode)
    ell, m, _ = mode
    baseline = m / 2.0

    if strategy == "main":
        if mode in ZERO_DAMPED:
            target = (values - baseline) / kappas
        elif mode in NONZERO_DAMPED:
            target = values
        else:
            raise ValueError(f"unsupported mode {mode}")
        matrix = design_matrix(kappas, mode, strategy)
        coeffs, residuals, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
        weights = np.ones_like(kappas)
        direct_weight_definition = "uniform transformed-target residual"
    elif strategy == "direct_weighted":
        if mode not in ZERO_DAMPED:
            raise ValueError("direct constrained strategy applies only to zero-damped modes")
        # Weighted direct-frequency LS: minimize sum |omega-model|^2/kappa.
        # The pre-multiplier is therefore kappa**(-1/2).
        weights = kappas ** -0.5
        matrix = design_matrix(kappas, mode, strategy) * weights[:, None]
        target = (values - baseline) * weights
        coeffs, residuals, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
        direct_weight_definition = "residual multiplier kappa**(-1/2), sample weight 1/kappa"
    else:
        raise ValueError(f"unknown fit strategy: {strategy}")

    prediction = evaluate_coefficients(spins_array, mode, coeffs, strategy)
    residual = prediction - values
    relative = abs(residual) / np.maximum(abs(values), np.finfo(float).tiny)
    return coeffs, {
        "strategy": strategy,
        "powers": list(range(5) if mode in ZERO_DAMPED else range(7))
        if strategy == "main"
        else list(range(1, 6)),
        "target": (
            "(omega-m/2)/kappa"
            if strategy == "main" and mode in ZERO_DAMPED
            else "omega-m/2 (direct constrained)"
            if strategy == "direct_weighted"
            else "omega"
        ),
        "weight_definition": direct_weight_definition,
        "rank": int(rank),
        "singular_values": singular,
        "coefficient_polar": [_polar(complex(value)) for value in coeffs],
        "coefficient_rectangular": [complex(value) for value in coeffs],
        "training_residual": {
            "max_abs_frequency": float(np.max(abs(residual))),
            "rms_abs_frequency": float(np.sqrt(np.mean(abs(residual) ** 2))),
            "max_relative_frequency": float(np.max(relative)),
            "residual_norm": float(np.linalg.norm(residual)),
        },
        "transformed_residual_norm": float(np.linalg.norm((residual / kappas) if mode in ZERO_DAMPED else residual)),
        "weighted_direct_residual_norm": float(np.linalg.norm(weights * residual)),
    }


def evaluate_coefficients(
    spins: Sequence[float], mode: tuple[int, int, int], coefficients: Sequence[complex], strategy: str = "main"
) -> np.ndarray:
    spins_array = np.asarray(spins, dtype=float)
    kappas = _mode_kappas(spins_array, mode)
    coeffs = np.asarray(coefficients, dtype=complex)
    baseline = mode[1] / 2.0
    if strategy == "main":
        result = design_matrix(kappas, mode, strategy) @ coeffs
        return result * kappas + baseline if mode in ZERO_DAMPED else result
    if strategy == "direct_weighted":
        if mode not in ZERO_DAMPED:
            raise ValueError("direct constrained strategy applies only to zero-damped modes")
        return baseline + design_matrix(kappas, mode, strategy) @ coeffs
    raise ValueError(f"unknown fit strategy: {strategy}")


def _printed_polar(mode: tuple[int, int, int]) -> tuple[float, list[list[float]]]:
    """Read the public transcription's printed polar coefficients."""
    # This import is intentionally limited to the safe printed-fit module;
    # its values are not numerical training data.
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from torchwave.approximants.kerr_qnm_london_published import _COEFFS, _GENERAL  # type: ignore

    lm = mode[:2]
    if lm in _COEFFS:
        base, coefficients = _COEFFS[lm]
    elif lm in _GENERAL:
        base, coefficients = 0.0, _GENERAL[lm]
    else:
        raise ValueError(f"printed coefficients unavailable for mode {mode}")
    return float(base), [[float(radius), float(angle)] for radius, angle in coefficients]


def compare_printed(
    coefficients: Sequence[complex], mode: tuple[int, int, int], rounding_half_width: float = 5e-5
) -> dict[str, Any]:
    """Compare independently fitted polar coefficients to four-decimal print intervals."""
    base, printed = _printed_polar(mode)
    generated = [_polar(complex(value)) for value in coefficients]
    rows = []
    for index, (got, reference) in enumerate(zip(generated, printed)):
        radius_delta = got[0] - reference[0]
        phase_delta = _wrap_phase(got[1] - reference[1])
        rows.append(
            {
                "index": index,
                "generated": got,
                "printed": reference,
                "radius_delta": float(radius_delta),
                "phase_delta_wrapped": float(phase_delta),
                "within_radius_rounding_interval": bool(abs(radius_delta) <= rounding_half_width),
                "within_phase_rounding_interval": bool(abs(phase_delta) <= rounding_half_width),
            }
        )
    return {
        "baseline": base,
        "rounding_interval_half_width": rounding_half_width,
        "coefficients": rows,
        "all_within_printed_rounding_intervals": all(
            row["within_radius_rounding_interval"] and row["within_phase_rounding_interval"] for row in rows
        ),
        "max_abs_radius_delta": float(max(abs(row["radius_delta"]) for row in rows)),
        "max_abs_phase_delta_wrapped": float(max(abs(row["phase_delta_wrapped"]) for row in rows)),
    }


def _root_values(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    spins = np.asarray([float(row["spin"]) for row in rows], dtype=float)
    values = np.asarray([complex(row["omega_paper"]) for row in rows], dtype=complex)
    return spins, values


def _fit_grid_rows(rows: Sequence[Mapping[str, Any]], mode: tuple[int, int, int], count: int) -> dict[str, Any]:
    indices = nested_indices(count)
    spins, values = _root_values([rows[index] for index in indices])
    coefficients, quality = fit_coefficients(spins, values, mode, "main")
    result = {
        "count": count,
        "indices_from_257": indices,
        "fit": quality,
        "coefficients_rectangular": coefficients,
        "coefficients_polar": [_polar(complex(value)) for value in coefficients],
        "printed_comparison": compare_printed(coefficients, mode),
    }
    if mode in ZERO_DAMPED:
        direct_coefficients, direct_quality = fit_coefficients(spins, values, mode, "direct_weighted")
        result["direct_constrained_weighted_alternative"] = {
            **direct_quality,
            "coefficients_rectangular": direct_coefficients,
            "coefficients_polar": [_polar(complex(value)) for value in direct_coefficients],
            "printed_comparison": compare_printed(direct_coefficients, mode),
        }
    return result


def _check_fit(
    check_rows: Sequence[Mapping[str, Any]], mode: tuple[int, int, int], coefficients: Sequence[complex], strategy: str
) -> dict[str, Any]:
    spins, values = _root_values(check_rows)
    predicted = evaluate_coefficients(spins, mode, coefficients, strategy)
    residual = predicted - values
    relative = abs(residual) / np.maximum(abs(values), np.finfo(float).tiny)
    return {
        "node_count": len(check_rows),
        "theta_nodes": [float(row["theta"]) for row in check_rows],
        "spin_nodes": spins,
        "max_abs_frequency": float(np.max(abs(residual))),
        "rms_abs_frequency": float(np.sqrt(np.mean(abs(residual) ** 2))),
        "max_relative_frequency": float(np.max(relative)),
        "worst_node_index": int(np.argmax(abs(residual))),
        "worst_spin": float(spins[np.argmax(abs(residual))]),
    }


def _convergence(previous: Mapping[str, Any] | None, current: Mapping[str, Any]) -> dict[str, Any] | None:
    if previous is None:
        return None
    old = np.asarray([complex(value) for value in previous["coefficients_rectangular"]])
    new = np.asarray([complex(value) for value in current["coefficients_rectangular"]])
    delta = abs(new - old)
    return {
        "from_count": int(previous["count"]),
        "to_count": int(current["count"]),
        "max_abs_coefficient_delta": float(np.max(delta)),
        "rms_abs_coefficient_delta": float(np.sqrt(np.mean(delta**2))),
        "coefficient_abs_deltas": delta,
    }


def _check_rows_for_mode(mode: tuple[int, int, int], solve: Any) -> list[dict[str, Any]]:
    rows = []
    for theta in check_theta_grid():
        spin = SPIN_LIMIT * math.sin(float(theta))
        # A no-seed call starts its own clean Schwarzschild/spin continuation.
        root = _solve_certified(solve, spin, mode)
        rows.append(_root_row(theta, spin, root))
    return rows


def run_benchmark(solve: Any | None = None) -> dict[str, Any]:
    """Small fixed preflight benchmark used before the full six-mode run."""
    solve = _load_solver() if solve is None else solve
    spins = (0.0, 0.5, 0.9, SPIN_LIMIT, -0.5, -SPIN_LIMIT)
    rows = []
    started = time.monotonic()
    for spin in spins:
        tick = time.monotonic()
        root = _solve_certified(solve, spin, MODES[0])
        rows.append(
            {
                "spin": spin,
                "omega_solver": complex(root.omega),
                "elapsed_seconds": time.monotonic() - tick,
                "cf_depth": root.cf_depth,
                "estimated_frequency_error": root.metadata.get("estimated_frequency_error"),
                "converged": root.converged,
            }
        )
    return {
        "mode": MODES[0],
        "spins": spins,
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
        "purpose": "preflight before the fixed N=65,129,257 reconstruction",
    }


def _versions() -> dict[str, Any]:
    result: dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__}
    for package in ("scipy", "gmpy2"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): _sha256(path) for path in (Path(__file__).resolve(), SOLVER_PATH, PRINTED_PATH)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "reviews" / "2026-09-13-parity-review" / "verification" / "london-qnm-fit",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("/tmp/torchwave-london-qnm-fit"))
    parser.add_argument("--benchmark", action="store_true", help="run only the fixed small preflight benchmark")
    parser.add_argument("--modes", nargs="+", help="optional subset, labels like 2,2,0; full run is predeclared six modes")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    solve = _load_solver()
    if args.benchmark:
        result = run_benchmark(solve)
        _write_json(args.output_dir / "benchmark.json", result)
        print(json.dumps(_jsonable(result), indent=2))
        return 0

    modes = MODES
    if args.modes:
        modes = tuple(tuple(int(value) for value in label.split(",")) for label in args.modes)
        if any(mode not in MODES for mode in modes) or len(set(modes)) != len(modes):
            parser.error("modes must be a subset of the predeclared six fundamentals")

    before_hashes = _source_hashes()
    started = time.monotonic()
    report: dict[str, Any] = {
        "provenance": {
            "paper": "L. London and E. Fauchon-Jones, arXiv:1810.03550v2",
            "paper_url": "https://arxiv.org/html/1810.03550",
            "equations": "Eqs. (4)-(6), (21)-(32); Fig. 1 caption",
            "solver_source": "independent Leaver continued fraction and spheroidal angular eigenproblem",
            "solver_convention": "exp(-i omega t), omega=omega_r-i*omega_d",
            "paper_convention": "tilde{omega}=omega_r+i/tau; independent roots conjugated before fit",
            "printed_reference": "src/torchwave/approximants/kerr_qnm_london_published.py, public four-decimal transcription",
            "clean_room_scope": "No LAL, PyCBC, positive, author software, calibration files, waveform outputs, or public target binary used",
            "historical_limit": "The paper does not specify the historical training count or all GMVP selections; this is a bounded independent reconstruction.",
        },
        "choices": {
            "modes": modes,
            "overtone": 0,
            "spin_limit": SPIN_LIMIT,
            "spin_grid": "a=0.995*sin(theta), theta uniform on [-pi/2, pi/2]",
            "predeclared_grid_counts": GRID_COUNTS,
            "grid_generation": "one N=257 grid, nested subgrids N=65 and N=129 by theta-node stride",
            "zero_damped_modes": sorted(ZERO_DAMPED),
            "zero_damped_powers": [0, 1, 2, 3, 4],
            "zero_damped_baseline": "m/2",
            "nonzero_damped_modes": sorted(NONZERO_DAMPED),
            "nonzero_damped_powers": list(range(7)),
            "least_squares": "numpy.linalg.lstsq, complex values, rcond=None",
            "main_zero_damped_target": "(omega-m/2)/kappa",
            "direct_alternative": "omega=m/2+sum(c_k*kappa**(k+1)); residual multiplier kappa**(-1/2), sample weight 1/kappa",
            "check_nodes": f"{CHECK_NODE_COUNT} fixed nodes theta=-pi/2+(i+{CHECK_NODE_OFFSET})*pi/{CHECK_NODE_COUNT}; no training seeds",
            "solver_certification": "strict=True; reject converged=False, nonphysical roots, or estimated frequency error >=1e-9",
            "branch_recovery": "fixed midpoint continuation only after a solver failure or jump; no fit-count adaptation",
        },
        "source_hashes_before": before_hashes,
        "versions": _versions(),
        "modes": {},
    }

    # Keep the progress file machine-readable after every completed mode.  Raw
    # roots live under /tmp and are hashed into the durable report.
    for mode_index, mode in enumerate(modes):
        label = ",".join(str(value) for value in mode)
        print(f"generating mode={label} ({mode_index + 1}/{len(modes)})", flush=True)
        grid_started = time.monotonic()
        rows = generate_training_grid(mode, solve)
        raw_path = args.raw_dir / f"mode-{label}-grid257.json"
        _write_json(raw_path, rows)
        check_rows = _check_rows_for_mode(mode, solve)
        check_path = args.raw_dir / f"mode-{label}-check.json"
        _write_json(check_path, check_rows)

        grid_results = []
        previous: dict[str, Any] | None = None
        for count in GRID_COUNTS:
            current = _fit_grid_rows(rows, mode, count)
            current["convergence_from_previous"] = _convergence(previous, current)
            current["independent_check_main"] = _check_fit(
                check_rows, mode, current["coefficients_rectangular"], "main"
            )
            if "direct_constrained_weighted_alternative" in current:
                direct = current["direct_constrained_weighted_alternative"]
                direct["independent_check"] = _check_fit(
                    check_rows, mode, direct["coefficients_rectangular"], "direct_weighted"
                )
            grid_results.append(current)
            previous = current

        mode_result = {
            "mode": mode,
            "grid_results": grid_results,
            "root_summary": {
                "training_root_count": len(rows),
                "check_root_count": len(check_rows),
                "all_training_roots_converged": all(bool(row["converged"]) for row in rows),
                "all_check_roots_converged": all(bool(row["converged"]) for row in check_rows),
                "max_cf_depth_training": max(int(row["cf_depth"]) for row in rows),
                "max_estimated_frequency_error_training": max(
                    float(row["metadata"]["estimated_frequency_error"]) for row in rows
                ),
                "max_continuation_refinement_steps": max(
                    int(row["continuation_refinement_steps"]) for row in rows
                ),
            },
            "raw_artifacts": {
                "training_path": str(raw_path),
                "training_sha256": _sha256(raw_path),
                "check_path": str(check_path),
                "check_sha256": _sha256(check_path),
            },
            "elapsed_seconds": time.monotonic() - grid_started,
        }
        report["modes"][label] = mode_result
        _write_json(args.output_dir / "progress.json", report)
        print(f"completed mode={label} elapsed={mode_result['elapsed_seconds']:.2f}s", flush=True)

    after_hashes = _source_hashes()
    if after_hashes != before_hashes:
        raise RuntimeError("clean-room inputs changed during reconstruction")
    report["source_hashes_after"] = after_hashes
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "completed"
    _write_json(args.output_dir / "report.json", report)
    _write_json(args.output_dir / "progress.json", report)
    print(f"wrote {args.output_dir / 'report.json'} in {report['elapsed_seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

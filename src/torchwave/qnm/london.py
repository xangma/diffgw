"""Clean-room London N=61 Kerr QNM fitting, evaluation, and grid generation.

Reconstructs the London & Fauchon-Jones (2019, arXiv:1810.03550v2) Kerr quasi-normal
mode (QNM) polynomial fits on the declared exact N=61 grid using the clean-room
Leaver continued-fraction solver in torchwave.qnm.solver.

The convention is exp(-i omega t), omega = omega_r - i*omega_d in the solver,
conjugated to tilde{omega} = omega_r + i/tau before fitting, following Eqs. (4)-(6)
and (21)-(32) of the paper.
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

from torchwave.approximants.kerr_qnm_london_published import _COEFFS, _GENERAL
from torchwave.qnm.solver import solve

EXACT_GRID_COUNT: int = 61
SPIN_LIMIT: float = 0.995
MODES: tuple[tuple[int, int, int], ...] = (
    (2, 2, 0),
    (2, 1, 0),
    (3, 3, 0),
    (3, 2, 0),
    (4, 4, 0),
    (4, 3, 0),
)
ZERO_DAMPED: frozenset[tuple[int, int, int]] = frozenset(
    ((2, 2, 0), (3, 3, 0), (4, 4, 0), (4, 3, 0))
)
NONZERO_DAMPED: frozenset[tuple[int, int, int]] = frozenset(
    ((2, 1, 0), (3, 2, 0))
)
CHECK_NODE_COUNT: int = 33
CHECK_NODE_OFFSET: float = 0.37


def _load_solver():
    """Return the Leaver continued-fraction solver function."""
    return solve


def _load_helper():
    """Return this module for clean-room helper compatibility."""
    return sys.modules[__name__]


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


def _polar(z: complex) -> list[float]:
    """Return radius and a phase in [0, 2*pi), matching paper display."""
    radius = abs(z)
    phase = math.atan2(z.imag, z.real) % (2 * math.pi)
    return [float(radius), float(phase)]


def _wrap_phase(delta: float) -> float:
    return float((delta + math.pi) % (2 * math.pi) - math.pi)


def kappa_value(spin: float, ell: int, m: int) -> float:
    """Equation (21) of London & Fauchon-Jones (2019)."""
    if not -1.0 <= float(spin) <= 1.0:
        raise ValueError("spin must lie in [-1, 1]")
    return math.log(2.0 - float(spin), 3.0) ** (1.0 / (2 + ell - abs(m)))


def _mode_kappas(spins: np.ndarray, mode: tuple[int, int, int]) -> np.ndarray:
    ell, m, _ = mode
    return np.asarray([kappa_value(float(spin), ell, m) for spin in spins], dtype=float)


def exact_theta_grid(count: int = EXACT_GRID_COUNT) -> np.ndarray:
    """Return the declared exact N=61 theta grid."""
    if count != EXACT_GRID_COUNT:
        raise ValueError(f"count must be the declared exact count {EXACT_GRID_COUNT}")
    return np.linspace(-math.pi / 2, math.pi / 2, count, dtype=float)


def exact_spin_grid(count: int = EXACT_GRID_COUNT) -> np.ndarray:
    """Return ``a=0.995*sin(theta)`` at exact N=61 nodes."""
    return SPIN_LIMIT * np.sin(exact_theta_grid(count))


def check_theta_grid() -> np.ndarray:
    """Return the fixed off-grid nodes used by the clean diagnostic."""
    return -math.pi / 2 + (np.arange(CHECK_NODE_COUNT, dtype=float) + CHECK_NODE_OFFSET) * (
        math.pi / CHECK_NODE_COUNT
    )


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


def evaluate_coefficients(
    spins: Sequence[float],
    mode: tuple[int, int, int],
    coefficients: Sequence[complex],
    strategy: str = "main",
) -> np.ndarray:
    """Evaluate the fitted polynomial model at given spin values."""
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


def fit_coefficients(
    spins: Sequence[float],
    frequencies: Sequence[complex],
    mode: tuple[int, int, int],
    strategy: str = "main",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit complex coefficients with generic NumPy least squares.

    ``main`` follows the paper's constrained ansatz by fitting
    ``(omega-m/2)/kappa`` for zero-damped modes, with powers 0..4. For
    nonzero-damped modes it fits omega with powers 0..6. The separate
    ``direct_weighted`` sensitivity check fits the constrained frequency
    directly with powers 1..5 and residual multiplier kappa**(-1/2).
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
        "transformed_residual_norm": float(
            np.linalg.norm((residual / kappas) if mode in ZERO_DAMPED else residual)
        ),
        "weighted_direct_residual_norm": float(np.linalg.norm(weights * residual)),
    }


def _printed_polar(mode: tuple[int, int, int]) -> tuple[float, list[list[float]]]:
    """Read the public transcription's printed polar coefficients."""
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
            row["within_radius_rounding_interval"] and row["within_phase_rounding_interval"]
            for row in rows
        ),
        "max_abs_radius_delta": float(max(abs(row["radius_delta"]) for row in rows)),
        "max_abs_phase_delta_wrapped": float(max(abs(row["phase_delta_wrapped"]) for row in rows)),
    }


def _root_values(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    spins = np.asarray([float(row["spin"]) for row in rows], dtype=float)
    values = np.asarray([complex(row["omega_paper"]) for row in rows], dtype=complex)
    return spins, values


def _root_is_usable(root: Any) -> bool:
    """Require numerical convergence and physical solver branch."""
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
    solve_fn: Any,
    spin: float,
    mode: tuple[int, int, int],
    omega_seed: complex | None = None,
    angular_seed: complex | None = None,
) -> Any:
    """Call solver in strict mode and reject weak roots."""
    ell, m, n = mode
    root = solve_fn(
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
    return {
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


def _continued_point(
    solve_fn: Any,
    target_spin: float,
    mode: tuple[int, int, int],
    previous_spin: float,
    previous_root: Any,
    max_refinement: int = 4,
) -> tuple[Any, int]:
    """Continue to a point, inserting fixed midpoint steps after a failure."""

    def attempt(spin: float, seed_root: Any) -> Any:
        candidate = _solve_certified(
            solve_fn,
            spin,
            mode,
            omega_seed=complex(seed_root.omega),
            angular_seed=complex(seed_root.separation_constant),
        )
        jump = abs(complex(candidate.omega) - complex(seed_root.omega))
        if jump > 0.15:
            raise RuntimeError(f"continuation branch jump {jump:g} at spin={spin}")
        return candidate

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


def generate_exact_training_grid(
    mode: tuple[int, int, int],
    solve: Any | None = None,
) -> list[dict[str, Any]]:
    """Generate all exact N=61 roots by fresh center and spin continuation."""
    solver = _load_solver() if solve is None else solve
    theta = exact_theta_grid()
    spins = exact_spin_grid()
    center = EXACT_GRID_COUNT // 2
    rows: list[dict[str, Any] | None] = [None] * EXACT_GRID_COUNT

    center_root = _solve_certified(solver, 0.0, mode)
    rows[center] = _root_row(theta[center], spins[center], center_root)

    for indices in (range(center - 1, -1, -1), range(center + 1, EXACT_GRID_COUNT)):
        previous_spin = 0.0
        previous_root = center_root
        for index in indices:
            root, refinements = _continued_point(
                solver, float(spins[index]), mode, previous_spin, previous_root
            )
            rows[index] = _root_row(theta[index], spins[index], root, refinements)
            previous_spin = float(spins[index])
            previous_root = root

    if any(row is None for row in rows):
        raise AssertionError(f"incomplete exact N=61 continuation for {mode}")
    return [row for row in rows if row is not None]


def _check_fit(
    check_rows: Sequence[Mapping[str, Any]],
    mode: tuple[int, int, int],
    coefficients: Sequence[complex],
    strategy: str,
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


def _check_rows_for_mode(mode: tuple[int, int, int], solve: Any | None = None) -> list[dict[str, Any]]:
    solver = _load_solver() if solve is None else solve
    rows = []
    for theta in check_theta_grid():
        spin = SPIN_LIMIT * math.sin(float(theta))
        root = _solve_certified(solver, spin, mode)
        rows.append(_root_row(theta, spin, root))
    return rows


def fit_exact_rows(
    rows: Sequence[Mapping[str, Any]],
    check_rows: Sequence[Mapping[str, Any]],
    mode: tuple[int, int, int],
) -> dict[str, Any]:
    """Fit one exact grid and attach training/check/rounding diagnostics."""
    spins, values = _root_values(rows)
    coefficients, quality = fit_coefficients(spins, values, mode, "main")
    result: dict[str, Any] = {
        "count": EXACT_GRID_COUNT,
        "grid_definition": "exact fresh N=61 theta grid; no interpolation",
        "theta_nodes": [float(row["theta"]) for row in rows],
        "spin_nodes": spins,
        "fit": quality,
        "coefficients_rectangular": coefficients,
        "coefficients_polar": [_polar(complex(value)) for value in coefficients],
        "printed_comparison": compare_printed(coefficients, mode),
        "independent_check_main": _check_fit(check_rows, mode, coefficients, "main"),
    }
    if mode in ZERO_DAMPED:
        direct_coefficients, direct_quality = fit_coefficients(spins, values, mode, "direct_weighted")
        direct = {
            **direct_quality,
            "coefficients_rectangular": direct_coefficients,
            "coefficients_polar": [_polar(complex(value)) for value in direct_coefficients],
            "printed_comparison": compare_printed(direct_coefficients, mode),
            "independent_check": _check_fit(check_rows, mode, direct_coefficients, "direct_weighted"),
        }
        result["direct_constrained_weighted_alternative"] = direct
    return result


_fit_exact_rows = fit_exact_rows


def _mode_result(
    mode: tuple[int, int, int],
    rows: Sequence[Mapping[str, Any]],
    check_rows: Sequence[Mapping[str, Any]],
    raw_dir: Path,
) -> dict[str, Any]:
    label = ",".join(str(value) for value in mode)
    training_path = raw_dir / f"mode-{label}-grid61.json"
    check_path = raw_dir / f"mode-{label}-check.json"
    _write_json(training_path, rows)
    _write_json(check_path, check_rows)
    fit = fit_exact_rows(rows, check_rows, mode)
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


def _versions() -> dict[str, Any]:
    result: dict[str, Any] = {"python": platform.python_version(), "numpy": np.__version__}
    for package in ("scipy", "gmpy2"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _source_hashes() -> dict[str, str]:
    mod_path = Path(__file__).resolve()
    solver_path = mod_path.parent / "solver.py"
    printed_path = mod_path.parents[1] / "approximants" / "kerr_qnm_london_published.py"
    hashes: dict[str, str] = {}
    for key, path in (
        ("src/torchwave/qnm/london.py", mod_path),
        ("src/torchwave/qnm/solver.py", solver_path),
        ("src/torchwave/approximants/kerr_qnm_london_published.py", printed_path),
    ):
        if path.exists():
            hashes[key] = _sha256(path)
    return hashes


def _parse_modes(
    parser: argparse.ArgumentParser, labels: Sequence[str] | None
) -> tuple[tuple[int, int, int], ...]:
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
    parser = argparse.ArgumentParser(description="TorchWave London N=61 Kerr QNM reconstruction")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/reviews/2026-09-13-parity-review/verification/london-qnm-fit-exact61"),
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("/tmp/torchwave-london-qnm-fit-exact61"))
    parser.add_argument("--log-path", type=Path, help="path recorded in the report for the redirected job log")
    parser.add_argument("--modes", nargs="+", help="optional subset, labels like 2,2,0")
    args = parser.parse_args(argv)

    modes = _parse_modes(parser, args.modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    solver = _load_solver()
    source_hashes_before = _source_hashes()
    started = time.monotonic()
    command = [sys.executable, "-m", "torchwave.qnm.london", *(argv if argv is not None else sys.argv[1:])]
    report: dict[str, Any] = {
        "provenance": {
            "paper": "L. London and E. Fauchon-Jones, arXiv:1810.03550v2",
            "paper_url": "https://arxiv.org/html/1810.03550",
            "equations": "Eqs. (4)-(6), (21)-(32); Fig. 1 caption",
            "source": "published rendered paper plus the clean-room Leaver continued-fraction solver",
            "solver_source": "src/torchwave/qnm/solver.py: radial continued fraction and spheroidal angular eigenproblem",
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
        rows = generate_exact_training_grid(mode, solver)
        check_rows = _check_rows_for_mode(mode, solver)
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
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())

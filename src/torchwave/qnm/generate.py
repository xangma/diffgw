"""Generate original Kerr QNM data solely from the published eigenproblem.

Requires NumPy/SciPy and torchwave.qnm.solver. No network, external QNM
solver, frequency dataset or waveform oracle is read.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import make_interp_spline

from torchwave.qnm.solver import solve

MODES = tuple(
    (l, m, n)
    for n in range(3)
    for l, m in ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
) + ((2, -2, 0),)

DEGREE = 5
SPIN_MAX = 0.99999
BEGIN = "# BEGIN GENERATED INDEPENDENT DATA\n"
END = "# END GENERATED INDEPENDENT DATA"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize(value: Any) -> Any:
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(serialize(value), indent=2, allow_nan=False) + "\n")


def root_row(a: float, root: Any) -> Dict[str, Any]:
    if not root.converged or root.omega.real <= 0 or root.omega.imag >= 0:
        raise RuntimeError(f"Unconverged or nonphysical root at a={a}: {root}")
    return {"spin": float(a), **asdict(root)}


def continuation(mode: Tuple[int, int, int], spins: np.ndarray) -> List[Dict[str, Any]]:
    """Follow a labelled Schwarzschild root without external starting data."""
    l, m, n = mode
    rows, previous = [], []
    for i, a in enumerate(spins):
        if not previous:
            root = solve(float(a), l, m, n)
        else:
            seed, angular_seed = previous[-1][1].omega, previous[-1][1].separation_constant
            if len(previous) == 2:
                x0, p0 = previous[0]
                x1, p1 = previous[1]
                ratio = (-np.log1p(-a) - x1) / (x1 - x0)
                seed += ratio * (p1.omega - p0.omega)
                angular_seed += ratio * (p1.separation_constant - p0.separation_constant)
            root = solve(float(a), l, m, n, omega_seed=seed, angular_seed=angular_seed)
        rows.append(root_row(a, root))
        previous = (previous + [(-np.log1p(-a), root)])[-2:]
        if i and i % 100 == 0:
            print(f"mode={mode} continuation={i}/{len(spins)} a={a:.8f}", flush=True)
    return rows


def interpolator(rows: List[Dict[str, Any]]):
    x = -np.log1p(-np.array([r["spin"] for r in rows]))
    y = np.array([[r["omega"].real, r["omega"].imag] for r in rows])
    return make_interp_spline(x, y, k=DEGREE)


def check_points(mode: Tuple[int, int, int], spins: np.ndarray, spline: Any, strict: bool):
    rows = []
    l, m, n = mode
    guesses = spline(-np.log1p(-spins))
    for a, guess in zip(spins, guesses):
        root = solve(float(a), l, m, n, omega_seed=complex(*guess), strict=strict)
        rows.append(root_row(a, root))
    actual = np.array([[r["omega"].real, r["omega"].imag] for r in rows])
    delta = np.abs(actual - guesses)
    return rows, {
        "count": len(rows),
        "max_abs": delta.max(axis=0).tolist(),
        "max_relative": (delta / np.abs(actual)).max(axis=0).tolist(),
        "worst_spin": spins[delta.argmax(axis=0)].tolist(),
    }


def reverse_check(mode: Tuple[int, int, int], rows: List[Dict[str, Any]]):
    """Check branch closure without using forward interior roots as seeds."""
    l, m, n = mode
    previous, reverse_rows = [], []
    for row in reversed(rows):
        a = row["spin"]
        x = -np.log1p(-a)
        if not previous:
            seed, angular_seed = row["omega"], row["separation_constant"]
        else:
            seed, angular_seed = previous[-1][1].omega, previous[-1][1].separation_constant
            if len(previous) == 2:
                x0, p0 = previous[0]
                x1, p1 = previous[1]
                ratio = (x - x1) / (x1 - x0)
                seed += ratio * (p1.omega - p0.omega)
                angular_seed += ratio * (p1.separation_constant - p0.separation_constant)
        root = solve(a, l, m, n, omega_seed=seed, angular_seed=angular_seed)
        reverse_rows.append(root_row(a, root))
        previous = (previous + [(x, root)])[-2:]
    reverse_rows.reverse()
    error = np.array([abs(a["omega"] - b["omega"]) for a, b in zip(rows, reverse_rows)])
    return reverse_rows, {
        "count": len(rows),
        "max_complex_abs": float(error.max()),
        "worst_spin": rows[int(error.argmax())]["spin"],
        "schwarzschild_closure": float(error[0]),
        "max_coordinate_step": float(max(np.diff(-np.log1p(-np.array([r["spin"] for r in rows]))))),
    }


def branch_separations(splines: Dict, records: Dict):
    """Reject coalesced overtone labels relative to their resolution errors."""
    result = {}
    x = np.linspace(0.0, -np.log1p(-SPIN_MAX), 1601)
    for first in splines:
        for second in splines:
            if first[:2] != second[:2] or first[2] >= second[2]:
                continue
            a, b = splines[first](x), splines[second](x)
            separation = np.linalg.norm(a - b, axis=1)
            first_label, second_label = (",".join(map(str, m)) for m in (first, second))
            error = max(
                records[first_label]["max_estimated_frequency_error"],
                records[second_label]["max_estimated_frequency_error"],
                np.finfo(float).eps,
            )
            ratio = float(separation.min() / error)
            if ratio <= 100:
                raise RuntimeError(f"Unresolved overtone separation {first}, {second}: {ratio}")
            result[f"{first_label};{second_label}"] = {
                "minimum_complex_separation": float(separation.min()),
                "separation_to_error_ratio": ratio,
            }
    return result


def render_splines(splines: Dict) -> str:
    lines = ["# Original clean-room QNM data derived via Leaver solver.", "_SPLINES = {"]
    for mode, spline in splines.items():
        lines += [f"    {mode}: (", "        ("]
        for start in range(0, len(spline.t), 4):
            lines.append("            " + ", ".join(repr(float(v)) for v in spline.t[start : start + 4]) + ",")
        lines += ["        ),", "        ("]
        lines += [f"            ({float(row[0])!r}, {float(row[1])!r})," for row in spline.c]
        lines += ["        ),", "    ),"]
    return "\n".join(lines + ["}", ""])


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="TorchWave Kerr QNM Data Generator")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save output files")
    parser.add_argument("--modes", nargs="+", help="Mode labels l,m,n; default all 16")
    parser.add_argument("--points", type=int, default=401, help="Initial grid points per mode")
    parser.add_argument("--max-points", type=int, default=1601, help="Max grid points per mode")
    parser.add_argument("--holdout-count", type=int, default=100, help="Number of random holdouts per mode")
    parser.add_argument("--absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--relative-tolerance", type=float, default=1e-10)
    args = parser.parse_args(argv)

    if args.points < 11 or args.max_points < args.points or args.holdout_count < 1:
        parser.error("Require points>=11, max-points>=points and holdout-count>=1")
    modes = tuple(tuple(map(int, label.split(","))) for label in args.modes) if args.modes else MODES
    if any(mode not in MODES for mode in modes) or len(set(modes)) != len(modes):
        parser.error("Unknown or duplicate mode")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    generated_file = Path(__file__).resolve().parents[1] / "approximants/kerr_qnm_generated.py"

    start = time.monotonic()
    results, splines, fixtures = {}, {}, {}
    for index, mode in enumerate(modes):
        label = ",".join(map(str, mode))
        count = args.points
        while True:
            x = np.linspace(0.0, -np.log1p(-SPIN_MAX), count)
            spins = -np.expm1(-x)
            spins[[0, -1]] = [0.0, SPIN_MAX]
            rows = continuation(mode, spins)
            spline = interpolator(rows)
            dev_spins = -np.expm1(-(x[:-1] + x[1:]) / 2)
            dev_rows, development = check_points(mode, dev_spins, spline, strict=False)
            write_json(args.output_dir / f"{label}-training-{count}.json", rows)
            write_json(args.output_dir / f"{label}-development-{count}.json", dev_rows)
            print(f"mode={mode} points={count} development={development['max_abs']}", flush=True)
            if max(development["max_abs"]) <= args.absolute_tolerance / 2 and max(development["max_relative"]) <= args.relative_tolerance / 2:
                break
            count = 2 * count - 1
            if count > args.max_points:
                raise RuntimeError(f"Development error exceeds tolerance for {mode}: {development}")

        reverse_rows, reverse = reverse_check(mode, rows)
        write_json(args.output_dir / f"{label}-reverse.json", reverse_rows)
        if reverse["max_complex_abs"] > 5 * args.absolute_tolerance:
            raise RuntimeError(f"Reverse branch closure failed for {mode}: {reverse}")

        rng = np.random.default_rng(20260912 + MODES.index(mode))
        hold_x = rng.uniform(0.0, -np.log1p(-SPIN_MAX), args.holdout_count)
        hold_spins = np.unique(np.r_[-np.expm1(-hold_x), 0.00013, 0.3, 0.9, 0.99, 0.999, 0.9999])
        holdouts, holdout = check_points(mode, hold_spins, spline, strict=True)
        endpoints, endpoint_check = check_points(mode, np.array([0.0, SPIN_MAX]), spline, strict=True)
        if (
            max(holdout["max_abs"] + endpoint_check["max_abs"]) > args.absolute_tolerance
            or max(holdout["max_relative"] + endpoint_check["max_relative"]) > args.relative_tolerance
        ):
            raise RuntimeError(f"Final validation failed for {mode}: {holdout}, {endpoint_check}")
        splines[mode] = spline
        fixtures[label] = {"rows": holdouts, "endpoints": endpoints}
        results[label] = {
            "training_count": count,
            "development": development,
            "holdout": holdout,
            "endpoint_resolution": endpoint_check,
            "reverse_continuation": reverse,
            "max_estimated_frequency_error": max(r["metadata"]["estimated_frequency_error"] for r in rows + holdouts + endpoints),
            "max_radial_residual": max(abs(r["radial_residual"]) for r in rows + holdouts + endpoints),
            "max_cf_depth": max(r["cf_depth"] for r in rows + holdouts + endpoints),
        }
        write_json(args.output_dir / "progress.json", results)
        print(f"mode={mode} verified={holdout['max_abs']} elapsed={time.monotonic()-start:.1f}s", flush=True)

    separations = branch_separations(splines, results)
    if generated_file.exists():
        template = generated_file.read_text()
        if BEGIN in template and END in template:
            prefix, rest = template.split(BEGIN)
            _, suffix = rest.split(END)
            output = args.output_dir / "kerr_qnm_generated.py"
            output.write_text(prefix + BEGIN + render_splines(splines) + END + suffix)

    fixture = {
        "source": "TorchWave's independent Kerr eigenproblem solves",
        "license": "MIT",
        "external_numerical_inputs": [],
        "modes": fixtures,
    }
    write_json(args.output_dir / "holdouts.json", fixture)
    report = {
        "source": fixture["source"],
        "license": "MIT",
        "external_numerical_inputs": [],
        "spin_domain": [0.0, SPIN_MAX],
        "coordinate": "-log(1-a)",
        "spline_degree": DEGREE,
        "absolute_tolerance": args.absolute_tolerance,
        "relative_tolerance": args.relative_tolerance,
        "holdout_seed": 20260912,
        "modes": results,
        "overtone_separations": separations,
        "elapsed_seconds": time.monotonic() - start,
    }
    write_json(args.output_dir / "result.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Check native signed-spin QNM interpolation at predeclared Cook interstices.

The Cook table locates intervals but supplies no answer at these points. The
strict direct Kerr solver uses the native value only as a branch seed, so this
checks interpolation against the generating equation, not independent physics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import jax
import numpy as np
import torch

from . import qualify_generator as q

jax.config.update("jax_enable_x64", True)
from jaxwave.approximants.kerr_qnm_interp import evaluate_kerr_qnm_220  # noqa: E402
from torchwave.approximants.kerr_qnm_generated import signed_qnm_220  # noqa: E402
from torchwave.qnm.solver import solve  # noqa: E402


PLAN = q.ROOT / "dev/native_rom/evidence/2026-09-25/qnm_interstitial_solver_plan.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-archive", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "diffgw.seobnrv4.qnm-interstitial-solver-plan.v1"
            or plan["release_qualified"] is not False
            or plan["strict_solver"] is not True
            or plan["positive_spin_magnitudes"] != [
                0.0005, 0.5005, 0.9005, 0.9905, 0.99971875, 0.999859375,
                0.9998875, 0.9999,
            ]
            or plan["native_minus_solver_absolute_tolerance"] != 1e-9
            or plan["torch_minus_jax_absolute_tolerance"] != 1e-12):
        parser.error("Frozen interstitial QNM plan changed")
    for relative, expected in plan["source_sha256"].items():
        if q.sha256(q.ROOT / relative) != expected:
            parser.error(f"Pinned source changed: {relative}")
    archive_hash = q.sha256(args.author_archive)
    if archive_hash != plan["author_archive_sha256"]:
        parser.error("Cook archive SHA-256 mismatch")
    with h5py.File(args.author_archive) as archive:
        author_spins = {
            branch: values[values <= 0.9999]
            for branch, path in plan["author_datasets"].items()
            for values in [archive[path][:, 0]]
        }
    for branch, values in author_spins.items():
        if values.ndim != 1 or values[0] != 0 or not np.all(np.diff(values) > 0):
            raise ValueError(f"Unexpected Cook spin grid: {branch}")
    magnitudes = np.asarray(plan["positive_spin_magnitudes"])
    spins = np.concatenate((magnitudes, -magnitudes))
    with torch.no_grad():
        torch_values = np.stack([
            value.cpu().numpy() for value in signed_qnm_220(
                torch.tensor(spins, dtype=torch.float64))
        ], axis=-1)
    jax_values = np.stack([
        np.asarray(value) for value in evaluate_kerr_qnm_220(jax.numpy.asarray(spins))
    ], axis=-1)
    rows = []
    for index, signed_spin in enumerate(spins):
        magnitude = abs(float(signed_spin))
        branch = "positive" if signed_spin > 0 else "negative"
        grid = author_spins[branch]
        right = int(np.searchsorted(grid, magnitude, side="left"))
        if index % len(magnitudes) < 6:
            if right == 0 or right == len(grid) or not grid[right - 1] < magnitude < grid[right]:
                raise ValueError(f"Declared interstitial spin is not between Cook rows: {signed_spin}")
        elif index % len(magnitudes) == 6:
            if right != len(grid) or grid[-1] != 0.999875:
                raise ValueError("Cook terminal gap changed")
        elif magnitude != 0.9999:
            raise ValueError("Native domain edge changed")
        native = torch_values[index]
        root = solve(magnitude, 2, 2 if signed_spin > 0 else -2, 0,
                     omega_seed=complex(native[0], -native[1]), strict=True)
        if not root.converged:
            raise RuntimeError(f"Strict QNM solver failed at {signed_spin}")
        solved = np.array([root.omega.real, -root.omega.imag])
        rows.append({
            "signed_spin": float(signed_spin),
            "cook_left_spin": float(grid[right - 1]) if right else None,
            "cook_right_spin": float(grid[right]) if right < len(grid) else None,
            "torch": native.tolist(), "jax": jax_values[index].tolist(),
            "strict_solver": solved.tolist(),
            "torch_minus_solver_abs": abs(native - solved).tolist(),
            "jax_minus_solver_abs": abs(jax_values[index] - solved).tolist(),
            "torch_minus_jax_abs": abs(native - jax_values[index]).tolist(),
            "solver_depth": root.cf_depth,
            "solver_estimated_frequency_error": root.metadata["estimated_frequency_error"],
        })
    max_native_solver = max(max(row[backend + "_minus_solver_abs"])
                            for row in rows for backend in ("torch", "jax"))
    max_torch_jax = max(max(row["torch_minus_jax_abs"]) for row in rows)
    report = {
        "schema": "diffgw.seobnrv4.qnm-interstitial-solver.v1",
        "status": "diagnostic_complete", "release_qualified": False,
        "runtime": {
            "python": sys.version,
            "h5py": h5py.__version__,
            "jax": jax.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "scope": dict(q.SCOPE, description="Strict direct-equation interstitial QNM controls on both signed-spin branches.",
                      known_limitations=[
                          "The strict solver generated the native spline data and uses native values as branch seeds; this is not independent physical validation.",
                          "Finite interstitial points cannot establish a continuous-domain bound.",
                          "This does not decide historical interpolation compatibility or qualify whole waveforms.",
                      ]),
        "plan_sha256": q.sha256(PLAN), "script_sha256": q.sha256(__file__),
        "source_sha256": plan["source_sha256"],
        "author_archive_sha256": archive_hash,
        "rows": rows,
        "max_native_minus_solver_abs": max_native_solver,
        "max_torch_minus_jax_abs": max_torch_jax,
        "gates": {
            "native_matches_strict_solver_at_declared_points":
                max_native_solver < plan["native_minus_solver_absolute_tolerance"],
            "torch_jax_agree_at_declared_points":
                max_torch_jax < plan["torch_minus_jax_absolute_tolerance"],
        },
    }
    q.write_evidence(args.json_out, report)
    if not all(report["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

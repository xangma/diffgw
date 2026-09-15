#!/usr/bin/env python3
"""Review cached TEOB tolerance replay arrays without re-running the ODE.

The generating run retains adaptive knots, so state differences are evaluated
on the exact shared public grid whenever native grids differ.  Mode arrays are
compared only on their common public prefix.  This keeps the convergence claim
separate from the baseline and avoids index-wise comparisons of adaptive
trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts import diagnose_teob_paired_driver_clean as driver


LEVELS = ("baseline", "tight_0p1", "tight_0p01")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def complex_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    d = a - b
    return {
        "samples": int(a.size),
        "relative_l2": float(np.linalg.norm(d) / max(np.linalg.norm(b), 1e-300)),
        "max_abs": float(np.max(np.abs(d))) if d.size else None,
        "rms_abs": float(np.sqrt(np.mean(np.abs(d) ** 2))) if d.size else None,
    }


def native_grid(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    same = bool(np.array_equal(a, b))
    return {
        "same": same,
        "samples_a": int(a.size),
        "samples_b": int(b.size),
        "support_a": [float(a[0]), float(a[-1])],
        "support_b": [float(b[0]), float(b[-1])],
        "max_abs_difference": float(np.max(np.abs(a - b))) if same else None,
    }


def state_on_shared_public_grid(
    ta: np.ndarray, sa: np.ndarray, tb: np.ndarray, sb: np.ndarray,
    public_time: np.ndarray,
) -> dict[str, Any]:
    lo = max(float(ta[0]), float(tb[0]), float(public_time[0]))
    hi = min(float(ta[-1]), float(tb[-1]), float(public_time[-1]))
    q = public_time[(public_time >= lo) & (public_time <= hi)]
    if q.size < 2:
        return {"samples": int(q.size), "available": False}
    ia = np.column_stack([
        driver._natural_cubic_numpy(ta, sa[:, j], q) for j in range(sa.shape[1])
    ])
    ib = np.column_stack([
        driver._natural_cubic_numpy(tb, sb[:, j], q) for j in range(sb.shape[1])
    ])
    delta = ia - ib
    return {
        "available": True,
        "samples": int(q.size),
        "query_support": [float(q[0]), float(q[-1])],
        "max_abs_by_component": np.max(np.abs(delta), axis=0).tolist(),
        "rms_abs_by_component": np.sqrt(np.mean(delta ** 2, axis=0)).tolist(),
        "endpoint_abs_by_component": np.abs(delta[-1]).tolist(),
    }


def mode_on_common_public_prefix(a: np.ndarray, b: np.ndarray, public_time: np.ndarray) -> dict[str, Any]:
    n = min(a.size, b.size, public_time.size)
    return {
        "public_prefix_samples": int(n),
        "public_prefix_support": [float(public_time[0]), float(public_time[n - 1])] if n else None,
        "same_public_subset": bool(n == a.size == b.size),
        "metrics": complex_metrics(a[:n], b[:n]),
    }


def review(report_path: Path, cache_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    arrays = np.load(cache_path)
    public_time = arrays["public_time"]
    pairs: dict[str, Any] = {}
    for left, right in zip(LEVELS, LEVELS[1:]):
        ta, tb = arrays[f"{left}_trajectory_time"], arrays[f"{right}_trajectory_time"]
        sa, sb = arrays[f"{left}_trajectory_state"], arrays[f"{right}_trajectory_state"]
        ea, eb = report["levels"][left]["events"], report["levels"][right]["events"]
        pairs[f"{left}_vs_{right}"] = {
            "native_trajectory_grid": native_grid(ta, tb),
            "trajectory_state_on_shared_public_grid": state_on_shared_public_grid(
                ta, sa, tb, sb, public_time,
            ),
            "pre_nqc_mode_on_common_public_subset": mode_on_common_public_prefix(
                arrays[f"{left}_pre_nqc_mode"], arrays[f"{right}_pre_nqc_mode"], public_time,
            ),
            "processed_mode_on_common_public_subset": mode_on_common_public_prefix(
                arrays[f"{left}_processed_mode"], arrays[f"{right}_processed_mode"], public_time,
            ),
            "event_delta_initial_mass": {
                key: float(eb[key] - ea[key])
                for key in ("peak_time_initial_mass", "attachment_time_initial_mass", "t_match_initial_mass")
            },
            "event_index_delta": {
                key: int(eb[key] - ea[key])
                for key in ("attachment_sampled_index_joined", "peak_knot_index_auxiliary", "output_boundary_index_public_grid")
            },
        }
    return {
        "method": "cache-only postprocessing; no ODE rerun",
        "pairs_in_tolerance_order": pairs,
        "interpretation": (
            "Native states and pre-NQC modes differ little at the sampled tolerances, "
            "whereas the processed mode changes substantially. This directs investigation "
            "to amplification during NQC and attachment processing. It does not establish "
            "asymptotic convergence or exclude derivative/interpolation amplification of "
            "integration errors. The specific cause remains unresolved."
        ),
        "limits": [
            "Native trajectory grids are adaptive and were never compared by index.",
            "Mode arrays use only their exact common public prefix; no alignment or fitting was applied.",
            "This clean replay cannot identify which documented NQC or event convention causes the processed sensitivity.",
            "No experiment holding event times or the auxiliary grid fixed was run.",
            "The generating report's approach-to-baseline convergence criterion is superseded: convergence must be assessed across successively refined levels, not by proximity to the baseline.",
        ],
        "input_sha256": {str(report_path): sha256(report_path), str(cache_path): sha256(cache_path)},
        "generating_report_source_sha256": report.get("source_sha256"),
        "review_script_sha256": sha256(Path(__file__).resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = review(args.report, args.cache)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output), "pairs": list(result["pairs_in_tolerance_order"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

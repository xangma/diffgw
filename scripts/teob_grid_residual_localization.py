#!/usr/bin/env python3
"""Localize frozen grid-matching residuals against the public TD fixture.

This is cache-only.  It compares identical elapsed TD indices; it performs no
time/phase alignment, fitting, evolution, or correction.
The grid boundaries come from the accepted cache and the grid variant's own
matching-controls event times.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "docs/reviews/2026-09-15-teob-grid-nqc"
SOURCE = Path(__file__).resolve()
sys.path.insert(0, str(ROOT))
from scripts.teob_fd_cache_controls import (  # noqa: E402
    _validate_public_td_sidecar,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(z: np.lib.npyio.NpzFile, name: str) -> float:
    return float(np.asarray(z[name]).reshape(()))


def rel_l2(value: np.ndarray, reference: np.ndarray) -> float:
    den = float(np.linalg.norm(reference))
    return float(np.linalg.norm(value - reference) / den) if den else float(np.linalg.norm(value - reference))


def metrics(value: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    d = value - reference
    den = float(np.linalg.norm(reference))
    return {
        "samples": int(value.size),
        "relative_l2": float(np.linalg.norm(d) / den) if den else float(np.linalg.norm(d)),
        "value_norm": float(np.linalg.norm(value)),
        "reference_norm": den,
        "residual_norm": float(np.linalg.norm(d)),
        "max_abs": float(np.max(np.abs(d))) if value.size else 0.0,
    }


def phase_quantiles(value: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    valid = (np.abs(value) > max(scale * 1.0e-12, 1.0e-300)) & (np.abs(reference) > max(scale * 1.0e-12, 1.0e-300))
    if not np.any(valid):
        return {"valid_samples": 0, "status": "no_nonzero_overlap"}
    residual = np.unwrap(np.angle(value[valid] / reference[valid]))
    q = np.quantile(np.abs(residual), [0.5, 0.9, 0.99, 1.0])
    return {
        "status": "diagnostic_only_no_alignment",
        "valid_samples": int(valid.sum()),
        "abs_residual_rad_quantiles": {k: float(v) for k, v in zip(("q50", "q90", "q99", "q100"), q)},
        "signed_residual_rad_first": float(residual[0]),
        "signed_residual_rad_last": float(residual[-1]),
    }


def first_threshold(value: np.ndarray, reference: np.ndarray, time: np.ndarray, threshold: float, *, phase: bool = False) -> dict[str, Any] | None:
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    valid = (np.abs(reference) > max(scale * 1.0e-12, 1.0e-300)) & (np.abs(value) > max(scale * 1.0e-12, 1.0e-300))
    if phase:
        error = np.full(reference.shape, np.nan)
        error[valid] = np.abs(np.unwrap(np.angle(value[valid] / reference[valid])))
    else:
        error = np.full(reference.shape, np.nan)
        error[valid] = np.abs(np.abs(value[valid]) - np.abs(reference[valid])) / np.abs(reference[valid])
    hits = np.flatnonzero(np.isfinite(error) & (error >= threshold))
    if not hits.size:
        return None
    i = int(hits[0])
    return {"index": i, "td_time_dimensionless": float(time[i]), "error": float(error[i]), "threshold": threshold}


def verify_public_fixture(case: str, controls: dict[str, Any], request: dict[str, Any], root: Path) -> tuple[dict[str, Any], Path]:
    inputs = controls.get("inputs", {})
    out: dict[str, Any] = {"status": "complete", "case": case}
    public_path: Path | None = None
    for label, key in (("continuous", "continuous_npz"), ("public_fd", "public_fd_npz"), ("public_td", "public_td_npz"), ("public_td_sidecar", "public_td_json")):
        raw = inputs.get(key)
        if not raw:
            out[label] = {"status": "absent"}
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        row = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            out[label] = row
            continue
        row["sha256"] = sha256(path)
        expected = inputs.get(key.replace("_npz", "_sha256").replace("_json", "_json_sha256"))
        if expected is not None:
            row["declared_sha256"] = expected
            row["hash_match"] = row["sha256"] == expected
        if label == "public_td":
            public_path = path
        if label == "public_td_sidecar":
            side = json.loads(path.read_text())
            _validate_public_td_sidecar(side, request)
            row["request"] = side.get("request")
            row["sampling"] = side.get("sampling")
            row["validated_with"] = "scripts.teob_fd_cache_controls._validate_public_td_sidecar"
            row["fixture_hash_match"] = side.get("outputs", {}).get("public_td_sha256") == inputs.get("public_td_sha256")
            if not row["fixture_hash_match"]:
                raise ValueError("public TD sidecar output hash does not match controls.json")
        out[label] = row
    if public_path is None or not public_path.exists():
        raise FileNotFoundError("validated public_td_npz fixture is required")
    if not out.get("public_td", {}).get("hash_match", False):
        raise ValueError("public TD fixture hash does not match controls.json")
    return out, public_path


def localize(case: str, input_root: Path, output_root: Path) -> dict[str, Any]:
    case_dir = input_root / case
    continuous_path = case_dir / "continuous.npz"
    grid_path = DEFAULT_ROOT / case / "nominal.npz"
    derivative_controls_path = DEFAULT_ROOT / "derivative-controls.json"
    controls_path = case_dir / "controls.json"
    for p in (continuous_path, grid_path, derivative_controls_path, controls_path):
        if not p.exists():
            raise FileNotFoundError(p)
    derivative_controls = json.loads(derivative_controls_path.read_text())
    report = derivative_controls["cases"][case]
    controls = json.loads(controls_path.read_text())
    request = controls["request"]
    continuous_hash = sha256(continuous_path)
    grid_hash = sha256(grid_path)
    nominal = report["variants"]["nominal"]
    if nominal.get("output_sha256") != grid_hash:
        raise ValueError("derivative-controls nominal output hash does not match nominal.npz")
    if report.get("input_sha256") != continuous_hash:
        raise ValueError("derivative-controls input hash does not match continuous cache")
    fixture_verification, public_path = verify_public_fixture(case, controls, request, ROOT)
    with np.load(continuous_path, allow_pickle=False) as cont, np.load(grid_path, allow_pickle=False) as grid:
        time = np.asarray(cont["td_time"], dtype=np.float64)
        gt = np.asarray(grid["td_time"], dtype=np.float64)
        if not np.array_equal(time, gt):
            raise ValueError("continuous and nominal grid TD time axes differ")
        n = time.size
        with np.load(public_path, allow_pickle=False) as public:
            hp_c = np.asarray(public["exact_start_raw_h_plus"], dtype=np.float64)
            hc_c = np.asarray(public["exact_start_raw_h_cross"], dtype=np.float64)
        hp_g, hc_g = np.asarray(grid["h_plus_td"]), np.asarray(grid["h_cross_td"])
        if any(x.shape != (n,) for x in (hp_c, hc_c, hp_g, hc_g)):
            raise ValueError("TD arrays do not share one-dimensional common support")
        total_mass_seconds = scalar(cont, "total_mass_seconds")
        cached_dt_seconds = float((time[1] - time[0]) * total_mass_seconds) if n > 1 else float("nan")
        if not math.isclose(cached_dt_seconds, float(request["dt"]), rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError(f"matching cache timestep does not match controls request: {cached_dt_seconds} != {request['dt']}")
        zc, zg = hp_c - 1j * hc_c, hp_g - 1j * hc_g
        d = zg - zc
        # Grid boundaries: accepted trajectory PA/auxiliary times plus the
        # grid variant's own match and ringdown peak. Continuous event times
        # are intentionally not used here.
        pa_end = float(np.asarray(cont["pa_time"])[-1])
        aux_start = float(np.asarray(cont["nqc_auxiliary_time"])[0])
        match = float(nominal["matching_time_M"])
        attachment = float(nominal["attachment_time_M"])
        rd_peak = float(nominal["rd_peak_M"])
        boundaries = [0.0, pa_end, aux_start, match, attachment, float(time[-1]) + np.finfo(float).eps]
        if not all(boundaries[i] <= boundaries[i + 1] for i in range(len(boundaries) - 1)):
            raise ValueError(f"grid localization boundaries are not ordered: {boundaries}")
        cuts = [0, int(np.searchsorted(time, pa_end, side="right")), int(np.searchsorted(time, aux_start, side="left")), int(np.searchsorted(time, match, side="left")), int(np.searchsorted(time, attachment, side="left")), n]
        segment_names = ("PA_span", "early_ODE", "late_pre_NQC", "matching_window", "post_attachment")
        segment_rows: list[dict[str, Any]] = []
        total_energy = float(np.vdot(d, d).real)
        for name, lo, hi, blo, bhi in zip(segment_names, cuts[:-1], cuts[1:], boundaries[:-1], boundaries[1:]):
            row: dict[str, Any] = {"name": name, "declared_time_bounds_M": [blo, bhi], "index_bounds": [int(lo), int(hi)], "samples": int(max(0, hi - lo)), "empty": hi <= lo}
            if hi > lo:
                sl = slice(lo, hi)
                row["time_range_M"] = [float(time[lo]), float(time[hi - 1])]
                row["h_plus"] = metrics(hp_g[sl], hp_c[sl])
                row["h_cross"] = metrics(hc_g[sl], hc_c[sl])
                row["hp_minus_i_hc"] = metrics(zg[sl], zc[sl])
                row["phase_ratio"] = phase_quantiles(zg[sl], zc[sl])
                row["squared_residual_energy"] = float(np.vdot(d[sl], d[sl]).real)
                row["share_of_total_squared_residual"] = row["squared_residual_energy"] / total_energy if total_energy else None
            else:
                row.update({"h_plus": None, "h_cross": None, "hp_minus_i_hc": None, "phase_ratio": None, "squared_residual_energy": 0.0, "share_of_total_squared_residual": None})
            segment_rows.append(row)
        cumulative = np.cumsum(np.abs(d) ** 2) / total_energy if total_energy else np.zeros(n)
        amplitude_rel = np.divide(np.abs(np.abs(zg) - np.abs(zc)), np.abs(zc), out=np.full(n, np.nan), where=np.abs(zc) > max(float(np.max(np.abs(zc))) * 1e-12, 1e-300))
        phase_abs = np.full(n, np.nan)
        valid = np.abs(zg) > max(float(np.max(np.abs(zc))) * 1e-12, 1e-300)
        valid &= np.abs(zc) > max(float(np.max(np.abs(zc))) * 1e-12, 1e-300)
        phase_abs[valid] = np.abs(np.unwrap(np.angle(zg[valid] / zc[valid])))
        output_root.mkdir(parents=True, exist_ok=True)
        arrays_path = output_root / f"localization-{case}.npz"
        np.savez_compressed(arrays_path, td_time=time, complex_residual=d, normalized_cumulative_residual_energy=cumulative, amplitude_relative_error=amplitude_rel, phase_absolute_error_rad=phase_abs)
        result = {
            "status": "complete", "case": case,
            "request": request,
            "input_hashes": {"continuous_npz": continuous_hash, "nominal_grid_npz": grid_hash, "derivative_controls_json": sha256(derivative_controls_path), "controls_json": sha256(controls_path), "derivative_controls_declared_input": report.get("input_sha256"), "nominal_declared_output": nominal.get("output_sha256"), "public_td_npz": sha256(public_path), "public_td_json": sha256(Path(controls["inputs"]["public_td_json"]))},
            "matching_fixture": {"parameters": report.get("case", {}).get("params"), "grid": report.get("case", {}).get("grid"), "td_dt_seconds_from_cache": cached_dt_seconds, "td_dt_seconds_requested": float(request["dt"]), "td_dt_exact_within_1e-15": True, "nominal_output_hash_match": True, "reference": "public TD exact-start fixture"},
            "fixture_verification": fixture_verification,
            "grid_event_boundaries": {"pa_end_M": pa_end, "nqc_auxiliary_start_M": aux_start, "match_M": match, "attachment_time_M": attachment, "rd_peak_M": rd_peak, "rule": "PA <= accepted pa_time[-1]; early ODE until accepted nqc_auxiliary_time[0]; late/pre-NQC until nominal grid matching_time_M; matching window until nominal attachment_time_M; post-attachment after nominal attachment_time_M"},
            "global_td": {"samples": n, "td_time_M": [float(time[0]), float(time[-1])], "hp_minus_i_hc": metrics(zg, zc), "h_plus": metrics(hp_g, hp_c), "h_cross": metrics(hc_g, hc_c), "phase_ratio": phase_quantiles(zg, zc), "first_amplitude_relative_ge_1e-6": first_threshold(zg, zc, time, 1e-6), "first_phase_abs_rad_ge_1e-6": first_threshold(zg, zc, time, 1e-6, phase=True)},
            "segments": segment_rows,
            "outputs": {"arrays_npz": str(arrays_path)},
            "fd": {"status": "omitted", "reason": "FD attribution is covered independently; this report localizes TD residuals only."},
            "method_limits": ["TD comparisons use identical elapsed output indices against the validated public exact-start fixture, with no fitted shifts.", "The matching window can be one or zero samples at the cached TD spacing; empty segments remain explicit.", "TD localization does not assign the public residual to a unique FD band."],
            "provenance": {"host": socket.gethostname(), "cwd": str(Path.cwd()), "command": shlex.join([sys.executable, *sys.argv]), "script_sha256": sha256(SOURCE), "python": sys.executable},
        }
    report_path = output_root / f"localization-{case}.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("catalog", "bbh_reference"), required=True)
    parser.add_argument("--input-root", type=Path, default=ROOT / "docs/reviews/2026-09-15-teob-fd-attribution")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = localize(args.case, args.input_root, args.output_root)
    print(json.dumps({"status": result["status"], "case": result["case"], "output": result["outputs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

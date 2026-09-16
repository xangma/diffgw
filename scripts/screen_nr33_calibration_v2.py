#!/usr/bin/env python3
"""NR-only quality screen for the replacement XHM (3,3) calibration set.

The policy is an input, not a set of defaults.  This screen reads only the
fitter-compatible manifest and its NPZ files, uses exact level metadata for
the FFT/QNM calculation, and never downloads, fits, or calls LAL.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.fit_nr33_amplitude_pilot import FFTControls, frequency_amplitude, independent_qnm33


REQUIRED_GATE_KEYS = (
    "max_reference_spin_magnitude", "max_eccentricity_upper_bound",
    "min_post_peak_duration_M", "max_extraction_relative_l2",
    "max_resolution_relative_l2", "max_preprocessing_relative_l2",
    "min_training_simulations", "min_holdout_simulations",
)
REQUIRED_CONTROL_KEYS = ("sample_dt", "start_taper_duration", "end_taper_duration", "pad_factor", "fmax")
REQUIRED_VARIANTS = ("sample_dt", "end_taper_duration")
DEFAULT_OUTPUT = Path("docs/reviews/2026-09-15-independent-calibration-v2/quality-screen.json")
DEFAULT_QUALIFIED = Path("docs/reviews/2026-09-15-independent-calibration-v2/nr33-qualified-manifest.json")
PINNED_POLICY_SHA256 = "1a863811b90c410cd5a9b0b52a0bd53696b869a0d503cb8fa61d9c054842138d"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value is not finite")
    return result


def _field(row: dict[str, Any], *names: str) -> Any:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
        if name in metadata and metadata[name] is not None:
            return metadata[name]
    raise KeyError(names[0])


def _parse_bound(value: Any) -> tuple[float, str]:
    """Parse a finite eccentricity or an explicit numeric upper bound."""
    if isinstance(value, str):
        text = value.strip()
        upper = text.startswith("<")
        if upper:
            text = text[1:].strip()
        number = _finite(text)
        if number < 0:
            raise ValueError("eccentricity bound must be non-negative")
        return number, "upper_bound" if upper else "value"
    number = _finite(value)
    if number < 0:
        raise ValueError("eccentricity must be non-negative")
    return number, "value"


def _vector(row: dict[str, Any], *names: str) -> np.ndarray:
    value = np.asarray(_field(row, *names), dtype=float).reshape(-1)
    if value.size != 3 or not np.all(np.isfinite(value)):
        raise ValueError("spin must be a finite three-vector")
    return value


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("quality policy schema_version must be 1")
    gates = policy.get("gates")
    controls = policy.get("controls")
    variants = policy.get("preprocessing_variants")
    band = policy.get("frequency_band")
    for key in REQUIRED_GATE_KEYS:
        if not isinstance(gates, dict) or key not in gates:
            raise ValueError(f"quality policy missing gates.{key}")
        _finite(gates[key])
    for key in REQUIRED_CONTROL_KEYS:
        if not isinstance(controls, dict) or key not in controls:
            raise ValueError(f"quality policy missing controls.{key}")
        _finite(controls[key])
    if not isinstance(variants, list) or len(variants) != 2:
        raise ValueError("quality policy must declare exactly two preprocessing_variants")
    seen = set()
    for variant in variants:
        if not isinstance(variant, dict) or len(variant) != 1:
            raise ValueError("each preprocessing variant must have one explicit control")
        key = next(iter(variant))
        if key not in REQUIRED_VARIANTS or key in seen:
            raise ValueError("preprocessing variants must declare sample_dt and end_taper_duration")
        seen.add(key)
        _finite(variant[key])
    if seen != set(REQUIRED_VARIANTS):
        raise ValueError("preprocessing variants must declare sample_dt and end_taper_duration")
    if not isinstance(band, dict) or band.get("minimum") != "0.95 * independent f_ring_33 from highest N3 matching remnant metadata" or band.get("maximum") != 0.3:
        raise ValueError("quality policy frequency_band is not the declared fixed policy")
    for key in ("target_mass_ratios", "training_target_q", "holdout_target_q"):
        if not isinstance(policy.get(key), list) or not policy[key]:
            raise ValueError(f"quality policy missing non-empty {key}")
    actual_sha256 = sha256(path)
    if actual_sha256 != PINNED_POLICY_SHA256:
        raise ValueError(f"quality policy SHA256 is not pinned: {actual_sha256}")
    return policy


def read_manifest(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    rows = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain a non-empty cases list")
    result = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id") or not row.get("path"):
            raise ValueError("each manifest case needs id and path")
        item = copy.deepcopy(row)
        item["path"] = str((path.parent / row["path"]).resolve()) if not Path(row["path"]).is_absolute() else str(Path(row["path"]))
        result.append(item)
    return result


def _level_order(row: dict[str, Any]) -> int:
    text = str(_field(row, "resolution"))
    match = re.fullmatch(r"Lev(\d+)", text)
    if not match:
        raise ValueError(f"invalid resolution {text!r}")
    return int(match.group(1))


def _order(row: dict[str, Any]) -> int:
    match = re.search(r"/N(\d+)$", str(row["id"]))
    if not match:
        value = _field(row, "extrapolation_order")
        return int(value)
    return int(match.group(1))


def _source_split(row: dict[str, Any]) -> str:
    try:
        value = _field(row, "source_split", "split")
    except KeyError:
        return ""
    return str(value)


def _declared_target_q(row: dict[str, Any], targets: list[float]) -> float:
    """Read the predeclared target; never infer it from measured q."""
    target = _finite(_field(row, "target_q"))
    if not any(math.isclose(target, float(value), rel_tol=0.0, abs_tol=1e-12)
               for value in targets):
        raise ValueError(f"target_q {target} is not in the frozen target list")
    return target


def _candidate_rank(row: dict[str, Any]) -> int:
    rank = int(_field(row, "candidate_rank"))
    if rank < 0:
        raise ValueError("candidate_rank must be non-negative")
    return rank


def _is_deprecated(row: dict[str, Any]) -> bool:
    values = [row.get(key) for key in ("deprecated", "source_deprecated", "is_deprecated")]
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    values += [metadata.get(key) for key in ("deprecated", "source_deprecated", "is_deprecated")]
    return any(value is True or str(value).lower() == "deprecated" for value in values)


def _metadata_gate(row: dict[str, Any]) -> dict[str, Any]:
    required = ("q", "resolution", "extrapolation_order", "reference_time", "relaxation_time",
                "final_mass_fraction", "final_spin", "initial_spin1", "initial_spin2",
                "reference_spin1", "reference_spin2", "peak_time", "time_end", "eccentricity")
    missing = []
    errors = []
    for name in required:
        try:
            _field(row, name)
        except KeyError:
            missing.append(name)
    exact = not missing and not _is_deprecated(row)
    if exact:
        # The level/order encoded by the record id must agree with the
        # catalog metadata used by the QNM and FFT calculation.
        try:
            level_match = re.search(r"/Lev(\d+)/N\d+$", str(row["id"]))
            exact = (level_match is not None and
                     int(level_match.group(1)) == _level_order(row) and
                     int(_field(row, "extrapolation_order")) == _order(row))
            if not exact:
                errors.append("record id level/order disagrees with metadata")
        except (KeyError, ValueError):
            exact = False
            errors.append("invalid record id level/order")
    if exact:
        try:
            for name in ("q", "reference_time", "relaxation_time", "final_mass_fraction",
                         "peak_time", "time_end"):
                _finite(_field(row, name))
            _parse_bound(_field(row, "eccentricity"))
            for name in ("final_spin", "initial_spin1", "initial_spin2",
                         "reference_spin1", "reference_spin2"):
                _vector(row, name)
        except (KeyError, ValueError):
            exact = False
            errors.append("non-finite or malformed metadata value")
    return {"pass": exact, "missing_fields": missing, "errors": errors,
            "deprecated": _is_deprecated(row),
            "source_split": _source_split(row)}


def _load_wave(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(row["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        expected = _field(row, "npz_sha256")
    except KeyError as exc:
        raise ValueError(f"manifest lacks required NPZ SHA256 for {row['id']}") from exc
    if sha256(path).lower() != str(expected).lower():
        raise ValueError(f"NPZ SHA256 mismatch for {row['id']}")
    with np.load(path, allow_pickle=False) as data:
        if "time" not in data or "h33" not in data or "h22" not in data:
            raise ValueError(f"{path} must contain time, h33, and h22")
        t = np.asarray(data["time"], dtype=float).reshape(-1)
        h33 = np.asarray(data["h33"], dtype=complex).reshape(-1)
        h22 = np.asarray(data["h22"], dtype=complex).reshape(-1)
    if t.size < 32 or h33.size != t.size or h22.size != t.size or not np.all(np.isfinite(t)) or not np.all(np.isfinite(h33)) or not np.all(np.isfinite(h22)) or np.any(np.diff(t) <= 0):
        raise ValueError(f"invalid waveform arrays in {path}")
    return t, h33, h22


def _spectrum(row: dict[str, Any], controls: FFTControls) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    t, h33, _ = _load_wave(row)
    relaxation = _finite(_field(row, "relaxation_time"))
    peak = _finite(_field(row, "peak_time"))
    f, amp, details = frequency_amplitude(t, h33, controls, relaxation, peak, return_metadata=True)
    return f, amp, details


def _common_metrics(base: tuple[np.ndarray, np.ndarray], others: Iterable[tuple[np.ndarray, np.ndarray]], lower: float, upper: float) -> dict[str, Any]:
    fb, ab = base
    arrays = [base, *others]
    lo = max(lower, *(float(f[0]) for f, _ in arrays))
    hi = min(upper, *(float(f[-1]) for f, _ in arrays))
    grid = fb[(fb >= lo) & (fb <= hi)]
    if grid.size < 8 or not (float(grid[0]) >= lo and float(grid[-1]) <= hi):
        raise ValueError("no adequate common frequency grid in fixed band")
    if np.linalg.norm(np.interp(grid, fb, ab)) == 0:
        raise ValueError("zero baseline spectrum")
    values = []
    for f, amp in others:
        if grid[0] < f[0] or grid[-1] > f[-1]:
            raise ValueError("comparison would require frequency extrapolation")
        values.append(np.interp(grid, f, amp))
    baseline = np.interp(grid, fb, ab)
    denom = float(np.linalg.norm(baseline))
    return {"relative_l2": [float(np.linalg.norm(value - baseline) / denom) for value in values],
            "frequency_Mf_range": [float(grid[0]), float(grid[-1])], "bins": int(grid.size),
            "denominator": "highest-resolution N3 baseline", "no_extrapolation": True}


def _gate(name: str, passed: bool, evidence: Any, limit: Any = None, required: bool = True) -> dict[str, Any]:
    item = {"name": name, "pass": bool(passed), "required": required, "evidence": evidence}
    if limit is not None:
        item["limit"] = limit
    return item


def screen_simulation(rows: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    """Screen one simulation; rows must contain its N2/N3 levels."""
    gates = policy["gates"]
    controls_cfg = policy["controls"]
    controls = FFTControls(**controls_cfg)
    variants = policy["preprocessing_variants"]
    levels: dict[int, dict[int, dict[str, Any]]] = {}
    for row in rows:
        levels.setdefault(_level_order(row), {})[_order(row)] = row
    available = sorted(levels, reverse=True)
    result: dict[str, Any] = {"sxs_id": str(_field(rows[0], "sxs_id")),
                              "available_resolutions": [f"Lev{x}" for x in available],
                              "primary_case_id": None, "source_split": "",
                              "gates": [], "passed": False}
    if len(available) < 2:
        result["evaluation_status"] = "unavailable_required_data"
        result["gates"].append(_gate("two_resolutions_N2_N3", False, result["available_resolutions"], "at least two"))
        return result
    high, next_level = available[:2]
    missing = {f"Lev{level}": [order for order in (2, 3) if order not in levels[level]]
               for level in (high, next_level) if any(order not in levels[level] for order in (2, 3))}
    if missing:
        result["evaluation_status"] = "unavailable_required_data"
        result["gates"].append(_gate("two_resolutions_N2_N3", False, {"missing_orders": missing}))
        return result
    high2, high3 = levels[high][2], levels[high][3]
    next3 = levels[next_level][3]
    result.update({"highest_resolution": f"Lev{high}", "next_resolution": f"Lev{next_level}",
                   "primary_case_id": high3["id"], "source_split": _source_split(high3)})
    metadata_rows = [high2, high3, next3]
    metadata_results = {row["id"]: _metadata_gate(row) for row in metadata_rows}
    result["metadata"] = metadata_results
    result["gates"].append(_gate("exact_nondeprecated_level_metadata", all(x["pass"] for x in metadata_results.values()), metadata_results))
    try:
        spin1 = _vector(high3, "reference_spin1")
        spin2 = _vector(high3, "reference_spin2")
        magnitudes = [float(np.linalg.norm(spin1)), float(np.linalg.norm(spin2))]
        max_spin = max(magnitudes)
        result["gates"].append(_gate("near_nonspin", max_spin <= gates["max_reference_spin_magnitude"], {"magnitudes": magnitudes, "maximum": max_spin}, gates["max_reference_spin_magnitude"]))
    except (KeyError, ValueError) as exc:
        result["gates"].append(_gate("near_nonspin", False, str(exc), gates["max_reference_spin_magnitude"]))
    try:
        eccentricity, eccentricity_kind = _parse_bound(_field(high3, "eccentricity", "e"))
        result["gates"].append(_gate("eccentricity", eccentricity <= gates["max_eccentricity_upper_bound"], {"value": eccentricity, "kind": eccentricity_kind}, gates["max_eccentricity_upper_bound"]))
    except (KeyError, ValueError) as exc:
        result["gates"].append(_gate("eccentricity", False, str(exc), gates["max_eccentricity_upper_bound"]))
    try:
        t, _, h22 = _load_wave(high3)
        peak_index = int(np.argmax(np.abs(h22)))
        peak_from_h22 = float(t[peak_index])
        peak_metadata = _finite(_field(high3, "peak_time"))
        post_peak = float(t[-1] - peak_from_h22)
        result["h22_peak"] = {"metadata_time": peak_metadata, "decoded_time": peak_from_h22,
                               "metadata_decoded_difference_M": peak_metadata - peak_from_h22,
                               "duration_to_end_M": post_peak, "duration_source": "decoded h22"}
        result["gates"].append(_gate("post_peak_duration", post_peak >= gates["min_post_peak_duration_M"], post_peak, gates["min_post_peak_duration_M"]))
    except (KeyError, ValueError, FileNotFoundError) as exc:
        result["evaluation_status"] = "failed_numerical_quality"
        result["gates"].append(_gate("post_peak_duration", False, str(exc), gates["min_post_peak_duration_M"]))
        return result
    try:
        qnm = independent_qnm33({**high3.get("metadata", {}), **high3})
        lower = 0.95 * float(qnm["f_ring_33"])
        upper = float(controls.fmax)
        spectra = {}
        for label, row, ctl in (("high_N3", high3, controls), ("high_N2", high2, controls), ("next_N3", next3, controls)):
            spectra[label] = _spectrum(row, ctl)[:2]
        variant_spectra = []
        for variant in variants:
            cfg = dict(controls_cfg); cfg.update(variant)
            variant_spectra.append(_spectrum(high3, FFTControls(**cfg))[:2])
        result["qnm"] = qnm
        result["frequency_band"] = {"minimum": lower, "maximum": upper}
        all_metrics = _common_metrics(spectra["high_N3"],
                                       [spectra["high_N2"], spectra["next_N3"], *variant_spectra],
                                       lower, upper)
        extraction = all_metrics["relative_l2"][0]
        resolution = all_metrics["relative_l2"][1]
        preprocessing = dict(all_metrics)
        preprocessing["relative_l2"] = all_metrics["relative_l2"][2:]
        result["metrics"] = {"extraction_high_N2_vs_high_N3_relative_l2": extraction,
                              "resolution_next_N3_vs_high_N3_relative_l2": resolution,
                              "preprocessing_variant_relative_l2": preprocessing}
        result["gates"] += [
            _gate("high_resolution_extraction", extraction <= gates["max_extraction_relative_l2"], extraction, gates["max_extraction_relative_l2"]),
            _gate("next_resolution", resolution <= gates["max_resolution_relative_l2"], resolution, gates["max_resolution_relative_l2"]),
            _gate("preprocessing_variations", all(x <= gates["max_preprocessing_relative_l2"] for x in preprocessing["relative_l2"]), preprocessing, gates["max_preprocessing_relative_l2"]),
        ]
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
        result["evaluation_status"] = "failed_numerical_quality"
        result["gates"].append(_gate("spectrum_metrics", False, str(exc)))
    result["passed"] = all(item["pass"] for item in result["gates"] if item["required"])
    result.setdefault("evaluation_status", "passed" if result["passed"] else "failed_quality_gate")
    return result


def screen_manifest(manifest_path: Path, policy_path: Path, qualified_path: Path | None = None) -> dict[str, Any]:
    policy = load_policy(policy_path)
    rows = read_manifest(manifest_path)
    excluded = set(policy.get("excluded_sxs_ids", []))
    grouped: dict[str, list[dict[str, Any]]] = {}
    ignored = []
    declared: dict[str, tuple[float, int]] = {}
    for row in rows:
        sid = str(_field(row, "sxs_id"))
        if sid in excluded:
            ignored.append(row["id"])
            continue
        target = _declared_target_q(row, policy["target_mass_ratios"])
        rank = _candidate_rank(row)
        prior = declared.get(sid)
        if prior is not None and prior != (target, rank):
            raise ValueError(f"all records for {sid} must share target_q and candidate_rank")
        declared[sid] = (target, rank)
        grouped.setdefault(sid, []).append(row)
    simulations = []
    for sid, sim_rows in grouped.items():
        screen = screen_simulation(sim_rows, policy)
        target, rank = declared[sid]
        screen["target_q"] = target
        screen["candidate_rank"] = rank
        expected_split = ("holdout" if any(math.isclose(target, float(q), rel_tol=0.0, abs_tol=1e-12)
                                          for q in policy["holdout_target_q"]) else "training")
        # Lower-order/resolution rows are deliberately validation_only.  The
        # split gate applies to the selected primary (highest N3), which
        # screen_simulation records in source_split.
        actual_split = str(screen["source_split"]).lower()
        split_pass = actual_split == expected_split or (expected_split == "training" and actual_split == "train")
        screen["gates"].append(_gate("frozen_source_split", split_pass,
                                      {"actual": actual_split, "expected": expected_split}))
        screen["passed"] = bool(screen["passed"] and split_pass)
        simulations.append(screen)
    selected = []
    for target in policy["target_mass_ratios"]:
        candidates = [x for x in simulations if x["target_q"] == float(target) and x["passed"]]
        if candidates:
            selected.append(min(candidates, key=lambda x: (x["candidate_rank"], x["sxs_id"])))
    train = [x for x in selected if x["source_split"].lower() in {"train", "training"}]
    holdout = [x for x in selected if x["source_split"].lower() == "holdout"]
    all_targets = len(selected) == len(policy["target_mass_ratios"])
    enough = len(train) >= int(policy["gates"]["min_training_simulations"]) and len(holdout) >= int(policy["gates"]["min_holdout_simulations"])
    overall = all_targets and enough
    qualified_info = None
    if overall and qualified_path is not None:
        chosen_by_id = {x["primary_case_id"]: x for x in selected}
        chosen = []
        for row in rows:
            if row["id"] in chosen_by_id:
                selected_case = chosen_by_id[row["id"]]
                item = copy.deepcopy(row)
                item["target_q"] = selected_case["target_q"]
                item["candidate_rank"] = selected_case["candidate_rank"]
                item["sxs_id"] = str(_field(row, "sxs_id"))
                item["resolution"] = str(_field(row, "resolution"))
                item["extrapolation_order"] = int(_field(row, "extrapolation_order"))
                item["source_split"] = _source_split(row)
                chosen.append(item)
        payload = {"cases": chosen, "screening": {"policy_sha256": sha256(policy_path), "manifest_sha256": sha256(manifest_path), "selected_primary_case_ids": [x["primary_case_id"] for x in selected]}}
        qualified_path.parent.mkdir(parents=True, exist_ok=True)
        qualified_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        qualified_info = {"path": str(qualified_path), "sha256": sha256(qualified_path), "case_count": len(chosen)}
    return {"schema_version": 1, "overall_pass": overall, "policy_sha256": sha256(policy_path),
            "input_manifest_sha256": sha256(manifest_path), "selected_primary_case_ids": [x["primary_case_id"] for x in selected],
            "simulations": simulations, "ignored_excluded_case_ids": ignored,
            "qualified_manifest": qualified_info, "counts": {"selected": len(selected), "training": len(train), "holdout": len(holdout), "required_targets": len(policy["target_mass_ratios"])}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qualified-output", type=Path, default=DEFAULT_QUALIFIED)
    args = parser.parse_args()
    report = screen_manifest(args.manifest, args.policy, args.qualified_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"overall_pass": report["overall_pass"], "selected": len(report["selected_primary_case_ids"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

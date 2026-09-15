"""Bounded tests for the frozen NR-only quality screen."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.screen_nr33_calibration_v2 import (
    _parse_bound,
    load_policy,
    sha256,
    screen_manifest,
    screen_simulation,
)


ROOT = Path(__file__).parents[1]
POLICY = ROOT / "docs/reviews/2026-09-15-independent-calibration-v2/quality-policy.json"


def _wave(path: Path, scale: float = 1.0) -> None:
    t = np.arange(0.0, 1000.1, 0.1)
    h22 = np.exp(-((t - 800.0) / 20.0) ** 2).astype(complex)
    envelope = np.zeros_like(t)
    support = (t >= 600.0) & (t <= 900.0)
    envelope[support] = np.sin(np.pi * (t[support] - 600.0) / 300.0) ** 4
    h33 = scale * envelope * np.exp(2j * np.pi * 0.05 * t)
    np.savez(path, time=t, h22=h22, h33=h33)


def _row(path: Path, level: int, order: int, split: str = "holdout") -> dict:
    return {
        "id": f"SXS:BBH:9999/Lev{level}/N{order}",
        "path": str(path),
        "sxs_id": "SXS:BBH:9999",
        "q": 3.0,
        "target_q": 3.0,
        "candidate_rank": 0,
        "resolution": f"Lev{level}",
        "extrapolation_order": order,
        "reference_time": 0.0,
        "relaxation_time": 100.0,
        "final_mass_fraction": 0.95,
        "final_spin": [0.5, 0.0, 0.0],
        "initial_spin1": [0.0, 0.0, 0.0],
        "initial_spin2": [0.0, 0.0, 0.0],
        "reference_spin1": [0.0, 0.0, 0.0],
        "reference_spin2": [0.0, 0.0, 0.0],
        "peak_time": 800.0,
        "time_end": 1000.0,
        "eccentricity": "<1e-3",
        "f_ring_33": 0.05,
        "f_damp_33": 0.01,
        "source_split": split,
        "npz_sha256": sha256(path),
    }


def test_bound_accepts_explicit_upper_bound() -> None:
    assert _parse_bound("<1e-3") == (pytest.approx(1e-3), "upper_bound")


def test_policy_values_are_mandatory(tmp_path: Path) -> None:
    policy = json.loads(POLICY.read_text())
    del policy["gates"]["max_resolution_relative_l2"]
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="max_resolution_relative_l2"):
        load_policy(path)


def test_policy_numeric_change_is_rejected_by_pinned_hash(tmp_path: Path) -> None:
    policy = json.loads(POLICY.read_text())
    policy["gates"]["max_resolution_relative_l2"] = 0.02
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="SHA256 is not pinned"):
        load_policy(path)


def test_screen_identical_highest_two_resolutions_passes(tmp_path: Path) -> None:
    policy = load_policy(POLICY)
    path = tmp_path / "wave.npz"
    _wave(path)
    result = screen_simulation(
        [_row(path, level, order) for level in (4, 5) for order in (2, 3)], policy
    )
    assert result["passed"] is True
    assert result["primary_case_id"].endswith("/Lev5/N3")
    assert all(gate["pass"] for gate in result["gates"])


def test_screen_rejects_highest_resolution_extraction_disagreement(tmp_path: Path) -> None:
    policy = load_policy(POLICY)
    good = tmp_path / "good.npz"
    bad = tmp_path / "bad.npz"
    _wave(good)
    _wave(bad, scale=1.1)
    rows = [_row(good, level, order) for level in (4, 5) for order in (2, 3)]
    rows[-2]["path"] = str(bad)  # highest Lev5/N2
    rows[-2]["npz_sha256"] = sha256(bad)
    result = screen_simulation(rows, policy)
    assert result["passed"] is False
    assert result["evaluation_status"] == "failed_quality_gate"
    extraction = next(g for g in result["gates"] if g["name"] == "high_resolution_extraction")
    assert extraction["pass"] is False
    assert extraction["evidence"] > policy["gates"]["max_extraction_relative_l2"]


def test_spin_gate_uses_vector_norm(tmp_path: Path) -> None:
    policy = load_policy(POLICY)
    path = tmp_path / "wave.npz"
    _wave(path)
    rows = [_row(path, level, order) for level in (4, 5) for order in (2, 3)]
    rows[-1]["reference_spin1"] = [8e-5, 8e-5, 0.0]  # norm exceeds 1e-4.
    result = screen_simulation(rows, policy)
    gate = next(g for g in result["gates"] if g["name"] == "near_nonspin")
    assert gate["pass"] is False
    assert gate["evidence"]["maximum"] > policy["gates"]["max_reference_spin_magnitude"]


def test_screen_does_not_skip_higher_incomplete_resolution(tmp_path: Path) -> None:
    policy = load_policy(POLICY)
    path = tmp_path / "wave.npz"
    _wave(path)
    rows = [_row(path, 4, order) for order in (2, 3)]
    rows += [_row(path, 5, 3)]  # Lev5 is highest but lacks N2.
    result = screen_simulation(rows, policy)
    assert result["passed"] is False
    assert result["evaluation_status"] == "unavailable_required_data"
    gate = next(g for g in result["gates"] if g["name"] == "two_resolutions_N2_N3")
    assert gate["evidence"]["missing_orders"] == {"Lev5": [2]}


def test_manifest_contract_does_not_qualify_incomplete_targets(tmp_path: Path) -> None:
    path = tmp_path / "wave.npz"
    _wave(path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": [_row(path, level, order) for level in (4, 5) for order in (2, 3)]}))
    qualified = tmp_path / "qualified.json"
    report = screen_manifest(manifest, POLICY, qualified)
    assert report["overall_pass"] is False
    assert report["input_manifest_sha256"]
    assert len(report["selected_primary_case_ids"]) == 1
    assert report["qualified_manifest"] is None
    assert not qualified.exists()
    assert {"sxs_id", "target_q", "source_split", "passed", "primary_case_id", "gates"} <= set(report["simulations"][0])


def test_manifest_uses_explicit_candidate_rank_and_rejects_validation_alias(tmp_path: Path) -> None:
    path = tmp_path / "wave.npz"
    _wave(path)
    rows = []
    for sid, rank, split in (("SXS:BBH:9998", 1, "holdout"), ("SXS:BBH:9997", 0, "validation_only")):
        candidate = [_row(path, level, order, split) for level in (4, 5) for order in (2, 3)]
        for row in candidate:
            row["sxs_id"] = sid
            row["id"] = row["id"].replace("SXS:BBH:9999", sid)
            row["candidate_rank"] = rank
        rows.extend(candidate)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": rows}))
    report = screen_manifest(manifest, POLICY)
    assert report["selected_primary_case_ids"] == ["SXS:BBH:9998/Lev5/N3"]


def test_manifest_uses_primary_split_when_lower_rows_are_validation_only(tmp_path: Path) -> None:
    path = tmp_path / "wave.npz"
    _wave(path)
    rows = []
    for level in (4, 5):
        for order in (2, 3):
            rows.append(_row(path, level, order, "holdout" if level == 5 and order == 3 else "validation_only"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": rows}))
    report = screen_manifest(manifest, POLICY)
    simulation = report["simulations"][0]
    split_gate = next(g for g in simulation["gates"] if g["name"] == "frozen_source_split")
    assert split_gate["pass"] is True
    assert simulation["source_split"] == "holdout"

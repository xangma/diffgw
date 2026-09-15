from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.teob_fixed_event_replay import _forced_peak, metrics
from torchwave.approximants.eob import teob_ringdown_assembly_clean as ringdown


def test_metrics_uses_complex_l2_and_rms() -> None:
    got = metrics(np.array([3 + 4j, 0j]), np.array([0 + 4j, 0j]))
    assert got["samples"] == 2
    assert got["max_abs"] == 3.0
    assert got["rms_abs"] == 3.0 / np.sqrt(2.0)
    assert got["relative_l2"] == 0.75


def test_metrics_rejects_shape_mismatch() -> None:
    with np.testing.assert_raises(ValueError):
        metrics(np.ones(3, dtype=complex), np.ones(2, dtype=complex))


def test_event_override_is_scoped_to_local_ringdown_module() -> None:
    original = ringdown._peak_polynomial
    with _forced_peak(torch.tensor(12.5, dtype=torch.float64)):
        assert ringdown._peak_polynomial is not original
    assert ringdown._peak_polynomial is original


def test_controlled_pair_uses_one_grid_and_all_fixed_event_times() -> None:
    report_path = (Path(__file__).resolve().parents[1]
                   / "docs/reviews/2026-09-15-parity-strategy-experiment/eob"
                   / "native-fixed-event-reassembly.json")
    report = json.loads(report_path.read_text())
    pair = report["controlled_pair"]
    assert pair["common_grid_equal"] is True
    assert pair["fixed_event_times_equal"] is True
    rows = pair["variants"]["common_grid_fixed_events"]
    left, right = rows[pair["levels"][0]], rows[pair["levels"][1]]
    assert left["auxiliary_time"] == right["auxiliary_time"]
    for row in (left, right):
        assert row["nqc"]["peak_grid_time_M"] == pair["fixed_nqc_peak_time_M"]
        assert row["nqc"]["matching_time_M"] == pair["fixed_nqc_time_M"]
        assert row["events"]["peak_time_initial_mass"] == pair["fixed_rd_peak_time_M"]

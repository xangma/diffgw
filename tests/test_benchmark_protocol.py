"""Tests for DiffGW Benchmarking Protocol, Artifact Sealing, and Parity Gatekeeping."""

import copy
import numpy as np
import pytest

from torchwave.protocol import (
    artifact_content_sha256,
    bootstrap_median_ci,
    calculate_sample_statistics,
    canonical_sha256,
    collect_system_provenance,
    percentile,
    seal_artifact,
    verify_scientific_gate,
    verify_seal,
)


def test_canonical_sha256_stability():
    """Verify that canonical JSON serialization produces deterministic hashes regardless of key order."""
    d1 = {"model": "TaylorF2", "batch_size": 10, "precision": "float64"}
    d2 = {"precision": "float64", "model": "TaylorF2", "batch_size": 10}
    assert canonical_sha256(d1) == canonical_sha256(d2)


def test_artifact_sealing_and_tamper_detection():
    """Verify that seal_artifact seals payload and detect any tampering."""
    payload = {
        "benchmark": "test_run",
        "values": [1.2, 3.4, 5.6],
        "metadata": {"device": "cpu"},
    }
    sealed = seal_artifact(payload)
    assert "content_sha256" in sealed
    assert sealed["schema_version"] == 1
    assert verify_seal(sealed) is True

    # Tamper with payload
    tampered = copy.deepcopy(sealed)
    tampered["benchmark"] = "modified_run"
    assert verify_seal(tampered) is False

    tampered_val = copy.deepcopy(sealed)
    tampered_val["values"][0] = 99.9
    assert verify_seal(tampered_val) is False


def test_bootstrap_median_ci_and_percentiles():
    """Verify statistical bootstrap CI and percentile calculations."""
    samples = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
    p50 = percentile(samples, 0.5)
    assert p50 == pytest.approx(11.5, rel=1e-3)

    low_ci, high_ci = bootstrap_median_ci(samples, confidence=0.95, resamples=1000, seed=42)
    assert low_ci <= p50 <= high_ci
    assert low_ci >= min(samples)
    assert high_ci <= max(samples)

    stats = calculate_sample_statistics(samples, batch_size=10)
    assert stats["batch_size"] == 10
    assert stats["median_ms"] == pytest.approx(11.5)
    assert stats["wf_per_sec"] == pytest.approx(10 / (11.5 / 1000.0), rel=1e-3)
    assert "median_ci_95" in stats


def test_scientific_gate_pass_and_fail():
    """Verify that Gate 1 passes on matching waveforms and fails on perturbed waveforms."""
    n_bins = 1000
    h_ref = np.exp(1j * np.linspace(0, 100, n_bins))

    # Test exact match
    gate_pass = verify_scientific_gate(
        candidate_hp=h_ref,
        candidate_hc=h_ref,
        reference_hp=h_ref,
        reference_hc=h_ref,
    )
    assert gate_pass["passed"] is True
    assert gate_pass["max_l2_error"] < 1e-12
    assert gate_pass["max_mismatch"] < 1e-15

    # Test perturbed waveform failing gate
    h_pert = h_ref * 1.05  # 5% amplitude error
    gate_fail = verify_scientific_gate(
        candidate_hp=h_pert,
        candidate_hc=h_pert,
        reference_hp=h_ref,
        reference_hc=h_ref,
    )
    assert gate_fail["passed"] is False
    assert gate_fail["max_l2_error"] > 1e-3


def test_system_provenance_collection():
    """Verify that system provenance collects host, CPU, Python, and git metadata."""
    prov = collect_system_provenance()
    assert "host" in prov
    assert "cpu" in prov
    assert "gpu" in prov
    assert "python" in prov
    assert "packages" in prov
    assert "git" in prov

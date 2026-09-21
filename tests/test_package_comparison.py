"""The external-comparison gate must not bless invalid or misaligned results."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "compare_packages", Path(__file__).parents[1] / "scripts" / "compare_packages.py"
)
comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparison)


def reference():
    wave = np.exp(1j * np.linspace(0, 2, 32)) * 1e-22
    return np.broadcast_to(wave, (2, 3, 32)).copy()


def test_gate_checks_every_binary_and_polarization():
    ref = reference()
    assert comparison.compare_waveforms(ref, ref)["qualified"]
    candidate = ref.copy()
    candidate[1, 2] *= 1.01
    verdict = comparison.compare_waveforms(candidate, ref)
    assert not verdict["qualified"]
    assert verdict["max_relative_l2"] == pytest.approx(0.01)


def test_good_overlap_does_not_hide_phase_convention_error():
    ref = reference()
    verdict = comparison.compare_waveforms(ref * 1j, ref)
    assert verdict["max_mismatch"] < 1e-14
    assert verdict["max_relative_l2"] == pytest.approx(np.sqrt(2))
    assert not verdict["qualified"]


@pytest.mark.parametrize("kind", ["nan", "zero", "shape", "precision"])
def test_invalid_results_fail_closed(kind):
    ref = reference()
    candidate = ref.copy()
    if kind == "nan":
        candidate[0, 1, 3] = np.nan
    elif kind == "zero":
        candidate[0, 1] = 0
    elif kind == "precision":
        candidate = candidate.astype(np.complex64)
    else:
        candidate = candidate[:, :1]
    with pytest.raises(ValueError):
        comparison.compare_waveforms(candidate, ref)


def test_batches_use_distinct_reproducible_binaries():
    for model in comparison.MODELS:
        single = comparison.parameter_batch(model, 1)
        batch = comparison.parameter_batch(model, 10)
        np.testing.assert_array_equal(single, batch[:1])
        assert len(np.unique(batch, axis=0)) == 10


def test_latency_units_and_bootstrap_reproducibility():
    samples = [1.0, 2.0, 3.0] * 5
    result = comparison.statistics(samples, 10)
    assert result == comparison.statistics(samples, 10)
    assert result["median_ms"] == 2.0
    assert result["amortized_ms"] == 0.2
    assert result["waveforms_per_second"] == 5000.0

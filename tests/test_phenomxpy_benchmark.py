"""The published TD benchmark must reject misleading agreement results."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def benchmark():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("phenomxpy_benchmark", scripts / "compare_phenomxpy.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(scripts))


def test_gate_checks_amplitude_sign_and_each_binary(benchmark):
    reference = np.random.default_rng(12).normal(size=(2, 3, 32))
    assert benchmark.compare_polarizations(reference.copy(), reference)["qualified"]
    for factor in (1.001, -1.0):
        candidate = reference.copy()
        candidate[1, 2] *= factor
        result = benchmark.compare_polarizations(candidate, reference)
        assert not result["qualified"]
        assert result["relative_l2"][1][2] > benchmark.MAX_L2
    # The direct error cannot be hidden by unit normalized overlap.
    result = benchmark.compare_polarizations(2 * reference, reference)
    assert result["max_mismatch"] < 1e-14
    assert not result["qualified"]


@pytest.mark.parametrize("fault", ["nan", "zero", "float32", "shape"])
def test_gate_rejects_invalid_results(benchmark, fault):
    reference = np.ones((2, 3, 32), dtype=np.float64)
    candidate = reference.copy()
    if fault == "nan":
        candidate[0, 1, 2] = np.nan
    elif fault == "zero":
        candidate[1, 2] = 0
    elif fault == "float32":
        candidate = candidate.astype(np.float32)
    else:
        candidate = candidate[:, :, :-1]
    with pytest.raises(ValueError):
        benchmark.compare_polarizations(candidate, reference)

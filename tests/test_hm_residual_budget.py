"""Analytical checks of the non-gating complex-residual diagnostic."""

import numpy as np
import pytest

from scripts.diagnose_imrphenomhm_sxs_2158 import frequency_budget, residual_budget
from scripts.run_imrphenomhm_sxs_2158 import FREQUENCIES


@pytest.mark.parametrize("scale,phase", [(1.0, 0.0), (1.5, 0.0), (1.0, 0.4), (0.8, -0.7)])
def test_known_amplitude_and_phase(scale, phase):
    reference = np.array([0, 1 + 2j, -3j, 4])
    result = residual_budget(reference, scale * np.exp(1j * phase) * reference, np.array([2, 3, 5, 7]))
    assert result["amplitude_residual_power"] == pytest.approx((1 - scale) ** 2, abs=1e-14)
    assert result["phase_residual_power"] == pytest.approx(2 * scale * (1 - np.cos(phase)), abs=1e-14)
    assert result["match"] == pytest.approx(np.cos(phase))


def test_frequency_partition_and_coherent_cancellation():
    reference = np.ones_like(FREQUENCIES, dtype=complex)
    candidate = reference * (1 + 0.1 * np.exp(1j * FREQUENCIES))
    psd = 1 + FREQUENCIES ** 2
    rows = frequency_budget(reference, candidate, psd)
    assert sum(row["residual_power"] for row in rows) == pytest.approx(0.01)
    # Equal and opposite mode errors cancel: their powers cannot be added.
    first = residual_budget(reference, 1.2 * reference, psd)
    second = residual_budget(reference, 0.8 * reference, psd)
    combined = residual_budget(2 * reference, 1.2 * reference + 0.8 * reference, psd)
    assert first["residual_power"] + second["residual_power"] > 0
    assert combined["residual_power"] == 0


def test_zero_candidate_above_mode_cutoff():
    result = residual_budget(np.array([1j, 2]), np.zeros(2), np.ones(2))
    assert result["match"] is None
    assert result["norm_ratio"] == 0
    assert result["amplitude_residual_power"] == 1
    assert result["phase_residual_power"] == 0


@pytest.mark.parametrize("psd", [[1, 0], [1, -1], [1, np.nan]])
def test_invalid_psd_rejected(psd):
    with pytest.raises(ValueError):
        residual_budget(np.ones(2), np.ones(2), psd)

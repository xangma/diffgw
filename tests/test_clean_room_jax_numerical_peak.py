"""Cross-backend checks for the development-only full-series peak convention."""

import jax
import numpy as np
import pytest
import torch

from dev.native_rom.eob.jax.numerical_attachment_clean import full_orbital_peak
from torchwave.approximants.eob.numerical_attachment import _full_orbital_peak

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("values", [
    2.0 - (np.linspace(0, 6, 61) - 3.27) ** 2,
    np.array([0.1, 0.2, 0.3, 0.3, 0.5, 0.4, 0.2]),
    np.array([0.1, 0.3, 0.5, 0.7, 0.71, 0.7101, 0.7099, 0.6]),
])
def test_full_orbital_peak_matches_torch_sign_product_rule(values):
    time = np.arange(len(values), dtype=np.float64) * 0.1
    expected = _full_orbital_peak(
        torch.tensor(time, dtype=torch.float64),
        torch.tensor(values, dtype=torch.float64),
    )
    actual = full_orbital_peak(time, values)
    assert int(actual.sample_index) == int(expected.sample_index)
    np.testing.assert_allclose(float(actual.time), float(expected.time), rtol=0, atol=1e-5)
    np.testing.assert_allclose(float(actual.frequency), float(expected.frequency),
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(float(actual.curvature), float(expected.curvature),
                               rtol=1e-9, atol=1e-9)


def test_full_orbital_peak_requires_a_resolved_turn():
    with pytest.raises(ValueError, match="no non-increasing sample"):
        full_orbital_peak(np.arange(5.0), np.arange(5.0))

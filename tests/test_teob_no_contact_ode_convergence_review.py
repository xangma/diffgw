from __future__ import annotations

import numpy as np

from scripts.teob_no_contact_ode_convergence_review import mode_on_common_public_prefix, native_grid


def test_native_grid_does_not_compare_adaptive_knots_by_index() -> None:
    result = native_grid(np.array([0.0, 1.0]), np.array([0.0, 0.5, 1.0]))
    assert result["same"] is False
    assert result["max_abs_difference"] is None


def test_mode_comparison_uses_common_public_prefix() -> None:
    public = np.arange(5.0)
    result = mode_on_common_public_prefix(
        np.array([1 + 0j, 2 + 0j, 3 + 0j]),
        np.array([1 + 0j, 2 + 1e-3j, 3 + 0j, 4 + 0j]),
        public,
    )
    assert result["public_prefix_samples"] == 3
    assert result["same_public_subset"] is False
    assert result["metrics"]["max_abs"] == 1e-3

from __future__ import annotations

import numpy as np

from scripts.teob_no_contact_ode_convergence import LEVELS, array_metrics


def test_tolerance_levels_are_predeclared_monotone() -> None:
    assert list(LEVELS) == ["baseline", "tight_0p1", "tight_0p01"]
    assert LEVELS["tight_0p1"][0] == LEVELS["baseline"][0] * 0.1
    assert LEVELS["tight_0p01"][1] == LEVELS["baseline"][1] * 0.01


def test_array_metrics_has_no_alignment_or_fit() -> None:
    ref = np.array([1.0 + 0j, 2.0 + 0j])
    got = ref + np.array([0.0 + 1e-3j, 0.0 + 2e-3j])
    result = array_metrics(got, ref)
    assert result["samples"] == 2
    assert result["relative_l2"] > 0.0

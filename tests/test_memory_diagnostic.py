"""Visible known mismatch between the legacy memory carrier and published 22 memory."""

import numpy as np
import pytest
import torch

from diffgw_reference import memory_dominant_22 as ref
from torchwave.approximants import memory as tm


@pytest.mark.xfail(strict=True, reason="legacy 1/sqrt(6) flux sum omits Favata/Talbot angular coupling")
def test_legacy_mode_sum_is_not_the_published_22_coefficient():
    times = torch.tensor([0.0, 0.1, 0.4, 1.0], dtype=torch.float64)
    slope = 2e-20
    distance_m = 3.085677581491367e24
    mode = slope * times
    legacy = tm.compute_memory_h20(times, {(2, 2): mode, (2, -2): mode}, distance=distance_m)
    target = float(ref.h20_linear_mode(1, 0, slope, distance_m))
    np.testing.assert_allclose(legacy[-1].item(), target, rtol=5e-13, atol=0)

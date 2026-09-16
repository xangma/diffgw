"""Focused numeric regression for Appendix-E E11 (the published (3,3) mode)."""

import torch

from torchwave.approximants.xhm_amplitude_clean import hhat_lm


def test_e11_33_independent_checkpoint():
    value = hhat_lm((3, 3), 0.117, 0.21, 0.4, 0.13, 0.04)
    expected = torch.tensor(
        -0.00008587732332306313 - 0.02891412680205477j,
        dtype=torch.complex128,
    )
    torch.testing.assert_close(value, expected, rtol=3e-13, atol=3e-15)

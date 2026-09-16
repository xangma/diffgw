"""Focused numeric regressions for Appendix-E E12, E14, and E15."""

import torch

from torchwave.approximants.xhm_amplitude_clean import hhat_lm


def test_e12_32_e14_44_e15_43_independent_checkpoints():
    cases = (
        ((3, 2), 0.001468665524521052 + 0.0000014082471570887055j),
        ((4, 4), -0.002546541267466567 - 0.0000009903578405098715j),
        ((4, 3), -0.0000007932901031548655 - 0.00008061703201584492j),
    )
    for mode, expected in cases:
        value = hhat_lm(mode, 0.117, 0.21, 0.4, 0.13, 0.04)
        torch.testing.assert_close(
            value, torch.tensor(expected, dtype=torch.complex128), rtol=3e-13, atol=3e-15
        )

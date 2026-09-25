"""Analytic physical projection of paired quadrupole modes."""

# SPDX-License-Identifier: MIT


import torch

from torchwave.constants import C_SI, MPC_SI, MSUN_SEC, PI
from torchwave.approximants.eob.conditioned_waveform_clean import project_22_pair_to_polarizations


def test_projection_uses_both_modes_phase_and_physical_distance_scale():
    h22 = torch.tensor([1.0 + 2.0j, -0.5 + 0.25j], dtype=torch.complex128)
    h2m2 = torch.tensor([0.25 - 0.75j, 0.5 + 0.125j], dtype=torch.complex128)
    distance = torch.tensor(120.0, dtype=torch.float64)
    inclination = torch.tensor(0.4, dtype=torch.float64)
    phic = torch.tensor(0.2, dtype=torch.float64)
    hp, hc = project_22_pair_to_polarizations(
        h22,
        h2m2,
        mass1=30.0,
        mass2=20.0,
        distance=distance,
        inclination=inclination,
        phic=phic,
    )

    c = torch.cos(inclination)
    pair = (
        (1.0 + c) ** 2 * h22 * torch.exp(-2j * phic)
        + (1.0 - c) ** 2 * h2m2 * torch.exp(2j * phic)
    )
    scale = (30.0 + 20.0) * MSUN_SEC * C_SI / (distance * MPC_SI)
    expected = (torch.sqrt(torch.tensor(5.0 / (64.0 * PI))) * pair * scale)
    torch.testing.assert_close(hp, expected.real)
    torch.testing.assert_close(hc, -expected.imag)

"""Tests for TaylorF2 approximant."""

import torch
import pytest

from torchwave.approximants.taylorf2 import TaylorF2
from torchwave import get_fd_waveform


def test_taylorf2_phase_coefficients():
    # Test equal-mass non-spinning coefficients
    m1 = torch.tensor(1.4)
    m2 = torch.tensor(1.4)
    chi1 = torch.tensor(0.0)
    chi2 = torch.tensor(0.0)

    (
        phi0, phi2, phi3, phi4, phi5, phi5_log,
        phi6, phi6_log, phi7, phi10_t, phi12_t
    ) = TaylorF2.phase_coefficients(m1, m2, chi1, chi2)

    assert torch.isclose(phi0, torch.tensor(1.0))
    # 1PN: 3715/756 + 55/9 * (1/4) = 3715/756 + 55/36 = 4870/756 = 2435/378 ~ 6.4417989
    expected_phi2 = 3715.0 / 756.0 + (55.0 / 9.0) * 0.25
    assert torch.isclose(phi2, torch.tensor(expected_phi2))
    # 1.5PN: -16*pi
    assert torch.isclose(phi3, torch.tensor(-16.0 * torch.pi))
    # 3PN: Buonanno et al. 2009 / Arun et al. 2008
    from torchwave.constants import EULER_GAMMA, PI
    expected_phi6 = (
        (11583231236531.0 / 4694215680.0)
        - (6848.0 / 21.0) * EULER_GAMMA
        - (640.0 / 3.0) * (PI * PI)
        + ((-15737765635.0 / 3048192.0) + (2255.0 / 12.0) * (PI * PI)) * 0.25
        + (76055.0 / 1728.0) * (0.25**2)
        - (127825.0 / 1296.0) * (0.25**3)
    )
    assert torch.isclose(phi6, torch.tensor(expected_phi6, dtype=phi6.dtype))
    assert torch.isclose(phi6_log, torch.tensor(-6848.0 / 21.0, dtype=phi6.dtype))


def test_taylorf2_distance_scaling():
    f = torch.linspace(30.0, 300.0, 50)
    hp_100, _ = get_fd_waveform(
        "TaylorF2",
        mass1=1.4,
        mass2=1.4,
        sample_frequencies=f,
        distance=100.0,
    )
    hp_200, _ = get_fd_waveform(
        "TaylorF2",
        mass1=1.4,
        mass2=1.4,
        sample_frequencies=f,
        distance=200.0,
    )

    ratio = torch.abs(hp_100) / torch.abs(hp_200)
    assert torch.allclose(ratio, torch.tensor(2.0, dtype=torch.float64), atol=1e-10)


def test_taylorf2_tidal_dephasing():
    f = torch.linspace(40.0, 800.0, 100)
    # Point particle (no tides)
    hp_pp, _ = get_fd_waveform(
        "TaylorF2",
        mass1=1.4,
        mass2=1.4,
        sample_frequencies=f,
        lambda1=0.0,
        lambda2=0.0,
    )
    # Binary neutron star with strong tides
    hp_tidal, _ = get_fd_waveform(
        "TaylorF2",
        mass1=1.4,
        mass2=1.4,
        sample_frequencies=f,
        lambda1=800.0,
        lambda2=800.0,
    )

    # Tides accelerate inspiral, causing accumulated phase difference at high frequency
    import numpy as np
    phase_pp = np.unwrap(torch.angle(hp_pp).numpy())
    phase_tidal = np.unwrap(torch.angle(hp_tidal).numpy())
    delta_phase = abs(phase_tidal[-1] - phase_pp[-1])
    assert delta_phase > 0.1, "Tidal deformability must induce measurable phase shift"

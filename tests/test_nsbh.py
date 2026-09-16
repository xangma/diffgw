"""Unit tests for IMRPhenomNSBH approximant."""

import math
import pytest
import torch

from torchwave import get_fd_waveform
from torchwave.approximants.imrphenom_nsbh import (
    IMRPhenomNSBH,
    ns_compactness_from_lambda,
    nsbh_frequencies,
)


def test_ns_compactness_from_lambda():
    """Verify Yagi-Yunes compactness C(Lambda) relation and monotonicity."""
    lam_vals = torch.tensor([100.0, 400.0, 1000.0], dtype=torch.float64)
    compactness = ns_compactness_from_lambda(lam_vals)

    # For typical NS (Lambda ~ 400), compactness is ~0.16 - 0.18
    assert 0.15 < compactness[1].item() < 0.20

    # More deformable (larger Lambda) corresponds to larger radius and lower compactness
    assert compactness[0].item() > compactness[1].item() > compactness[2].item()

    # BH limit (Lambda -> 0)
    c_bh = ns_compactness_from_lambda(torch.tensor([0.0], dtype=torch.float64))
    assert c_bh.item() == 0.5


def test_nsbh_frequencies_disruption_regimes():
    """Verify disruption vs non-disruption frequency ordering."""
    # Case 1: High mass ratio, zero spin -> Non-disrupting plunge
    # (m1 = 15 Msun, m2 = 1.4 Msun, chi1 = 0, Lambda2 = 300)
    # The NS falls inside the horizon/ISCO before tidal disruption can occur
    m1_nondisrupt = torch.tensor(15.0, dtype=torch.float64)
    m2_nondisrupt = torch.tensor(1.4, dtype=torch.float64)
    chi1_nondisrupt = torch.tensor(0.0, dtype=torch.float64)
    lam2_nondisrupt = torch.tensor(300.0, dtype=torch.float64)

    f_tide_1, f_isco_1, f_rd_1, c_1 = nsbh_frequencies(
        m1_nondisrupt, m2_nondisrupt, chi1_nondisrupt, lam2_nondisrupt
    )
    # When non-disrupting, r_tide is inside ISCO, so f_tide > f_isco
    assert f_tide_1.item() > f_isco_1.item()

    # Case 2: Low mass ratio, high prograde spin -> Complete tidal disruption
    # (m1 = 3.0 Msun, m2 = 1.2 Msun, chi1 = 0.8, Lambda2 = 800)
    # Tidal forces shred the NS far outside the ISCO
    m1_disrupt = torch.tensor(3.0, dtype=torch.float64)
    m2_disrupt = torch.tensor(1.2, dtype=torch.float64)
    chi1_disrupt = torch.tensor(0.8, dtype=torch.float64)
    lam2_disrupt = torch.tensor(800.0, dtype=torch.float64)

    f_tide_2, f_isco_2, f_rd_2, c_2 = nsbh_frequencies(
        m1_disrupt, m2_disrupt, chi1_disrupt, lam2_disrupt
    )
    # When disrupting, r_tide is outside ISCO, so f_tide < f_isco
    assert f_tide_2.item() < f_isco_2.item()


def test_imrphenom_nsbh_disruption_cutoff():
    """Verify exponential suppression of high-frequency amplitude for disrupting systems."""
    # High frequency grid covering post-ISCO
    f = torch.linspace(40.0, 6000.0, 600, dtype=torch.float64)

    # Disrupting parameters
    m1, m2 = 3.0, 1.2
    chi1 = 0.75
    lam2 = 600.0

    model = IMRPhenomNSBH()
    hp_disrupt, hc_disrupt = model(
        f=f,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        lambda2=lam2,
    )

    f_tide, f_isco, _, _ = nsbh_frequencies(
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64),
        torch.tensor(chi1, dtype=torch.float64),
        torch.tensor(lam2, dtype=torch.float64),
    )

    # At frequencies well above f_tide, amplitude should be heavily attenuated
    high_f_mask = f > (f_tide.item() * 1.5)
    low_f_mask = (f > 100.0) & (f < f_tide.item() * 0.8)

    high_amp = torch.max(hp_disrupt.abs()[high_f_mask]).item()
    low_amp = torch.max(hp_disrupt.abs()[low_f_mask]).item()

    assert high_amp < 0.05 * low_amp


def test_imrphenom_nsbh_batching():
    """Verify batched evaluation across masses, spins, and tidal deformabilities."""
    B = 3
    N = 64
    f = torch.linspace(30.0, 400.0, N, dtype=torch.float64)

    m1_b = torch.tensor([5.0, 8.0, 4.0], dtype=torch.float64)
    m2_b = torch.tensor([1.4, 1.3, 1.5], dtype=torch.float64)
    chi1_b = torch.tensor([0.2, -0.1, 0.5], dtype=torch.float64)
    chi2_b = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    lam2_b = torch.tensor([350.0, 500.0, 200.0], dtype=torch.float64)
    inc_b = torch.tensor([0.0, 0.5, 0.8], dtype=torch.float64)

    model = IMRPhenomNSBH()
    hp_b, hc_b = model(
        f=f,
        mass1=m1_b,
        mass2=m2_b,
        chi1=chi1_b,
        chi2=chi2_b,
        lambda2=lam2_b,
        inclination=inc_b,
    )

    assert hp_b.shape == (B, N)
    assert hc_b.shape == (B, N)

    for b in range(B):
        hp_s, hc_s = model(
            f=f,
            mass1=m1_b[b],
            mass2=m2_b[b],
            chi1=chi1_b[b],
            chi2=chi2_b[b],
            lambda2=lam2_b[b],
            inclination=inc_b[b],
        )
        assert torch.allclose(hp_b[b], hp_s, atol=1e-25)
        assert torch.allclose(hc_b[b], hc_s, atol=1e-25)


def test_imrphenom_nsbh_autograd():
    """Verify finite non-zero gradients with respect to m1, m2, chi1, lambda2, and inclination."""
    f = torch.linspace(30.0, 300.0, 32, dtype=torch.float64)

    m1 = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(1.4, dtype=torch.float64, requires_grad=True)
    chi1 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    lam2 = torch.tensor(400.0, dtype=torch.float64, requires_grad=True)
    inc = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)

    hp, hc = IMRPhenomNSBH()(
        f=f,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        lambda2=lam2,
        inclination=inc,
    )

    loss = (hp.abs() ** 2 + hc.abs() ** 2).sum()
    loss.backward()

    for name, param in [("m1", m1), ("m2", m2), ("chi1", chi1), ("lam2", lam2), ("inc", inc)]:
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad), f"Gradient for {name} is NaN"
        assert param.grad.abs().item() > 0.0, f"Gradient for {name} is zero"


def test_nsbh_get_fd_waveform():
    """Verify waveform generation through get_fd_waveform entrypoint."""
    f = torch.linspace(30.0, 300.0, 50, dtype=torch.float64)

    hp1, hc1 = get_fd_waveform("IMRPhenomNSBH", mass1=6.0, mass2=1.4, chi1=0.2, lambda2=450.0, sample_frequencies=f)
    hp2, hc2 = get_fd_waveform("NSBH", mass1=6.0, mass2=1.4, chi1=0.2, lambda2=450.0, sample_frequencies=f)

    assert torch.allclose(hp1, hp2, atol=1e-25)
    assert torch.allclose(hc1, hc2, atol=1e-25)

"""Tests for IMRPhenomD approximant."""

import torch
import pytest

from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave import get_fd_waveform
from torchwave.constants import MSUN_SEC


def test_imrphenomd_remnant_fits():
    # SXS:BBH:0001 benchmark from Husa et al. 2016 Table I:
    # q=1, chi1=0, chi2=0 -> M_f ~ 0.9516, a_f ~ 0.6865, M*f_RD ~ 0.0881
    m1 = torch.tensor(20.0)
    m2 = torch.tensor(20.0)
    s1 = torch.tensor(0.0)
    s2 = torch.tensor(0.0)

    m_final, a_final, f_rd, f_damp = IMRPhenomD.remnant_properties(m1, m2, s1, s2)

    total_m = 40.0
    m_frac = (m_final / total_m).item()
    af_val = a_final.item()
    frd_val = f_rd.item()

    assert abs(m_frac - 0.9516) < 0.01, f"Expected M_f/M ~ 0.9516, got {m_frac}"
    assert abs(af_val - 0.6865) < 0.01, f"Expected a_f ~ 0.6865, got {af_val}"
    assert abs(frd_val - 0.0881) < 0.01, f"Expected M*f_RD ~ 0.0881, got {frd_val}"


def test_imrphenomd_face_on_polarizations():
    # For face-on binary (inclination = 0), |h_+| == |h_x|
    f = torch.linspace(20.0, 500.0, 100)
    hp, hc = get_fd_waveform(
        "IMRPhenomD",
        mass1=30.0,
        mass2=30.0,
        chi1=0.2,
        chi2=-0.1,
        sample_frequencies=f,
        inclination=0.0,
    )

    amp_plus = torch.abs(hp)
    amp_cross = torch.abs(hc)
    assert torch.allclose(amp_plus, amp_cross, atol=1e-25)


def test_imrphenomd_continuity_across_boundaries():
    # Sample tightly across the transition boundaries:
    # Inspiral/Intermediate phase join at M*f = 0.018
    # Intermediate/Ringdown phase join at M*f = 0.5 * f_RD
    m1 = torch.tensor(30.0)
    m2 = torch.tensor(30.0)
    total_sec = 60.0 * MSUN_SEC

    f_join1 = 0.018 / total_sec
    _, _, f_rd, _ = IMRPhenomD.remnant_properties(m1, m2, torch.tensor(0.0), torch.tensor(0.0))
    f_join2 = (0.5 * f_rd.item()) / total_sec

    # Evaluate just below and just above f_join1
    eps = 1e-4
    f_pts = torch.tensor([f_join1 - eps, f_join1 + eps, f_join2 - eps, f_join2 + eps], dtype=torch.float64)

    hp, _ = get_fd_waveform(
        "IMRPhenomD",
        mass1=30.0,
        mass2=30.0,
        sample_frequencies=f_pts,
    )

    # Verify no discontinuous jumps or NaNs at transition points
    assert not torch.isnan(hp).any()
    # Continuous amplitude across f_join1
    amp1_below = torch.abs(hp[0])
    amp1_above = torch.abs(hp[1])
    assert torch.isclose(amp1_below, amp1_above, rtol=1e-2)

    # Continuous amplitude across f_join2
    amp2_below = torch.abs(hp[2])
    amp2_above = torch.abs(hp[3])
    assert torch.isclose(amp2_below, amp2_above, rtol=1e-2)


def test_imrphenomd_taylorf2_early_inspiral_overlap():
    # At low frequencies (Mf << 0.01), both TaylorF2 and IMRPhenomD phase are governed by PN expansions
    m1 = 15.0
    m2 = 15.0
    total_sec = 30.0 * MSUN_SEC
    # M*f = 0.005 -> f ~ 33.8 Hz
    f_low = torch.linspace(20.0, 35.0, 30, dtype=torch.float64)

    hp_tf2, _ = get_fd_waveform("TaylorF2", mass1=m1, mass2=m2, sample_frequencies=f_low)
    hp_phd, _ = get_fd_waveform("IMRPhenomD", mass1=m1, mass2=m2, sample_frequencies=f_low)

    # Amplitudes should match closely at low frequency (accounting for 3.5PN amplitude corrections in IMRPhenomD)
    amp_tf2 = torch.abs(hp_tf2)
    amp_phd = torch.abs(hp_phd)
    rel_diff = torch.abs(amp_phd - amp_tf2) / amp_tf2
    assert torch.max(rel_diff).item() < 0.10, f"Expected <10% amplitude difference at early inspiral, got {torch.max(rel_diff).item()}"


def test_imrphenomd_corner_cases():
    # Corner case: q = 18, chi1 = 0.85, chi2 = -0.85
    f = torch.linspace(20.0, 500.0, 100, dtype=torch.float64)
    hp, hc = get_fd_waveform(
        "IMRPhenomD",
        mass1=90.0,
        mass2=5.0,  # q = 18
        chi1=0.85,
        chi2=-0.85,
        sample_frequencies=f,
    )
    assert not torch.isnan(hp).any()
    assert not torch.isnan(hc).any()
    assert (torch.abs(hp) > 0.0).any()


def test_uniform_grid_generation():
    # Test generation with f_lower, f_final, delta_f
    hp, hc = get_fd_waveform(
        "IMRPhenomD",
        mass1=30.0,
        mass2=30.0,
        f_lower=20.0,
        f_final=200.0,
        delta_f=1.0,
    )
    assert hp.shape == (181,)
    assert not torch.isnan(hp).any()

"""Tests for physical waveform cutoffs and smooth tapering windows."""

import pytest
import torch
import math

from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.waveforms import get_fd_waveform
from torchwave.constants import MSUN_SEC, PI


def test_imrphenomd_opt_in_planck_cutoff_zero_amplitude():
    """The optional Planck taper vanishes above (f_RD + 5*f_damp)/M_sec."""
    m1, m2 = 30.0, 20.0
    chi1, chi2 = 0.5, -0.3
    total_mass = m1 + m2
    m_sec = total_mass * MSUN_SEC

    # Get remnant ringdown and damping frequencies
    _, _, f_rd, f_damp = IMRPhenomD.remnant_properties(
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64),
        torch.tensor(chi1, dtype=torch.float64),
        torch.tensor(chi2, dtype=torch.float64),
    )
    f_cut = ((f_rd + 5.0 * f_damp) / m_sec).item()

    # Generate frequencies above f_cut
    f_above = torch.linspace(f_cut + 5.0, f_cut + 200.0, 50, dtype=torch.float64)
    hp, hc = get_fd_waveform(
        'IMRPhenomD',
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        sample_frequencies=f_above,
        window_type='planck',
    )

    # Strictly zero amplitude
    assert torch.all(hp == 0.0), 'IMRPhenomD hp is non-zero above f_cut'
    assert torch.all(hc == 0.0), 'IMRPhenomD hc is non-zero above f_cut'


def test_imrphenomd_opt_in_planck_rolloff_continuity():
    """The optional Planck taper rolls off smoothly in its chosen window."""
    m1, m2 = 40.0, 30.0
    chi1, chi2 = 0.2, 0.1
    total_mass = m1 + m2
    m_sec = total_mass * MSUN_SEC

    _, _, f_rd, f_damp = IMRPhenomD.remnant_properties(
        torch.tensor(m1, dtype=torch.float64),
        torch.tensor(m2, dtype=torch.float64),
        torch.tensor(chi1, dtype=torch.float64),
        torch.tensor(chi2, dtype=torch.float64),
    )
    f_cut = ((f_rd + 5.0 * f_damp) / m_sec).item()
    f_roll_start = ((f_rd + 3.0 * f_damp) / m_sec).item()

    # Sample densely through roll-off window
    f = torch.linspace(f_roll_start - 20.0, f_cut + 20.0, 500, dtype=torch.float64)
    hp, _ = get_fd_waveform(
        'IMRPhenomD',
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        sample_frequencies=f,
        window_type='planck',
    )
    amp = hp.abs()

    # Check that amplitude reaches exactly zero at f_cut
    idx_above_cut = f >= f_cut
    assert torch.all(amp[idx_above_cut] == 0.0)

    # Check that before f_cut amplitude is positive and smoothly decreasing
    idx_roll = (f >= f_roll_start) & (f < f_cut)
    roll_amp = amp[idx_roll]
    assert torch.all(roll_amp > 0.0)

    # Check finite differences (slopes) have no explosive spikes
    damp_df = torch.diff(roll_amp) / torch.diff(f[idx_roll])
    assert torch.all(torch.isfinite(damp_df))
    assert torch.all(damp_df <= 0.0), 'Amplitude should be monotonically non-increasing in window'


def test_taylorf2_smooth_isco_cutoff():
    """Verify TaylorF2 smooth_cutoff rolls amplitude smoothly to zero at ISCO."""
    m1, m2 = 1.4, 1.4
    total_mass = m1 + m2
    m_sec = total_mass * MSUN_SEC
    # Schwarzschild ISCO frequency: f_ISCO = 1 / (6^(3/2) * pi * M_sec)
    f_isco = 1.0 / ((6.0 ** 1.5) * PI * m_sec)

    f = torch.linspace(f_isco * 0.8, f_isco * 1.2, 200, dtype=torch.float64)

    # Without smooth cutoff: sharp step
    hp_sharp, _ = get_fd_waveform('TaylorF2', mass1=m1, mass2=m2, sample_frequencies=f, smooth_cutoff=False)
    # With smooth cutoff: smooth cosine tapering
    hp_smooth, _ = get_fd_waveform('TaylorF2', mass1=m1, mass2=m2, sample_frequencies=f, smooth_cutoff=True, roll_off_delta=0.1)

    amp_sharp = hp_sharp.abs()
    amp_smooth = hp_smooth.abs()

    # Both must be zero above ISCO
    idx_above = f > f_isco
    assert torch.all(amp_sharp[idx_above] == 0.0)
    assert torch.all(amp_smooth[idx_above] == 0.0)

    # In the window [0.9*f_isco, f_isco], amp_smooth < amp_sharp and monotonically decreasing
    idx_window = (f >= 0.9 * f_isco) & (f <= f_isco)
    assert torch.all(amp_smooth[idx_window] <= amp_sharp[idx_window])
    d_smooth = torch.diff(amp_smooth[idx_window])
    assert torch.all(d_smooth <= 1e-30)


def test_windowing_autograd_differentiability():
    """Verify gradients propagate cleanly through smooth windowing without NaN/inf."""
    m1 = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    c1 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)

    f = torch.linspace(30.0, 500.0, 100, dtype=torch.float64)
    hp, hc = get_fd_waveform('IMRPhenomD', mass1=m1, mass2=m2, chi1=c1, sample_frequencies=f, window_type='planck')

    loss = hp.abs().sum() + hc.abs().sum()
    loss.backward()

    for p, name in [(m1, 'm1'), (m2, 'm2'), (c1, 'c1')]:
        assert p.grad is not None, f'Gradient for {name} is None'
        assert torch.isfinite(p.grad), f'Gradient for {name} is not finite: {p.grad}'
        assert p.grad.item() != 0.0, f'Gradient for {name} is zero'

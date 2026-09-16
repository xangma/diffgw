"""Tests for time-domain waveform generation and conditioning (get_td_waveform).

Literature citations:
- Allen et al. (2012), Phys. Rev. D 85, 122006 (arXiv:gr-qc/0509116)
- McKechan, Robinson, Sathyaprakash (2010), Class. Quantum Grav. 27, 084020 (arXiv:1003.2939)
"""

import math
import pytest
import torch

from torchwave.waveforms import get_td_waveform, get_fd_waveform


def test_td_waveform_generation():
    """Verify basic get_td_waveform generation returning real tensors."""
    hp, hc = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=30.0,
        mass2=25.0,
        chi1=0.2,
        chi2=-0.1,
        delta_t=1.0 / 2048.0,
        f_lower=20.0,
        distance=100.0,
    )

    assert hp.is_floating_point() and not hp.is_complex()
    assert hc.is_floating_point() and not hc.is_complex()
    assert hp.shape == hc.shape
    assert hp.dim() == 1
    assert torch.all(torch.isfinite(hp))
    assert torch.all(torch.isfinite(hc))
    assert torch.any(hp.abs() > 0.0)


def test_td_waveform_parseval_energy_conservation():
    """Verify exact Parseval theorem energy conservation between TD and FD:
    integral |h(t)|^2 dt == 2 * integral_{f>0} |h~(f)|^2 df
    """
    delta_t = 1.0 / 4096.0
    f_lower = 25.0
    m1, m2 = 35.0, 30.0

    hp_td, hc_td = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        delta_t=delta_t,
        f_lower=f_lower,
        distance=100.0,
    )

    # Compute energy in time domain: sum |h(t)|^2 * delta_t
    energy_td_plus = torch.sum(hp_td.double() ** 2.0) * delta_t
    energy_td_cross = torch.sum(hc_td.double() ** 2.0) * delta_t

    # Compute energy in frequency domain via rfft of hp_td
    # Discrete rfft gives continuous FD strain H(f) = rfft(h(t)) * delta_t
    h_fd_plus = torch.fft.rfft(hp_td) * delta_t
    h_fd_cross = torch.fft.rfft(hc_td) * delta_t
    n_samples = hp_td.shape[-1]
    delta_f = 1.0 / (n_samples * delta_t)

    # Frequency domain energy: 2 * sum_{k=1}^{N/2-1} |H(f)|^2 * df + |H(0)|^2*df + |H(Nyq)|^2*df
    energy_fd_plus = (
        2.0 * torch.sum(torch.abs(h_fd_plus[1:-1]) ** 2.0)
        + torch.abs(h_fd_plus[0]) ** 2.0
        + torch.abs(h_fd_plus[-1]) ** 2.0
    ) * delta_f

    energy_fd_cross = (
        2.0 * torch.sum(torch.abs(h_fd_cross[1:-1]) ** 2.0)
        + torch.abs(h_fd_cross[0]) ** 2.0
        + torch.abs(h_fd_cross[-1]) ** 2.0
    ) * delta_f

    assert torch.isclose(energy_td_plus, energy_fd_plus, rtol=1e-5)
    assert torch.isclose(energy_td_cross, energy_fd_cross, rtol=1e-5)


def test_td_waveform_coalescence_centering():
    """Verify waveform peak is positioned near the end of the time series (coalescence centering)."""
    delta_t = 1.0 / 2048.0
    hp, hc = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=30.0,
        mass2=30.0,
        delta_t=delta_t,
        f_lower=25.0,
        distance=100.0,
    )

    peak_idx = torch.argmax(hp.abs()).item()
    n_total = hp.shape[-1]

    # Peak should occur in the last 10% of the buffer (near T - t_pad)
    assert peak_idx > int(0.70 * n_total), f"Peak at index {peak_idx} out of {n_total}"


def test_td_waveform_batching_parity():
    """Verify batched get_td_waveform matches unbatched evaluation."""
    m1 = torch.tensor([30.0, 40.0], dtype=torch.float64)
    m2 = torch.tensor([25.0, 20.0], dtype=torch.float64)
    c1 = torch.tensor([0.1, -0.2], dtype=torch.float64)
    c2 = torch.tensor([-0.1, 0.1], dtype=torch.float64)

    hp_b, hc_b = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        chi1=c1,
        chi2=c2,
        delta_t=1.0 / 2048.0,
        f_lower=25.0,
        distance=100.0,
    )

    assert hp_b.dim() == 2
    assert hp_b.shape[0] == 2

    for i in range(2):
        hp_s, hc_s = get_td_waveform(
            approximant="IMRPhenomD",
            mass1=m1[i],
            mass2=m2[i],
            chi1=c1[i],
            chi2=c2[i],
            delta_t=1.0 / 2048.0,
            f_lower=25.0,
            distance=100.0,
        )
        assert torch.allclose(hp_b[i], hp_s, rtol=1e-8, atol=1e-25)
        assert torch.allclose(hc_b[i], hc_s, rtol=1e-8, atol=1e-25)


def test_td_waveform_autograd():
    """Verify autograd produces finite non-zero gradients through get_td_waveform."""
    m1 = torch.tensor(35.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(25.0, dtype=torch.float64, requires_grad=True)
    c1 = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    c2 = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True)

    hp, hc = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        chi1=c1,
        chi2=c2,
        delta_t=1.0 / 1024.0,
        f_lower=30.0,
        distance=100.0,
    )

    loss = (hp ** 2.0).sum()
    loss.backward()

    for p, name in [(m1, "m1"), (m2, "m2"), (c1, "c1"), (c2, "c2")]:
        assert p.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(p.grad), f"Gradient for {name} is not finite: {p.grad}"
        assert p.grad.item() != 0.0, f"Gradient for {name} is zero"

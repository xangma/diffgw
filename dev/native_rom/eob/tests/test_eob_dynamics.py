"""Tests for Effective-One-Body (EOB) dynamics and ODE trajectory integration.

Clean-room literature benchmarks:
- Buonanno & Damour (1999), Phys. Rev. D 59, 084006 (arXiv:gr-qc/9811091)
- Bohé et al. (2017), Phys. Rev. D 95, 044028 (arXiv:1611.03703)
"""

import torch

from dev.native_rom.eob.dispatch import torch_get_fd_waveform as get_fd_waveform


def test_seobnr_ode_face_on_polarizations():
    """At face-on (iota = 0), h_plus and h_cross have equal amplitudes."""
    f = torch.linspace(25.0, 150.0, 50, dtype=torch.float64)
    hp, hc = get_fd_waveform(
        approximant="SEOBNR_ODE",
        mass1=36.0,
        mass2=29.0,
        chi1=0.0,
        chi2=0.0,
        sample_frequencies=f,
        distance=410.0,
        inclination=0.0,
        dt_geom=1.0,
        num_steps=150,
    )

    assert hp.shape == f.shape
    assert hc.shape == f.shape
    assert torch.all(torch.isfinite(hp))
    assert torch.all(torch.isfinite(hc))

    # At iota = 0, |h+| == |hx|
    assert torch.allclose(hp.abs(), hc.abs(), rtol=1e-5)


def test_seobnr_ode_distance_scaling():
    """Waveform amplitude scales inversely with luminosity distance 1/d_L."""
    f = torch.linspace(30.0, 100.0, 30, dtype=torch.float64)
    hp1, _ = get_fd_waveform(
        approximant="SEOBNR_ODE",
        mass1=30.0,
        mass2=30.0,
        sample_frequencies=f,
        distance=100.0,
        dt_geom=1.0,
        num_steps=100,
    )
    hp2, _ = get_fd_waveform(
        approximant="SEOBNR_ODE",
        mass1=30.0,
        mass2=30.0,
        sample_frequencies=f,
        distance=200.0,
        dt_geom=1.0,
        num_steps=100,
    )

    ratio = hp1.abs() / hp2.abs()
    valid = hp2.abs() > 1e-26
    assert torch.allclose(ratio[valid], torch.tensor(2.0, dtype=torch.float64), rtol=1e-4)


def test_seobnr_ode_autograd_differentiability():
    """EOB trajectory integration is end-to-end differentiable via PyTorch autograd."""
    m1 = torch.tensor(36.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(29.0, dtype=torch.float64, requires_grad=True)
    f = torch.linspace(25.0, 100.0, 20, dtype=torch.float64)

    hp, _ = get_fd_waveform(
        approximant="SEOBNR_ODE",
        mass1=m1,
        mass2=m2,
        sample_frequencies=f,
        distance=410.0,
        dt_geom=1.0,
        num_steps=100,
    )

    loss = hp.abs().sum()
    loss.backward()

    assert m1.grad is not None and torch.isfinite(m1.grad)
    assert m2.grad is not None and torch.isfinite(m2.grad)

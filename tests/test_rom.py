"""Tests for SEOBNR Reduced-Order Model (ROM) Surrogate.

Clean-room literature benchmarks:
- Pürrer (2014), Class. Quantum Grav. 31, 195010 (arXiv:1402.4146)
- Pürrer (2016), Phys. Rev. D 93, 064041 (arXiv:1512.02248)
- Bohé et al. (2017), Phys. Rev. D 95, 044028 (arXiv:1611.03703)
"""

import pytest
import torch

@pytest.fixture(autouse=True)
def official_rom_data(monkeypatch, official_rom_path):
    """The canonical dispatcher requires the external official release."""
    monkeypatch.setenv("TORCHWAVE_DATA_PATH", str(official_rom_path.parent))

from torchwave.waveforms import get_fd_waveform


def test_rom_waveform_generation():
    """Verify ROM evaluates correctly and returns finite polarizations."""
    f = torch.linspace(20.0, 300.0, 100, dtype=torch.float64)
    hp, hc = get_fd_waveform(
        approximant="SEOBNR_ROM",
        mass1=36.0,
        mass2=29.0,
        chi1=0.2,
        chi2=-0.1,
        sample_frequencies=f,
        distance=410.0,
        inclination=0.0,
    )

    assert hp.shape == f.shape
    assert hc.shape == f.shape
    assert torch.all(torch.isfinite(hp))
    assert torch.all(torch.isfinite(hc))

    # At face-on, |h+| == |hx|
    assert torch.allclose(hp.abs(), hc.abs(), rtol=1e-5)


def test_rom_distance_scaling():
    """Verify exact 1/d_L scaling in ROM amplitude."""
    f = torch.linspace(20.0, 200.0, 50, dtype=torch.float64)
    hp1, _ = get_fd_waveform(
        approximant="SEOBNR_ROM",
        mass1=30.0,
        mass2=25.0,
        sample_frequencies=f,
        distance=100.0,
    )
    hp2, _ = get_fd_waveform(
        approximant="SEOBNR_ROM",
        mass1=30.0,
        mass2=25.0,
        sample_frequencies=f,
        distance=300.0,
    )

    ratio = hp1.abs() / hp2.abs()
    assert torch.allclose(ratio, torch.tensor(3.0, dtype=torch.float64), rtol=1e-5)


def test_rom_batching_parity():
    """Verify batched evaluation exactly matches unbatched evaluation."""
    f = torch.linspace(20.0, 250.0, 60, dtype=torch.float64)
    m1_list = [36.0, 45.0, 20.0]
    m2_list = [29.0, 30.0, 15.0]
    c1_list = [0.1, -0.2, 0.4]
    c2_list = [-0.05, 0.1, -0.3]

    # Batched evaluation
    m1_b = torch.tensor(m1_list, dtype=torch.float64)
    m2_b = torch.tensor(m2_list, dtype=torch.float64)
    c1_b = torch.tensor(c1_list, dtype=torch.float64)
    c2_b = torch.tensor(c2_list, dtype=torch.float64)

    hp_batched, hc_batched = get_fd_waveform(
        approximant="SEOBNR_ROM",
        mass1=m1_b,
        mass2=m2_b,
        chi1=c1_b,
        chi2=c2_b,
        sample_frequencies=f,
        distance=400.0,
    )

    assert hp_batched.shape == (len(m1_list), len(f))

    # Compare each against individual evaluation
    for i in range(len(m1_list)):
        hp_single, hc_single = get_fd_waveform(
            approximant="SEOBNR_ROM",
            mass1=m1_list[i],
            mass2=m2_list[i],
            chi1=c1_list[i],
            chi2=c2_list[i],
            sample_frequencies=f,
            distance=400.0,
        )
        assert torch.allclose(hp_batched[i], hp_single, rtol=1e-7, atol=1e-28)
        assert torch.allclose(hc_batched[i], hc_single, rtol=1e-7, atol=1e-28)


def test_rom_autograd_differentiability():
    """Verify analytical gradients exist and are finite for all input parameters."""
    m1 = torch.tensor(36.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(29.0, dtype=torch.float64, requires_grad=True)
    c1 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    c2 = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True)

    f = torch.linspace(20.0, 200.0, 50, dtype=torch.float64)

    hp, hc = get_fd_waveform(
        approximant="SEOBNR_ROM",
        mass1=m1,
        mass2=m2,
        chi1=c1,
        chi2=c2,
        sample_frequencies=f,
        distance=410.0,
    )

    loss = hp.abs().sum() + hc.abs().sum()
    loss.backward()

    for p, name in [(m1, "m1"), (m2, "m2"), (c1, "c1"), (c2, "c2")]:
        assert p.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(p.grad), f"Gradient for {name} is not finite"
        assert p.grad.item() != 0.0, f"Gradient for {name} is zero"

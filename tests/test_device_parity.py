"""Tests for device parity across CPU, CUDA, and Apple Silicon MPS."""

import torch
import pytest

from torchwave import get_fd_waveform


def test_mps_device_parity():
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Silicon MPS is not available on this machine")

    f_cpu = torch.linspace(20.0, 500.0, 128, dtype=torch.float32, device="cpu")
    f_mps = torch.linspace(20.0, 500.0, 128, dtype=torch.float32, device="mps")

    # TaylorF2
    hp_cpu_tf2, _ = get_fd_waveform(
        "TaylorF2",
        mass1=1.4,
        mass2=1.4,
        sample_frequencies=f_cpu,
        device="cpu",
        dtype=torch.float32,
    )
    hp_mps_tf2, _ = get_fd_waveform(
        "TaylorF2",
        mass1=1.4,
        mass2=1.4,
        sample_frequencies=f_mps,
        device="mps",
        dtype=torch.float32,
    )
    # Compare real and imaginary parts to avoid float32 squared underflow (10^-23)^2 = 10^-46
    assert torch.allclose(hp_cpu_tf2.real, hp_mps_tf2.cpu().real, rtol=1e-4, atol=2e-25)
    assert torch.allclose(hp_cpu_tf2.imag, hp_mps_tf2.cpu().imag, rtol=1e-4, atol=2e-25)

    # IMRPhenomD
    hp_cpu_phd, _ = get_fd_waveform(
        "IMRPhenomD",
        mass1=30.0,
        mass2=30.0,
        chi1=0.2,
        chi2=-0.1,
        sample_frequencies=f_cpu,
        device="cpu",
        dtype=torch.float32,
    )
    hp_mps_phd, _ = get_fd_waveform(
        "IMRPhenomD",
        mass1=30.0,
        mass2=30.0,
        chi1=0.2,
        chi2=-0.1,
        sample_frequencies=f_mps,
        device="mps",
        dtype=torch.float32,
    )
    assert torch.allclose(hp_cpu_phd.real, hp_mps_phd.cpu().real, rtol=1e-4, atol=1e-25)
    assert torch.allclose(hp_cpu_phd.imag, hp_mps_phd.cpu().imag, rtol=1e-4, atol=1e-25)

    # IMRPhenomP (Precessing)
    hp_cpu_php, _ = get_fd_waveform(
        "IMRPhenomP",
        mass1=25.0,
        mass2=15.0,
        spin1x=0.2,
        spin1y=0.1,
        spin1z=0.1,
        sample_frequencies=f_cpu,
        device="cpu",
        dtype=torch.float32,
        f_ref=30.0,
    )
    hp_mps_php, _ = get_fd_waveform(
        "IMRPhenomP",
        mass1=25.0,
        mass2=15.0,
        spin1x=0.2,
        spin1y=0.1,
        spin1z=0.1,
        sample_frequencies=f_mps,
        device="mps",
        dtype=torch.float32,
        f_ref=30.0,
    )
    assert torch.allclose(hp_cpu_php.real, hp_mps_php.cpu().real, rtol=1e-4, atol=1e-25)
    assert torch.allclose(hp_cpu_php.imag, hp_mps_php.cpu().imag, rtol=1e-4, atol=1e-25)

    # SEOBNR_ROM (Surrogate)
    hp_cpu_rom, _ = get_fd_waveform(
        "SEOBNR_ROM",
        mass1=35.0,
        mass2=25.0,
        chi1=0.2,
        chi2=-0.1,
        sample_frequencies=f_cpu,
        device="cpu",
        dtype=torch.float32,
        f_ref=30.0,
    )
    hp_mps_rom, _ = get_fd_waveform(
        "SEOBNR_ROM",
        mass1=35.0,
        mass2=25.0,
        chi1=0.2,
        chi2=-0.1,
        sample_frequencies=f_mps,
        device="mps",
        dtype=torch.float32,
        f_ref=30.0,
    )
    assert torch.allclose(hp_cpu_rom.real, hp_mps_rom.cpu().real, rtol=1e-4, atol=1e-25)
    assert torch.allclose(hp_cpu_rom.imag, hp_mps_rom.cpu().imag, rtol=1e-4, atol=1e-25)

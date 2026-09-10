"""Tests for batching consistency and automatic differentiation (autograd)."""

import torch
import pytest

from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.taylorf2 import TaylorF2
from torchwave import get_fd_waveform


def test_batching_parity_taylorf2():
    f = torch.linspace(25.0, 400.0, 64, dtype=torch.float64)

    m1_batch = torch.tensor([1.4, 1.6, 2.0], dtype=torch.float64)
    m2_batch = torch.tensor([1.2, 1.4, 1.5], dtype=torch.float64)
    s1_batch = torch.tensor([0.05, -0.1, 0.2], dtype=torch.float64)
    s2_batch = torch.tensor([-0.02, 0.05, -0.15], dtype=torch.float64)

    hp_batch, hc_batch = TaylorF2()(
        f,
        mass1=m1_batch,
        mass2=m2_batch,
        chi1=s1_batch,
        chi2=s2_batch,
    )

    assert hp_batch.shape == (3, 64)
    assert hc_batch.shape == (3, 64)

    # Compare each batch row with single evaluation
    for i in range(3):
        hp_single, hc_single = TaylorF2()(
            f,
            mass1=m1_batch[i].item(),
            mass2=m2_batch[i].item(),
            chi1=s1_batch[i].item(),
            chi2=s2_batch[i].item(),
        )
        assert torch.allclose(hp_batch[i], hp_single, atol=1e-15, rtol=1e-10)
        assert torch.allclose(hc_batch[i], hc_single, atol=1e-15, rtol=1e-10)


def test_batching_parity_imrphenomd():
    f = torch.linspace(20.0, 400.0, 80, dtype=torch.float64)

    m1_batch = torch.tensor([25.0, 35.0, 45.0], dtype=torch.float64)
    m2_batch = torch.tensor([20.0, 25.0, 30.0], dtype=torch.float64)
    s1_batch = torch.tensor([0.2, -0.3, 0.5], dtype=torch.float64)
    s2_batch = torch.tensor([-0.1, 0.4, -0.2], dtype=torch.float64)

    hp_batch, hc_batch = IMRPhenomD()(
        f,
        mass1=m1_batch,
        mass2=m2_batch,
        chi1=s1_batch,
        chi2=s2_batch,
    )

    assert hp_batch.shape == (3, 80)
    assert hc_batch.shape == (3, 80)

    for i in range(3):
        hp_single, hc_single = IMRPhenomD()(
            f,
            mass1=m1_batch[i].item(),
            mass2=m2_batch[i].item(),
            chi1=s1_batch[i].item(),
            chi2=s2_batch[i].item(),
        )
        assert torch.allclose(hp_batch[i], hp_single, atol=1e-15, rtol=1e-10)
        assert torch.allclose(hc_batch[i], hc_single, atol=1e-15, rtol=1e-10)


def test_autograd_differentiability_imrphenomd():
    f = torch.linspace(20.0, 300.0, 50, dtype=torch.float64)

    m1 = torch.tensor([30.0], dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor([25.0], dtype=torch.float64, requires_grad=True)
    chi1 = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    chi2 = torch.tensor([-0.1], dtype=torch.float64, requires_grad=True)
    dist = torch.tensor([400.0], dtype=torch.float64, requires_grad=True)
    inc = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

    hp, hc = IMRPhenomD()(
        f,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        distance=dist,
        inclination=inc,
    )

    # Define a scalar loss (e.g. power spectral energy)
    loss = (torch.real(hp * torch.conj(hp)) + torch.real(hc * torch.conj(hc))).sum()
    loss.backward()

    for p_name, param in [
        ("m1", m1),
        ("m2", m2),
        ("chi1", chi1),
        ("chi2", chi2),
        ("distance", dist),
        ("inclination", inc),
    ]:
        assert param.grad is not None, f"Gradient for {p_name} is None"
        assert not torch.isnan(param.grad).any(), f"Gradient for {p_name} contains NaN"
        assert param.grad.abs().item() > 0.0, f"Gradient for {p_name} is zero"


def test_autograd_differentiability_taylorf2():
    f = torch.linspace(30.0, 300.0, 40, dtype=torch.float64)

    m1 = torch.tensor([1.4], dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
    chi1 = torch.tensor([0.05], dtype=torch.float64, requires_grad=True)
    chi2 = torch.tensor([-0.02], dtype=torch.float64, requires_grad=True)
    l1 = torch.tensor([300.0], dtype=torch.float64, requires_grad=True)
    l2 = torch.tensor([250.0], dtype=torch.float64, requires_grad=True)

    hp, hc = TaylorF2()(
        f,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        lambda1=l1,
        lambda2=l2,
    )

    loss = (torch.real(hp * torch.conj(hp))).sum()
    loss.backward()

    for p_name, param in [
        ("m1", m1),
        ("m2", m2),
    ]:
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
        assert param.grad.abs().item() > 0.0


def test_autograd_with_fref():
    """Verify gradients propagate with f_ref specified across all frequency-domain models."""
    f = torch.linspace(25.0, 200.0, 40, dtype=torch.float64)

    for approx in ("TaylorF2", "IMRPhenomD", "IMRPhenomP", "SEOBNR_ROM"):
        m1 = torch.tensor(30.0 if "Phenom" in approx or "ROM" in approx else 1.5, dtype=torch.float64, requires_grad=True)
        m2 = torch.tensor(20.0 if "Phenom" in approx or "ROM" in approx else 1.3, dtype=torch.float64, requires_grad=True)
        c1 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        c2 = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True)
        t_c = torch.tensor(0.01, dtype=torch.float64, requires_grad=True)
        phi_c = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

        hp, hc = get_fd_waveform(
            approx,
            mass1=m1,
            mass2=m2,
            chi1=c1,
            chi2=c2,
            tc=t_c,
            phic=phi_c,
            sample_frequencies=f,
            f_ref=30.0,
        )

        loss = (hp.abs() ** 2 + hc.abs() ** 2 + (hp * torch.conj(hc)).real).sum()
        loss.backward()

        for p_name, param in [("m1", m1), ("m2", m2), ("c1", c1), ("c2", c2), ("tc", t_c), ("phic", phi_c)]:
            assert param.grad is not None, f"{approx}: Gradient for {p_name} is None"
            assert torch.isfinite(param.grad), f"{approx}: Gradient for {p_name} is not finite: {param.grad}"
            assert param.grad.item() != 0.0, f"{approx}: Gradient for {p_name} is zero"



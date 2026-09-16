"""Acceptance and reconstruction unit tests for clean-room IMRPhenomXHM."""

import math
import pytest
import torch

from torchwave.approximants.imrphenom_xhm import IMRPhenomXHM, _pn_amplitude_ratio
from torchwave.waveforms import get_fd_waveform


def test_supported_modes():
    """Verify supported modes include (2,2), (2,1), (3,3), (3,2), (4,4)."""
    model = IMRPhenomXHM()
    expected_modes = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))
    assert model.supported_modes == expected_modes
    assert model.modes == expected_modes


def test_forward_pass_finite_strain():
    """Verify forward pass produces finite non-zero strain across frequency grid."""
    model = IMRPhenomXHM()
    f = torch.linspace(20.0, 1024.0, 500)
    m1 = torch.tensor(35.0)
    m2 = torch.tensor(20.0)
    chi1 = torch.tensor(0.4)
    chi2 = torch.tensor(-0.3)

    hp, hc = model(f, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, inclination=0.6)

    assert hp.shape == f.shape
    assert hc.shape == f.shape
    assert torch.isfinite(hp).all()
    assert torch.isfinite(hc).all()
    assert hp.abs().max() > 0.0
    assert hc.abs().max() > 0.0
    # GW strain amplitude sanity check (astrophysical range ~ 1e-24 to 1e-21)
    assert hp.abs().max() < 1e-19
    assert hc.abs().max() < 1e-19


def test_individual_mode_reconstruction():
    """Verify each individual mode can be isolated and reconstructed with physical amplitudes."""
    model = IMRPhenomXHM()
    f = torch.linspace(20.0, 800.0, 400)
    m1 = torch.tensor(40.0)
    m2 = torch.tensor(15.0)
    chi1 = torch.tensor(0.5)
    chi2 = torch.tensor(-0.2)

    hp, hc, modes_dict = model(
        f,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        inclination=0.7,
        return_modes=True,
    )

    for lm in ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4)):
        assert lm in modes_dict
        mode_hp, mode_hc = modes_dict[lm]
        assert torch.isfinite(mode_hp).all()
        assert torch.isfinite(mode_hc).all()
        amp_max = mode_hp.abs().max().item()
        assert amp_max > 0.0, f"Mode {lm} should have non-zero strain"
        assert amp_max < 1e-19, f"Mode {lm} amplitude {amp_max} exceeded physical strain range"


def test_equal_mass_symmetry_and_unequal_spin_non_vanishing():
    """Verify odd-m modes vanish for equal mass/equal spin, but persist for unequal spin."""
    model = IMRPhenomXHM()
    f = torch.linspace(25.0, 600.0, 250)

    # Case A: Equal-mass, equal-spin (m1=m2, chi1=chi2) -> (2,1) and (3,3) vanish identically
    _, _, unproj_eq = model(
        f,
        mass1=30.0,
        mass2=30.0,
        chi1=0.4,
        chi2=0.4,
        return_unprojected_modes=True,
    )
    assert unproj_eq[(2, 1)].abs().max().item() == 0.0
    assert unproj_eq[(3, 3)].abs().max().item() == 0.0
    assert unproj_eq[(2, 2)].abs().max().item() > 0.0

    # Case B: Equal-mass, unequal-spin (m1=m2, chi1=0.6, chi2=-0.6) -> (2,1) and (3,3) are strictly non-zero
    _, _, unproj_uneq = model(
        f,
        mass1=30.0,
        mass2=30.0,
        chi1=0.6,
        chi2=-0.6,
        return_unprojected_modes=True,
    )
    h21_amp = unproj_uneq[(2, 1)].abs().max().item()
    h33_amp = unproj_uneq[(3, 3)].abs().max().item()
    assert h21_amp > 0.0, "(2,1) mode must be non-zero for equal-mass unequal-spin binaries"
    assert h33_amp > 0.0, "(3,3) mode must be non-zero for equal-mass unequal-spin binaries"
    assert h21_amp < 1e-20
    assert h33_amp < 1e-20


def test_mode_32_mixing():
    """Verify mode (3,2) executes spherical-spheroidal mode mixing."""
    model = IMRPhenomXHM(modes=((3, 2),))
    f = torch.linspace(20.0, 800.0, 400)
    m1 = torch.tensor(35.0)
    m2 = torch.tensor(20.0)
    chi1 = torch.tensor(0.5)
    chi2 = torch.tensor(-0.3)

    hp, hc, unproj = model(
        f,
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        return_unprojected_modes=True,
    )
    h32 = unproj[(3, 2)]
    assert torch.isfinite(h32).all()
    assert h32.abs().max().item() > 0.0
    assert h32.abs().max().item() < 1e-20


def test_batched_tensor_inputs():
    """Verify batched parameter evaluation with matching outputs."""
    model = IMRPhenomXHM()
    f = torch.linspace(20.0, 500.0, 200)

    m1 = torch.tensor([30.0, 45.0, 25.0])
    m2 = torch.tensor([20.0, 15.0, 25.0])
    chi1 = torch.tensor([0.4, -0.3, 0.0])
    chi2 = torch.tensor([-0.2, 0.5, 0.0])

    hp, hc = model(f, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, inclination=0.5)

    assert hp.shape == (3, 200)
    assert hc.shape == (3, 200)
    assert torch.isfinite(hp).all()
    assert torch.isfinite(hc).all()


def test_autograd_differentiability():
    """Verify exact autograd backward pass yields finite, non-zero parameter gradients."""
    model = IMRPhenomXHM()
    f = torch.linspace(30.0, 400.0, 100)

    m1 = torch.tensor(32.0, requires_grad=True)
    m2 = torch.tensor(22.0, requires_grad=True)
    chi1 = torch.tensor(0.3, requires_grad=True)
    chi2 = torch.tensor(-0.15, requires_grad=True)

    hp, hc = model(f, mass1=m1, mass2=m2, chi1=chi1, chi2=chi2, inclination=0.6)
    loss = hp.abs().sum() + hc.abs().sum()
    loss.backward()

    for p, name in ((m1, "m1"), (m2, "m2"), (chi1, "chi1"), (chi2, "chi2")):
        assert p.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(p.grad), f"Gradient for {name} is not finite: {p.grad}"
        assert p.grad.item() != 0.0, f"Gradient for {name} is zero"


def test_waveform_factory_integration():
    """Verify top-level get_fd_waveform dispatches correctly to clean-room IMRPhenomXHM."""
    f = torch.linspace(25.0, 512.0, 256)
    hp, hc = get_fd_waveform(
        approximant="IMRPhenomXHM",
        mass1=30.0,
        mass2=20.0,
        chi1=0.2,
        chi2=-0.1,
        sample_frequencies=f,
        inclination=0.5,
    )
    assert hp.shape == f.shape
    assert hc.shape == f.shape
    assert torch.isfinite(hp).all()
    assert torch.isfinite(hc).all()
    assert hp.abs().max() > 0.0

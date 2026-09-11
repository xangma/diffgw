"""Tests for Resummed Tidal Effective-One-Body (TEOBResumS) model.

Clean-room literature benchmarks:
- Damour & Nagar (2010), Phys. Rev. D 81, 084016 (arXiv:0906.0096)
- Bernuzzi et al. (2015), Phys. Rev. Lett. 114, 161103 (arXiv:1412.4553)
- Nagar et al. (2018), Phys. Rev. D 98, 104052 (arXiv:1805.03891)
"""

import math
import pytest
import torch

from torchwave.approximants.teobresums import (
    TEOBResumS,
    TEOBResumSHamiltonian,
    compute_tidal_merger_frequency,
)
from torchwave.approximants.eob.hamiltonian import EOBHamiltonian
from torchwave.waveforms import get_fd_waveform


def test_point_mass_limit_potentials():
    """When Lambda_1 = Lambda_2 = 0, A(u) reduces exactly to the point-mass potential."""
    nu = torch.tensor(0.25, dtype=torch.float64)
    chi1 = torch.tensor(0.1, dtype=torch.float64)
    chi2 = torch.tensor(-0.1, dtype=torch.float64)

    # Point-mass EOB Hamiltonian
    ham_pm = EOBHamiltonian(nu=nu, chi1=chi1, chi2=chi2)

    # TEOBResumS Hamiltonian with zero tidal deformabilities
    ham_tidal_zero = TEOBResumSHamiltonian(
        nu=nu,
        chi1=chi1,
        chi2=chi2,
        lambda1=torch.tensor(0.0, dtype=torch.float64),
        lambda2=torch.tensor(0.0, dtype=torch.float64),
    )

    r = torch.tensor([5.0, 8.0, 15.0, 30.0], dtype=torch.float64)
    a_pm, b_pm, xi_pm = ham_pm.metric_potentials(r)
    a_teob, b_teob, xi_teob = ham_tidal_zero.metric_potentials(r)

    assert torch.allclose(a_teob, a_pm, atol=1e-12, rtol=1e-12)
    assert torch.allclose(b_teob, b_pm, atol=1e-12, rtol=1e-12)
    assert torch.allclose(xi_teob, xi_pm, atol=1e-12, rtol=1e-12)


def test_tidal_merger_frequency_shift():
    """Verify that increasing Lambda accelerates merger and shifts merger frequency to lower values (d f_mrg / d Lambda < 0)."""
    m1 = 1.4
    m2 = 1.4

    # Point-mass (Lambda = 0)
    f_mrg_0 = compute_tidal_merger_frequency(m1, m2, lambda1=0.0, lambda2=0.0)

    # Moderate tidal deformability (Lambda = 400)
    f_mrg_mid = compute_tidal_merger_frequency(m1, m2, lambda1=400.0, lambda2=400.0)

    # Large tidal deformability (Lambda = 1000)
    f_mrg_large = compute_tidal_merger_frequency(m1, m2, lambda1=1000.0, lambda2=1000.0)

    # Strictly decreasing merger frequency with increasing Lambda
    assert f_mrg_0.item() > f_mrg_mid.item()
    assert f_mrg_mid.item() > f_mrg_large.item()

    # Verify analytical gradient d(f_mrg)/d(Lambda) < 0 via autograd
    lam = torch.tensor(400.0, dtype=torch.float64, requires_grad=True)
    f_mrg_var = compute_tidal_merger_frequency(m1, m2, lambda1=lam, lambda2=lam)
    f_mrg_var.backward()

    assert lam.grad is not None
    assert lam.grad.item() < 0.0


def test_teobresums_hamiltonian_autograd():
    """Verify that conservative dynamics and potentials are end-to-end autograd differentiable."""
    nu = torch.tensor(0.24, dtype=torch.float64, requires_grad=True)
    chi1 = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    chi2 = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    lam1 = torch.tensor(350.0, dtype=torch.float64, requires_grad=True)
    lam2 = torch.tensor(450.0, dtype=torch.float64, requires_grad=True)

    ham = TEOBResumSHamiltonian(nu=nu, chi1=chi1, chi2=chi2, lambda1=lam1, lambda2=lam2)
    r = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    pr = torch.tensor(-0.01, dtype=torch.float64, requires_grad=True)
    pphi = torch.tensor(3.5, dtype=torch.float64, requires_grad=True)

    h_real = ham.real_hamiltonian(r, pr, pphi)
    h_real.backward()

    for p in (nu, chi1, chi2, lam1, lam2, r, pr, pphi):
        assert p.grad is not None and torch.isfinite(p.grad)


def test_teobresums_waveform_generation():
    """Verify waveform generation for BNS with TEOBResumS."""
    f = torch.linspace(25.0, 120.0, 40, dtype=torch.float64)

    hp, hc = get_fd_waveform(
        approximant="teobresums",
        mass1=1.4,
        mass2=1.3,
        chi1=0.05,
        chi2=-0.02,
        lambda1=380.0,
        lambda2=420.0,
        sample_frequencies=f,
        distance=40.0,
        inclination=0.0,
        dt_geom=1.0,
        num_steps=150,
    )

    assert hp.shape == f.shape
    assert hc.shape == f.shape
    assert torch.all(torch.isfinite(hp))
    assert torch.all(torch.isfinite(hc))
    assert torch.any(hp.abs() > 0.0)

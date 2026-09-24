"""Tests for published nonspinning TEOB components.

Clean-room literature benchmarks:
- Damour & Nagar (2010), Phys. Rev. D 81, 084016 (arXiv:0906.0096)
- Bernuzzi et al. (2015), Phys. Rev. Lett. 114, 161103 (arXiv:1412.4553)
- Nagar et al. (2018), Phys. Rev. D 98, 104052 (arXiv:1805.03891)
"""

import math
import pytest
import torch

from torchwave.approximants.teobresums import TEOBResumSHamiltonian
from torchwave.waveforms import get_fd_waveform


def test_point_mass_limit_potentials():
    """When Lambda_1 = Lambda_2 = 0, A(u) reduces exactly to the point-mass potential."""
    nu = torch.tensor(0.25, dtype=torch.float64)
    chi1 = torch.tensor(0.0, dtype=torch.float64)
    chi2 = torch.tensor(0.0, dtype=torch.float64)

    # TEOBResumS Hamiltonian with zero tidal deformabilities
    ham_tidal_zero = TEOBResumSHamiltonian(
        nu=nu,
        chi1=chi1,
        chi2=chi2,
        lambda1=torch.tensor(0.0, dtype=torch.float64),
        lambda2=torch.tensor(0.0, dtype=torch.float64),
    )

    r = torch.tensor([5.0, 8.0, 15.0, 30.0], dtype=torch.float64)
    a_teob, b_teob, xi_teob = ham_tidal_zero.metric_potentials(r)
    a_pm, b_pm, xi_pm = ham_tidal_zero.point_mass_metric_potentials(r)

    assert torch.allclose(a_teob, a_pm, atol=1e-12, rtol=1e-12)
    assert torch.allclose(b_teob, b_pm, atol=1e-12, rtol=1e-12)
    assert torch.allclose(xi_teob, xi_pm, atol=1e-12, rtol=1e-12)


def test_teobresums_hamiltonian_autograd():
    """Verify that conservative dynamics and potentials are end-to-end autograd differentiable."""
    nu = torch.tensor(0.24, dtype=torch.float64, requires_grad=True)

    ham = TEOBResumSHamiltonian(nu=nu)
    r = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    pr = torch.tensor(-0.01, dtype=torch.float64, requires_grad=True)
    pphi = torch.tensor(3.5, dtype=torch.float64, requires_grad=True)

    h_real = ham.real_hamiltonian(r, pr, pphi)
    h_real.backward()

    for p in (nu, r, pr, pphi):
        assert p.grad is not None and torch.isfinite(p.grad)

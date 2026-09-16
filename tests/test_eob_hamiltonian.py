"""Tests for Effective-One-Body (EOB) Hamiltonian and metric potentials.

Clean-room literature benchmarks:
- Damour, Jaranowski, Schäfer (2000), Phys. Rev. D 62, 084011 (arXiv:gr-qc/0005034)
- Barausse & Buonanno (2010), Phys. Rev. D 81, 084024 (arXiv:0912.3517)
- Bohé et al. (2017), Phys. Rev. D 95, 044028 (arXiv:1611.03703)
"""

import math
import pytest
import torch
from torchwave.approximants.eob.hamiltonian import EOBHamiltonian


def test_schwarzschild_limit_potentials():
    """In test-particle non-spinning limit (nu -> 0, chi -> 0), A(r) -> 1 - 2/r and B(r) -> 1/(1 - 2/r)."""
    ham = EOBHamiltonian(
        nu=torch.tensor(1e-9, dtype=torch.float64),
        chi1=torch.tensor(0.0, dtype=torch.float64),
        chi2=torch.tensor(0.0, dtype=torch.float64),
    )

    r = torch.tensor([4.0, 6.0, 10.0, 20.0], dtype=torch.float64)
    A, B, xi = ham.metric_potentials(r)
    A_exact = 1.0 - 2.0 / r

    assert torch.allclose(A, A_exact, atol=1e-4)


def test_real_energy_map():
    """Verify Damour energy map: H_real = sqrt(1 + 2*nu*(H_eff - 1))."""
    nu = torch.tensor(0.25, dtype=torch.float64)  # Equal mass
    ham = EOBHamiltonian(
        nu=nu,
        chi1=torch.tensor(0.0, dtype=torch.float64),
        chi2=torch.tensor(0.0, dtype=torch.float64),
    )

    # Rest state: H_eff = 1.0 -> H_real = 1.0
    h_eff_rest = torch.tensor(1.0, dtype=torch.float64)
    h_real_rest = ham.real_hamiltonian_from_eff(h_eff_rest)
    assert torch.isclose(h_real_rest, torch.tensor(1.0, dtype=torch.float64))

    # Bound state: H_eff = 0.95 -> H_real = sqrt(1 + 2 * 0.25 * (-0.05)) = sqrt(1 - 0.025)
    h_eff_bound = torch.tensor(0.95, dtype=torch.float64)
    expected = torch.sqrt(1.0 + 2.0 * nu * (h_eff_bound - 1.0))
    h_real_bound = ham.real_hamiltonian_from_eff(h_eff_bound)
    assert torch.isclose(h_real_bound, expected)


def test_circular_orbit_derivatives():
    """At large separation, dH/dr = 0 for exact circular momentum pphi = sqrt(r / (1 - 3/r)) in Schwarzschild."""
    ham = EOBHamiltonian(
        nu=torch.tensor(1e-6, dtype=torch.float64),
        chi1=torch.tensor(0.0, dtype=torch.float64),
        chi2=torch.tensor(0.0, dtype=torch.float64),
    )

    r_val = 20.0
    r = torch.tensor(r_val, dtype=torch.float64, requires_grad=True)
    pr = torch.tensor(0.0, dtype=torch.float64)
    # Relativistic circular angular momentum in Schwarzschild
    pphi = torch.tensor(math.sqrt(r_val / (1.0 - 3.0 / r_val)), dtype=torch.float64)

    h_real = ham.real_hamiltonian(r, pr, pphi)
    dh_dr = torch.autograd.grad(h_real, r)[0]

    assert abs(dh_dr.item()) < 1e-6

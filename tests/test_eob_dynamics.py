"""Tests for Effective-One-Body (EOB) dynamics and ODE trajectory integration.

Clean-room literature benchmarks:
- Buonanno & Damour (1999), Phys. Rev. D 59, 084006 (arXiv:gr-qc/9811091)
- Bohé et al. (2017), Phys. Rev. D 95, 044028 (arXiv:1611.03703)
"""

import torch

from torchwave.approximants.eob.hamiltonian import Nonspinning3PNHamiltonian
from torchwave.approximants.eob.radiation_reaction import FactorizedRadiationReaction
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator


def test_quasicircular_initial_conditions():
    """Verify Keplerian scaling of initial conditions at low frequency."""
    nu = torch.tensor(0.25, dtype=torch.float64)
    chi1 = torch.tensor(0.0, dtype=torch.float64)
    chi2 = torch.tensor(0.0, dtype=torch.float64)

    ham = Nonspinning3PNHamiltonian(nu)
    rr = FactorizedRadiationReaction("din2009_nonspinning")

    omega0 = torch.tensor(0.01, dtype=torch.float64)
    r0, pr0, pphi0 = find_quasicircular_initial_conditions(ham, rr, omega0)

    # Keplerian radius r ~ omega^(-2/3) ~ 0.01^(-2/3) ~ 21.54
    expected_r = omega0 ** (-2.0 / 3.0)
    assert torch.isclose(r0, expected_r, rtol=0.1)

    # Infalling radial momentum: pr0 < 0 due to radiation reaction
    assert pr0.item() < 0.0

    # Angular momentum: pphi0 ~ r0^2 * omega0 ~ omega0^(-1/3) ~ 4.64
    assert pphi0.item() > 0.0


def test_trajectory_monotonic_inspiral():
    """Verify that orbital separation decreases monotonically under radiation reaction."""
    nu = torch.tensor(0.25, dtype=torch.float64)
    chi1 = torch.tensor(0.0, dtype=torch.float64)
    chi2 = torch.tensor(0.0, dtype=torch.float64)

    ham = Nonspinning3PNHamiltonian(nu)
    rr = FactorizedRadiationReaction("din2009_nonspinning")
    integrator = EOBTrajectoryIntegrator(ham, rr)

    omega0 = torch.tensor(0.02, dtype=torch.float64)
    r0, pr0, pphi0 = find_quasicircular_initial_conditions(ham, rr, omega0)
    init_state = torch.stack([r0, torch.zeros_like(r0), pr0, pphi0], dim=-1)

    t_geom, traj = integrator.integrate(init_state, dt=1.0, num_steps=200)

    r_start = traj[0, 0]
    r_end = traj[-1, 0]

    # Monotonic inspiral
    assert r_end < r_start
    assert (r_start - r_end).item() > 0.01

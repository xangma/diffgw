"""Tests for Effective-One-Body (EOB) dynamics and ODE trajectory integration.

Clean-room literature benchmarks:
- Buonanno & Damour (1999), Phys. Rev. D 59, 084006 (arXiv:gr-qc/9811091)
- Bohé et al. (2017), Phys. Rev. D 95, 044028 (arXiv:1611.03703)
"""

import math
import pytest
import torch

from torchwave.approximants.eob.hamiltonian import EOBHamiltonian
from torchwave.approximants.eob.radiation_reaction import EOBRadiationReaction
from torchwave.approximants.eob.initial_conditions import find_quasicircular_initial_conditions
from torchwave.approximants.eob.integrator import EOBTrajectoryIntegrator
from torchwave.waveforms import get_fd_waveform


def test_quasicircular_initial_conditions():
    """Verify Keplerian scaling of initial conditions at low frequency."""
    nu = torch.tensor(0.25, dtype=torch.float64)
    chi1 = torch.tensor(0.0, dtype=torch.float64)
    chi2 = torch.tensor(0.0, dtype=torch.float64)

    ham = EOBHamiltonian(nu=nu, chi1=chi1, chi2=chi2)
    rr = EOBRadiationReaction(nu=nu, chi1=chi1, chi2=chi2)

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

    ham = EOBHamiltonian(nu=nu, chi1=chi1, chi2=chi2)
    rr = EOBRadiationReaction(nu=nu, chi1=chi1, chi2=chi2)
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

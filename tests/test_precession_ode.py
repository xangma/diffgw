"""Tests for differentiable SpinPrecessionODE dynamics engine.

Literature citations:
- Chatziioannou et al. (2017), Phys. Rev. D 95, 104004 (arXiv:1703.03967)
- Kidder (1995), Phys. Rev. D 52, 821 (arXiv:gr-qc/9506014)
- Racine (2008), Phys. Rev. D 78, 044021 (arXiv:0803.1848)
- Pratten et al. (2021), Phys. Rev. D 103, 104056 (arXiv:2004.06503)
"""

import math
import pytest
import torch

from torchwave.approximants.precession_ode import SpinPrecessionODE
from torchwave.approximants.imrphenom_xphm import IMRPhenomXPHM
from torchwave.waveforms import get_fd_waveform


def test_j_conservation_conservative_limit():
    """Verify |J| = |L + S1 + S2| is strictly conserved to < 10^(-6) in the conservative limit."""
    m1 = torch.tensor(36.0, dtype=torch.float64)
    m2 = torch.tensor(24.0, dtype=torch.float64)
    total_m = m1 + m2
    eta = (m1 * m2) / (total_m * total_m)

    m1_norm = m1 / total_m
    m2_norm = m2 / total_m

    # Generic precessing 3D spins
    chi1 = torch.tensor([0.3, -0.2, 0.4], dtype=torch.float64)
    chi2 = torch.tensor([-0.1, 0.4, -0.3], dtype=torch.float64)

    s1_0 = (m1_norm ** 2) * chi1
    s2_0 = (m2_norm ** 2) * chi2

    # Separation r = 18 M
    r_val = 18.0
    v_val = 1.0 / math.sqrt(r_val)
    l_mag = eta / v_val
    l_0 = l_mag * torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    j0 = s1_0 + s2_0 + l_0
    j0_mag = torch.norm(j0)

    # Integrate 1000 steps without radiation reaction
    sol = SpinPrecessionODE.integrate_time(
        s1_0=s1_0,
        s2_0=s2_0,
        l_0=l_0,
        m1=m1,
        m2=m2,
        total_time=1500.0,
        n_steps=1000,
        radiation_reaction=False,
    )

    j_traj = sol["j"]
    j_mags = torch.norm(j_traj, dim=-1)
    max_rel_err = torch.max(torch.abs(j_mags - j0_mag) / j0_mag).item()

    assert max_rel_err < 1e-6, f"Max relative error {max_rel_err:.2e} exceeds 1e-6 tolerance"


def test_precession_angles_properties():
    """Verify extraction of (alpha, beta, epsilon) angles produces bounded, well-behaved curves."""
    f_grid = torch.linspace(20.0, 300.0, 100, dtype=torch.float64)
    m1 = 40.0
    m2 = 20.0

    alpha, beta, epsilon = SpinPrecessionODE.compute_angles_frequency(
        f=f_grid,
        mass1=m1,
        mass2=m2,
        spin1x=0.4,
        spin1y=0.2,
        spin1z=0.1,
        spin2x=-0.2,
        spin2y=0.3,
        spin2z=-0.1,
        f_ref=20.0,
    )

    assert alpha.shape == (100,)
    assert beta.shape == (100,)
    assert epsilon.shape == (100,)

    assert torch.all(torch.isfinite(alpha))
    assert torch.all(torch.isfinite(beta))
    assert torch.all(torch.isfinite(epsilon))

    # Opening angle beta in [0, pi]
    assert torch.all(beta >= 0.0)
    assert torch.all(beta <= math.pi)

    # Minimal rotation initial condition
    assert abs(epsilon[0].item()) < 1e-5


def test_precession_ode_autograd_differentiability():
    """Verify PyTorch backward autograd through SpinPrecessionODE dynamics produces finite gradients."""
    m1 = torch.tensor(35.0, dtype=torch.float64, requires_grad=True)
    m2 = torch.tensor(25.0, dtype=torch.float64, requires_grad=True)
    s1x = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    s1y = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
    s1z = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

    f_grid = torch.linspace(25.0, 200.0, 40, dtype=torch.float64)

    alpha, beta, epsilon = SpinPrecessionODE.compute_angles_frequency(
        f=f_grid,
        mass1=m1,
        mass2=m2,
        spin1x=s1x,
        spin1y=s1y,
        spin1z=s1z,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=0.0,
        f_ref=25.0,
    )

    loss = torch.sum(alpha ** 2 + beta ** 2 + epsilon ** 2)
    loss.backward()

    for p_name, param in [("m1", m1), ("m2", m2), ("s1x", s1x), ("s1y", s1y), ("s1z", s1z)]:
        assert param.grad is not None, f"Gradient for {p_name} is None"
        assert torch.isfinite(param.grad), f"Gradient for {p_name} is non-finite: {param.grad}"
        assert abs(param.grad.item()) > 0.0, f"Gradient for {p_name} is zero"


def test_imrphenom_xphm_precession_method_ode():
    """Verify IMRPhenomXPHM with precession_method='ode' runs seamlessly."""
    f = torch.linspace(25.0, 300.0, 100, dtype=torch.float64)
    model = IMRPhenomXPHM()

    hp_ode, hc_ode = model(
        f=f,
        mass1=45.0,
        mass2=25.0,
        spin1x=0.3,
        spin1y=0.2,
        spin1z=0.1,
        spin2x=-0.2,
        spin2y=0.1,
        spin2z=-0.1,
        distance=400.0,
        inclination=math.pi / 4.0,
        precession_method="ode",
    )

    assert hp_ode.shape == (100,)
    assert hc_ode.shape == (100,)
    assert torch.all(torch.isfinite(hp_ode))
    assert torch.all(torch.isfinite(hc_ode))
    assert torch.max(torch.abs(hp_ode)) > 0.0

    # Also check through high-level get_fd_waveform
    hp_api, hc_api = get_fd_waveform(
        approximant="IMRPhenomXPHM",
        mass1=45.0,
        mass2=25.0,
        spin1x=0.3,
        spin1y=0.2,
        spin1z=0.1,
        spin2x=-0.2,
        spin2y=0.1,
        spin2z=-0.1,
        distance=400.0,
        inclination=math.pi / 4.0,
        sample_frequencies=f,
        precession_method="ode",
    )

    assert torch.allclose(hp_ode, hp_api, rtol=1e-6, atol=1e-28)
    assert torch.allclose(hc_ode, hc_api, rtol=1e-6, atol=1e-28)

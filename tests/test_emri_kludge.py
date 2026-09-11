"""Unit tests for Relativistic EMRI Kludge waveform model for LISA."""

import math
import pytest
import torch

from torchwave.constants import MSUN_SEC, PI
from torchwave.approximants.emri_kludge import (
    EMRIKludge,
    kerr_geodesic_frequencies,
    peters_mathews_radiation_reaction,
)


def test_kerr_geodesic_frequencies():
    """Verify Kerr orbital frequencies match analytical circular-equatorial limits."""
    # Central SMBH mass M = 1e6 M_sun
    m = 1.0e6
    spin = 0.6
    p = 10.0
    inc = 0.0
    e = 0.0

    # 1. Dimensionless geometric frequency verification
    omega_phi, omega_r, omega_theta, gamma_dot, alpha_dot = kerr_geodesic_frequencies(
        m, spin, p, e, inc, in_hz=False
    )

    # In geometric units: Omega_phi = 1 / (p^(3/2) + a)
    expected_omega_phi = 1.0 / (p ** 1.5 + spin)
    assert math.isclose(omega_phi.item(), expected_omega_phi, rel_tol=1e-7)

    # For equatorial circular orbit, polar frequency matches azimuthal: Omega_theta != Omega_phi only when precessing
    assert alpha_dot.item() > 0.0, "Lense-Thirring precession rate should be positive for a > 0"
    assert gamma_dot.item() > 0.0, "Periastron advance rate should be positive"

    # Physical frequency in Hz (rad/s divided by 2*pi)
    omega_phi_hz, _, _, _, _ = kerr_geodesic_frequencies(
        m, spin, p, e, inc, in_hz=True
    )
    expected_hz = expected_omega_phi / (m * MSUN_SEC)
    assert math.isclose(omega_phi_hz.item(), expected_hz, rel_tol=1e-7)


def test_peters_mathews_radiation_reaction():
    """Verify radiation reaction decreases semi-latus rectum dp/dt < 0 and eccentricity de/dt <= 0."""
    m = 1.0e6
    mu = 10.0
    p = 8.0

    # Non-zero eccentricity
    e_nonzero = 0.3
    dp_dt, de_dt = peters_mathews_radiation_reaction(m, mu, p, e_nonzero)

    assert dp_dt.item() < 0.0, f"dp/dt must be strictly negative, got {dp_dt.item()}"
    assert de_dt.item() < 0.0, f"de/dt must be strictly negative for e > 0, got {de_dt.item()}"

    # Circular orbit limit (e = 0): dp/dt < 0, de/dt == 0
    e_zero = 0.0
    dp_dt_circ, de_dt_circ = peters_mathews_radiation_reaction(m, mu, p, e_zero)
    assert dp_dt_circ.item() < 0.0, "dp/dt must be negative even for circular orbits"
    assert math.isclose(de_dt_circ.item(), 0.0, abs_tol=1e-15), "de/dt must be zero for circular orbits"


def test_emri_polarizations_and_lisa_projection():
    """Verify EMRI waveform generation and projection into LISA TDI channels A, E, T."""
    model = EMRIKludge(n_harmonics=5)

    # 1000 time points spanning 1000 seconds
    t = torch.linspace(0.0, 1000.0, 500, dtype=torch.float64)

    hp, hc = model.polarizations(
        t=t,
        total_mass=1.0e6,
        secondary_mass=10.0,
        spin=0.5,
        p0=8.0,
        e0=0.15,
        inclination=0.3,
        distance=1000.0,
    )

    assert len(hp) == len(t)
    assert len(hc) == len(t)
    assert not torch.any(torch.isnan(hp))
    assert not torch.any(torch.isnan(hc))
    assert hp.abs().max().item() > 0.0, "Plus polarization must be non-zero"
    assert hc.abs().max().item() > 0.0, "Cross polarization must be non-zero"

    # Project to LISA TDI channels
    tdi_responses = model.project_to_lisa(
        t=t,
        theta=PI / 4.0,
        phi=PI / 3.0,
        psi=PI / 6.0,
        total_mass=1.0e6,
        secondary_mass=10.0,
        spin=0.5,
        p0=8.0,
        e0=0.15,
        inclination=0.3,
        distance=1000.0,
        channels=("A", "E", "T"),
    )

    assert "A" in tdi_responses
    assert "E" in tdi_responses
    assert "T" in tdi_responses

    s_a = tdi_responses["A"]
    s_e = tdi_responses["E"]
    s_t = tdi_responses["T"]

    assert not torch.any(torch.isnan(s_a))
    assert not torch.any(torch.isnan(s_e))
    assert s_a.abs().max().item() > 0.0, "Channel A response must be non-zero"
    assert s_e.abs().max().item() > 0.0, "Channel E response must be non-zero"


def test_emri_autograd():
    """Verify EMRI waveform autograd backward pass through physical parameters."""
    model = EMRIKludge(n_harmonics=3)
    t = torch.linspace(0.0, 200.0, 100, dtype=torch.float64)

    p0 = torch.tensor(8.0, dtype=torch.float64, requires_grad=True)
    e0 = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    spin = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)

    hp, hc = model.polarizations(
        t=t,
        total_mass=1.0e6,
        secondary_mass=10.0,
        spin=spin,
        p0=p0,
        e0=e0,
        inclination=0.2,
        distance=1000.0,
    )

    loss = torch.sum(hp ** 2 + hc ** 2)
    loss.backward()

    for name, param in [("p0", p0), ("e0", e0), ("spin", spin)]:
        assert param.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(param.grad), f"Gradient for {name} is NaN"
        assert param.grad.abs().item() > 0.0, f"Gradient for {name} is zero"

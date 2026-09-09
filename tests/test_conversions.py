"""Tests for kinematic and parameter conversions."""

import math
import torch
import pytest

from torchwave.conversions import (
    component_masses_to_eta,
    component_masses_to_delta,
    component_masses_to_chirp_mass,
    spin_combinations,
    total_spin_for_remnant,
    tidal_combinations,
)


def test_equal_mass_kinematics():
    m1 = torch.tensor(10.0)
    m2 = torch.tensor(10.0)

    eta = component_masses_to_eta(m1, m2)
    delta = component_masses_to_delta(m1, m2)
    m_chirp = component_masses_to_chirp_mass(m1, m2)

    assert torch.isclose(eta, torch.tensor(0.25), atol=1e-10)
    assert torch.isclose(delta, torch.tensor(0.0), atol=1e-10)
    expected_m_chirp = (0.25 ** 0.6) * 20.0
    assert torch.isclose(m_chirp, torch.tensor(expected_m_chirp), atol=1e-10)


def test_unequal_mass_kinematics():
    m1 = torch.tensor(30.0)
    m2 = torch.tensor(10.0)

    eta = component_masses_to_eta(m1, m2)
    delta = component_masses_to_delta(m1, m2)

    # eta = 300 / 1600 = 3/16 = 0.1875
    assert torch.isclose(eta, torch.tensor(0.1875), atol=1e-10)
    # delta = 20 / 40 = 0.5
    assert torch.isclose(delta, torch.tensor(0.5), atol=1e-10)


def test_spin_combinations():
    m1 = torch.tensor(20.0)
    m2 = torch.tensor(20.0)
    s1 = torch.tensor(0.5)
    s2 = torch.tensor(-0.3)

    chi_s, chi_a, chi_pn, chi_hat = spin_combinations(m1, m2, s1, s2)

    assert torch.isclose(chi_s, torch.tensor(0.1), atol=1e-10)
    assert torch.isclose(chi_a, torch.tensor(0.4), atol=1e-10)
    # For equal mass, delta = 0, so chi_hat == chi_s == 0.1
    assert torch.isclose(chi_hat, torch.tensor(0.1), atol=1e-10)


def test_tidal_combinations():
    m1 = torch.tensor(1.4)
    m2 = torch.tensor(1.4)
    l1 = torch.tensor(400.0)
    l2 = torch.tensor(400.0)

    lambda_tilde, delta_lambda_tilde = tidal_combinations(m1, m2, l1, l2)
    # For identical equal-mass NS, delta=0, l1=l2, so delta_lambda_tilde = 0
    # and lambda_tilde = (8/13) * (1 + 7/4 - 31/16) * 800 = 400
    assert torch.isclose(lambda_tilde, torch.tensor(400.0), atol=1e-5)
    assert torch.isclose(delta_lambda_tilde, torch.tensor(0.0), atol=1e-5)

"""Tests for Kerr ISCO radius and frequency calculations and their integration with TaylorF2.

Formulas derived strictly from Bardeen, Press, Teukolsky (1972) Eq. 2.10.
"""

import math
import pytest
import torch

from torchwave.conversions import kerr_isco_radius, kerr_isco_frequency
from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.constants import MSUN_SEC, PI


def test_kerr_isco_schwarzschild_limit():
    """At chi=0, Kerr ISCO must match Schwarzschild ISCO r=6M, f = 1/(6^(3/2) * pi * M)."""
    chi = torch.tensor([0.0], dtype=torch.float64)
    r_isco = kerr_isco_radius(chi)
    assert torch.isclose(r_isco, torch.tensor([6.0], dtype=torch.float64), atol=1e-6)

    m = torch.tensor([20.0], dtype=torch.float64)
    f_isco = kerr_isco_frequency(m, chi)
    f_schw = 1.0 / (6.0 ** 1.5 * PI * (m * MSUN_SEC))
    assert torch.isclose(f_isco, f_schw, rtol=1e-5)


def test_kerr_isco_extremal_limits():
    """At prograde extremal spin chi->1, r_ISCO->1M; at retrograde extremal spin chi->-1, r_ISCO->9M."""
    chi_pro = torch.tensor([0.999], dtype=torch.float64)
    r_pro = kerr_isco_radius(chi_pro)
    assert 1.0 <= r_pro.item() < 1.5

    chi_ret = torch.tensor([-0.999], dtype=torch.float64)
    r_ret = kerr_isco_radius(chi_ret)
    assert 8.8 < r_ret.item() <= 9.0


def test_kerr_isco_monotonicity():
    """r_ISCO must monotonically decrease as chi increases from retrograde to prograde."""
    chis = torch.linspace(-0.95, 0.95, 100, dtype=torch.float64)
    r_iscos = kerr_isco_radius(chis)
    diffs = r_iscos[1:] - r_iscos[:-1]
    assert torch.all(diffs < 0.0)


def test_kerr_isco_autograd_continuity():
    """Verify C1 smoothness and autograd differentiability through chi=0 without NaNs."""
    chis = torch.linspace(-0.5, 0.5, 101, dtype=torch.float64, requires_grad=True)
    r_iscos = kerr_isco_radius(chis)
    loss = r_iscos.sum()
    loss.backward()

    assert not torch.isnan(chis.grad).any()
    assert not torch.isinf(chis.grad).any()

    # Analytical derivative at chi=0: dr/dchi = -4 * sqrt(2/3) ~ -3.265986
    chi_zero = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    r_zero = kerr_isco_radius(chi_zero)
    r_zero.backward()
    expected_grad = -4.0 * math.sqrt(2.0 / 3.0)
    assert torch.isclose(chi_zero.grad, torch.tensor([expected_grad], dtype=torch.float64), rtol=1e-3)


def test_taylorf2_kerr_isco_cutoff():
    """TaylorF2 with isco_type='kerr' should apply prograde/retrograde cutoff frequencies."""
    f = torch.linspace(20.0, 1000.0, 500, dtype=torch.float64)
    m1, m2 = 10.0, 10.0

    # Schwarzschild (chi=0)
    hp_schw, _ = TaylorF2()(f, m1, m2, chi1=0.0, chi2=0.0, f_isco_cutoff=True, isco_type="schwarzschild")
    # Prograde Kerr (chi=0.8)
    hp_kerr_pro, _ = TaylorF2()(f, m1, m2, chi1=0.8, chi2=0.8, f_isco_cutoff=True, isco_type="kerr")
    # Retrograde Kerr (chi=-0.8)
    hp_kerr_ret, _ = TaylorF2()(f, m1, m2, chi1=-0.8, chi2=-0.8, f_isco_cutoff=True, isco_type="kerr")

    # Prograde ISCO frequency > Schwarzschild ISCO frequency > Retrograde ISCO frequency
    idx_schw = torch.where(torch.abs(hp_schw) > 0.0)[0][-1]
    idx_pro = torch.where(torch.abs(hp_kerr_pro) > 0.0)[0][-1]
    idx_ret = torch.where(torch.abs(hp_kerr_ret) > 0.0)[0][-1]

    assert f[idx_pro] > f[idx_schw]
    assert f[idx_schw] > f[idx_ret]

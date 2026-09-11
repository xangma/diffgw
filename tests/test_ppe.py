"""Unit tests for Parameterized Post-Einsteinian (ppE) and modified dispersion."""

import math
import pytest
import torch

from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.approximants.imrphenom_xas import IMRPhenomXAS
from torchwave.approximants.ppe import (
    ParameterizedPostEinstein,
    ppe_phase_shift,
    modified_dispersion_phase_shift,
)


def test_gr_recovery_limits():
    """Verify beta_ppE = 0 and lambda_g -> inf recovers exact GR waveform to machine precision."""
    f = torch.linspace(25.0, 300.0, 100, dtype=torch.float64)
    m1 = 30.0
    m2 = 25.0
    dist = 100.0

    # Test across multiple underlying approximants
    for approx_cls in [IMRPhenomD, IMRPhenomXAS]:
        base_model = approx_cls()
        ppe_model = ParameterizedPostEinstein(approximant=base_model)

        # Pure GR evaluation
        hp_gr, hc_gr = base_model(f=f, mass1=m1, mass2=m2, distance=dist)

        # ppE model with beta=0 and lambda_g=inf
        hp_ppe, hc_ppe = ppe_model(
            f=f,
            mass1=m1,
            mass2=m2,
            distance=dist,
            beta_ppe=0.0,
            lambda_g=float("inf"),
        )

        assert torch.allclose(hp_ppe, hp_gr, atol=1e-15, rtol=1e-15), f"Failed for {approx_cls}"
        assert torch.allclose(hc_ppe, hc_gr, atol=1e-15, rtol=1e-15), f"Failed for {approx_cls}"


def test_ppe_pn_order_scaling():
    """Verify ppE phase shift power-law frequency dependence v^b = (pi*M*f)^(b/3)."""
    f = torch.tensor([50.0, 100.0, 200.0], dtype=torch.float64)
    total_mass = 50.0
    beta = 0.1

    # PN orders: b = -7 (-1PN dipole), b = -2 (1.5PN), b = 0 (2.5PN)
    for b in [-7, -2, 0]:
        d_psi = ppe_phase_shift(f, total_mass=total_mass, beta_ppe=beta, b_ppe=b)

        if b == 0:
            # For b = 0 (2.5PN), phase shift is independent of frequency: v^0 = 1
            assert torch.allclose(d_psi, torch.tensor([beta, beta, beta], dtype=torch.float64), atol=1e-12)
        else:
            # Ratio when frequency doubles: (2 f / f)^(b/3) = 2^(b/3)
            ratio = (d_psi[1] / d_psi[0]).item()
            expected_ratio = 2.0 ** (float(b) / 3.0)
            assert math.isclose(ratio, expected_ratio, rel_tol=1e-6)


def test_modified_dispersion_zero_limit():
    """Verify modified dispersion phase shift vanishes as lambda_g -> inf."""
    f = torch.linspace(20.0, 200.0, 50, dtype=torch.float64)

    # 1. Infinite Compton wavelength (massless graviton)
    shift_inf = modified_dispersion_phase_shift(f, lambda_g=float("inf"))
    assert torch.all(shift_inf == 0.0)

    # 2. Finite Compton wavelength (massive graviton)
    # lambda_g ~ 1e15 meters (~ 1 light year)
    l_g_finite = 1.0e15
    shift_finite = modified_dispersion_phase_shift(f, distance=100.0, lambda_g=l_g_finite)

    assert torch.all(shift_finite < 0.0), "Massive graviton dispersion shift must be negative"
    assert not torch.any(torch.isnan(shift_finite))

    # Phase shift ~ f^(-1), so 2x frequency gives 0.5x shift
    assert math.isclose((shift_finite[1] / shift_finite[0]).item(), (f[0] / f[1]).item(), rel_tol=1e-6)


def test_ppe_autograd():
    """Verify non-zero autograd gradients through ppE waveform parameters."""
    f = torch.linspace(30.0, 250.0, 64, dtype=torch.float64)
    ppe_model = ParameterizedPostEinstein(approximant=IMRPhenomD())

    beta_var = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
    m1_var = torch.tensor(32.0, dtype=torch.float64, requires_grad=True)

    hp, hc = ppe_model(
        f=f,
        mass1=m1_var,
        mass2=28.0,
        distance=100.0,
        beta_ppe=beta_var,
        b_ppe=-2,
    )

    loss = torch.sum(torch.real(hp * torch.conj(hp)))
    loss.backward()

    assert beta_var.grad is not None and not torch.isnan(beta_var.grad)
    assert m1_var.grad is not None and not torch.isnan(m1_var.grad)
    assert beta_var.grad.abs().item() > 0.0

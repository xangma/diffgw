"""Unit tests for Pulsar Timing Array (PTA) network and Hellings-Downs likelihood."""

import math
import pytest
import torch

from torchwave.constants import PI
from torchwave.detectors.pta import (
    PTANetwork,
    hellings_downs_correlation,
    stochastic_gwb_spectrum,
    pulsar_line_of_sight,
    angular_separation,
    SEC_PER_YEAR,
    F_YEAR,
)


def test_pulsar_geometry_and_separations():
    """Verify pulsar 3D Cartesian line-of-sight unit vectors and angular separations."""
    # North pole: dec = pi/2
    p_np = pulsar_line_of_sight(ra=0.0, dec=PI / 2.0)
    assert torch.allclose(p_np, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-12)

    # South pole: dec = -pi/2
    p_sp = pulsar_line_of_sight(ra=0.0, dec=-PI / 2.0)
    assert torch.allclose(p_sp, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64), atol=1e-12)

    # Antipodal separation should be exactly pi
    zeta_ns, cos_zeta = angular_separation(p_np, p_sp)
    assert math.isclose(zeta_ns.item(), PI, abs_tol=1e-12)
    assert math.isclose(cos_zeta.item(), -1.0, abs_tol=1e-12)


def test_hellings_downs_correlation_values():
    """Verify Hellings-Downs correlation satisfies analytical limits:

    - Cross-term limit at zero separation: Gamma(0) = 0.5
    - Auto-correlation term (delta_ab = 1): Gamma_aa = 1.0
    - Exact analytical value at antipodal separation (zeta = pi): Gamma(pi) = 0.25
    - Consistency with Allen (2023) / NANOGrav cosmic variance normalization 0.5 * (1 - ln 2) ~ 0.1534
    """
    # 1. Cross-term at zeta = 0 (distinct pulsars along same line of sight)
    zeta_zero = torch.tensor(0.0, dtype=torch.float64)
    gamma_cross_0 = hellings_downs_correlation(zeta_zero, is_auto=False)
    assert math.isclose(gamma_cross_0.item(), 0.5, abs_tol=1e-10), f"Expected 0.5, got {gamma_cross_0.item()}"

    # Auto-correlation at zeta = 0 (same pulsar)
    gamma_auto_0 = hellings_downs_correlation(zeta_zero, is_auto=True)
    assert math.isclose(gamma_auto_0.item(), 1.0, abs_tol=1e-10), f"Expected 1.0, got {gamma_auto_0.item()}"

    # 2. Antipodal separation zeta = pi
    zeta_pi = torch.tensor(PI, dtype=torch.float64)
    gamma_cross_pi = hellings_downs_correlation(zeta_pi, is_auto=False)
    assert math.isclose(gamma_cross_pi.item(), 0.25, abs_tol=1e-10), f"Expected 0.25, got {gamma_cross_pi.item()}"

    # 3. Verify the cosmic variance / unpolarized normalization constant 0.5 * (1 - ln 2) ~ 0.1534 (Allen 2023)
    c_allen = 0.5 * (1.0 - math.log(2.0))
    assert math.isclose(c_allen, 0.153426, rel_tol=1e-4)

    # 4. Characteristic dip at 90 degrees
    zeta_90 = torch.tensor(PI / 2.0, dtype=torch.float64)
    gamma_cross_90 = hellings_downs_correlation(zeta_90, is_auto=False)
    # At x = 0.5: Gamma = 1/2 - 1/8 + 3/4 * ln(1/2) = 3/8 - 3/4 * ln(2) ~ -0.1448
    expected_90 = 0.375 - 0.75 * math.log(2.0)
    assert math.isclose(gamma_cross_90.item(), expected_90, rel_tol=1e-6)
    assert gamma_cross_90.item() < 0.0, "HD curve must dip negative around 90 degrees"


def test_stochastic_gwb_spectrum():
    """Verify stochastic GWB strain spectral density S_h(f) scaling."""
    f = torch.tensor([1e-8, 3.16875e-8, 1e-7], dtype=torch.float64)
    a_gwb = 2.0e-15
    gamma = 13.0 / 3.0  # SMBHB standard inspiral index

    s_h = stochastic_gwb_spectrum(f, a_gwb=a_gwb, gamma=gamma)
    assert torch.all(s_h > 0.0)
    assert not torch.any(torch.isnan(s_h))

    # Slope check: S_h ~ f^(2 - 13/3) = f^(-7/3), so higher frequency has lower power
    assert s_h[0].item() > s_h[1].item() > s_h[2].item()


def test_pta_covariance_and_likelihood():
    """Verify PTA residual covariance matrix C is positive definite and log-likelihood is differentiable."""
    # Array of 5 pulsars with diverse sky positions
    names = ["J0030+0451", "J0437-4715", "J1713+0747", "J1744-1134", "J1909-3744"]
    ra = [0.13, 1.21, 4.51, 4.64, 5.01]
    dec = [0.08, -0.82, 0.13, -0.20, -0.65]
    noise_std = [1e-7, 8e-8, 5e-8, 1.2e-7, 6e-8]

    pta = PTANetwork(names, ra, dec, noise_std=noise_std)

    # 1. Covariance positive-definiteness
    a_gwb = 2.5e-15
    gamma = 13.0 / 3.0
    cov = pta.covariance_matrix(a_gwb=a_gwb, gamma=gamma)

    # Check symmetry
    assert torch.allclose(cov, cov.T, atol=1e-18)

    # Check positive eigenvalues
    eigvals = torch.linalg.eigvalsh(cov)
    assert torch.all(eigvals > 0.0), f"Non-positive eigenvalues found: {eigvals}"

    # Cholesky decomposition must succeed
    chol = torch.linalg.cholesky(cov)
    assert not torch.any(torch.isnan(chol))

    # 2. Differentiable log-likelihood evaluation
    torch.manual_seed(42)
    residuals = torch.randn(5, dtype=torch.float64) * 1e-7

    a_var = torch.tensor(a_gwb, dtype=torch.float64, requires_grad=True)
    gam_var = torch.tensor(gamma, dtype=torch.float64, requires_grad=True)

    ln_l = pta.log_likelihood(residuals, a_gwb=a_var, gamma=gam_var)

    assert not torch.isnan(ln_l)
    assert not torch.isinf(ln_l)

    ln_l.backward()

    assert a_var.grad is not None and not torch.isnan(a_var.grad)
    assert gam_var.grad is not None and not torch.isnan(gam_var.grad)

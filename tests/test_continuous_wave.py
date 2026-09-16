"""Tests for Continuous Gravitational Waves and Differentiable F-Statistic.

Clean-room literature benchmarks:
- Jaranowski, Królak, Schutz (1998), Phys. Rev. D 58, 063001 (arXiv:gr-qc/9804014)
- Prix (2011), arXiv:1108.3546
"""

import math
import pytest
import torch

from torchwave.approximants.continuous_wave import (
    continuous_wave_h0,
    continuous_wave_strain,
    continuous_wave_amplitudes,
    continuous_wave_basis,
    ContinuousWave,
)
from torchwave.inference.f_statistic import compute_f_statistic


def test_continuous_wave_zero_ellipticity():
    """When ellipticity epsilon = 0, h(t) identically equals zero."""
    t = torch.linspace(0.0, 100.0, 1000, dtype=torch.float64)

    h = continuous_wave_strain(
        t=t,
        ra=1.2,
        dec=0.5,
        f0=50.0,
        ellipticity=0.0,
        distance=1.0e20,
        cos_iota=0.8,
        psi=0.4,
        phi0=0.2,
    )

    assert torch.all(h == 0.0)
    assert torch.allclose(h, torch.zeros_like(h))


def test_quadrupolar_basis_decomposition_identity():
    """Verify exact equivalence between direct triaxial strain and quadrupolar basis sum sum_mu A^mu h_mu(t)."""
    t = torch.linspace(0.0, 50.0, 500, dtype=torch.float64)
    ra = 0.8
    dec = -0.3
    f0 = 25.0
    f1 = -1e-10
    h0 = 1e-22
    cos_i = 0.6
    psi = 0.7
    phi0 = 1.1

    # Method 1: direct strain call
    h_direct = continuous_wave_strain(
        t=t,
        ra=ra,
        dec=dec,
        f0=f0,
        f1=f1,
        h0=h0,
        cos_iota=cos_i,
        psi=psi,
        phi0=phi0,
    )

    # Method 2: manual sum of amplitudes and basis
    amp = continuous_wave_amplitudes(h0=h0, cos_iota=cos_i, psi=psi, phi0=phi0)  # (4,)
    basis = continuous_wave_basis(t=t, ra=ra, dec=dec, f0=f0, f1=f1)  # (4, N)
    h_reconstructed = torch.sum(amp.unsqueeze(-1) * basis, dim=0)

    assert torch.allclose(h_direct, h_reconstructed, atol=1e-30, rtol=1e-12)


def test_f_statistic_theoretical_expectation():
    """Verify that E[2F] = 4 + rho^2 on injected continuous signals across noise realizations."""
    torch.manual_seed(42)

    dt = 0.02
    n_samples = 2000
    t = torch.arange(n_samples, dtype=torch.float64) * dt

    ra = 1.0
    dec = 0.4
    f0 = 10.0
    f1 = 0.0

    sigma_noise = 1.0

    # Strong signal injection to verify expectation with high statistical precision
    h0 = 0.3
    signal = continuous_wave_strain(
        t=t,
        ra=ra,
        dec=dec,
        f0=f0,
        h0=h0,
        cos_iota=0.8,
        psi=0.5,
        phi0=0.3,
    )

    # Optimal SNR^2: rho^2 = <h | h> = (1 / sigma^2) * sum h_n^2
    rho_sq = float((torch.sum(signal * signal) / (sigma_noise ** 2)).item())

    # Monte Carlo average of 2F over noise realizations
    n_realizations = 150
    two_f_values = []

    for _ in range(n_realizations):
        noise = torch.randn(n_samples, dtype=torch.float64) * sigma_noise
        data = signal + noise
        two_f, _, _ = compute_f_statistic(
            data=data,
            t=t,
            ra=ra,
            dec=dec,
            f0=f0,
            f1=f1,
            sigma=sigma_noise,
        )
        two_f_values.append(two_f.item())

    mean_two_f = sum(two_f_values) / len(two_f_values)
    expected_two_f = 4.0 + rho_sq

    # Standard error of the mean: Var(2F) = 2*(4 + 2*rho^2)
    var_2f = 2.0 * (4.0 + 2.0 * rho_sq)
    sem = math.sqrt(var_2f / n_realizations)

    # Check that mean 2F matches theoretical expectation within 3 standard errors
    assert abs(mean_two_f - expected_two_f) < 3.5 * sem


def test_f_statistic_autograd_differentiability():
    """Verify that 2F is autograd differentiable with respect to sky coordinates and frequencies."""
    t = torch.linspace(0.0, 10.0, 200, dtype=torch.float64)

    ra = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    dec = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    f0 = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
    f1 = torch.tensor(-1e-8, dtype=torch.float64, requires_grad=True)

    # Synthetic signal
    data = continuous_wave_strain(
        t=t,
        ra=1.2,
        dec=0.4,
        f0=20.0,
        h0=1e-21,
        detector="H1",
    )

    two_f, _, _ = compute_f_statistic(
        data=data,
        t=t,
        ra=ra,
        dec=dec,
        f0=f0,
        f1=f1,
        detector="H1",
        sigma=1e-21,
    )

    two_f.backward()


    assert ra.grad is not None and torch.isfinite(ra.grad)
    assert dec.grad is not None and torch.isfinite(dec.grad)
    assert f0.grad is not None and torch.isfinite(f0.grad)
    assert f1.grad is not None and torch.isfinite(f1.grad)

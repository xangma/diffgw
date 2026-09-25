"""Unit tests for differentiable Fisher information matrix, covariance, and MAP optimizer."""

import math
import pytest
import torch
from torchwave import get_fd_waveform
from torchwave.detectors.psd import aLIGOZeroDetHighPower
from torchwave.detectors.network import DetectorNetwork
from torchwave.inference.fisher import (
    optimal_snr,
    matched_filter_snr,
    fisher_information_matrix,
    covariance_matrix,
    sky_localization_area,
    optimize_map,
)


def test_optimal_snr_scaling():
    """Verify optimal SNR is positive and strictly scales as 1 / d_L."""
    f = torch.linspace(20.0, 500.0, 200, dtype=torch.float64)
    df = (f[1] - f[0]).item()
    psd = aLIGOZeroDetHighPower(f)

    hp1, _ = get_fd_waveform("TaylorF2", mass1=1.4, mass2=1.4, sample_frequencies=f, distance=100.0)
    hp2, _ = get_fd_waveform("TaylorF2", mass1=1.4, mass2=1.4, sample_frequencies=f, distance=200.0)

    snr1 = optimal_snr(hp1, psd, df=df)
    snr2 = optimal_snr(hp2, psd, df=df)

    assert snr1 > 0.0
    assert snr2 > 0.0
    assert torch.isclose(snr1 / snr2, torch.tensor(2.0, dtype=torch.float64), rtol=1e-4)


def test_matched_filter_snr():
    """Verify matched filter SNR equals optimal SNR when template equals data."""
    f = torch.linspace(20.0, 500.0, 200, dtype=torch.float64)
    df = (f[1] - f[0]).item()
    psd = aLIGOZeroDetHighPower(f)

    hp, _ = get_fd_waveform("TaylorF2", mass1=1.4, mass2=1.4, sample_frequencies=f, distance=100.0)
    opt_snr = optimal_snr(hp, psd, df=df)
    mf_snr = matched_filter_snr(hp, hp, psd, df=df)

    assert torch.isclose(opt_snr, mf_snr, rtol=1e-5)


def test_fisher_symmetric_positive_definite():
    """Verify that Fisher matrix is symmetric and strictly positive definite (all eigenvalues > 0)."""
    f = torch.linspace(20.0, 400.0, 150, dtype=torch.float64)
    df = (f[1] - f[0]).item()
    psd = aLIGOZeroDetHighPower(f)

    def waveform_fn(p_dict):
        m1 = p_dict["mass1"]
        m2 = p_dict["mass2"]
        dist = p_dict["distance"]
        hp, _ = get_fd_waveform("TaylorF2", mass1=m1, mass2=m2, distance=dist, sample_frequencies=f)
        return hp

    # At equal masses, exchanging mass1 and mass2 is an exact local
    # degeneracy, so their two Fisher columns cannot be independent.
    params = {
        "mass1": 1.6,
        "mass2": 1.2,
        "distance": 100.0,
    }

    gamma, names = fisher_information_matrix(
        parameters_dict=params,
        waveform_fn=waveform_fn,
        psd_dict=psd,
        df=df,
        f_low=20.0,
        f_high=400.0,
    )

    assert gamma.shape == (3, 3)
    # Check symmetry: |Gamma - Gamma^T| < 1e-10
    sym_diff = torch.max(torch.abs(gamma - gamma.transpose(0, 1))).item()
    assert sym_diff < 1e-8, f"Fisher matrix not symmetric: diff={sym_diff}"

    # Check positive definiteness: all eigenvalues > 0
    eigenvalues = torch.linalg.eigvalsh(gamma)
    min_ev = torch.min(eigenvalues).item()
    assert min_ev > 0.0, f"Fisher matrix has non-positive eigenvalue: {min_ev}"

    # Invert to covariance matrix
    sigma = covariance_matrix(gamma)
    assert sigma.shape == (3, 3)
    cov_eigenvalues = torch.linalg.eigvalsh(sigma)
    assert torch.min(cov_eigenvalues).item() > 0.0


def test_sky_localization_area():
    """Verify sky localization area computation formula."""
    # Synthetic covariance matrix for [mass, ra, dec]
    cov = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, (0.01)**2, 0.5 * 0.01 * 0.02],
        [0.0, 0.5 * 0.01 * 0.02, (0.02)**2],
    ], dtype=torch.float64)

    # Dec = 0.5 rad
    area = sky_localization_area(cov, dec_idx=2, ra_idx=1, dec=0.5)
    expected_det = (0.01**2) * (0.02**2) - (0.5 * 0.01 * 0.02)**2
    expected_area = 2.0 * math.pi * math.log(10.0) * math.sqrt(expected_det) * math.cos(0.5)

    assert torch.isclose(area, torch.tensor(expected_area, dtype=torch.float64), rtol=1e-5)


def test_map_optimizer_convergence():
    """Verify MAP optimizer converges close to true injection parameters for high SNR signal (SNR >= 30)."""
    f = torch.linspace(20.0, 300.0, 100, dtype=torch.float64)
    df = (f[1] - f[0]).item()
    psd = aLIGOZeroDetHighPower(f)

    true_params = {"mass1": 15.0, "mass2": 15.0}

    def waveform_fn(p_dict):
        m1 = p_dict["mass1"]
        m2 = p_dict["mass2"]
        hp, _ = get_fd_waveform("TaylorF2", mass1=m1, mass2=m2, distance=20.0, sample_frequencies=f)
        return hp

    # High SNR injection
    clean_data = waveform_fn({k: torch.tensor(v, dtype=torch.float64) for k, v in true_params.items()})
    inj_snr = optimal_snr(clean_data, psd, df=df).item()
    assert inj_snr >= 30.0, f"Injection SNR {inj_snr} is less than 30"

    # Start near the true value
    init_params = {"mass1": 14.9, "mass2": 14.9}
    bounds = {"mass1": (10.0, 20.0), "mass2": (10.0, 20.0)}

    best_p, history = optimize_map(
        data=clean_data,
        psd=psd,
        initial_params=init_params,
        waveform_fn=waveform_fn,
        bounds=bounds,
        lr=0.01,
        steps=50,
        df=df,
        f_low=20.0,
        f_high=300.0,
    )

    # Check loss decreased
    assert history[-1] < history[0]
    # Check convergence close to true values (within 3% relative error)
    rel_err_m1 = abs(best_p["mass1"] - true_params["mass1"]) / true_params["mass1"]
    rel_err_m2 = abs(best_p["mass2"] - true_params["mass2"]) / true_params["mass2"]
    assert rel_err_m1 < 0.05
    assert rel_err_m2 < 0.05

"""Tests for relative binning (heterodyned) accelerated likelihood."""

import math
import torch

from torchwave.approximants.imrphenomd import IMRPhenomD
from torchwave.detectors.psd import aLIGOZeroDetHighPower
from torchwave.inference.likelihood import WhittleLikelihood
from torchwave.inference.relative_binning import (
    RelativeBinningLikelihood,
    compute_relative_binning_bins,
)


def test_relative_binning_bins_properties():
    """Verify bin edges are strictly increasing and adapt to PN power law."""
    edges = compute_relative_binning_bins(
        f_min=20.0,
        f_max=500.0,
        n_bins=40,
        gamma=-5.0 / 3.0,
    )
    assert len(edges) == 41
    assert math.isclose(edges[0].item(), 20.0, rel_tol=1e-6)
    assert math.isclose(edges[-1].item(), 500.0, rel_tol=1e-6)
    # Strictly increasing
    assert torch.all(edges[1:] > edges[:-1])

    # Bins should be narrower at low frequencies (where dPsi/df is largest)
    low_width = (edges[1] - edges[0]).item()
    high_width = (edges[-1] - edges[-2]).item()
    assert low_width < high_width


def test_relative_binning_accuracy():
    """Check absolute log-likelihood error below 0.05 for declared nearby masses."""
    device = torch.device("cpu")
    # Dense frequency grid: 32768 points from 20 to 400 Hz
    n_dense = 32768
    f_dense = torch.linspace(20.0, 400.0, n_dense, dtype=torch.float64, device=device)
    df = (f_dense[1] - f_dense[0]).item()
    psd = aLIGOZeroDetHighPower(f_dense)

    model = IMRPhenomD()

    # Fiducial reference parameters (GW150914-like)
    m1_0 = 36.0
    m2_0 = 29.0
    dist_0 = 410.0

    hp_0, _ = model(f=f_dense, mass1=m1_0, mass2=m2_0, distance=dist_0)

    # Synthetic data: the same plus polarization in two detector-labelled streams
    data = {"H1": hp_0, "L1": hp_0}
    psds = {"H1": psd, "L1": psd}
    h0 = {"H1": hp_0, "L1": hp_0}

    # Full dense Whittle likelihood
    full_likelihood = WhittleLikelihood(data=data, psds=psds, df=df, frequencies=f_dense)

    # Relative binning likelihood with M = 40 bins
    n_bins = 40
    rel_likelihood = RelativeBinningLikelihood(
        data=data,
        psds=psds,
        h0=h0,
        frequencies=f_dense,
        n_bins=n_bins,
        df=df,
    )
    bin_centers = rel_likelihood.bin_centers

    # 1. Exact match at fiducial reference parameter: ln L = 0
    hp_0_centers, _ = model(f=bin_centers, mass1=m1_0, mass2=m2_0, distance=dist_0)
    ln_l_rel_0 = rel_likelihood({"H1": hp_0_centers, "L1": hp_0_centers})
    ln_l_full_0 = full_likelihood({"H1": hp_0, "L1": hp_0})

    assert math.isclose(ln_l_rel_0.item(), 0.0, abs_tol=1e-8)
    assert math.isclose(ln_l_full_0.item(), 0.0, abs_tol=1e-8)

    # 2. One nearby mass perturbation; this does not establish posterior-wide accuracy
    m1_test = 36.02
    m2_test = 28.98
    hp_test_dense, _ = model(f=f_dense, mass1=m1_test, mass2=m2_test, distance=dist_0)
    hp_test_centers, _ = model(f=bin_centers, mass1=m1_test, mass2=m2_test, distance=dist_0)

    ln_l_full = full_likelihood({"H1": hp_test_dense, "L1": hp_test_dense})
    ln_l_rel = rel_likelihood({"H1": hp_test_centers, "L1": hp_test_centers})

    delta_ln_l = abs(ln_l_rel.item() - ln_l_full.item())
    assert delta_ln_l < 0.05, f"Log-likelihood discrepancy {delta_ln_l} exceeds 0.05"

    # 3. Check numerical accuracy across a small batch of nearby masses.
    # Performance belongs in repeated, environment-recorded benchmarks.
    batch_size = 16
    m1_batch = torch.linspace(35.98, 36.02, batch_size, dtype=torch.float64)
    m2_batch = torch.linspace(28.98, 29.02, batch_size, dtype=torch.float64)
    hp_dense_b, _ = model(f=f_dense, mass1=m1_batch, mass2=m2_batch, distance=dist_0)
    hp_centers_b, _ = model(f=bin_centers, mass1=m1_batch, mass2=m2_batch, distance=dist_0)
    full_batch = full_likelihood({"H1": hp_dense_b, "L1": hp_dense_b})
    relative_batch = rel_likelihood({"H1": hp_centers_b, "L1": hp_centers_b})
    assert relative_batch.shape == (batch_size,)
    assert torch.isfinite(relative_batch).all()
    coarse_error = torch.max(torch.abs(relative_batch - full_batch)).item()

    # Forty bins do not meet the scalar case's 0.05 bound across this batch.
    # Refining to 80 bins improves the approximation for these fixed inputs.
    refined = RelativeBinningLikelihood(
        data=data, psds=psds, h0=h0, frequencies=f_dense, n_bins=80, df=df,
    )
    hp_refined, _ = model(f=refined.bin_centers, mass1=m1_batch, mass2=m2_batch, distance=dist_0)
    refined_batch = refined({"H1": hp_refined, "L1": hp_refined})
    refined_error = torch.max(torch.abs(refined_batch - full_batch)).item()
    assert refined_error < coarse_error
    assert refined_error < 0.05


def test_relative_binning_autograd_gradients():
    """Verify autograd gradient error ||grad_rel - grad_full|| / ||grad_full|| < 1e-2."""
    device = torch.device("cpu")
    n_dense = 4096
    f_dense = torch.linspace(25.0, 400.0, n_dense, dtype=torch.float64, device=device)
    df = (f_dense[1] - f_dense[0]).item()
    psd = aLIGOZeroDetHighPower(f_dense)

    model = IMRPhenomD()

    m1_0 = 36.0
    m2_0 = 29.0
    dist_0 = 410.0

    hp_0, _ = model(f=f_dense, mass1=m1_0, mass2=m2_0, distance=dist_0, phic=0.0)
    data = {"H1": hp_0}
    psds = {"H1": psd}
    h0 = {"H1": hp_0}

    full_lik = WhittleLikelihood(data=data, psds=psds, df=df, frequencies=f_dense)
    rel_lik = RelativeBinningLikelihood(data=data, psds=psds, h0=h0, frequencies=f_dense, n_bins=50, df=df)
    bin_centers = rel_lik.bin_centers

    # Differentiate with respect to physical parameters: distance and coalescence phase phic
    dist_val = 405.0
    phic_val = 0.05

    # 1. Full gradient
    d_f = torch.tensor(dist_val, dtype=torch.float64, requires_grad=True)
    phic_f = torch.tensor(phic_val, dtype=torch.float64, requires_grad=True)
    hp_f, _ = model(f=f_dense, mass1=m1_0, mass2=m2_0, distance=d_f, phic=phic_f)
    ln_l_f = full_lik({"H1": hp_f})
    ln_l_f.backward()
    grad_full = torch.tensor([d_f.grad.item(), phic_f.grad.item()], dtype=torch.float64)

    # 2. Relative binning gradient
    d_r = torch.tensor(dist_val, dtype=torch.float64, requires_grad=True)
    phic_r = torch.tensor(phic_val, dtype=torch.float64, requires_grad=True)
    hp_r, _ = model(f=bin_centers, mass1=m1_0, mass2=m2_0, distance=d_r, phic=phic_r)
    ln_l_r = rel_lik({"H1": hp_r})
    ln_l_r.backward()
    grad_rel = torch.tensor([d_r.grad.item(), phic_r.grad.item()], dtype=torch.float64)

    norm_full = torch.norm(grad_full)
    norm_diff = torch.norm(grad_rel - grad_full)
    rel_error = (norm_diff / norm_full).item()

    assert norm_full > 0.0
    assert rel_error < 1.0e-2, f"Relative gradient error {rel_error:.4e} exceeds 1e-2 threshold"

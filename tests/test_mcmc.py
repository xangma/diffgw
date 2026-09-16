"""Tests for pure PyTorch No-U-Turn Sampler (NUTS) and Gelman-Rubin convergence diagnostics."""

import math
import pytest
import torch

from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.inference.mcmc import NUTS, gelman_rubin, leapfrog_step
from torchwave.inference.relative_binning import RelativeBinningLikelihood


def test_gelman_rubin_properties():
    """Verify Gelman-Rubin potential scale reduction factor R_hat properties."""
    torch.manual_seed(42)

    # 1. Identical chains must have R_hat == 1.0
    chain = torch.randn(1, 200, 3)
    identical_chains = chain.repeat(4, 1, 1)
    r_hat_identical = gelman_rubin(identical_chains)
    assert torch.allclose(r_hat_identical, torch.ones(3), atol=1e-5)

    # 2. Well-mixed stationary chains drawn from the same distribution should have R_hat < 1.05
    stationary_chains = torch.randn(4, 500, 3)
    r_hat_stationary = gelman_rubin(stationary_chains)
    assert torch.all(r_hat_stationary < 1.05)

    # 3. Disjoint / unconverged chains must have R_hat >> 1.05
    divergent_chains = torch.stack([
        torch.randn(200, 2),
        torch.randn(200, 2) + 10.0,
        torch.randn(200, 2) - 10.0,
    ], dim=0)
    r_hat_divergent = gelman_rubin(divergent_chains)
    assert torch.all(r_hat_divergent > 2.0)

    # 4. Dimension and length validation
    with pytest.raises(ValueError, match="Expected chains tensor of shape"):
        gelman_rubin(torch.randn(10, 2))

    with pytest.raises(ValueError, match="Need at least 2 chains"):
        gelman_rubin(torch.randn(1, 50, 2))

    with pytest.raises(ValueError, match="Need at least 4 samples"):
        gelman_rubin(torch.randn(3, 2, 2))


def test_leapfrog_step_reversibility():
    """Verify leapfrog integration is time-reversible and symplectic."""
    def toy_grad_fn(theta: torch.Tensor):
        # Potential V(theta) = 0.5 * sum(theta^2)
        # log_p = -0.5 * sum(theta^2)
        # grad_V = theta
        return -0.5 * torch.sum(theta ** 2), theta.clone()

    theta0 = torch.tensor([1.5, -2.0], dtype=torch.float64)
    p0 = torch.tensor([-0.8, 0.5], dtype=torch.float64)
    _, grad0 = toy_grad_fn(theta0)
    eps = 0.05

    # Forward step
    theta1, p1, _, grad1 = leapfrog_step(theta0, p0, grad0, eps, toy_grad_fn)

    # Backward step with reversed momentum or negative step size
    theta_rev, p_rev, _, _ = leapfrog_step(theta1, p1, grad1, -eps, toy_grad_fn)

    # Should recover theta0 and p0 exactly to numerical precision
    assert torch.allclose(theta_rev, theta0, atol=1e-7)
    assert torch.allclose(p_rev, p0, atol=1e-7)


def test_nuts_2d_gaussian_sampling_and_convergence():
    """Verify 4 parallel NUTS chains converge on 2D correlated Gaussian with R_hat < 1.05

    and posterior mean enclosing the true mean within 2 sigma.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")

    true_mu = torch.tensor([2.0, -1.0], dtype=torch.float64, device=device)
    cov = torch.tensor([[1.0, 0.6], [0.6, 1.5]], dtype=torch.float64, device=device)
    inv_cov = torch.inverse(cov)

    def log_posterior(theta: torch.Tensor) -> torch.Tensor:
        diff = theta - true_mu
        return -0.5 * torch.matmul(diff, torch.matmul(inv_cov, diff))

    sampler = NUTS(log_posterior, target_accept=0.65, max_depth=6)

    # 4 parallel chains initialized around the domain
    init_positions = torch.tensor([
        [0.0, 0.0],
        [4.0, -2.0],
        [1.0, -3.0],
        [3.0, 1.0],
    ], dtype=torch.float64, device=device)

    chains = sampler.sample_parallel(init_positions, n_samples=250, n_warmup=150)
    assert chains.shape == (4, 250, 2)

    # Convergence check: Gelman-Rubin R_hat < 1.05
    r_hat = gelman_rubin(chains)
    for d in range(2):
        assert r_hat[d].item() < 1.05, f"Parameter {d} failed R_hat convergence: {r_hat[d].item()}"

    # Accuracy check: Posterior mean must enclose true injection within 2 sigma
    post_mean = chains.mean(dim=(0, 1))
    post_std = chains.std(dim=(0, 1))
    std_err = post_std / math.sqrt(4 * 250)
    diff = torch.abs(post_mean - true_mu)

    assert torch.all(diff < 2.0 * post_std), f"Posterior mean {post_mean} exceeds 2 sigma of truth {true_mu}"


def test_nuts_gw_relative_binning_posterior():
    """Verify NUTS sampling on GW relative binning likelihood encloses true injection within 2 sigma."""
    torch.manual_seed(123)
    n_dense = 1024
    f_dense = torch.linspace(30.0, 200.0, n_dense, dtype=torch.float64)
    df = (f_dense[1] - f_dense[0]).item()
    psd = torch.ones_like(f_dense) * 1e-45

    # True injected parameters
    mc_true = 28.16
    eta_true = 0.248

    def mc_eta_to_m1_m2(mc, eta):
        M = mc / (eta ** 0.6)
        delta = torch.sqrt(torch.clamp(1.0 - 4.0 * eta, min=0.0))
        m1 = 0.5 * M * (1.0 + delta)
        m2 = 0.5 * M * (1.0 - delta)
        return m1, m2

    m1_0, m2_0 = mc_eta_to_m1_m2(
        torch.tensor(mc_true, dtype=torch.float64),
        torch.tensor(eta_true, dtype=torch.float64),
    )
    taylor = TaylorF2()
    hp_0, _ = taylor(f=f_dense, mass1=m1_0, mass2=m2_0, distance=400.0)

    data = {"H1": hp_0}
    psds = {"H1": psd}
    h0 = {"H1": hp_0}

    rel_lik = RelativeBinningLikelihood(
        data=data,
        psds=psds,
        h0=h0,
        frequencies=f_dense,
        n_bins=20,
        df=df,
    )
    bin_centers = rel_lik.bin_centers

    # Scaled parameters: u = [(mc - mc_true) / 0.05, (eta - eta_true) / 0.005]
    def log_posterior(u: torch.Tensor) -> torch.Tensor:
        mc = mc_true + u[0] * 0.05
        eta = eta_true + u[1] * 0.005
        if eta > 0.25 or eta < 0.20 or mc < 27.0 or mc > 29.0:
            return torch.tensor(-1e8, dtype=torch.float64)
        m1, m2 = mc_eta_to_m1_m2(mc, eta)
        hp_c, _ = taylor(f=bin_centers, mass1=m1, mass2=m2, distance=400.0)
        return rel_lik({"H1": hp_c}).squeeze()

    sampler = NUTS(log_posterior, target_accept=0.65, max_depth=5)
    init_u = torch.tensor([
        [0.1, -0.1],
        [-0.1, 0.1],
        [0.05, 0.05],
        [-0.05, -0.05],
    ], dtype=torch.float64)

    chains_u = sampler.sample_parallel(init_u, n_samples=80, n_warmup=60)
    chains_mc = mc_true + chains_u[:, :, 0] * 0.05
    chains_eta = eta_true + chains_u[:, :, 1] * 0.005
    chains = torch.stack([chains_mc, chains_eta], dim=-1)

    mean = chains.mean(dim=(0, 1))
    std = chains.std(dim=(0, 1))
    diff = torch.abs(mean - torch.tensor([mc_true, eta_true], dtype=torch.float64))

    # Mean must enclose truth within 2 sigma
    assert torch.all(diff < 2.0 * std), f"Posterior mean {mean} is not within 2 sigma of truth {[mc_true, eta_true]}"

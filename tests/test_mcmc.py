"""Tests for pure PyTorch No-U-Turn Sampler (NUTS) and Gelman-Rubin convergence diagnostics."""

import math
import pytest
import torch

from torchwave.approximants.taylorf2 import TaylorF2
from torchwave.inference.mcmc import NUTS, gelman_rubin, leapfrog_step
from torchwave.inference.relative_binning import RelativeBinningLikelihood


def _fixed_random(monkeypatch):
    monkeypatch.setattr(torch, "randn_like", lambda theta: torch.ones_like(theta))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.75]))


def _averaged_step_size(epsilon, acceptances, target_accept=0.65):
    """Independent scalar application of Hoffman & Gelman Algorithm 5."""
    h_bar, log_bar = 0.0, 0.0
    mu = math.log(10.0 * epsilon)
    for m, acceptance in enumerate(acceptances, 1):
        eta = 1.0 / (m + 10.0)
        h_bar = (1.0 - eta) * h_bar + eta * (target_accept - acceptance)
        log_eps = mu - math.sqrt(m) * h_bar / 0.05
        weight = m ** -0.75
        log_bar = weight * log_eps + (1.0 - weight) * log_bar
    return math.exp(log_bar)


def _scripted_tree(records, acceptances, *, stop=True):
    def tree(theta, momentum, grad, log_u, direction, depth, epsilon, h0, **kwargs):
        records.append({"epsilon": epsilon, "momentum": momentum.clone(), "depth": depth})
        alpha, n_alpha = acceptances[depth]
        endpoint = theta + direction * (2 ** depth)
        return (endpoint, momentum, grad, endpoint, momentum, grad,
                endpoint, grad, (-0.5 * endpoint.square().sum()).detach(),
                1, not stop, alpha, n_alpha)
    return tree


def test_nuts_warmup_handoff_uses_averaged_step_size(monkeypatch):
    records = []
    sampler = NUTS(lambda theta: -0.5 * theta.square().sum(), max_depth=1)
    monkeypatch.setattr(sampler, "_build_tree", _scripted_tree(records, [(0.8, 1)]))
    sampler.sample_chain(torch.zeros(1, dtype=torch.float64), n_samples=2,
                         n_warmup=3, epsilon=0.1)
    expected = _averaged_step_size(0.1, [0.8] * 3)
    assert [record["epsilon"] for record in records[3:]] == pytest.approx([expected] * 2)


def test_nuts_diagonal_metric_momentum_transition(monkeypatch):
    _fixed_random(monkeypatch)
    records = []
    metric = torch.tensor([4.0, 0.25], dtype=torch.float64)
    sampler = NUTS(lambda theta: -0.5 * theta.square().sum(), inv_mass=metric, max_depth=1)
    monkeypatch.setattr(sampler, "_build_tree", _scripted_tree(records, [(1.0, 1)]))
    sampler.sample_chain(torch.zeros(2, dtype=torch.float64), n_samples=1,
                         n_warmup=0, epsilon=0.1)
    assert torch.equal(records[0]["momentum"], torch.tensor([0.5, 2.0], dtype=torch.float64))


def test_nuts_diagonal_metric_momentum_epsilon_search(monkeypatch):
    import torchwave.inference.mcmc as mcmc

    _fixed_random(monkeypatch)
    momenta = []
    original = mcmc.leapfrog_step

    def record_momentum(theta, momentum, *args, **kwargs):
        momenta.append(momentum.clone())
        return original(theta, momentum, *args, **kwargs)

    monkeypatch.setattr(mcmc, "leapfrog_step", record_momentum)
    sampler = NUTS(lambda theta: -0.5 * theta.square().sum(),
                   inv_mass=torch.tensor([4.0, 0.25], dtype=torch.float64))
    theta = torch.zeros(2, dtype=torch.float64)
    sampler.find_reasonable_epsilon(theta, *sampler.value_and_grad(theta))
    assert torch.equal(momenta[0], torch.tensor([0.5, 2.0], dtype=torch.float64))


def test_nuts_algorithm6_top_level_selection(monkeypatch):
    """Algorithm 6 selects a valid equally sized new subtree with probability one."""
    _fixed_random(monkeypatch)
    sampler = NUTS(lambda theta: -0.5 * theta.square().sum(), max_depth=1)
    monkeypatch.setattr(sampler, "_build_tree", _scripted_tree([], [(1.0, 1)], stop=False))
    samples = sampler.sample_chain(torch.zeros(1, dtype=torch.float64), n_samples=1,
                                  n_warmup=0, epsilon=0.1)
    assert torch.equal(samples, torch.ones(1, 1, dtype=torch.float64))


def test_nuts_adaptation_uses_final_doubling_acceptance(monkeypatch):
    _fixed_random(monkeypatch)
    records = []
    sampler = NUTS(lambda theta: -0.5 * theta.square().sum(), max_depth=2)
    monkeypatch.setattr(sampler, "_build_tree",
                        _scripted_tree(records, [(0.2, 1), (2.4, 3)], stop=False))
    sampler.sample_chain(torch.zeros(1, dtype=torch.float64), n_samples=1,
                         n_warmup=1, epsilon=0.1)
    assert records[2]["epsilon"] == pytest.approx(_averaged_step_size(0.1, [0.8]))


@pytest.mark.parametrize("target", [
    lambda theta: theta.sum() * float("nan"),
    lambda theta: theta.sum() * float("inf"),
    lambda theta: -torch.sqrt(theta[0].square()),
])
def test_nuts_rejects_invalid_initial_target(target):
    sampler = NUTS(target, max_depth=1)
    with pytest.raises(ValueError, match="initial.*finite"):
        sampler.sample_chain(torch.zeros(1, dtype=torch.float64), n_samples=1,
                             n_warmup=0, epsilon=0.1)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), "gradient"])
def test_nuts_invalid_proposal_has_zero_acceptance(monkeypatch, invalid):
    sampler = NUTS(lambda theta: -0.5 * theta.square().sum())

    def invalid_target(theta):
        if invalid == "gradient":
            return torch.zeros((), dtype=theta.dtype), torch.full_like(theta, float("nan"))
        return torch.tensor(invalid, dtype=theta.dtype), torch.zeros_like(theta)

    monkeypatch.setattr(sampler, "value_and_grad", invalid_target)
    theta = torch.zeros(1, dtype=torch.float64)
    tree = sampler._build_tree(theta, torch.ones_like(theta), theta, -1.0, 1, 0, 0.1, 0.5)
    assert tree[9:13] == (0, False, 0.0, 1)


@pytest.mark.parametrize("epsilon", [0.2, 100.0])
def test_nuts_transition_reuses_selected_log_probability(epsilon):
    """Only the initial state and actual leapfrogs evaluate the target."""
    calls = []

    def log_probability(theta):
        calls.append(theta.requires_grad)
        return -0.5 * theta.square().sum()

    torch.manual_seed(184)
    sampler = NUTS(log_probability, max_depth=3)
    _, diagnostics = sampler.sample_chain(
        torch.tensor([0.1, -0.15], dtype=torch.float64),
        n_samples=12, n_warmup=4, epsilon=epsilon,
        adapt_step_size=False, return_diagnostics=True,
    )
    assert len(calls) == 1 + int(diagnostics["leapfrog_count"].sum().item())
    assert all(calls)
    if epsilon == 100.0:
        assert diagnostics["divergent"].all()


@pytest.mark.parametrize("depth", [0, 1, 3])
@pytest.mark.parametrize("direction", [-1, 1])
def test_nuts_recursive_candidate_retains_its_log_probability(depth, direction):
    """Recursive selection must carry the value of the chosen state, detached."""
    def log_probability(theta):
        x, y = theta.unbind()
        return -0.5 * (x.square() + ((y - 0.3 * x.square()) / 0.7).square())

    torch.manual_seed(185)
    theta = torch.tensor([0.1, -0.15], dtype=torch.float64)
    momentum = torch.tensor([0.3, -0.8], dtype=torch.float64)
    sampler = NUTS(log_probability)
    logp, gradient = sampler.value_and_grad(theta)
    energy = float((-logp + sampler.kinetic_energy(momentum)).item())
    tree = sampler._build_tree(
        theta, momentum, gradient, -energy - 3.0, direction, depth, 0.12, energy,
    )
    selected_theta, selected_logp = tree[6], tree[8]
    assert torch.equal(selected_logp, log_probability(selected_theta))
    assert not selected_logp.requires_grad


def test_nuts_constant_finite_target_has_zero_gradient():
    sampler = NUTS(lambda theta: theta.new_tensor(-2.0))
    value, gradient = sampler.value_and_grad(torch.ones(2, dtype=torch.float64))
    assert value.item() == -2.0
    assert torch.equal(gradient, torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize("metric", [
    [0.0], [-1.0], [float("nan")], [float("inf")], [[1.0]],
])
def test_nuts_rejects_invalid_diagonal_metric(metric):
    with pytest.raises(ValueError, match="finite positive diagonal"):
        NUTS(lambda theta: -theta.square().sum(), inv_mass=torch.tensor(metric))


def test_nuts_rejects_mismatched_metric_shape():
    sampler = NUTS(lambda theta: -theta.square().sum(), inv_mass=torch.ones(2))
    with pytest.raises(ValueError, match="parameter vector shape"):
        sampler.sample_chain(torch.zeros(1), n_samples=1, n_warmup=0, epsilon=0.1)


def test_nuts_invalid_proposal_diagnostics_and_rejection(monkeypatch):
    _fixed_random(monkeypatch)

    def bounded_target(theta):
        return -0.5 * theta.square().sum() if bool((theta == 0).all()) else theta.sum() * float("nan")

    sampler = NUTS(bounded_target, max_depth=2)
    progress = []
    samples, diagnostics = sampler.sample_chain(
        torch.zeros(1, dtype=torch.float64), n_samples=2, n_warmup=1,
        epsilon=0.1, adapt_step_size=False, return_diagnostics=True,
        progress_callback=lambda iteration, total, row: progress.append((iteration, total, row)),
    )
    assert torch.equal(samples, torch.zeros(2, 1, dtype=torch.float64))
    assert diagnostics["divergent"].tolist() == [True] * 3
    assert diagnostics["acceptance_stat"].tolist() == [0.0] * 3
    assert diagnostics["tree_depth"].tolist() == [1] * 3
    assert diagnostics["leapfrog_count"].tolist() == [1] * 3
    assert diagnostics["hit_max_depth"].tolist() == [False] * 3
    assert diagnostics["warmup"].tolist() == [True, False, False]
    assert diagnostics["step_size"].tolist() == [0.1] * 3
    assert diagnostics["energy"].tolist() == [0.5] * 3
    assert [iteration for iteration, total, row in progress] == [1, 2, 3]
    assert all(total == 3 and row["divergent"] for iteration, total, row in progress)
    for key, values in diagnostics.items():
        assert values.shape == (3,)
        assert values.dtype in (torch.bool, torch.int64, torch.float64)
        assert values.device == samples.device


def test_nuts_depth_cap_counts_actual_leapfrogs(monkeypatch):
    _fixed_random(monkeypatch)
    # Constant finite target produces a straight trajectory, reaching the cap.
    sampler = NUTS(lambda theta: theta.new_tensor(0.0), max_depth=3)
    samples, diagnostics = sampler.sample_chain(
        torch.zeros(1, dtype=torch.float64), n_samples=1, n_warmup=0,
        epsilon=0.1, return_diagnostics=True,
    )
    assert torch.isfinite(samples).all()
    assert diagnostics["tree_depth"].item() == 3
    assert diagnostics["leapfrog_count"].item() == 7
    assert diagnostics["hit_max_depth"].item()
    assert not diagnostics["divergent"].item()
    assert diagnostics["acceptance_stat"].item() == 1.0


def test_nuts_diagnostics_preserve_default_samples_and_parallel_api():
    sampler = NUTS(lambda theta: -0.5 * theta.square().sum(), max_depth=3)
    positions = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    torch.manual_seed(813)
    ordinary = sampler.sample_parallel(positions, n_samples=4, n_warmup=3, epsilon=0.1)
    torch.manual_seed(813)
    samples, diagnostics = sampler.sample_parallel(
        positions, n_samples=4, n_warmup=3, epsilon=0.1, return_diagnostics=True,
    )
    assert isinstance(ordinary, torch.Tensor)
    assert torch.equal(ordinary, samples)
    assert all(values.shape == (2, 7) for values in diagnostics.values())
    assert torch.equal(diagnostics["step_size"][:, 3:], diagnostics["step_size"][:, 3:4].expand(-1, 4))


def test_nuts_diagonal_metric_analytical_gaussian():
    """Check native mass consistency against known means, covariance and marginal CDFs."""
    torch.manual_seed(1063)
    mean = torch.tensor([1.0, -2.0], dtype=torch.float64)
    variance = torch.tensor([0.25, 4.0], dtype=torch.float64)
    sampler = NUTS(lambda theta: -0.5 * ((theta - mean).square() / variance).sum(),
                   inv_mass=variance, target_accept=0.8, max_depth=6)
    positions = mean + variance.sqrt() * torch.tensor(
        [[-2.0, 2.0], [2.0, -2.0], [-1.0, -1.0], [1.0, 1.0]], dtype=torch.float64,
    )
    chains = sampler.sample_parallel(positions, n_samples=600, n_warmup=300)
    standardized = ((chains - mean) / variance.sqrt()).reshape(-1, 2)
    assert torch.all(standardized.mean(0).abs() < 0.2)
    assert torch.all((standardized.var(0) - 1.0).abs() < 0.2)
    covariance = torch.cov(standardized.T)
    assert covariance[0, 1].abs() < 0.15
    assert torch.all(gelman_rubin(chains) < 1.05)
    ordered = standardized.sort(dim=0).values
    cdf = 0.5 * (1.0 + torch.erf(ordered / math.sqrt(2.0)))
    n = len(ordered)
    empirical_upper = torch.arange(1, n + 1, dtype=torch.float64).unsqueeze(1) / n
    empirical_lower = empirical_upper - 1.0 / n
    assert torch.max(torch.maximum(empirical_upper - cdf, cdf - empirical_lower)) < 0.07


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


@pytest.mark.parametrize("inv_mass", [None, torch.tensor([4.0, 0.25], dtype=torch.float64)])
def test_leapfrog_step_reversibility(inv_mass):
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
    theta1, p1, _, grad1 = leapfrog_step(theta0, p0, grad0, eps, toy_grad_fn, inv_mass)

    # Backward step with reversed momentum or negative step size
    theta_rev, p_rev, _, _ = leapfrog_step(theta1, p1, grad1, -eps, toy_grad_fn, inv_mass)

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
